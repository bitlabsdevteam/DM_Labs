import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _uniform_baseline_summary(vocab_size: Optional[int]) -> Dict[str, float]:
    if vocab_size is None or int(vocab_size) <= 0:
        return {
            "vocab_size": None,
            "uniform_random_avg_cross_entropy": float("nan"),
            "uniform_random_pseudo_perplexity": float("nan"),
            "uniform_random_bits_per_masked_token": float("nan"),
        }
    vocab_size = int(vocab_size)
    uniform_ce = math.log(float(vocab_size))
    return {
        "vocab_size": vocab_size,
        "uniform_random_avg_cross_entropy": uniform_ce,
        "uniform_random_pseudo_perplexity": float(vocab_size),
        "uniform_random_bits_per_masked_token": uniform_ce / math.log(2.0),
    }


def _view_calibration(avg_cross_entropy: float, bits_per_masked_token: float, vocab_size: Optional[int]) -> Dict[str, float]:
    if vocab_size is None or int(vocab_size) <= 0:
        return {
            "uniform_random_pseudo_perplexity": float("nan"),
            "bits_saved_vs_uniform": float("nan"),
            "denoising_skill": float("nan"),
        }
    if avg_cross_entropy is None or bits_per_masked_token is None:
        return {
            "uniform_random_pseudo_perplexity": float(vocab_size),
            "bits_saved_vs_uniform": float("nan"),
            "denoising_skill": float("nan"),
        }
    avg_cross_entropy = float(avg_cross_entropy)
    bits_per_masked_token = float(bits_per_masked_token)
    if math.isnan(avg_cross_entropy) or math.isnan(bits_per_masked_token):
        return {
            "uniform_random_pseudo_perplexity": float(vocab_size),
            "bits_saved_vs_uniform": float("nan"),
            "denoising_skill": float("nan"),
        }
    uniform_ce = math.log(float(vocab_size))
    uniform_bits = uniform_ce / math.log(2.0)
    return {
        "uniform_random_pseudo_perplexity": float(vocab_size),
        "bits_saved_vs_uniform": uniform_bits - bits_per_masked_token,
        "denoising_skill": 1.0 - (avg_cross_entropy / uniform_ce),
    }


def _normalize_excluded_token_ids(excluded_token_ids: Optional[Sequence[Optional[int]]]) -> List[int]:
    if excluded_token_ids is None:
        return []
    return sorted({int(token_id) for token_id in excluded_token_ids if token_id is not None})


def _resolve_timestep_grid(T: int, timestep_grid: Optional[Sequence[int]] = None) -> List[int]:
    if timestep_grid is None:
        timestep_grid = [1, max(1, T // 4), max(1, T // 2), max(1, (3 * T) // 4), T]
    return sorted({int(max(1, min(T, t))) for t in timestep_grid})


def _safe_exp(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    return math.exp(min(80.0, value))


def _expected_top1_correct_count(logits: Tensor, targets: Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    max_logits = logits.max(dim=-1, keepdim=True).values
    is_tied_max = logits == max_logits
    tie_counts = is_tied_max.sum(dim=-1).clamp(min=1).to(dtype=torch.float64)
    target_is_tied_max = is_tied_max.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).to(dtype=torch.float64)
    return float((target_is_tied_max / tie_counts).sum().item())


def _timestep_fraction_from_int(timestep: int, T: int) -> float:
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    return float(max(1, min(T, int(timestep)))) / float(T)


def _resolve_eval_timesteps(timestep_plan: Dict[str, object], T: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    source_t = timestep_plan.get("t")
    if source_t is not None and not isinstance(source_t, torch.Tensor):
        source_t = torch.tensor(source_t, dtype=torch.long)
    if isinstance(source_t, torch.Tensor):
        source_t = source_t.detach().cpu().long()

    if "u" in timestep_plan and timestep_plan["u"] is not None:
        u = timestep_plan["u"]
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32)
        u = u.to(device=device, dtype=torch.float32)
        eval_t = torch.clamp(torch.round(u * float(T)), min=1, max=T).long()
        if source_t is None:
            source_t = eval_t.detach().cpu()
        return eval_t, u, source_t

    if source_t is None:
        raise ValueError("Each timestep_plan must provide either 'u' (normalized timestep fractions) or 't' (integer timesteps).")

    eval_t = source_t.to(device=device, dtype=torch.long)
    u = (eval_t.float() / float(T)).detach().cpu()
    return eval_t, u.to(device=device), source_t


def _bootstrap_metric_interval(
    batch_records: Sequence[Dict[str, float]],
    *,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, object]:
    if not batch_records:
        return {
            "method": "bootstrap_over_sampled_eval_batches",
            "n_samples": 0,
            "replicates": 0,
            "avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = len(batch_records)
    ce_samples: List[float] = []

    for _ in range(n_samples):
        indices = torch.randint(0, n, (n,), generator=rng)
        nll_sum = 0.0
        masked_tokens = 0
        for idx in indices.tolist():
            rec = batch_records[idx]
            nll_sum += float(rec["nll_sum"])
            masked_tokens += int(rec["masked_tokens"])
        if masked_tokens == 0:
            continue
        ce_samples.append(nll_sum / masked_tokens)

    if not ce_samples:
        return {
            "method": "bootstrap_over_sampled_eval_batches",
            "n_samples": n,
            "replicates": 0,
            "avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    xs = torch.tensor(ce_samples, dtype=torch.float64)
    quantiles = torch.quantile(xs, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    mean_ce = float(xs.mean().item())

    def _summary(transform: Callable[[float], float]) -> Dict[str, float]:
        transformed = [transform(float(v)) for v in xs.tolist()]
        tx = torch.tensor(transformed, dtype=torch.float64)
        q = torch.quantile(tx, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
        return {
            "mean": float(tx.mean().item()),
            "p05": float(q[0]),
            "p50": float(q[1]),
            "p95": float(q[2]),
        }

    return {
        "method": "bootstrap_over_sampled_eval_batches",
        "n_samples": n,
        "replicates": len(ce_samples),
        "avg_cross_entropy": {
            "mean": mean_ce,
            "p05": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p95": float(quantiles[2]),
        },
        "pseudo_perplexity": _summary(_safe_exp),
        "bits_per_masked_token": _summary(lambda ce: ce / math.log(2.0)),
    }


def _bootstrap_example_uniform_metric_interval(
    example_records: Sequence[Dict[str, float]],
    *,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, object]:
    if not example_records:
        return {
            "method": "bootstrap_over_example_records_uniform",
            "n_examples": 0,
            "replicates": 0,
            "timestep_uniform_avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_uniform_pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_uniform_bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_uniform_masked_token_accuracy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = len(example_records)
    ce_samples: List[float] = []
    acc_samples: List[float] = []

    for _ in range(n_samples):
        indices = torch.randint(0, n, (n,), generator=rng)
        sampled_ce: List[float] = []
        sampled_acc: List[float] = []
        for idx in indices.tolist():
            rec = example_records[idx]
            sampled_ce.append(float(rec["avg_cross_entropy"]))
            sampled_acc.append(float(rec["masked_token_accuracy"]))
        if not sampled_ce:
            continue
        ce_samples.append(sum(sampled_ce) / len(sampled_ce))
        acc_samples.append(sum(sampled_acc) / len(sampled_acc))

    if not ce_samples:
        return {
            "method": "bootstrap_over_example_records_uniform",
            "n_examples": n,
            "replicates": 0,
            "timestep_uniform_avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_uniform_pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_uniform_bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_uniform_masked_token_accuracy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    ce_tensor = torch.tensor(ce_samples, dtype=torch.float64)
    ce_q = torch.quantile(ce_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    ppx_tensor = torch.tensor([_safe_exp(float(v)) for v in ce_samples], dtype=torch.float64)
    ppx_q = torch.quantile(ppx_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    bits_tensor = torch.tensor([float(v) / math.log(2.0) for v in ce_samples], dtype=torch.float64)
    bits_q = torch.quantile(bits_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    acc_tensor = torch.tensor(acc_samples, dtype=torch.float64)
    acc_q = torch.quantile(acc_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()

    return {
        "method": "bootstrap_over_example_records_uniform",
        "n_examples": n,
        "replicates": len(ce_samples),
        "timestep_uniform_avg_cross_entropy": {"mean": float(ce_tensor.mean().item()), "p05": float(ce_q[0]), "p50": float(ce_q[1]), "p95": float(ce_q[2])},
        "timestep_uniform_pseudo_perplexity": {"mean": float(ppx_tensor.mean().item()), "p05": float(ppx_q[0]), "p50": float(ppx_q[1]), "p95": float(ppx_q[2])},
        "timestep_uniform_bits_per_masked_token": {"mean": float(bits_tensor.mean().item()), "p05": float(bits_q[0]), "p50": float(bits_q[1]), "p95": float(bits_q[2])},
        "timestep_uniform_masked_token_accuracy": {"mean": float(acc_tensor.mean().item()), "p05": float(acc_q[0]), "p50": float(acc_q[1]), "p95": float(acc_q[2])},
    }


def _bootstrap_reweighted_metric_interval(
    example_records: Sequence[Dict[str, float]],
    *,
    weight_key: str,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, object]:
    metric_prefix = weight_key.replace("_weight", "")
    ce_key = f"{metric_prefix}_avg_cross_entropy"
    ppx_key = f"{metric_prefix}_pseudo_perplexity"
    bits_key = f"{metric_prefix}_bits_per_masked_token"
    acc_key = f"{metric_prefix}_masked_token_accuracy"

    if not example_records:
        return {
            "method": f"bootstrap_over_example_records_with_{weight_key}",
            "n_examples": 0,
            "replicates": 0,
            ce_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            ppx_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            bits_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            acc_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = len(example_records)
    ce_samples: List[float] = []
    acc_samples: List[float] = []

    for _ in range(n_samples):
        indices = torch.randint(0, n, (n,), generator=rng)
        weighted_nll = 0.0
        weighted_tokens = 0.0
        weighted_correct = 0.0
        for idx in indices.tolist():
            rec = example_records[idx]
            weight = float(rec.get(weight_key, 0.0))
            if weight <= 0.0:
                continue
            weighted_nll += float(rec["nll_sum"]) * weight
            weighted_tokens += float(rec["masked_tokens"]) * weight
            weighted_correct += float(rec.get("correct_masked_tokens", 0)) * weight
        if weighted_tokens <= 0.0:
            continue
        ce_samples.append(weighted_nll / weighted_tokens)
        acc_samples.append(weighted_correct / weighted_tokens)

    if not ce_samples:
        return {
            "method": f"bootstrap_over_example_records_with_{weight_key}",
            "n_examples": n,
            "replicates": 0,
            ce_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            ppx_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            bits_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            acc_key: {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    ce_tensor = torch.tensor(ce_samples, dtype=torch.float64)
    ce_q = torch.quantile(ce_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    ppx_tensor = torch.tensor([_safe_exp(float(v)) for v in ce_samples], dtype=torch.float64)
    ppx_q = torch.quantile(ppx_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    bits_tensor = torch.tensor([float(v) / math.log(2.0) for v in ce_samples], dtype=torch.float64)
    bits_q = torch.quantile(bits_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    acc_tensor = torch.tensor(acc_samples, dtype=torch.float64)
    acc_q = torch.quantile(acc_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()

    return {
        "method": f"bootstrap_over_example_records_with_{weight_key}",
        "n_examples": n,
        "replicates": len(ce_samples),
        ce_key: {"mean": float(ce_tensor.mean().item()), "p05": float(ce_q[0]), "p50": float(ce_q[1]), "p95": float(ce_q[2])},
        ppx_key: {"mean": float(ppx_tensor.mean().item()), "p05": float(ppx_q[0]), "p50": float(ppx_q[1]), "p95": float(ppx_q[2])},
        bits_key: {"mean": float(bits_tensor.mean().item()), "p05": float(bits_q[0]), "p50": float(bits_q[1]), "p95": float(bits_q[2])},
        acc_key: {"mean": float(acc_tensor.mean().item()), "p05": float(acc_q[0]), "p50": float(acc_q[1]), "p95": float(acc_q[2])},
    }


def _bootstrap_grid_uniform_metric_interval(
    grid_batch_records: Sequence[Dict[str, float]],
    *,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, object]:
    if not grid_batch_records:
        return {
            "method": "bootstrap_over_grid_batch_timestep_records",
            "n_records": 0,
            "replicates": 0,
            "grid_uniform_avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "grid_uniform_pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "grid_uniform_bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "grid_uniform_masked_token_accuracy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n = len(grid_batch_records)
    ce_samples: List[float] = []
    acc_samples: List[float] = []

    for _ in range(n_samples):
        indices = torch.randint(0, n, (n,), generator=rng)
        ce_values: List[float] = []
        acc_values: List[float] = []
        for idx in indices.tolist():
            rec = grid_batch_records[idx]
            ce_values.append(float(rec["avg_cross_entropy"]))
            acc_values.append(float(rec["masked_token_accuracy"]))
        if not ce_values:
            continue
        ce_samples.append(sum(ce_values) / len(ce_values))
        acc_samples.append(sum(acc_values) / len(acc_values))

    if not ce_samples:
        return {
            "method": "bootstrap_over_grid_batch_timestep_records",
            "n_records": n,
            "replicates": 0,
            "grid_uniform_avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "grid_uniform_pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "grid_uniform_bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "grid_uniform_masked_token_accuracy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    ce_tensor = torch.tensor(ce_samples, dtype=torch.float64)
    ce_q = torch.quantile(ce_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    ppx_tensor = torch.tensor([_safe_exp(float(v)) for v in ce_samples], dtype=torch.float64)
    ppx_q = torch.quantile(ppx_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    bits_tensor = torch.tensor([float(v) / math.log(2.0) for v in ce_samples], dtype=torch.float64)
    bits_q = torch.quantile(bits_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    acc_tensor = torch.tensor(acc_samples, dtype=torch.float64)
    acc_q = torch.quantile(acc_tensor, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()

    return {
        "method": "bootstrap_over_grid_batch_timestep_records",
        "n_records": n,
        "replicates": len(ce_samples),
        "grid_uniform_avg_cross_entropy": {
            "mean": float(ce_tensor.mean().item()),
            "p05": float(ce_q[0]),
            "p50": float(ce_q[1]),
            "p95": float(ce_q[2]),
        },
        "grid_uniform_pseudo_perplexity": {
            "mean": float(ppx_tensor.mean().item()),
            "p05": float(ppx_q[0]),
            "p50": float(ppx_q[1]),
            "p95": float(ppx_q[2]),
        },
        "grid_uniform_bits_per_masked_token": {
            "mean": float(bits_tensor.mean().item()),
            "p05": float(bits_q[0]),
            "p50": float(bits_q[1]),
            "p95": float(bits_q[2]),
        },
        "grid_uniform_masked_token_accuracy": {
            "mean": float(acc_tensor.mean().item()),
            "p05": float(acc_q[0]),
            "p50": float(acc_q[1]),
            "p95": float(acc_q[2]),
        },
    }


def _metric_percentiles(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        return {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    xs = torch.tensor(list(samples), dtype=torch.float64)
    q = torch.quantile(xs, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
    return {
        "mean": float(xs.mean().item()),
        "p05": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
    }


def _bootstrap_timestep_metric_interval(
    timestep_metrics: Sequence[Dict[str, float]],
    *,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, object]:
    if not timestep_metrics:
        return {
            "method": "bootstrap_over_timestep_metrics",
            "n_timesteps": 0,
            "replicates": 0,
            "timestep_macro_avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_macro_pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_macro_bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_macro_masked_token_accuracy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_auc_avg_cross_entropy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_auc_pseudo_perplexity": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_auc_bits_per_masked_token": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
            "timestep_auc_masked_token_accuracy": {"mean": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")},
        }

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n_timesteps = len(timestep_metrics)
    macro_ce_samples: List[float] = []
    macro_acc_samples: List[float] = []
    auc_ce_samples: List[float] = []
    auc_acc_samples: List[float] = []

    for _ in range(n_samples):
        indices = torch.randint(0, n_timesteps, (n_timesteps,), generator=rng)
        sampled = [dict(timestep_metrics[idx]) for idx in indices.tolist()]
        sampled.sort(key=lambda row: (float(row.get("timestep_fraction", float("nan"))), int(row.get("source_plan_timestep", row.get("timestep", 0)))))
        for rank, row in enumerate(sampled):
            row["timestep_fraction"] = float(rank) / float(max(1, n_timesteps - 1)) if n_timesteps > 1 else float(row.get("timestep_fraction", 1.0))

        macro_metrics = _compute_timestep_macro_metrics(sampled)
        auc_metrics = _compute_timestep_auc_metrics(sampled)
        macro_ce_samples.append(float(macro_metrics["timestep_macro_avg_cross_entropy"]))
        macro_acc_samples.append(float(macro_metrics["timestep_macro_masked_token_accuracy"]))
        auc_ce_samples.append(float(auc_metrics["timestep_auc_avg_cross_entropy"]))
        auc_acc_samples.append(float(auc_metrics["timestep_auc_masked_token_accuracy"]))

    return {
        "method": "bootstrap_over_timestep_metrics",
        "n_timesteps": n_timesteps,
        "replicates": n_samples,
        "timestep_macro_avg_cross_entropy": _metric_percentiles(macro_ce_samples),
        "timestep_macro_pseudo_perplexity": _metric_percentiles([_safe_exp(v) for v in macro_ce_samples]),
        "timestep_macro_bits_per_masked_token": _metric_percentiles([v / math.log(2.0) for v in macro_ce_samples]),
        "timestep_macro_masked_token_accuracy": _metric_percentiles(macro_acc_samples),
        "timestep_auc_avg_cross_entropy": _metric_percentiles(auc_ce_samples),
        "timestep_auc_pseudo_perplexity": _metric_percentiles([_safe_exp(v) for v in auc_ce_samples]),
        "timestep_auc_bits_per_masked_token": _metric_percentiles([v / math.log(2.0) for v in auc_ce_samples]),
        "timestep_auc_masked_token_accuracy": _metric_percentiles(auc_acc_samples),
    }


def _compute_timestep_macro_metrics(timestep_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not timestep_metrics:
        return {
            "timestep_macro_avg_cross_entropy": float("nan"),
            "timestep_macro_pseudo_perplexity": float("nan"),
            "timestep_macro_bits_per_masked_token": float("nan"),
            "timestep_macro_masked_token_accuracy": float("nan"),
            "timestep_macro_timestep_count": 0,
        }

    avg_cross_entropy_values = [float(row["avg_cross_entropy"]) for row in timestep_metrics]
    masked_token_accuracy_values = [float(row["masked_token_accuracy"]) for row in timestep_metrics]
    mean_ce = sum(avg_cross_entropy_values) / len(avg_cross_entropy_values)
    mean_acc = sum(masked_token_accuracy_values) / len(masked_token_accuracy_values)
    return {
        "timestep_macro_avg_cross_entropy": mean_ce,
        "timestep_macro_pseudo_perplexity": _safe_exp(mean_ce),
        "timestep_macro_bits_per_masked_token": mean_ce / math.log(2.0),
        "timestep_macro_masked_token_accuracy": mean_acc,
        "timestep_macro_timestep_count": len(timestep_metrics),
    }


def _compute_timestep_auc_metrics(timestep_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not timestep_metrics:
        return {
            "timestep_auc_avg_cross_entropy": float("nan"),
            "timestep_auc_pseudo_perplexity": float("nan"),
            "timestep_auc_bits_per_masked_token": float("nan"),
            "timestep_auc_masked_token_accuracy": float("nan"),
            "timestep_auc_timestep_count": 0,
            "timestep_auc_fraction_span": float("nan"),
        }

    rows = sorted(
        (
            {
                "timestep_fraction": float(row.get("timestep_fraction", float("nan"))),
                "avg_cross_entropy": float(row["avg_cross_entropy"]),
                "masked_token_accuracy": float(row["masked_token_accuracy"]),
            }
            for row in timestep_metrics
        ),
        key=lambda row: row["timestep_fraction"],
    )

    if len(rows) == 1:
        mean_ce = rows[0]["avg_cross_entropy"]
        mean_acc = rows[0]["masked_token_accuracy"]
        return {
            "timestep_auc_avg_cross_entropy": mean_ce,
            "timestep_auc_pseudo_perplexity": _safe_exp(mean_ce),
            "timestep_auc_bits_per_masked_token": mean_ce / math.log(2.0),
            "timestep_auc_masked_token_accuracy": mean_acc,
            "timestep_auc_timestep_count": 1,
            "timestep_auc_fraction_span": 0.0,
        }

    x = torch.tensor([row["timestep_fraction"] for row in rows], dtype=torch.float64)
    if bool(torch.any(torch.diff(x) <= 0)):
        raise ValueError("timestep_fraction values must be strictly increasing to compute timestep AUC metrics.")

    span = float((x[-1] - x[0]).item())
    if span <= 0.0:
        mean_ce = rows[0]["avg_cross_entropy"]
        mean_acc = rows[0]["masked_token_accuracy"]
        return {
            "timestep_auc_avg_cross_entropy": mean_ce,
            "timestep_auc_pseudo_perplexity": _safe_exp(mean_ce),
            "timestep_auc_bits_per_masked_token": mean_ce / math.log(2.0),
            "timestep_auc_masked_token_accuracy": mean_acc,
            "timestep_auc_timestep_count": len(rows),
            "timestep_auc_fraction_span": span,
        }

    ce_y = torch.tensor([row["avg_cross_entropy"] for row in rows], dtype=torch.float64)
    acc_y = torch.tensor([row["masked_token_accuracy"] for row in rows], dtype=torch.float64)
    ce_auc = float(torch.trapezoid(ce_y, x).item() / span)
    acc_auc = float(torch.trapezoid(acc_y, x).item() / span)
    return {
        "timestep_auc_avg_cross_entropy": ce_auc,
        "timestep_auc_pseudo_perplexity": _safe_exp(ce_auc),
        "timestep_auc_bits_per_masked_token": ce_auc / math.log(2.0),
        "timestep_auc_masked_token_accuracy": acc_auc,
        "timestep_auc_timestep_count": len(rows),
        "timestep_auc_fraction_span": span,
    }


def _bootstrap_paired_comparison_interval(
    cosine_batch_metrics: Sequence[Dict[str, float]],
    linear_batch_metrics: Sequence[Dict[str, float]],
    *,
    cosine_example_metrics: Optional[Sequence[Dict[str, float]]] = None,
    linear_example_metrics: Optional[Sequence[Dict[str, float]]] = None,
    cosine_grid_batch_metrics: Optional[Sequence[Dict[str, float]]] = None,
    linear_grid_batch_metrics: Optional[Sequence[Dict[str, float]]] = None,
    cosine_timestep_metrics: Optional[Sequence[Dict[str, float]]] = None,
    linear_timestep_metrics: Optional[Sequence[Dict[str, float]]] = None,
    vocab_size: Optional[int] = None,
    n_samples: int = 500,
    seed: int = 0,
) -> Dict[str, object]:
    if not cosine_batch_metrics or not linear_batch_metrics:
        return {
            "method": "paired_bootstrap_over_shared_eval_plan",
            "n_pairs": 0,
            "replicates": 0,
            "delta_linear_minus_cosine": {},
        }
    if len(cosine_batch_metrics) != len(linear_batch_metrics):
        raise ValueError("Paired bootstrap requires the same number of cosine and linear batch records.")
    if cosine_example_metrics is not None and linear_example_metrics is not None and len(cosine_example_metrics) != len(linear_example_metrics):
        raise ValueError("Paired bootstrap requires the same number of cosine and linear example records.")
    if cosine_grid_batch_metrics is not None and linear_grid_batch_metrics is not None and len(cosine_grid_batch_metrics) != len(linear_grid_batch_metrics):
        raise ValueError("Paired bootstrap requires the same number of cosine and linear grid batch records.")
    if cosine_timestep_metrics is not None and linear_timestep_metrics is not None:
        cosine_timestep_by_step = {int(row.get("source_plan_timestep", row["timestep"])): row for row in cosine_timestep_metrics}
        linear_timestep_by_step = {int(row.get("source_plan_timestep", row["timestep"])): row for row in linear_timestep_metrics}
        shared_timestep_keys = sorted(set(cosine_timestep_by_step) & set(linear_timestep_by_step))
        if len(shared_timestep_keys) != len(cosine_timestep_by_step) or len(shared_timestep_keys) != len(linear_timestep_by_step):
            raise ValueError("Paired timestep-macro bootstrap requires the same shared timestep grid for cosine and linear metrics.")
    else:
        cosine_timestep_by_step = {}
        linear_timestep_by_step = {}
        shared_timestep_keys = []

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    n_batches = len(cosine_batch_metrics)
    batch_delta_samples = {
        "avg_cross_entropy": [],
        "pseudo_perplexity": [],
        "bits_per_masked_token": [],
        "masked_token_accuracy": [],
    }
    example_delta_samples = {
        "timestep_uniform_avg_cross_entropy": [],
        "timestep_uniform_pseudo_perplexity": [],
        "timestep_uniform_bits_per_masked_token": [],
        "timestep_uniform_masked_token_accuracy": [],
        "schedule_reweighted_avg_cross_entropy": [],
        "schedule_reweighted_pseudo_perplexity": [],
        "schedule_reweighted_bits_per_masked_token": [],
        "schedule_reweighted_masked_token_accuracy": [],
    }
    grid_delta_samples = {
        "grid_uniform_avg_cross_entropy": [],
        "grid_uniform_pseudo_perplexity": [],
        "grid_uniform_bits_per_masked_token": [],
        "grid_uniform_masked_token_accuracy": [],
    }
    timestep_macro_delta_samples = {
        "timestep_macro_avg_cross_entropy": [],
        "timestep_macro_pseudo_perplexity": [],
        "timestep_macro_bits_per_masked_token": [],
        "timestep_macro_masked_token_accuracy": [],
        "timestep_auc_avg_cross_entropy": [],
        "timestep_auc_pseudo_perplexity": [],
        "timestep_auc_bits_per_masked_token": [],
        "timestep_auc_masked_token_accuracy": [],
    }

    for _ in range(n_samples):
        batch_indices = torch.randint(0, n_batches, (n_batches,), generator=rng)
        cosine_nll = 0.0
        linear_nll = 0.0
        cosine_masked = 0
        linear_masked = 0
        cosine_correct = 0
        linear_correct = 0

        for idx in batch_indices.tolist():
            cosine_rec = cosine_batch_metrics[idx]
            linear_rec = linear_batch_metrics[idx]
            cosine_nll += float(cosine_rec["nll_sum"])
            linear_nll += float(linear_rec["nll_sum"])
            cosine_masked += int(cosine_rec["masked_tokens"])
            linear_masked += int(linear_rec["masked_tokens"])
            cosine_correct += float(cosine_rec.get("correct_masked_tokens", 0))
            linear_correct += float(linear_rec.get("correct_masked_tokens", 0))

        if cosine_masked > 0 and linear_masked > 0:
            cosine_ce = cosine_nll / cosine_masked
            linear_ce = linear_nll / linear_masked
            batch_delta_samples["avg_cross_entropy"].append(linear_ce - cosine_ce)
            batch_delta_samples["pseudo_perplexity"].append(_safe_exp(linear_ce) - _safe_exp(cosine_ce))
            batch_delta_samples["bits_per_masked_token"].append((linear_ce - cosine_ce) / math.log(2.0))
            batch_delta_samples["masked_token_accuracy"].append((linear_correct / linear_masked) - (cosine_correct / cosine_masked))

        if cosine_example_metrics is not None and linear_example_metrics is not None:
            n_examples = len(cosine_example_metrics)
            if n_examples > 0:
                example_indices = torch.randint(0, n_examples, (n_examples,), generator=rng)
                cosine_example_ce = []
                linear_example_ce = []
                cosine_example_acc = []
                linear_example_acc = []
                for idx in example_indices.tolist():
                    cosine_rec = cosine_example_metrics[idx]
                    linear_rec = linear_example_metrics[idx]
                    cosine_example_ce.append(float(cosine_rec["avg_cross_entropy"]))
                    linear_example_ce.append(float(linear_rec["avg_cross_entropy"]))
                    cosine_example_acc.append(float(cosine_rec["masked_token_accuracy"]))
                    linear_example_acc.append(float(linear_rec["masked_token_accuracy"]))
                if cosine_example_ce and linear_example_ce:
                    cosine_uniform_ce = sum(cosine_example_ce) / len(cosine_example_ce)
                    linear_uniform_ce = sum(linear_example_ce) / len(linear_example_ce)
                    example_delta_samples["timestep_uniform_avg_cross_entropy"].append(linear_uniform_ce - cosine_uniform_ce)
                    example_delta_samples["timestep_uniform_pseudo_perplexity"].append(_safe_exp(linear_uniform_ce) - _safe_exp(cosine_uniform_ce))
                    example_delta_samples["timestep_uniform_bits_per_masked_token"].append((linear_uniform_ce - cosine_uniform_ce) / math.log(2.0))
                    example_delta_samples["timestep_uniform_masked_token_accuracy"].append(
                        (sum(linear_example_acc) / len(linear_example_acc)) - (sum(cosine_example_acc) / len(cosine_example_acc))
                    )

                    cosine_rw_nll = 0.0
                    linear_rw_nll = 0.0
                    cosine_rw_tokens = 0.0
                    linear_rw_tokens = 0.0
                    cosine_rw_correct = 0.0
                    linear_rw_correct = 0.0
                    for idx in example_indices.tolist():
                        cosine_rec = cosine_example_metrics[idx]
                        linear_rec = linear_example_metrics[idx]
                        cosine_weight = float(cosine_rec.get("schedule_reweighted_weight", 0.0))
                        linear_weight = float(linear_rec.get("schedule_reweighted_weight", 0.0))
                        if cosine_weight > 0.0:
                            cosine_rw_nll += float(cosine_rec["nll_sum"]) * cosine_weight
                            cosine_rw_tokens += float(cosine_rec["masked_tokens"]) * cosine_weight
                            cosine_rw_correct += float(cosine_rec.get("correct_masked_tokens", 0)) * cosine_weight
                        if linear_weight > 0.0:
                            linear_rw_nll += float(linear_rec["nll_sum"]) * linear_weight
                            linear_rw_tokens += float(linear_rec["masked_tokens"]) * linear_weight
                            linear_rw_correct += float(linear_rec.get("correct_masked_tokens", 0)) * linear_weight
                    if cosine_rw_tokens > 0.0 and linear_rw_tokens > 0.0:
                        cosine_rw_ce = cosine_rw_nll / cosine_rw_tokens
                        linear_rw_ce = linear_rw_nll / linear_rw_tokens
                        example_delta_samples["schedule_reweighted_avg_cross_entropy"].append(linear_rw_ce - cosine_rw_ce)
                        example_delta_samples["schedule_reweighted_pseudo_perplexity"].append(_safe_exp(linear_rw_ce) - _safe_exp(cosine_rw_ce))
                        example_delta_samples["schedule_reweighted_bits_per_masked_token"].append((linear_rw_ce - cosine_rw_ce) / math.log(2.0))
                        example_delta_samples["schedule_reweighted_masked_token_accuracy"].append(
                            (linear_rw_correct / linear_rw_tokens) - (cosine_rw_correct / cosine_rw_tokens)
                        )

        if cosine_grid_batch_metrics is not None and linear_grid_batch_metrics is not None:
            n_grid_records = len(cosine_grid_batch_metrics)
            if n_grid_records > 0:
                grid_indices = torch.randint(0, n_grid_records, (n_grid_records,), generator=rng)
                cosine_grid_ce = []
                linear_grid_ce = []
                cosine_grid_acc = []
                linear_grid_acc = []
                for idx in grid_indices.tolist():
                    cosine_rec = cosine_grid_batch_metrics[idx]
                    linear_rec = linear_grid_batch_metrics[idx]
                    cosine_grid_ce.append(float(cosine_rec["avg_cross_entropy"]))
                    linear_grid_ce.append(float(linear_rec["avg_cross_entropy"]))
                    cosine_grid_acc.append(float(cosine_rec["masked_token_accuracy"]))
                    linear_grid_acc.append(float(linear_rec["masked_token_accuracy"]))
                if cosine_grid_ce and linear_grid_ce:
                    cosine_uniform_ce = sum(cosine_grid_ce) / len(cosine_grid_ce)
                    linear_uniform_ce = sum(linear_grid_ce) / len(linear_grid_ce)
                    grid_delta_samples["grid_uniform_avg_cross_entropy"].append(linear_uniform_ce - cosine_uniform_ce)
                    grid_delta_samples["grid_uniform_pseudo_perplexity"].append(_safe_exp(linear_uniform_ce) - _safe_exp(cosine_uniform_ce))
                    grid_delta_samples["grid_uniform_bits_per_masked_token"].append((linear_uniform_ce - cosine_uniform_ce) / math.log(2.0))
                    grid_delta_samples["grid_uniform_masked_token_accuracy"].append(
                        (sum(linear_grid_acc) / len(linear_grid_acc)) - (sum(cosine_grid_acc) / len(cosine_grid_acc))
                    )

        if shared_timestep_keys:
            n_timesteps = len(shared_timestep_keys)
            timestep_indices = torch.randint(0, n_timesteps, (n_timesteps,), generator=rng)
            cosine_sampled_timestep_metrics = []
            linear_sampled_timestep_metrics = []
            for rank, idx in enumerate(timestep_indices.tolist()):
                timestep = shared_timestep_keys[idx]
                cosine_rec = dict(cosine_timestep_by_step[timestep])
                linear_rec = dict(linear_timestep_by_step[timestep])
                bootstrap_fraction = float(rank) / float(max(1, n_timesteps - 1)) if n_timesteps > 1 else float(cosine_rec.get("timestep_fraction", 1.0))
                cosine_rec["timestep_fraction"] = bootstrap_fraction
                linear_rec["timestep_fraction"] = bootstrap_fraction
                cosine_sampled_timestep_metrics.append(cosine_rec)
                linear_sampled_timestep_metrics.append(linear_rec)

            cosine_macro_metrics = _compute_timestep_macro_metrics(cosine_sampled_timestep_metrics)
            linear_macro_metrics = _compute_timestep_macro_metrics(linear_sampled_timestep_metrics)
            cosine_auc_metrics = _compute_timestep_auc_metrics(cosine_sampled_timestep_metrics)
            linear_auc_metrics = _compute_timestep_auc_metrics(linear_sampled_timestep_metrics)
            timestep_macro_delta_samples["timestep_macro_avg_cross_entropy"].append(
                float(linear_macro_metrics["timestep_macro_avg_cross_entropy"]) - float(cosine_macro_metrics["timestep_macro_avg_cross_entropy"])
            )
            timestep_macro_delta_samples["timestep_macro_pseudo_perplexity"].append(
                float(linear_macro_metrics["timestep_macro_pseudo_perplexity"]) - float(cosine_macro_metrics["timestep_macro_pseudo_perplexity"])
            )
            timestep_macro_delta_samples["timestep_macro_bits_per_masked_token"].append(
                float(linear_macro_metrics["timestep_macro_bits_per_masked_token"]) - float(cosine_macro_metrics["timestep_macro_bits_per_masked_token"])
            )
            timestep_macro_delta_samples["timestep_macro_masked_token_accuracy"].append(
                float(linear_macro_metrics["timestep_macro_masked_token_accuracy"]) - float(cosine_macro_metrics["timestep_macro_masked_token_accuracy"])
            )
            timestep_macro_delta_samples["timestep_auc_avg_cross_entropy"].append(
                float(linear_auc_metrics["timestep_auc_avg_cross_entropy"]) - float(cosine_auc_metrics["timestep_auc_avg_cross_entropy"])
            )
            timestep_macro_delta_samples["timestep_auc_pseudo_perplexity"].append(
                float(linear_auc_metrics["timestep_auc_pseudo_perplexity"]) - float(cosine_auc_metrics["timestep_auc_pseudo_perplexity"])
            )
            timestep_macro_delta_samples["timestep_auc_bits_per_masked_token"].append(
                float(linear_auc_metrics["timestep_auc_bits_per_masked_token"]) - float(cosine_auc_metrics["timestep_auc_bits_per_masked_token"])
            )
            timestep_macro_delta_samples["timestep_auc_masked_token_accuracy"].append(
                float(linear_auc_metrics["timestep_auc_masked_token_accuracy"]) - float(cosine_auc_metrics["timestep_auc_masked_token_accuracy"])
            )

    def _with_win_probability(metric_name: str, values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {**_metric_percentiles(values), "probability_linear_better": float("nan")}
        higher_is_better = metric_name in {"masked_token_accuracy", "timestep_uniform_masked_token_accuracy", "schedule_reweighted_masked_token_accuracy", "grid_uniform_masked_token_accuracy", "timestep_macro_masked_token_accuracy", "timestep_auc_masked_token_accuracy"}
        wins = sum(v > 0.0 for v in values) if higher_is_better else sum(v < 0.0 for v in values)
        return {**_metric_percentiles(values), "probability_linear_better": float(wins / len(values))}

    summary = {key: _with_win_probability(key, values) for key, values in batch_delta_samples.items()}
    summary.update({key: _with_win_probability(key, values) for key, values in example_delta_samples.items()})
    summary.update({key: _with_win_probability(key, values) for key, values in grid_delta_samples.items()})
    summary.update({key: _with_win_probability(key, values) for key, values in timestep_macro_delta_samples.items()})

    if vocab_size is not None and int(vocab_size) > 1:
        uniform_ce = math.log(float(vocab_size))
        calibration_delta_map = {
            "sampled_bits_saved_vs_uniform": ("bits_per_masked_token", -1.0),
            "sampled_denoising_skill": ("avg_cross_entropy", -1.0 / uniform_ce),
            "timestep_uniform_bits_saved_vs_uniform": ("timestep_uniform_bits_per_masked_token", -1.0),
            "timestep_uniform_denoising_skill": ("timestep_uniform_avg_cross_entropy", -1.0 / uniform_ce),
            "schedule_reweighted_bits_saved_vs_uniform": ("schedule_reweighted_bits_per_masked_token", -1.0),
            "schedule_reweighted_denoising_skill": ("schedule_reweighted_avg_cross_entropy", -1.0 / uniform_ce),
            "grid_uniform_bits_saved_vs_uniform": ("grid_uniform_bits_per_masked_token", -1.0),
            "grid_uniform_denoising_skill": ("grid_uniform_avg_cross_entropy", -1.0 / uniform_ce),
            "timestep_macro_bits_saved_vs_uniform": ("timestep_macro_bits_per_masked_token", -1.0),
            "timestep_macro_denoising_skill": ("timestep_macro_avg_cross_entropy", -1.0 / uniform_ce),
            "timestep_auc_bits_saved_vs_uniform": ("timestep_auc_bits_per_masked_token", -1.0),
            "timestep_auc_denoising_skill": ("timestep_auc_avg_cross_entropy", -1.0 / uniform_ce),
        }
        for calibration_key, (source_key, scale) in calibration_delta_map.items():
            source_summary = summary.get(source_key)
            if not source_summary:
                continue
            summary[calibration_key] = {
                "mean": float(source_summary.get("mean", float("nan"))) * scale,
                "p05": float(source_summary.get("p95", float("nan"))) * scale if scale < 0.0 else float(source_summary.get("p05", float("nan"))) * scale,
                "p50": float(source_summary.get("p50", float("nan"))) * scale,
                "p95": float(source_summary.get("p05", float("nan"))) * scale if scale < 0.0 else float(source_summary.get("p95", float("nan"))) * scale,
                "probability_linear_better": 1.0 - float(source_summary.get("probability_linear_better", float("nan"))),
            }
    return {
        "method": "paired_bootstrap_over_shared_eval_plan",
        "n_pairs": n_batches,
        "replicates": n_samples,
        "delta_linear_minus_cosine": summary,
    }


def mask_ratio_cosine_schedule(t: Tensor, T: int, offset: float = 0.008, exponent: float = 2.0) -> Tensor:
    normalized_t = t.float() / float(T)
    v_start = math.cos(offset / (1.0 + offset) * math.pi / 2.0) ** exponent
    v_current = torch.cos(((normalized_t + offset) / (1.0 + offset)) * (math.pi / 2.0)) ** exponent
    ratio = 1.0 - (v_current / v_start)
    return torch.clamp(ratio, min=0.0, max=1.0)


def mask_ratio_linear_schedule(t: Tensor, T: int) -> Tensor:
    return torch.clamp(t.float() / float(T), min=0.0, max=1.0)


def _resolve_schedule_fn(schedule_name: Optional[str]) -> Optional[Callable[[Tensor, int], Tensor]]:
    if schedule_name is None:
        return None
    normalized = schedule_name.lower().strip()
    if normalized == "cosine":
        return mask_ratio_cosine_schedule
    if normalized == "linear":
        return mask_ratio_linear_schedule
    raise ValueError(f"Unknown schedule_name={schedule_name!r}. Expected 'cosine' or 'linear'.")


def corruption_factory(schedule_name: str) -> Callable[..., Tuple[Tensor, Tensor, Tensor]]:
    schedule_fn = _resolve_schedule_fn(schedule_name)
    if schedule_fn is None:
        raise ValueError(f"Unknown schedule_name={schedule_name!r}. Expected 'cosine' or 'linear'.")

    def _corrupt_with_mask(
        input_ids: Tensor,
        attention_mask: Tensor,
        t: Tensor,
        mask_token_id: int,
        T: int,
        excluded_token_ids: Optional[Sequence[Optional[int]]] = None,
        generator: Optional[torch.Generator] = None,
        rand: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return corrupt_with_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            t=t,
            mask_token_id=mask_token_id,
            T=T,
            schedule_fn=schedule_fn,
            excluded_token_ids=excluded_token_ids,
            generator=generator,
            rand=rand,
        )

    return _corrupt_with_mask


def corrupt_with_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    t: Tensor,
    mask_token_id: int,
    T: int,
    schedule_fn: Callable[[Tensor, int], Tensor],
    excluded_token_ids: Optional[Sequence[Optional[int]]] = None,
    generator: Optional[torch.Generator] = None,
    rand: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    B, L = input_ids.shape
    ratio = schedule_fn(t, T).unsqueeze(1)

    can_mask = attention_mask.bool().clone()
    for token_id in _normalize_excluded_token_ids(excluded_token_ids):
        can_mask &= input_ids != token_id

    if rand is None:
        rand = torch.rand((B, L), device=input_ids.device, generator=generator)
    else:
        rand = rand.to(input_ids.device)
        if tuple(rand.shape) != (B, L):
            raise ValueError(f"rand must have shape {(B, L)}, got {tuple(rand.shape)}")

    mask_positions = (rand < ratio) & can_mask

    noisy = input_ids.clone()
    noisy[mask_positions] = mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]
    return noisy, labels, mask_positions


@torch.no_grad()
def build_eval_plan(
    dataloader: Iterable[Dict[str, Tensor]],
    *,
    T: int,
    n_batches: int = 20,
    timestep_grid: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    resolved_grid = _resolve_timestep_grid(T, timestep_grid)
    plan_batches: List[Dict[str, object]] = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        input_ids = batch["input_ids"].detach().cpu()
        attention_mask = batch["attention_mask"].detach().cpu()
        B, L = input_ids.shape
        sampled_t = torch.randint(1, T + 1, (B,), generator=generator)
        sampled_u = sampled_t.float() / float(T)

        timestep_plans = [
            {
                "kind": "sampled",
                "t": sampled_t,
                "u": sampled_u,
                "rand": torch.rand((B, L), generator=generator),
            }
        ]
        for t in resolved_grid:
            timestep_plans.append(
                {
                    "kind": "grid",
                    "t": torch.full((B,), int(t), dtype=torch.long),
                    "u": torch.full((B,), float(t) / float(T), dtype=torch.float32),
                    "rand": torch.rand((B, L), generator=generator),
                }
            )

        plan_batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "timestep_plans": timestep_plans,
                "batch_size": B,
                "seq_len": L,
                "active_tokens": int(attention_mask.sum().item()),
            }
        )

    return {
        "n_batches": len(plan_batches),
        "seed": seed,
        "T": T,
        "timestep_grid": resolved_grid,
        "batches": plan_batches,
    }


def _empty_eval_result(
    n_batches: int,
    seed: int,
    excluded_token_ids: Optional[Sequence[Optional[int]]],
    vocab_size: Optional[int] = None,
) -> Dict[str, object]:
    return {
        "metric": "diffusion_pseudo_perplexity",
        **_uniform_baseline_summary(vocab_size),
        "avg_cross_entropy": float("nan"),
        "pseudo_perplexity": float("nan"),
        "bits_per_masked_token": float("nan"),
        "masked_token_accuracy": float("nan"),
        "timestep_uniform_avg_cross_entropy": float("nan"),
        "timestep_uniform_pseudo_perplexity": float("nan"),
        "timestep_uniform_bits_per_masked_token": float("nan"),
        "timestep_uniform_masked_token_accuracy": float("nan"),
        "schedule_reweighted_avg_cross_entropy": float("nan"),
        "schedule_reweighted_pseudo_perplexity": float("nan"),
        "schedule_reweighted_bits_per_masked_token": float("nan"),
        "schedule_reweighted_masked_token_accuracy": float("nan"),
        "schedule_reweighted_effective_sample_size": float("nan"),
        "schedule_reweighted_effective_sample_size_fraction": float("nan"),
        "schedule_reweighted_nonzero_examples": 0,
        "schedule_reweighted_estimated_eligible_token_count": 0.0,
        "timestep_uniform_confidence_intervals": _bootstrap_example_uniform_metric_interval([], n_samples=0, seed=seed),
        "grid_uniform_avg_cross_entropy": float("nan"),
        "grid_uniform_pseudo_perplexity": float("nan"),
        "grid_uniform_bits_per_masked_token": float("nan"),
        "grid_uniform_masked_token_accuracy": float("nan"),
        "timestep_macro_avg_cross_entropy": float("nan"),
        "timestep_macro_pseudo_perplexity": float("nan"),
        "timestep_macro_bits_per_masked_token": float("nan"),
        "timestep_macro_masked_token_accuracy": float("nan"),
        "timestep_macro_timestep_count": 0,
        "timestep_auc_avg_cross_entropy": float("nan"),
        "timestep_auc_pseudo_perplexity": float("nan"),
        "timestep_auc_bits_per_masked_token": float("nan"),
        "timestep_auc_masked_token_accuracy": float("nan"),
        "timestep_auc_timestep_count": 0,
        "timestep_auc_fraction_span": float("nan"),
        "sampled_example_count": 0,
        "masked_tokens": 0,
        "n_batches": n_batches,
        "seed": seed,
        "timestep_metrics": [],
        "sampled_batch_metrics": [],
        "sampled_example_metrics": [],
        "grid_batch_metrics": [],
        "sampled_timestep_histogram": {},
        "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
        "confidence_intervals": _bootstrap_metric_interval([], n_samples=0, seed=seed),
        "schedule_reweighted_confidence_intervals": _bootstrap_reweighted_metric_interval([], weight_key="schedule_reweighted_weight", n_samples=0, seed=seed),
        "grid_uniform_confidence_intervals": _bootstrap_grid_uniform_metric_interval([], n_samples=0, seed=seed),
        "timestep_confidence_intervals": _bootstrap_timestep_metric_interval([], n_samples=0, seed=seed),
        "calibration": {},
        "notes": [
            "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
            "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
            "Timestep-uniform CE reports the mean masked-token CE across uniformly sampled timesteps, decoupling the metric from schedule-dependent mask counts.",
            "Schedule-reweighted CE applies inverse expected mask-ratio weights to sampled masked tokens, estimating a uniform-over-mask-eligible-token-and-timestep denoising objective directly from sampled batches.",
            "Grid-uniform CE reports the mean batch-level denoising CE over a fixed cached timestep grid, giving an explicit schedule-agnostic diagnostic aggregate.",
        ],
    }


@torch.no_grad()
def evaluate_diffusion_pseudo_perplexity_from_plan(
    model,
    eval_plan: Dict[str, object],
    corruption_fn: Callable[..., Tuple[Tensor, Tensor, Tensor]],
    mask_token_id: int,
    T: int,
    excluded_token_ids: Optional[Sequence[Optional[int]]] = None,
    schedule_name: Optional[str] = None,
    bootstrap_samples: int = 500,
) -> Dict[str, object]:
    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_masked_tokens = 0
    total_correct = 0.0
    sampled_example_ce_sum = 0.0
    observed_vocab_size: Optional[int] = None
    sampled_example_accuracy_sum = 0.0
    sampled_example_count = 0
    schedule_fn = _resolve_schedule_fn(schedule_name)
    schedule_reweighted_nll_sum = 0.0
    schedule_reweighted_masked_tokens = 0.0
    schedule_reweighted_correct = 0.0
    schedule_reweighted_token_masses: List[float] = []
    schedule_reweighted_nonzero_examples = 0
    timestep_records: Dict[int, Dict[str, float]] = {}
    sampled_batch_metrics: List[Dict[str, float]] = []
    sampled_example_metrics: List[Dict[str, float]] = []
    grid_batch_metrics: List[Dict[str, float]] = []
    sampled_timestep_histogram: Dict[int, int] = {}

    for batch_idx, batch_plan in enumerate(eval_plan["batches"]):
        input_ids = batch_plan["input_ids"].to(device)
        attention_mask = batch_plan["attention_mask"].to(device)
        batch_active_tokens = int(batch_plan.get("active_tokens", int(attention_mask.sum().item())))

        for timestep_plan in batch_plan["timestep_plans"]:
            eval_t, timestep_fraction, source_timestep = _resolve_eval_timesteps(timestep_plan, T, device)
            rand = timestep_plan["rand"].to(device)
            noisy_ids, labels, mask_positions = corruption_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                t=eval_t,
                mask_token_id=mask_token_id,
                T=T,
                excluded_token_ids=excluded_token_ids,
                rand=rand,
            )
            masked_count = int(mask_positions.sum().item())
            if masked_count == 0:
                continue

            logits = model(noisy_ids, timesteps=eval_t, attention_mask=attention_mask)
            if observed_vocab_size is None:
                observed_vocab_size = int(logits.size(-1))
            masked_logits = logits[mask_positions]
            masked_targets = labels[mask_positions]
            token_nll_sum = float(F.cross_entropy(masked_logits, masked_targets, reduction="sum").item())
            correct = _expected_top1_correct_count(masked_logits, masked_targets)

            if timestep_plan["kind"] == "sampled":
                total_nll += token_nll_sum
                total_masked_tokens += masked_count
                total_correct += correct

                batch_ce = token_nll_sum / masked_count
                t_values = eval_t.detach().cpu().tolist()
                u_values = timestep_fraction.detach().cpu().tolist()
                source_t_values = source_timestep.tolist()
                for value in t_values:
                    value = int(value)
                    sampled_timestep_histogram[value] = sampled_timestep_histogram.get(value, 0) + 1
                sampled_batch_metrics.append(
                    {
                        "batch_index": batch_idx,
                        "nll_sum": token_nll_sum,
                        "masked_tokens": masked_count,
                        "correct_masked_tokens": float(correct),
                        "avg_cross_entropy": batch_ce,
                        "pseudo_perplexity": _safe_exp(batch_ce),
                        "bits_per_masked_token": batch_ce / math.log(2.0),
                        "masked_token_accuracy": correct / masked_count,
                        "mean_mask_fraction": masked_count / max(1, batch_active_tokens),
                        "sampled_timesteps": t_values,
                        "sampled_timestep_fractions": [float(v) for v in u_values],
                        "source_plan_timesteps": [int(v) for v in source_t_values],
                    }
                )

                for example_idx in range(input_ids.size(0)):
                    example_mask = mask_positions[example_idx]
                    example_masked_tokens = int(example_mask.sum().item())
                    if example_masked_tokens == 0:
                        continue
                    example_logits = logits[example_idx][example_mask]
                    example_targets = labels[example_idx][example_mask]
                    example_nll_sum = float(F.cross_entropy(example_logits, example_targets, reduction="sum").item())
                    example_ce = example_nll_sum / example_masked_tokens
                    example_correct = _expected_top1_correct_count(example_logits, example_targets)
                    example_accuracy = example_correct / example_masked_tokens
                    sampled_example_ce_sum += example_ce
                    sampled_example_accuracy_sum += example_accuracy
                    sampled_example_count += 1
                    schedule_reweighted_weight = 0.0
                    if schedule_fn is not None:
                        expected_mask_ratio = float(schedule_fn(eval_t[example_idx:example_idx + 1], T).item())
                        if expected_mask_ratio > 0.0:
                            schedule_reweighted_weight = 1.0 / expected_mask_ratio
                            weighted_token_mass = example_masked_tokens * schedule_reweighted_weight
                            schedule_reweighted_nll_sum += example_nll_sum * schedule_reweighted_weight
                            schedule_reweighted_masked_tokens += weighted_token_mass
                            schedule_reweighted_correct += example_correct * schedule_reweighted_weight
                            schedule_reweighted_token_masses.append(float(weighted_token_mass))
                            schedule_reweighted_nonzero_examples += 1
                    sampled_example_metrics.append(
                        {
                            "batch_index": batch_idx,
                            "example_index": example_idx,
                            "timestep": int(eval_t[example_idx].item()),
                            "source_plan_timestep": int(source_timestep[example_idx].item()),
                            "timestep_fraction": float(timestep_fraction[example_idx].item()),
                            "nll_sum": example_nll_sum,
                            "masked_tokens": example_masked_tokens,
                            "correct_masked_tokens": example_correct,
                            "avg_cross_entropy": example_ce,
                            "pseudo_perplexity": _safe_exp(example_ce),
                            "bits_per_masked_token": example_ce / math.log(2.0),
                            "masked_token_accuracy": example_accuracy,
                            "schedule_reweighted_weight": schedule_reweighted_weight,
                        }
                    )
            else:
                key = int(source_timestep[0].item())
                batch_ce = token_nll_sum / masked_count
                batch_accuracy = correct / masked_count
                rec = timestep_records.setdefault(
                    key,
                    {
                        "nll_sum": 0.0,
                        "masked_tokens": 0,
                        "mask_ratio_sum": 0.0,
                        "examples": 0,
                        "correct": 0,
                        "timestep": int(eval_t[0].item()),
                        "source_plan_timestep": key,
                        "timestep_fraction": float(timestep_fraction[0].item()),
                    },
                )
                rec["nll_sum"] += token_nll_sum
                rec["masked_tokens"] += masked_count
                rec["mask_ratio_sum"] += masked_count / max(1, batch_active_tokens)
                rec["examples"] += 1
                rec["correct"] += correct
                grid_batch_metrics.append(
                    {
                        "batch_index": batch_idx,
                        "timestep": int(eval_t[0].item()),
                        "source_plan_timestep": key,
                        "timestep_fraction": float(timestep_fraction[0].item()),
                        "nll_sum": token_nll_sum,
                        "masked_tokens": masked_count,
                        "correct_masked_tokens": float(correct),
                        "avg_cross_entropy": batch_ce,
                        "pseudo_perplexity": _safe_exp(batch_ce),
                        "bits_per_masked_token": batch_ce / math.log(2.0),
                        "masked_token_accuracy": batch_accuracy,
                        "mean_mask_fraction": masked_count / max(1, batch_active_tokens),
                    }
                )

    timestep_metrics = []
    for t in sorted(timestep_records):
        rec = timestep_records[t]
        if rec["masked_tokens"] == 0:
            continue
        ce = rec["nll_sum"] / rec["masked_tokens"]
        timestep_metrics.append(
            {
                "timestep": int(rec.get("timestep", t)),
                "source_plan_timestep": int(rec.get("source_plan_timestep", t)),
                "timestep_fraction": float(rec.get("timestep_fraction", _timestep_fraction_from_int(t, T))),
                "avg_cross_entropy": ce,
                "pseudo_perplexity": _safe_exp(ce),
                "bits_per_masked_token": ce / math.log(2.0),
                "masked_tokens": rec["masked_tokens"],
                "mean_mask_fraction": rec["mask_ratio_sum"] / max(1, rec["examples"]),
                "masked_token_accuracy": rec["correct"] / rec["masked_tokens"],
            }
        )

    if total_masked_tokens == 0 and not grid_batch_metrics:
        result = _empty_eval_result(
            n_batches=int(eval_plan.get("n_batches", 0)),
            seed=int(eval_plan.get("seed", 0)),
            excluded_token_ids=excluded_token_ids,
            vocab_size=observed_vocab_size,
        )
    else:
        aggregate_nll = total_nll
        aggregate_masked_tokens = total_masked_tokens
        aggregate_correct = total_correct
        if aggregate_masked_tokens == 0 and grid_batch_metrics:
            aggregate_nll = sum(float(row["nll_sum"]) for row in grid_batch_metrics)
            aggregate_masked_tokens = sum(int(row["masked_tokens"]) for row in grid_batch_metrics)
            aggregate_correct = sum(float(row.get("correct_masked_tokens", 0.0)) for row in grid_batch_metrics)

        avg_ce = aggregate_nll / aggregate_masked_tokens if aggregate_masked_tokens > 0 else float("nan")

        if sampled_example_count:
            timestep_uniform_avg_ce = sampled_example_ce_sum / sampled_example_count
            timestep_uniform_masked_token_accuracy = sampled_example_accuracy_sum / sampled_example_count
        elif timestep_metrics:
            timestep_uniform_avg_ce = sum(float(row["avg_cross_entropy"]) for row in timestep_metrics) / len(timestep_metrics)
            timestep_uniform_masked_token_accuracy = sum(float(row["masked_token_accuracy"]) for row in timestep_metrics) / len(timestep_metrics)
        else:
            timestep_uniform_avg_ce = float("nan")
            timestep_uniform_masked_token_accuracy = float("nan")

        if schedule_reweighted_masked_tokens > 0.0:
            schedule_reweighted_avg_ce = schedule_reweighted_nll_sum / schedule_reweighted_masked_tokens
            schedule_reweighted_masked_token_accuracy = schedule_reweighted_correct / schedule_reweighted_masked_tokens
        elif timestep_metrics:
            schedule_reweighted_avg_ce = sum(float(row["avg_cross_entropy"]) for row in timestep_metrics) / len(timestep_metrics)
            schedule_reweighted_masked_token_accuracy = sum(float(row["masked_token_accuracy"]) for row in timestep_metrics) / len(timestep_metrics)
        else:
            schedule_reweighted_avg_ce = float("nan")
            schedule_reweighted_masked_token_accuracy = float("nan")

        if schedule_reweighted_token_masses:
            schedule_reweighted_mass_sum = float(sum(schedule_reweighted_token_masses))
            schedule_reweighted_mass_sq_sum = float(sum(mass * mass for mass in schedule_reweighted_token_masses))
            schedule_reweighted_effective_sample_size = (
                (schedule_reweighted_mass_sum * schedule_reweighted_mass_sum) / schedule_reweighted_mass_sq_sum
                if schedule_reweighted_mass_sq_sum > 0.0
                else float("nan")
            )
            schedule_reweighted_effective_sample_size_fraction = (
                schedule_reweighted_effective_sample_size / float(len(schedule_reweighted_token_masses))
                if schedule_reweighted_token_masses
                else float("nan")
            )
        else:
            schedule_reweighted_mass_sum = 0.0
            schedule_reweighted_effective_sample_size = float("nan")
            schedule_reweighted_effective_sample_size_fraction = float("nan")

        if grid_batch_metrics:
            grid_uniform_avg_ce = sum(float(row["avg_cross_entropy"]) for row in grid_batch_metrics) / len(grid_batch_metrics)
            grid_uniform_masked_token_accuracy = sum(float(row["masked_token_accuracy"]) for row in grid_batch_metrics) / len(grid_batch_metrics)
        else:
            grid_uniform_avg_ce = float("nan")
            grid_uniform_masked_token_accuracy = float("nan")
        timestep_macro_metrics = _compute_timestep_macro_metrics(timestep_metrics)
        timestep_auc_metrics = _compute_timestep_auc_metrics(timestep_metrics)
        ci_example_metrics = sampled_example_metrics
        if not ci_example_metrics and timestep_metrics:
            ci_example_metrics = [
                {
                    "batch_index": -1,
                    "example_index": idx,
                    "timestep": int(row["timestep"]),
                    "source_plan_timestep": int(row.get("source_plan_timestep", row["timestep"])),
                    "timestep_fraction": float(row["timestep_fraction"]),
                    "nll_sum": float(row["avg_cross_entropy"]) * float(row["masked_tokens"]),
                    "masked_tokens": int(row["masked_tokens"]),
                    "correct_masked_tokens": int(round(float(row["masked_token_accuracy"]) * float(row["masked_tokens"]))),
                    "avg_cross_entropy": float(row["avg_cross_entropy"]),
                    "pseudo_perplexity": float(row["pseudo_perplexity"]),
                    "bits_per_masked_token": float(row["bits_per_masked_token"]),
                    "masked_token_accuracy": float(row["masked_token_accuracy"]),
                    "schedule_reweighted_weight": 1.0,
                }
                for idx, row in enumerate(timestep_metrics)
            ]

        result = {
            "metric": "diffusion_pseudo_perplexity",
            **_uniform_baseline_summary(observed_vocab_size),
            "avg_cross_entropy": avg_ce,
            "pseudo_perplexity": _safe_exp(avg_ce),
            "bits_per_masked_token": avg_ce / math.log(2.0),
            "masked_token_accuracy": aggregate_correct / aggregate_masked_tokens if aggregate_masked_tokens > 0 else float("nan"),
            "timestep_uniform_avg_cross_entropy": timestep_uniform_avg_ce,
            "timestep_uniform_pseudo_perplexity": _safe_exp(timestep_uniform_avg_ce),
            "timestep_uniform_bits_per_masked_token": timestep_uniform_avg_ce / math.log(2.0) if not math.isnan(timestep_uniform_avg_ce) else float("nan"),
            "timestep_uniform_masked_token_accuracy": timestep_uniform_masked_token_accuracy,
            "schedule_reweighted_avg_cross_entropy": schedule_reweighted_avg_ce,
            "schedule_reweighted_pseudo_perplexity": _safe_exp(schedule_reweighted_avg_ce),
            "schedule_reweighted_bits_per_masked_token": schedule_reweighted_avg_ce / math.log(2.0) if not math.isnan(schedule_reweighted_avg_ce) else float("nan"),
            "schedule_reweighted_masked_token_accuracy": schedule_reweighted_masked_token_accuracy,
            "schedule_reweighted_effective_sample_size": schedule_reweighted_effective_sample_size,
            "schedule_reweighted_effective_sample_size_fraction": schedule_reweighted_effective_sample_size_fraction,
            "schedule_reweighted_nonzero_examples": schedule_reweighted_nonzero_examples,
            "schedule_reweighted_estimated_eligible_token_count": schedule_reweighted_mass_sum,
            "grid_uniform_avg_cross_entropy": grid_uniform_avg_ce,
            "grid_uniform_pseudo_perplexity": _safe_exp(grid_uniform_avg_ce),
            "grid_uniform_bits_per_masked_token": grid_uniform_avg_ce / math.log(2.0) if not math.isnan(grid_uniform_avg_ce) else float("nan"),
            "grid_uniform_masked_token_accuracy": grid_uniform_masked_token_accuracy,
            **timestep_macro_metrics,
            **timestep_auc_metrics,
            "sampled_example_count": sampled_example_count,
            "masked_tokens": total_masked_tokens,
            "n_batches": int(eval_plan.get("n_batches", 0)),
            "seed": int(eval_plan.get("seed", 0)),
            "timestep_metrics": timestep_metrics,
            "sampled_batch_metrics": sampled_batch_metrics,
            "sampled_example_metrics": sampled_example_metrics,
            "grid_batch_metrics": grid_batch_metrics,
            "sampled_timestep_histogram": {str(k): int(v) for k, v in sorted(sampled_timestep_histogram.items())},
            "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
            "confidence_intervals": _bootstrap_metric_interval(
                sampled_batch_metrics,
                n_samples=bootstrap_samples,
                seed=int(eval_plan.get("seed", 0)),
            ),
            "timestep_uniform_confidence_intervals": _bootstrap_example_uniform_metric_interval(
                ci_example_metrics,
                n_samples=bootstrap_samples,
                seed=int(eval_plan.get("seed", 0)),
            ),
            "schedule_reweighted_confidence_intervals": _bootstrap_reweighted_metric_interval(
                ci_example_metrics,
                weight_key="schedule_reweighted_weight",
                n_samples=bootstrap_samples,
                seed=int(eval_plan.get("seed", 0)),
            ),
            "grid_uniform_confidence_intervals": _bootstrap_grid_uniform_metric_interval(
                grid_batch_metrics,
                n_samples=bootstrap_samples,
                seed=int(eval_plan.get("seed", 0)),
            ),
            "timestep_confidence_intervals": _bootstrap_timestep_metric_interval(
                timestep_metrics,
                n_samples=bootstrap_samples,
                seed=int(eval_plan.get("seed", 0)),
            ),
            "notes": [
                "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
                "The exported uniform-random baseline uses the model vocabulary size, so bits-saved and denoising-skill views are calibrated against random guessing on the same tokenizer.",
                "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
                "Timestep-uniform CE averages per-example masked-token CE over uniformly sampled timesteps, making schedule comparisons less sensitive to different mask-count profiles.",
                "Schedule-reweighted CE applies inverse expected mask-ratio weights to sampled masked tokens, estimating a uniform-over-mask-eligible-token-and-timestep denoising objective directly from sampled batches.",
                "Schedule-reweighted diagnostics expose an importance-sampling effective sample size (ESS), so users can tell when inverse-mask-ratio weighting becomes statistically fragile.",
                "Grid-uniform CE averages batch-level denoising CE over a fixed cached timestep grid, giving an explicit schedule-agnostic comparison surface.",
                "Timestep-macro CE averages token-weighted per-timestep denoising CE equally over the explicit diagnostic grid, making each denoising stage contribute the same top-level weight.",
                "Timestep-AUC CE integrates token-weighted per-timestep CE over normalized timestep fraction with trapezoidal weighting, so irregular diagnostic grids do not silently over-weight dense regions.",
                "Confidence intervals are computed by bootstrapping over sampled evaluation batches from the shared cached plan.",
                "Grid-uniform confidence intervals are computed by bootstrapping over cached batch-timestep diagnostic records.",
                "Timestep-macro and timestep-AUC confidence intervals are computed by bootstrapping over cached timestep diagnostics on the shared grid.",
                "Timestep diagnostics are evaluated on a shared cached batch/noise plan so schedule comparisons are paired and reproducible.",
            ],
        }

    result["eval_protocol"] = {
        "T": T,
        "schedule_name": schedule_name,
        "timestep_grid": list(eval_plan.get("timestep_grid", [])),
        "timestep_grid_fractions": [_timestep_fraction_from_int(t, int(eval_plan.get("T", T))) for t in list(eval_plan.get("timestep_grid", []))],
        "eval_plan_T": int(eval_plan.get("T", T)),
        "normalized_timestep_remapping": bool(int(eval_plan.get("T", T)) != int(T)),
        "paired_noise": True,
        "paired_batches": True,
        "sampled_timestep_distribution": "uniform_integer_1_to_eval_plan_T_then_remapped_by_fraction_per_model",
        "schedule_reweighted_aggregation": "inverse_expected_mask_ratio_weighting_over_sampled_masked_tokens",
        "grid_uniform_aggregation": "mean_over_cached_batch_timestep_records",
        "timestep_macro_aggregation": "mean_over_token_weighted_per_timestep_metrics_on_cached_grid",
        "timestep_auc_aggregation": "normalized_trapezoid_integral_over_token_weighted_per_timestep_metrics_on_cached_grid",
        "bootstrap_samples": bootstrap_samples,
    }
    result["calibration"] = _build_calibration_view_summaries(result)
    result["quality_summary"] = _build_eval_quality_summary(result)
    return result


@torch.no_grad()
def evaluate_diffusion_pseudo_perplexity(
    model,
    dataloader: Iterable[Dict[str, Tensor]],
    corruption_fn: Callable[..., Tuple[Tensor, Tensor, Tensor]],
    mask_token_id: int,
    T: int,
    n_batches: int = 20,
    timestep_grid: Optional[Sequence[int]] = None,
    excluded_token_ids: Optional[Sequence[Optional[int]]] = None,
    seed: int = 0,
    schedule_name: Optional[str] = None,
    bootstrap_samples: int = 500,
) -> Dict[str, object]:
    eval_plan = build_eval_plan(
        dataloader,
        T=T,
        n_batches=n_batches,
        timestep_grid=timestep_grid,
        seed=seed,
    )
    return evaluate_diffusion_pseudo_perplexity_from_plan(
        model=model,
        eval_plan=eval_plan,
        corruption_fn=corruption_fn,
        mask_token_id=mask_token_id,
        T=T,
        excluded_token_ids=excluded_token_ids,
        schedule_name=schedule_name,
        bootstrap_samples=bootstrap_samples,
    )


def load_diffusion_checkpoint(artifact_dir, device, config_cls, model_cls):
    artifact_dir = Path(artifact_dir)
    cfg_dict = json.loads((artifact_dir / "config.json").read_text())
    cfg = config_cls(**cfg_dict)
    model = model_cls(cfg).to(device)
    state = torch.load(artifact_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _calibration_interval_from_ce_ci(ce_ci: Optional[Dict[str, float]], vocab_size: Optional[int]) -> Dict[str, float]:
    if vocab_size is None or int(vocab_size) <= 0 or not ce_ci:
        return {
            "bits_saved_vs_uniform_ci_p05": float("nan"),
            "bits_saved_vs_uniform_ci_p95": float("nan"),
            "denoising_skill_ci_p05": float("nan"),
            "denoising_skill_ci_p95": float("nan"),
        }
    try:
        ce_p05 = float(ce_ci.get("p05", float("nan")))
        ce_p95 = float(ce_ci.get("p95", float("nan")))
    except (TypeError, ValueError):
        ce_p05 = float("nan")
        ce_p95 = float("nan")
    if math.isnan(ce_p05) or math.isnan(ce_p95):
        return {
            "bits_saved_vs_uniform_ci_p05": float("nan"),
            "bits_saved_vs_uniform_ci_p95": float("nan"),
            "denoising_skill_ci_p05": float("nan"),
            "denoising_skill_ci_p95": float("nan"),
        }

    uniform_ce = math.log(float(vocab_size))
    uniform_bits = uniform_ce / math.log(2.0)
    return {
        "bits_saved_vs_uniform_ci_p05": uniform_bits - (ce_p95 / math.log(2.0)),
        "bits_saved_vs_uniform_ci_p95": uniform_bits - (ce_p05 / math.log(2.0)),
        "denoising_skill_ci_p05": 1.0 - (ce_p95 / uniform_ce),
        "denoising_skill_ci_p95": 1.0 - (ce_p05 / uniform_ce),
    }


EVAL_CALIBRATION_VIEW_SPECS = {
    "sampled": {
        "avg_cross_entropy_key": "avg_cross_entropy",
        "bits_key": "bits_per_masked_token",
        "ci_container_key": "confidence_intervals",
        "ci_ce_key": "avg_cross_entropy",
    },
    "timestep_uniform": {
        "avg_cross_entropy_key": "timestep_uniform_avg_cross_entropy",
        "bits_key": "timestep_uniform_bits_per_masked_token",
        "ci_container_key": "timestep_uniform_confidence_intervals",
        "ci_ce_key": "timestep_uniform_avg_cross_entropy",
    },
    "schedule_reweighted": {
        "avg_cross_entropy_key": "schedule_reweighted_avg_cross_entropy",
        "bits_key": "schedule_reweighted_bits_per_masked_token",
        "ci_container_key": "schedule_reweighted_confidence_intervals",
        "ci_ce_key": "schedule_reweighted_avg_cross_entropy",
    },
    "grid_uniform": {
        "avg_cross_entropy_key": "grid_uniform_avg_cross_entropy",
        "bits_key": "grid_uniform_bits_per_masked_token",
        "ci_container_key": "grid_uniform_confidence_intervals",
        "ci_ce_key": "grid_uniform_avg_cross_entropy",
    },
    "timestep_macro": {
        "avg_cross_entropy_key": "timestep_macro_avg_cross_entropy",
        "bits_key": "timestep_macro_bits_per_masked_token",
        "ci_container_key": "timestep_confidence_intervals",
        "ci_ce_key": "timestep_macro_avg_cross_entropy",
    },
    "timestep_auc": {
        "avg_cross_entropy_key": "timestep_auc_avg_cross_entropy",
        "bits_key": "timestep_auc_bits_per_masked_token",
        "ci_container_key": "timestep_confidence_intervals",
        "ci_ce_key": "timestep_auc_avg_cross_entropy",
    },
}


def _build_calibration_view_summaries(summary: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    vocab_size = summary.get("vocab_size")
    calibration_views: Dict[str, Dict[str, float]] = {}
    for view_name, spec in EVAL_CALIBRATION_VIEW_SPECS.items():
        ci_container = summary.get(spec["ci_container_key"]) or {}
        ce_ci = ci_container.get(spec["ci_ce_key"]) or {}
        calibration_views[view_name] = {
            **_view_calibration(
                summary.get(spec["avg_cross_entropy_key"]),
                summary.get(spec["bits_key"]),
                vocab_size,
            ),
            **_calibration_interval_from_ce_ci(ce_ci, vocab_size),
        }
    return calibration_views


def _comparison_calibration_views(summary: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    return _build_calibration_view_summaries(summary)


def _paired_delta_confidence_summary(
    delta_summary: Optional[Dict[str, object]],
    *,
    better_direction: str,
) -> Dict[str, object]:
    if not delta_summary:
        return {
            "ci_excludes_zero": False,
            "winner_probability": float("nan"),
            "loser_probability": float("nan"),
            "practically_tied": True,
        }

    p05 = delta_summary.get("p05")
    p95 = delta_summary.get("p95")
    probability_linear_better = delta_summary.get("probability_linear_better")
    try:
        p05 = float(p05)
        p95 = float(p95)
        probability_linear_better = float(probability_linear_better)
    except (TypeError, ValueError):
        return {
            "ci_excludes_zero": False,
            "winner_probability": float("nan"),
            "loser_probability": float("nan"),
            "practically_tied": True,
        }

    if math.isnan(probability_linear_better):
        winner_probability = float("nan")
        loser_probability = float("nan")
    else:
        winner_probability = probability_linear_better if better_direction == "higher" else 1.0 - probability_linear_better
        loser_probability = 1.0 - winner_probability

    ci_excludes_zero = (p05 > 0.0) or (p95 < 0.0)
    return {
        "ci_excludes_zero": bool(ci_excludes_zero),
        "winner_probability": float(winner_probability),
        "loser_probability": float(loser_probability),
        "practically_tied": not bool(ci_excludes_zero),
    }


def _schedule_reweighted_reliability_bucket(effective_sample_size_fraction: float) -> str:
    if effective_sample_size_fraction is None:
        return "unknown"
    try:
        value = float(effective_sample_size_fraction)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(value):
        return "unknown"
    if value >= 0.5:
        return "strong"
    if value >= 0.25:
        return "usable"
    return "fragile"


def _recommend_primary_eval_view(result: Dict[str, object]) -> Dict[str, object]:
    quality_summary = result.get("quality_summary") or {}
    sampled_example_count = int(result.get("sampled_example_count", 0) or 0)
    timestep_count = int(result.get("timestep_macro_timestep_count", 0) or 0)
    grid_count = len(result.get("grid_batch_metrics") or [])
    ess_bucket = quality_summary.get("schedule_reweighted_reliability") or _schedule_reweighted_reliability_bucket(
        result.get("schedule_reweighted_effective_sample_size_fraction")
    )

    if sampled_example_count > 0 and ess_bucket in {"strong", "usable"}:
        return {
            "view": "schedule_reweighted_sampled",
            "metric_key": "schedule_reweighted_pseudo_perplexity",
            "better_direction": "lower",
            "rationale": "Use the schedule-reweighted sampled pseudo-perplexity as the primary diffusion perplexity-style metric because it directly estimates a uniform-over-mask-eligible-token-and-timestep denoising objective and the ESS diagnostic is not fragile.",
            "caveat": None if ess_bucket == "strong" else "ESS is usable rather than strong, so small deltas should still be cross-checked against timestep-AUC or timestep-uniform views.",
        }
    if timestep_count >= 3:
        return {
            "view": "fixed_grid_timestep_auc",
            "metric_key": "timestep_auc_pseudo_perplexity",
            "better_direction": "lower",
            "rationale": "Use the fixed-grid timestep-AUC pseudo-perplexity as the primary fallback because it integrates over normalized timestep fraction on an explicit shared grid and is less sensitive to fragile importance weights.",
            "caveat": "This is a grid diagnostic rather than a sampled objective estimate, so keep the schedule-reweighted view visible when its ESS becomes strong.",
        }
    if sampled_example_count > 0:
        return {
            "view": "timestep_uniform_sampled",
            "metric_key": "timestep_uniform_pseudo_perplexity",
            "better_direction": "lower",
            "rationale": "Use the timestep-uniform sampled pseudo-perplexity as the primary fallback because it keeps equal top-level weight on sampled timesteps without relying on inverse-mask-ratio importance weights.",
            "caveat": "This view is less target-faithful than schedule-reweighted evaluation when ESS is strong.",
        }
    if grid_count > 0:
        return {
            "view": "fixed_grid_batch_uniform",
            "metric_key": "grid_uniform_pseudo_perplexity",
            "better_direction": "lower",
            "rationale": "Use the fixed-grid batch-uniform pseudo-perplexity as a last-resort fallback because sampled per-example metrics were unavailable but cached grid diagnostics exist.",
            "caveat": "This should be treated as a coarse diagnostic summary, not a preferred primary metric.",
        }
    return {
        "view": "token_weighted_sampled",
        "metric_key": "pseudo_perplexity",
        "better_direction": "lower",
        "rationale": "Fallback to token-weighted sampled pseudo-perplexity because no stronger schedule-corrected or grid-integrated diagnostic was available.",
        "caveat": "Interpret carefully; this view can be more sensitive to schedule-dependent mask-count profiles.",
    }



def _build_eval_quality_summary(result: Dict[str, object]) -> Dict[str, object]:
    protocol = result.get("eval_protocol") or {}
    notes: List[str] = []
    warnings: List[str] = []

    sampled_example_count = int(result.get("sampled_example_count", 0) or 0)
    masked_tokens = int(result.get("masked_tokens", 0) or 0)
    timestep_count = int(result.get("timestep_macro_timestep_count", 0) or 0)
    ess_fraction = result.get("schedule_reweighted_effective_sample_size_fraction")
    ess_bucket = _schedule_reweighted_reliability_bucket(ess_fraction)

    if sampled_example_count > 0:
        notes.append(f"Sampled evaluation covers {sampled_example_count} per-example corruption draws and {masked_tokens} masked tokens.")
    else:
        warnings.append("Sampled evaluation did not record per-example corruption draws; timestep-uniform and schedule-reweighted views may be fallback estimates.")

    if protocol.get("normalized_timestep_remapping"):
        notes.append("Checkpoint comparisons/evaluations remap cached plan timesteps by normalized timestep fraction because diffusion step counts differ.")

    if timestep_count >= 3:
        notes.append(f"Fixed-grid timestep diagnostics span {timestep_count} timesteps, which is enough to make timestep-macro and timestep-AUC summaries informative.")
    elif timestep_count > 0:
        warnings.append(f"Fixed-grid timestep diagnostics only span {timestep_count} timesteps; macro/AUC views are available but thin.")
    else:
        warnings.append("No fixed-grid timestep diagnostics were recorded.")

    if ess_bucket == "strong":
        notes.append("Schedule-reweighted sampled evaluation has strong effective sample size support.")
    elif ess_bucket == "usable":
        warnings.append("Schedule-reweighted sampled evaluation is usable but somewhat variance-sensitive; consult the ESS before over-interpreting small deltas.")
    elif ess_bucket == "fragile":
        warnings.append("Schedule-reweighted sampled evaluation is variance-fragile because the effective sample size fraction is low.")
    else:
        warnings.append("Schedule-reweighted sampled evaluation did not produce a finite effective sample size diagnostic.")

    primary_view = _recommend_primary_eval_view({**result, "quality_summary": {"schedule_reweighted_reliability": ess_bucket}})

    return {
        "sampled_example_count": sampled_example_count,
        "masked_tokens": masked_tokens,
        "timestep_macro_timestep_count": timestep_count,
        "schedule_reweighted_effective_sample_size": result.get("schedule_reweighted_effective_sample_size"),
        "schedule_reweighted_effective_sample_size_fraction": ess_fraction,
        "schedule_reweighted_reliability": ess_bucket,
        "normalized_timestep_remapping": bool(protocol.get("normalized_timestep_remapping", False)),
        "recommended_primary_view": primary_view,
        "notes": notes,
        "warnings": warnings,
    }


def _recommend_comparison_primary_metric(comparison: Dict[str, object]) -> Dict[str, object]:
    protocol = comparison.get("comparison_protocol") or {}
    models = comparison.get("models") or []
    cosine_quality = ((models[0].get("quality_summary") or {}) if len(models) > 0 else {})
    linear_quality = ((models[1].get("quality_summary") or {}) if len(models) > 1 else {})
    cosine_primary = cosine_quality.get("recommended_primary_view") or {}
    linear_primary = linear_quality.get("recommended_primary_view") or {}

    default_primary = {
        "metric": "timestep_auc_pseudo_perplexity",
        "view": "fixed_grid_timestep_auc",
        "rationale": "Default to timestep-AUC pseudo-perplexity for conservative shared comparison behavior.",
    }

    if protocol.get("normalized_timestep_remapping"):
        return {
            **default_primary,
            "rationale": "The comparison remaps timesteps across different diffusion step counts, so timestep-AUC pseudo-perplexity is the safest shared primary metric because it integrates over a common normalized timestep grid instead of relying on schedule-specific sampled weights.",
        }

    cosine_metric = cosine_primary.get("metric_key")
    linear_metric = linear_primary.get("metric_key")
    if cosine_metric and cosine_metric == linear_metric:
        return {
            "metric": str(cosine_metric),
            "view": str(cosine_primary.get("view") or linear_primary.get("view") or "fixed_grid_timestep_auc"),
            "rationale": str(cosine_primary.get("rationale") or linear_primary.get("rationale") or default_primary["rationale"]),
        }
    if "schedule_reweighted_pseudo_perplexity" in {cosine_metric, linear_metric}:
        return {
            **default_primary,
            "rationale": "The two schedules disagree on whether schedule-reweighted sampled evaluation is reliable enough to be primary, so the comparison falls back to the shared conservative timestep-AUC view.",
        }
    if cosine_metric or linear_metric:
        return {
            "metric": str(cosine_metric or linear_metric),
            "view": str(cosine_primary.get("view") or linear_primary.get("view") or "fixed_grid_timestep_auc"),
            "rationale": "The comparison inherits the only available primary-view recommendation from the evaluated checkpoints.",
        }
    return default_primary


def _build_comparison_decision_summary(comparison: Dict[str, object]) -> Dict[str, object]:
    winner = comparison.get("winner") or {}
    confidence = comparison.get("winner_confidence") or {}
    primary_metric = _recommend_comparison_primary_metric(comparison)
    primary_metric_key = str(primary_metric.get("metric") or "timestep_auc_pseudo_perplexity")
    primary_metric_view = str(primary_metric.get("view") or "fixed_grid_timestep_auc")
    primary_rationale = str(primary_metric.get("rationale") or "Default to timestep-AUC pseudo-perplexity for conservative shared comparison behavior.")

    tracked_metrics = [
        ("sampled_pseudo_perplexity", "pseudo_perplexity", "lower"),
        ("timestep_uniform_pseudo_perplexity", "timestep_uniform_pseudo_perplexity", "lower"),
        ("schedule_reweighted_pseudo_perplexity", "schedule_reweighted_pseudo_perplexity", "lower"),
        ("grid_uniform_pseudo_perplexity", "grid_uniform_pseudo_perplexity", "lower"),
        ("timestep_macro_pseudo_perplexity", "timestep_macro_pseudo_perplexity", "lower"),
        ("timestep_auc_pseudo_perplexity", "timestep_auc_pseudo_perplexity", "lower"),
        ("sampled_accuracy", "masked_token_accuracy", "higher"),
        ("timestep_auc_accuracy", "timestep_auc_masked_token_accuracy", "higher"),
        ("sampled_bits_saved_vs_uniform", "sampled_bits_saved_vs_uniform", "higher"),
        ("timestep_auc_bits_saved_vs_uniform", "timestep_auc_bits_saved_vs_uniform", "higher"),
    ]

    rows = []
    cosine_wins = 0
    linear_wins = 0
    tied = 0
    decisive = 0
    for public_name, key, better_direction in tracked_metrics:
        metric_conf = confidence.get(key) or {}
        metric_winner = winner.get(key)
        practically_tied = bool(metric_conf.get("practically_tied", True))
        ci_excludes_zero = bool(metric_conf.get("ci_excludes_zero", False))
        winner_probability = metric_conf.get("winner_probability")
        if practically_tied:
            tied += 1
        elif metric_winner == "cosine_schedule":
            cosine_wins += 1
        elif metric_winner == "linear_schedule_baseline":
            linear_wins += 1
        if ci_excludes_zero:
            decisive += 1
        rows.append(
            {
                "metric": public_name,
                "winner": metric_winner,
                "better_direction": better_direction,
                "winner_probability": winner_probability,
                "ci_excludes_zero": ci_excludes_zero,
                "practically_tied": practically_tied,
                "is_recommended_primary_metric": bool(key == primary_metric_key),
            }
        )

    primary_conf = confidence.get(primary_metric_key) or {}
    primary_winner = winner.get(primary_metric_key)
    primary_decisive = bool(primary_conf.get("ci_excludes_zero", False))
    if primary_decisive and primary_winner == "cosine_schedule":
        headline = "cosine_schedule_leads_on_primary_metric"
    elif primary_decisive and primary_winner == "linear_schedule_baseline":
        headline = "linear_schedule_leads_on_primary_metric"
    elif cosine_wins > linear_wins:
        headline = "cosine_schedule_leads"
    elif linear_wins > cosine_wins:
        headline = "linear_schedule_leads"
    else:
        headline = "no_clear_schedule_lead"

    return {
        "headline": headline,
        "tracked_metric_count": len(tracked_metrics),
        "decisive_metric_count": decisive,
        "practically_tied_metric_count": tied,
        "cosine_schedule_win_count": cosine_wins,
        "linear_schedule_win_count": linear_wins,
        "recommended_primary_metric": {
            "metric": primary_metric_key,
            "view": primary_metric_view,
            "winner": primary_winner,
            "winner_probability": primary_conf.get("winner_probability"),
            "ci_excludes_zero": primary_conf.get("ci_excludes_zero"),
            "practically_tied": primary_conf.get("practically_tied"),
            "rationale": primary_rationale,
        },
        "tracked_metrics": rows,
    }


def compare_schedule_checkpoints(
    cosine_dir,
    linear_dir,
    *,
    device,
    config_cls,
    model_cls,
    dataloader,
    mask_token_id: int,
    excluded_token_ids: Optional[Sequence[Optional[int]]] = None,
    n_batches: int = 20,
    timestep_grid: Optional[Sequence[int]] = None,
    seed: int = 0,
    bootstrap_samples: int = 500,
):
    cosine_model, cosine_cfg = load_diffusion_checkpoint(cosine_dir, device, config_cls, model_cls)
    linear_model = None
    linear_cfg = None
    if linear_dir:
        linear_model, linear_cfg = load_diffusion_checkpoint(linear_dir, device, config_cls, model_cls)

    shared_T = min([cfg.diffusion_steps for cfg in [cosine_cfg, linear_cfg] if cfg is not None])
    shared_timestep_grid = _resolve_timestep_grid(shared_T, timestep_grid)
    eval_plan = build_eval_plan(
        dataloader,
        T=shared_T,
        n_batches=n_batches,
        timestep_grid=shared_timestep_grid,
        seed=seed,
    )

    cosine_result = evaluate_diffusion_pseudo_perplexity_from_plan(
        model=cosine_model,
        eval_plan=eval_plan,
        corruption_fn=corruption_factory("cosine"),
        mask_token_id=mask_token_id,
        T=cosine_cfg.diffusion_steps,
        excluded_token_ids=excluded_token_ids,
        schedule_name="cosine",
        bootstrap_samples=bootstrap_samples,
    )
    models = [{"tag": "cosine_schedule", **cosine_result}]

    if linear_model is not None:
        linear_result = evaluate_diffusion_pseudo_perplexity_from_plan(
            model=linear_model,
            eval_plan=eval_plan,
            corruption_fn=corruption_factory("linear"),
            mask_token_id=mask_token_id,
            T=linear_cfg.diffusion_steps,
            excluded_token_ids=excluded_token_ids,
            schedule_name="linear",
            bootstrap_samples=bootstrap_samples,
        )
        models.append({"tag": "linear_schedule_baseline", **linear_result})

    comparison = {
        "metric": "diffusion_pseudo_perplexity",
        "comparison_protocol": {
            "n_batches": n_batches,
            "seed": seed,
            "shared_timestep_grid": shared_timestep_grid,
            "shared_timestep_grid_fractions": [_timestep_fraction_from_int(t, shared_T) for t in shared_timestep_grid],
            "shared_eval_plan_T": shared_T,
            "cosine_diffusion_steps": int(cosine_cfg.diffusion_steps),
            "linear_diffusion_steps": int(linear_cfg.diffusion_steps) if linear_cfg is not None else None,
            "normalized_timestep_remapping": bool(linear_cfg is not None and int(cosine_cfg.diffusion_steps) != int(linear_cfg.diffusion_steps)),
            "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
            "paired_batches": True,
            "paired_uniform_noise": True,
            "bootstrap_samples": bootstrap_samples,
            "notes": [
                "Both checkpoints are evaluated on the same cached batches.",
                "Both schedules reuse the same underlying uniform random matrices, so differences come from the schedule mapping and the model, not fresh mask draws.",
                "If diffusion_steps differ, shared eval-plan timesteps are remapped by normalized timestep fraction before each model is evaluated.",
                "The comparison reports sampled-token, timestep-uniform sampled, schedule-reweighted sampled, fixed-grid-uniform, fixed-grid timestep-macro, and fixed-grid timestep-AUC aggregates.",
            ],
        },
        "models": models,
    }

    if len(models) == 2:
        cosine_metrics = {int(row.get("source_plan_timestep", row["timestep"])): row for row in models[0].get("timestep_metrics", [])}
        linear_metrics = {int(row.get("source_plan_timestep", row["timestep"])): row for row in models[1].get("timestep_metrics", [])}
        shared_timesteps = sorted(set(cosine_metrics) & set(linear_metrics))
        cosine_calibration = _comparison_calibration_views(models[0])
        linear_calibration = _comparison_calibration_views(models[1])
        comparison["delta"] = {
            "pseudo_perplexity": models[1]["pseudo_perplexity"] - models[0]["pseudo_perplexity"],
            "avg_cross_entropy": models[1]["avg_cross_entropy"] - models[0]["avg_cross_entropy"],
            "bits_per_masked_token": models[1]["bits_per_masked_token"] - models[0]["bits_per_masked_token"],
            "masked_token_accuracy": models[1]["masked_token_accuracy"] - models[0]["masked_token_accuracy"],
            "timestep_uniform_pseudo_perplexity": models[1]["timestep_uniform_pseudo_perplexity"] - models[0]["timestep_uniform_pseudo_perplexity"],
            "timestep_uniform_avg_cross_entropy": models[1]["timestep_uniform_avg_cross_entropy"] - models[0]["timestep_uniform_avg_cross_entropy"],
            "timestep_uniform_bits_per_masked_token": models[1]["timestep_uniform_bits_per_masked_token"] - models[0]["timestep_uniform_bits_per_masked_token"],
            "timestep_uniform_masked_token_accuracy": models[1]["timestep_uniform_masked_token_accuracy"] - models[0]["timestep_uniform_masked_token_accuracy"],
            "schedule_reweighted_pseudo_perplexity": models[1]["schedule_reweighted_pseudo_perplexity"] - models[0]["schedule_reweighted_pseudo_perplexity"],
            "schedule_reweighted_avg_cross_entropy": models[1]["schedule_reweighted_avg_cross_entropy"] - models[0]["schedule_reweighted_avg_cross_entropy"],
            "schedule_reweighted_bits_per_masked_token": models[1]["schedule_reweighted_bits_per_masked_token"] - models[0]["schedule_reweighted_bits_per_masked_token"],
            "schedule_reweighted_masked_token_accuracy": models[1]["schedule_reweighted_masked_token_accuracy"] - models[0]["schedule_reweighted_masked_token_accuracy"],
            "grid_uniform_pseudo_perplexity": models[1]["grid_uniform_pseudo_perplexity"] - models[0]["grid_uniform_pseudo_perplexity"],
            "grid_uniform_avg_cross_entropy": models[1]["grid_uniform_avg_cross_entropy"] - models[0]["grid_uniform_avg_cross_entropy"],
            "grid_uniform_bits_per_masked_token": models[1]["grid_uniform_bits_per_masked_token"] - models[0]["grid_uniform_bits_per_masked_token"],
            "grid_uniform_masked_token_accuracy": models[1]["grid_uniform_masked_token_accuracy"] - models[0]["grid_uniform_masked_token_accuracy"],
            "timestep_macro_pseudo_perplexity": models[1]["timestep_macro_pseudo_perplexity"] - models[0]["timestep_macro_pseudo_perplexity"],
            "timestep_macro_avg_cross_entropy": models[1]["timestep_macro_avg_cross_entropy"] - models[0]["timestep_macro_avg_cross_entropy"],
            "timestep_macro_bits_per_masked_token": models[1]["timestep_macro_bits_per_masked_token"] - models[0]["timestep_macro_bits_per_masked_token"],
            "timestep_macro_masked_token_accuracy": models[1]["timestep_macro_masked_token_accuracy"] - models[0]["timestep_macro_masked_token_accuracy"],
            "timestep_auc_pseudo_perplexity": models[1]["timestep_auc_pseudo_perplexity"] - models[0]["timestep_auc_pseudo_perplexity"],
            "timestep_auc_avg_cross_entropy": models[1]["timestep_auc_avg_cross_entropy"] - models[0]["timestep_auc_avg_cross_entropy"],
            "timestep_auc_bits_per_masked_token": models[1]["timestep_auc_bits_per_masked_token"] - models[0]["timestep_auc_bits_per_masked_token"],
            "timestep_auc_masked_token_accuracy": models[1]["timestep_auc_masked_token_accuracy"] - models[0]["timestep_auc_masked_token_accuracy"],
            "sampled_bits_saved_vs_uniform": linear_calibration["sampled"]["bits_saved_vs_uniform"] - cosine_calibration["sampled"]["bits_saved_vs_uniform"],
            "sampled_denoising_skill": linear_calibration["sampled"]["denoising_skill"] - cosine_calibration["sampled"]["denoising_skill"],
            "timestep_uniform_bits_saved_vs_uniform": linear_calibration["timestep_uniform"]["bits_saved_vs_uniform"] - cosine_calibration["timestep_uniform"]["bits_saved_vs_uniform"],
            "timestep_uniform_denoising_skill": linear_calibration["timestep_uniform"]["denoising_skill"] - cosine_calibration["timestep_uniform"]["denoising_skill"],
            "schedule_reweighted_bits_saved_vs_uniform": linear_calibration["schedule_reweighted"]["bits_saved_vs_uniform"] - cosine_calibration["schedule_reweighted"]["bits_saved_vs_uniform"],
            "schedule_reweighted_denoising_skill": linear_calibration["schedule_reweighted"]["denoising_skill"] - cosine_calibration["schedule_reweighted"]["denoising_skill"],
            "grid_uniform_bits_saved_vs_uniform": linear_calibration["grid_uniform"]["bits_saved_vs_uniform"] - cosine_calibration["grid_uniform"]["bits_saved_vs_uniform"],
            "grid_uniform_denoising_skill": linear_calibration["grid_uniform"]["denoising_skill"] - cosine_calibration["grid_uniform"]["denoising_skill"],
            "timestep_macro_bits_saved_vs_uniform": linear_calibration["timestep_macro"]["bits_saved_vs_uniform"] - cosine_calibration["timestep_macro"]["bits_saved_vs_uniform"],
            "timestep_macro_denoising_skill": linear_calibration["timestep_macro"]["denoising_skill"] - cosine_calibration["timestep_macro"]["denoising_skill"],
            "timestep_auc_bits_saved_vs_uniform": linear_calibration["timestep_auc"]["bits_saved_vs_uniform"] - cosine_calibration["timestep_auc"]["bits_saved_vs_uniform"],
            "timestep_auc_denoising_skill": linear_calibration["timestep_auc"]["denoising_skill"] - cosine_calibration["timestep_auc"]["denoising_skill"],
        }
        comparison["calibration"] = {
            "cosine_schedule": cosine_calibration,
            "linear_schedule_baseline": linear_calibration,
        }
        comparison["winner"] = {
            "pseudo_perplexity": "cosine_schedule" if models[0]["pseudo_perplexity"] <= models[1]["pseudo_perplexity"] else "linear_schedule_baseline",
            "avg_cross_entropy": "cosine_schedule" if models[0]["avg_cross_entropy"] <= models[1]["avg_cross_entropy"] else "linear_schedule_baseline",
            "masked_token_accuracy": "cosine_schedule" if models[0]["masked_token_accuracy"] >= models[1]["masked_token_accuracy"] else "linear_schedule_baseline",
            "timestep_uniform_pseudo_perplexity": "cosine_schedule" if models[0]["timestep_uniform_pseudo_perplexity"] <= models[1]["timestep_uniform_pseudo_perplexity"] else "linear_schedule_baseline",
            "timestep_uniform_avg_cross_entropy": "cosine_schedule" if models[0]["timestep_uniform_avg_cross_entropy"] <= models[1]["timestep_uniform_avg_cross_entropy"] else "linear_schedule_baseline",
            "timestep_uniform_bits_per_masked_token": "cosine_schedule" if models[0]["timestep_uniform_bits_per_masked_token"] <= models[1]["timestep_uniform_bits_per_masked_token"] else "linear_schedule_baseline",
            "timestep_uniform_masked_token_accuracy": "cosine_schedule" if models[0]["timestep_uniform_masked_token_accuracy"] >= models[1]["timestep_uniform_masked_token_accuracy"] else "linear_schedule_baseline",
            "schedule_reweighted_pseudo_perplexity": "cosine_schedule" if models[0]["schedule_reweighted_pseudo_perplexity"] <= models[1]["schedule_reweighted_pseudo_perplexity"] else "linear_schedule_baseline",
            "schedule_reweighted_avg_cross_entropy": "cosine_schedule" if models[0]["schedule_reweighted_avg_cross_entropy"] <= models[1]["schedule_reweighted_avg_cross_entropy"] else "linear_schedule_baseline",
            "schedule_reweighted_masked_token_accuracy": "cosine_schedule" if models[0]["schedule_reweighted_masked_token_accuracy"] >= models[1]["schedule_reweighted_masked_token_accuracy"] else "linear_schedule_baseline",
            "grid_uniform_pseudo_perplexity": "cosine_schedule" if models[0]["grid_uniform_pseudo_perplexity"] <= models[1]["grid_uniform_pseudo_perplexity"] else "linear_schedule_baseline",
            "grid_uniform_avg_cross_entropy": "cosine_schedule" if models[0]["grid_uniform_avg_cross_entropy"] <= models[1]["grid_uniform_avg_cross_entropy"] else "linear_schedule_baseline",
            "grid_uniform_bits_per_masked_token": "cosine_schedule" if models[0]["grid_uniform_bits_per_masked_token"] <= models[1]["grid_uniform_bits_per_masked_token"] else "linear_schedule_baseline",
            "grid_uniform_masked_token_accuracy": "cosine_schedule" if models[0]["grid_uniform_masked_token_accuracy"] >= models[1]["grid_uniform_masked_token_accuracy"] else "linear_schedule_baseline",
            "timestep_macro_pseudo_perplexity": "cosine_schedule" if models[0]["timestep_macro_pseudo_perplexity"] <= models[1]["timestep_macro_pseudo_perplexity"] else "linear_schedule_baseline",
            "timestep_macro_avg_cross_entropy": "cosine_schedule" if models[0]["timestep_macro_avg_cross_entropy"] <= models[1]["timestep_macro_avg_cross_entropy"] else "linear_schedule_baseline",
            "timestep_macro_masked_token_accuracy": "cosine_schedule" if models[0]["timestep_macro_masked_token_accuracy"] >= models[1]["timestep_macro_masked_token_accuracy"] else "linear_schedule_baseline",
            "timestep_auc_pseudo_perplexity": "cosine_schedule" if models[0]["timestep_auc_pseudo_perplexity"] <= models[1]["timestep_auc_pseudo_perplexity"] else "linear_schedule_baseline",
            "timestep_auc_avg_cross_entropy": "cosine_schedule" if models[0]["timestep_auc_avg_cross_entropy"] <= models[1]["timestep_auc_avg_cross_entropy"] else "linear_schedule_baseline",
            "timestep_auc_masked_token_accuracy": "cosine_schedule" if models[0]["timestep_auc_masked_token_accuracy"] >= models[1]["timestep_auc_masked_token_accuracy"] else "linear_schedule_baseline",
            "sampled_bits_saved_vs_uniform": "cosine_schedule" if cosine_calibration["sampled"]["bits_saved_vs_uniform"] >= linear_calibration["sampled"]["bits_saved_vs_uniform"] else "linear_schedule_baseline",
            "sampled_denoising_skill": "cosine_schedule" if cosine_calibration["sampled"]["denoising_skill"] >= linear_calibration["sampled"]["denoising_skill"] else "linear_schedule_baseline",
            "timestep_uniform_bits_saved_vs_uniform": "cosine_schedule" if cosine_calibration["timestep_uniform"]["bits_saved_vs_uniform"] >= linear_calibration["timestep_uniform"]["bits_saved_vs_uniform"] else "linear_schedule_baseline",
            "timestep_uniform_denoising_skill": "cosine_schedule" if cosine_calibration["timestep_uniform"]["denoising_skill"] >= linear_calibration["timestep_uniform"]["denoising_skill"] else "linear_schedule_baseline",
            "schedule_reweighted_bits_saved_vs_uniform": "cosine_schedule" if cosine_calibration["schedule_reweighted"]["bits_saved_vs_uniform"] >= linear_calibration["schedule_reweighted"]["bits_saved_vs_uniform"] else "linear_schedule_baseline",
            "schedule_reweighted_denoising_skill": "cosine_schedule" if cosine_calibration["schedule_reweighted"]["denoising_skill"] >= linear_calibration["schedule_reweighted"]["denoising_skill"] else "linear_schedule_baseline",
            "grid_uniform_bits_saved_vs_uniform": "cosine_schedule" if cosine_calibration["grid_uniform"]["bits_saved_vs_uniform"] >= linear_calibration["grid_uniform"]["bits_saved_vs_uniform"] else "linear_schedule_baseline",
            "grid_uniform_denoising_skill": "cosine_schedule" if cosine_calibration["grid_uniform"]["denoising_skill"] >= linear_calibration["grid_uniform"]["denoising_skill"] else "linear_schedule_baseline",
            "timestep_macro_bits_saved_vs_uniform": "cosine_schedule" if cosine_calibration["timestep_macro"]["bits_saved_vs_uniform"] >= linear_calibration["timestep_macro"]["bits_saved_vs_uniform"] else "linear_schedule_baseline",
            "timestep_macro_denoising_skill": "cosine_schedule" if cosine_calibration["timestep_macro"]["denoising_skill"] >= linear_calibration["timestep_macro"]["denoising_skill"] else "linear_schedule_baseline",
            "timestep_auc_bits_saved_vs_uniform": "cosine_schedule" if cosine_calibration["timestep_auc"]["bits_saved_vs_uniform"] >= linear_calibration["timestep_auc"]["bits_saved_vs_uniform"] else "linear_schedule_baseline",
            "timestep_auc_denoising_skill": "cosine_schedule" if cosine_calibration["timestep_auc"]["denoising_skill"] >= linear_calibration["timestep_auc"]["denoising_skill"] else "linear_schedule_baseline",
        }
        comparison["timestep_deltas"] = [
            {
                "source_plan_timestep": t,
                "timestep_fraction": cosine_metrics[t].get("timestep_fraction", linear_metrics[t].get("timestep_fraction")),
                "cosine_timestep": cosine_metrics[t]["timestep"],
                "linear_timestep": linear_metrics[t]["timestep"],
                "pseudo_perplexity": linear_metrics[t]["pseudo_perplexity"] - cosine_metrics[t]["pseudo_perplexity"],
                "avg_cross_entropy": linear_metrics[t]["avg_cross_entropy"] - cosine_metrics[t]["avg_cross_entropy"],
                "bits_per_masked_token": linear_metrics[t]["bits_per_masked_token"] - cosine_metrics[t]["bits_per_masked_token"],
                "masked_token_accuracy": linear_metrics[t]["masked_token_accuracy"] - cosine_metrics[t]["masked_token_accuracy"],
                "cosine_mean_mask_fraction": cosine_metrics[t]["mean_mask_fraction"],
                "linear_mean_mask_fraction": linear_metrics[t]["mean_mask_fraction"],
                "mask_fraction_delta_linear_minus_cosine": linear_metrics[t]["mean_mask_fraction"] - cosine_metrics[t]["mean_mask_fraction"],
            }
            for t in shared_timesteps
        ]
        comparison["delta_confidence_intervals"] = _bootstrap_paired_comparison_interval(
            cosine_batch_metrics=models[0].get("sampled_batch_metrics", []),
            linear_batch_metrics=models[1].get("sampled_batch_metrics", []),
            cosine_example_metrics=models[0].get("sampled_example_metrics", []),
            linear_example_metrics=models[1].get("sampled_example_metrics", []),
            cosine_grid_batch_metrics=models[0].get("grid_batch_metrics", []),
            linear_grid_batch_metrics=models[1].get("grid_batch_metrics", []),
            cosine_timestep_metrics=models[0].get("timestep_metrics", []),
            linear_timestep_metrics=models[1].get("timestep_metrics", []),
            vocab_size=models[0].get("vocab_size") or models[1].get("vocab_size"),
            n_samples=bootstrap_samples,
            seed=seed,
        )
        comparison["winner_confidence"] = {
            "pseudo_perplexity": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("pseudo_perplexity"), better_direction="lower"),
            "avg_cross_entropy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("avg_cross_entropy"), better_direction="lower"),
            "bits_per_masked_token": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("bits_per_masked_token"), better_direction="lower"),
            "masked_token_accuracy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("masked_token_accuracy"), better_direction="higher"),
            "timestep_uniform_pseudo_perplexity": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_uniform_pseudo_perplexity"), better_direction="lower"),
            "timestep_uniform_avg_cross_entropy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_uniform_avg_cross_entropy"), better_direction="lower"),
            "timestep_uniform_bits_per_masked_token": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_uniform_bits_per_masked_token"), better_direction="lower"),
            "timestep_uniform_masked_token_accuracy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_uniform_masked_token_accuracy"), better_direction="higher"),
            "schedule_reweighted_pseudo_perplexity": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("schedule_reweighted_pseudo_perplexity"), better_direction="lower"),
            "schedule_reweighted_avg_cross_entropy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("schedule_reweighted_avg_cross_entropy"), better_direction="lower"),
            "schedule_reweighted_bits_per_masked_token": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("schedule_reweighted_bits_per_masked_token"), better_direction="lower"),
            "schedule_reweighted_masked_token_accuracy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("schedule_reweighted_masked_token_accuracy"), better_direction="higher"),
            "grid_uniform_pseudo_perplexity": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("grid_uniform_pseudo_perplexity"), better_direction="lower"),
            "grid_uniform_avg_cross_entropy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("grid_uniform_avg_cross_entropy"), better_direction="lower"),
            "grid_uniform_bits_per_masked_token": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("grid_uniform_bits_per_masked_token"), better_direction="lower"),
            "grid_uniform_masked_token_accuracy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("grid_uniform_masked_token_accuracy"), better_direction="higher"),
            "timestep_macro_pseudo_perplexity": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_macro_pseudo_perplexity"), better_direction="lower"),
            "timestep_macro_avg_cross_entropy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_macro_avg_cross_entropy"), better_direction="lower"),
            "timestep_macro_bits_per_masked_token": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_macro_bits_per_masked_token"), better_direction="lower"),
            "timestep_macro_masked_token_accuracy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_macro_masked_token_accuracy"), better_direction="higher"),
            "timestep_auc_pseudo_perplexity": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_auc_pseudo_perplexity"), better_direction="lower"),
            "timestep_auc_avg_cross_entropy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_auc_avg_cross_entropy"), better_direction="lower"),
            "timestep_auc_bits_per_masked_token": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_auc_bits_per_masked_token"), better_direction="lower"),
            "timestep_auc_masked_token_accuracy": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_auc_masked_token_accuracy"), better_direction="higher"),
            "sampled_bits_saved_vs_uniform": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("sampled_bits_saved_vs_uniform"), better_direction="higher"),
            "sampled_denoising_skill": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("sampled_denoising_skill"), better_direction="higher"),
            "timestep_uniform_bits_saved_vs_uniform": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_uniform_bits_saved_vs_uniform"), better_direction="higher"),
            "timestep_uniform_denoising_skill": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_uniform_denoising_skill"), better_direction="higher"),
            "schedule_reweighted_bits_saved_vs_uniform": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("schedule_reweighted_bits_saved_vs_uniform"), better_direction="higher"),
            "schedule_reweighted_denoising_skill": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("schedule_reweighted_denoising_skill"), better_direction="higher"),
            "grid_uniform_bits_saved_vs_uniform": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("grid_uniform_bits_saved_vs_uniform"), better_direction="higher"),
            "grid_uniform_denoising_skill": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("grid_uniform_denoising_skill"), better_direction="higher"),
            "timestep_macro_bits_saved_vs_uniform": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_macro_bits_saved_vs_uniform"), better_direction="higher"),
            "timestep_macro_denoising_skill": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_macro_denoising_skill"), better_direction="higher"),
            "timestep_auc_bits_saved_vs_uniform": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_auc_bits_saved_vs_uniform"), better_direction="higher"),
            "timestep_auc_denoising_skill": _paired_delta_confidence_summary(comparison["delta_confidence_intervals"]["delta_linear_minus_cosine"].get("timestep_auc_denoising_skill"), better_direction="higher"),
        }
        comparison["decision_summary"] = _build_comparison_decision_summary(comparison)
    return comparison


def _json_ready(value):
    if is_dataclass(value):
        return asdict(value)
    return value


def export_eval_result(path, tag: str, result: Dict[str, object]) -> Dict[str, object]:
    payload = {"tag": tag, **{k: _json_ready(v) for k, v in result.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def export_eval_plan(path, eval_plan: Dict[str, object]) -> Dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(eval_plan, path)
    return {
        "path": str(path),
        "n_batches": int(eval_plan.get("n_batches", 0)),
        "T": int(eval_plan.get("T", 0)),
        "timestep_grid": list(eval_plan.get("timestep_grid", [])),
        "seed": int(eval_plan.get("seed", 0)),
    }


def load_eval_plan(path) -> Dict[str, object]:
    return torch.load(Path(path), map_location="cpu")


def export_schedule_comparison(path, comparison: Dict[str, object]) -> Dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison
