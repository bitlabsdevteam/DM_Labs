import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


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


def _bootstrap_paired_comparison_interval(
    cosine_batch_metrics: Sequence[Dict[str, float]],
    linear_batch_metrics: Sequence[Dict[str, float]],
    *,
    cosine_example_metrics: Optional[Sequence[Dict[str, float]]] = None,
    linear_example_metrics: Optional[Sequence[Dict[str, float]]] = None,
    cosine_grid_batch_metrics: Optional[Sequence[Dict[str, float]]] = None,
    linear_grid_batch_metrics: Optional[Sequence[Dict[str, float]]] = None,
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
    }
    grid_delta_samples = {
        "grid_uniform_avg_cross_entropy": [],
        "grid_uniform_pseudo_perplexity": [],
        "grid_uniform_bits_per_masked_token": [],
        "grid_uniform_masked_token_accuracy": [],
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
            cosine_correct += int(cosine_rec.get("correct_masked_tokens", 0))
            linear_correct += int(linear_rec.get("correct_masked_tokens", 0))

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

    def _with_win_probability(metric_name: str, values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {**_metric_percentiles(values), "probability_linear_better": float("nan")}
        higher_is_better = metric_name in {"masked_token_accuracy", "timestep_uniform_masked_token_accuracy", "grid_uniform_masked_token_accuracy"}
        wins = sum(v > 0.0 for v in values) if higher_is_better else sum(v < 0.0 for v in values)
        return {**_metric_percentiles(values), "probability_linear_better": float(wins / len(values))}

    summary = {key: _with_win_probability(key, values) for key, values in batch_delta_samples.items()}
    summary.update({key: _with_win_probability(key, values) for key, values in example_delta_samples.items()})
    summary.update({key: _with_win_probability(key, values) for key, values in grid_delta_samples.items()})
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


def corruption_factory(schedule_name: str) -> Callable[..., Tuple[Tensor, Tensor, Tensor]]:
    schedule_name = schedule_name.lower().strip()
    if schedule_name == "cosine":
        schedule_fn = mask_ratio_cosine_schedule
    elif schedule_name == "linear":
        schedule_fn = mask_ratio_linear_schedule
    else:
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

        timestep_plans = [{"kind": "sampled", "t": sampled_t, "rand": torch.rand((B, L), generator=generator)}]
        for t in resolved_grid:
            timestep_plans.append(
                {
                    "kind": "grid",
                    "t": torch.full((B,), int(t), dtype=torch.long),
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


def _empty_eval_result(n_batches: int, seed: int, excluded_token_ids: Optional[Sequence[Optional[int]]]) -> Dict[str, object]:
    return {
        "metric": "diffusion_pseudo_perplexity",
        "avg_cross_entropy": float("nan"),
        "pseudo_perplexity": float("nan"),
        "bits_per_masked_token": float("nan"),
        "masked_token_accuracy": float("nan"),
        "timestep_uniform_avg_cross_entropy": float("nan"),
        "timestep_uniform_pseudo_perplexity": float("nan"),
        "timestep_uniform_bits_per_masked_token": float("nan"),
        "grid_uniform_avg_cross_entropy": float("nan"),
        "grid_uniform_pseudo_perplexity": float("nan"),
        "grid_uniform_bits_per_masked_token": float("nan"),
        "grid_uniform_masked_token_accuracy": float("nan"),
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
        "grid_uniform_confidence_intervals": _bootstrap_grid_uniform_metric_interval([], n_samples=0, seed=seed),
        "notes": [
            "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
            "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
            "Timestep-uniform CE reports the mean masked-token CE across uniformly sampled timesteps, decoupling the metric from schedule-dependent mask counts.",
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
    total_correct = 0
    sampled_example_ce_sum = 0.0
    sampled_example_count = 0
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
            eval_t = timestep_plan["t"].to(device)
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
            masked_logits = logits[mask_positions]
            masked_targets = labels[mask_positions]
            token_nll_sum = float(F.cross_entropy(masked_logits, masked_targets, reduction="sum").item())
            predictions = masked_logits.argmax(dim=-1)
            correct = int((predictions == masked_targets).sum().item())

            if timestep_plan["kind"] == "sampled":
                total_nll += token_nll_sum
                total_masked_tokens += masked_count
                total_correct += correct

                batch_ce = token_nll_sum / masked_count
                t_values = eval_t.detach().cpu().tolist()
                for value in t_values:
                    value = int(value)
                    sampled_timestep_histogram[value] = sampled_timestep_histogram.get(value, 0) + 1
                sampled_batch_metrics.append(
                    {
                        "batch_index": batch_idx,
                        "nll_sum": token_nll_sum,
                        "masked_tokens": masked_count,
                        "correct_masked_tokens": correct,
                        "avg_cross_entropy": batch_ce,
                        "pseudo_perplexity": _safe_exp(batch_ce),
                        "bits_per_masked_token": batch_ce / math.log(2.0),
                        "masked_token_accuracy": correct / masked_count,
                        "mean_mask_fraction": masked_count / max(1, batch_active_tokens),
                        "sampled_timesteps": t_values,
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
                    example_correct = int((example_logits.argmax(dim=-1) == example_targets).sum().item())
                    sampled_example_ce_sum += example_ce
                    sampled_example_count += 1
                    sampled_example_metrics.append(
                        {
                            "batch_index": batch_idx,
                            "example_index": example_idx,
                            "timestep": int(eval_t[example_idx].item()),
                            "nll_sum": example_nll_sum,
                            "masked_tokens": example_masked_tokens,
                            "correct_masked_tokens": example_correct,
                            "avg_cross_entropy": example_ce,
                            "pseudo_perplexity": _safe_exp(example_ce),
                            "bits_per_masked_token": example_ce / math.log(2.0),
                            "masked_token_accuracy": example_correct / example_masked_tokens,
                        }
                    )
            else:
                key = int(eval_t[0].item())
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
                        "timestep": key,
                        "nll_sum": token_nll_sum,
                        "masked_tokens": masked_count,
                        "correct_masked_tokens": correct,
                        "avg_cross_entropy": batch_ce,
                        "pseudo_perplexity": _safe_exp(batch_ce),
                        "bits_per_masked_token": batch_ce / math.log(2.0),
                        "masked_token_accuracy": batch_accuracy,
                        "mean_mask_fraction": masked_count / max(1, batch_active_tokens),
                    }
                )

    if total_masked_tokens == 0:
        result = _empty_eval_result(
            n_batches=int(eval_plan.get("n_batches", 0)),
            seed=int(eval_plan.get("seed", 0)),
            excluded_token_ids=excluded_token_ids,
        )
    else:
        avg_ce = total_nll / total_masked_tokens
        timestep_metrics = []
        for t in sorted(timestep_records):
            rec = timestep_records[t]
            if rec["masked_tokens"] == 0:
                continue
            ce = rec["nll_sum"] / rec["masked_tokens"]
            timestep_metrics.append(
                {
                    "timestep": t,
                    "avg_cross_entropy": ce,
                    "pseudo_perplexity": _safe_exp(ce),
                    "bits_per_masked_token": ce / math.log(2.0),
                    "masked_tokens": rec["masked_tokens"],
                    "mean_mask_fraction": rec["mask_ratio_sum"] / max(1, rec["examples"]),
                    "masked_token_accuracy": rec["correct"] / rec["masked_tokens"],
                }
            )

        timestep_uniform_avg_ce = sampled_example_ce_sum / sampled_example_count if sampled_example_count else float("nan")
        if grid_batch_metrics:
            grid_uniform_avg_ce = sum(float(row["avg_cross_entropy"]) for row in grid_batch_metrics) / len(grid_batch_metrics)
            grid_uniform_masked_token_accuracy = sum(float(row["masked_token_accuracy"]) for row in grid_batch_metrics) / len(grid_batch_metrics)
        else:
            grid_uniform_avg_ce = float("nan")
            grid_uniform_masked_token_accuracy = float("nan")

        result = {
            "metric": "diffusion_pseudo_perplexity",
            "avg_cross_entropy": avg_ce,
            "pseudo_perplexity": _safe_exp(avg_ce),
            "bits_per_masked_token": avg_ce / math.log(2.0),
            "masked_token_accuracy": total_correct / total_masked_tokens,
            "timestep_uniform_avg_cross_entropy": timestep_uniform_avg_ce,
            "timestep_uniform_pseudo_perplexity": _safe_exp(timestep_uniform_avg_ce),
            "timestep_uniform_bits_per_masked_token": timestep_uniform_avg_ce / math.log(2.0) if not math.isnan(timestep_uniform_avg_ce) else float("nan"),
            "grid_uniform_avg_cross_entropy": grid_uniform_avg_ce,
            "grid_uniform_pseudo_perplexity": _safe_exp(grid_uniform_avg_ce),
            "grid_uniform_bits_per_masked_token": grid_uniform_avg_ce / math.log(2.0) if not math.isnan(grid_uniform_avg_ce) else float("nan"),
            "grid_uniform_masked_token_accuracy": grid_uniform_masked_token_accuracy,
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
            "grid_uniform_confidence_intervals": _bootstrap_grid_uniform_metric_interval(
                grid_batch_metrics,
                n_samples=bootstrap_samples,
                seed=int(eval_plan.get("seed", 0)),
            ),
            "notes": [
                "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
                "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
                "Timestep-uniform CE averages per-example masked-token CE over uniformly sampled timesteps, making schedule comparisons less sensitive to different mask-count profiles.",
                "Grid-uniform CE averages batch-level denoising CE over a fixed cached timestep grid, giving an explicit schedule-agnostic comparison surface.",
                "Confidence intervals are computed by bootstrapping over sampled evaluation batches from the shared cached plan.",
                "Grid-uniform confidence intervals are computed by bootstrapping over cached batch-timestep diagnostic records.",
                "Timestep diagnostics are evaluated on a shared cached batch/noise plan so schedule comparisons are paired and reproducible.",
            ],
        }

    result["eval_protocol"] = {
        "T": T,
        "schedule_name": schedule_name,
        "timestep_grid": list(eval_plan.get("timestep_grid", [])),
        "paired_noise": True,
        "paired_batches": True,
        "sampled_timestep_distribution": "uniform_integer_1_to_T",
        "grid_uniform_aggregation": "mean_over_cached_batch_timestep_records",
        "bootstrap_samples": bootstrap_samples,
    }
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
            "shared_eval_plan_T": shared_T,
            "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
            "paired_batches": True,
            "paired_uniform_noise": True,
            "bootstrap_samples": bootstrap_samples,
            "notes": [
                "Both checkpoints are evaluated on the same cached batches.",
                "Both schedules reuse the same underlying uniform random matrices, so differences come from the schedule mapping and the model, not fresh mask draws.",
                "The comparison reports both sampled-timestep and fixed-grid-uniform aggregates.",
            ],
        },
        "models": models,
    }

    if len(models) == 2:
        cosine_metrics = {row["timestep"]: row for row in models[0].get("timestep_metrics", [])}
        linear_metrics = {row["timestep"]: row for row in models[1].get("timestep_metrics", [])}
        shared_timesteps = sorted(set(cosine_metrics) & set(linear_metrics))
        comparison["delta"] = {
            "pseudo_perplexity": models[1]["pseudo_perplexity"] - models[0]["pseudo_perplexity"],
            "avg_cross_entropy": models[1]["avg_cross_entropy"] - models[0]["avg_cross_entropy"],
            "bits_per_masked_token": models[1]["bits_per_masked_token"] - models[0]["bits_per_masked_token"],
            "masked_token_accuracy": models[1]["masked_token_accuracy"] - models[0]["masked_token_accuracy"],
            "timestep_uniform_pseudo_perplexity": models[1]["timestep_uniform_pseudo_perplexity"] - models[0]["timestep_uniform_pseudo_perplexity"],
            "timestep_uniform_avg_cross_entropy": models[1]["timestep_uniform_avg_cross_entropy"] - models[0]["timestep_uniform_avg_cross_entropy"],
            "grid_uniform_pseudo_perplexity": models[1]["grid_uniform_pseudo_perplexity"] - models[0]["grid_uniform_pseudo_perplexity"],
            "grid_uniform_avg_cross_entropy": models[1]["grid_uniform_avg_cross_entropy"] - models[0]["grid_uniform_avg_cross_entropy"],
            "grid_uniform_masked_token_accuracy": models[1]["grid_uniform_masked_token_accuracy"] - models[0]["grid_uniform_masked_token_accuracy"],
        }
        comparison["winner"] = {
            "pseudo_perplexity": "cosine_schedule" if models[0]["pseudo_perplexity"] <= models[1]["pseudo_perplexity"] else "linear_schedule_baseline",
            "avg_cross_entropy": "cosine_schedule" if models[0]["avg_cross_entropy"] <= models[1]["avg_cross_entropy"] else "linear_schedule_baseline",
            "masked_token_accuracy": "cosine_schedule" if models[0]["masked_token_accuracy"] >= models[1]["masked_token_accuracy"] else "linear_schedule_baseline",
            "timestep_uniform_pseudo_perplexity": "cosine_schedule" if models[0]["timestep_uniform_pseudo_perplexity"] <= models[1]["timestep_uniform_pseudo_perplexity"] else "linear_schedule_baseline",
            "timestep_uniform_avg_cross_entropy": "cosine_schedule" if models[0]["timestep_uniform_avg_cross_entropy"] <= models[1]["timestep_uniform_avg_cross_entropy"] else "linear_schedule_baseline",
            "grid_uniform_pseudo_perplexity": "cosine_schedule" if models[0]["grid_uniform_pseudo_perplexity"] <= models[1]["grid_uniform_pseudo_perplexity"] else "linear_schedule_baseline",
            "grid_uniform_avg_cross_entropy": "cosine_schedule" if models[0]["grid_uniform_avg_cross_entropy"] <= models[1]["grid_uniform_avg_cross_entropy"] else "linear_schedule_baseline",
            "grid_uniform_masked_token_accuracy": "cosine_schedule" if models[0]["grid_uniform_masked_token_accuracy"] >= models[1]["grid_uniform_masked_token_accuracy"] else "linear_schedule_baseline",
        }
        comparison["timestep_deltas"] = [
            {
                "timestep": t,
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
            n_samples=bootstrap_samples,
            seed=seed,
        )
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


def export_schedule_comparison(path, comparison: Dict[str, object]) -> Dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison
