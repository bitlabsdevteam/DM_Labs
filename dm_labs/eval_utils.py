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
        "sampled_example_count": 0,
        "masked_tokens": 0,
        "n_batches": n_batches,
        "seed": seed,
        "timestep_metrics": [],
        "sampled_batch_metrics": [],
        "sampled_example_metrics": [],
        "sampled_timestep_histogram": {},
        "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
        "confidence_intervals": _bootstrap_metric_interval([], n_samples=0, seed=seed),
        "notes": [
            "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
            "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
            "Timestep-uniform CE reports the mean masked-token CE across uniformly sampled timesteps, decoupling the metric from schedule-dependent mask counts.",
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
                            "avg_cross_entropy": example_ce,
                            "pseudo_perplexity": _safe_exp(example_ce),
                            "bits_per_masked_token": example_ce / math.log(2.0),
                            "masked_token_accuracy": example_correct / example_masked_tokens,
                        }
                    )
            else:
                key = int(eval_t[0].item())
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
        result = {
            "metric": "diffusion_pseudo_perplexity",
            "avg_cross_entropy": avg_ce,
            "pseudo_perplexity": _safe_exp(avg_ce),
            "bits_per_masked_token": avg_ce / math.log(2.0),
            "masked_token_accuracy": total_correct / total_masked_tokens,
            "timestep_uniform_avg_cross_entropy": timestep_uniform_avg_ce,
            "timestep_uniform_pseudo_perplexity": _safe_exp(timestep_uniform_avg_ce),
            "timestep_uniform_bits_per_masked_token": timestep_uniform_avg_ce / math.log(2.0) if not math.isnan(timestep_uniform_avg_ce) else float("nan"),
            "sampled_example_count": sampled_example_count,
            "masked_tokens": total_masked_tokens,
            "n_batches": int(eval_plan.get("n_batches", 0)),
            "seed": int(eval_plan.get("seed", 0)),
            "timestep_metrics": timestep_metrics,
            "sampled_batch_metrics": sampled_batch_metrics,
            "sampled_example_metrics": sampled_example_metrics,
            "sampled_timestep_histogram": {str(k): int(v) for k, v in sorted(sampled_timestep_histogram.items())},
            "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
            "confidence_intervals": _bootstrap_metric_interval(
                sampled_batch_metrics,
                n_samples=bootstrap_samples,
                seed=int(eval_plan.get("seed", 0)),
            ),
            "notes": [
                "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
                "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
                "Timestep-uniform CE averages per-example masked-token CE over uniformly sampled timesteps, making schedule comparisons less sensitive to different mask-count profiles.",
                "Confidence intervals are computed by bootstrapping over sampled evaluation batches from the shared cached plan.",
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
        }
        comparison["winner"] = {
            "pseudo_perplexity": "cosine_schedule" if models[0]["pseudo_perplexity"] <= models[1]["pseudo_perplexity"] else "linear_schedule_baseline",
            "avg_cross_entropy": "cosine_schedule" if models[0]["avg_cross_entropy"] <= models[1]["avg_cross_entropy"] else "linear_schedule_baseline",
            "masked_token_accuracy": "cosine_schedule" if models[0]["masked_token_accuracy"] >= models[1]["masked_token_accuracy"] else "linear_schedule_baseline",
            "timestep_uniform_pseudo_perplexity": "cosine_schedule" if models[0]["timestep_uniform_pseudo_perplexity"] <= models[1]["timestep_uniform_pseudo_perplexity"] else "linear_schedule_baseline",
            "timestep_uniform_avg_cross_entropy": "cosine_schedule" if models[0]["timestep_uniform_avg_cross_entropy"] <= models[1]["timestep_uniform_avg_cross_entropy"] else "linear_schedule_baseline",
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
