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
        "masked_tokens": 0,
        "n_batches": n_batches,
        "seed": seed,
        "timestep_metrics": [],
        "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
        "notes": [
            "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
            "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
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
) -> Dict[str, object]:
    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_masked_tokens = 0
    timestep_records: Dict[int, Dict[str, float]] = {}

    for batch_plan in eval_plan["batches"]:
        input_ids = batch_plan["input_ids"].to(device)
        attention_mask = batch_plan["attention_mask"].to(device)

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

            if timestep_plan["kind"] == "sampled":
                total_nll += token_nll_sum
                total_masked_tokens += masked_count
            else:
                key = int(eval_t[0].item())
                rec = timestep_records.setdefault(key, {"nll_sum": 0.0, "masked_tokens": 0, "mask_ratio_sum": 0.0, "examples": 0})
                rec["nll_sum"] += token_nll_sum
                rec["masked_tokens"] += masked_count
                rec["mask_ratio_sum"] += masked_count / max(1, int(attention_mask.sum().item()))
                rec["examples"] += 1

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
                    "pseudo_perplexity": math.exp(ce),
                    "bits_per_masked_token": ce / math.log(2.0),
                    "masked_tokens": rec["masked_tokens"],
                    "mean_mask_fraction": rec["mask_ratio_sum"] / max(1, rec["examples"]),
                }
            )

        result = {
            "metric": "diffusion_pseudo_perplexity",
            "avg_cross_entropy": avg_ce,
            "pseudo_perplexity": math.exp(avg_ce),
            "bits_per_masked_token": avg_ce / math.log(2.0),
            "masked_tokens": total_masked_tokens,
            "n_batches": int(eval_plan.get("n_batches", 0)),
            "seed": int(eval_plan.get("seed", 0)),
            "timestep_metrics": timestep_metrics,
            "excluded_token_ids": _normalize_excluded_token_ids(excluded_token_ids),
            "notes": [
                "Pseudo-perplexity is computed from masked-token denoising NLL, not autoregressive next-token likelihood.",
                "Aggregate CE is token-weighted over all masked positions, avoiding per-batch averaging bias.",
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
):
    cosine_model, cosine_cfg = load_diffusion_checkpoint(cosine_dir, device, config_cls, model_cls)
    linear_model = None
    linear_cfg = None
    if linear_dir:
        linear_model, linear_cfg = load_diffusion_checkpoint(linear_dir, device, config_cls, model_cls)

    shared_T = min(
        [cfg.diffusion_steps for cfg in [cosine_cfg, linear_cfg] if cfg is not None]
    )
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
            "notes": [
                "Both checkpoints are evaluated on the same cached batches.",
                "Both schedules reuse the same underlying uniform random matrices, so differences come from the schedule mapping and the model, not fresh mask draws.",
            ],
        },
        "models": models,
    }

    if len(models) == 2:
        comparison["delta"] = {
            "pseudo_perplexity": models[1]["pseudo_perplexity"] - models[0]["pseudo_perplexity"],
            "avg_cross_entropy": models[1]["avg_cross_entropy"] - models[0]["avg_cross_entropy"],
            "bits_per_masked_token": models[1]["bits_per_masked_token"] - models[0]["bits_per_masked_token"],
        }
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
