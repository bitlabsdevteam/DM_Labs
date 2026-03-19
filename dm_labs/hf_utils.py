import json
import math
import os
from pathlib import Path
from typing import Optional


def _view_calibration(avg_cross_entropy, bits_per_masked_token, vocab_size: Optional[int]) -> dict:
    if vocab_size is None or int(vocab_size) <= 0:
        return {
            "uniform_random_pseudo_perplexity": None,
            "bits_saved_vs_uniform": None,
            "denoising_skill": None,
        }
    if avg_cross_entropy is None or bits_per_masked_token is None:
        return {
            "uniform_random_pseudo_perplexity": float(vocab_size),
            "bits_saved_vs_uniform": None,
            "denoising_skill": None,
        }
    try:
        avg_cross_entropy = float(avg_cross_entropy)
        bits_per_masked_token = float(bits_per_masked_token)
    except (TypeError, ValueError):
        return {
            "uniform_random_pseudo_perplexity": float(vocab_size),
            "bits_saved_vs_uniform": None,
            "denoising_skill": None,
        }
    if math.isnan(avg_cross_entropy) or math.isnan(bits_per_masked_token):
        return {
            "uniform_random_pseudo_perplexity": float(vocab_size),
            "bits_saved_vs_uniform": float("nan"),
            "denoising_skill": float("nan"),
        }
    uniform_bits = math.log(float(vocab_size)) / math.log(2.0)
    return {
        "uniform_random_pseudo_perplexity": float(vocab_size),
        "bits_saved_vs_uniform": uniform_bits - bits_per_masked_token,
        "denoising_skill": 1.0 - (avg_cross_entropy / math.log(float(vocab_size))),
    }


EVAL_VIEW_SPECS = [
    {
        "view": "token_weighted_sampled",
        "avg_cross_entropy_key": "avg_cross_entropy",
        "pseudo_perplexity_key": "pseudo_perplexity",
        "bits_key": "bits_per_masked_token",
        "accuracy_key": "masked_token_accuracy",
        "ci_container_key": "confidence_intervals",
        "ci_ce_key": "avg_cross_entropy",
        "ci_metric_key": "pseudo_perplexity",
        "ci_accuracy_key": "masked_token_accuracy",
        "aggregation": "token-weighted over sampled masked tokens",
    },
    {
        "view": "timestep_uniform_sampled",
        "avg_cross_entropy_key": "timestep_uniform_avg_cross_entropy",
        "pseudo_perplexity_key": "timestep_uniform_pseudo_perplexity",
        "bits_key": "timestep_uniform_bits_per_masked_token",
        "accuracy_key": "timestep_uniform_masked_token_accuracy",
        "ci_container_key": "timestep_uniform_confidence_intervals",
        "ci_ce_key": "timestep_uniform_avg_cross_entropy",
        "ci_metric_key": "timestep_uniform_pseudo_perplexity",
        "ci_accuracy_key": "timestep_uniform_masked_token_accuracy",
        "aggregation": "uniform mean over sampled per-example timesteps",
    },
    {
        "view": "schedule_reweighted_sampled",
        "avg_cross_entropy_key": "schedule_reweighted_avg_cross_entropy",
        "pseudo_perplexity_key": "schedule_reweighted_pseudo_perplexity",
        "bits_key": "schedule_reweighted_bits_per_masked_token",
        "accuracy_key": "schedule_reweighted_masked_token_accuracy",
        "ci_container_key": "schedule_reweighted_confidence_intervals",
        "ci_ce_key": "schedule_reweighted_avg_cross_entropy",
        "ci_metric_key": "schedule_reweighted_pseudo_perplexity",
        "ci_accuracy_key": "schedule_reweighted_masked_token_accuracy",
        "aggregation": "inverse-expected-mask-ratio weighting over sampled masked tokens",
    },
    {
        "view": "fixed_grid_batch_uniform",
        "avg_cross_entropy_key": "grid_uniform_avg_cross_entropy",
        "pseudo_perplexity_key": "grid_uniform_pseudo_perplexity",
        "bits_key": "grid_uniform_bits_per_masked_token",
        "accuracy_key": "grid_uniform_masked_token_accuracy",
        "ci_container_key": "grid_uniform_confidence_intervals",
        "ci_ce_key": "grid_uniform_avg_cross_entropy",
        "ci_metric_key": "grid_uniform_pseudo_perplexity",
        "ci_accuracy_key": "grid_uniform_masked_token_accuracy",
        "aggregation": "uniform mean over cached batch-timestep records on the fixed grid",
    },
    {
        "view": "fixed_grid_timestep_macro",
        "avg_cross_entropy_key": "timestep_macro_avg_cross_entropy",
        "pseudo_perplexity_key": "timestep_macro_pseudo_perplexity",
        "bits_key": "timestep_macro_bits_per_masked_token",
        "accuracy_key": "timestep_macro_masked_token_accuracy",
        "ci_container_key": "timestep_confidence_intervals",
        "ci_ce_key": "timestep_macro_avg_cross_entropy",
        "ci_metric_key": "timestep_macro_pseudo_perplexity",
        "ci_accuracy_key": "timestep_macro_masked_token_accuracy",
        "aggregation": "uniform mean over token-weighted per-timestep metrics on the fixed grid",
    },
    {
        "view": "fixed_grid_timestep_auc",
        "avg_cross_entropy_key": "timestep_auc_avg_cross_entropy",
        "pseudo_perplexity_key": "timestep_auc_pseudo_perplexity",
        "bits_key": "timestep_auc_bits_per_masked_token",
        "accuracy_key": "timestep_auc_masked_token_accuracy",
        "ci_container_key": "timestep_confidence_intervals",
        "ci_ce_key": "timestep_auc_avg_cross_entropy",
        "ci_metric_key": "timestep_auc_pseudo_perplexity",
        "ci_accuracy_key": "timestep_auc_masked_token_accuracy",
        "aggregation": "normalized trapezoid integral over token-weighted per-timestep metrics on the fixed grid",
    },
]


COMPARISON_VIEW_SPECS = [
    {
        "metric_view": "sampled_pseudo_perplexity",
        "delta_key": "pseudo_perplexity",
        "winner_key": "pseudo_perplexity",
        "ci_key": "pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "pseudo_perplexity"),
        "linear_value_key": ("model", "pseudo_perplexity"),
    },
    {
        "metric_view": "sampled_bits_per_masked_token",
        "delta_key": "bits_per_masked_token",
        "winner_key": "bits_per_masked_token",
        "ci_key": "bits_per_masked_token",
        "better_direction": "lower",
        "cosine_value_key": ("model", "bits_per_masked_token"),
        "linear_value_key": ("model", "bits_per_masked_token"),
    },
    {
        "metric_view": "sampled_bits_saved_vs_uniform",
        "delta_key": "sampled_bits_saved_vs_uniform",
        "winner_key": "sampled_bits_saved_vs_uniform",
        "ci_key": "sampled_bits_saved_vs_uniform",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "sampled", "bits_saved_vs_uniform"),
        "linear_value_key": ("calibration", "sampled", "bits_saved_vs_uniform"),
    },
    {
        "metric_view": "sampled_denoising_skill",
        "delta_key": "sampled_denoising_skill",
        "winner_key": "sampled_denoising_skill",
        "ci_key": "sampled_denoising_skill",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "sampled", "denoising_skill"),
        "linear_value_key": ("calibration", "sampled", "denoising_skill"),
    },
    {
        "metric_view": "timestep_uniform_pseudo_perplexity",
        "delta_key": "timestep_uniform_pseudo_perplexity",
        "winner_key": "timestep_uniform_pseudo_perplexity",
        "ci_key": "timestep_uniform_pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "timestep_uniform_pseudo_perplexity"),
        "linear_value_key": ("model", "timestep_uniform_pseudo_perplexity"),
    },
    {
        "metric_view": "timestep_uniform_bits_per_masked_token",
        "delta_key": "timestep_uniform_bits_per_masked_token",
        "winner_key": "timestep_uniform_bits_per_masked_token",
        "ci_key": "timestep_uniform_bits_per_masked_token",
        "better_direction": "lower",
        "cosine_value_key": ("model", "timestep_uniform_bits_per_masked_token"),
        "linear_value_key": ("model", "timestep_uniform_bits_per_masked_token"),
    },
    {
        "metric_view": "timestep_uniform_bits_saved_vs_uniform",
        "delta_key": "timestep_uniform_bits_saved_vs_uniform",
        "winner_key": "timestep_uniform_bits_saved_vs_uniform",
        "ci_key": "timestep_uniform_bits_saved_vs_uniform",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "timestep_uniform", "bits_saved_vs_uniform"),
        "linear_value_key": ("calibration", "timestep_uniform", "bits_saved_vs_uniform"),
    },
    {
        "metric_view": "timestep_uniform_denoising_skill",
        "delta_key": "timestep_uniform_denoising_skill",
        "winner_key": "timestep_uniform_denoising_skill",
        "ci_key": "timestep_uniform_denoising_skill",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "timestep_uniform", "denoising_skill"),
        "linear_value_key": ("calibration", "timestep_uniform", "denoising_skill"),
    },
    {
        "metric_view": "schedule_reweighted_pseudo_perplexity",
        "delta_key": "schedule_reweighted_pseudo_perplexity",
        "winner_key": "schedule_reweighted_pseudo_perplexity",
        "ci_key": "schedule_reweighted_pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "schedule_reweighted_pseudo_perplexity"),
        "linear_value_key": ("model", "schedule_reweighted_pseudo_perplexity"),
    },
    {
        "metric_view": "schedule_reweighted_bits_per_masked_token",
        "delta_key": "schedule_reweighted_bits_per_masked_token",
        "winner_key": "schedule_reweighted_bits_per_masked_token",
        "ci_key": "schedule_reweighted_bits_per_masked_token",
        "better_direction": "lower",
        "cosine_value_key": ("model", "schedule_reweighted_bits_per_masked_token"),
        "linear_value_key": ("model", "schedule_reweighted_bits_per_masked_token"),
    },
    {
        "metric_view": "schedule_reweighted_bits_saved_vs_uniform",
        "delta_key": "schedule_reweighted_bits_saved_vs_uniform",
        "winner_key": "schedule_reweighted_bits_saved_vs_uniform",
        "ci_key": "schedule_reweighted_bits_saved_vs_uniform",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "schedule_reweighted", "bits_saved_vs_uniform"),
        "linear_value_key": ("calibration", "schedule_reweighted", "bits_saved_vs_uniform"),
    },
    {
        "metric_view": "schedule_reweighted_denoising_skill",
        "delta_key": "schedule_reweighted_denoising_skill",
        "winner_key": "schedule_reweighted_denoising_skill",
        "ci_key": "schedule_reweighted_denoising_skill",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "schedule_reweighted", "denoising_skill"),
        "linear_value_key": ("calibration", "schedule_reweighted", "denoising_skill"),
    },
    {
        "metric_view": "grid_uniform_pseudo_perplexity",
        "delta_key": "grid_uniform_pseudo_perplexity",
        "winner_key": "grid_uniform_pseudo_perplexity",
        "ci_key": "grid_uniform_pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "grid_uniform_pseudo_perplexity"),
        "linear_value_key": ("model", "grid_uniform_pseudo_perplexity"),
    },
    {
        "metric_view": "grid_uniform_bits_per_masked_token",
        "delta_key": "grid_uniform_bits_per_masked_token",
        "winner_key": "grid_uniform_bits_per_masked_token",
        "ci_key": "grid_uniform_bits_per_masked_token",
        "better_direction": "lower",
        "cosine_value_key": ("model", "grid_uniform_bits_per_masked_token"),
        "linear_value_key": ("model", "grid_uniform_bits_per_masked_token"),
    },
    {
        "metric_view": "grid_uniform_bits_saved_vs_uniform",
        "delta_key": "grid_uniform_bits_saved_vs_uniform",
        "winner_key": "grid_uniform_bits_saved_vs_uniform",
        "ci_key": "grid_uniform_bits_saved_vs_uniform",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "grid_uniform", "bits_saved_vs_uniform"),
        "linear_value_key": ("calibration", "grid_uniform", "bits_saved_vs_uniform"),
    },
    {
        "metric_view": "grid_uniform_denoising_skill",
        "delta_key": "grid_uniform_denoising_skill",
        "winner_key": "grid_uniform_denoising_skill",
        "ci_key": "grid_uniform_denoising_skill",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "grid_uniform", "denoising_skill"),
        "linear_value_key": ("calibration", "grid_uniform", "denoising_skill"),
    },
    {
        "metric_view": "timestep_macro_pseudo_perplexity",
        "delta_key": "timestep_macro_pseudo_perplexity",
        "winner_key": "timestep_macro_pseudo_perplexity",
        "ci_key": "timestep_macro_pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "timestep_macro_pseudo_perplexity"),
        "linear_value_key": ("model", "timestep_macro_pseudo_perplexity"),
    },
    {
        "metric_view": "timestep_macro_bits_per_masked_token",
        "delta_key": "timestep_macro_bits_per_masked_token",
        "winner_key": "timestep_macro_bits_per_masked_token",
        "ci_key": "timestep_macro_bits_per_masked_token",
        "better_direction": "lower",
        "cosine_value_key": ("model", "timestep_macro_bits_per_masked_token"),
        "linear_value_key": ("model", "timestep_macro_bits_per_masked_token"),
    },
    {
        "metric_view": "timestep_macro_bits_saved_vs_uniform",
        "delta_key": "timestep_macro_bits_saved_vs_uniform",
        "winner_key": "timestep_macro_bits_saved_vs_uniform",
        "ci_key": "timestep_macro_bits_saved_vs_uniform",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "timestep_macro", "bits_saved_vs_uniform"),
        "linear_value_key": ("calibration", "timestep_macro", "bits_saved_vs_uniform"),
    },
    {
        "metric_view": "timestep_macro_denoising_skill",
        "delta_key": "timestep_macro_denoising_skill",
        "winner_key": "timestep_macro_denoising_skill",
        "ci_key": "timestep_macro_denoising_skill",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "timestep_macro", "denoising_skill"),
        "linear_value_key": ("calibration", "timestep_macro", "denoising_skill"),
    },
    {
        "metric_view": "timestep_auc_pseudo_perplexity",
        "delta_key": "timestep_auc_pseudo_perplexity",
        "winner_key": "timestep_auc_pseudo_perplexity",
        "ci_key": "timestep_auc_pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "timestep_auc_pseudo_perplexity"),
        "linear_value_key": ("model", "timestep_auc_pseudo_perplexity"),
    },
    {
        "metric_view": "timestep_auc_bits_per_masked_token",
        "delta_key": "timestep_auc_bits_per_masked_token",
        "winner_key": "timestep_auc_bits_per_masked_token",
        "ci_key": "timestep_auc_bits_per_masked_token",
        "better_direction": "lower",
        "cosine_value_key": ("model", "timestep_auc_bits_per_masked_token"),
        "linear_value_key": ("model", "timestep_auc_bits_per_masked_token"),
    },
    {
        "metric_view": "timestep_auc_bits_saved_vs_uniform",
        "delta_key": "timestep_auc_bits_saved_vs_uniform",
        "winner_key": "timestep_auc_bits_saved_vs_uniform",
        "ci_key": "timestep_auc_bits_saved_vs_uniform",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "timestep_auc", "bits_saved_vs_uniform"),
        "linear_value_key": ("calibration", "timestep_auc", "bits_saved_vs_uniform"),
    },
    {
        "metric_view": "timestep_auc_denoising_skill",
        "delta_key": "timestep_auc_denoising_skill",
        "winner_key": "timestep_auc_denoising_skill",
        "ci_key": "timestep_auc_denoising_skill",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "timestep_auc", "denoising_skill"),
        "linear_value_key": ("calibration", "timestep_auc", "denoising_skill"),
    },
    {
        "metric_view": "sampled_accuracy",
        "delta_key": "masked_token_accuracy",
        "winner_key": "masked_token_accuracy",
        "ci_key": "masked_token_accuracy",
        "better_direction": "higher",
        "cosine_value_key": ("model", "masked_token_accuracy"),
        "linear_value_key": ("model", "masked_token_accuracy"),
    },
    {
        "metric_view": "timestep_uniform_accuracy",
        "delta_key": "timestep_uniform_masked_token_accuracy",
        "winner_key": "timestep_uniform_masked_token_accuracy",
        "ci_key": "timestep_uniform_masked_token_accuracy",
        "better_direction": "higher",
        "cosine_value_key": ("model", "timestep_uniform_masked_token_accuracy"),
        "linear_value_key": ("model", "timestep_uniform_masked_token_accuracy"),
    },
    {
        "metric_view": "schedule_reweighted_accuracy",
        "delta_key": "schedule_reweighted_masked_token_accuracy",
        "winner_key": "schedule_reweighted_masked_token_accuracy",
        "ci_key": "schedule_reweighted_masked_token_accuracy",
        "better_direction": "higher",
        "cosine_value_key": ("model", "schedule_reweighted_masked_token_accuracy"),
        "linear_value_key": ("model", "schedule_reweighted_masked_token_accuracy"),
    },
    {
        "metric_view": "grid_uniform_accuracy",
        "delta_key": "grid_uniform_masked_token_accuracy",
        "winner_key": "grid_uniform_masked_token_accuracy",
        "ci_key": "grid_uniform_masked_token_accuracy",
        "better_direction": "higher",
        "cosine_value_key": ("model", "grid_uniform_masked_token_accuracy"),
        "linear_value_key": ("model", "grid_uniform_masked_token_accuracy"),
    },
    {
        "metric_view": "timestep_macro_accuracy",
        "delta_key": "timestep_macro_masked_token_accuracy",
        "winner_key": "timestep_macro_masked_token_accuracy",
        "ci_key": "timestep_macro_masked_token_accuracy",
        "better_direction": "higher",
        "cosine_value_key": ("model", "timestep_macro_masked_token_accuracy"),
        "linear_value_key": ("model", "timestep_macro_masked_token_accuracy"),
    },
    {
        "metric_view": "timestep_auc_accuracy",
        "delta_key": "timestep_auc_masked_token_accuracy",
        "winner_key": "timestep_auc_masked_token_accuracy",
        "ci_key": "timestep_auc_masked_token_accuracy",
        "better_direction": "higher",
        "cosine_value_key": ("model", "timestep_auc_masked_token_accuracy"),
        "linear_value_key": ("model", "timestep_auc_masked_token_accuracy"),
    },
]



def _extract_ci(summary: Optional[dict], container_key: Optional[str], metric_key: Optional[str]) -> dict:
    if not summary or not container_key or not metric_key:
        return {}
    return (summary.get(container_key) or {}).get(metric_key) or {}


def _calibration_interval_from_ce_ci(ce_ci: dict, vocab_size: Optional[int]) -> dict:
    if vocab_size is None or int(vocab_size) <= 0 or not ce_ci:
        return {}
    ce_p05 = ce_ci.get("p05")
    ce_p95 = ce_ci.get("p95")
    if ce_p05 is None or ce_p95 is None:
        return {}
    try:
        ce_p05 = float(ce_p05)
        ce_p95 = float(ce_p95)
    except (TypeError, ValueError):
        return {}
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


def build_eval_view_rows(eval_summary: Optional[dict] = None) -> list:
    if not eval_summary:
        return []

    rows = []
    vocab_size = eval_summary.get("vocab_size")
    for spec in EVAL_VIEW_SPECS:
        metric_ci = _extract_ci(eval_summary, spec["ci_container_key"], spec["ci_metric_key"])
        accuracy_ci = _extract_ci(eval_summary, spec["ci_container_key"], spec["ci_accuracy_key"])
        ce_ci = _extract_ci(eval_summary, spec["ci_container_key"], spec.get("ci_ce_key"))
        avg_cross_entropy = eval_summary.get(spec["avg_cross_entropy_key"])
        bits_per_masked_token = eval_summary.get(spec["bits_key"])
        rows.append(
            {
                "view": spec["view"],
                "aggregation": spec["aggregation"],
                "avg_cross_entropy": avg_cross_entropy,
                "pseudo_perplexity": eval_summary.get(spec["pseudo_perplexity_key"]),
                "bits_per_masked_token": bits_per_masked_token,
                "masked_token_accuracy": eval_summary.get(spec["accuracy_key"]),
                **_view_calibration(avg_cross_entropy, bits_per_masked_token, vocab_size),
                **_calibration_interval_from_ce_ci(ce_ci, vocab_size),
                "pseudo_perplexity_ci_p05": metric_ci.get("p05"),
                "pseudo_perplexity_ci_p95": metric_ci.get("p95"),
                "masked_token_accuracy_ci_p05": accuracy_ci.get("p05"),
                "masked_token_accuracy_ci_p95": accuracy_ci.get("p95"),
            }
        )
    return rows


def _nested_get(container, path):
    value = container
    for key in path:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def build_schedule_comparison_rows(comparison_summary: Optional[dict] = None) -> list:
    if not comparison_summary or len(comparison_summary.get("models", [])) < 2:
        return []

    delta = comparison_summary.get("delta") or {}
    winner = comparison_summary.get("winner") or {}
    winner_confidence = comparison_summary.get("winner_confidence") or {}
    delta_ci = ((comparison_summary.get("delta_confidence_intervals") or {}).get("delta_linear_minus_cosine") or {})
    models = comparison_summary.get("models") or []
    cosine_model = models[0] if len(models) > 0 else {}
    linear_model = models[1] if len(models) > 1 else {}
    calibration = comparison_summary.get("calibration") or {}
    cosine_calibration = calibration.get("cosine_schedule") or {}
    linear_calibration = calibration.get("linear_schedule_baseline") or {}
    rows = []
    for spec in COMPARISON_VIEW_SPECS:
        ci = delta_ci.get(spec["ci_key"]) or {}
        confidence = winner_confidence.get(spec["winner_key"]) or {}
        cosine_value = _nested_get(cosine_model, spec["cosine_value_key"][1:]) if spec["cosine_value_key"][0] == "model" else _nested_get(cosine_calibration, spec["cosine_value_key"][1:])
        linear_value = _nested_get(linear_model, spec["linear_value_key"][1:]) if spec["linear_value_key"][0] == "model" else _nested_get(linear_calibration, spec["linear_value_key"][1:])
        rows.append(
            {
                "metric_view": spec["metric_view"],
                "better_direction": spec["better_direction"],
                "cosine_value": cosine_value,
                "linear_value": linear_value,
                "delta_linear_minus_cosine": delta.get(spec["delta_key"]),
                "winner": winner.get(spec["winner_key"]),
                "bootstrap_p05": ci.get("p05"),
                "bootstrap_p95": ci.get("p95"),
                "probability_linear_better": ci.get("probability_linear_better"),
                "winner_probability": confidence.get("winner_probability"),
                "ci_excludes_zero": confidence.get("ci_excludes_zero"),
                "practically_tied": confidence.get("practically_tied"),
            }
        )
    return rows


def write_eval_summary(local_artifact_dir: str, eval_summary: Optional[dict] = None) -> Optional[str]:
    if not eval_summary:
        return None
    local_artifact_dir = Path(local_artifact_dir)
    out_path = local_artifact_dir / "eval_summary.json"
    out_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    return str(out_path)


def write_schedule_comparison(local_artifact_dir: str, comparison_summary: Optional[dict] = None) -> Optional[str]:
    if not comparison_summary:
        return None
    local_artifact_dir = Path(local_artifact_dir)
    out_path = local_artifact_dir / "schedule_comparison.json"
    out_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    return str(out_path)


def write_eval_plan(local_artifact_dir: str, eval_plan=None) -> Optional[str]:
    if eval_plan is None:
        return None
    local_artifact_dir = Path(local_artifact_dir)
    out_path = local_artifact_dir / "eval_plan.pt"
    import torch

    torch.save(eval_plan, out_path)
    return str(out_path)


def validate_hf_export_bundle(local_artifact_dir: str, repo_id: Optional[str] = None) -> dict:
    local_artifact_dir = Path(local_artifact_dir)
    manifest_path = local_artifact_dir / "hf_export_manifest.json"
    readme_path = local_artifact_dir / "README.md"
    eval_summary_path = local_artifact_dir / "eval_summary.json"
    comparison_path = local_artifact_dir / "schedule_comparison.json"
    eval_plan_path = local_artifact_dir / "eval_plan.pt"

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checks = {
        "artifact_dir_exists": local_artifact_dir.exists(),
        "readme_exists": readme_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "eval_summary_exists": eval_summary_path.exists(),
        "schedule_comparison_exists": comparison_path.exists(),
        "eval_plan_exists": eval_plan_path.exists(),
        "manifest_repo_id_matches": (repo_id is None) or (manifest.get("repo_id") == repo_id),
    }
    return {
        "repo_id": repo_id or manifest.get("repo_id"),
        "local_artifact_dir": str(local_artifact_dir),
        "checks": checks,
        "ready_for_upload": bool(checks["artifact_dir_exists"] and checks["readme_exists"] and checks["manifest_exists"] and checks["manifest_repo_id_matches"]),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "eval_summary_path": str(eval_summary_path) if eval_summary_path.exists() else None,
        "comparison_summary_path": str(comparison_path) if comparison_path.exists() else None,
        "eval_plan_path": str(eval_plan_path) if eval_plan_path.exists() else None,
    }



def write_hf_export_bundle(
    local_artifact_dir: str,
    repo_id: str,
    *,
    eval_summary: Optional[dict] = None,
    comparison_summary: Optional[dict] = None,
    eval_plan=None,
    overwrite_model_card: bool = False,
) -> dict:
    local_artifact_dir = Path(local_artifact_dir)
    local_artifact_dir.mkdir(parents=True, exist_ok=True)
    eval_summary_path = write_eval_summary(local_artifact_dir, eval_summary=eval_summary)
    comparison_path = write_schedule_comparison(local_artifact_dir, comparison_summary=comparison_summary)
    eval_plan_path = write_eval_plan(local_artifact_dir, eval_plan=eval_plan)
    readme_path = ensure_hf_model_card(
        local_artifact_dir,
        repo_id,
        eval_summary=eval_summary,
        comparison_summary=comparison_summary,
        overwrite=overwrite_model_card,
    )
    manifest = {
        "repo_id": repo_id,
        "local_artifact_dir": str(local_artifact_dir),
        "readme_path": readme_path,
        "eval_summary_path": eval_summary_path,
        "comparison_summary_path": comparison_path,
        "eval_plan_path": eval_plan_path,
        "has_eval_summary": eval_summary is not None,
        "has_comparison_summary": comparison_summary is not None,
        "has_eval_plan": eval_plan is not None,
    }
    manifest_path = local_artifact_dir / "hf_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["validation"] = validate_hf_export_bundle(local_artifact_dir, repo_id=repo_id)
    return manifest


def ensure_hf_model_card(
    local_artifact_dir: str,
    repo_id: str,
    eval_summary: Optional[dict] = None,
    comparison_summary: Optional[dict] = None,
    overwrite: bool = False,
) -> str:
    local_artifact_dir = Path(local_artifact_dir)
    readme_path = local_artifact_dir / "README.md"
    if readme_path.exists() and not overwrite:
        return str(readme_path)

    metrics_block = ""
    protocol_block = ""
    quality_block = ""
    comparison_block = ""
    decision_block = ""
    if eval_summary:
        eval_rows = build_eval_view_rows(eval_summary)
        metrics_lines = [
            "\n## Evaluation summary\n",
            "| view | aggregation | avg_cross_entropy | pseudo_perplexity | uniform_random_pseudo_perplexity | bits_per_masked_token | bits_saved_vs_uniform | bits_saved_ci_p05 | bits_saved_ci_p95 | denoising_skill | denoising_skill_ci_p05 | denoising_skill_ci_p95 | masked_token_accuracy | pseudo_perplexity_ci_p05 | pseudo_perplexity_ci_p95 | accuracy_ci_p05 | accuracy_ci_p95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in eval_rows:
            metrics_lines.append(
                f"| {row['view']} | {row['aggregation']} | {row['avg_cross_entropy']} | {row['pseudo_perplexity']} | {row['uniform_random_pseudo_perplexity']} | {row['bits_per_masked_token']} | {row['bits_saved_vs_uniform']} | {row.get('bits_saved_vs_uniform_ci_p05')} | {row.get('bits_saved_vs_uniform_ci_p95')} | {row['denoising_skill']} | {row.get('denoising_skill_ci_p05')} | {row.get('denoising_skill_ci_p95')} | {row['masked_token_accuracy']} | {row['pseudo_perplexity_ci_p05']} | {row['pseudo_perplexity_ci_p95']} | {row['masked_token_accuracy_ci_p05']} | {row['masked_token_accuracy_ci_p95']} |"
            )
        metrics_lines.extend(
            [
                "",
                f"- metric: {eval_summary.get('metric', 'diffusion_pseudo_perplexity')}",
                f"- vocab_size: {eval_summary.get('vocab_size')}",
                f"- uniform_random_pseudo_perplexity: {eval_summary.get('uniform_random_pseudo_perplexity')}",
                f"- uniform_random_avg_cross_entropy: {eval_summary.get('uniform_random_avg_cross_entropy')}",
                f"- uniform_random_bits_per_masked_token: {eval_summary.get('uniform_random_bits_per_masked_token')}",
                f"- sampled_example_count: {eval_summary.get('sampled_example_count')}",
                f"- masked_tokens: {eval_summary.get('masked_tokens')}",
                f"- n_batches: {eval_summary.get('n_batches')}",
                f"- schedule_reweighted_nonzero_examples: {eval_summary.get('schedule_reweighted_nonzero_examples')}",
                f"- schedule_reweighted_estimated_eligible_token_count: {eval_summary.get('schedule_reweighted_estimated_eligible_token_count')}",
                f"- schedule_reweighted_effective_sample_size: {eval_summary.get('schedule_reweighted_effective_sample_size')}",
                f"- schedule_reweighted_effective_sample_size_fraction: {eval_summary.get('schedule_reweighted_effective_sample_size_fraction')}",
                f"- timestep_macro_timestep_count: {eval_summary.get('timestep_macro_timestep_count')}",
                f"- timestep_auc_timestep_count: {eval_summary.get('timestep_auc_timestep_count')}",
                f"- timestep_auc_fraction_span: {eval_summary.get('timestep_auc_fraction_span')}",
            ]
        )
        metrics_block = "\n".join(metrics_lines) + "\n"
        protocol = eval_summary.get("eval_protocol") or {}
        if protocol:
            protocol_block = (
                "\n## Evaluation protocol\n\n"
                f"- schedule_name: {protocol.get('schedule_name')}\n"
                f"- diffusion_steps: {protocol.get('T')}\n"
                f"- timestep_grid: {protocol.get('timestep_grid')}\n"
                f"- timestep_grid_fractions: {protocol.get('timestep_grid_fractions')}\n"
                f"- eval_plan_T: {protocol.get('eval_plan_T')}\n"
                f"- normalized_timestep_remapping: {protocol.get('normalized_timestep_remapping')}\n"
                f"- paired_noise: {protocol.get('paired_noise')}\n"
                f"- paired_batches: {protocol.get('paired_batches')}\n"
                f"- sampled_timestep_distribution: {protocol.get('sampled_timestep_distribution')}\n"
                f"- schedule_reweighted_aggregation: {protocol.get('schedule_reweighted_aggregation')}\n"
                f"- grid_uniform_aggregation: {protocol.get('grid_uniform_aggregation')}\n"
                f"- timestep_macro_aggregation: {protocol.get('timestep_macro_aggregation')}\n"
                f"- timestep_auc_aggregation: {protocol.get('timestep_auc_aggregation')}\n"
                f"- bootstrap_samples: {protocol.get('bootstrap_samples')}\n"
            )
        quality_summary = eval_summary.get("quality_summary") or {}
        if quality_summary:
            quality_lines = [
                "\n## Evaluation quality summary\n",
                f"- schedule_reweighted_reliability: {quality_summary.get('schedule_reweighted_reliability')}",
                f"- schedule_reweighted_effective_sample_size: {quality_summary.get('schedule_reweighted_effective_sample_size')}",
                f"- schedule_reweighted_effective_sample_size_fraction: {quality_summary.get('schedule_reweighted_effective_sample_size_fraction')}",
                f"- sampled_example_count: {quality_summary.get('sampled_example_count')}",
                f"- masked_tokens: {quality_summary.get('masked_tokens')}",
                f"- timestep_macro_timestep_count: {quality_summary.get('timestep_macro_timestep_count')}",
                f"- normalized_timestep_remapping: {quality_summary.get('normalized_timestep_remapping')}",
            ]
            notes = quality_summary.get("notes") or []
            warnings = quality_summary.get("warnings") or []
            if notes:
                quality_lines.append("- notes:")
                quality_lines.extend([f"  - {note}" for note in notes])
            if warnings:
                quality_lines.append("- warnings:")
                quality_lines.extend([f"  - {warning}" for warning in warnings])
            quality_block = "\n".join(quality_lines) + "\n"

    comparison_rows = build_schedule_comparison_rows(comparison_summary)
    if comparison_rows:
        comparison_lines = [
            "\n## Schedule comparison\n",
            "| metric_view | better_direction | cosine_value | linear_value | delta_linear_minus_cosine | winner | winner_probability | ci_excludes_zero | practically_tied | bootstrap_p05 | bootstrap_p95 | probability_linear_better |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
        for row in comparison_rows:
            comparison_lines.append(
                f"| {row['metric_view']} | {row['better_direction']} | {row['cosine_value']} | {row['linear_value']} | {row['delta_linear_minus_cosine']} | {row['winner']} | {row.get('winner_probability')} | {row.get('ci_excludes_zero')} | {row.get('practically_tied')} | {row['bootstrap_p05']} | {row['bootstrap_p95']} | {row['probability_linear_better']} |"
            )
        delta = comparison_summary.get("delta") or {}
        comparison_lines.extend(
            [
                "",
                f"- avg_cross_entropy_delta_linear_minus_cosine: {delta.get('avg_cross_entropy')}",
                f"- timestep_uniform_avg_cross_entropy_delta_linear_minus_cosine: {delta.get('timestep_uniform_avg_cross_entropy')}",
                f"- grid_uniform_avg_cross_entropy_delta_linear_minus_cosine: {delta.get('grid_uniform_avg_cross_entropy')}",
                f"- timestep_macro_avg_cross_entropy_delta_linear_minus_cosine: {delta.get('timestep_macro_avg_cross_entropy')}",
                f"- timestep_macro_bits_per_masked_token_delta_linear_minus_cosine: {delta.get('timestep_macro_bits_per_masked_token')}",
                f"- timestep_auc_avg_cross_entropy_delta_linear_minus_cosine: {delta.get('timestep_auc_avg_cross_entropy')}",
                f"- timestep_auc_bits_per_masked_token_delta_linear_minus_cosine: {delta.get('timestep_auc_bits_per_masked_token')}",
            ]
        )
        comparison_block = "\n".join(comparison_lines) + "\n"
        decision_summary = comparison_summary.get("decision_summary") or {}
        if decision_summary:
            decision_lines = [
                "\n## Schedule decision summary\n",
                f"- headline: {decision_summary.get('headline')}",
                f"- tracked_metric_count: {decision_summary.get('tracked_metric_count')}",
                f"- decisive_metric_count: {decision_summary.get('decisive_metric_count')}",
                f"- practically_tied_metric_count: {decision_summary.get('practically_tied_metric_count')}",
                f"- cosine_schedule_win_count: {decision_summary.get('cosine_schedule_win_count')}",
                f"- linear_schedule_win_count: {decision_summary.get('linear_schedule_win_count')}",
            ]
            tracked = decision_summary.get('tracked_metrics') or []
            if tracked:
                decision_lines.extend([
                    "",
                    "| metric | winner | better_direction | winner_probability | ci_excludes_zero | practically_tied |",
                    "| --- | --- | --- | ---: | --- | --- |",
                ])
                for row in tracked:
                    decision_lines.append(
                        f"| {row.get('metric')} | {row.get('winner')} | {row.get('better_direction')} | {row.get('winner_probability')} | {row.get('ci_excludes_zero')} | {row.get('practically_tied')} |"
                    )
            decision_block = "\n".join(decision_lines) + "\n"

    readme_path.write_text(
        f"""---
library_name: pytorch
tags:
- diffusion-language-model
- tinystories
- denoising-language-model
- masked-diffusion
---

# {repo_id}

This model repo contains artifacts exported from the DM_Labs notebook for a discrete diffusion language model trained on TinyStories.

## Contents

- `model.pt`
- `config.json`
- tokenizer files
- optional `eval_summary.json`
- optional `schedule_comparison.json`
- optional `eval_plan.pt` shared cached evaluation plan artifact
- optional `hf_export_manifest.json` bundle manifest covering all exported metadata files
- optional schedule-comparison JSON artifacts
- optional bundle validation metadata inside `hf_export_manifest.json`
{metrics_block}{protocol_block}{quality_block}{comparison_block}{decision_block}## Evaluation note

The reported pseudo-perplexity is based on masked-token denoising NLL under a diffusion corruption process. It is **not** autoregressive next-token perplexity.

To make the pseudo-perplexity-style views easier to interpret, the exported tables also calibrate each aggregate against a same-vocabulary uniform-random baseline, exposing `bits_saved_vs_uniform` and `denoising_skill = 1 - CE / log(|V|)`.

For reproducibility, the notebook can export the exact evaluation summary used for this upload into `eval_summary.json`, the paired linear-vs-cosine summary into `schedule_comparison.json`, and the shared cached batch/timestep/noise plan into `eval_plan.pt`.
""",
        encoding="utf-8",
    )
    return str(readme_path)


def upload_checkpoint_to_hub(
    local_artifact_dir: str,
    *,
    hf_token: Optional[str] = None,
    hf_username: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    private: bool = False,
    eval_summary: Optional[dict] = None,
    comparison_summary: Optional[dict] = None,
    eval_plan=None,
    overwrite_model_card: bool = False,
    commit_message: str = "Upload DM_Labs diffusion LM artifacts",
) -> str:
    hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    hf_username = hf_username or os.getenv("HF_USERNAME")
    hf_repo_id = hf_repo_id or os.getenv("HF_REPO_ID")

    if hf_token is None:
        raise ValueError("Set HF_TOKEN or HUGGINGFACE_TOKEN before uploading to Hugging Face Hub.")
    if hf_repo_id is None:
        if not hf_username:
            raise ValueError("Set HF_USERNAME or HF_REPO_ID before uploading to Hugging Face Hub.")
        hf_repo_id = f"{hf_username}/tinystories-diffusion-lm"

    write_hf_export_bundle(
        local_artifact_dir,
        hf_repo_id,
        eval_summary=eval_summary,
        comparison_summary=comparison_summary,
        eval_plan=eval_plan,
        overwrite_model_card=overwrite_model_card,
    )

    from huggingface_hub import create_repo, login, upload_folder

    login(token=hf_token)
    create_repo(repo_id=hf_repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(
        repo_id=hf_repo_id,
        folder_path=local_artifact_dir,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{hf_repo_id}"
