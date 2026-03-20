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
        "estimand": "observed masked-token denoising CE under the sampled schedule-induced corruption mix",
        "sample_axis": "sampled masked tokens pooled across cached batches",
        "weighting": "token-weighted",
        "comparison_semantics": "useful descriptive sampled aggregate; not schedule-corrected",
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
        "estimand": "uniform-over-sampled-timesteps per-example denoising CE",
        "sample_axis": "sampled per-example corruption draws",
        "weighting": "equal weight per sampled example/timestep draw",
        "comparison_semantics": "schedule-agnostic sampled view with equal top-level timestep weight",
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
        "estimand": "uniform-over-eligible-token-and-timestep denoising CE estimated from sampled draws",
        "sample_axis": "sampled masked tokens with inverse expected mask-ratio correction",
        "weighting": "importance-weighted by inverse expected mask ratio",
        "comparison_semantics": "closest sampled estimator to a schedule-corrected diffusion perplexity objective when ESS is healthy",
    },
    {
        "view": "schedule_reweighted_ht",
        "avg_cross_entropy_key": "schedule_reweighted_ht_avg_cross_entropy",
        "pseudo_perplexity_key": "schedule_reweighted_ht_pseudo_perplexity",
        "bits_key": "schedule_reweighted_ht_bits_per_masked_token",
        "accuracy_key": "schedule_reweighted_ht_masked_token_accuracy",
        "ci_container_key": "schedule_reweighted_ht_confidence_intervals",
        "ci_ce_key": "schedule_reweighted_ht_avg_cross_entropy",
        "ci_metric_key": "schedule_reweighted_ht_pseudo_perplexity",
        "ci_accuracy_key": "schedule_reweighted_ht_masked_token_accuracy",
        "aggregation": "Horvitz-Thompson inverse-mask-ratio estimate normalized by exact eligible-token count",
        "estimand": "population-normalized uniform-over-eligible-token-and-timestep denoising CE",
        "sample_axis": "sampled masked tokens plus exact eligible-token denominator",
        "weighting": "Horvitz-Thompson inverse-probability weighting",
        "comparison_semantics": "schedule-corrected population-style estimator with explicit eligible-token normalization",
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
        "estimand": "fixed diagnostic-grid denoising CE averaged over cached batch-timestep records",
        "sample_axis": "cached batch-timestep grid records",
        "weighting": "equal weight per cached batch-timestep record",
        "comparison_semantics": "explicit shared-grid diagnostic; depends on chosen timestep grid",
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
        "estimand": "fixed-grid denoising CE with equal top-level weight per diagnostic timestep",
        "sample_axis": "token-weighted per-timestep summaries on the cached grid",
        "weighting": "equal weight per timestep after within-timestep token aggregation",
        "comparison_semantics": "good for stage-balanced schedule comparison on a shared grid",
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
        "estimand": "trajectory-integrated denoising CE over normalized timestep fraction on the fixed grid",
        "sample_axis": "token-weighted per-timestep summaries on the cached normalized-time grid",
        "weighting": "trapezoidal integration over normalized timestep fraction",
        "comparison_semantics": "most conservative shared comparison view when schedules only align by normalized timestep fraction",
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
        "comparison_scope": "sampled schedule-mix aggregate",
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
        "comparison_scope": "sampled equal-timestep view",
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
        "comparison_scope": "sampled schedule-corrected objective estimate",
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
        "metric_view": "schedule_reweighted_ht_pseudo_perplexity",
        "delta_key": "schedule_reweighted_ht_pseudo_perplexity",
        "winner_key": "schedule_reweighted_ht_pseudo_perplexity",
        "ci_key": "schedule_reweighted_ht_pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "schedule_reweighted_ht_pseudo_perplexity"),
        "linear_value_key": ("model", "schedule_reweighted_ht_pseudo_perplexity"),
        "comparison_scope": "sampled Horvitz-Thompson schedule-corrected objective estimate",
    },
    {
        "metric_view": "schedule_reweighted_ht_bits_per_masked_token",
        "delta_key": "schedule_reweighted_ht_bits_per_masked_token",
        "winner_key": "schedule_reweighted_ht_bits_per_masked_token",
        "ci_key": "schedule_reweighted_ht_bits_per_masked_token",
        "better_direction": "lower",
        "cosine_value_key": ("model", "schedule_reweighted_ht_bits_per_masked_token"),
        "linear_value_key": ("model", "schedule_reweighted_ht_bits_per_masked_token"),
    },
    {
        "metric_view": "schedule_reweighted_ht_bits_saved_vs_uniform",
        "delta_key": "schedule_reweighted_ht_bits_saved_vs_uniform",
        "winner_key": "schedule_reweighted_ht_bits_saved_vs_uniform",
        "ci_key": "schedule_reweighted_ht_bits_saved_vs_uniform",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "schedule_reweighted_ht", "bits_saved_vs_uniform"),
        "linear_value_key": ("calibration", "schedule_reweighted_ht", "bits_saved_vs_uniform"),
    },
    {
        "metric_view": "schedule_reweighted_ht_denoising_skill",
        "delta_key": "schedule_reweighted_ht_denoising_skill",
        "winner_key": "schedule_reweighted_ht_denoising_skill",
        "ci_key": "schedule_reweighted_ht_denoising_skill",
        "better_direction": "higher",
        "cosine_value_key": ("calibration", "schedule_reweighted_ht", "denoising_skill"),
        "linear_value_key": ("calibration", "schedule_reweighted_ht", "denoising_skill"),
    },
    {
        "metric_view": "grid_uniform_pseudo_perplexity",
        "delta_key": "grid_uniform_pseudo_perplexity",
        "winner_key": "grid_uniform_pseudo_perplexity",
        "ci_key": "grid_uniform_pseudo_perplexity",
        "better_direction": "lower",
        "cosine_value_key": ("model", "grid_uniform_pseudo_perplexity"),
        "linear_value_key": ("model", "grid_uniform_pseudo_perplexity"),
        "comparison_scope": "shared fixed-grid batch-timestep diagnostic",
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
        "comparison_scope": "shared fixed-grid equal-timestep diagnostic",
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
        "comparison_scope": "shared normalized-timestep trajectory diagnostic",
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
        "metric_view": "schedule_reweighted_ht_accuracy",
        "delta_key": "schedule_reweighted_ht_masked_token_accuracy",
        "winner_key": "schedule_reweighted_ht_masked_token_accuracy",
        "ci_key": "schedule_reweighted_ht_masked_token_accuracy",
        "better_direction": "higher",
        "cosine_value_key": ("model", "schedule_reweighted_ht_masked_token_accuracy"),
        "linear_value_key": ("model", "schedule_reweighted_ht_masked_token_accuracy"),
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
                "estimand": spec.get("estimand"),
                "sample_axis": spec.get("sample_axis"),
                "weighting": spec.get("weighting"),
                "comparison_semantics": spec.get("comparison_semantics"),
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


def build_eval_protocol_rows(eval_summary: Optional[dict] = None) -> list:
    if not eval_summary:
        return []

    protocol = eval_summary.get("eval_protocol") or {}
    quality_summary = eval_summary.get("quality_summary") or {}
    recommended_view = ((quality_summary.get("recommended_primary_view") or {}).get("view"))
    rows = []
    for spec in EVAL_VIEW_SPECS:
        rows.append(
            {
                "view": spec["view"],
                "estimand": spec.get("estimand"),
                "sample_axis": spec.get("sample_axis"),
                "weighting": spec.get("weighting"),
                "comparison_semantics": spec.get("comparison_semantics"),
                "is_recommended_primary_view": bool(spec["view"] == recommended_view),
                "normalized_timestep_remapping": bool(protocol.get("normalized_timestep_remapping", False)),
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
                "comparison_scope": spec.get("comparison_scope", "paired shared-plan comparison metric"),
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


def build_comparison_protocol_rows(comparison_summary: Optional[dict] = None) -> list:
    if not comparison_summary or len(comparison_summary.get("models", [])) < 2:
        return []

    decision_summary = comparison_summary.get("decision_summary") or {}
    recommended_metric = ((decision_summary.get("recommended_primary_metric") or {}).get("metric"))
    protocol = comparison_summary.get("comparison_protocol") or {}
    rows = []
    for spec in COMPARISON_VIEW_SPECS:
        rows.append(
            {
                "metric_view": spec["metric_view"],
                "better_direction": spec["better_direction"],
                "comparison_scope": spec.get("comparison_scope", "paired shared-plan comparison metric"),
                "is_recommended_primary_metric": bool(spec["winner_key"] == recommended_metric),
                "normalized_timestep_remapping": bool(protocol.get("normalized_timestep_remapping", False)),
                "paired_batches": bool(protocol.get("paired_batches", False)),
                "paired_uniform_noise": bool(protocol.get("paired_uniform_noise", False)),
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


def inspect_hf_artifact_dir(local_artifact_dir: str) -> dict:
    local_artifact_dir = Path(local_artifact_dir)
    model_path = local_artifact_dir / "model.pt"
    config_path = local_artifact_dir / "config.json"
    tokenizer_candidates = {
        "tokenizer_json": local_artifact_dir / "tokenizer.json",
        "tokenizer_config_json": local_artifact_dir / "tokenizer_config.json",
        "special_tokens_map_json": local_artifact_dir / "special_tokens_map.json",
        "vocab_json": local_artifact_dir / "vocab.json",
        "merges_txt": local_artifact_dir / "merges.txt",
    }
    tokenizer_present = {name: path.exists() for name, path in tokenizer_candidates.items()}
    tokenizer_family = {
        "fast_tokenizer_bundle": bool(tokenizer_present["tokenizer_json"] and tokenizer_present["tokenizer_config_json"]),
        "bpe_vocab_merge_pair": bool(tokenizer_present["vocab_json"] and tokenizer_present["merges_txt"]),
        "special_tokens_map": bool(tokenizer_present["special_tokens_map_json"]),
    }
    return {
        "local_artifact_dir": str(local_artifact_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "tokenizer_paths": {name: str(path) for name, path in tokenizer_candidates.items()},
        "checks": {
            "artifact_dir_exists": local_artifact_dir.exists(),
            "model_exists": model_path.exists(),
            "config_exists": config_path.exists(),
            **tokenizer_present,
            "tokenizer_assets_present": bool(any(tokenizer_present.values())),
            "tokenizer_loadable_bundle_present": bool(tokenizer_family["fast_tokenizer_bundle"] or tokenizer_family["bpe_vocab_merge_pair"]),
            "special_tokens_map_present": tokenizer_family["special_tokens_map"],
        },
    }


def summarize_eval_for_hf(eval_summary: Optional[dict] = None) -> dict:
    if not eval_summary:
        return {}
    quality_summary = eval_summary.get("quality_summary") or {}
    primary = eval_summary.get("primary_metric_snapshot") or {}
    if not primary:
        recommended = quality_summary.get("recommended_primary_view") or {}
        metric_key = recommended.get("metric_key")
        view_name = recommended.get("view")
        matching_spec = next((spec for spec in EVAL_VIEW_SPECS if spec["view"] == view_name), None)
        metric_ci = {}
        accuracy_value = None
        calibration = {}
        if matching_spec is not None:
            metric_ci = _extract_ci(eval_summary, matching_spec.get("ci_container_key"), matching_spec.get("ci_metric_key"))
            accuracy_value = eval_summary.get(matching_spec.get("accuracy_key"))
            calibration = _view_calibration(
                eval_summary.get(matching_spec.get("avg_cross_entropy_key")),
                eval_summary.get(matching_spec.get("bits_key")),
                eval_summary.get("vocab_size"),
            )
            calibration.update(
                _calibration_interval_from_ce_ci(
                    _extract_ci(eval_summary, matching_spec.get("ci_container_key"), matching_spec.get("ci_ce_key")),
                    eval_summary.get("vocab_size"),
                )
            )
        primary = {
            "view": view_name,
            "metric_key": metric_key,
            "metric_value": eval_summary.get(metric_key) if metric_key else None,
            "metric_confidence_interval": metric_ci,
            "better_direction": recommended.get("better_direction"),
            "estimand": matching_spec.get("estimand") if matching_spec else None,
            "sample_axis": matching_spec.get("sample_axis") if matching_spec else None,
            "weighting": matching_spec.get("weighting") if matching_spec else None,
            "comparison_semantics": matching_spec.get("comparison_semantics") if matching_spec else None,
            "masked_token_accuracy_value": accuracy_value,
            "bits_saved_vs_uniform": calibration.get("bits_saved_vs_uniform"),
            "bits_saved_vs_uniform_ci_p05": calibration.get("bits_saved_vs_uniform_ci_p05"),
            "bits_saved_vs_uniform_ci_p95": calibration.get("bits_saved_vs_uniform_ci_p95"),
            "denoising_skill": calibration.get("denoising_skill"),
            "denoising_skill_ci_p05": calibration.get("denoising_skill_ci_p05"),
            "denoising_skill_ci_p95": calibration.get("denoising_skill_ci_p95"),
            "rationale": recommended.get("rationale"),
            "caveat": recommended.get("caveat"),
        }
    metric_key = primary.get("metric_key")
    return {
        "metric": eval_summary.get("metric"),
        "primary_view": primary.get("view"),
        "primary_metric_key": metric_key,
        "primary_metric_value": primary.get("metric_value") if metric_key else None,
        "primary_metric_ci": primary.get("metric_confidence_interval") or {},
        "primary_better_direction": primary.get("better_direction"),
        "primary_estimand": primary.get("estimand"),
        "primary_sample_axis": primary.get("sample_axis"),
        "primary_weighting": primary.get("weighting"),
        "primary_comparison_semantics": primary.get("comparison_semantics"),
        "primary_bits_saved_vs_uniform": primary.get("bits_saved_vs_uniform"),
        "primary_denoising_skill": primary.get("denoising_skill"),
        "primary_masked_token_accuracy": primary.get("masked_token_accuracy_value"),
        "primary_rationale": primary.get("rationale"),
        "primary_caveat": primary.get("caveat"),
        "schedule_reweighted_reliability": quality_summary.get("schedule_reweighted_reliability"),
        "schedule_reweighted_effective_sample_size_fraction": eval_summary.get("schedule_reweighted_effective_sample_size_fraction"),
        "schedule_reweighted_exact_eligible_token_count": eval_summary.get("schedule_reweighted_exact_eligible_token_count"),
        "schedule_reweighted_expected_masked_token_count": eval_summary.get("schedule_reweighted_expected_masked_token_count"),
        "sampled_example_count": eval_summary.get("sampled_example_count"),
        "masked_tokens": eval_summary.get("masked_tokens"),
        "normalized_timestep_remapping": bool((eval_summary.get("eval_protocol") or {}).get("normalized_timestep_remapping", False)),
    }


def summarize_comparison_for_hf(comparison_summary: Optional[dict] = None) -> dict:
    if not comparison_summary:
        return {}
    decision_summary = comparison_summary.get("decision_summary") or {}
    primary = comparison_summary.get("primary_metric_snapshot") or {}
    if not primary:
        recommended = decision_summary.get("recommended_primary_metric") or {}
        metric_key = recommended.get("metric")
        primary = {
            "metric": metric_key,
            "view": recommended.get("view"),
            "winner": recommended.get("winner"),
            "comparison_scope": next((spec.get("comparison_scope") for spec in COMPARISON_VIEW_SPECS if spec.get("winner_key") == metric_key), None),
            "delta_linear_minus_cosine": (comparison_summary.get("delta") or {}).get(metric_key) if metric_key else None,
            "winner_confidence": (comparison_summary.get("winner_confidence") or {}).get(metric_key) or {},
            "delta_confidence_interval": ((comparison_summary.get("delta_confidence_intervals") or {}).get("delta_linear_minus_cosine") or {}).get(metric_key) or {},
            "rationale": recommended.get("rationale"),
        }
    winner_confidence = primary.get("winner_confidence") or {}
    delta_ci = primary.get("delta_confidence_interval") or {}
    return {
        "headline": decision_summary.get("headline"),
        "primary_metric": primary.get("metric"),
        "primary_view": primary.get("view"),
        "primary_comparison_scope": primary.get("comparison_scope"),
        "primary_winner": primary.get("winner"),
        "primary_winner_probability": winner_confidence.get("winner_probability"),
        "primary_ci_excludes_zero": winner_confidence.get("ci_excludes_zero"),
        "primary_practically_tied": winner_confidence.get("practically_tied"),
        "primary_delta_linear_minus_cosine": primary.get("delta_linear_minus_cosine"),
        "primary_delta_ci": delta_ci,
        "primary_cosine_value": primary.get("cosine_value"),
        "primary_linear_value": primary.get("linear_value"),
        "primary_rationale": primary.get("rationale"),
        "normalized_timestep_remapping": bool((comparison_summary.get("comparison_protocol") or {}).get("normalized_timestep_remapping", False)),
        "tracked_metric_count": decision_summary.get("tracked_metric_count"),
        "decisive_metric_count": decision_summary.get("decisive_metric_count"),
    }


def _format_report_scalar(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}g}"
    return str(value)



def render_eval_summary_report(
    eval_summary: Optional[dict] = None,
    comparison_summary: Optional[dict] = None,
) -> str:
    lines = []

    eval_snapshot = summarize_eval_for_hf(eval_summary)
    if eval_snapshot:
        eval_rows = {row["view"]: row for row in build_eval_view_rows(eval_summary)}
        primary_view = eval_snapshot.get("primary_view")
        primary_row = eval_rows.get(primary_view or "", {})
        lines.extend(
            [
                "Single-model evaluation",
                f"- primary_view: {_format_report_scalar(primary_view)}",
                f"- primary_metric: {_format_report_scalar(eval_snapshot.get('primary_metric_key'))} = {_format_report_scalar(eval_snapshot.get('primary_metric_value'))} ({_format_report_scalar(eval_snapshot.get('primary_better_direction'))} is better)",
                f"- primary_metric_ci_p05: {_format_report_scalar((eval_snapshot.get('primary_metric_ci') or {}).get('p05'))}",
                f"- primary_metric_ci_p95: {_format_report_scalar((eval_snapshot.get('primary_metric_ci') or {}).get('p95'))}",
                f"- primary_estimand: {_format_report_scalar(eval_snapshot.get('primary_estimand'))}",
                f"- primary_sample_axis: {_format_report_scalar(eval_snapshot.get('primary_sample_axis'))}",
                f"- primary_weighting: {_format_report_scalar(eval_snapshot.get('primary_weighting'))}",
                f"- primary_comparison_semantics: {_format_report_scalar(eval_snapshot.get('primary_comparison_semantics'))}",
                f"- schedule_reweighted_reliability: {_format_report_scalar(eval_snapshot.get('schedule_reweighted_reliability'))}",
                f"- sampled_example_count: {_format_report_scalar(eval_snapshot.get('sampled_example_count'))}",
                f"- masked_tokens: {_format_report_scalar(eval_snapshot.get('masked_tokens'))}",
                f"- normalized_timestep_remapping: {_format_report_scalar(eval_snapshot.get('normalized_timestep_remapping'))}",
            ]
        )
        if primary_row:
            lines.extend(
                [
                    f"- primary_bits_per_masked_token: {_format_report_scalar(primary_row.get('bits_per_masked_token'))}",
                    f"- primary_bits_saved_vs_uniform: {_format_report_scalar(eval_snapshot.get('primary_bits_saved_vs_uniform'))}",
                    f"- primary_denoising_skill: {_format_report_scalar(eval_snapshot.get('primary_denoising_skill'))}",
                    f"- primary_masked_token_accuracy: {_format_report_scalar(eval_snapshot.get('primary_masked_token_accuracy'))}",
                ]
            )
        lines.extend(
            [
                f"- schedule_reweighted_ht_pseudo_perplexity: {_format_report_scalar(eval_summary.get('schedule_reweighted_ht_pseudo_perplexity'))}",
                f"- schedule_reweighted_ht_bits_saved_vs_uniform: {_format_report_scalar(((eval_summary.get('calibration') or {}).get('schedule_reweighted_ht') or {}).get('bits_saved_vs_uniform'))}",
                f"- schedule_reweighted_exact_eligible_token_count: {_format_report_scalar(eval_summary.get('schedule_reweighted_exact_eligible_token_count'))}",
            ]
        )
        rationale = eval_snapshot.get("primary_rationale")
        if rationale:
            lines.append(f"- rationale: {rationale}")

    comparison_snapshot = summarize_comparison_for_hf(comparison_summary)
    if comparison_snapshot:
        comparison_rows = {row["metric_view"]: row for row in build_schedule_comparison_rows(comparison_summary)}
        primary_metric = comparison_snapshot.get("primary_metric")
        primary_row = comparison_rows.get(primary_metric or "", {})
        if lines:
            lines.append("")
        lines.extend(
            [
                "Linear-vs-cosine comparison",
                f"- headline: {_format_report_scalar(comparison_snapshot.get('headline'))}",
                f"- primary_metric: {_format_report_scalar(primary_metric)}",
                f"- primary_view: {_format_report_scalar(comparison_snapshot.get('primary_view'))}",
                f"- primary_comparison_scope: {_format_report_scalar(comparison_snapshot.get('primary_comparison_scope'))}",
                f"- primary_winner: {_format_report_scalar(comparison_snapshot.get('primary_winner'))}",
                f"- winner_probability: {_format_report_scalar(comparison_snapshot.get('primary_winner_probability'))}",
                f"- ci_excludes_zero: {_format_report_scalar(comparison_snapshot.get('primary_ci_excludes_zero'))}",
                f"- practically_tied: {_format_report_scalar(comparison_snapshot.get('primary_practically_tied'))}",
                f"- delta_linear_minus_cosine: {_format_report_scalar(comparison_snapshot.get('primary_delta_linear_minus_cosine'))}",
                f"- delta_ci_p05: {_format_report_scalar((comparison_snapshot.get('primary_delta_ci') or {}).get('p05'))}",
                f"- delta_ci_p95: {_format_report_scalar((comparison_snapshot.get('primary_delta_ci') or {}).get('p95'))}",
                f"- normalized_timestep_remapping: {_format_report_scalar(comparison_snapshot.get('normalized_timestep_remapping'))}",
            ]
        )
        if primary_row:
            lines.extend(
                [
                    f"- cosine_value: {_format_report_scalar(primary_row.get('cosine_value'))}",
                    f"- linear_value: {_format_report_scalar(primary_row.get('linear_value'))}",
                    f"- better_direction: {_format_report_scalar(primary_row.get('better_direction'))}",
                    f"- bootstrap_p05: {_format_report_scalar(primary_row.get('bootstrap_p05'))}",
                    f"- bootstrap_p95: {_format_report_scalar(primary_row.get('bootstrap_p95'))}",
                    f"- probability_linear_better: {_format_report_scalar(primary_row.get('probability_linear_better'))}",
                ]
            )

    return "\n".join(lines)



def render_hf_preflight_report(validation: Optional[dict] = None) -> str:
    validation = validation or {}
    checks = validation.get("checks") or {}
    missing_required = validation.get("missing_required") or []
    warnings = validation.get("warnings") or []
    lines = [
        "Hugging Face export preflight",
        f"- repo_id: {_format_report_scalar(validation.get('repo_id'))}",
        f"- local_artifact_dir: {_format_report_scalar(validation.get('local_artifact_dir'))}",
        f"- ready_for_upload: {_format_report_scalar(validation.get('ready_for_upload'))}",
        f"- eval_report_exists: {_format_report_scalar(checks.get('eval_report_exists'))}",
        f"- preflight_report_exists: {_format_report_scalar(checks.get('preflight_report_exists'))}",
        f"- model_exists: {_format_report_scalar(checks.get('model_exists'))}",
        f"- config_exists: {_format_report_scalar(checks.get('config_exists'))}",
        f"- tokenizer_loadable_bundle_present: {_format_report_scalar(checks.get('tokenizer_loadable_bundle_present'))}",
        f"- readme_exists: {_format_report_scalar(checks.get('readme_exists'))}",
        f"- manifest_exists: {_format_report_scalar(checks.get('manifest_exists'))}",
        f"- eval_summary_exists: {_format_report_scalar(checks.get('eval_summary_exists'))}",
        f"- schedule_comparison_exists: {_format_report_scalar(checks.get('schedule_comparison_exists'))}",
        f"- eval_plan_exists: {_format_report_scalar(checks.get('eval_plan_exists'))}",
    ]
    if missing_required:
        lines.append(f"- missing_required: {', '.join(missing_required)}")
    if warnings:
        lines.append("- warnings:")
        lines.extend([f"  - {warning}" for warning in warnings])
    return "\n".join(lines)



def write_eval_summary_report(
    local_artifact_dir: str,
    *,
    eval_summary: Optional[dict] = None,
    comparison_summary: Optional[dict] = None,
) -> Optional[str]:
    report = render_eval_summary_report(eval_summary=eval_summary, comparison_summary=comparison_summary)
    if not report.strip():
        return None
    local_artifact_dir = Path(local_artifact_dir)
    out_path = local_artifact_dir / "eval_summary_report.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    return str(out_path)



def write_hf_preflight_report(local_artifact_dir: str, validation: Optional[dict] = None) -> Optional[str]:
    report = render_hf_preflight_report(validation)
    if not report.strip():
        return None
    local_artifact_dir = Path(local_artifact_dir)
    out_path = local_artifact_dir / "hf_preflight_report.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    return str(out_path)



def validate_hf_export_bundle(local_artifact_dir: str, repo_id: Optional[str] = None) -> dict:
    local_artifact_dir = Path(local_artifact_dir)
    manifest_path = local_artifact_dir / "hf_export_manifest.json"
    readme_path = local_artifact_dir / "README.md"
    eval_summary_path = local_artifact_dir / "eval_summary.json"
    comparison_path = local_artifact_dir / "schedule_comparison.json"
    eval_plan_path = local_artifact_dir / "eval_plan.pt"
    eval_report_path = local_artifact_dir / "eval_summary_report.txt"
    preflight_report_path = local_artifact_dir / "hf_preflight_report.txt"

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifact_inspection = inspect_hf_artifact_dir(local_artifact_dir)
    checks = {
        "artifact_dir_exists": local_artifact_dir.exists(),
        "readme_exists": readme_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "eval_summary_exists": eval_summary_path.exists(),
        "schedule_comparison_exists": comparison_path.exists(),
        "eval_plan_exists": eval_plan_path.exists(),
        "eval_report_exists": eval_report_path.exists(),
        "preflight_report_exists": preflight_report_path.exists(),
        "manifest_repo_id_matches": (repo_id is None) or (manifest.get("repo_id") == repo_id),
        **artifact_inspection.get("checks", {}),
    }
    missing_required = [
        name for name in [
            "artifact_dir_exists",
            "readme_exists",
            "manifest_exists",
            "manifest_repo_id_matches",
            "model_exists",
            "config_exists",
        ]
        if not checks.get(name)
    ]
    warnings = []
    if not checks.get("tokenizer_assets_present"):
        warnings.append("Tokenizer assets were not found in the artifact directory.")
    elif not checks.get("tokenizer_loadable_bundle_present"):
        warnings.append("Some tokenizer files exist, but the bundle may be incomplete for AutoTokenizer loading.")
    if not checks.get("special_tokens_map_present"):
        warnings.append("special_tokens_map.json is missing; upload may still work, but tokenizer behavior can be less explicit.")

    return {
        "repo_id": repo_id or manifest.get("repo_id"),
        "local_artifact_dir": str(local_artifact_dir),
        "checks": checks,
        "missing_required": missing_required,
        "warnings": warnings,
        "ready_for_upload": bool(not missing_required),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "eval_summary_path": str(eval_summary_path) if eval_summary_path.exists() else None,
        "comparison_summary_path": str(comparison_path) if comparison_path.exists() else None,
        "eval_plan_path": str(eval_plan_path) if eval_plan_path.exists() else None,
        "eval_report_path": str(eval_report_path) if eval_report_path.exists() else None,
        "preflight_report_path": str(preflight_report_path) if preflight_report_path.exists() else None,
        "artifact_inspection": artifact_inspection,
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
    eval_report_path = write_eval_summary_report(
        local_artifact_dir,
        eval_summary=eval_summary,
        comparison_summary=comparison_summary,
    )
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
        "eval_report_path": eval_report_path,
        "preflight_report_path": None,
        "has_eval_summary": eval_summary is not None,
        "has_comparison_summary": comparison_summary is not None,
        "has_eval_plan": eval_plan is not None,
        "has_eval_report": eval_report_path is not None,
        "eval_snapshot": summarize_eval_for_hf(eval_summary),
        "comparison_snapshot": summarize_comparison_for_hf(comparison_summary),
    }
    manifest_path = local_artifact_dir / "hf_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    validation = validate_hf_export_bundle(local_artifact_dir, repo_id=repo_id)
    preflight_report_path = write_hf_preflight_report(local_artifact_dir, validation=validation)
    manifest["preflight_report_path"] = preflight_report_path
    manifest["has_preflight_report"] = preflight_report_path is not None
    manifest["validation"] = validation
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
    export_snapshot_block = ""
    if eval_summary:
        eval_rows = build_eval_view_rows(eval_summary)
        metrics_lines = [
            "\n## Evaluation summary\n",
            "| view | aggregation | estimand | sample_axis | weighting | avg_cross_entropy | pseudo_perplexity | uniform_random_pseudo_perplexity | bits_per_masked_token | bits_saved_vs_uniform | bits_saved_ci_p05 | bits_saved_ci_p95 | denoising_skill | denoising_skill_ci_p05 | denoising_skill_ci_p95 | masked_token_accuracy | pseudo_perplexity_ci_p05 | pseudo_perplexity_ci_p95 | accuracy_ci_p05 | accuracy_ci_p95 |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in eval_rows:
            metrics_lines.append(
                f"| {row['view']} | {row['aggregation']} | {row.get('estimand')} | {row.get('sample_axis')} | {row.get('weighting')} | {row['avg_cross_entropy']} | {row['pseudo_perplexity']} | {row['uniform_random_pseudo_perplexity']} | {row['bits_per_masked_token']} | {row['bits_saved_vs_uniform']} | {row.get('bits_saved_vs_uniform_ci_p05')} | {row.get('bits_saved_vs_uniform_ci_p95')} | {row['denoising_skill']} | {row.get('denoising_skill_ci_p05')} | {row.get('denoising_skill_ci_p95')} | {row['masked_token_accuracy']} | {row['pseudo_perplexity_ci_p05']} | {row['pseudo_perplexity_ci_p95']} | {row['masked_token_accuracy_ci_p05']} | {row['masked_token_accuracy_ci_p95']} |"
            )
        protocol_rows = build_eval_protocol_rows(eval_summary)
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
                f"- schedule_reweighted_exact_eligible_token_count: {eval_summary.get('schedule_reweighted_exact_eligible_token_count')}",
                f"- schedule_reweighted_expected_masked_token_count: {eval_summary.get('schedule_reweighted_expected_masked_token_count')}",
                f"- schedule_reweighted_effective_sample_size: {eval_summary.get('schedule_reweighted_effective_sample_size')}",
                f"- schedule_reweighted_effective_sample_size_fraction: {eval_summary.get('schedule_reweighted_effective_sample_size_fraction')}",
                f"- timestep_macro_timestep_count: {eval_summary.get('timestep_macro_timestep_count')}",
                f"- timestep_auc_timestep_count: {eval_summary.get('timestep_auc_timestep_count')}",
                f"- timestep_auc_fraction_span: {eval_summary.get('timestep_auc_fraction_span')}",
            ]
        )
        if protocol_rows:
            metrics_lines.extend([
                "",
                "### Metric estimands and comparison semantics",
                "",
                "| view | estimand | sample_axis | weighting | comparison_semantics | recommended_primary |",
                "| --- | --- | --- | --- | --- | --- |",
            ])
            for row in protocol_rows:
                metrics_lines.append(
                    f"| {row['view']} | {row['estimand']} | {row['sample_axis']} | {row['weighting']} | {row['comparison_semantics']} | {row['is_recommended_primary_view']} |"
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
                f"- schedule_reweighted_ht_aggregation: {protocol.get('schedule_reweighted_ht_aggregation')}\n"
                f"- grid_uniform_aggregation: {protocol.get('grid_uniform_aggregation')}\n"
                f"- timestep_macro_aggregation: {protocol.get('timestep_macro_aggregation')}\n"
                f"- timestep_auc_aggregation: {protocol.get('timestep_auc_aggregation')}\n"
                f"- bootstrap_samples: {protocol.get('bootstrap_samples')}\n"
            )
        quality_summary = eval_summary.get("quality_summary") or {}
        if quality_summary:
            recommended_primary_view = quality_summary.get("recommended_primary_view") or {}
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
            if recommended_primary_view:
                quality_lines.extend(
                    [
                        "- recommended_primary_view:",
                        f"  - view: {recommended_primary_view.get('view')}",
                        f"  - metric_key: {recommended_primary_view.get('metric_key')}",
                        f"  - better_direction: {recommended_primary_view.get('better_direction')}",
                        f"  - rationale: {recommended_primary_view.get('rationale')}",
                        f"  - caveat: {recommended_primary_view.get('caveat')}",
                    ]
                )
            notes = quality_summary.get("notes") or []
            warnings = quality_summary.get("warnings") or []
            if notes:
                quality_lines.append("- notes:")
                quality_lines.extend([f"  - {note}" for note in notes])
            if warnings:
                quality_lines.append("- warnings:")
                quality_lines.extend([f"  - {warning}" for warning in warnings])
            quality_block = "\n".join(quality_lines) + "\n"

    eval_snapshot = summarize_eval_for_hf(eval_summary)
    comparison_snapshot = summarize_comparison_for_hf(comparison_summary)
    if eval_snapshot or comparison_snapshot:
        snapshot_lines = ["\n## Export bundle snapshot\n"]
        if eval_snapshot:
            snapshot_lines.extend(
                [
                    "- eval_snapshot:",
                    f"  - primary_view: {eval_snapshot.get('primary_view')}",
                    f"  - primary_metric_key: {eval_snapshot.get('primary_metric_key')}",
                    f"  - primary_metric_value: {eval_snapshot.get('primary_metric_value')}",
                    f"  - primary_metric_ci: {eval_snapshot.get('primary_metric_ci')}",
                    f"  - primary_bits_saved_vs_uniform: {eval_snapshot.get('primary_bits_saved_vs_uniform')}",
                    f"  - primary_denoising_skill: {eval_snapshot.get('primary_denoising_skill')}",
                    f"  - schedule_reweighted_reliability: {eval_snapshot.get('schedule_reweighted_reliability')}",
                    f"  - normalized_timestep_remapping: {eval_snapshot.get('normalized_timestep_remapping')}",
                ]
            )
        if comparison_snapshot:
            snapshot_lines.extend(
                [
                    "- comparison_snapshot:",
                    f"  - headline: {comparison_snapshot.get('headline')}",
                    f"  - primary_metric: {comparison_snapshot.get('primary_metric')}",
                    f"  - primary_winner: {comparison_snapshot.get('primary_winner')}",
                    f"  - primary_winner_probability: {comparison_snapshot.get('primary_winner_probability')}",
                    f"  - primary_delta_linear_minus_cosine: {comparison_snapshot.get('primary_delta_linear_minus_cosine')}",
                    f"  - primary_delta_ci: {comparison_snapshot.get('primary_delta_ci')}",
                    f"  - primary_practically_tied: {comparison_snapshot.get('primary_practically_tied')}",
                    f"  - normalized_timestep_remapping: {comparison_snapshot.get('normalized_timestep_remapping')}",
                ]
            )
        export_snapshot_block = "\n".join(snapshot_lines) + "\n"

    comparison_rows = build_schedule_comparison_rows(comparison_summary)
    if comparison_rows:
        comparison_lines = [
            "\n## Schedule comparison\n",
            "| metric_view | better_direction | comparison_scope | cosine_value | linear_value | delta_linear_minus_cosine | winner | winner_probability | ci_excludes_zero | practically_tied | bootstrap_p05 | bootstrap_p95 | probability_linear_better |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
        for row in comparison_rows:
            comparison_lines.append(
                f"| {row['metric_view']} | {row['better_direction']} | {row.get('comparison_scope')} | {row['cosine_value']} | {row['linear_value']} | {row['delta_linear_minus_cosine']} | {row['winner']} | {row.get('winner_probability')} | {row.get('ci_excludes_zero')} | {row.get('practically_tied')} | {row['bootstrap_p05']} | {row['bootstrap_p95']} | {row['probability_linear_better']} |"
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
        comparison_protocol_rows = build_comparison_protocol_rows(comparison_summary)
        if comparison_protocol_rows:
            comparison_lines.extend([
                "",
                "### Comparison semantics",
                "",
                "| metric_view | comparison_scope | recommended_primary | paired_batches | paired_uniform_noise | normalized_timestep_remapping |",
                "| --- | --- | --- | --- | --- | --- |",
            ])
            for row in comparison_protocol_rows:
                comparison_lines.append(
                    f"| {row['metric_view']} | {row['comparison_scope']} | {row['is_recommended_primary_metric']} | {row['paired_batches']} | {row['paired_uniform_noise']} | {row['normalized_timestep_remapping']} |"
                )
        comparison_block = "\n".join(comparison_lines) + "\n"
        decision_summary = comparison_summary.get("decision_summary") or {}
        if decision_summary:
            recommended_primary_metric = decision_summary.get('recommended_primary_metric') or {}
            decision_lines = [
                "\n## Schedule decision summary\n",
                f"- headline: {decision_summary.get('headline')}",
                f"- tracked_metric_count: {decision_summary.get('tracked_metric_count')}",
                f"- decisive_metric_count: {decision_summary.get('decisive_metric_count')}",
                f"- practically_tied_metric_count: {decision_summary.get('practically_tied_metric_count')}",
                f"- cosine_schedule_win_count: {decision_summary.get('cosine_schedule_win_count')}",
                f"- linear_schedule_win_count: {decision_summary.get('linear_schedule_win_count')}",
            ]
            if recommended_primary_metric:
                decision_lines.extend(
                    [
                        "- recommended_primary_metric:",
                        f"  - metric: {recommended_primary_metric.get('metric')}",
                        f"  - view: {recommended_primary_metric.get('view')}",
                        f"  - winner: {recommended_primary_metric.get('winner')}",
                        f"  - winner_probability: {recommended_primary_metric.get('winner_probability')}",
                        f"  - ci_excludes_zero: {recommended_primary_metric.get('ci_excludes_zero')}",
                        f"  - practically_tied: {recommended_primary_metric.get('practically_tied')}",
                        f"  - rationale: {recommended_primary_metric.get('rationale')}",
                    ]
                )
            tracked = decision_summary.get('tracked_metrics') or []
            if tracked:
                decision_lines.extend([
                    "",
                    "| metric | winner | better_direction | winner_probability | ci_excludes_zero | practically_tied | recommended_primary |",
                    "| --- | --- | --- | ---: | --- | --- | --- |",
                ])
                for row in tracked:
                    decision_lines.append(
                        f"| {row.get('metric')} | {row.get('winner')} | {row.get('better_direction')} | {row.get('winner_probability')} | {row.get('ci_excludes_zero')} | {row.get('practically_tied')} | {row.get('is_recommended_primary_metric')} |"
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
- optional `eval_summary_report.txt` plain-text notebook/HF evaluation summary
- optional `hf_preflight_report.txt` plain-text Hugging Face upload preflight summary
- optional `hf_export_manifest.json` bundle manifest covering all exported metadata files
- optional schedule-comparison JSON artifacts
- optional bundle validation metadata inside `hf_export_manifest.json`
{metrics_block}{protocol_block}{quality_block}{export_snapshot_block}{comparison_block}{decision_block}## Evaluation note

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
