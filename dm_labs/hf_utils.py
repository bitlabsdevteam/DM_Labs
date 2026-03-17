import json
import os
from pathlib import Path
from typing import Optional


EVAL_VIEW_SPECS = [
    {
        "view": "token_weighted_sampled",
        "avg_cross_entropy_key": "avg_cross_entropy",
        "pseudo_perplexity_key": "pseudo_perplexity",
        "bits_key": "bits_per_masked_token",
        "accuracy_key": "masked_token_accuracy",
        "ci_container_key": "confidence_intervals",
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
        "ci_container_key": None,
        "ci_metric_key": None,
        "ci_accuracy_key": None,
        "aggregation": "uniform mean over sampled per-example timesteps",
    },
    {
        "view": "schedule_reweighted_sampled",
        "avg_cross_entropy_key": "schedule_reweighted_avg_cross_entropy",
        "pseudo_perplexity_key": "schedule_reweighted_pseudo_perplexity",
        "bits_key": "schedule_reweighted_bits_per_masked_token",
        "accuracy_key": "schedule_reweighted_masked_token_accuracy",
        "ci_container_key": "schedule_reweighted_confidence_intervals",
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
    },
    {
        "metric_view": "timestep_uniform_pseudo_perplexity",
        "delta_key": "timestep_uniform_pseudo_perplexity",
        "winner_key": "timestep_uniform_pseudo_perplexity",
        "ci_key": "timestep_uniform_pseudo_perplexity",
        "better_direction": "lower",
    },
    {
        "metric_view": "schedule_reweighted_pseudo_perplexity",
        "delta_key": "schedule_reweighted_pseudo_perplexity",
        "winner_key": "schedule_reweighted_pseudo_perplexity",
        "ci_key": "schedule_reweighted_pseudo_perplexity",
        "better_direction": "lower",
    },
    {
        "metric_view": "grid_uniform_pseudo_perplexity",
        "delta_key": "grid_uniform_pseudo_perplexity",
        "winner_key": "grid_uniform_pseudo_perplexity",
        "ci_key": "grid_uniform_pseudo_perplexity",
        "better_direction": "lower",
    },
    {
        "metric_view": "timestep_macro_pseudo_perplexity",
        "delta_key": "timestep_macro_pseudo_perplexity",
        "winner_key": "timestep_macro_pseudo_perplexity",
        "ci_key": "timestep_macro_pseudo_perplexity",
        "better_direction": "lower",
    },
    {
        "metric_view": "timestep_auc_pseudo_perplexity",
        "delta_key": "timestep_auc_pseudo_perplexity",
        "winner_key": "timestep_auc_pseudo_perplexity",
        "ci_key": "timestep_auc_pseudo_perplexity",
        "better_direction": "lower",
    },
    {
        "metric_view": "sampled_accuracy",
        "delta_key": "masked_token_accuracy",
        "winner_key": "masked_token_accuracy",
        "ci_key": "masked_token_accuracy",
        "better_direction": "higher",
    },
    {
        "metric_view": "timestep_uniform_accuracy",
        "delta_key": "timestep_uniform_masked_token_accuracy",
        "winner_key": "timestep_uniform_masked_token_accuracy",
        "ci_key": "timestep_uniform_masked_token_accuracy",
        "better_direction": "higher",
    },
    {
        "metric_view": "schedule_reweighted_accuracy",
        "delta_key": "schedule_reweighted_masked_token_accuracy",
        "winner_key": "schedule_reweighted_masked_token_accuracy",
        "ci_key": "schedule_reweighted_masked_token_accuracy",
        "better_direction": "higher",
    },
    {
        "metric_view": "grid_uniform_accuracy",
        "delta_key": "grid_uniform_masked_token_accuracy",
        "winner_key": "grid_uniform_masked_token_accuracy",
        "ci_key": "grid_uniform_masked_token_accuracy",
        "better_direction": "higher",
    },
    {
        "metric_view": "timestep_macro_accuracy",
        "delta_key": "timestep_macro_masked_token_accuracy",
        "winner_key": "timestep_macro_masked_token_accuracy",
        "ci_key": "timestep_macro_masked_token_accuracy",
        "better_direction": "higher",
    },
    {
        "metric_view": "timestep_auc_accuracy",
        "delta_key": "timestep_auc_masked_token_accuracy",
        "winner_key": "timestep_auc_masked_token_accuracy",
        "ci_key": "timestep_auc_masked_token_accuracy",
        "better_direction": "higher",
    },
]


def _extract_ci(summary: Optional[dict], container_key: Optional[str], metric_key: Optional[str]) -> dict:
    if not summary or not container_key or not metric_key:
        return {}
    return (summary.get(container_key) or {}).get(metric_key) or {}


def build_eval_view_rows(eval_summary: Optional[dict] = None) -> list:
    if not eval_summary:
        return []

    rows = []
    for spec in EVAL_VIEW_SPECS:
        metric_ci = _extract_ci(eval_summary, spec["ci_container_key"], spec["ci_metric_key"])
        accuracy_ci = _extract_ci(eval_summary, spec["ci_container_key"], spec["ci_accuracy_key"])
        rows.append(
            {
                "view": spec["view"],
                "aggregation": spec["aggregation"],
                "avg_cross_entropy": eval_summary.get(spec["avg_cross_entropy_key"]),
                "pseudo_perplexity": eval_summary.get(spec["pseudo_perplexity_key"]),
                "bits_per_masked_token": eval_summary.get(spec["bits_key"]),
                "masked_token_accuracy": eval_summary.get(spec["accuracy_key"]),
                "pseudo_perplexity_ci_p05": metric_ci.get("p05"),
                "pseudo_perplexity_ci_p95": metric_ci.get("p95"),
                "masked_token_accuracy_ci_p05": accuracy_ci.get("p05"),
                "masked_token_accuracy_ci_p95": accuracy_ci.get("p95"),
            }
        )
    return rows


def build_schedule_comparison_rows(comparison_summary: Optional[dict] = None) -> list:
    if not comparison_summary or len(comparison_summary.get("models", [])) < 2:
        return []

    delta = comparison_summary.get("delta") or {}
    winner = comparison_summary.get("winner") or {}
    delta_ci = ((comparison_summary.get("delta_confidence_intervals") or {}).get("delta_linear_minus_cosine") or {})
    rows = []
    for spec in COMPARISON_VIEW_SPECS:
        ci = delta_ci.get(spec["ci_key"]) or {}
        rows.append(
            {
                "metric_view": spec["metric_view"],
                "better_direction": spec["better_direction"],
                "delta_linear_minus_cosine": delta.get(spec["delta_key"]),
                "winner": winner.get(spec["winner_key"]),
                "bootstrap_p05": ci.get("p05"),
                "bootstrap_p95": ci.get("p95"),
                "probability_linear_better": ci.get("probability_linear_better"),
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
    comparison_block = ""
    if eval_summary:
        eval_rows = build_eval_view_rows(eval_summary)
        metrics_lines = [
            "\n## Evaluation summary\n",
            "| view | aggregation | avg_cross_entropy | pseudo_perplexity | bits_per_masked_token | masked_token_accuracy | pseudo_perplexity_ci_p05 | pseudo_perplexity_ci_p95 | accuracy_ci_p05 | accuracy_ci_p95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in eval_rows:
            metrics_lines.append(
                f"| {row['view']} | {row['aggregation']} | {row['avg_cross_entropy']} | {row['pseudo_perplexity']} | {row['bits_per_masked_token']} | {row['masked_token_accuracy']} | {row['pseudo_perplexity_ci_p05']} | {row['pseudo_perplexity_ci_p95']} | {row['masked_token_accuracy_ci_p05']} | {row['masked_token_accuracy_ci_p95']} |"
            )
        metrics_lines.extend(
            [
                "",
                f"- metric: {eval_summary.get('metric', 'diffusion_pseudo_perplexity')}",
                f"- sampled_example_count: {eval_summary.get('sampled_example_count')}",
                f"- masked_tokens: {eval_summary.get('masked_tokens')}",
                f"- n_batches: {eval_summary.get('n_batches')}",
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

    comparison_rows = build_schedule_comparison_rows(comparison_summary)
    if comparison_rows:
        comparison_lines = [
            "\n## Schedule comparison\n",
            "| metric_view | better_direction | delta_linear_minus_cosine | winner | bootstrap_p05 | bootstrap_p95 | probability_linear_better |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for row in comparison_rows:
            comparison_lines.append(
                f"| {row['metric_view']} | {row['better_direction']} | {row['delta_linear_minus_cosine']} | {row['winner']} | {row['bootstrap_p05']} | {row['bootstrap_p95']} | {row['probability_linear_better']} |"
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
- optional schedule-comparison JSON artifacts
{metrics_block}{protocol_block}{comparison_block}## Evaluation note

The reported pseudo-perplexity is based on masked-token denoising NLL under a diffusion corruption process. It is **not** autoregressive next-token perplexity.

For reproducibility, the notebook can export the exact evaluation summary used for this upload into `eval_summary.json` and the paired linear-vs-cosine summary into `schedule_comparison.json`.
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

    write_eval_summary(local_artifact_dir, eval_summary=eval_summary)
    write_schedule_comparison(local_artifact_dir, comparison_summary=comparison_summary)
    ensure_hf_model_card(
        local_artifact_dir,
        hf_repo_id,
        eval_summary=eval_summary,
        comparison_summary=comparison_summary,
        overwrite=overwrite_model_card,
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
