import json
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import create_repo, login, upload_folder


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
        metrics_block = (
            "\n## Evaluation summary\n\n"
            "| view | avg_cross_entropy | pseudo_perplexity | bits_per_masked_token | masked_token_accuracy |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
            f"| token_weighted_sampled | {eval_summary.get('avg_cross_entropy')} | {eval_summary.get('pseudo_perplexity')} | {eval_summary.get('bits_per_masked_token')} | {eval_summary.get('masked_token_accuracy')} |\n"
            f"| timestep_uniform_sampled | {eval_summary.get('timestep_uniform_avg_cross_entropy')} | {eval_summary.get('timestep_uniform_pseudo_perplexity')} | {eval_summary.get('timestep_uniform_bits_per_masked_token')} | n/a |\n"
            f"| fixed_grid_batch_uniform | {eval_summary.get('grid_uniform_avg_cross_entropy')} | {eval_summary.get('grid_uniform_pseudo_perplexity')} | {eval_summary.get('grid_uniform_bits_per_masked_token')} | {eval_summary.get('grid_uniform_masked_token_accuracy')} |\n"
            f"| fixed_grid_timestep_macro | {eval_summary.get('timestep_macro_avg_cross_entropy')} | {eval_summary.get('timestep_macro_pseudo_perplexity')} | {eval_summary.get('timestep_macro_bits_per_masked_token')} | {eval_summary.get('timestep_macro_masked_token_accuracy')} |\n\n"
            f"- metric: {eval_summary.get('metric', 'diffusion_pseudo_perplexity')}\n"
            f"- sampled_example_count: {eval_summary.get('sampled_example_count')}\n"
            f"- masked_tokens: {eval_summary.get('masked_tokens')}\n"
            f"- n_batches: {eval_summary.get('n_batches')}\n"
            f"- timestep_macro_timestep_count: {eval_summary.get('timestep_macro_timestep_count')}\n"
        )
        ci = eval_summary.get("confidence_intervals") or {}
        ppx_ci = ci.get("pseudo_perplexity") or {}
        if ppx_ci:
            metrics_block += (
                f"- pseudo_perplexity_bootstrap_p05_p95: [{ppx_ci.get('p05')}, {ppx_ci.get('p95')}]\n"
            )
        grid_ci = eval_summary.get("grid_uniform_confidence_intervals") or {}
        grid_ppx_ci = grid_ci.get("grid_uniform_pseudo_perplexity") or {}
        if grid_ppx_ci:
            metrics_block += (
                f"- grid_uniform_pseudo_perplexity_bootstrap_p05_p95: [{grid_ppx_ci.get('p05')}, {grid_ppx_ci.get('p95')}]\n"
            )
        protocol = eval_summary.get("eval_protocol") or {}
        if protocol:
            protocol_block = (
                "\n## Evaluation protocol\n\n"
                f"- schedule_name: {protocol.get('schedule_name')}\n"
                f"- diffusion_steps: {protocol.get('T')}\n"
                f"- timestep_grid: {protocol.get('timestep_grid')}\n"
                f"- paired_noise: {protocol.get('paired_noise')}\n"
                f"- paired_batches: {protocol.get('paired_batches')}\n"
                f"- sampled_timestep_distribution: {protocol.get('sampled_timestep_distribution')}\n"
                f"- grid_uniform_aggregation: {protocol.get('grid_uniform_aggregation')}\n"
                f"- timestep_macro_aggregation: {protocol.get('timestep_macro_aggregation')}\n"
                f"- bootstrap_samples: {protocol.get('bootstrap_samples')}\n"
            )

    if comparison_summary and len(comparison_summary.get("models", [])) >= 2:
        delta = comparison_summary.get("delta") or {}
        winner = comparison_summary.get("winner") or {}
        delta_ci = ((comparison_summary.get("delta_confidence_intervals") or {}).get("delta_linear_minus_cosine") or {})
        ppx_ci = delta_ci.get("pseudo_perplexity") or {}
        uniform_ppx_ci = delta_ci.get("timestep_uniform_pseudo_perplexity") or {}
        grid_ppx_ci = delta_ci.get("grid_uniform_pseudo_perplexity") or {}
        acc_ci = delta_ci.get("masked_token_accuracy") or {}
        grid_acc_ci = delta_ci.get("grid_uniform_masked_token_accuracy") or {}
        comparison_block = (
            "\n## Schedule comparison\n\n"
            "| metric_view | delta_linear_minus_cosine | winner | extra |\n"
            "| --- | ---: | --- | --- |\n"
            f"| sampled_pseudo_perplexity | {delta.get('pseudo_perplexity')} | {winner.get('pseudo_perplexity')} | bootstrap_p05_p95=[{ppx_ci.get('p05')}, {ppx_ci.get('p95')}], p_linear_better={ppx_ci.get('probability_linear_better')} |\n"
            f"| timestep_uniform_pseudo_perplexity | {delta.get('timestep_uniform_pseudo_perplexity')} | {winner.get('timestep_uniform_pseudo_perplexity')} | bootstrap_p05_p95=[{uniform_ppx_ci.get('p05')}, {uniform_ppx_ci.get('p95')}], p_linear_better={uniform_ppx_ci.get('probability_linear_better')} |\n"
            f"| grid_uniform_pseudo_perplexity | {delta.get('grid_uniform_pseudo_perplexity')} | {winner.get('grid_uniform_pseudo_perplexity')} | bootstrap_p05_p95=[{grid_ppx_ci.get('p05')}, {grid_ppx_ci.get('p95')}], p_linear_better={grid_ppx_ci.get('probability_linear_better')} |\n"
            f"| timestep_macro_pseudo_perplexity | {delta.get('timestep_macro_pseudo_perplexity')} | {winner.get('timestep_macro_pseudo_perplexity')} | equal weight per diagnostic timestep |\n"
            f"| sampled_accuracy | {delta.get('masked_token_accuracy')} | {winner.get('masked_token_accuracy')} | bootstrap_p05_p95=[{acc_ci.get('p05')}, {acc_ci.get('p95')}], p_linear_better={acc_ci.get('probability_linear_better')} |\n"
            f"| grid_uniform_accuracy | {delta.get('grid_uniform_masked_token_accuracy')} | {winner.get('grid_uniform_masked_token_accuracy')} | bootstrap_p05_p95=[{grid_acc_ci.get('p05')}, {grid_acc_ci.get('p95')}], p_linear_better={grid_acc_ci.get('probability_linear_better')} |\n"
            f"| timestep_macro_accuracy | {delta.get('timestep_macro_masked_token_accuracy')} | {winner.get('timestep_macro_masked_token_accuracy')} | equal weight per diagnostic timestep |\n\n"
            f"- avg_cross_entropy_delta_linear_minus_cosine: {delta.get('avg_cross_entropy')}\n"
            f"- timestep_uniform_avg_cross_entropy_delta_linear_minus_cosine: {delta.get('timestep_uniform_avg_cross_entropy')}\n"
            f"- grid_uniform_avg_cross_entropy_delta_linear_minus_cosine: {delta.get('grid_uniform_avg_cross_entropy')}\n"
            f"- timestep_macro_avg_cross_entropy_delta_linear_minus_cosine: {delta.get('timestep_macro_avg_cross_entropy')}\n"
            f"- timestep_macro_bits_per_masked_token_delta_linear_minus_cosine: {delta.get('timestep_macro_bits_per_masked_token')}\n"
        )

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
{metrics_block}{protocol_block}{comparison_block}
## Evaluation note

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
    login(token=hf_token)
    create_repo(repo_id=hf_repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(
        repo_id=hf_repo_id,
        folder_path=local_artifact_dir,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{hf_repo_id}"
