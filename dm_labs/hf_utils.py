import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, login, upload_folder


def ensure_hf_model_card(local_artifact_dir: str, repo_id: str, eval_summary: Optional[dict] = None) -> str:
    local_artifact_dir = Path(local_artifact_dir)
    readme_path = local_artifact_dir / "README.md"
    if readme_path.exists():
        return str(readme_path)

    metrics_block = ""
    if eval_summary:
        metrics_block = (
            "\n## Evaluation summary\n\n"
            f"- metric: {eval_summary.get('metric', 'diffusion_pseudo_perplexity')}\n"
            f"- avg_cross_entropy: {eval_summary.get('avg_cross_entropy')}\n"
            f"- pseudo_perplexity: {eval_summary.get('pseudo_perplexity')}\n"
            f"- bits_per_masked_token: {eval_summary.get('bits_per_masked_token')}\n"
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
- optional evaluation JSON artifacts
{metrics_block}
## Evaluation note

The reported pseudo-perplexity is based on masked-token denoising NLL under a diffusion corruption process. It is **not** autoregressive next-token perplexity.
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

    ensure_hf_model_card(local_artifact_dir, hf_repo_id, eval_summary=eval_summary)
    login(token=hf_token)
    HfApi()
    create_repo(repo_id=hf_repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(
        repo_id=hf_repo_id,
        folder_path=local_artifact_dir,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{hf_repo_id}"
