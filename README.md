# DM_Labs

Diffusion Language Model experiments, notebooks, and evaluation work.

## Current Contents

- `Custom_Final_Diffusion_LLM_from_scratch_TinyStories_Cosine_schedule.ipynb`
  - Colab notebook for training a diffusion language model from scratch on TinyStories
  - uses a cosine noising / masking schedule
  - includes notebook cells for:
    - Hugging Face deployment
    - diffusion pseudo-perplexity evaluation
    - linear-vs-cosine comparison
    - FairSteer-style evaluation artifact export
- `dm_labs/modeling.py`
  - extracted diffusion Transformer model/config code from the notebook
- `dm_labs/data_utils.py`
  - extracted token-block dataset + collation utilities
- `dm_labs/eval_utils.py`
  - reusable diffusion evaluation helpers
  - token-weighted pseudo-perplexity computation
  - vocabulary-calibrated uniform-random baselines so each denoising aggregate can also be read as bits saved vs random guessing / denoising skill
  - timestep-uniform denoising CE / pseudo-perplexity
  - mask-ratio-reweighted sampled denoising CE / pseudo-perplexity for schedule-corrected sampled evaluation
  - fixed-grid-uniform denoising CE / pseudo-perplexity over cached diagnostic timesteps
  - equal-weight-over-timestep macro denoising CE / pseudo-perplexity on the cached diagnostic grid
  - normalized timestep-fraction AUC denoising CE / pseudo-perplexity over the cached diagnostic grid
  - bootstrap confidence intervals over sampled eval batches
  - bootstrap confidence intervals for timestep-uniform sampled denoising aggregates over per-example records
  - bootstrap confidence intervals over cached batch-timestep grid records
  - masked-token accuracy + timestep-slice denoising diagnostics
  - cached shared-batch/shared-noise evaluation plans for reproducible comparisons
  - paired cosine-vs-linear checkpoint comparison using the same underlying corruption draws
  - paired bootstrap confidence intervals for linear-vs-cosine metric deltas on the shared eval plan
  - timestep-grid bootstrap confidence intervals for timestep-macro and timestep-AUC denoising aggregates in single-model eval artifacts
  - paired timestep-macro and timestep-AUC bootstrap intervals over the shared diagnostic grid, so equal-weight-over-timestep and trajectory-integral comparisons carry uncertainty too
  - per-timestep delta reporting for linear-vs-cosine comparisons
- `dm_labs/hf_utils.py`
  - reusable Hugging Face model-card + upload helpers
  - persists `eval_summary.json` and `schedule_comparison.json` alongside uploaded artifacts
  - surfaces timestep-uniform, schedule-reweighted, grid-uniform, timestep-macro, and timestep-AUC evaluation views in exported model cards

## Project Focus

This repo is for practical experimentation around:

- diffusion language models (DLMs)
- discrete masked denoising for text generation
- cosine vs linear noising schedules
- perplexity-style evaluation for denoising-based models
- implementation-oriented research and iteration

## Important Evaluation Note

This project distinguishes between:

- **autoregressive perplexity**
- **diffusion-compatible perplexity-style evaluation**

A diffusion language model is **not** evaluated the same way as an autoregressive language model.

So this repo uses a **diffusion pseudo-perplexity** / denoising-based evaluation approach where appropriate, rather than blindly copying standard next-token perplexity.

Current protocol improvements in the repo:
- aggregate masked-token NLL by **token count**, not by naive per-batch averaging
- also report a **timestep-uniform denoising CE / pseudo-perplexity** that averages per-example masked-token CE over uniformly sampled timesteps
- attach **bootstrap confidence intervals** for the timestep-uniform sampled view, so the schedule-agnostic sampled aggregate carries uncertainty too
- also report a **schedule-reweighted sampled denoising CE / pseudo-perplexity** that applies inverse expected mask-ratio weights, so sampled-batch evaluation better approximates a uniform-over-mask-eligible-token-and-timestep denoising objective
- also report a **grid-uniform denoising CE / pseudo-perplexity** that averages over a fixed cached timestep grid shared across schedule comparisons
- also report a **timestep-macro denoising CE / pseudo-perplexity** that gives equal top-level weight to each diagnostic timestep on the shared grid
- also report a **timestep-AUC denoising CE / pseudo-perplexity** that integrates over normalized timestep fraction, so irregular diagnostic grids do not over-weight densely sampled regions
- report **bits per masked token** alongside pseudo-perplexity
- report **masked-token accuracy** as a complementary denoising quality signal
- expose **bootstrap confidence intervals** over sampled evaluation batches
- expose **timestep-conditioned diagnostics** so schedule quality can be inspected across early/mid/late denoising
- compare cosine vs linear checkpoints under a **shared cached batch set, shared timestep grid, and shared underlying uniform noise draws**
- when checkpoints use different `diffusion_steps`, remap the shared plan by **normalized timestep fraction** so comparisons stay corruption-matched instead of integer-step-mismatched
- export **per-timestep linear-minus-cosine deltas**, including mask-fraction deltas, for tighter schedule analysis
- attach **paired bootstrap delta intervals** so schedule winners are reported with uncertainty, not just point estimates
- attach **timestep-grid bootstrap intervals** for both **timestep-macro** and **timestep-AUC** denoising summaries in single-model eval artifacts
- attach **paired timestep-macro and timestep-AUC bootstrap intervals** so equal-weight-over-timestep and trajectory-level schedule claims are uncertainty-aware too
- surface a **normalized timestep-fraction AUC** view so schedule comparisons can summarize the denoising trajectory without depending on equal timestep spacing
- expose a **fixed-grid shared-timestep aggregate** plus paired uncertainty so cosine-vs-linear claims can be checked on an explicit common denoising surface
- persist evaluation protocol metadata into exported JSON and Hugging Face upload artifacts
- calibrate denoising CE / pseudo-perplexity views against a same-tokenizer uniform-random baseline, exposing bits-saved and denoising-skill summaries in notebook/HF outputs
- surface when normalized timestep remapping was used in cross-checkpoint comparisons and Hugging Face artifacts

## Current Notebook Capabilities

The main notebook currently supports:

1. training a diffusion language model from scratch on a TinyStories slice
2. cosine-schedule corruption / denoising
3. sample generation
4. artifact packaging
5. Hugging Face Hub deployment cell
6. diffusion pseudo-perplexity evaluation cell
7. linear-noising baseline vs cosine comparison cell
8. FairSteer-style JSON artifact export for evaluation results

## Intended Workflow

1. iterate on notebook implementation
2. validate evaluation logic carefully for diffusion models
3. compare schedule variants fairly
4. push each meaningful implementation update to this repo

## Roadmap Direction

Near-term priorities include:

- improving diffusion-specific evaluation rigor
- clarifying the relationship between denoising loss and perplexity-style metrics
- comparing linear and cosine schedules under a shared protocol
- improving artifact export and reproducibility
- extending model deployment and benchmarking flow

## Repo Philosophy

This repository is implementation-first.

The goal is to produce:
- real working notebook/code changes
- technically honest evaluation logic
- clean experiment artifacts
- fast iteration with disciplined version control
