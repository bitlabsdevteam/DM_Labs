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
