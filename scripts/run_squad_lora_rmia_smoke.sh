#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

uv run python ./cli/squad_lora_rmia.py \
  runtime.seed=42 \
  workflow.stage=smoke \
  workflow.profile=smoke \
  llm_model=pythia_410m \
  data.smoke_candidate_limit=4 \
  data.smoke_eval_limit=4 \
  train.smoke_max_steps=1 \
  shadow.smoke_count=1
