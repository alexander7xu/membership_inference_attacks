#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

for model in pythia_410m olmo_1b_hf; do
  if [[ "$model" == "olmo_1b_hf" ]]; then
    train_batch_size=8
    gradient_accumulation_steps=4
  else
    train_batch_size=16
    gradient_accumulation_steps=2
  fi

  uv run python ./cli/squad_fullft_rmia.py \
    runtime.seed=42 \
    workflow.stage=smoke \
    workflow.profile=smoke \
    llm_model="$model" \
    train.per_device_train_batch_size="$train_batch_size" \
    train.gradient_accumulation_steps="$gradient_accumulation_steps" \
    train.smoke_max_steps=20 \
    data.smoke_candidate_limit=4 \
    data.smoke_eval_limit=4 \
    shadow.smoke_count=1 \
    eval.batch_size=4 \
    eval.generation_batch_size=4 \
    generation.batch_size=4 \
    attack.batch_size=4
done
