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

  common_overrides=(
    runtime.seed=42
    workflow.profile=formal
    llm_model="$model"
    train.per_device_train_batch_size="$train_batch_size"
    train.gradient_accumulation_steps="$gradient_accumulation_steps"
    train.dataloader_num_workers=4
    eval.batch_size=96
    eval.generation_batch_size=96
    generation.batch_size=96
    attack.batch_size=96
  )

  uv run python ./cli/squad_fullft_rmia.py \
    "${common_overrides[@]}" workflow.stage=formal
  uv run python ./cli/squad_fullft_rmia.py \
    "${common_overrides[@]}" workflow.stage=generate_base
  uv run python ./cli/squad_fullft_rmia.py \
    "${common_overrides[@]}" workflow.stage=plot_rmia_feature
done
