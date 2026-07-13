#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-96}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-96}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

for model in pythia_410m; do
  uv run python ./cli/squad_lora_rmia.py \
    runtime.seed=42 \
    workflow.stage=formal \
    workflow.profile=formal \
    llm_model="${model}" \
    train.per_device_train_batch_size="${TRAIN_BATCH_SIZE}" \
    train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    train.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    eval.batch_size="${INFERENCE_BATCH_SIZE}" \
    eval.generation_batch_size="${GENERATION_BATCH_SIZE}" \
    generation.batch_size="${GENERATION_BATCH_SIZE}" \
    attack.batch_size="${INFERENCE_BATCH_SIZE}"
done
