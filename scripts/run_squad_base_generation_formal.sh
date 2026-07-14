#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-96}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-96}"
TOKENIZATION_WORKERS="${TOKENIZATION_WORKERS:-8}"

for model in pythia_410m olmo_1b_hf; do
  uv run python ./cli/squad_lora_rmia.py \
    runtime.seed=42 \
    workflow.stage=generate_base \
    workflow.profile=formal \
    llm_model="${model}" \
    tokenizer.preprocessing_workers="${TOKENIZATION_WORKERS}" \
    eval.batch_size="${INFERENCE_BATCH_SIZE}" \
    generation.batch_size="${GENERATION_BATCH_SIZE}"
done
