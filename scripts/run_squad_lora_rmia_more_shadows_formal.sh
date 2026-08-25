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
CONFIG_NAME=squad_lora_rmia_more_shadows
OUTPUT_ROOT=outputs/squad_lora_rmia_more_shadows

run_stage() {
  local model=$1
  local stage=$2
  printf '[%s] model=%s stage=%s start\n' "$(date --iso-8601=seconds)" "$model" "$stage"
  uv run python ./cli/squad_lora_rmia.py \
    --config-name="$CONFIG_NAME" \
    runtime.seed=42 \
    workflow.stage="$stage" \
    workflow.profile=formal \
    llm_model="$model" \
    train.per_device_train_batch_size="$TRAIN_BATCH_SIZE" \
    train.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    train.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    eval.batch_size="$INFERENCE_BATCH_SIZE" \
    eval.generation_batch_size="$GENERATION_BATCH_SIZE" \
    generation.batch_size="$GENERATION_BATCH_SIZE" \
    attack.batch_size="$INFERENCE_BATCH_SIZE"
  printf '[%s] model=%s stage=%s complete\n' "$(date --iso-8601=seconds)" "$model" "$stage"
}

for model in pythia_410m olmo_1b_hf; do
  run_stage "$model" prepare_shadow_reuse
  run_stage "$model" build_candidates
  run_stage "$model" train_shadows
  run_stage "$model" attack
  run_stage "$model" validate
done

formal_root="$OUTPUT_ROOT/formal"
test -f "$formal_root/pythia_410m/_SUCCESS"
test -f "$formal_root/olmo_1b_hf/_SUCCESS"
sha256sum \
  "$formal_root/pythia_410m/completion_manifest.json" \
  "$formal_root/olmo_1b_hf/completion_manifest.json" \
  > "$formal_root/completion_manifest.sha256"
printf 'verified\n' > "$formal_root/_SUCCESS"
printf '[%s] formal workflow complete\n' "$(date --iso-8601=seconds)"
