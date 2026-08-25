#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

if [[ $# -ne 1 ]]; then
  printf 'usage: %s {pythia_410m|olmo_1b_hf}\n' "$0" >&2
  exit 2
fi

model=$1
case "$model" in
  pythia_410m | olmo_1b_hf) ;;
  *)
    printf 'unsupported model: %s\n' "$model" >&2
    exit 2
    ;;
esac

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-96}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-96}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
CONFIG_NAME=squad_lora_rmia_more_epochs
OUTPUT_ROOT=outputs/squad_lora_rmia_more_epochs

run_stage() {
  local stage=$1
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

for stage in \
  prepare \
  train_target \
  evaluate \
  generate_base \
  generate \
  build_candidates \
  train_shadows; do
  run_stage "$stage"
done

model_root="$OUTPUT_ROOT/formal/$model"
target_root="$model_root/target/seed_42"
test -f "$target_root/epoch_validation_metrics.json"
test -f "$target_root/adapter/adapter_config.json"
test -f "$target_root/adapter/adapter_model.safetensors"
test -f "$target_root/experiment.md"
for index in 00 01 02 03 04; do
  shadow_root="$model_root/shadows/shadow_$index"
  test -f "$shadow_root/adapter/adapter_config.json"
  test -f "$shadow_root/adapter/adapter_model.safetensors"
  test -f "$shadow_root/experiment.md"
done
printf '[%s] model=%s target and five shadows complete\n' \
  "$(date --iso-8601=seconds)" "$model"
