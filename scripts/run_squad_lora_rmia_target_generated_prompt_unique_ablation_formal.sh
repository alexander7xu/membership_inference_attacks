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
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
CONFIG_NAME=squad_lora_rmia_target_generated_prompt_unique_ablation
OUTPUT_ROOT=outputs/squad_lora_rmia_target_generated_prompt_unique_ablation

run_stage() {
  local stage=$1
  if [[ "$stage" == "attack" ]]; then
    export LLM_MIA_ATTACK_BATCH_SIZE="$ATTACK_BATCH_SIZE"
  else
    unset LLM_MIA_ATTACK_BATCH_SIZE
  fi
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
    eval.generation_batch_size="$INFERENCE_BATCH_SIZE" \
    generation.batch_size="$INFERENCE_BATCH_SIZE"
  printf '[%s] model=%s stage=%s complete\n' "$(date --iso-8601=seconds)" "$model" "$stage"
}

for stage in \
  prepare_shadow_reuse \
  generate \
  build_candidates \
  train_shadows \
  attack \
  compare_gold_iid \
  plot_rmia_feature \
  validate; do
  run_stage "$stage"
done

model_root="$OUTPUT_ROOT/formal/$model"
test -f "$model_root/_SUCCESS"
test -f "$model_root/completion_manifest.json"
test -f "$model_root/attack/online_rmia_scores.jsonl"
test -f "$model_root/control/gold_iid/comparison.json"
test -f "$model_root/analysis/relative_log_likelihood/relative_log_likelihood_ecdf.png"
printf '[%s] model=%s target-generated shadow-distribution online RMIA complete\n' \
  "$(date --iso-8601=seconds)" "$model"
