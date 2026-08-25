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

ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-96}"
printf '[%s] model=%s stage=plot_feature_matrix start\n' \
  "$(date --iso-8601=seconds)" "$model"
uv run python ./cli/squad_lora_rmia.py \
  --config-name=squad_lora_mia_feature_matrix \
  runtime.seed=42 \
  workflow.stage=plot_feature_matrix \
  workflow.profile=formal \
  llm_model="$model" \
  attack.batch_size="$ATTACK_BATCH_SIZE"
printf '[%s] model=%s stage=plot_feature_matrix complete\n' \
  "$(date --iso-8601=seconds)" "$model"

model_root="outputs/squad_lora_mia_feature_matrix/formal/$model"
test -f "$model_root/_SUCCESS"
test -f "$model_root/completion_manifest.json"
for setting in one_epoch_rmia one_epoch_lira ten_epoch_rmia \
  one_epoch_rmia_gold_iid one_epoch_rmia_target_generated; do
  test -f "$model_root/settings/$setting/population_centered_3x3_ecdf.png"
  test -f "$model_root/settings/$setting/manifest.json"
done
printf '[%s] model=%s population-centered feature matrix complete\n' \
  "$(date --iso-8601=seconds)" "$model"
