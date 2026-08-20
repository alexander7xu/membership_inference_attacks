#!/usr/bin/env bash
set -euo pipefail

source /home/c01xuju/CISPA-home/home/.envrc

cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
  printf 'usage: %s {pythia-shadows|olmo-full}\n' "$0" >&2
  exit 2
fi

mode=$1
case "$mode" in
  pythia-shadows)
    partition="xe8545"
    command='exec uv run python ./cli/squad_lora_rmia.py --config-name=squad_lora_rmia_more_epochs runtime.seed=42 workflow.stage=train_shadows workflow.profile=formal llm_model=pythia_410m train.per_device_train_batch_size=16 train.gradient_accumulation_steps=2 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96'
    ;;
  olmo-full)
    partition="tmp"
    command='exec bash ./scripts/run_squad_lora_rmia_more_epochs_formal.sh olmo_1b_hf'
    ;;
  *)
    printf 'unsupported repair mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac

project_root=$(pwd)
shared_project_root=$(readlink -f "$project_root")
stamp=$(date -u +%Y%m%d_%H%M%S)
job_name="squad_lora_10e_${mode}_${stamp}"
log_dir="$shared_project_root/logs"
mkdir -p "$log_dir"

cache_root="/home/c01xuju/.cache"
data_root="/home/c01xuju/.local/share"
job_command="set +e; source /home/c01xuju/.envrc; set -euo pipefail"
job_command+="; command -v uv >/dev/null"
job_command+="; export WANDB_MODE=offline"
job_command+="; export TOKENIZERS_PARALLELISM=false"
job_command+="; export TORCH_DISABLE_NATIVE_JIT=1"
job_command+="; export XDG_CACHE_HOME=$cache_root"
job_command+="; export WANDB_CACHE_DIR=$cache_root/wandb"
job_command+="; export WANDB_DATA_DIR=$data_root/wandb"
job_command+="; export TRITON_CACHE_DIR=$cache_root/triton"
job_command+="; mkdir -p $cache_root/wandb $cache_root/triton $data_root/wandb"
job_command+="; cd $project_root; $command"

sbatch \
  --parsable \
  --nodes=1 \
  --ntasks=1 \
  --partition="$partition" \
  --cpus-per-task=32 \
  --gpus-per-node=1 \
  --job-name="$job_name" \
  --output="$log_dir/%x_%j.log" \
  --time=6-23:30:00 \
  --container-image="$PYXIS_CONTAINER_IMAGE" \
  --container-workdir="$project_root" \
  --container-mounts="$PYXIS_CONTAINER_MOUNTS" \
  --container-name="$job_name" \
  --wrap="exec bash -lc '$job_command'"
