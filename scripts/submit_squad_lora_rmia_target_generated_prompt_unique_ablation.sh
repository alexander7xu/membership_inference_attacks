#!/usr/bin/env bash
set -euo pipefail

source /home/c01xuju/CISPA-home/home/.envrc
cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
  printf 'usage: %s {pythia-full|olmo-full}\n' "$0" >&2
  exit 2
fi

mode=$1
case "$mode" in
  pythia-full)
    partition="xe8545"
    model="pythia_410m"
    ;;
  olmo-full)
    partition="tmp"
    model="olmo_1b_hf"
    ;;
  *)
    printf 'unsupported target-generated RMIA mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac

project_root=$(pwd)
shared_project_root=$(readlink -f "$project_root")
stamp=$(date -u +%Y%m%d_%H%M%S)
job_name="squad_lora_rmia_target_generated_prompt_unique_${model}_${stamp}"
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
job_command+="; cd $project_root"
job_command+="; exec bash ./scripts/run_squad_lora_rmia_target_generated_prompt_unique_ablation_formal.sh $model"

exclude_args=()
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
  exclude_args+=(--exclude="$EXCLUDE_NODES")
fi

dependency_args=()
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  dependency_args+=(--dependency="afterok:$AFTEROK_JOB_ID")
fi

sbatch \
  --parsable \
  "${exclude_args[@]}" \
  "${dependency_args[@]}" \
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
