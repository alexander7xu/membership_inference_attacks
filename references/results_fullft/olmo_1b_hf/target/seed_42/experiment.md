# Experiment Record

## Purpose

target full-parameter fine-tuning for OLMo-1B-hf on SQuAD QA.

## Hypothesis

Target-trained SQuAD records exhibit different target/reference score distributions than IID, OOD, and generated non-members.

## Command

`uv run python ./cli/squad_fullft_rmia.py runtime.seed=42 workflow.profile=formal llm_model=olmo_1b_hf train.per_device_train_batch_size=8 train.gradient_accumulation_steps=4 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 workflow.stage=formal`

## Hydra Overrides


## Fingerprint

- Config SHA256: `56655b5506f5462dda786da9fcbb740273e04975beda17719156b8e4422256b8`

## Source State

```json
{
  "branch": "codex/squad-lora-rmia",
  "commit": "c3d3cd79a4b09aa12187ec462e1776033647b506",
  "short_commit": "c3d3cd7",
  "status": "?? cli/debug_squad_lora_gpu.py\n?? scripts/debug_gpu_utilization_suite.sh\n?? scripts/debug_inference_gpu_utilization.sh\n?? scripts/debug_training_gpu_utilization.sh"
}
```

## Environment

```json
{
  "accelerate": "1.14.0",
  "cuda": "12.6",
  "cuda_available": true,
  "datasets": "5.0.0",
  "gpu": "NVIDIA A800-SXM4-80GB",
  "gpu_driver": "550.54.14",
  "peft": "0.19.1",
  "platform": "Linux-4.18.0-305.el8.x86_64-x86_64-with-glibc2.35",
  "python": "3.13.13 (main, Jun  2 2026, 22:27:49) [Clang 22.1.3 ]",
  "torch": "2.13.0+cu126",
  "transformers": "5.13.1",
  "uv": "uv 0.11.28 (x86_64-unknown-linux-gnu)",
  "uv_lock_sha256": "2691b73f92add6bc799345e4813eee9db6c31378ab813a2c5ac9adea2f185ccf"
}
```

## W&B

- Run ID: `6ay0yitf`

## Artifacts

- model: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/target/seed_42/model`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/target/seed_42/metrics.json`
- resolved_config: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/target/seed_42/resolved_config.yaml`
- wandb_target_artifact: `6ay0yitf-target:023cacbb4f55cec1b7e205b540e98d34`

## Metrics

- train/train_runtime: `884.2711`
- train/train_samples_per_second: `49.531`
- train/train_steps_per_second: `1.548`
- train/total_flos: `8.567521734662554e+16`
- train/train_loss: `0.15937774359004353`
- train/epoch: `1.0`
- train/records: `43799.0`
- train/model_seed: `42.0`
- train/trainable_parameters: `1176764416.0`
- train/total_parameters: `1176764416.0`
- train/peak_gpu_memory_bytes: `14514216960.0`
- train/final_learning_rate: `1.5071590052750567e-07`
- train/final_grad_norm: `5.3125`

## Conclusion

target full-parameter checkpoint saved from the final epoch/step without checkpoint selection.

## Achieved Purpose

yes

## Next Action

Evaluate utility or score candidates with this final checkpoint.
