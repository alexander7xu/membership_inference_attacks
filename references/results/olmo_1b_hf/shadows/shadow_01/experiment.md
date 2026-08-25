# Experiment Record

## Purpose

shadow LoRA fine-tuning for OLMo-1B-hf on SQuAD QA.

## Hypothesis

Target-trained SQuAD records exhibit different target/reference score distributions than IID, OOD, and generated non-members.

## Command

`/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/.venv/bin/python3 ./cli/squad_lora_rmia.py runtime.seed=42 workflow.stage=formal workflow.profile=formal llm_model=olmo_1b_hf train.per_device_train_batch_size=16 train.gradient_accumulation_steps=2 train.dataloader_num_workers=4 tokenizer.preprocessing_workers=8 eval.batch_size=96 eval.loss_chunk_tokens=64 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 attack.loss_chunk_tokens=64`

## Hydra Overrides


## Fingerprint

- Config SHA256: `9117ab94135600b9bf610bcda34aa37507469c41b9c4d32b069c6fa6e97dcda7`

## Reproducible Source Snapshot

- Branch: `codex/squad-lora-rmia`
- Plan commit: `b31db23342202ab001f9fe53bbbb16c0440dcf69`
- Source snapshot commit: `c0c177d27cbb48f4fbb9aa3e9c0fb0b98782a75d`
- Execution base commit: `3f8e7030098d0310c34efb1c74f3e7574da8ac4e` with the dirty worktree recorded below.
- Provenance note: the source snapshot was committed retrospectively from the recorded execution worktree; the original execution source state is preserved verbatim.

## Execution Source State

```json
{
  "branch": "codex/squad-lora-rmia",
  "commit": "3f8e7030098d0310c34efb1c74f3e7574da8ac4e",
  "short_commit": "3f8e703",
  "status": "M README.md\n M scripts/run_squad_lora_rmia_formal.sh\n?? cli/debug_squad_lora_gpu.py\n?? conf/llm_model/olmo_1b_hf.yaml\n?? scripts/debug_gpu_utilization_suite.sh\n?? scripts/debug_inference_gpu_utilization.sh\n?? scripts/debug_training_gpu_utilization.sh"
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
  "gpu_driver": "570.86.10",
  "peft": "0.19.1",
  "platform": "Linux-5.4.0-165-generic-x86_64-with-glibc2.35",
  "python": "3.13.13 (main, Jun  2 2026, 22:27:49) [Clang 22.1.3 ]",
  "torch": "2.13.0+cu126",
  "transformers": "5.13.1",
  "uv": "uv 0.11.28 (x86_64-unknown-linux-gnu)",
  "uv_lock_sha256": "2691b73f92add6bc799345e4813eee9db6c31378ab813a2c5ac9adea2f185ccf"
}
```

## W&B

- Run ID: `0l18l0ay`

## Artifacts

- adapter: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/shadows/shadow_01/adapter`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/shadows/shadow_01/metrics.json`
- resolved_config: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/shadows/shadow_01/resolved_config.yaml`
- wandb_shadow_01_artifact: `0l18l0ay-shadow_01:08965057d74efa6c6e942b01292c0001`

## Metrics

- train/train_runtime: `1123.0437`
- train/train_samples_per_second: `38.999`
- train/train_steps_per_second: `1.219`
- train/total_flos: `1.3117727203078963e+17`
- train/train_loss: `0.14070792077143232`
- train/epoch: `1.0`
- train/records: `43799.0`
- train/trainable_parameters: `12058624.0`

## Conclusion

shadow LoRA adapter saved from the final epoch/step.

## Achieved Purpose

yes

## Next Action

Evaluate utility or score candidates with this final adapter.
