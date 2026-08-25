# Experiment Record

## Purpose

Generate reusable target-model candidate completions for OLMo-1B-hf.

## Hypothesis

Generated records will mostly be non-members unless prompt and generated completion exactly reconstruct a target training record.

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

- public_generated: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/generated/public_generated.jsonl`
- private_generated_labels: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/generated/private_generated_labels.jsonl`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/generated/metrics.json`
- wandb_generated_artifact: `0l18l0ay-generated:4271610e13c716bb0d7094ee544e516e`

## Metrics

- gen_from_squad_train/exact_match: `0.0`
- gen_from_squad_train/f1: `0.0043744362183684255`
- gen_from_squad_train/duplicate_rate: `0.0`
- gen_from_squad_train/exact_reconstruction_rate: `0.0`
- gen_from_squad_train/completion_loss: `1.1550489364310697`
- gen_from_squad_train/completion_perplexity: `3.1741787467063824`
- gen_from_squad_train/completion_tokens: `55030.0`
- gen_from_squad_train/examples: `1024.0`
- gen_from_squad_train/generated_completion_tokens_mean: `53.740234375`
- gen_from_squad_train/generated_completion_tokens_min: `27.0`
- gen_from_squad_train/generated_completion_tokens_max: `64.0`
- gen_from_squad_validation/exact_match: `0.0`
- gen_from_squad_validation/f1: `0.005835328776299696`
- gen_from_squad_validation/duplicate_rate: `0.0`
- gen_from_squad_validation/exact_reconstruction_rate: `0.0`
- gen_from_squad_validation/completion_loss: `1.1611084763025898`
- gen_from_squad_validation/completion_perplexity: `3.1934712020508718`
- gen_from_squad_validation/completion_tokens: `55115.0`
- gen_from_squad_validation/examples: `1024.0`
- gen_from_squad_validation/generated_completion_tokens_mean: `53.8232421875`
- gen_from_squad_validation/generated_completion_tokens_min: `23.0`
- gen_from_squad_validation/generated_completion_tokens_max: `64.0`
- gen_from_trivia_validation/exact_match: `0.0`
- gen_from_trivia_validation/f1: `0.005465035417898123`
- gen_from_trivia_validation/duplicate_rate: `0.4375`
- gen_from_trivia_validation/exact_reconstruction_rate: `0.0`
- gen_from_trivia_validation/completion_loss: `2.489986302232102`
- gen_from_trivia_validation/completion_perplexity: `12.061110909015383`
- gen_from_trivia_validation/completion_tokens: `10274.0`
- gen_from_trivia_validation/examples: `1024.0`
- gen_from_trivia_validation/generated_completion_tokens_mean: `49.2353515625`
- gen_from_trivia_validation/generated_completion_tokens_min: `21.0`
- gen_from_trivia_validation/generated_completion_tokens_max: `64.0`

## Conclusion

Generated artifacts and private provenance labels were saved for reuse.

## Achieved Purpose

yes

## Next Action

Build public attack candidates and shadow inclusion masks.
