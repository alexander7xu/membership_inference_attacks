# Experiment Record

## Purpose

Generate reusable base-model completions for OLMo-1B-hf from the SQuAD train, SQuAD validation, and TriviaQA validation sources.

## Hypothesis

Base-model generation on the same deterministic source subsets provides a reproducible pre-LoRA reference for generated-data comparisons.

## Command

`uv run python ./cli/squad_fullft_rmia.py runtime.seed=42 workflow.profile=formal llm_model=olmo_1b_hf train.per_device_train_batch_size=8 train.gradient_accumulation_steps=4 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 workflow.stage=generate_base`

## Hydra Overrides


## Fingerprint

- Config SHA256: `f31870bad273aae2645ff724015975f901a4433bd7a63535d10115359f2bb19b`

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

- Run ID: `eufyp3fi`

## Artifacts

- public_generated: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/base/public_generated.jsonl`
- private_generated_labels: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/base/private_generated_labels.jsonl`
- metrics: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/base/metrics.json`
- resolved_config: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/base/resolved_config.yaml`
- model: `allenai/OLMo-1B-hf@aee7752d9c08ee4775e9b0091426d8410e8f6a89`
- checkpoint: `base-model`
- wandb_generated_base_artifact: `eufyp3fi-generated_base:b8c9ebd5fe20645aebd516f63d126b91`

## Metrics

- gen_from_squad_train/exact_match: `0.0`
- gen_from_squad_train/f1: `0.0058089275769362845`
- gen_from_squad_train/duplicate_rate: `0.0`
- gen_from_squad_train/exact_reconstruction_rate: `0.0`
- gen_from_squad_train/completion_loss: `1.4346213333774358`
- gen_from_squad_train/completion_perplexity: `4.198055044448689`
- gen_from_squad_train/completion_tokens: `65536.0`
- gen_from_squad_train/examples: `1024.0`
- gen_from_squad_train/generated_completion_tokens_mean: `64.0`
- gen_from_squad_train/generated_completion_tokens_min: `64.0`
- gen_from_squad_train/generated_completion_tokens_max: `64.0`
- gen_from_squad_validation/exact_match: `0.0`
- gen_from_squad_validation/f1: `0.007142570671148995`
- gen_from_squad_validation/duplicate_rate: `0.0`
- gen_from_squad_validation/exact_reconstruction_rate: `0.0`
- gen_from_squad_validation/completion_loss: `1.4538369566434994`
- gen_from_squad_validation/completion_perplexity: `4.279503322194138`
- gen_from_squad_validation/completion_tokens: `65536.0`
- gen_from_squad_validation/examples: `1024.0`
- gen_from_squad_validation/generated_completion_tokens_mean: `64.0`
- gen_from_squad_validation/generated_completion_tokens_min: `64.0`
- gen_from_squad_validation/generated_completion_tokens_max: `64.0`
- gen_from_trivia_validation/exact_match: `0.0`
- gen_from_trivia_validation/f1: `0.006502704466529964`
- gen_from_trivia_validation/duplicate_rate: `0.4375`
- gen_from_trivia_validation/exact_reconstruction_rate: `0.0`
- gen_from_trivia_validation/completion_loss: `2.403062955573386`
- gen_from_trivia_validation/completion_perplexity: `11.056991641056241`
- gen_from_trivia_validation/completion_tokens: `12505.0`
- gen_from_trivia_validation/examples: `1024.0`
- gen_from_trivia_validation/generated_completion_tokens_mean: `64.0`
- gen_from_trivia_validation/generated_completion_tokens_min: `64.0`
- gen_from_trivia_validation/generated_completion_tokens_max: `64.0`

## Conclusion

Base-model generations and source provenance were saved for reuse.

## Achieved Purpose

yes

## Next Action

Compare base and fine-tuned generation metrics and distributions.
