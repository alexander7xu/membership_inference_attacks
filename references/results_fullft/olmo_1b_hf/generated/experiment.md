# Experiment Record

## Purpose

Generate reusable target-model candidate completions for OLMo-1B-hf.

## Hypothesis

Generated records will mostly be non-members unless prompt and generated completion exactly reconstruct a target training record.

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

- public_generated: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/public_generated.jsonl`
- private_generated_labels: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/private_generated_labels.jsonl`
- metrics: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/metrics.json`
- resolved_config: `outputs/squad_fullft_rmia/formal/olmo_1b_hf/generated/resolved_config.yaml`
- model: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/target/seed_42/model`
- checkpoint: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/target/seed_42/model`
- wandb_generated_artifact: `6ay0yitf-generated:0e496ba44fb2425ac7b3771bc98e42f8`

## Metrics

- fullft_gen_from_squad_train/exact_match: `0.0`
- fullft_gen_from_squad_train/f1: `0.009401447335103912`
- fullft_gen_from_squad_train/duplicate_rate: `0.0`
- fullft_gen_from_squad_train/exact_reconstruction_rate: `0.0`
- fullft_gen_from_squad_train/completion_loss: `0.865848737561772`
- fullft_gen_from_squad_train/completion_perplexity: `2.377022698518067`
- fullft_gen_from_squad_train/completion_tokens: `54843.0`
- fullft_gen_from_squad_train/examples: `1024.0`
- fullft_gen_from_squad_train/generated_completion_tokens_mean: `53.5576171875`
- fullft_gen_from_squad_train/generated_completion_tokens_min: `11.0`
- fullft_gen_from_squad_train/generated_completion_tokens_max: `64.0`
- fullft_gen_from_squad_validation/exact_match: `0.0`
- fullft_gen_from_squad_validation/f1: `0.011610016690856116`
- fullft_gen_from_squad_validation/duplicate_rate: `0.0`
- fullft_gen_from_squad_validation/exact_reconstruction_rate: `0.0`
- fullft_gen_from_squad_validation/completion_loss: `0.8858309244269167`
- fullft_gen_from_squad_validation/completion_perplexity: `2.4249985450913014`
- fullft_gen_from_squad_validation/completion_tokens: `53754.0`
- fullft_gen_from_squad_validation/examples: `1024.0`
- fullft_gen_from_squad_validation/generated_completion_tokens_mean: `52.494140625`
- fullft_gen_from_squad_validation/generated_completion_tokens_min: `4.0`
- fullft_gen_from_squad_validation/generated_completion_tokens_max: `64.0`
- fullft_gen_from_trivia_validation/exact_match: `0.0`
- fullft_gen_from_trivia_validation/f1: `0.04032607721766071`
- fullft_gen_from_trivia_validation/duplicate_rate: `0.4375`
- fullft_gen_from_trivia_validation/exact_reconstruction_rate: `0.0`
- fullft_gen_from_trivia_validation/completion_loss: `2.9140968634617352`
- fullft_gen_from_trivia_validation/completion_perplexity: `18.43215812785251`
- fullft_gen_from_trivia_validation/completion_tokens: `6826.0`
- fullft_gen_from_trivia_validation/examples: `1024.0`
- fullft_gen_from_trivia_validation/generated_completion_tokens_mean: `24.0673828125`
- fullft_gen_from_trivia_validation/generated_completion_tokens_min: `5.0`
- fullft_gen_from_trivia_validation/generated_completion_tokens_max: `64.0`

## Conclusion

Generated artifacts and private provenance labels were saved for reuse.

## Achieved Purpose

yes

## Next Action

Build public attack candidates and shadow inclusion masks.
