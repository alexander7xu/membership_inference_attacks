# Experiment Record

## Purpose

Generate reusable target-model candidate completions for Pythia-410M.

## Hypothesis

Generated records will mostly be non-members unless prompt and generated completion exactly reconstruct a target training record.

## Command

`uv run python ./cli/squad_fullft_rmia.py runtime.seed=42 workflow.profile=formal llm_model=pythia_410m train.per_device_train_batch_size=16 train.gradient_accumulation_steps=2 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 workflow.stage=formal`

## Hydra Overrides


## Fingerprint

- Config SHA256: `8ed3376a0bbffe7e5fe71ca5df329d796b279f15a042059d7c732eb038855030`

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

- Run ID: `y3rp58g2`

## Artifacts

- public_generated: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/public_generated.jsonl`
- private_generated_labels: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/private_generated_labels.jsonl`
- metrics: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/metrics.json`
- resolved_config: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/resolved_config.yaml`
- model: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/target/seed_42/model`
- checkpoint: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/target/seed_42/model`
- wandb_generated_artifact: `y3rp58g2-generated:3be99a06531d54345d45bdc652ca7080`

## Metrics

- fullft_gen_from_squad_train/exact_match: `0.0`
- fullft_gen_from_squad_train/f1: `0.10638435656479397`
- fullft_gen_from_squad_train/duplicate_rate: `0.0`
- fullft_gen_from_squad_train/exact_reconstruction_rate: `0.0`
- fullft_gen_from_squad_train/completion_loss: `0.04821161090092392`
- fullft_gen_from_squad_train/completion_perplexity: `1.0493926947642211`
- fullft_gen_from_squad_train/completion_tokens: `65454.0`
- fullft_gen_from_squad_train/examples: `1024.0`
- fullft_gen_from_squad_train/generated_completion_tokens_mean: `63.919921875`
- fullft_gen_from_squad_train/generated_completion_tokens_min: `62.0`
- fullft_gen_from_squad_train/generated_completion_tokens_max: `64.0`
- fullft_gen_from_squad_validation/exact_match: `0.0`
- fullft_gen_from_squad_validation/f1: `0.12187249354524293`
- fullft_gen_from_squad_validation/duplicate_rate: `0.0`
- fullft_gen_from_squad_validation/exact_reconstruction_rate: `0.0`
- fullft_gen_from_squad_validation/completion_loss: `0.04785915266253757`
- fullft_gen_from_squad_validation/completion_perplexity: `1.049022892837345`
- fullft_gen_from_squad_validation/completion_tokens: `65471.0`
- fullft_gen_from_squad_validation/examples: `1024.0`
- fullft_gen_from_squad_validation/generated_completion_tokens_mean: `63.9365234375`
- fullft_gen_from_squad_validation/generated_completion_tokens_min: `61.0`
- fullft_gen_from_squad_validation/generated_completion_tokens_max: `64.0`
- fullft_gen_from_trivia_validation/exact_match: `0.0`
- fullft_gen_from_trivia_validation/f1: `0.06408555210757683`
- fullft_gen_from_trivia_validation/duplicate_rate: `0.4375`
- fullft_gen_from_trivia_validation/exact_reconstruction_rate: `0.0`
- fullft_gen_from_trivia_validation/completion_loss: `0.16371185560756907`
- fullft_gen_from_trivia_validation/completion_perplexity: `1.1778748681519613`
- fullft_gen_from_trivia_validation/completion_tokens: `12504.0`
- fullft_gen_from_trivia_validation/examples: `1024.0`
- fullft_gen_from_trivia_validation/generated_completion_tokens_mean: `63.953125`
- fullft_gen_from_trivia_validation/generated_completion_tokens_min: `62.0`
- fullft_gen_from_trivia_validation/generated_completion_tokens_max: `64.0`

## Conclusion

Generated artifacts and private provenance labels were saved for reuse.

## Achieved Purpose

yes

## Next Action

Build public attack candidates and shadow inclusion masks.
