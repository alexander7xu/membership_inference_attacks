# Experiment Record

## Purpose

Generate reusable base-model completions for Pythia-410M from the SQuAD train, SQuAD validation, and TriviaQA validation sources.

## Hypothesis

Base-model generation on the same deterministic source subsets provides a reproducible pre-LoRA reference for generated-data comparisons.

## Command

`uv run python ./cli/squad_fullft_rmia.py runtime.seed=42 workflow.profile=formal llm_model=pythia_410m train.per_device_train_batch_size=16 train.gradient_accumulation_steps=2 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 workflow.stage=generate_base`

## Hydra Overrides


## Fingerprint

- Config SHA256: `58f8152f045f2d49dd85297b7ca6ee994027b6e74ba142c4831fba7cc2534e35`

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

- Run ID: `htfz85hr`

## Artifacts

- public_generated: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/base/public_generated.jsonl`
- private_generated_labels: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/base/private_generated_labels.jsonl`
- metrics: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/base/metrics.json`
- resolved_config: `outputs/squad_fullft_rmia/formal/pythia_410m/generated/base/resolved_config.yaml`
- model: `EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`
- checkpoint: `base-model`
- wandb_generated_base_artifact: `htfz85hr-generated_base:468d829aa3513d4fcb9affc012dbdc14`

## Metrics

- gen_from_squad_train/exact_match: `0.0`
- gen_from_squad_train/f1: `0.056371794596988246`
- gen_from_squad_train/duplicate_rate: `0.0`
- gen_from_squad_train/exact_reconstruction_rate: `0.0`
- gen_from_squad_train/completion_loss: `0.791102398133437`
- gen_from_squad_train/completion_perplexity: `2.2058267857292053`
- gen_from_squad_train/completion_tokens: `65054.0`
- gen_from_squad_train/examples: `1024.0`
- gen_from_squad_train/generated_completion_tokens_mean: `63.529296875`
- gen_from_squad_train/generated_completion_tokens_min: `12.0`
- gen_from_squad_train/generated_completion_tokens_max: `65.0`
- gen_from_squad_validation/exact_match: `0.0`
- gen_from_squad_validation/f1: `0.06136622288133174`
- gen_from_squad_validation/duplicate_rate: `0.0`
- gen_from_squad_validation/exact_reconstruction_rate: `0.0`
- gen_from_squad_validation/completion_loss: `0.7988473663603777`
- gen_from_squad_validation/completion_perplexity: `2.2229771729712025`
- gen_from_squad_validation/completion_tokens: `65228.0`
- gen_from_squad_validation/examples: `1024.0`
- gen_from_squad_validation/generated_completion_tokens_mean: `63.69921875`
- gen_from_squad_validation/generated_completion_tokens_min: `61.0`
- gen_from_squad_validation/generated_completion_tokens_max: `65.0`
- gen_from_trivia_validation/exact_match: `0.0`
- gen_from_trivia_validation/f1: `0.05127095813841343`
- gen_from_trivia_validation/duplicate_rate: `0.4375`
- gen_from_trivia_validation/exact_reconstruction_rate: `0.0`
- gen_from_trivia_validation/completion_loss: `1.1410435705698063`
- gen_from_trivia_validation/completion_perplexity: `3.1300330718102587`
- gen_from_trivia_validation/completion_tokens: `12378.0`
- gen_from_trivia_validation/examples: `1024.0`
- gen_from_trivia_validation/generated_completion_tokens_mean: `63.3173828125`
- gen_from_trivia_validation/generated_completion_tokens_min: `43.0`
- gen_from_trivia_validation/generated_completion_tokens_max: `66.0`

## Conclusion

Base-model generations and source provenance were saved for reuse.

## Achieved Purpose

yes

## Next Action

Compare base and fine-tuned generation metrics and distributions.
