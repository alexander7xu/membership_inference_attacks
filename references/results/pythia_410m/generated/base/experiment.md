# Experiment Record

## Purpose

Generate reusable base-model completions for Pythia-410M from the SQuAD train, SQuAD validation, and TriviaQA validation sources.

## Hypothesis

Base-model generation on the same deterministic source subsets provides a reproducible pre-LoRA reference for generated-data comparisons.

## Command

`uv run python ./cli/squad_lora_rmia.py runtime.seed=42 workflow.stage=generate_base workflow.profile=formal workflow.force=true llm_model=pythia_410m tokenizer.preprocessing_workers=8 eval.batch_size=96 generation.batch_size=96`

## Hydra Overrides


## Fingerprint

- Config SHA256: `f2d707573f05beffdc022cde0644564802ac934d0ecce2e33d3322658cff77f0`

## Source State

```json
{
  "branch": "codex/squad-lora-rmia",
  "commit": "b15c9890bb809e186cdebe82589658e290a40edd",
  "short_commit": "b15c989",
  "status": "M cli/squad_lora_rmia.py\n M conf/squad_lora_rmia.yaml\n M src/llm_mia/workflow.py\n?? cli/debug_squad_lora_gpu.py\n?? scripts/debug_gpu_utilization_suite.sh\n?? scripts/debug_inference_gpu_utilization.sh\n?? scripts/debug_training_gpu_utilization.sh\n?? scripts/run_squad_base_generation_formal.sh\n?? src/llm_mia/plotting.py\n?? tests/test_llm_mia_cli.py\n?? tests/test_llm_mia_generation.py\n?? tests/test_llm_mia_plotting.py"
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

- Run ID: `zaadbf7k`

## Artifacts

- public_generated: `outputs/squad_lora_rmia/formal/pythia_410m/generated/base/public_generated.jsonl`
- private_generated_labels: `outputs/squad_lora_rmia/formal/pythia_410m/generated/base/private_generated_labels.jsonl`
- metrics: `outputs/squad_lora_rmia/formal/pythia_410m/generated/base/metrics.json`
- resolved_config: `outputs/squad_lora_rmia/formal/pythia_410m/generated/base/resolved_config.yaml`
- model: `EleutherAI/pythia-410m@9879c9b5f8bea9051dcb0e68dff21493d67e9d4f`
- adapter: `base-model`
- wandb_generated_base_artifact: `zaadbf7k-generated_base:1d6577e4e99c27825ec22e1bbdd28ce0`

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

Compare base and target-LoRA generation metrics and distributions.
