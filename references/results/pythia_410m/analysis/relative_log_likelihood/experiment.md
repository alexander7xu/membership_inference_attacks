# Experiment Record

## Purpose

Compare nine candidate-group relative_log_likelihood distributions for Pythia-410M.

## Hypothesis

Data source and text origin change the target-versus-shadow relative log-likelihood distribution.

## Command

`uv run python ./cli/squad_lora_rmia.py workflow.stage=plot_rmia_feature workflow.profile=formal runtime.seed=42 llm_model=pythia_410m`

## Hydra Overrides


## Fingerprint

- Config SHA256: `4e5a7f029b0756036f5ebfe44a764350405cd4d942491913d4cb1de593dde7c7`

## Source State

```json
{
  "branch": "codex/squad-lora-rmia",
  "commit": "601a0fd5a98b0497bb718e8ff866cc7270f2fd25",
  "short_commit": "601a0fd",
  "status": "M conf/squad_lora_rmia.yaml\n M src/llm_mia/workflow.py\n?? cli/debug_squad_lora_gpu.py\n?? scripts/debug_gpu_utilization_suite.sh\n?? scripts/debug_inference_gpu_utilization.sh\n?? scripts/debug_training_gpu_utilization.sh\n?? src/llm_mia/plotting.py\n?? tests/test_llm_mia_plotting.py"
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

- Run ID: `fmu49q22`

## Artifacts

- figure: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/analysis/relative_log_likelihood/relative_log_likelihood_ecdf.png`
- sampled_candidates: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/analysis/relative_log_likelihood/sampled_candidates.jsonl`
- resolved_config: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/analysis/relative_log_likelihood/resolved_config.yaml`
- manifest: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/analysis/relative_log_likelihood/manifest.json`
- wandb_rmia_feature_plot_artifact: `fmu49q22-rmia_feature_plot:68c7ca4bc4c9203513335cb4025e906a`

## Metrics

- gold_squad_target_train_count: `512.0`
- gold_squad_target_train_mean: `0.0252268568226384`
- gold_squad_target_train_median: `0.004158680563913353`
- base_gen_from_squad_train_count: `512.0`
- base_gen_from_squad_train_mean: `-0.03644123893103412`
- base_gen_from_squad_train_median: `-0.02206433817296727`
- lora_gen_from_squad_train_count: `512.0`
- lora_gen_from_squad_train_mean: `-0.0003673346519890081`
- lora_gen_from_squad_train_median: `-0.0012946437565205277`
- gold_squad_validation_count: `512.0`
- gold_squad_validation_mean: `-0.05831165003441336`
- gold_squad_validation_median: `-0.0032162001804901946`
- base_gen_from_squad_validation_count: `512.0`
- base_gen_from_squad_validation_mean: `-0.03639760992355202`
- base_gen_from_squad_validation_median: `-0.027844622428620402`
- lora_gen_from_squad_validation_count: `512.0`
- lora_gen_from_squad_validation_mean: `-0.0006644722402762357`
- lora_gen_from_squad_validation_median: `-0.0007794181017187745`
- gold_trivia_validation_count: `512.0`
- gold_trivia_validation_mean: `-0.4092947508286577`
- gold_trivia_validation_median: `-0.266547684136752`
- base_gen_from_trivia_validation_count: `512.0`
- base_gen_from_trivia_validation_mean: `-0.3060094164281831`
- base_gen_from_trivia_validation_median: `-0.18374268648609388`
- lora_gen_from_trivia_validation_count: `512.0`
- lora_gen_from_trivia_validation_mean: `0.15665887863579214`
- lora_gen_from_trivia_validation_median: `0.05191579906918601`

## Conclusion

The figure is descriptive: color encodes the three data sources, line style encodes gold/base-generated/LoRA-generated text, and every curve uses an equal-size deterministic sample.

## Achieved Purpose

yes

## Next Action

Interpret group separation alongside token length and reference scores.
