# Experiment Record

## Purpose

Compare nine candidate-group relative_log_likelihood distributions for Pythia-410M.

## Hypothesis

Data source and text origin change the target-versus-shadow relative log-likelihood distribution.

## Command

`uv run python ./cli/squad_fullft_rmia.py runtime.seed=42 workflow.profile=formal llm_model=pythia_410m train.per_device_train_batch_size=16 train.gradient_accumulation_steps=2 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 workflow.stage=plot_rmia_feature`

## Hydra Overrides


## Fingerprint

- Config SHA256: `f68fdb6216fb4c1f93a230d915be63f93c3dd8822be69599f050e9baa7bd3a90`

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

- Run ID: `8lve8146`

## Artifacts

- figure: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/analysis/relative_log_likelihood/relative_log_likelihood_ecdf.png`
- sampled_candidates: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/analysis/relative_log_likelihood/sampled_candidates.jsonl`
- resolved_config: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/analysis/relative_log_likelihood/resolved_config.yaml`
- manifest: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/analysis/relative_log_likelihood/manifest.json`
- wandb_rmia_feature_plot_artifact: `8lve8146-rmia_feature_plot:a078b890ea6da6b9f059f5dd90f097e6`

## Metrics

- gold_squad_target_train_count: `512.0`
- gold_squad_target_train_mean: `-0.050233699147442225`
- gold_squad_target_train_median: `0.006605270474906765`
- base_gen_from_squad_train_count: `512.0`
- base_gen_from_squad_train_mean: `0.017001035807481617`
- base_gen_from_squad_train_median: `0.016985028361618526`
- fullft_gen_from_squad_train_count: `512.0`
- fullft_gen_from_squad_train_mean: `0.0005927088638723197`
- fullft_gen_from_squad_train_median: `-0.0009321229460628232`
- gold_squad_validation_count: `512.0`
- gold_squad_validation_mean: `-0.12417318072693198`
- gold_squad_validation_median: `-0.015506866111946897`
- base_gen_from_squad_validation_count: `512.0`
- base_gen_from_squad_validation_mean: `0.02836768334673223`
- base_gen_from_squad_validation_median: `0.018970667775098482`
- fullft_gen_from_squad_validation_count: `512.0`
- fullft_gen_from_squad_validation_mean: `-9.86250458117575e-05`
- fullft_gen_from_squad_validation_median: `-0.0014783784259210753`
- gold_trivia_validation_count: `512.0`
- gold_trivia_validation_mean: `-0.41104050168373096`
- gold_trivia_validation_median: `-0.25118787018382727`
- base_gen_from_trivia_validation_count: `512.0`
- base_gen_from_trivia_validation_mean: `-0.04922345721170931`
- base_gen_from_trivia_validation_median: `-0.013963272601530408`
- fullft_gen_from_trivia_validation_count: `512.0`
- fullft_gen_from_trivia_validation_mean: `0.10431503244107071`
- fullft_gen_from_trivia_validation_median: `0.02590343761395144`

## Conclusion

The figure is descriptive: color encodes the three data sources, line style encodes gold/base-generated/full-parameter-generated text, and every curve uses an equal-size deterministic sample.

## Achieved Purpose

yes

## Next Action

Interpret group separation alongside token length and reference scores.
