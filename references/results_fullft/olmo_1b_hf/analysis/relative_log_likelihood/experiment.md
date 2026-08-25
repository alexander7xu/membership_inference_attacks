# Experiment Record

## Purpose

Compare nine candidate-group relative_log_likelihood distributions for OLMo-1B-hf.

## Hypothesis

Data source and text origin change the target-versus-shadow relative log-likelihood distribution.

## Command

`uv run python ./cli/squad_fullft_rmia.py runtime.seed=42 workflow.profile=formal llm_model=olmo_1b_hf train.per_device_train_batch_size=8 train.gradient_accumulation_steps=4 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 workflow.stage=plot_rmia_feature`

## Hydra Overrides


## Fingerprint

- Config SHA256: `9a9da4269aa1b8113917eef4451e90c36a3c7be2ed19ece6eb56c6a177ecb8cd`

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

- Run ID: `x8k0g6r0`

## Artifacts

- figure: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/analysis/relative_log_likelihood/relative_log_likelihood_ecdf.png`
- sampled_candidates: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/analysis/relative_log_likelihood/sampled_candidates.jsonl`
- resolved_config: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/analysis/relative_log_likelihood/resolved_config.yaml`
- manifest: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/analysis/relative_log_likelihood/manifest.json`
- wandb_rmia_feature_plot_artifact: `x8k0g6r0-rmia_feature_plot:55ed1ea0dbcd08fdfdf00343a97151d7`

## Metrics

- gold_squad_target_train_count: `512.0`
- gold_squad_target_train_mean: `0.025820082399052844`
- gold_squad_target_train_median: `0.0023091256923669532`
- base_gen_from_squad_train_count: `512.0`
- base_gen_from_squad_train_mean: `-0.3800567506605527`
- base_gen_from_squad_train_median: `-0.37754675528909676`
- fullft_gen_from_squad_train_count: `512.0`
- fullft_gen_from_squad_train_mean: `-1.026448442774133`
- fullft_gen_from_squad_train_median: `-0.7353135935818546`
- gold_squad_validation_count: `512.0`
- gold_squad_validation_mean: `-0.04795969289641959`
- gold_squad_validation_median: `-0.00035991495204488983`
- base_gen_from_squad_validation_count: `512.0`
- base_gen_from_squad_validation_mean: `-0.37680046254423954`
- base_gen_from_squad_validation_median: `-0.37486756368398094`
- fullft_gen_from_squad_validation_count: `512.0`
- fullft_gen_from_squad_validation_mean: `-1.0160140619358233`
- fullft_gen_from_squad_validation_median: `-0.7337750720909768`
- gold_trivia_validation_count: `512.0`
- gold_trivia_validation_mean: `-0.86480417516462`
- gold_trivia_validation_median: `-0.18277755572857213`
- base_gen_from_trivia_validation_count: `512.0`
- base_gen_from_trivia_validation_mean: `-1.7970488742800366`
- base_gen_from_trivia_validation_median: `-1.657307096121173`
- fullft_gen_from_trivia_validation_count: `512.0`
- fullft_gen_from_trivia_validation_mean: `-4.046050440620103`
- fullft_gen_from_trivia_validation_median: `-4.081695333156633`

## Conclusion

The figure is descriptive: color encodes the three data sources, line style encodes gold/base-generated/full-parameter-generated text, and every curve uses an equal-size deterministic sample.

## Achieved Purpose

yes

## Next Action

Interpret group separation alongside token length and reference scores.
