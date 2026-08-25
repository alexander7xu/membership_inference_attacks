# Experiment Record

## Purpose

Analyze text-only RMIA score distributions for Pythia-410M.

## Hypothesis

Target-trained SQuAD records exhibit different target/reference score distributions than IID, OOD, and generated non-members.

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

- scores: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/attack/online_rmia_scores.jsonl`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/attack/metrics.json`
- public_candidates: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/candidates/public_candidates.jsonl`
- evaluator_mapping: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/candidates/evaluator_mapping.jsonl`
- shadow_masks: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/pythia_410m/candidates/shadow_masks.csv`
- wandb_attack_artifact: `y3rp58g2-attack:bf9b170f100cd42df2a2950e4a4e65ad`

## Metrics

- population_calibration_examples: `1024.0`
- threshold_at_population_fpr_0.01: `0.9980449657869013`
- tpr_gold_train_at_population_fpr_0.01: `0.0048828125`
- fpr_fullft_gen_from_squad_train_at_population_fpr_0.01: `0.0`
- fpr_fullft_gen_from_squad_validation_at_population_fpr_0.01: `0.0`
- fpr_fullft_gen_from_trivia_validation_at_population_fpr_0.01: `0.034722222222222224`
- fpr_gold_squad_target_train_at_population_fpr_0.01: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.01: `0.005859375`
- fpr_gold_trivia_validation_at_population_fpr_0.01: `0.008680555555555556`
- threshold_at_population_fpr_0.05: `0.9951124144672532`
- tpr_gold_train_at_population_fpr_0.05: `0.0478515625`
- fpr_fullft_gen_from_squad_train_at_population_fpr_0.05: `0.0`
- fpr_fullft_gen_from_squad_validation_at_population_fpr_0.05: `0.0`
- fpr_fullft_gen_from_trivia_validation_at_population_fpr_0.05: `0.1423611111111111`
- fpr_gold_squad_target_train_at_population_fpr_0.05: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.05: `0.03125`
- fpr_gold_trivia_validation_at_population_fpr_0.05: `0.052083333333333336`
- auc_gold_train_vs_fullft_gen_from_squad_train: `0.5369911193847656`
- auc_ci_lower_gold_train_vs_fullft_gen_from_squad_train: `0.5105652809143066`
- auc_ci_upper_gold_train_vs_fullft_gen_from_squad_train: `0.5613336563110352`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_fullft_gen_from_squad_train: `0.1171875`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_fullft_gen_from_squad_train: `0.25390625`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_fullft_gen_from_squad_train: `0.220703125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_fullft_gen_from_squad_train: `0.3349609375`
- auc_gold_train_vs_fullft_gen_from_squad_validation: `0.5413632392883301`
- auc_ci_lower_gold_train_vs_fullft_gen_from_squad_validation: `0.5152621269226074`
- auc_ci_upper_gold_train_vs_fullft_gen_from_squad_validation: `0.5686593055725098`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_fullft_gen_from_squad_validation: `0.1181640625`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_fullft_gen_from_squad_validation: `0.25390625`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_fullft_gen_from_squad_validation: `0.220703125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_fullft_gen_from_squad_validation: `0.3134765625`
- auc_gold_train_vs_fullft_gen_from_trivia_validation: `0.41598934597439235`
- auc_ci_lower_gold_train_vs_fullft_gen_from_trivia_validation: `0.38671451144748265`
- auc_ci_upper_gold_train_vs_fullft_gen_from_trivia_validation: `0.4446512858072917`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_fullft_gen_from_trivia_validation: `0.0`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_fullft_gen_from_trivia_validation: `0.0029296875`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_fullft_gen_from_trivia_validation: `0.0009765625`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_fullft_gen_from_trivia_validation: `0.0166015625`
- auc_gold_train_vs_gold_squad_validation: `0.6043887138366699`
- auc_ci_lower_gold_train_vs_gold_squad_validation: `0.5790557861328125`
- auc_ci_upper_gold_train_vs_gold_squad_validation: `0.6285452842712402`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_squad_validation: `0.0009765625`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_squad_validation: `0.01953125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_squad_validation: `0.0517578125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_squad_validation: `0.107421875`
- auc_gold_train_vs_gold_trivia_validation: `0.7384406195746528`
- auc_ci_lower_gold_train_vs_gold_trivia_validation: `0.7099032931857638`
- auc_ci_upper_gold_train_vs_gold_trivia_validation: `0.7676467895507812`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_trivia_validation: `0.0`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_trivia_validation: `0.0146484375`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_trivia_validation: `0.0078125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_trivia_validation: `0.0751953125`
- num_candidates: `5248.0`

## Conclusion

Scores used text and shadow masks only; evaluator labels were applied after scoring.

## Achieved Purpose

yes

## Next Action

Compare model families and true membership-conditioned distributions.
