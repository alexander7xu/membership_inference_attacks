# Experiment Record

## Purpose

Analyze text-only RMIA score distributions for OLMo-1B-hf.

## Hypothesis

Target-trained SQuAD records exhibit different target/reference score distributions than IID, OOD, and generated non-members.

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

- scores: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/attack/online_rmia_scores.jsonl`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/attack/metrics.json`
- public_candidates: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/candidates/public_candidates.jsonl`
- evaluator_mapping: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/candidates/evaluator_mapping.jsonl`
- shadow_masks: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/candidates/shadow_masks.csv`
- wandb_attack_artifact: `6ay0yitf-attack:b5b88718a7c7760c858b91c92de10af6`

## Metrics

- population_calibration_examples: `1024.0`
- threshold_at_population_fpr_0.01: `0.9990224828934506`
- tpr_gold_train_at_population_fpr_0.01: `0.0625`
- fpr_fullft_gen_from_squad_train_at_population_fpr_0.01: `0.0009765625`
- fpr_fullft_gen_from_squad_validation_at_population_fpr_0.01: `0.0`
- fpr_fullft_gen_from_trivia_validation_at_population_fpr_0.01: `0.001736111111111111`
- fpr_gold_squad_target_train_at_population_fpr_0.01: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.01: `0.0234375`
- fpr_gold_trivia_validation_at_population_fpr_0.01: `0.041666666666666664`
- threshold_at_population_fpr_0.05: `0.9970674486803519`
- tpr_gold_train_at_population_fpr_0.05: `0.126953125`
- fpr_fullft_gen_from_squad_train_at_population_fpr_0.05: `0.001953125`
- fpr_fullft_gen_from_squad_validation_at_population_fpr_0.05: `0.0009765625`
- fpr_fullft_gen_from_trivia_validation_at_population_fpr_0.05: `0.003472222222222222`
- fpr_gold_squad_target_train_at_population_fpr_0.05: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.05: `0.05078125`
- fpr_gold_trivia_validation_at_population_fpr_0.05: `0.06944444444444445`
- auc_gold_train_vs_fullft_gen_from_squad_train: `0.9958028793334961`
- auc_ci_lower_gold_train_vs_fullft_gen_from_squad_train: `0.9916210174560547`
- auc_ci_upper_gold_train_vs_fullft_gen_from_squad_train: `0.9989328384399414`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_fullft_gen_from_squad_train: `0.9912109375`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_fullft_gen_from_squad_train: `0.9990234375`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_fullft_gen_from_squad_train: `0.9912109375`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_fullft_gen_from_squad_train: `0.9990234375`
- auc_gold_train_vs_fullft_gen_from_squad_validation: `0.9953770637512207`
- auc_ci_lower_gold_train_vs_fullft_gen_from_squad_validation: `0.9917025566101074`
- auc_ci_upper_gold_train_vs_fullft_gen_from_squad_validation: `0.9985194206237793`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_fullft_gen_from_squad_validation: `0.98046875`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_fullft_gen_from_squad_validation: `0.9990234375`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_fullft_gen_from_squad_validation: `0.9921875`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_fullft_gen_from_squad_validation: `0.9990234375`
- auc_gold_train_vs_fullft_gen_from_trivia_validation: `0.9948255750868056`
- auc_ci_lower_gold_train_vs_fullft_gen_from_trivia_validation: `0.9890543619791667`
- auc_ci_upper_gold_train_vs_fullft_gen_from_trivia_validation: `0.9990615844726562`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_fullft_gen_from_trivia_validation: `0.1103515625`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_fullft_gen_from_trivia_validation: `0.998046875`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_fullft_gen_from_trivia_validation: `0.9970703125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_fullft_gen_from_trivia_validation: `1.0`
- auc_gold_train_vs_gold_squad_validation: `0.6177496910095215`
- auc_ci_lower_gold_train_vs_gold_squad_validation: `0.5974521636962891`
- auc_ci_upper_gold_train_vs_gold_squad_validation: `0.637047290802002`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_squad_validation: `0.0009765625`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_squad_validation: `0.0078125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_squad_validation: `0.048828125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_squad_validation: `0.14453125`
- auc_gold_train_vs_gold_trivia_validation: `0.7657106187608508`
- auc_ci_lower_gold_train_vs_gold_trivia_validation: `0.7411617702907985`
- auc_ci_upper_gold_train_vs_gold_trivia_validation: `0.7899848090277778`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_trivia_validation: `0.0`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_trivia_validation: `0.0078125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_trivia_validation: `0.001953125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_trivia_validation: `0.0810546875`
- num_candidates: `5248.0`

## Conclusion

Scores used text and shadow masks only; evaluator labels were applied after scoring.

## Achieved Purpose

yes

## Next Action

Compare model families and true membership-conditioned distributions.
