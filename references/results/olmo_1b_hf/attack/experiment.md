# Experiment Record

## Purpose

Analyze text-only RMIA score distributions for OLMo-1B-hf.

## Hypothesis

Target-trained SQuAD records exhibit different target/reference score distributions than IID, OOD, and generated non-members.

## Command

`/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/.venv/bin/python3 ./cli/squad_lora_rmia.py runtime.seed=42 workflow.stage=formal workflow.profile=formal llm_model=olmo_1b_hf train.per_device_train_batch_size=16 train.gradient_accumulation_steps=2 train.dataloader_num_workers=4 tokenizer.preprocessing_workers=8 eval.batch_size=96 eval.loss_chunk_tokens=64 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96 attack.loss_chunk_tokens=64`

## Hydra Overrides


## Fingerprint

- Config SHA256: `9117ab94135600b9bf610bcda34aa37507469c41b9c4d32b069c6fa6e97dcda7`

## Reproducible Source Snapshot

- Branch: `codex/squad-lora-rmia`
- Plan commit: `b31db23342202ab001f9fe53bbbb16c0440dcf69`
- Source snapshot commit: `c0c177d27cbb48f4fbb9aa3e9c0fb0b98782a75d`
- Execution base commit: `3f8e7030098d0310c34efb1c74f3e7574da8ac4e` with the dirty worktree recorded below.
- Provenance note: the source snapshot was committed retrospectively from the recorded execution worktree; the original execution source state is preserved verbatim.

## Execution Source State

```json
{
  "branch": "codex/squad-lora-rmia",
  "commit": "3f8e7030098d0310c34efb1c74f3e7574da8ac4e",
  "short_commit": "3f8e703",
  "status": "M README.md\n M scripts/run_squad_lora_rmia_formal.sh\n?? cli/debug_squad_lora_gpu.py\n?? conf/llm_model/olmo_1b_hf.yaml\n?? scripts/debug_gpu_utilization_suite.sh\n?? scripts/debug_inference_gpu_utilization.sh\n?? scripts/debug_training_gpu_utilization.sh"
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

- Run ID: `0l18l0ay`

## Artifacts

- scores: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/attack/online_rmia_scores.jsonl`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/attack/metrics.json`
- public_candidates: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/candidates/public_candidates.jsonl`
- shadow_masks: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/candidates/shadow_masks.csv`
- wandb_attack_artifact: `0l18l0ay-attack:c03b2b433536b11d64c5dc9713e7146c`

## Metrics

- population_calibration_examples: `1024.0`
- threshold_at_population_fpr_0.01: `0.9980449657869013`
- tpr_gold_train_at_population_fpr_0.01: `0.0908203125`
- fpr_gen_from_squad_train_at_population_fpr_0.01: `0.0`
- fpr_gen_from_squad_validation_at_population_fpr_0.01: `0.0`
- fpr_gen_from_trivia_validation_at_population_fpr_0.01: `0.0`
- fpr_gold_squad_target_train_at_population_fpr_0.01: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.01: `0.0390625`
- fpr_gold_trivia_validation_at_population_fpr_0.01: `0.03993055555555555`
- threshold_at_population_fpr_0.05: `0.9980449657869013`
- tpr_gold_train_at_population_fpr_0.05: `0.0908203125`
- fpr_gen_from_squad_train_at_population_fpr_0.05: `0.0`
- fpr_gen_from_squad_validation_at_population_fpr_0.05: `0.0`
- fpr_gen_from_trivia_validation_at_population_fpr_0.05: `0.0`
- fpr_gold_squad_target_train_at_population_fpr_0.05: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.05: `0.0390625`
- fpr_gold_trivia_validation_at_population_fpr_0.05: `0.03993055555555555`
- auc_gold_train_vs_gen_from_squad_train: `0.999415397644043`
- auc_ci_lower_gold_train_vs_gen_from_squad_train: `0.9981803894042969`
- auc_ci_upper_gold_train_vs_gen_from_squad_train: `1.0`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gen_from_squad_train: `0.9970703125`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gen_from_squad_train: `1.0`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gen_from_squad_train: `0.9970703125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gen_from_squad_train: `1.0`
- auc_gold_train_vs_gen_from_squad_validation: `0.9994268417358398`
- auc_ci_lower_gold_train_vs_gen_from_squad_validation: `0.9982204437255859`
- auc_ci_upper_gold_train_vs_gen_from_squad_validation: `1.0`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gen_from_squad_validation: `0.9970703125`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gen_from_squad_validation: `1.0`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gen_from_squad_validation: `0.9970703125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gen_from_squad_validation: `1.0`
- auc_gold_train_vs_gen_from_trivia_validation: `0.9999237060546875`
- auc_ci_lower_gold_train_vs_gen_from_trivia_validation: `0.9997389051649306`
- auc_ci_upper_gold_train_vs_gen_from_trivia_validation: `1.0`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gen_from_trivia_validation: `0.9970703125`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gen_from_trivia_validation: `1.0`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gen_from_trivia_validation: `0.9970703125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gen_from_trivia_validation: `1.0`
- auc_gold_train_vs_gold_squad_validation: `0.5980129241943359`
- auc_ci_lower_gold_train_vs_gold_squad_validation: `0.5798773765563965`
- auc_ci_upper_gold_train_vs_gold_squad_validation: `0.6166701316833496`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_squad_validation: `0.0068359375`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_squad_validation: `0.0205078125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_squad_validation: `0.0126953125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_squad_validation: `0.1123046875`
- auc_gold_train_vs_gold_trivia_validation: `0.7515394422743056`
- auc_ci_lower_gold_train_vs_gold_trivia_validation: `0.7274754842122396`
- auc_ci_upper_gold_train_vs_gold_trivia_validation: `0.7745641072591146`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_trivia_validation: `0.0`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_trivia_validation: `0.017578125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_trivia_validation: `0.009765625`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_trivia_validation: `0.1103515625`
- num_candidates: `5248.0`

## Conclusion

Scores used text and shadow masks only; evaluator labels were applied after scoring.

## Achieved Purpose

yes

## Next Action

Compare model families and true membership-conditioned distributions.
