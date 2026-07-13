# Experiment Record

## Purpose

Analyze text-only RMIA score distributions for Pythia-410M.

## Hypothesis

Target-trained SQuAD records exhibit different target/reference score distributions than IID, OOD, and generated non-members.

## Command

`/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/.venv/bin/python3 ./cli/squad_lora_rmia.py runtime.seed=42 workflow.stage=formal workflow.profile=formal llm_model=pythia_410m train.per_device_train_batch_size=16 train.gradient_accumulation_steps=2 train.dataloader_num_workers=4 eval.batch_size=96 eval.generation_batch_size=96 generation.batch_size=96 attack.batch_size=96`

## Hydra Overrides


## Fingerprint

- Config SHA256: `f893a0509a57865d79abb171131576bd4116f1d0a8496c36238bad0c2a006fd7`

## Reproducible Source Snapshot

- Branch: `codex/squad-lora-rmia`
- Plan commit: `b31db23342202ab001f9fe53bbbb16c0440dcf69`
- Source snapshot commit: `6833dafb1a5814919a9a44fb47673fdb5b0aa57e`
- Execution base commit: `6447cce3dcf07d2797e9a41355b966732611d4f0` with the dirty worktree recorded below.
- Provenance note: the source snapshot was committed retrospectively from the recorded execution worktree; the original execution source state is preserved verbatim.

## Execution Source State

```json
{
  "branch": "master",
  "commit": "6447cce3dcf07d2797e9a41355b966732611d4f0",
  "short_commit": "6447cce",
  "status": "M .gitignore\n M README.md\n M pyproject.toml\n M src/attacker/lira.py\n M src/attacker/rmia.py\n M src/dataset/torchvision_dataset.py\n M src/experiment/records.py\n M src/model/torchvision_model.py\n M src/utils/config_class.py\n M uv.lock\n?? cli/debug_squad_lora_gpu.py\n?? cli/squad_lora_rmia.py\n?? conf/llm_model/\n?? conf/squad_lora_rmia.yaml\n?? references/\n?? scripts/\n?? src/llm_mia/\n?? tests/test_llm_mia_analysis.py\n?? tests/test_llm_mia_data.py\n?? tests/test_llm_mia_scoring.py\n?? tests/test_llm_mia_tokenization.py\n?? tests/test_llm_mia_tracking.py"
}
```

## Environment

```json
{
  "accelerate": "1.14.0",
  "cuda": "13.0",
  "cuda_available": true,
  "datasets": "5.0.0",
  "gpu": "NVIDIA A800-SXM4-80GB",
  "gpu_driver": "590.48.01",
  "peft": "0.19.1",
  "platform": "Linux-5.14.0-570.17.1.el9_6.x86_64-x86_64-with-glibc2.35",
  "python": "3.13.13 (main, Jun  2 2026, 22:27:49) [Clang 22.1.3 ]",
  "torch": "2.13.0+cu130",
  "transformers": "5.13.0",
  "uv": "uv 0.11.28 (x86_64-unknown-linux-gnu)",
  "uv_lock_sha256": "83924a9093237f4b7d0d05dc1668e91f08e06eb46d32aa0656de569a9562bdcc"
}
```

## W&B

- Run ID: `ihwkse5b`

## Artifacts

- scores: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/attack/online_rmia_scores.jsonl`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/attack/metrics.json`
- public_candidates: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/candidates/public_candidates.jsonl`
- shadow_masks: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/candidates/shadow_masks.csv`
- wandb_attack_artifact: `ihwkse5b-attack:ac93a811bee2a2c27ae8230e0d2685c5`

## Metrics

- population_calibration_examples: `1024.0`
- threshold_at_population_fpr_0.01: `0.9980449657869013`
- tpr_gold_train_at_population_fpr_0.01: `0.0263671875`
- fpr_gen_from_squad_train_at_population_fpr_0.01: `0.0`
- fpr_gen_from_squad_validation_at_population_fpr_0.01: `0.0`
- fpr_gen_from_trivia_validation_at_population_fpr_0.01: `0.07465277777777778`
- fpr_gold_squad_target_train_at_population_fpr_0.01: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.01: `0.015625`
- fpr_gold_trivia_validation_at_population_fpr_0.01: `0.024305555555555556`
- threshold_at_population_fpr_0.05: `0.9931573802541545`
- tpr_gold_train_at_population_fpr_0.05: `0.076171875`
- fpr_gen_from_squad_train_at_population_fpr_0.05: `0.0`
- fpr_gen_from_squad_validation_at_population_fpr_0.05: `0.0`
- fpr_gen_from_trivia_validation_at_population_fpr_0.05: `0.203125`
- fpr_gold_squad_target_train_at_population_fpr_0.05: `nan`
- fpr_gold_squad_validation_at_population_fpr_0.05: `0.046875`
- fpr_gold_trivia_validation_at_population_fpr_0.05: `0.06770833333333333`
- auc_gold_train_vs_gen_from_squad_train: `0.5758490562438965`
- auc_ci_lower_gold_train_vs_gen_from_squad_train: `0.5510697364807129`
- auc_ci_upper_gold_train_vs_gen_from_squad_train: `0.6015048027038574`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gen_from_squad_train: `0.1435546875`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gen_from_squad_train: `0.1875`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gen_from_squad_train: `0.2255859375`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gen_from_squad_train: `0.3056640625`
- auc_gold_train_vs_gen_from_squad_validation: `0.5812273025512695`
- auc_ci_lower_gold_train_vs_gen_from_squad_validation: `0.5552835464477539`
- auc_ci_upper_gold_train_vs_gen_from_squad_validation: `0.6057615280151367`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gen_from_squad_validation: `0.142578125`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gen_from_squad_validation: `0.1904296875`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gen_from_squad_validation: `0.232421875`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gen_from_squad_validation: `0.306640625`
- auc_gold_train_vs_gen_from_trivia_validation: `0.38331943088107634`
- auc_ci_lower_gold_train_vs_gen_from_trivia_validation: `0.3527679443359375`
- auc_ci_upper_gold_train_vs_gen_from_trivia_validation: `0.41157616509331596`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gen_from_trivia_validation: `0.0`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gen_from_trivia_validation: `0.0078125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gen_from_trivia_validation: `0.0048828125`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gen_from_trivia_validation: `0.0205078125`
- auc_gold_train_vs_gold_squad_validation: `0.5702090263366699`
- auc_ci_lower_gold_train_vs_gold_squad_validation: `0.5455794334411621`
- auc_ci_upper_gold_train_vs_gold_squad_validation: `0.5961875915527344`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_squad_validation: `0.0029296875`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_squad_validation: `0.03125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_squad_validation: `0.037109375`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_squad_validation: `0.0908203125`
- auc_gold_train_vs_gold_trivia_validation: `0.7280773586697048`
- auc_ci_lower_gold_train_vs_gold_trivia_validation: `0.699511210123698`
- auc_ci_upper_gold_train_vs_gold_trivia_validation: `0.7584186130099826`
- tpr_at_fpr_0.01_ci_lower_gold_train_vs_gold_trivia_validation: `0.001953125`
- tpr_at_fpr_0.01_ci_upper_gold_train_vs_gold_trivia_validation: `0.017578125`
- tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_trivia_validation: `0.0244140625`
- tpr_at_fpr_0.05_ci_upper_gold_train_vs_gold_trivia_validation: `0.0732421875`
- num_candidates: `5248.0`

## Conclusion

Scores used text and shadow masks only; evaluator labels were applied after scoring.

## Achieved Purpose

yes

## Next Action

Compare model families and true membership-conditioned distributions.
