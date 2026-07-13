# Experiment Record

## Purpose

Generate reusable target-model candidate completions for Pythia-410M.

## Hypothesis

Generated records will mostly be non-members unless prompt and generated completion exactly reconstruct a target training record.

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

- Run ID: `qgdul778`

## Artifacts

- public_generated: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/generated/public_generated.jsonl`
- private_generated_labels: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/generated/private_generated_labels.jsonl`
- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/generated/metrics.json`
- wandb_generated_artifact: `qgdul778-generated:42f7da6e56312c01f982ad262e0e7632`

## Metrics

- gen_from_squad_train/exact_match: `0.0`
- gen_from_squad_train/f1: `0.10879797926945271`
- gen_from_squad_train/duplicate_rate: `0.0`
- gen_from_squad_train/exact_reconstruction_rate: `0.0`
- gen_from_squad_train/completion_loss: `0.06700080762108368`
- gen_from_squad_train/completion_perplexity: `1.0692963417605217`
- gen_from_squad_train/completion_tokens: `65466.0`
- gen_from_squad_train/examples: `1024.0`
- gen_from_squad_train/generated_completion_tokens_mean: `63.931640625`
- gen_from_squad_train/generated_completion_tokens_min: `62.0`
- gen_from_squad_train/generated_completion_tokens_max: `64.0`
- gen_from_squad_validation/exact_match: `0.0`
- gen_from_squad_validation/f1: `0.12547291972828375`
- gen_from_squad_validation/duplicate_rate: `0.0`
- gen_from_squad_validation/exact_reconstruction_rate: `0.0`
- gen_from_squad_validation/completion_loss: `0.06926039089073609`
- gen_from_squad_validation/completion_perplexity: `1.0717152377038837`
- gen_from_squad_validation/completion_tokens: `65466.0`
- gen_from_squad_validation/examples: `1024.0`
- gen_from_squad_validation/generated_completion_tokens_mean: `63.931640625`
- gen_from_squad_validation/generated_completion_tokens_min: `60.0`
- gen_from_squad_validation/generated_completion_tokens_max: `64.0`
- gen_from_trivia_validation/exact_match: `0.0`
- gen_from_trivia_validation/f1: `0.0661159281272204`
- gen_from_trivia_validation/duplicate_rate: `0.4375`
- gen_from_trivia_validation/exact_reconstruction_rate: `0.0`
- gen_from_trivia_validation/completion_loss: `0.16203037401658446`
- gen_from_trivia_validation/completion_perplexity: `1.175895957461889`
- gen_from_trivia_validation/completion_tokens: `12507.0`
- gen_from_trivia_validation/examples: `1024.0`
- gen_from_trivia_validation/generated_completion_tokens_mean: `63.9228515625`
- gen_from_trivia_validation/generated_completion_tokens_min: `62.0`
- gen_from_trivia_validation/generated_completion_tokens_max: `64.0`

## Conclusion

Generated artifacts and private provenance labels were saved for reuse.

## Achieved Purpose

yes

## Next Action

Build public attack candidates and shadow inclusion masks.
