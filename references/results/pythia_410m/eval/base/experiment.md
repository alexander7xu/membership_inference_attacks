# Experiment Record

## Purpose

Evaluate base Pythia-410M on SQuAD IID and TriviaQA OOD QA.

## Hypothesis

Final LoRA should reduce completion loss and improve QA EM/F1 versus the base model.

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

- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/pythia_410m/eval/base/metrics.json`
- adapter: `base-model`
- wandb_eval_base_artifact: `qgdul778-eval_base:ef86f40d335d53418860a87fe3b73c3a`

## Metrics

- squad_validation/completion_loss: `3.2950899553468758`
- squad_validation/completion_perplexity: `26.979840941693695`
- squad_validation/completion_tokens: `43806.0`
- squad_validation/examples: `10570.0`
- squad_validation/generation_exact_match: `0.0`
- squad_validation/generation_f1: `0.06372630696167875`
- squad_validation/generation_examples: `10570.0`
- trivia_validation/completion_loss: `4.269746588400465`
- trivia_validation/completion_perplexity: `71.50351350340975`
- trivia_validation/completion_tokens: `57515.0`
- trivia_validation/examples: `17944.0`
- trivia_validation/generation_exact_match: `5.572893446277307e-05`
- trivia_validation/generation_f1: `0.05381532232598053`
- trivia_validation/generation_examples: `17944.0`

## Conclusion

Evaluation completed without checkpoint selection.

## Achieved Purpose

yes

## Next Action

Use final target adapter for generated candidates and RMIA scoring.
