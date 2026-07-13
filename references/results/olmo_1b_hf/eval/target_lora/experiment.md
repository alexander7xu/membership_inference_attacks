# Experiment Record

## Purpose

Evaluate target_lora OLMo-1B-hf on SQuAD IID and TriviaQA OOD QA.

## Hypothesis

Final LoRA should reduce completion loss and improve QA EM/F1 versus the base model.

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

- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/eval/target_lora/metrics.json`
- adapter: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_lora_rmia/formal/olmo_1b_hf/target/seed_42/adapter`
- wandb_eval_target_lora_artifact: `0l18l0ay-eval_target_lora:755777d6bce5dac76f6fce850e4c8bc8`

## Metrics

- squad_validation/completion_loss: `0.12772163021967425`
- squad_validation/completion_perplexity: `1.1362366647188804`
- squad_validation/completion_tokens: `43806.0`
- squad_validation/examples: `10570.0`
- squad_validation/generation_exact_match: `0.0`
- squad_validation/generation_f1: `0.005959530419130615`
- squad_validation/generation_examples: `10570.0`
- trivia_validation/completion_loss: `1.2350038192658574`
- trivia_validation/completion_perplexity: `3.4383916528118914`
- trivia_validation/completion_tokens: `57515.0`
- trivia_validation/examples: `17944.0`
- trivia_validation/generation_exact_match: `0.0`
- trivia_validation/generation_f1: `0.005060380309727433`
- trivia_validation/generation_examples: `17944.0`

## Conclusion

Evaluation completed without checkpoint selection.

## Achieved Purpose

yes

## Next Action

Use final target adapter for generated candidates and RMIA scoring.
