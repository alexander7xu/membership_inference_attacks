# Experiment Record

## Purpose

Evaluate base OLMo-1B-hf on SQuAD IID and TriviaQA OOD QA.

## Hypothesis

Final full-parameter fine-tuning should reduce completion loss and improve QA EM/F1 versus the base model.

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

- metrics: `/hpc2hdd/home/ytang740/LHB/membership_inference_attacks/outputs/squad_fullft_rmia/formal/olmo_1b_hf/eval/base/metrics.json`
- checkpoint: `base-model`
- wandb_eval_base_artifact: `6ay0yitf-eval_base:6e0a83f916114d3ef01d6e44a9d0b94a`

## Metrics

- squad_validation/completion_loss: `1.262316715221767`
- squad_validation/completion_perplexity: `3.5335983530657047`
- squad_validation/completion_tokens: `43806.0`
- squad_validation/examples: `10570.0`
- squad_validation/generation_exact_match: `0.0`
- squad_validation/generation_f1: `0.0075952025250757`
- squad_validation/generation_examples: `10570.0`
- trivia_validation/completion_loss: `1.8690898008391328`
- trivia_validation/completion_perplexity: `6.482393444188543`
- trivia_validation/completion_tokens: `57515.0`
- trivia_validation/examples: `17944.0`
- trivia_validation/generation_exact_match: `0.0`
- trivia_validation/generation_f1: `0.006477351724183613`
- trivia_validation/generation_examples: `17944.0`

## Conclusion

Evaluation completed without checkpoint selection.

## Achieved Purpose

yes

## Next Action

Use the final target checkpoint for generation and RMIA scoring.
