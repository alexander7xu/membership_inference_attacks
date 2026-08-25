# OLMo-1B-hf Formal Results

This directory is the lightweight, reviewable record for the completed
OLMo-1B-hf SQuAD LoRA and online RMIA experiment.

## Provenance

- Experiment plan: `b31db23342202ab001f9fe53bbbb16c0440dcf69`
- Reproducible source snapshot: `c0c177d27cbb48f4fbb9aa3e9c0fb0b98782a75d`
- Target-training execution base: `6447cce3dcf07d2797e9a41355b966732611d4f0` plus its dirty
  `master` worktree.
- Downstream execution base: `3f8e7030098d0310c34efb1c74f3e7574da8ac4e` plus its dirty
  `codex/squad-lora-rmia` worktree for evaluation, generation, shadows, and attack.

The source snapshot is retrospective: it captures the OLMo experiment files
that were present in the recorded dirty worktree. It does not rewrite or conceal
the original execution state. `provenance.json` stores SHA256 digests of every
original lightweight record copied into this package.

The saved target adapter was trained under Torch 2.13.0+cu130 and reused for
downstream stages under Torch 2.13.0+cu126. Exact package, CUDA, driver, and
lockfile versions remain in the corresponding experiment documents.

## Scope

Committed records include resolved Hydra configurations, aggregate metrics,
artifact manifests, five shadow-model summaries, and experiment conclusions.
Large or sensitive artifacts remain under the ignored `outputs/` tree and are
identified by the committed manifests and original run records.

The two `fpr_gold_squad_target_train_*` values in `attack/metrics.json` are
non-finite because false-positive rate is not defined for the positive member
group itself. The corresponding member statistic is TPR.

Generation exact match is zero and token F1 is very low for both base and target
evaluations. Completion loss and perplexity improved after LoRA fine-tuning, so
generation quality and completion likelihood must be interpreted separately.
