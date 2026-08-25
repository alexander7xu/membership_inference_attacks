# Pythia-410M Formal Results

This directory is the lightweight, reviewable record for the completed
Pythia-410M SQuAD LoRA and online RMIA experiment.

## Provenance

- Experiment plan: `b31db23342202ab001f9fe53bbbb16c0440dcf69`
- Reproducible source snapshot: `6833dafb1a5814919a9a44fb47673fdb5b0aa57e`
- Original execution base: `6447cce3dcf07d2797e9a41355b966732611d4f0` plus the dirty worktree
  preserved in each experiment document.

The source snapshot is retrospective: it captures the experiment implementation
that was present in the recorded dirty worktree. It does not rewrite or conceal
the original execution state. `provenance.json` stores SHA256 digests of every
original lightweight record copied into this package.

## Scope

Committed records include resolved Hydra configurations, aggregate metrics,
artifact manifests, five shadow-model summaries, and experiment conclusions.
Large or sensitive artifacts remain under the ignored `outputs/` tree and are
identified by the committed manifests and original run records.

The two `fpr_gold_squad_target_train_*` values in `attack/metrics.json` are
non-finite because false-positive rate is not defined for the positive member
group itself. The corresponding member statistic is TPR.
