# OLMo-1B-hf Full-FT Formal Results

This directory is the lightweight, reviewable record for the completed
OLMo-1B-hf SQuAD full fine-tuning and online RMIA experiment.

## Provenance

- Experiment plan: `references/squad_fullft_rmia_experiment_plan.md`
- Tracked source commit: `c3d3cd79a4b09aa12187ec462e1776033647b506`
- Original output: `outputs/squad_fullft_rmia/formal/olmo_1b_hf`

The recorded source status contains four pre-existing untracked GPU debug scripts. They are not part of this result snapshot; the tracked source matched `c3d3cd79a4b09aa12187ec462e1776033647b506`.

## Scope

The snapshot includes resolved Hydra configurations, aggregate metrics,
artifact manifests, target and five-shadow training summaries, model-weight
digests and success markers, trainer states, base-generation reuse evidence,
online RMIA records, and the nine-group relative-log-likelihood figure.

Large or row-level artifacts remain in the ignored `outputs/` tree. Full-model
weights, generated text, candidate tables, attack score rows, W&B local state,
datasets, and caches are deliberately not committed. `provenance.json` records
the original path, byte size, and SHA-256 digest of every copied artifact.
