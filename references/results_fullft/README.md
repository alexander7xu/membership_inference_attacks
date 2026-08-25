# Full Fine-Tuning Formal Results

This package contains the lightweight formal records for the completed
Pythia-410M and OLMo-1B-hf SQuAD full fine-tuning and online RMIA runs.

- Plan and tracked execution source: `c3d3cd79a4b09aa12187ec462e1776033647b506`
- Target models: 2
- Shadow models: 10 total, 5 per model family
- Copied source artifacts: 142
- Copied source bytes: 921867
- Final acceptance: 192 recorded hashes verified across 20,234,754,987 bytes;
  no validation errors

The per-model `provenance.json` files bind each committed record to its original
ignored output. Large models, row-level generated/candidate/attack data, W&B
local state, datasets, and caches remain uncommitted.
