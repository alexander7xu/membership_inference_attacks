# 1-Epoch RMIA Target-Generated Distribution Ablation

## Objective

Measure the effect of replacing gold IID SQuAD non-members with source-aligned completions from the corresponding trained target model. The target adapter, 1,024 target members, five source-prompt slices, masks, population set, shadow filler, optimizer, LoRA settings, and one-epoch training budget remain fixed.

## Candidate Suite

Each model has 6,144 candidates:

- 1,024 byte-identical `gold_squad_target_train` members from the completed gold-IID experiment.
- Five groups of 1,024 target-generated non-members, one for each gold-IID SQuAD validation slice.

Generation uses the model-specific one-epoch target adapter with greedy decoding (`do_sample=false`, `temperature=0`, `max_new_tokens=64`). Every generated record binds its source ID, prompt hash, gold control candidate ID, target checkpoint hash, and generation configuration. Generated answers that exactly match the gold answer are retained and reported through EM/F1; exact target-training reconstruction and forbidden content overlap must remain zero.

## Masks And Shadow Training

Masks are inherited by `(slice index, source_id)` from the completed gold-IID experiment. Target-member IDs and masks are unchanged. The candidate suite must contain 6,144 raw and canonical rows with no conflicts, and per-shadow IN counts must be `[3001, 3088, 3064, 3051, 3071]`.

Each model trains five new one-epoch shadows with 43,799 unique records. Auxiliary SQuAD filler remains unchanged from the gold-IID control, including row order, with counts `[40798, 40711, 40735, 40748, 40728]`. Target, candidate, population, and OUT-candidate leakage must be zero.

## Attack And Primary Comparison

Online RMIA keeps gamma 0.5, the original 1,024-record population, and 2,000 bootstrap replicates. Metrics are reported for each generated slice and for all 5,120 generated non-members pooled.

The primary comparison aligns target members by candidate ID and non-members by `(slice, source_id)`. Generated-minus-gold AUC and TPR differences use identical bootstrap resampling indices in both arms. Pythia and OLMo use their own target adapters, so cross-model comparisons are descriptive.

## Workflow And Scheduling

The formal stages are:

1. `prepare_shadow_reuse`
2. `generate`
3. `build_candidates`
4. `train_shadows`
5. `attack`
6. `compare_gold_iid`
7. `plot_rmia_feature`
8. `validate`

Pythia runs on `xe8545`; OLMo runs on the 80 GB `tmp` partition. The two jobs are independent and may run concurrently. Scoring, comparison, plotting, and final validation begin only after all five shadows for that model are complete.

## Acceptance Criteria

Both models must finish five shadows and produce 6,144 finite RMIA scores, five slice metrics, pooled metrics, paired comparison intervals, an ECDF, offline W&B records, a completion manifest, and `_SUCCESS`. Final validation rechecks source completion, target/evaluation trees, population, source prompts, generation provenance, masks, exact filler order, checkpoint completion, and all SHA256 hashes.
