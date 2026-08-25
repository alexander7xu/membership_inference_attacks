# Prompt-Unique Gold-IID Control and Target-Generated RMIA

## Scope

This correction preserves the 1-epoch, five-shadow online RMIA design while making the five gold-IID source slices globally unique by prompt. Existing formal gold-IID artifacts remain immutable.

## Source Replacements

Two duplicated prompts in `gold_squad_validation_04` are replaced at their original positions:

- `5726e65e708984140094d53e`, duplicated by `5726eb76f1498d1400e8efdb` in slice 01.
- `56d98fbfdc89441400fdb563`, duplicated by `56d6f0770d65d21400198269` in slice 03.

Replacement records are selected from the pinned SQuAD validation revision with seed 42 and stable source-ID hashing. Target training, population, auxiliary filler, all existing candidate source IDs, prompt hashes, and content hashes are excluded. Each replacement inherits the removed candidate's five-bit mask.

## Matched Arms

The prompt-unique gold-IID control is written to `outputs/squad_lora_rmia_iid_prompt_unique_control`. It retains 5,120 non-members, 6,144 total candidates, zero canonicalization loss, and IN counts `[3001, 3088, 3064, 3051, 3071]`.

The target-generated arm is written to `outputs/squad_lora_rmia_target_generated_prompt_unique_ablation`. It reuses the verified control target, evaluation, population, source IDs, and masks, then generates one deterministic completion per unique prompt with the model-specific target adapter.

Both arms retrain all five shadows. The target-generated arm starts only after the matching control job succeeds. Pythia uses `xe8545`; OLMo uses `tmp`.

## Acceptance

- 5,120 globally unique non-member prompts and 6,144 unique canonical candidates per model.
- Replacement provenance binds old/new source IDs, positions, prompt/content hashes, inherited masks, selection rule, exclusion-set hashes, and source artifact hashes.
- Five complete 43,799-record shadows per arm, with no target, candidate, population, or OUT-candidate filler leakage.
- Complete attack, comparison, plot, offline W&B, completion manifest, and `_SUCCESS` artifacts with all SHA256 checks passing.