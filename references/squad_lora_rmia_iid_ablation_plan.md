# SQuAD LoRA RMIA IID Shadow-Distribution Ablation

## Question

Does replacing the mixed shadow candidate distribution with the target task's
gold SQuAD QA distribution change five-shadow online RMIA performance?

## Fixed factors

Both model families reuse the verified 1-epoch target adapter and utility
evaluation. LoRA rank, optimizer, learning rate, batch size, gradient
accumulation, BF16 precision, shadow count, shadow train size, mask seed,
population calibration, RMIA gamma, and bootstrap procedure remain unchanged.

## Treatment

The candidate suite has 6,144 unique gold SQuAD records: 1,024 target members
and five disjoint 1,024-record SQuAD validation non-member slices. Slice 00 is
the exact IID slice used by the safe mixed-distribution control. The other four
slices are selected deterministically from the pinned SQuAD validation revision.
All non-members are disjoint from the full target train, population, and each
other by source ID and model-input content hash.

Each candidate receives the original conditioned Bernoulli(0.5) five-shadow
mask under seed 1042. Shared members and slice 00 retain the control candidate
IDs and masks exactly. Every shadow contains 43,799 unique records. Auxiliary
fillers exclude the full target train, all candidates, and the population, so an
OUT candidate cannot enter through filler sampling.

## Control

The completed safe 1-epoch LiRA shadow adapters are verified against the
historical mixed candidate tables and masks, then rescored with online RMIA.
The control is written under the ablation output root and never modifies LiRA
artifacts. The primary comparison is restricted to the shared 1,024 members,
1,024 IID non-members, masks, target adapter, and population.

## Reporting

Report ROC-AUC and TPR at 1% and 5% FPR with 2,000-bootstrap 95% intervals for
each treatment slice and for the pooled 5,120 IID non-members. Report the
shared-slice treatment-control comparison as the primary estimate. Historical
unsafe RMIA results are auxiliary context only.

## Execution and acceptance

Pythia-410M runs on `xe8545`; OLMo-1B-hf runs on `tmp`. Both jobs use separate
model roots and logs. Completion requires five verified 1-epoch shadow adapters,
6,144 finite treatment scores, 1,024 population scores, safe control outputs,
dynamic plots, offline W&B records, completion manifests, `_SUCCESS`, and all
lineage and SHA256 checks.
