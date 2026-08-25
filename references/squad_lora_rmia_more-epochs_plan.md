# SQuAD LoRA RMIA Plan with 10-Epoch Target and Shadow Fine-Tuning

## Objective

Repeat the formal experiment in `references/squad_lora_rmia_experiment_plan.md`
with an intended training-length change: train every target LoRA adapter and every
shadow LoRA adapter for 10 epochs instead of 1 epoch. Target training also records
loss-only validation evidence at every epoch boundary so the longer run can be
inspected without selecting a checkpoint. Candidate-to-shadow masks are inherited
from the verified baseline. The current target-overlap-free filler policy is kept;
historical filler rows are not reproduced.

The experiment tests how longer fine-tuning changes QA utility, generated text,
completion-score distributions, and membership-conditioned RMIA results. It does
not assume that 10 epochs must improve utility or attack performance.

## Claim Boundary

The claim boundary from the source plan is unchanged:

- The language model receives only prompt and completion token IDs.
- Candidate IDs may be used only to join candidates to shadow masks.
- Group, provenance, membership, and row order remain evaluator-only metadata.
- Results are membership-conditioned score distributions and, for the specified
  population-calibrated statistic, auditor-capability RMIA results.
- This is not a label-blind deployable-attacker benchmark.

The verified 1-epoch run and this 10-epoch run form a descriptive comparison,
not a pure epoch-only causal contrast. Mask assignments are matched, but 36--41
historical filler records per shadow overlap target-training content and are replaced
by the current target-overlap-free policy. Conclusions must state this limitation.

## Pre-Registered Hypotheses

1. Ten epochs may increase the separation between target-training records and
   non-members because the adapters receive more updates on the same records.
2. Longer training may improve completion loss and QA metrics initially, but may
   also increase overfitting or reduce OOD utility.
3. Target-generated completions may change in quality, length, duplication, and
   exact-reconstruction rate, which can alter the generated candidate groups.
4. RMIA ROC-AUC or low-FPR TPR may increase, decrease, or remain unchanged; no
   attack improvement is assumed in advance.

## Controlled Configuration

| Area | Fixed setting |
| --- | --- |
| Target families | `EleutherAI/pythia-410m`; `allenai/OLMo-1B-hf` |
| Target runs | One freshly trained target LoRA adapter per model family |
| Shadow runs | Exactly five freshly trained LoRA shadow adapters per model family |
| Task | Completion-only causal next-token prediction on SQuAD QA |
| Prompt and labels | Same Context/Question/Answer format and completion-only masking |
| Maximum sequence length | 1024 |
| Precision | bf16 |
| LoRA | `target_modules=all-linear`, rank 16, alpha 32, dropout 0.05 |
| Learning rate | `1e-4` |
| Training epochs | **10 for every target and shadow adapter** |
| Target epoch evaluation | After every epoch, completion-only loss/perplexity separately on all configured SQuAD and TriviaQA validation records |
| Shadow epoch evaluation | None; no validation datasets or evaluation calls during shadow training |
| Batch settings | Same train, scoring, generation, and gradient-accumulation settings as the 1-epoch formal config |
| Optimizer and scheduler | Same types and hyperparameters as the 1-epoch formal config |
| Checkpoint choice | Final epoch only; no intermediate adapter save, best-checkpoint, or membership-result selection |
| Data | Same pinned SQuAD and TriviaQA revisions, split rules, row IDs, and hashes |
| Candidates | Same six 1,024-record groups and deterministic group-major order |
| Population | Same disjoint 1,024-row SQuAD validation calibration population |
| Shadow masks | Exact five-column assignments inherited from the verified baseline through validated gold IDs and generated-source alignment |
| Shadow filler | Deterministic target-overlap-free filler; not byte-identical to the historical 1-epoch filler |
| RMIA | `gamma=0.5`, logsumexp aggregation, 2,000 deterministic bootstrap resamples |
| Tracking | Hydra with `hydra.job.chdir=false`, project-root paths, mandatory seed, W&B offline |

Do not rescale the learning rate, alter warmup or scheduler hyperparameters, change
batch size, or change gradient accumulation to compensate for the longer run. The
number of optimizer updates and scheduler steps may increase only as the derived
consequence of changing the epoch count from 1 to 10. The recorded filler correction
is an explicit compatibility exception and limits causal interpretation.

## Output Isolation and Baseline Protection

Write the new experiment under the independent output root:

`outputs/squad_lora_rmia_more_epochs`

Treat `outputs/squad_lora_rmia/formal` as a read-only baseline. Never overwrite,
resume into, or append 10-epoch artifacts to the 1-epoch output tree.

The new run must record the immutable digests of the baseline configs, manifests,
metrics, and reports used for comparison. A compatibility report must show that
the resolved 10-epoch configuration differs from the baseline only in:

- the epoch value (`1` to `10`);
- experiment/run names and independent output paths;
- the mask-reuse dependency declaration and target-overlap-free filler evidence;
- derived total-step, scheduler-step, checkpoint-step, and progress counters;
- timestamps, process identifiers, hardware observations, and artifact digests
  that necessarily belong to the new run.

Any other semantic config difference is a hard failure unless this plan is
explicitly revised before formal execution.

## Reuse Contract

Reuse only inputs whose semantics are independent of target or shadow training
length and whose recorded hashes match:

- pinned model, tokenizer, and dataset revisions;
- SQuAD target-training and attacker-auxiliary split manifests;
- SQuAD candidate and calibration-population row IDs;
- TriviaQA validation candidate row IDs;
- materialized gold QA records and their content hashes;
- deterministic generation source prompts;
- the baseline resolved configs and reports as read-only comparison evidence;
- base-model evaluation and base-model generation artifacts, only when their full
  dependency manifests match because the base model is unchanged.

Regenerate every artifact that depends on a fine-tuned adapter:

- target adapters, target trainer state, metrics, and experiment records;
- target-LoRA utility evaluation;
- target-LoRA generated completions, generation metrics, and provenance records;
- the complete six-group candidate suite and its manifest;
- all five shadow adapters and shadow training records;
- target, shadow, population, and RMIA scores;
- attack metrics, figures, reports, completion manifests, and success markers.

Do not continue training from the 1-epoch target or shadow adapters. Every
10-epoch target and shadow run starts from the same pinned base model used by the
source experiment.

## Target Training Protocol

For each model family:

1. Verify the pinned base-model and tokenizer revisions and the target-training
   manifest against the source formal run.
2. Invoke the same Hydra training entry point and the same target-training
   overrides, changing only the experiment/output identifiers and the epoch value
   to 10.
3. Initialize the LoRA adapter from the pinned base model; do not load the
   1-epoch adapter as an initialization or checkpoint.
4. Train over the same 43,799 distinct target records in the same deterministic
   ordering policy and with the same seed behavior.
5. Preserve the existing optimizer, scheduler, precision, batching, gradient
   accumulation, clipping, and checkpoint settings.
6. At every epoch boundary, evaluate completion-only causal-LM loss separately on
   the complete configured SQuAD validation and TriviaQA validation record sets.
   With formal `data.eval_limit: null`, this means every valid record (currently
   10,570 SQuAD rows and 17,944 converted TriviaQA rows). Derive perplexity from
   each loss. Do not generate answers or compute EM/F1 in the training loop.
7. Preserve the named Trainer losses in `trainer/trainer_state.json` and write the
   ten merged epoch records, global steps, dataset counts, losses, and perplexities
   to `epoch_validation_metrics.json`.
8. Save and use only the adapter from the final epoch/step. Per-epoch validation
   evidence must never select a checkpoint, trigger early stopping, or create an
   intermediate adapter.
9. Record per-epoch and final training metrics already emitted by the existing
   trainer, including loss, learning rate, throughput, peak GPU memory, and
   gradient norm where available.

The final trainer state must show completion of epoch 10 and the expected derived
optimizer-step count for the unchanged dataset and batch configuration. The compact
epoch-validation artifact must contain exactly ten complete entries with both
validation datasets.

## Evaluation and Generated Records

After target epoch 10, evaluate the base model and final 10-epoch target adapter
once on the same full SQuAD validation data and selected TriviaQA validation rows
using the source plan's unchanged completion and deterministic generation metrics.
This final utility stage computes completion loss/perplexity plus generated-answer
EM/F1. It is distinct from the loss-only per-epoch target evaluations, which never
run generation, EM, or F1.

For each 10-epoch target adapter, generate 1,024 completions from each unchanged
source:

1. `squad_target_train`;
2. `squad_validation_candidates`;
3. the selected TriviaQA validation candidates.

Use the same `do_sample=false`, temperature, tokenizer, prompt construction,
truncation, and `max_new_tokens=64` settings. Preserve source row IDs and stable
candidate IDs. Recompute content hashes because the 10-epoch target may produce
different completions.

As in the source plan, a generated record is a true member only when its
materialized prompt-plus-completion SHA256 exactly matches a materialized target
training record. Source-prompt membership remains separate provenance metadata.

Record completion loss/perplexity, QA EM and token F1, completion-token and
generated-token lengths, duplicate rate, and exact-reconstruction rate. Bind all
outputs to the final adapter digest and generation manifest.

## Candidate Suite and Five Shadows

Build the same six groups in the same fixed order:

1. `gold_squad_target_train`;
2. `gold_squad_validation`;
3. `gold_trivia_validation`;
4. `gen_from_squad_train`;
5. `gen_from_squad_validation`;
6. `gen_from_trivia_validation`.

Keep the public text table separate from evaluator-only labels. Preserve stable
IDs and group-major order so corresponding baseline and 10-epoch records can be
paired by ID. Generated-record content hashes may differ as a direct consequence
of the controlled training change.

Inherit the exact five-column baseline mask rather than resampling it. Gold rows
must match baseline candidate ID, content hash, group, and source ID. Generated rows
must match baseline raw generation position, group, and source ID; their historical
content hash resolves the baseline canonical mask. Project raw assignments onto the
current content-canonical table and fail on any missing row, hash mismatch, identity
mismatch, uncovered canonical candidate, or conflicting duplicate assignment.

For shadow `j`:

1. Include the candidates selected by inherited mask column `j`.
2. Select deterministic auxiliary-pool filler under the current target-overlap-free
   rule and require exactly 43,799 distinct content hashes.
3. Record the historical filler divergence; do not restore the 36--41 contaminated
   baseline rows per shadow merely to manufacture an epoch-only comparison.
4. Initialize a fresh LoRA adapter from the pinned base model.
5. Train for 10 epochs with every other training setting unchanged. Supply no
   validation dataset, schedule no evaluation calls, and emit no epoch-validation
   artifact for any shadow.
6. Save the final adapter, exact training manifest, inclusion summary, resolved
   config, trainer state, metrics, source/environment state, W&B offline record,
   and `experiment.md`.

Generated candidate text may differ from the baseline. Mask assignments and source
prompt identities remain paired under the validated mapping; auxiliary filler IDs
follow the safer current policy and their historical differences are reported.

## Text-Only Scoring and RMIA

Keep the source score and attack definitions unchanged. For completion tokens
`y_1` through `y_n`:

```text
s_m(x) = (1 / n) * sum_t log P_m(y_t | prompt, y_<t)
```

For each candidate, split the five shadow scores by the unchanged candidate mask:

```text
log_ref_x = log((mean(exp(s_j(x)) for j in IN(x))
                + mean(exp(s_j(x)) for j in OUT(x))) / 2)
r_x = s_target(x) - log_ref_x

log_ref_z = log(mean(exp(s_j(z)) for j in all five shadows))
r_z = s_target(z) - log_ref_z

rmia_gamma(x) = mean_z[r_x - r_z > log(0.5)]
```

Compute means with logsumexp. The scorer must remain invariant to group names,
provenance, row order, and membership labels. Evaluator metadata is joined only
after target, shadow, population, and RMIA scores have been saved.

## Comparative Analysis

Produce every distribution, utility metric, attack metric, confidence interval,
histogram, and ECDF required by the source plan. Add a separate paired comparison
between the verified 1-epoch and 10-epoch runs.

For each model family and candidate group, report:

- change in SQuAD and TriviaQA completion loss, perplexity, EM, and token F1;
- change in generated-text length, duplication, and exact reconstruction;
- change in target, IN-reference, OUT-reference, relative, and RMIA score
  distributions;
- change in ROC-AUC, TPR at 1% and 5% FPR, and calibrated group false-positive
  rates, with deterministic bootstrap confidence intervals;
- train-versus-IID, train-versus-OOD, and train-versus-generated comparisons;
- training cost evidence: optimizer steps, elapsed time, throughput, and peak GPU
  memory.

Pair records by stable candidate ID when the materialized candidate is unchanged.
For target-generated candidates whose completion changes, report both the paired
source-prompt comparison and the new materialized-text distribution. Do not treat
changed generated text as if it were byte-identical paired evidence.

Keep utility, generation quality, raw score distributions, RMIA results, and
1-versus-10-epoch comparisons in separate report sections. Conclusions must describe
the comparison as observational, name the filler-policy difference, and list any
failed compatibility checks or unavoidable derived differences.

## Formal Records and Reproducibility

Use the source project's Hydra, uv, W&B-offline, manifest, and `experiment.md`
conventions. Every formal stage records:

- resolved config and command-line overrides;
- pinned model, tokenizer, and dataset revisions;
- source commit, dirty state, and lockfile hash;
- Python, PyTorch, CUDA, driver, GPU, Transformers, Datasets, PEFT, and Accelerate
  versions;
- seed, precision, batch sizes, gradient accumulation, learning rate, optimizer,
  scheduler, epoch count, and derived step counts;
- throughput, elapsed time, peak GPU memory, and gradient norm where available;
- input and output artifact paths, row counts, SHA256 hashes, and dependency
  digests;
- the baseline-run digests used by the comparison report.

Paths stored in records must be project-relative. Reuse requires exact manifest
and dependency matches; a mismatch fails rather than silently regenerating or
mixing baseline and 10-epoch evidence.

## Implementation and Execution Sequence

Use the restartable runner
`scripts/run_squad_lora_rmia_more_epochs_formal.sh` for a complete model run and
`scripts/submit_squad_lora_rmia_more_epochs_repair.sh` for the two recovery jobs.
The project-owned submitter calls Slurm directly so a failed child command produces
a failed job rather than a false successful status. A complete model run uses:

1. `prepare`;
2. `train_target`;
3. `evaluate`;
4. `generate_base`;
5. `generate`;
6. `build_candidates`;
7. `train_shadows` (all five shadows).

The initial authorized compute phase ends after all five shadow adapters are
complete for each model. Candidate/population scoring, RMIA, plotting, comparative
analysis, reporting, and final attack validation are explicitly deferred to a
later submission decision; they are not included in these two jobs.

Submit `pythia-shadows` to one 40GB GPU on `xe8545` and `olmo-full` to
one 80GB GPU on `tmp`. The model roots and logs are independent, so both jobs may
run concurrently. Keep batch size 16, gradient accumulation 2, BF16, one GPU, and
the `6-23:30:00` limit. Restartability comes from existing completeness checks; do
not use `workflow.force=true` for a routine resubmission.

Before formal launch, require:

```text
uv lock --check
bash -n scripts/run_squad_lora_rmia_more_epochs_formal.sh
bash -n scripts/submit_squad_lora_rmia_more_epochs_repair.sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Also require focused tests for the target-only epoch-evaluation policy, exact
config-diff whitelist, named validation datasets, final-only adapter saving,
shadow evaluation prohibition, and runner stage order. No training or model
download is required for implementation verification.

## Monitoring and Recovery

Longer training is expected to increase the training-stage optimizer updates by
approximately a factor of ten; evaluation, generation, scoring, and reporting
work remain approximately unchanged except for content-dependent variation.
Actual wall time must be measured rather than inferred.

Monitor the active model and stage, process state, latest completed epoch or
shadow, log progress, GPU utilization, memory, power, throughput, and any
OOM/CUDA/Traceback message. Use the project's existing monitoring interval and
recovery policy.

A resumed stage must match the full resolved-config fingerprint and trainer state.
Never resume a 10-epoch run from a 1-epoch adapter or mix baseline and new output
roots. If a fix changes data, masks, training updates, generation, or score
semantics, invalidate and rerun every dependent artifact.

## Acceptance Criteria

For completion of the initial target-and-shadow training phase, require for both
model families:

- one fresh target adapter whose trainer state completes epoch 10;
- exactly ten target epoch-evaluation entries, each containing separate loss and
  perplexity for the complete configured SQuAD and TriviaQA validation sets;
- `epoch_validation_metrics.json` and matching named losses in
  `trainer/trainer_state.json`;
- only the final epoch-10 target adapter saved and used by later stages;
- complete base and final-target utility records, with generated-answer EM/F1 run
  once after epoch 10 rather than during per-epoch evaluation;
- three regenerated target-LoRA generated groups with validated manifests;
- six complete candidate groups and a valid five-column mask;
- exactly five fresh shadow adapters whose trainer states complete epoch 10;
- zero shadow validation datasets, evaluation calls, named evaluation losses, or
  `epoch_validation_metrics.json` files;
- unchanged pinned revisions, split identities, target rows, candidate-source
  rows, inherited mask assignments, seeds, and all non-epoch scientific
  hyperparameters, plus an explicit record of target-overlap-free filler divergence;
- complete resolved configs, manifests, metrics, W&B-offline records, and
  `experiment.md` files for the submitted stages.

Full experiment acceptance additionally requires the deferred target, five-shadow,
population, and RMIA score artifacts; confidence intervals, figures, reports, final
validation; and an explicit 1-versus-10-epoch comparison that reports utility and
membership-signal outcomes without selecting or suppressing unfavorable results.

## References

- Source plan: `references/squad_lora_rmia_experiment_plan.md`
- SQuAD: https://huggingface.co/datasets/rajpurkar/squad
- TriviaQA: https://huggingface.co/datasets/mandarjoshi/trivia_qa
- Pythia-410M: https://huggingface.co/EleutherAI/pythia-410m
- OLMo-1B-hf: https://huggingface.co/allenai/OLMo-1B-hf
- RMIA: https://arxiv.org/abs/2312.03262
