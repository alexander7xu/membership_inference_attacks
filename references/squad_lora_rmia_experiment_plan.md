# SQuAD LoRA and Text-Only Membership Signal Analysis Plan

## Purpose and Claim Boundary

This formal experiment measures how text-only language-model scores differ between
records that were and were not used to fine-tune a target model. The evaluator
knows true membership, provenance group, and deterministic row order. Those
fields are analysis metadata, not model inputs.

This is not a benchmark of a label-blind deployable attacker. Results must be
described as membership-conditioned score distributions or, when the specified
population-calibrated statistic is used, auditor-capability RMIA results.

Candidate files keep stable group-major order and readable IDs. This is allowed
because it supports auditing and plotting. The required boundary is:

- The language model receives only prompt and completion token IDs.
- Score code may join a candidate ID to its shadow mask, but must not parse the
  ID, inspect row order, use a group field, or read a membership label.
- Group, provenance, membership, and order may be used only after all scores
  are saved, by the evaluator and report generator.
- A future label-blind benchmark must use an opaque-ID view. It is out of scope.

## Target Models and Fine-Tuning

Train one final-epoch LoRA adapter for each target model:

| Run key | Base model |
| --- | --- |
| pythia_410m | EleutherAI/pythia-410m |
| olmo_1b_hf | allenai/OLMo-1B-hf |

Use one Hydra training entry point for target and shadow models.

- Task: completion-only causal next-token prediction.
- Prompt format: Context, Question, then Answer as specified below.
- Maximum sequence length: 1024; bf16 precision.
- LoRA: target_modules=all-linear, rank 16, alpha 32, dropout 0.05.
- Training: one epoch, learning rate 1e-4, no best-checkpoint selection.
- Evaluate only the adapter saved at the final epoch.

## Data, Revisions, and Splits

Pin the Hugging Face revision commit for every dataset and model. Record the
repository ID and revision in the resolved config and every manifest.

Use rajpurkar/squad, config plain_text. Its 87,599 train rows are partitioned by
a stable hash of SQuAD ID and the configured seed:

| Split | Rows | Use |
| --- | ---: | --- |
| squad_target_train | 43,799 | target training only |
| squad_attacker_auxiliary_pool | 43,800 | fill source for shadow training |

The one-row asymmetry is intentional: every target and shadow model can train
on exactly 43,799 distinct records without sampling with replacement.

From SQuAD validation, create deterministic disjoint subsets:

| Split | Rows | Use |
| --- | ---: | --- |
| squad_validation_candidates | 1,024 | IID non-member candidate group |
| squad_validation_population | 1,024 | RMIA population calibration only |

Use mandarjoshi/trivia_qa, config rc, validation rows only, as the OOD group.
Build context from entity_pages.wiki_context, then search_results.search_context.
Never download or use TriviaQA train/test shards. Select 1,024 valid rows by
deterministic ID hash.

Every split manifest records revisions, split rule, seed, row IDs, counts, and
SHA256 hashes. Reuse is allowed only when the manifest and all input hashes
match; otherwise fail unless an explicit regeneration override is supplied.

## QA Records and Utility Evaluation

Represent each QA row as:

~~~text
Context:
{context}

Question:
{question}

Answer:

completion = " {first_gold_answer}"
~~~

Use completion-only labels: prompt tokens are -100 and completion tokens are
causal-LM targets. Shorten context around the answer when needed. Drop and
record rows with no remaining completion token after truncation; never silently
change group counts.

Evaluate base and final LoRA models on full SQuAD validation and selected
TriviaQA validation. These metrics never select a checkpoint:

- completion loss, perplexity, completion-token count, and evaluated rows;
- greedy Exact Match and token F1;
- do_sample=false, temperature=0, max_new_tokens=64.

Use SQuAD normalization for SQuAD. Match TriviaQA against all normalized aliases.

## Generated Records

For each target adapter, generate 1,024 completions from each source:

| Group | Source prompts |
| --- | --- |
| gen_from_squad_train | squad_target_train |
| gen_from_squad_validation | squad_validation_candidates |
| gen_from_trivia_validation | TriviaQA validation candidates |

A generated record is a true member only when its materialized prompt plus
completion SHA256 exactly equals a materialized target-training record. Source
prompt membership remains a separate provenance field.

Save public generated text and an evaluator-only provenance table. Record EM,
token F1, completion loss/perplexity, completion-token count, generated-token
length statistics, duplicate rate, and exact-reconstruction rate. The manifest
must bind adapter digest, tokenizer revision, generation config, source manifest,
file hashes, and row counts. Reuse requires an exact manifest match.

## Candidate Suite and Five Shadows

Build these six 1,024-record groups in this fixed order:

1. gold_squad_target_train
2. gold_squad_validation
3. gold_trivia_validation
4. gen_from_squad_train
5. gen_from_squad_validation
6. gen_from_trivia_validation

Store a public text table separately from evaluator labels. The public table
contains ID, prompt, completion, content hash, and generation metadata. The
evaluator table contains group, source ID, provenance, and true membership.

Train exactly five saved LoRA shadow adapters for each target family. Generate a
deterministic Bernoulli(0.5) inclusion mask per candidate, conditioned on one
to four IN shadows. This guarantees at least one IN and one OUT shadow.

For shadow j, let I_j be its included candidates. Its training set is I_j plus
a deterministic, non-overlapping sample of 43,799 - len(I_j) rows from
squad_attacker_auxiliary_pool. Require exactly 43,799 distinct content hashes;
fail on an invalid mask, duplicate, or insufficient filler. No target-training
record outside I_j may enter a shadow training set.

Save every shadow adapter, train manifest, mask manifest, resolved config,
command, metrics, source/dependency state, and experiment.md.

## Text-Only Scores and RMIA

For record x with completion tokens y_1 through y_n, define:

~~~text
s_m(x) = (1 / n) * sum_t log P_m(y_t | prompt, y_<t)
~~~

This normalized completion log probability is the primary descriptive score.
Report target, IN-reference, OUT-reference, and group-conditioned distributions.

Use the name RMIA only for the following population-calibrated statistic. Let
IN(x) and OUT(x) be the candidate's mask-selected shadow sets:

~~~text
log_ref_x = log((mean(exp(s_j(x)) for j in IN(x))
                + mean(exp(s_j(x)) for j in OUT(x))) / 2)
r_x = s_target(x) - log_ref_x

log_ref_z = log(mean(exp(s_j(z)) for j in all five shadows))
r_z = s_target(z) - log_ref_z

rmia_gamma(x) = mean_z[ r_x - r_z > log(gamma) ]
~~~

Here z ranges over squad_validation_population, which is disjoint from the IID
candidate group and from all training sets. Compute all means with logsumexp.
Set gamma=0.5 in Hydra before the run and never tune it from validation or
membership results.

The scorer may use a candidate ID only to locate its mask. It must not use
labels, group names, source IDs, or row positions. The evaluator applies these
fields only after target, shadow, and RMIA scores have been saved.

## Analysis and Reports

The primary result is a distributional comparison, not a deployable-security
claim. For each model, score family, and true group, report count, mean, standard
deviation, median, IQR, and 1/5/95/99 percentiles.

For true members versus every non-member group, report ROC-AUC, TPR at FPR 1%
and 5%, and deterministic bootstrap 95% confidence intervals using 2,000
resamples. Calibrate thresholds only on squad_validation_population, then report
IID, OOD, and generated-group false-positive rates. Produce histograms and ECDF
plots grouped by evaluator-only labels.

Keep QA utility, generation quality, raw score distributions, and RMIA results
in separate report sections. State that group metadata and order were available
to the evaluator but excluded from score computation.

## Formal Records and Acceptance

Use Hydra with hydra.job.chdir=false, a mandatory runtime.seed, project-root
relative paths, uv-locked dependencies, and W&B offline mode. The config must
cover runtime, experiment, model, data, optimizer, scheduler, loss, precision,
train, eval, checkpoint, W&B, and report.

Every formal record includes config and overrides; pinned revisions; source
commit and dirty state; lockfile hash; Python, PyTorch, CUDA, driver, GPU,
Transformers, Datasets, PEFT, and Accelerate versions; seed; precision; batch
sizes; gradient accumulation; throughput; peak GPU memory; learning rate; and
gradient norm where available.

Register or reference by immutable digest data manifests, adapters, generated
artifacts, candidate suite, masks, metrics, figures, and reports. experiment.md
uses project-relative artifact paths and concrete digests, never machine-specific
absolute paths.

Before formal launch, require tests for deterministic revision-bound splits,
completion-only masking, score invariance to IDs/groups/order, label-free score
inputs, valid masks, exact shadow size, RMIA calibration, bootstrap seeding,
manifest-match reuse, manifest-mismatch failure, final adapter load, and a
small target-plus-one-shadow smoke run.

Pass uv lock --check, uv run ruff check ., uv run ruff format --check ., and
uv run pytest before every formal launch. A run is accepted only when all six
candidate groups, five shadow adapters, reusable generated artifacts, metrics,
figures, manifests, and experiment.md files exist for both model families.

## References

- SQuAD: https://huggingface.co/datasets/rajpurkar/squad
- TriviaQA: https://huggingface.co/datasets/mandarjoshi/trivia_qa
- Pythia-410M: https://huggingface.co/EleutherAI/pythia-410m
- OLMo-1B-hf: https://huggingface.co/allenai/OLMo-1B-hf
- RMIA: https://arxiv.org/abs/2312.03262
