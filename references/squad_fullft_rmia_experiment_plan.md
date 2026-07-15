# SQuAD Full Fine-Tuning and Online RMIA Experiment Plan

## 1. Scope and Fixed Claims

This experiment repeats the SQuAD LoRA workflow with full-parameter fine-tuning
for both target and shadow models. It measures text-only membership signals for:

| Run key | Base model | Revision |
| --- | --- | --- |
| `pythia_410m` | `EleutherAI/pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` |
| `olmo_1b_hf` | `allenai/OLMo-1B-hf` | `aee7752d9c08ee4775e9b0091426d8410e8f6a89` |

The target and all five shadows start independently from the pinned base
revision. PEFT, LoRA, quantization, merged adapters, and warm-starting from any
fine-tuned checkpoint are forbidden.

The model receives only prompt and completion token IDs. Candidate ID may be
used only to look up a shadow mask. Group, provenance, row order, and membership
labels are evaluator-only fields joined after scores are saved. Results are
membership-conditioned analyses under the stated online-attacker capability,
not a label-blind deployable-attack benchmark.

Use only the final model after one epoch. Do not select a checkpoint or tune any
setting from validation or membership results.

## 2. Changes from the LoRA Experiment

Data, prompt formatting, objective, epoch count, generation, evaluation, RMIA,
and reporting remain unchanged. The intentional differences are:

| Setting | LoRA | Full fine-tuning |
| --- | --- | --- |
| Trainable parameters | LoRA adapters | All model parameters |
| Learning rate | `1e-4` | `2e-5` |
| Gradient checkpointing | Disabled | Enabled |
| Pythia micro-batch / accumulation | `16 / 2` | `16 / 2` |
| OLMo micro-batch / accumulation | `16 / 2` | `8 / 4` |
| Saved model | Adapter | Full model checkpoint |

Both families keep an effective batch size of 32. The lower learning rate is
fixed before execution because `1e-4` is a LoRA-specific setting. Comparisons
with prior LoRA results are descriptive, not a controlled causal ablation.

Use a dedicated interface and output namespace while reusing shared modules:

~~~text
conf/squad_fullft_rmia.yaml
conf/finetuning/full.yaml
cli/squad_fullft_rmia.py
scripts/run_squad_fullft_rmia_smoke.sh
scripts/run_squad_fullft_rmia_formal.sh
outputs/squad_fullft_rmia/{smoke,formal}/...
~~~

Hydra selects `finetuning=full`; that config constructs the full-fine-tuning
strategy through `_target_` and contains no LoRA parameters.

The existing `outputs/squad_lora_rmia` tree is read-only and must never be
overwritten or relabeled.

## 3. Data and QA Objective

Use these pinned datasets:

- `rajpurkar/squad`, `plain_text`, revision
  `7b6d24c440a36b6815f21b70d25016731768db1f`;
- `mandarjoshi/trivia_qa`, `rc`, revision
  `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`, validation shards only.

With `runtime.seed=42`, preserve the existing deterministic splits:

| Split | Rows | Use |
| --- | ---: | --- |
| `squad_target_train` | 43,799 | Target training |
| `squad_attacker_auxiliary_pool` | 43,800 | Shadow filler |
| `squad_validation_candidates` | 1,024 | IID candidates and prompts |
| `squad_validation_population` | 1,024 | RMIA calibration only |
| `trivia_validation_candidates` | 1,024 | OOD candidates and prompts |

The two SQuAD validation subsets are disjoint from each other and from every
training set. Build TriviaQA context from `entity_pages.wiki_context`, then
`search_results.search_context`. Never use TriviaQA train or test shards.

Each QA record is:

~~~text
Context:
{context}

Question:
{question}

Answer:

completion = " {first_gold_answer}"
~~~

Train with completion-only causal cross-entropy: prompt labels are `-100` and
only completion tokens contribute to loss. Maximum sequence length is 1,024.
Shorten context around the answer when necessary. Record dropped rows and fail
if a required group cannot retain its specified size.

## 4. Reuse Policy

Only two prior artifact classes may be reused:

| Artifact | Rule |
| --- | --- |
| Split files under `data/squad_lora_rmia` | Reuse only if dataset revisions, seed, split rule, ordered IDs, row counts, and file SHA256 values match. Treat files as read-only. |
| `outputs/squad_lora_rmia/formal/<model>/generated/base` | Reuse only for the same model family after validating model/tokenizer revision, source IDs, generation settings, row counts, manifest SHA256, and every referenced file SHA256. |

For reused base generation, require `do_sample=false`, `temperature=0`,
`max_new_tokens=64`, and exactly 1,024 source rows per group. If the prior
manifest lacks a required source-split digest, recompute the expected ordered
source IDs from the validated split files and record the comparison.

Write `reuse_manifest.json` under the new model output. It records the prior
project-relative path, source commit, prior manifest hash, validated fields, and
reused file hashes. Any missing file or mismatch forces regeneration.

Rerun base-model utility evaluation so base and full-FT metrics use the same
current evaluation code. Recompute all target/shadow checkpoints, full-FT
generations, candidate tables, masks, train manifests, scores, RMIA outputs,
figures, and reports. LoRA-generated text is outside this experiment.

## 5. Training and Checkpoints

Use the same recipe for a target and its five shadows:

| Setting | Value |
| --- | --- |
| Epochs | `1` |
| Precision | `bf16` |
| Optimizer | `adamw_torch` |
| Learning rate | `2e-5` |
| Weight decay | `0.0` |
| Scheduler / warmup | Linear / `0.03` |
| Maximum sequence length | `1024` |
| Tokenization / DataLoader workers | `8 / 4` |
| Gradient checkpointing / model cache | Enabled / disabled |
| Temporary checkpoint interval | `250` optimizer steps |
| Temporary checkpoints retained | `1` |

Use seed 42 for the target, seed 1042 for mask generation, and seed `1042 + j`
for shadow `j` model initialization, data order, and auxiliary filler.

Before every train run, assert that the model is not a `PeftModel`, has no
adapter, and `trainable_parameters == total_parameters`. Run a non-formal
20-step target-plus-one-shadow memory smoke test for each family. Formal batch
values never change automatically. The single-shadow smoke uses a deterministic,
bounded training-only inclusion mask and does not report RMIA metrics, which
require the five-shadow formal masks. If the smoke test OOMs, revise the config
and this plan, preserve effective batch 32, and rerun all gates.

Save target and shadow outputs under:

~~~text
outputs/squad_fullft_rmia/formal/<model>/target/seed_42/
outputs/squad_fullft_rmia/formal/<model>/shadows/shadow_00/
...
outputs/squad_fullft_rmia/formal/<model>/shadows/shadow_04/
~~~

Each completed run contains the full model in `safetensors` format, tokenizer
and model configs, trainer state, metrics, resolved config, train manifest,
`experiment.md`, and a file-level SHA256 manifest. Write the final model to a
temporary directory, reload it for one forward pass, verify its hashes, then
atomically publish it with a `_SUCCESS` marker. Reuse requires `_SUCCESS`, the
same config fingerprint, and matching hashes.

Temporary checkpoints include optimizer, scheduler, and RNG state for resume.
Delete them only after the final model is verified. Before launch, verify free
space for six final full models per family, one active resumable checkpoint, all
non-model artifacts, and a 30% margin; derive the estimate from the smoke run.

## 6. Utility and Generated Data

Evaluate the base and final full-FT target on full SQuAD validation and selected
TriviaQA validation. Report completion loss, perplexity, completion-token count,
evaluated rows, greedy Exact Match, token F1, throughput, and peak GPU memory.
Use SQuAD normalization for SQuAD and all normalized aliases for TriviaQA.

Use deterministic generation with `do_sample=false`, `temperature=0`,
`max_new_tokens=64`, and batch size 96. Scoring uses batch size 96 and
`loss_chunk_tokens=64`.

For each target, generate and evaluate 1,024 raw records from each source:

| Group | Prompt source |
| --- | --- |
| `fullft_gen_from_squad_train` | Deterministic subset of target train |
| `fullft_gen_from_squad_validation` | SQuAD validation candidates |
| `fullft_gen_from_trivia_validation` | TriviaQA validation candidates |

Save public text and evaluator-only provenance separately. The manifest binds
the target checkpoint digest, tokenizer revision, source split digest,
generation config, row count, and file hashes. Record EM, token F1, completion
loss/perplexity, completion length, duplicate rate, and exact-reconstruction
rate. A generated record is a member only if its materialized prompt plus
completion hash occurs in target training; prompt provenance is not membership.

## 7. Candidate Sets and Deduplication

The primary online-RMIA candidate suite contains six raw groups in this order:

1. `gold_squad_target_train`
2. `gold_squad_validation`
3. `gold_trivia_validation`
4. `fullft_gen_from_squad_train`
5. `fullft_gen_from_squad_validation`
6. `fullft_gen_from_trivia_validation`

Each starts with 1,024 rows. Keep this raw provenance table unchanged. For model
scoring and shadow masks, canonicalize by the SHA256 of materialized prompt plus
completion and score each unique content once. Store a separate one-to-many
mapping from canonical candidate ID to all source/group rows. Membership is
defined only by exact target-training content membership, so one canonical
candidate cannot have conflicting labels. Report raw, within-group unique, and
global unique counts.

The descriptive analysis uses a 3-by-3 matrix:

| Source | Gold | Reused base generation | New full-FT generation |
| --- | --- | --- | --- |
| SQuAD target train | `gold_squad_target_train` | `base_gen_from_squad_train` | `fullft_gen_from_squad_train` |
| SQuAD validation | `gold_squad_validation` | `base_gen_from_squad_validation` | `fullft_gen_from_squad_validation` |
| TriviaQA validation | `gold_trivia_validation` | `base_gen_from_trivia_validation` | `fullft_gen_from_trivia_validation` |

Base-generated groups are descriptive controls and never enter shadow training.
Require at least 512 unique records in every descriptive group; otherwise fail
instead of silently changing the plotting sample size.

## 8. Five Shadows and Attacker Capability

For every unique attack candidate, create a deterministic Bernoulli(0.5) mask
conditioned on one to four IN shadows. For shadow `j`, let `I_j` be its included
candidate records. Train it on `I_j` plus a deterministic sample without
replacement from `squad_attacker_auxiliary_pool` until the set contains exactly
43,799 unique content hashes. Filler must exclude `I_j`; no target-training
record outside `I_j` may enter. Fail on an invalid mask, duplicate, overlap,
insufficient filler, or incorrect count.

The attacker may access candidate text and opaque IDs; a target oracle returning
completion-token likelihoods; the pinned base model and tokenizer; the full-FT
recipe and train-set size; the disjoint auxiliary pool and population set; and
resources for five shadows. The attacker may not access target membership,
evaluator metadata, target training records other than unlabeled candidates,
fine-tuned target weights, gradients, optimizer state, or hidden activations.
Shadow IN/OUT masks are attacker-chosen reference memberships, not target labels.

Save all five full shadow models and their masks, train manifests, inclusion
summaries, metrics, configs, hashes, W&B records, and `experiment.md` files.

## 9. Scores and Analysis

For completion tokens `y_1..y_n`, define:

~~~text
s_m(x) = (1 / n) * sum_t log P_m(y_t | prompt, y_<t)
~~~

For attack candidate `x`, use its mask-selected shadows:

~~~text
log_ref_x = log((mean(exp(s_j(x)) for j in IN(x))
                + mean(exp(s_j(x)) for j in OUT(x))) / 2)
r_x = s_target(x) - log_ref_x

log_ref_z = log(mean(exp(s_j(z)) for j in all five shadows))
r_z = s_target(z) - log_ref_z

rmia_gamma(x) = mean_z[r_x - r_z > log(gamma)]
~~~

Compute all means with `logsumexp`. `z` is the 1,024-row SQuAD population set.
Fix `gamma=0.5`; do not tune it.

For the nine-group descriptive ECDF, use:

~~~text
relative_log_likelihood(x)
  = s_target(x) - log(mean(exp(s_j(x)) for j in all five shadows))
~~~

This feature is not the mask-balanced RMIA score. Plot 512 deterministic unique
records per group, with color for data source and line style for text origin.
Summary tables use every unique record in each group.

For each score and group, report count, mean, standard deviation, median, IQR,
and 1/5/95/99 percentiles. Compare target-train members with the label-0 subset
of each other attack group. Report ROC-AUC, TPR at FPR 1% and 5%, and
deterministic 95% bootstrap confidence intervals with 2,000 resamples.
Calibrate thresholds only on the population set and report IID, OOD, and
generated-group false-positive rates.

## 10. Execution Order and Outputs

Complete Pythia before starting OLMo. Within each family, run:

| Stage | Required output |
| --- | --- |
| Validate inputs | `reuse_manifest.json` and split checks |
| Smoke | Loadable target and one-shadow smoke models, memory record; no RMIA metrics |
| Train target | Verified full checkpoint, metrics, train manifest |
| Evaluate and generate | Base/target metrics and three full-FT generated files |
| Build candidates | Raw table, canonical public table, evaluator mapping, five-shadow masks |
| Train shadows | Five verified full checkpoints and train manifests |
| Score and attack | Raw target/shadow/population scores and online-RMIA metrics |
| Analyze and report | Nine-group figure/tables, manifests, `experiment.md` |

A stage is reusable only when its `_SUCCESS` marker, config fingerprint, input
digests, and output hashes all match. A partial directory is never complete.

Launch from the project root with:

~~~bash
bash scripts/run_squad_fullft_rmia_smoke.sh
bash scripts/run_squad_fullft_rmia_formal.sh
~~~

## 11. Reproducibility and Acceptance

Formal entry points use Hydra with mandatory `runtime.seed`,
`hydra.job.chdir=false`, project-root-relative paths, and the
`finetuning=full` config group. Replaceable components use Hydra `_target_`
construction. The full-FT resolved config must not contain a LoRA block.
Required config areas are `runtime`, `experiment`, `model`, `data`, `optimizer`,
`scheduler`, `loss`, `precision`, `train`, `eval`, `checkpoint`, `wandb`, and
`report`.

Use the committed `uv.lock`. W&B defaults to offline mode. Every stage records
the resolved config and overrides, config fingerprint, source commit and dirty
state, lockfile hash, dependency versions, model/data revisions, hardware,
driver/CUDA/PyTorch versions, seeds, precision, batch sizes, throughput, peak
memory, learning rate, gradient norm, W&B run ID, and immutable artifact digests.
Register large checkpoints as W&B reference artifacts; do not commit generated
data, checkpoints, or local W&B directories. Keep the training and scoring main
path in PyTorch tensors and retain jaxtyping plus beartype checks at concrete
tensor boundaries.

Each stage's `experiment.md` records purpose, hypothesis, exact command and
overrides, config fingerprint, source/dependency state, artifact identifiers,
metrics, conclusion, achieved-purpose status, and next action.

Before formal launch, tests must verify:

- config loading, seed derivation, full trainability, and no PEFT wrapper;
- split/reuse validation and mismatch rejection;
- completion-only masking and deterministic truncation;
- full-model save/load equivalence and interrupted-run resume;
- canonical candidate mapping, valid masks, exact shadow size, and no forbidden
  overlap;
- score independence from evaluator metadata and correct log-space RMIA;
- one target plus one shadow end-to-end smoke run.

Run and pass:

~~~bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
~~~

The experiment is complete only when each family has one verified target and
five verified shadow full checkpoints; base/target utility metrics; three
full-FT generated datasets and metrics; validated base-generation references;
candidate, mask, and population manifests; raw scores; online-RMIA metrics;
nine-group summaries and figure; W&B offline records; and complete
`experiment.md` files.

## References

- Prior plan: `references/squad_lora_rmia_experiment_plan.md`
- SQuAD: https://huggingface.co/datasets/rajpurkar/squad
- TriviaQA: https://huggingface.co/datasets/mandarjoshi/trivia_qa
- Pythia-410M: https://huggingface.co/EleutherAI/pythia-410m
- OLMo-1B-hf: https://huggingface.co/allenai/OLMo-1B-hf
- RMIA: https://arxiv.org/abs/2312.03262
