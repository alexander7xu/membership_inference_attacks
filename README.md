# Membership Inference Attacks

This repository contains reproducible experiment code for CIFAR-10
membership inference attacks using torchvision models and LiRA/RMIA-style
attackers. It also includes a formal SQuAD LoRA fine-tuning and online RMIA
workflow for causal language models.

## Environment

Formal runs use `uv` and the locked project environment:

```bash
uv sync --extra dev
```

`requirements.txt` is intentionally not used as a formal environment source.
Add dependencies in `pyproject.toml` and refresh `uv.lock`.

## CIFAR-10 Smoke Run

Launch experiments from the project root. Pass the seed explicitly so the
run record captures the exact source of randomness:

```bash
uv run python ./cli/attack_resnet_cifar10.py runtime.seed=42 attack=lira_online
```

Other configured attacks are:

```bash
attack=lira_offline
attack=rmia_online
attack=rmia_offline
```

## SQuAD LoRA And Online RMIA

The LLM workflow is documented in
`references/squad_lora_rmia_experiment_plan.md`.

Run a tiny end-to-end smoke pass first:

```bash
bash scripts/run_squad_lora_rmia_smoke.sh
```

Launch the formal Pythia-410M and OLMo-1B-hf runs:

```bash
bash scripts/run_squad_lora_rmia_formal.sh
```

The target and shadow models use the same Hydra training entry point. Formal
outputs are written under `outputs/squad_lora_rmia/`; generated data, LoRA
adapters, private labels, masks, metrics, manifests, and `experiment.md` files
are ignored by Git and tracked through run records.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Run Records

Each formal run writes:

- `resolved_config.yaml`
- `experiment.md`
- metrics and manifest JSON files
- score/reference files and figures when generated

W&B defaults to offline mode. Use `wandb sync` explicitly after inspecting a
run if online synchronization is desired.
