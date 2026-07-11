# Membership Inference Attacks

This repository contains reproducible experiment code for CIFAR-10
membership inference attacks using torchvision models and LiRA/RMIA-style
attackers.

## Environment

Formal runs use `uv` and the locked project environment:

```bash
uv sync --extra dev
```

`requirements.txt` is intentionally not used as a formal environment source.
Add dependencies in `pyproject.toml` and refresh `uv.lock`.

## Smoke Run

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

Hydra keeps the process working directory at the project root and writes
run artifacts under `outputs/`.

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
- score/reference CSV files
- ROC and learning-curve figures when generated

W&B defaults to offline mode. Use `wandb sync` explicitly after inspecting a
run if online synchronization is desired.
