from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.llm_mia import workflow


class FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata: dict[str, str]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.digest = "digest123"
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, name: str) -> None:
        self.files.append((path, name))

    def add_dir(self, path: str, name: str) -> None:
        self.files.append((path, name))


class FakeRun:
    id = "offline123"

    def __init__(self) -> None:
        self.logged: list[dict[str, float]] = []
        self.artifacts: list[FakeArtifact] = []

    def log(self, metrics: dict[str, float]) -> None:
        self.logged.append(metrics)

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)


def test_disabled_wandb_does_not_import_client(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"wandb": {"enabled": False}})
    assert workflow._start_wandb_run(cfg, project_root=tmp_path, command="test") is None


def test_tracking_logs_metrics_and_concrete_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_module = SimpleNamespace(Artifact=FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", artifact_module)
    metric_path = tmp_path / "metrics.json"
    metric_path.write_text("{}", encoding="utf-8")
    run = FakeRun()
    token = workflow._ACTIVE_WANDB_RUN.set(run)
    try:
        identifiers = workflow._log_wandb_stage(
            stage="attack",
            metrics={"auc": 0.75},
            paths={"metrics": metric_path},
        )
    finally:
        workflow._ACTIVE_WANDB_RUN.reset(token)

    assert run.logged == [{"attack/auc": 0.75}]
    assert run.artifacts[0].files == [(str(metric_path), "metrics")]
    assert identifiers == {"wandb_attack_artifact": "offline123-attack:digest123"}
