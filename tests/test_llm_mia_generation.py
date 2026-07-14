from pathlib import Path

from omegaconf import OmegaConf

from src.llm_mia import workflow


def _config():
    return OmegaConf.create(
        {
            "model": {
                "key": "test_model",
                "name_or_path": "owner/test-model",
                "revision": "revision123",
                "display_name": "Test model",
            },
            "paths": {
                "output_root": "outputs/squad_lora_rmia",
                "data_dir": "data/squad_lora_rmia",
            },
            "workflow": {"profile": "formal"},
        }
    )


def test_base_generation_uses_base_model_and_separate_directory(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _config()
    calls = []
    monkeypatch.setattr(workflow, "prepare_data", lambda *_: None)
    monkeypatch.setattr(
        workflow,
        "_generate_candidates",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    workflow.generate_base_candidates(
        cfg,
        project_root=tmp_path,
        command="test command",
        smoke=False,
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["adapter_path"] is None
    assert kwargs["model_run_id"] == "owner/test-model@revision123"
    assert kwargs["run_dir"] == (
        tmp_path
        / "outputs/squad_lora_rmia"
        / "formal"
        / "test_model"
        / "generated"
        / "base"
    )


def test_target_generation_keeps_existing_directory_and_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _config()
    adapter = tmp_path / "adapter"
    calls = []
    monkeypatch.setattr(workflow, "train_target", lambda *args, **kwargs: adapter)
    monkeypatch.setattr(
        workflow,
        "_generate_candidates",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    workflow.generate_target_candidates(
        cfg,
        project_root=tmp_path,
        command="test command",
        smoke=False,
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["adapter_path"] == adapter
    assert kwargs["model_run_id"] == str(adapter)
    assert kwargs["run_dir"] == (
        tmp_path / "outputs/squad_lora_rmia" / "formal" / "test_model" / "generated"
    )
