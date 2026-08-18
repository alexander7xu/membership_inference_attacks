from pathlib import Path


def test_no_committed_notebooks():
    notebooks = [
        path
        for path in Path(".").rglob("*.ipynb")
        if ".venv" not in path.parts and ".git" not in path.parts
    ]

    assert notebooks == []


def test_more_epochs_repair_submitter_routes_resources_and_preserves_exit_status():
    script = Path("scripts/submit_squad_lora_rmia_more_epochs_repair.sh").read_text(
        encoding="utf-8"
    )

    assert "pythia-shadows" in script
    assert "olmo-full" in script
    assert 'partition="xe8545"' in script
    assert 'partition="tmp"' in script
    assert "workflow.stage=train_shadows" in script
    assert "run_squad_lora_rmia_more_epochs_formal.sh olmo_1b_hf" in script
    assert "exec uv run python" in script
    assert "exec bash" in script
    assert "--wrap" in script
    assert "mysubmit" not in script
