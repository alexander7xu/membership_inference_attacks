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
    assert "printf -v quoted_command" not in script
    assert "job_command+=" in script
    assert "--wrap=\"exec bash -lc '$job_command'\"" in script
    assert "/home/c01xuju/CISPA-home/.home:/home/c01xuju:ro" in script
    assert "/home/c01xuju/CISPA-home:/home/c01xuju/CISPA-home:rw" in script
    assert 'cache_root="/home/c01xuju/CISPA-home/.cache"' in script
    assert 'data_root="/home/c01xuju/CISPA-home/.local/share"' in script
    assert "export TORCH_DISABLE_NATIVE_JIT=1" in script
    assert "export XDG_CACHE_HOME=$cache_root" in script
    assert "export WANDB_CACHE_DIR=$cache_root/wandb" in script
    assert "export WANDB_DATA_DIR=$data_root/wandb" in script
    assert "export TRITON_CACHE_DIR=$cache_root/triton" in script
    assert "mkdir -p $cache_root/wandb $cache_root/triton $data_root/wandb" in script
    assert "mysubmit" not in script
