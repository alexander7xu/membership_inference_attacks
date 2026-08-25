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
    assert "source /home/c01xuju/CISPA-home/home/.envrc" in script
    assert 'shared_project_root=$(readlink -f "$project_root")' in script
    assert 'log_dir="$shared_project_root/logs"' in script
    assert '--container-image="$PYXIS_CONTAINER_IMAGE"' in script
    assert '--container-mounts="$PYXIS_CONTAINER_MOUNTS"' in script
    assert "/home/c01xuju/CISPA-home/.docker_image" not in script
    assert 'cache_root="/home/c01xuju/.cache"' in script
    assert 'data_root="/home/c01xuju/.local/share"' in script
    assert "export TORCH_DISABLE_NATIVE_JIT=1" in script
    assert "export XDG_CACHE_HOME=$cache_root" in script
    assert "export WANDB_CACHE_DIR=$cache_root/wandb" in script
    assert "export WANDB_DATA_DIR=$data_root/wandb" in script
    assert "export TRITON_CACHE_DIR=$cache_root/triton" in script
    assert "mkdir -p $cache_root/wandb $cache_root/triton $data_root/wandb" in script
    assert "mysubmit" not in script
