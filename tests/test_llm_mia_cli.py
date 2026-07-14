from cli.squad_lora_rmia import reproducible_command


def test_reproducible_command_uses_uv_and_project_relative_entrypoint() -> None:
    command = reproducible_command(
        [
            "/project/.venv/bin/python3",
            "runtime.seed=42",
            "workflow.stage=generate_base",
        ]
    )

    assert command == (
        "uv run python ./cli/squad_lora_rmia.py runtime.seed=42 "
        "workflow.stage=generate_base"
    )
    assert "/project/" not in command
