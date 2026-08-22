from pathlib import Path


def test_lira_submitter_uses_compute_visible_log_path() -> None:
    script = Path("scripts/submit_squad_lora_lira.sh").read_text(encoding="utf-8")

    assert 'shared_project_root=$(readlink -f "$project_root")' in script
    assert 'log_dir="$shared_project_root/logs"' in script
    assert 'log_dir="$project_root/logs"' not in script
