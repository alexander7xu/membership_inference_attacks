from pathlib import Path


def test_no_committed_notebooks():
    notebooks = [
        path
        for path in Path(".").rglob("*.ipynb")
        if ".venv" not in path.parts and ".git" not in path.parts
    ]

    assert notebooks == []
