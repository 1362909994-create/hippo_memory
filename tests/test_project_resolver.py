from __future__ import annotations

from hippocampus_memory.project_resolver import resolve_project_name, write_project_config


def test_project_config_resolves_name_from_parent(tmp_path):
    root = tmp_path / "demo"
    child = root / "src"
    child.mkdir(parents=True)
    write_project_config(root, "configured-demo")

    assert resolve_project_name(cwd=child) == "configured-demo"
