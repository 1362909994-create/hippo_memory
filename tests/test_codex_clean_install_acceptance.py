from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_clean_project_can_be_deployed_and_diagnosed_through_cli(tmp_path: Path) -> None:
    root = tmp_path / "clean-project"
    root.mkdir()
    (root / "README.md").write_text("# Clean project\n", encoding="utf-8")

    deploy = subprocess.run(
        [
            sys.executable,
            "-m",
            "hippocampus_memory",
            "codex-deploy",
            "--root",
            str(root),
            "--project",
            "clean_project",
            "--no-index",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert deploy.returncode == 0, deploy.stderr
    assert (root / ".hippo" / "hippo.db").exists()
    assert (root / ".hippo" / "codex-mcp-config.json").exists()
    assert (root / ".hippo.toml").exists()
    assert "clean_project" in deploy.stdout
    assert "codex-mcp-config.json" in deploy.stdout

    doctor = subprocess.run(
        [
            sys.executable,
            "-m",
            "hippocampus_memory",
            "doctor",
            "--root",
            str(root),
            "--json",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert doctor.returncode == 0, doctor.stderr
    report = json.loads(doctor.stdout)
    assert report["diagnostic"] == "hippo_codex"
    assert report["ready"] is True
    assert Path(report["root"]) == root.resolve()
    assert report["db_exists"] is True
    assert report["codex_mcp_config_exists"] is True
    assert report["project_memory_has_hippo_block"] is True
    assert report["recommendations"] == []
