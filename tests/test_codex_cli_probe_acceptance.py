from __future__ import annotations

import shutil
import subprocess

import pytest


def test_real_codex_cli_noninteractive_probe() -> None:
    codex = shutil.which("codex")
    if codex is None:
        pytest.skip("Codex CLI is not installed on this machine")

    version = subprocess.run(
        [codex, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    mcp_help = subprocess.run(
        [codex, "mcp", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    mcp_list = subprocess.run(
        [codex, "mcp", "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert version.returncode == 0
    assert "codex" in version.stdout.casefold()
    assert mcp_help.returncode == 0
    assert "Manage external MCP servers" in mcp_help.stdout
    assert mcp_list.returncode == 0
