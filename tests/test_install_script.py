from __future__ import annotations

from pathlib import Path


def test_codex_one_step_installer_script_exists() -> None:
    script = Path("install-codex-hippo.ps1")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "Find-Python311" in text
    assert "Python.Python.3.11" in text
    assert "pip\", \"install\", \"--user" in text
    assert "codex-deploy" in text
    assert "AGENTS.md" in text
    assert "reasonix" not in text.casefold()
