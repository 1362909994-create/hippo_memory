from __future__ import annotations

from pathlib import Path


def test_reasonix_one_step_installer_script_exists() -> None:
    script = Path("install-reasonix-hippo.ps1")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "Find-Python311" in text
    assert "Python.Python.3.11" in text
    assert "pip\", \"install\", \"--user" in text
    assert "reasonix-install-shim" in text
    assert "reasonix-deploy" in text
    assert "reasonix code" in text
