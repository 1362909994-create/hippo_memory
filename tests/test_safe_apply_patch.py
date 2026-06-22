from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "safe_apply_patch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("safe_apply_patch", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.name is not None
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_add_update_delete_patch_round_trip(tmp_path: Path) -> None:
    module = load_module()
    patch = """*** Begin Patch
*** Add File: notes.txt
+alpha
+beta
*** Update File: notes.txt
@@
 alpha
-beta
+gamma
*** Delete File: notes.txt
*** End Patch
"""

    result = module.apply_patch_text(tmp_path, patch)

    assert result.changed_files == [tmp_path / "notes.txt"]
    assert not (tmp_path / "notes.txt").exists()


def test_dry_run_reports_changes_without_writing(tmp_path: Path) -> None:
    module = load_module()
    patch = """*** Begin Patch
*** Add File: notes.txt
+alpha
*** End Patch
"""

    result = module.apply_patch_text(tmp_path, patch, dry_run=True)

    assert result.changed_files == [tmp_path / "notes.txt"]
    assert result.dry_run is True
    assert not (tmp_path / "notes.txt").exists()


def test_update_requires_exact_context(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Update File: notes.txt
@@
 alpha
-missing
+gamma
*** End Patch
"""

    try:
        module.apply_patch_text(tmp_path, patch)
    except module.SafePatchError as exc:
        assert "context did not match" in str(exc)
    else:
        raise AssertionError("expected SafePatchError")

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_rejects_paths_outside_repository(tmp_path: Path) -> None:
    module = load_module()
    patch = """*** Begin Patch
*** Add File: ../escape.txt
+nope
*** End Patch
"""

    try:
        module.apply_patch_text(tmp_path, patch)
    except module.SafePatchError as exc:
        assert "outside repository" in str(exc)
    else:
        raise AssertionError("expected SafePatchError")

    assert not (tmp_path.parent / "escape.txt").exists()


def test_cli_can_apply_patch_from_stdin(tmp_path: Path) -> None:
    patch = """*** Begin Patch
*** Add File: cli.txt
+hello
*** End Patch
"""

    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--repo", str(tmp_path), "--patch", "-"],
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "cli.txt").read_text(encoding="utf-8") == "hello\n"
    assert "cli.txt" in completed.stdout


def test_cleanup_requires_successful_probe_and_yes(tmp_path: Path) -> None:
    module = load_module()
    script = tmp_path / "safe_apply_patch.py"
    script.write_text("print('temporary')\n", encoding="utf-8")

    failed = module.cleanup_if_native_ok(
        script_path=script,
        probe_command=[sys.executable, "-c", "import sys; sys.exit(3)"],
        yes=True,
    )
    assert failed is False
    assert script.exists()

    refused = module.cleanup_if_native_ok(
        script_path=script,
        probe_command=[sys.executable, "-c", "import sys; sys.exit(0)"],
        yes=False,
    )
    assert refused is False
    assert script.exists()

    cleaned = module.cleanup_if_native_ok(
        script_path=script,
        probe_command=[sys.executable, "-c", "import sys; sys.exit(0)"],
        yes=True,
    )
    assert cleaned is True
    assert not script.exists()
