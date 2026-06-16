from __future__ import annotations

import json

from hippocampus_memory.lsp_diagnostics import parse_pyright_output, run_python_diagnostics


def test_parse_pyright_output_normalizes_diagnostics(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    source = root / "app.py"
    source.write_text("value = missing_name\n", encoding="utf-8")
    payload = {
        "generalDiagnostics": [
            {
                "file": str(source),
                "severity": "error",
                "message": '"missing_name" is not defined',
                "range": {
                    "start": {"line": 0, "character": 8},
                    "end": {"line": 0, "character": 20},
                },
                "rule": "reportUndefinedVariable",
            }
        ]
    }

    diagnostics, error = parse_pyright_output(json.dumps(payload), root_path=root)

    assert error is None
    assert diagnostics[0].relative_path == "app.py"
    assert diagnostics[0].line == 1
    assert diagnostics[0].column == 9
    assert diagnostics[0].rule == "reportUndefinedVariable"


def test_run_python_diagnostics_handles_missing_checker(tmp_path):
    result = run_python_diagnostics(tmp_path, checker="definitely_missing_pyright_tool")

    assert result["available"] is False
    assert result["diagnostics"] == []
    assert "No basedpyright or pyright" in result["error"]
