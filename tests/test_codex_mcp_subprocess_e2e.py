from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.deploy import deploy_codex
from hippocampus_memory.memory_writer import MemoryWriter


def test_codex_mcp_config_command_runs_real_jsonrpc_subprocess(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    deploy_codex(root, project="demo", index_project=False)
    db = Database(root / ".hippo" / "hippo.db")
    db.initialize()
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Decision: Codex MCP subprocess must return turn metadata.",
        tags=["codex", "mcp", "subprocess"],
    )

    process = subprocess.Popen(
        [sys.executable, "-m", "hippocampus_memory", "mcp-project", "--root", str(root)],
        cwd=Path.cwd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        initialized = _jsonrpc(process, "initialize", request_id=1)
        tools = _jsonrpc(process, "tools/list", request_id=2)
        pack = _jsonrpc(
            process,
            "tools/call",
            request_id=3,
            params={
                "name": "memory_pack",
                "arguments": {"query": "Codex MCP subprocess turn metadata"},
            },
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    tool_names = [tool["name"] for tool in tools["result"]["tools"]]
    structured = pack["result"]["structuredContent"]
    assert initialized["result"]["serverInfo"]["name"] == "hippocampus-memory"
    assert "memory_pack" in tool_names
    assert "Memory Pack:" in structured["text"]
    assert structured["execution_trace"]
    assert structured["retrieved_memories"]
    assert structured["selected_memories"]
    assert structured["context_budget"]["memory_scheduler_report"]


def test_codex_mcp_subprocess_survives_non_utf8_stdio_encoding(tmp_path: Path) -> None:
    root = tmp_path / "unicode-project"
    root.mkdir()
    deploy_codex(root, project="unicode-demo", index_project=False)
    db = Database(root / ".hippo" / "hippo.db")
    db.initialize()
    MemoryWriter(db).write(
        project="unicode-demo",
        memory_type="decision",
        content="\ufeffDecision: Codex memory can contain ????? ? and emoji ?? safely.",
        tags=["codex", "unicode", "stdio"],
    )

    env = {**os.environ, "PYTHONIOENCODING": "ascii"}
    process = subprocess.Popen(
        [sys.executable, "-m", "hippocampus_memory", "mcp-project", "--root", str(root)],
        cwd=Path.cwd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )
    try:
        pack = _jsonrpc(
            process,
            "tools/call",
            request_id=1,
            params={
                "name": "context_auto",
                "arguments": {
                    "intent": "unicode stdio Codex memory",
                    "session_key": "codex",
                    "max_tokens": 300,
                },
            },
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    assert pack["result"]["structuredContent"]["execution_trace"]


def _jsonrpc(
    process: subprocess.Popen[str],
    method: str,
    *,
    request_id: int,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert process.stdin is not None
    assert process.stdout is not None
    request = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        request["params"] = params
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise AssertionError(f"MCP subprocess exited without response: {stderr}")
    response = json.loads(line)
    assert response["id"] == request_id
    assert "error" not in response, response
    return response
