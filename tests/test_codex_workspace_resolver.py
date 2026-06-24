from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hippocampus_memory.codex_workspace import CodexWorkspaceResolver
from hippocampus_memory.mcp_server import HippoMcpServer


def test_codex_workspace_resolver_creates_project_local_database(tmp_path: Path) -> None:
    workspace = tmp_path / "alpha"
    workspace.mkdir()
    (workspace / "README.md").write_text("# Alpha\n", encoding="utf-8")

    resolved = CodexWorkspaceResolver(
        auto_create=True,
        env={"CODEX_WORKSPACE_ROOT": str(workspace)},
    ).resolve({})

    assert resolved.root == workspace.resolve()
    assert resolved.project == "alpha"
    assert resolved.db.path == workspace.resolve() / ".hippo" / "hippo.db"
    assert resolved.db.path.exists()
    assert (workspace / ".hippo.toml").exists()
    assert resolved.arguments == {"project": "alpha"}


def test_mcp_server_resolves_different_codex_workspaces_per_call(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    fallback = tmp_path / "fallback.db"

    server = HippoMcpServer(
        CodexWorkspaceResolver.fallback_database(fallback),
        safe_tool_names=True,
        project_resolver=CodexWorkspaceResolver(auto_create=True),
    )

    server.call_tool(
        "memory_write",
        {
            "workspace_root": str(alpha),
            "content": "Alpha-only Codex memory boundary.",
            "memory_type": "decision",
        },
    )
    server.call_tool(
        "memory_write",
        {
            "workspace_root": str(beta),
            "content": "Beta-only Codex memory boundary.",
            "memory_type": "decision",
        },
    )

    alpha_result = server.call_tool(
        "memory_search",
        {"workspace_root": str(alpha), "query": "Codex memory boundary", "top_k": 5},
    )
    beta_result = server.call_tool(
        "memory_search",
        {"workspace_root": str(beta), "query": "Codex memory boundary", "top_k": 5},
    )

    alpha_text = alpha_result["text"]
    beta_text = beta_result["text"]
    assert "Alpha-only" in alpha_text
    assert "Beta-only" not in alpha_text
    assert "Beta-only" in beta_text
    assert "Alpha-only" not in beta_text
    assert (alpha / ".hippo" / "hippo.db").exists()
    assert (beta / ".hippo" / "hippo.db").exists()


def test_mcp_codex_subprocess_uses_current_working_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "gamma"
    workspace.mkdir()
    process = subprocess.Popen(
        [sys.executable, "-m", "hippocampus_memory", "mcp-codex"],
        cwd=workspace,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        _jsonrpc(
            process,
            "tools/call",
            request_id=1,
            params={
                "name": "memory_write",
                "arguments": {
                    "content": "Gamma current-working-directory memory.",
                    "memory_type": "decision",
                },
            },
        )
        search = _jsonrpc(
            process,
            "tools/call",
            request_id=2,
            params={
                "name": "memory_search",
                "arguments": {"query": "current-working-directory memory", "top_k": 3},
            },
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    structured = search["result"]["structuredContent"]
    assert "Gamma current-working-directory memory" in structured["text"]
    assert structured["decision"]["project"] == "gamma"
    assert {memory["project"] for memory in structured["selected_memories"]} == {"gamma"}
    assert (workspace / ".hippo" / "hippo.db").exists()


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
