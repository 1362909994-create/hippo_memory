from __future__ import annotations

import json

from typer.testing import CliRunner

from hippocampus_memory.cli import app
from hippocampus_memory.deploy import (
    REASONIX_PROJECT_SPEC,
    deploy_reasonix,
    ensure_reasonix_global_memory,
    install_reasonix_command_shims,
    install_reasonix_mcp_spec,
    patch_reasonix_status_bar,
    project_mcp_database,
    reasonix_fixed_mcp_spec,
    write_reasonix_bootstrap_context,
    write_reasonix_status_file,
)


def test_reasonix_deploy_creates_project_local_mcp(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "app.py").write_text("def hello():\n    return 'ok'\n", encoding="utf-8")
    config_path = tmp_path / "reasonix" / "config.json"

    result = deploy_reasonix(root, project="demo", config_path=config_path)

    assert result["project"] == "demo"
    assert (root / ".hippo.toml").exists()
    assert (root / ".hippo" / "hippo.db").exists()
    assert (root / ".hippo" / "hippo-mcp.ps1").exists()
    assert (root / ".hippo" / "reasonix-mcp-spec.txt").exists()
    assert (root / ".hippo" / "reasonix-global-mcp-spec.txt").read_text(
        encoding="utf-8"
    ).strip() == REASONIX_PROJECT_SPEC
    assert (root / "REASONIX.md").exists()
    memory_text = (root / "REASONIX.md").read_text(encoding="utf-8")
    assert "hippo_memory_context_auto" in memory_text
    assert "hippo_memory_memory_auto_store" in memory_text
    assert "token_savings_text" in memory_text
    assert "final user-facing Reasonix UI reply" in memory_text
    assert ".hippo/" in (root / ".gitignore").read_text(encoding="utf-8")
    assert result["index"]["indexed_files"] >= 1

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["mcp"] == [REASONIX_PROJECT_SPEC]
    global_memory = config_path.parent / "REASONIX.md"
    assert global_memory.exists()
    global_memory_text = global_memory.read_text(encoding="utf-8")
    assert "hippo_memory_context_auto" in global_memory_text
    assert "final user-facing Reasonix UI reply" in global_memory_text

    db = project_mcp_database(root / "src")
    assert db.path == root / ".hippo" / "hippo.db"


def test_reasonix_config_install_is_idempotent_and_replaces_same_server(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "apiKey": "keep-this",
                "mcp": ["filesystem=npx server", "hippo_memory=old command"],
                "mcpDisabled": ["hippo_memory", "other"],
            }
        ),
        encoding="utf-8",
    )

    assert install_reasonix_mcp_spec(config_path)
    assert not install_reasonix_mcp_spec(config_path)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["apiKey"] == "keep-this"
    assert cfg["mcp"] == ["filesystem=npx server", REASONIX_PROJECT_SPEC]
    assert cfg["mcpDisabled"] == ["other"]


def test_reasonix_fixed_spec_quotes_paths_with_spaces(tmp_path):
    script = tmp_path / "project with spaces" / ".hippo" / "hippo-mcp.ps1"
    script.parent.mkdir(parents=True)
    spec = reasonix_fixed_mcp_spec(script)

    assert spec.startswith("hippo_memory=powershell.exe")
    assert '"' in spec
    assert "project with spaces" in spec


def test_reasonix_deploy_cli(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    config_path = tmp_path / "reasonix.json"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "reasonix-deploy",
            "--root",
            str(root),
            "--project",
            "demo",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "reasonix_mcp_spec" in result.output
    assert (root / ".hippo" / "hippo.db").exists()
    assert REASONIX_PROJECT_SPEC in config_path.read_text(encoding="utf-8")


def test_reasonix_deploy_appends_existing_agents_instead_of_shadowing(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    agents = root / "AGENTS.md"
    agents.write_text("# Existing instructions\n", encoding="utf-8")

    result = deploy_reasonix(root, project="demo", install_global=False, index_project=False)

    assert result["reasonix_project_memory"] == str(agents)
    assert not (root / "REASONIX.md").exists()
    text = agents.read_text(encoding="utf-8")
    assert "# Existing instructions" in text
    assert "hippo_memory_context_auto" in text
    assert "hippo_memory_memory_auto_store" in text
    assert "token_savings_text" in text


def test_reasonix_deploy_can_skip_project_memory(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()

    result = deploy_reasonix(
        root,
        project="demo",
        install_global=False,
        index_project=False,
        project_memory=False,
    )

    assert result["reasonix_project_memory"] is None
    assert not (root / "REASONIX.md").exists()


def test_project_mcp_database_falls_back_to_global_db(tmp_path, monkeypatch):
    global_db = tmp_path / "global.db"
    monkeypatch.setenv("HIPPO_DB_PATH", str(global_db))
    root = tmp_path / "undeployed"
    root.mkdir()

    db = project_mcp_database(root)

    assert db.path == global_db
    assert global_db.exists()


def test_reasonix_global_memory_is_idempotent(tmp_path):
    path, updated = ensure_reasonix_global_memory(tmp_path)
    again, updated_again = ensure_reasonix_global_memory(tmp_path)

    assert path == again
    assert updated
    assert not updated_again
    text = path.read_text(encoding="utf-8")
    assert "hippo_memory_context_auto" in text
    assert text.count("hippocampus-memory:start") == 1


def test_reasonix_command_shim_wraps_code_invocations(tmp_path):
    bin_dir = tmp_path / "npm"
    bin_dir.mkdir()
    (bin_dir / "reasonix.ps1").write_text("original ps1\n", encoding="utf-8")
    (bin_dir / "reasonix.cmd").write_text("original cmd\n", encoding="utf-8")
    (bin_dir / "reasonix").write_text("original sh\n", encoding="utf-8")

    result = install_reasonix_command_shims(bin_dir)
    again = install_reasonix_command_shims(bin_dir)

    ps1 = (bin_dir / "reasonix.ps1").read_text(encoding="utf-8")
    cmd = (bin_dir / "reasonix.cmd").read_text(encoding="utf-8")
    sh = (bin_dir / "reasonix").read_text(encoding="utf-8")
    assert result["ps1_updated"]
    assert result["cmd_updated"]
    assert result["sh_updated"]
    assert not again["ps1_updated"]
    assert not again["cmd_updated"]
    assert not again["sh_updated"]
    assert "HIPPO_MEMORY_REASONIX_SHIM" in ps1
    assert "reasonix-bootstrap-context" in ps1
    assert "--status-output" in ps1
    assert "HIPPO_REASONIX_STATUS_FILE" in ps1
    assert "--system-append-file" in ps1
    assert "Get-RecentReasonixWorkspace" in ps1
    assert "Get-ReasonixWorkspaceFromMeta" in ps1
    assert "Resolve-DefaultCodeRoot" in ps1
    assert "Test-IsUnsafeCodeRoot" in ps1
    assert "Add-NewSessionDefault" in ps1
    assert "--new" in ps1
    assert "--resume" in ps1
    assert "reasonix.ps1" in cmd
    assert "reasonix.ps1" in sh
    assert "powershell.exe" in sh
    assert (bin_dir / "reasonix.ps1.hippo-original").exists()
    assert (bin_dir / "reasonix.cmd.hippo-original").exists()
    assert (bin_dir / "reasonix.hippo-original").exists()
    assert result["status_bar_patch"]["reason"] == "reasonix_cli_dir_not_found"


def test_reasonix_status_bar_patch_is_idempotent(tmp_path):
    bin_dir = tmp_path / "npm"
    cli_dir = bin_dir / "node_modules" / "reasonix" / "dist" / "cli"
    cli_dir.mkdir(parents=True)
    chunk = cli_dir / "chunk-demo.js"
    chunk.write_text(
        "\n".join(
            [
                'import { formatTokens } from "./chunk-demo2.js";',
                "function Pill({ children }) {",
                "  return /* @__PURE__ */ import_react16.default.createElement(",
                "    Box_default, {}, children",
                "  );",
                "}",
                "function Gap() {",
                "  return /* @__PURE__ */ import_react16.default.createElement(",
                '    Text, null, " "',
                "  );",
                "}",
                "function StatusRow({",
                "  statusBar = DEFAULT_STATUS_BAR_CONFIG",
                "}) {",
                "  return /* @__PURE__ */ import_react16.default.createElement(",
                "    Box_default, null,",
                "    statusBar.showCacheHit &&",
                "    /* @__PURE__ */ import_react16.default.createElement(",
                "      import_react16.default.Fragment, null,",
                "      /* @__PURE__ */ import_react16.default.createElement(Gap, null),",
                "      /* @__PURE__ */ import_react16.default.createElement(",
                "        Pill, null,",
                "        /* @__PURE__ */ import_react16.default.createElement(",
                "          Text, { color: TONE.accent },",
                '          `${t("statusBar.cache")} ${Math.round(status2.cacheHit * 100)}%`)))'
                ", statusBar.showCtxUsage && true);",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    result = patch_reasonix_status_bar(bin_dir)
    again = patch_reasonix_status_bar(bin_dir)

    text = chunk.read_text(encoding="utf-8")
    assert result["patched"]
    assert again["reason"] == "already_patched"
    assert "HIPPO_REASONIX_STATUS_BAR_PATCH" in text
    assert "HIPPO_REASONIX_STATUS_BAR_PATCH v11" in text
    assert "HIPPO_REASONIX_STATUS_FILE" in text
    assert "HippoSavingsPill" in text
    assert "sessionId: session.id" in text
    assert "promptTokens: status2.promptTokens" in text
    assert "turnCost: status2.cost" in text
    assert "workspace: session.workspace" in text
    assert "hasTurn: hasTurn" in text
    assert "const hasActivity = hasTurn === true" in text
    assert "hasActivity && run > 0" in text
    assert "refreshHippoReasonixStatusForWorkspace" in text
    assert "reasonix-bootstrap-context" in text
    assert "reasonix_active_turn_runs_v1" in text
    assert "legacy_saved_tokens" in text
    assert "last_saved_tokens" in text
    assert "context_count" in text
    assert "预计节省" in text
    assert "会话 ${formatTokens(data.sessionTotal)}" in text
    assert "statusBar.showCtxUsage" in text
    assert (cli_dir / "chunk-demo.js.hippo-status-original").exists()


def test_reasonix_bootstrap_context_includes_token_savings(tmp_path, monkeypatch):
    db_path = tmp_path / "global.db"
    monkeypatch.setenv("HIPPO_DB_PATH", str(db_path))
    root = tmp_path / "demo"
    root.mkdir()
    (root / "app.py").write_text("def hello():\n    return 'ok'\n", encoding="utf-8")
    deploy_reasonix(root, project="demo", install_global=False)

    output = tmp_path / "context.md"
    status_output = tmp_path / "status.json"
    path = write_reasonix_bootstrap_context(root, output, status_output=status_output)

    text = path.read_text(encoding="utf-8")
    status = json.loads(status_output.read_text(encoding="utf-8"))
    assert "Hippocampus Memory bootstrap for Reasonix" in text
    assert "Show this token savings line to the user:" in text
    assert "Token savings:" in text
    assert status["available"]
    assert status["scope"] == "reasonix_session"
    assert status["project"] == "demo"
    assert status["run_id"]
    ledger_dir = status["session_ledger_dir"].replace("\\", "/")
    assert ledger_dir.endswith(".hippo/reasonix-session-savings")
    assert status["saved_tokens"] >= 0
    assert status["session_saved_tokens"] == 0
    assert status["total_saved_tokens"] == 0
    assert status["project_total_saved_tokens"] >= status["saved_tokens"]


def test_reasonix_bootstrap_context_auto_deploys_new_project(tmp_path, monkeypatch):
    db_path = tmp_path / "global.db"
    monkeypatch.setenv("HIPPO_DB_PATH", str(db_path))
    root = tmp_path / "fresh"
    root.mkdir()
    (root / "app.py").write_text("def hello():\n    return 'ok'\n", encoding="utf-8")

    output = tmp_path / "context.md"
    status_output = tmp_path / "status.json"
    write_reasonix_bootstrap_context(root, output, status_output=status_output)

    status = json.loads(status_output.read_text(encoding="utf-8"))
    assert (root / ".hippo.toml").exists()
    assert (root / ".hippo" / "hippo.db").exists()
    assert status["available"]
    assert status["project"] == "fresh"
    assert status["saved_tokens"] >= 0


def test_reasonix_status_file_still_renders_when_no_savings(tmp_path):
    output = tmp_path / "status.json"

    write_reasonix_status_file(output, None, project="demo", root=tmp_path)

    status = json.loads(output.read_text(encoding="utf-8"))
    assert status["available"]
    assert status["scope"] == "reasonix_session"
    assert status["project"] == "demo"
    assert status["root"] == str(tmp_path)
    assert status["saved_tokens"] == 0
    assert status["session_saved_tokens"] == 0
    assert status["reason"] == "no_token_savings_available"
