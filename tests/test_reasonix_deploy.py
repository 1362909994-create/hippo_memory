from __future__ import annotations

import json

from typer.testing import CliRunner

from hippocampus_memory.cli import app
from hippocampus_memory.deploy import (
    REASONIX_PROJECT_SPEC,
    deploy_reasonix,
    install_reasonix_mcp_spec,
    project_mcp_database,
    reasonix_fixed_mcp_spec,
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
    assert "hippo_memory_context_callback" in (root / "REASONIX.md").read_text(
        encoding="utf-8"
    )
    assert ".hippo/" in (root / ".gitignore").read_text(encoding="utf-8")
    assert result["index"]["indexed_files"] >= 1

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["mcp"] == [REASONIX_PROJECT_SPEC]

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
    assert "hippo_memory_context_callback" in text


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
