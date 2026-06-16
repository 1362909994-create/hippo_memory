from __future__ import annotations

from hippocampus_memory.config import Settings
from hippocampus_memory.deploy import mcp_client_config, write_daemon_script
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.project_indexer import ProjectIndexer
from hippocampus_memory.summarizer import summarize_session
from hippocampus_memory.token_report import token_ledger_report, token_savings_report
from hippocampus_memory.vector_store import SQLiteVectorStore, create_vector_store


def test_chroma_vector_backend_falls_back_to_sqlite(db, tmp_path):
    settings = Settings(
        db_path=db.path,
        vector_backend="chroma",
        chroma_path=tmp_path / "chroma",
    )

    store = create_vector_store(db, settings)

    assert isinstance(store, SQLiteVectorStore)


def test_llm_summarizer_falls_back_to_rules_without_env(monkeypatch):
    monkeypatch.delenv("HIPPO_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("HIPPO_LLM_MODEL", raising=False)

    summary = summarize_session("用户：当前目标是先让 STM32 点亮 TFT 屏幕。", use_llm=True)

    assert summary["task_state"]


def test_rule_summarizer_extracts_higher_quality_candidates():
    summary = summarize_session(
        "\n".join(
            [
                "用户：我只想每个项目分开单独记忆，不想做全局记忆。",
                "Codex：决定采用项目级 token ledger。",
                "用户：失败点是旧 source chunk 可能误导 AI。",
                "用户：函数 callback_pack 会影响 callback session 去重。",
                "用户：未知点是 token 节省收益需要连续观察。",
            ]
        )
    )

    assert summary["project_context"][0]["memory_type"] == "user_preference"
    assert summary["decisions"]
    assert summary["failures"]
    assert summary["technical_facts"]
    assert summary["open_questions"]


def test_mcp_config_and_daemon_script_generation(tmp_path):
    config = mcp_client_config(command="python")
    assert config["mcpServers"]["hippocampus-memory"]["args"] == [
        "-m",
        "hippocampus_memory",
        "mcp",
    ]

    script = write_daemon_script(tmp_path / "start.ps1", port=9999)

    assert "9999" in script.read_text(encoding="utf-8")


def test_token_savings_report(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("def main():\n    return 'hello'\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Current task is to keep the token report useful.",
    )

    report = token_savings_report(db, "demo", "token report", model="gpt-4o")

    assert report["context_bundle_tokens"] > 0
    assert report["memory_pack_tokens"] > 0
    assert report["compact_pack_tokens"] > 0
    assert "gpt-4o" in report["token_counter"]
    assert report["context_bundle_strategy"] in {"auto:lean", "auto:full"}
    assert "token_counter_exact" in report
    assert "token_counter_note" in report
    assert report["full_context_bundle_tokens"] >= report["lean_context_bundle_tokens"]
    assert report["indexed_file_count"] == 1
    assert len(report["ledger_ids"]) == 3

    ledger = token_ledger_report(db, "demo")

    assert ledger["summary"]["entry_count"] == 3
    assert {entry["context_type"] for entry in ledger["recent_entries"]} == {
        "compact_pack",
        "context_bundle",
        "memory_pack",
    }
