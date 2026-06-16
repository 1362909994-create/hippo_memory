from __future__ import annotations

from hippocampus_memory.db import build_fts_query
from hippocampus_memory.lsp_diagnostics import CodeDiagnostic
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.utils import cjk_search_terms, text_similarity


def test_db_initializes_schema(db):
    stats = db.stats()
    assert stats["memories"] == 0
    assert stats["projects"] == 0
    assert stats["token_ledger_entries"] == 0
    assert stats["db_path"].endswith("hippo.db")


def test_token_ledger_is_project_scoped(db):
    alpha = db.insert_token_ledger(
        project="alpha",
        intent="alpha task",
        context_type="memory_pack",
        baseline_tokens=100,
        output_tokens=40,
    )
    db.insert_token_ledger(
        project="beta",
        intent="beta task",
        context_type="memory_pack",
        baseline_tokens=100,
        output_tokens=90,
    )

    summary = db.token_ledger_summary("alpha")
    entries = db.list_token_ledger("alpha")

    assert alpha
    assert summary["entry_count"] == 1
    assert summary["saved_tokens"] == 60
    assert len(entries) == 1
    assert entries[0]["project"] == "alpha"


def test_code_diagnostics_are_project_scoped(db):
    db.replace_code_diagnostics(
        project="alpha",
        source="basedpyright",
        diagnostics=[
            CodeDiagnostic(
                relative_path="app.py",
                severity="error",
                message="Undefined name",
                line=3,
                column=5,
                rule="reportUndefinedVariable",
                source="basedpyright",
            )
        ],
    )
    db.replace_code_diagnostics(
        project="beta",
        source="basedpyright",
        diagnostics=[
            CodeDiagnostic(
                relative_path="beta.py",
                severity="warning",
                message="Beta warning",
                line=1,
                column=1,
                source="basedpyright",
            )
        ],
    )

    diagnostics = db.list_code_diagnostics("alpha")

    assert len(diagnostics) == 1
    assert diagnostics[0]["relative_path"] == "app.py"
    assert db.stats()["code_diagnostics"] == 2


def test_supersede_memory_marks_old_memory_outdated(db):
    writer = MemoryWriter(db)
    old = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Old storage fact.",
    )
    new = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="New storage fact.",
    )

    assert db.supersede_memory(old.memory_id, new.memory_id)

    old_memory = db.get_memory(old.memory_id)
    assert old_memory.status == "outdated"
    assert old_memory.metadata["superseded_by"] == new.memory_id


def test_supersede_memory_does_not_cross_projects(db):
    writer = MemoryWriter(db)
    old = writer.write(project="alpha", memory_type="task_state", content="Alpha state.")
    new = writer.write(project="beta", memory_type="task_state", content="Beta state.")

    assert not db.supersede_memory(old.memory_id, new.memory_id)


def test_build_fts_query_expands_chinese_ngrams():
    query = build_fts_query("上下文压缩")

    assert '"上下文压缩"' in query
    assert '"上下"' in query
    assert '"上下文"' in query


def test_text_quality_fallbacks_handle_chinese_and_near_duplicates():
    terms = cjk_search_terms("上下文压缩")

    assert "上下文压缩" in terms
    assert "上下" in terms
    assert "上下文" in terms
    assert text_similarity(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
    ) > 0.9
