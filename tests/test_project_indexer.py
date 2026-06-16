from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.project_indexer import ProjectIndexer


def test_project_indexer_ignores_common_directories(db, tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / ".git").mkdir()
    (root / "build").mkdir()
    (root / "src" / "app.py").write_text(
        "import os\n\ndef main():\n    return os.name\n",
        encoding="utf-8",
    )
    (root / "node_modules" / "lib.py").write_text("def ignored(): pass\n", encoding="utf-8")
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "build" / "out.py").write_text("def ignored(): pass\n", encoding="utf-8")

    result = ProjectIndexer(db).index_project(root, "demo")

    assert result["indexed_files"] == 1
    with db.connect() as conn:
        rows = conn.execute("SELECT relative_path FROM files").fetchall()
    assert [row["relative_path"] for row in rows] == ["src/app.py"]


def test_project_indexer_ignores_generated_metadata_directories(db, tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "demo.egg-info").mkdir()
    (root / "src" / "app.py").write_text("def main():\n    return True\n", encoding="utf-8")
    (root / "demo.egg-info" / "dependency_links.txt").write_text("", encoding="utf-8")

    result = ProjectIndexer(db).index_project(root, "demo")

    assert result["indexed_files"] == 1
    with db.connect() as conn:
        paths = [row["relative_path"] for row in conn.execute("SELECT relative_path FROM files")]
    assert paths == ["src/app.py"]


def test_project_indexer_skips_effectively_empty_text_files(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("def main():\n    return True\n", encoding="utf-8")
    (root / "empty.txt").write_text("\n\t  \n", encoding="utf-8")

    result = ProjectIndexer(db).index_project(root, "demo")

    assert result["indexed_files"] == 1
    assert result["skipped_files"] == 1
    with db.connect() as conn:
        paths = [row["relative_path"] for row in conn.execute("SELECT relative_path FROM files")]
    assert paths == ["app.py"]


def test_project_indexer_marks_removed_files_missing(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    keep = root / "keep.py"
    remove = root / "remove.py"
    keep.write_text("def keep():\n    return True\n", encoding="utf-8")
    remove.write_text("def remove():\n    return False\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")

    remove.unlink()
    result = ProjectIndexer(db).index_project(root, "demo")

    assert result["stale_files"] == 1
    with db.connect() as conn:
        row = conn.execute(
            "SELECT status FROM files WHERE project = 'demo' AND relative_path = 'remove.py'"
        ).fetchone()
    assert row["status"] == "missing"


def test_project_indexer_uses_python_ast_for_calls(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text(
        "import json\n\nclass App:\n    pass\n\ndef run():\n    return json.dumps({'ok': True})\n",
        encoding="utf-8",
    )

    ProjectIndexer(db).index_project(root, "demo")

    with db.connect() as conn:
        row = conn.execute(
            "SELECT summary, symbols, imports, calls FROM files WHERE relative_path = 'app.py'"
        ).fetchone()
    assert "App" in row["symbols"]
    assert "run" in row["symbols"]
    assert "json" in row["imports"]
    assert "dumps" in row["calls"]
    assert "symbols=App, run" in row["summary"]


def test_project_indexer_writes_python_lsp_symbols_and_references(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text(
        "\n".join(
            [
                "class Service:",
                "    def run(self):",
                "        return helper(self.name)",
                "",
                "def helper(value):",
                "    \"\"\"Return the value.\"\"\"",
                "    return value",
            ]
        ),
        encoding="utf-8",
    )

    ProjectIndexer(db).index_project(root, "demo")

    symbols = db.list_code_symbols("demo", query="helper")
    references = db.list_code_references("demo", "helper")

    assert any(symbol["qualified_name"] == "helper" for symbol in symbols)
    assert any(symbol["qualified_name"] == "Service.run" for symbol in db.list_code_symbols("demo"))
    assert any(symbol["docstring"] == "Return the value." for symbol in symbols)
    assert any(reference["kind"] == "call" for reference in references)
    assert any("return helper" in str(reference["context"]) for reference in references)


def test_project_indexer_refreshes_lsp_index_when_file_changes(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("def old_name():\n    return 1\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")

    app.write_text("def new_name():\n    return 2\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")

    assert not db.list_code_symbols("demo", query="old_name")
    assert db.list_code_symbols("demo", query="new_name")


def test_project_indexer_clears_lsp_index_for_missing_files(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("def removed():\n    return 1\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")

    app.unlink()
    ProjectIndexer(db).index_project(root, "demo")

    assert not db.list_code_symbols("demo", query="removed")


def test_project_indexer_archives_old_chunks_when_file_changes(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("def old_name():\n    return 'old'\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    with db.connect() as conn:
        old_memory_id = conn.execute(
            "SELECT memory_id FROM chunks WHERE project = 'demo'"
        ).fetchone()["memory_id"]

    app.write_text("def new_name():\n    return 'new'\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")

    assert db.get_memory(old_memory_id).status == "archived"
    with db.connect() as conn:
        active_source_chunks = conn.execute(
            """
            SELECT COUNT(*)
            FROM memories
            WHERE project = 'demo'
              AND memory_type = 'source_chunk'
              AND status = 'active'
            """
        ).fetchone()[0]
        current_chunk = conn.execute(
            "SELECT content FROM chunks WHERE project = 'demo'"
        ).fetchone()["content"]
    assert active_source_chunks == 1
    assert "new_name" in current_chunk


def test_project_indexer_source_chunk_summaries_include_line_range(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("\n".join(f"line_{idx} = {idx}" for idx in range(130)), encoding="utf-8")

    ProjectIndexer(db).index_project(root, "demo")

    with db.connect() as conn:
        summaries = [
            row["summary"]
            for row in conn.execute(
                "SELECT summary FROM chunks WHERE project = 'demo' ORDER BY start_line"
            )
        ]
    assert "app.py:L1-120:" in summaries[0]
    assert "app.py:L121-130:" in summaries[1]


def test_project_indexer_refreshes_legacy_chunk_summaries(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("def stable():\n    return 'same'\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    with db.connect() as conn:
        old_memory_id = conn.execute("SELECT memory_id FROM chunks").fetchone()["memory_id"]
        conn.execute("UPDATE chunks SET summary = 'app.py: legacy summary'")
        conn.execute(
            "UPDATE memories SET summary = 'app.py: legacy summary' WHERE id = ?",
            (old_memory_id,),
        )

    ProjectIndexer(db).index_project(root, "demo")

    assert db.get_memory(old_memory_id).status == "archived"
    with db.connect() as conn:
        row = conn.execute("SELECT memory_id, summary FROM chunks").fetchone()
    assert row["memory_id"] != old_memory_id
    assert "app.py:L1-2:" in row["summary"]


def test_project_indexer_refreshes_legacy_memory_summaries(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("def stable():\n    return 'same'\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    with db.connect() as conn:
        old_memory_id = conn.execute("SELECT memory_id FROM chunks").fetchone()["memory_id"]
        conn.execute(
            "UPDATE memories SET summary = 'app.py: legacy summary' WHERE id = ?",
            (old_memory_id,),
        )

    ProjectIndexer(db).index_project(root, "demo")

    assert db.get_memory(old_memory_id).status == "archived"
    with db.connect() as conn:
        row = conn.execute("SELECT memory_id FROM chunks").fetchone()
    assert row["memory_id"] != old_memory_id
    assert "app.py:L1-2:" in db.get_memory(row["memory_id"]).summary


def test_project_indexer_archives_unlinked_project_source_memories(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("def stable():\n    return 'same'\n", encoding="utf-8")
    orphan = MemoryWriter(db).write(
        project="demo",
        memory_type="source_chunk",
        content="old unlinked project index chunk",
        source="project_index",
    )
    keep = MemoryWriter(db).write(
        project="demo",
        memory_type="source_chunk",
        content="user managed source note should remain active",
        source="manual",
    )

    result = ProjectIndexer(db).index_project(root, "demo")

    assert result["orphaned_source_memories"] == 1
    assert db.get_memory(orphan.memory_id).status == "archived"
    assert db.get_memory(keep.memory_id).status == "active"


def test_project_indexer_does_not_rewrite_unchanged_chunks(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    app = root / "app.py"
    app.write_text("def stable():\n    return 'same'\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    with db.connect() as conn:
        first_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        first_chunk_id = conn.execute("SELECT id FROM chunks").fetchone()["id"]

    ProjectIndexer(db).index_project(root, "demo")

    with db.connect() as conn:
        second_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        second_chunk_id = conn.execute("SELECT id FROM chunks").fetchone()["id"]
    assert second_count == first_count
    assert second_chunk_id == first_chunk_id
