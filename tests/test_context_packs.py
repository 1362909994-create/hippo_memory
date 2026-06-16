from __future__ import annotations

from hippocampus_memory.change_planner import ChangePlanner
from hippocampus_memory.code_map import CodeMapBuilder
from hippocampus_memory.context_bundle import ContextBundleBuilder
from hippocampus_memory.lsp_diagnostics import CodeDiagnostic
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.project_indexer import ProjectIndexer
from hippocampus_memory.project_profile import ProjectProfileBuilder
from hippocampus_memory.utils import estimate_tokens


def test_project_profile_includes_overview_risks_and_unknowns(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("# Demo\nA memory project.\n", encoding="utf-8")
    MemoryWriter(db).write(
        project="demo",
        memory_type="project_context",
        content="Demo is a local AI memory project.",
    )
    ProjectIndexer(db).index_project(root, "demo")

    profile = ProjectProfileBuilder(db).build("demo")

    assert "Project Profile:" in profile
    assert "Implementation shape:" in profile
    assert "Risk points:" in profile
    assert "Demo is a local AI memory project." in profile


def test_project_profile_infers_project_shape_from_index(db, tmp_path):
    root = tmp_path / "project"
    (root / "src" / "demo_pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "examples").mkdir()
    (root / "README.md").write_text("# Demo\nA local MCP server.\n", encoding="utf-8")
    (root / "src" / "demo_pkg" / "server.py").write_text(
        "class DemoServer:\n    def handle(self):\n        return 'ok'\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_server.py").write_text("def test_ok():\n    assert True\n")
    (root / "examples" / "demo.py").write_text("print('demo')\n")
    ProjectIndexer(db).index_project(root, "demo")

    profile = ProjectProfileBuilder(db).build("demo")

    assert "Inferred project understanding:" in profile
    assert "not confirmed memory" in profile
    assert "Likely source package roots: demo_pkg." in profile
    assert "Tests indexed:" in profile
    assert "Examples indexed:" in profile
    assert "DemoServer" in profile


def test_code_map_shows_symbols_and_imports(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "retriever.py").write_text(
        "import sqlite3\n\nclass Retriever:\n    def search(self):\n        return []\n",
        encoding="utf-8",
    )
    ProjectIndexer(db).index_project(root, "demo")

    code_map = CodeMapBuilder(db).build("demo", query="retriever search")

    assert "Code Map:" in code_map
    assert "retriever.py" in code_map
    assert "Retriever" in code_map
    assert "search" in code_map
    assert "Calls:" in code_map


def test_change_planner_outputs_minimal_change_guidance_and_tests(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "retriever.py").write_text(
        "class Retriever:\n    def search(self):\n        return []\n",
        encoding="utf-8",
    )
    ProjectIndexer(db).index_project(root, "demo")

    impact = ChangePlanner(db).plan("change search ranking", project="demo")

    assert "Code Impact Pack:" in impact
    assert "retriever.py" in impact
    assert "Search changes must preserve" in impact
    assert "tests/test_retriever.py" in impact


def test_change_planner_includes_call_impact(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text(
        "from b import helper\n\ndef main():\n    return helper()\n",
        encoding="utf-8",
    )
    (root / "b.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    db.replace_code_diagnostics(
        project="demo",
        source="basedpyright",
        diagnostics=[
            CodeDiagnostic(
                relative_path="b.py",
                severity="error",
                message="Return type is unknown",
                line=1,
                column=5,
                rule="reportUnknownVariableType",
                source="basedpyright",
            )
        ],
    )

    impact = ChangePlanner(db).plan("change helper behavior", project="demo")

    assert "Precise symbol impact:" in impact
    assert "helper [function] b.py:1" in impact
    assert "a.py:4 calls helper in main" in impact
    assert "Stored LSP diagnostics:" in impact
    assert "error b.py:1:5 Return type is unknown" in impact
    assert "Potential call impact:" in impact
    assert "a.py calls helper in b.py." in impact


def test_context_bundle_auto_uses_lean_strategy_for_focused_tasks(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "callback.py").write_text(
        "def callback_pack():\n    return 'small'\n",
        encoding="utf-8",
    )
    ProjectIndexer(db).index_project(root, "demo")
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Callback dedupe should avoid repeated injection.",
        importance=0.9,
    )
    builder = ContextBundleBuilder(db)

    auto = builder.build("demo", "fix callback dedupe", strategy="auto")
    full = builder.build("demo", "fix callback dedupe", strategy="full")

    assert "Strategy: auto:lean" in auto
    assert estimate_tokens(auto) < estimate_tokens(full)


def test_context_bundle_auto_keeps_full_strategy_for_project_overview(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")

    bundle = ContextBundleBuilder(db).build("demo", "understand project overview")

    assert "Strategy: auto:full" in bundle
