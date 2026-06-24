from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import hippocampus_memory
from hippocampus_memory.orchestrator import MemoryScheduler, TurnOrchestrator
from tests.utils.deterministic_replay import DeterministicReplay
from tests.utils.system_simulator import SimulationEvent

ENTRYPOINTS = [
    Path("hippocampus_memory/cli.py"),
    Path("hippocampus_memory/mcp_server.py"),
    Path("hippocampus_memory/api.py"),
]

FORBIDDEN_RUNTIME_IMPORTS = {
    "hippocampus_memory.callback": {"callback_pack"},
    "hippocampus_memory.context_bundle": {"ContextBundleBuilder"},
    "hippocampus_memory.memory_policy": {"auto_store_memories"},
    "hippocampus_memory.packer": {"MemoryPacker"},
    "hippocampus_memory.ranker": {"RANKER_VERSION", "explain_memory_score"},
    "hippocampus_memory.recall_policy": {"build_auto_context", "decide_recall"},
    "hippocampus_memory.retriever": {"Retriever"},
}

FORBIDDEN_RUNTIME_CALLS = {
    "callback_pack",
    "ContextBundleBuilder",
    "auto_store_memories",
    "MemoryPacker",
    "explain_memory_score",
    "build_auto_context",
    "decide_recall",
    "Retriever",
}


def test_all_hippocampus_modules_import_without_circular_failures() -> None:
    failures: list[str] = []
    for module in pkgutil.walk_packages(hippocampus_memory.__path__, "hippocampus_memory."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # pragma: no cover - assertion reports import graph failure
            failures.append(f"{module.name}: {type(exc).__name__}: {exc}")

    assert not failures
    assert TurnOrchestrator is not None
    assert MemoryScheduler is not None


def test_cli_mcp_api_do_not_bypass_turn_orchestrator() -> None:
    violations: dict[str, list[str]] = {}
    for path in ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_orchestrator = False
        path_violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                if node.module == "hippocampus_memory.orchestrator":
                    imported_orchestrator = "TurnOrchestrator" in names
                forbidden = FORBIDDEN_RUNTIME_IMPORTS.get(node.module or "")
                if forbidden:
                    for name in sorted(names & forbidden):
                        path_violations.append(f"forbidden import {node.module}.{name}")
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                if call_name in FORBIDDEN_RUNTIME_CALLS:
                    path_violations.append(f"forbidden runtime call {call_name}()")
        if not imported_orchestrator:
            path_violations.append("missing TurnOrchestrator import")
        if path_violations:
            violations[str(path)] = path_violations

    assert not violations


def test_simulation_harness_replays_deterministically(tmp_path) -> None:
    events = [
        SimulationEvent("write", {"content": "Decision: keep replay deterministic."}),
        SimulationEvent("recall", {"query": "continue replay deterministic"}),
        SimulationEvent("scheduler", {}),
        SimulationEvent("policy_feedback", {"signal": "successful_recall"}),
        SimulationEvent("compress", {}),
    ]

    result = DeterministicReplay(tmp_path).replay(events)

    assert result.matched, result.differences


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
