from __future__ import annotations

from hippocampus_memory.change_planner import ChangePlanner
from hippocampus_memory.code_map import CodeMapBuilder
from hippocampus_memory.db import Database
from hippocampus_memory.git_utils import format_git_snapshot, git_snapshot
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.project_profile import ProjectProfileBuilder
from hippocampus_memory.utils import estimate_tokens, utc_now


class ContextBundleBuilder:
    def __init__(self, db: Database) -> None:
        self.db = db

    def build(
        self,
        project: str,
        intent: str,
        max_tokens: int = 3500,
        include_code_map: bool = True,
        strategy: str = "auto",
    ) -> str:
        strategy = _normalize_strategy(strategy)
        if strategy == "auto":
            return self._build_auto(project, intent, max_tokens, include_code_map)
        if strategy == "lean":
            return self._build_lean(project, intent, max_tokens, include_code_map, "lean")
        if strategy == "pack":
            return self._build_pack_only(project, intent, max_tokens)
        return self._build_full(project, intent, max_tokens, include_code_map, "full")

    def _build_auto(
        self,
        project: str,
        intent: str,
        max_tokens: int,
        include_code_map: bool,
    ) -> str:
        if _wants_project_overview(intent):
            return self._build_full(project, intent, max_tokens, include_code_map, "auto:full")
        return self._build_lean(project, intent, max_tokens, include_code_map, "auto:lean")

    def _build_full(
        self,
        project: str,
        intent: str,
        max_tokens: int,
        include_code_map: bool,
        strategy: str,
    ) -> str:
        sections = [
            _section("Git Snapshot", format_git_snapshot(git_snapshot())),
            _section("Project Profile", ProjectProfileBuilder(self.db).build(project)),
            _section("Memory Pack", MemoryPacker(self.db).pack(intent, project=project)),
            _section("Code Impact Pack", ChangePlanner(self.db).plan(intent, project=project)),
        ]
        if include_code_map:
            sections.append(
                _section(
                    "Code Map",
                    CodeMapBuilder(self.db).build(project, query=intent, max_files=10),
                )
            )

        header = _header(project, intent, strategy)
        return _trim("\n".join([*header, *sections]), max_tokens=max_tokens)

    def _build_lean(
        self,
        project: str,
        intent: str,
        max_tokens: int,
        include_code_map: bool,
        strategy: str,
    ) -> str:
        impact_budget = max(500, min(1000, max_tokens // 2))
        pack_budget = max(350, min(900, max_tokens // 3))
        profile = ProjectProfileBuilder(self.db).build(project)
        sections = [
            _section("Project Profile", _compact_lines(profile, limit=10)),
            _section(
                "Memory Pack",
                MemoryPacker(self.db).pack(
                    intent,
                    project=project,
                    max_tokens=pack_budget,
                    compact=True,
                ),
            ),
            _section(
                "Code Impact Pack",
                ChangePlanner(self.db).plan(
                    intent,
                    project=project,
                    max_tokens=impact_budget,
                ),
            ),
        ]
        if include_code_map:
            sections.append(
                _section(
                    "Code Map",
                    CodeMapBuilder(self.db).build(project, query=intent, max_files=5),
                )
            )
        header = _header(project, intent, strategy)
        return _trim("\n".join([*header, *sections]), max_tokens=max_tokens)

    def _build_pack_only(self, project: str, intent: str, max_tokens: int) -> str:
        header = _header(project, intent, "pack")
        pack = MemoryPacker(self.db).pack(
            intent,
            project=project,
            max_tokens=max_tokens,
            compact=True,
        )
        return _trim("\n".join([*header, _section("Memory Pack", pack)]), max_tokens=max_tokens)


def _section(title: str, body: str) -> str:
    return f"\n--- {title} ---\n{body.strip()}"


def _header(project: str, intent: str, strategy: str) -> list[str]:
    return [
        "Hippocampus Context Bundle",
        f"Project: {project}",
        f"Intent: {intent.strip()}",
        f"Strategy: {strategy}",
        f"Generated at: {utc_now()}",
        "Use this as compressed external memory for the current coding task.",
        "Prefer minimal, well-tested changes that respect the risks and invariants below.",
    ]


def _normalize_strategy(strategy: str) -> str:
    normalized = strategy.strip().casefold()
    if normalized in {"auto", "full", "lean", "pack"}:
        return normalized
    raise ValueError("strategy must be one of: auto, full, lean, pack")


def _wants_project_overview(intent: str) -> bool:
    text = intent.casefold()
    terms = [
        "understand project",
        "project overview",
        "project profile",
        "onboard",
        "architecture",
        "整体项目",
        "项目理解",
        "项目概览",
        "架构",
    ]
    return any(term in text for term in terms)


def _compact_lines(text: str, limit: int) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join([*lines[:limit], "[Profile truncated for lean context.]"])


def _trim(text: str, max_tokens: int) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line])
        if estimate_tokens(candidate) > max_tokens:
            kept.append("[Context truncated to token budget.]")
            break
        kept.append(line)
    return "\n".join(kept)
