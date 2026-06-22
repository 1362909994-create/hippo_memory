from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hippocampus_memory.change_planner import ChangePlanner
from hippocampus_memory.code_intelligence import CodeIntelligence
from hippocampus_memory.code_map import CodeMapBuilder
from hippocampus_memory.db import Database
from hippocampus_memory.lsp_diagnostics import run_python_diagnostics
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator import TurnOrchestrator
from hippocampus_memory.project_profile import ProjectProfileBuilder
from hippocampus_memory.utils import dumps_json

TOOLS: dict[str, dict[str, Any]] = {
    "memory.write": {
        "description": "Write one long-term memory.",
        "inputSchema": {
            "type": "object",
            "required": ["content", "memory_type"],
            "properties": {
                "content": {"type": "string"},
                "memory_type": {"type": "string"},
                "project": {"type": "string"},
                "confidence": {"type": "number"},
                "importance": {"type": "number"},
            },
        },
    },
    "memory.search": {
        "description": "Search memories.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "project": {"type": "string"},
                "top_k": {"type": "integer"},
                "search_mode": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "dedupe_results": {"type": "boolean"},
            },
        },
    },
    "memory.pack": {
        "description": "Generate a Memory Pack.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "project": {"type": "string"},
                "source_chunk_limit": {"type": "integer"},
                "compact": {"type": "boolean"},
                "exclude_memory_ids": {"type": "array", "items": {"type": "string"}},
                "session_dedupe": {"type": "boolean"},
            },
        },
    },
    "memory.auto_store": {
        "description": "Automatically write, queue, or skip useful long-term memories.",
        "inputSchema": {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "project": {"type": "string"},
                "source": {"type": "string"},
                "mode": {"type": "string"},
                "max_candidates": {"type": "integer"},
                "allow_sensitive": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
            },
        },
    },
    "project.profile": {
        "description": "Generate a Project Profile.",
        "inputSchema": {
            "type": "object",
            "required": ["project"],
            "properties": {"project": {"type": "string"}},
        },
    },
    "project.impact": {
        "description": "Generate a Code Impact Pack.",
        "inputSchema": {
            "type": "object",
            "required": ["intent", "project"],
            "properties": {"intent": {"type": "string"}, "project": {"type": "string"}},
        },
    },
    "code.symbols": {
        "description": "List indexed code symbols with definition locations.",
        "inputSchema": {
            "type": "object",
            "required": ["project"],
            "properties": {
                "project": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    "code.references": {
        "description": "List indexed references/calls for a symbol.",
        "inputSchema": {
            "type": "object",
            "required": ["project", "symbol"],
            "properties": {
                "project": {"type": "string"},
                "symbol": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    "code.intelligence": {
        "description": "Generate a symbol-level impact summary for an intent.",
        "inputSchema": {
            "type": "object",
            "required": ["project", "intent"],
            "properties": {
                "project": {"type": "string"},
                "intent": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    "code.diagnostics": {
        "description": "List or refresh stored Python LSP diagnostics.",
        "inputSchema": {
            "type": "object",
            "required": ["project"],
            "properties": {
                "project": {"type": "string"},
                "path": {"type": "string"},
                "checker": {"type": "string"},
                "refresh": {"type": "boolean"},
                "limit": {"type": "integer"},
            },
        },
    },
    "context.bundle": {
        "description": "Generate a full Context Bundle.",
        "inputSchema": {
            "type": "object",
            "required": ["project", "intent"],
            "properties": {
                "project": {"type": "string"},
                "intent": {"type": "string"},
                "strategy": {"type": "string"},
            },
        },
    },
    "context.auto": {
        "description": "Automatically decide whether and how to recall external memory.",
        "inputSchema": {
            "type": "object",
            "required": ["intent"],
            "properties": {
                "intent": {"type": "string"},
                "project": {"type": "string"},
                "session_key": {"type": "string"},
                "max_tokens": {"type": "integer"},
                "include_code_map": {"type": "boolean"},
            },
        },
    },
    "context.callback": {
        "description": "Generate a project-scoped callback pack and remember injected memories.",
        "inputSchema": {
            "type": "object",
            "required": ["project", "intent"],
            "properties": {
                "project": {"type": "string"},
                "intent": {"type": "string"},
                "session_key": {"type": "string"},
                "max_tokens": {"type": "integer"},
                "source_chunk_limit": {"type": "integer"},
                "compact": {"type": "boolean"},
            },
        },
    },
    "candidate.list": {
        "description": "List queued memory candidates.",
        "inputSchema": {
            "type": "object",
            "properties": {"project": {"type": "string"}, "status": {"type": "string"}},
        },
    },
    "conflict.list": {
        "description": "List open memory conflicts.",
        "inputSchema": {
            "type": "object",
            "properties": {"project": {"type": "string"}, "limit": {"type": "integer"}},
        },
    },
}

SAFE_TOOL_NAMES = {name: name.replace(".", "_") for name in TOOLS}
CANONICAL_TOOL_NAMES = {safe_name: name for name, safe_name in SAFE_TOOL_NAMES.items()}
SERVER_INSTRUCTIONS = (
    "Hippocampus Memory provides external memory and compact context. "
    "For non-trivial coding, debugging, project-review, or architecture tasks, call "
    "`context_auto`/`hippo_memory_context_auto` before reading lots of files. "
    "When the result includes `token_savings_text` or says "
    "`Show this token savings line to the user:`, include that token savings line in "
    "the final user-facing reply. Near the end of meaningful work, call "
    "`memory_auto_store`/`hippo_memory_memory_auto_store` with a concise session summary."
)


class HippoMcpServer:
    def __init__(
        self,
        db: Database,
        *,
        safe_tool_names: bool = False,
        default_project: str | None = None,
    ) -> None:
        self.db = db
        self.safe_tool_names = safe_tool_names
        self.default_project = default_project

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hippocampus-memory", "version": "0.1.0"},
                    "instructions": SERVER_INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method in {"tools/list", "tools.list"}:
                result = {
                    "tools": [
                        {
                            "name": self._public_tool_name(name),
                            "description": meta["description"],
                            "inputSchema": meta["inputSchema"],
                        }
                        for name, meta in TOOLS.items()
                    ]
                }
            elif method in {"tools/call", "tools.call"}:
                result = _tool_result(
                    self.call_tool(str(params.get("name")), params.get("arguments") or {})
                )
            else:
                return _error(request_id, -32601, f"Unknown method: {method}")
        except Exception as exc:  # pragma: no cover - defensive boundary for stdio server
            return _error(request_id, -32000, str(exc))
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _public_tool_name(self, name: str) -> str:
        if not self.safe_tool_names:
            return name
        return SAFE_TOOL_NAMES[name]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        name = CANONICAL_TOOL_NAMES.get(name, name)
        if name == "memory.write":
            result = MemoryWriter(self.db).write(**arguments)
            return {"memory_id": result.memory_id, "created": result.created}
        if name == "memory.search":
            arguments = self._with_default_project(arguments)
            turn = TurnOrchestrator(self.db).run_turn(
                str(arguments["query"]),
                context={**arguments, "operation": "memory_search", "writeback": False},
                mode="preview",
            )
            return turn.runtime_payload()
        if name == "memory.pack":
            arguments = self._with_default_project(arguments)
            turn = TurnOrchestrator(self.db).run_turn(
                str(arguments["query"]),
                context={**arguments, "operation": "memory_pack", "writeback": False},
                mode="preview",
            )
            return turn.runtime_payload()
        if name == "memory.auto_store":
            arguments = self._with_default_project(arguments)
            store_mode = str(arguments.get("mode", "auto"))
            dry_run = bool(arguments.get("dry_run", False))
            turn = TurnOrchestrator(self.db).run_turn(
                str(arguments["text"]),
                context={
                    **arguments,
                    "operation": "memory_auto_store",
                    "store_mode": store_mode,
                    "writeback": False,
                },
                mode="preview" if dry_run or store_mode == "preview" else "write",
            )
            return turn.runtime_payload()
        if name == "project.profile":
            return {"text": ProjectProfileBuilder(self.db).build(str(arguments["project"]))}
        if name == "project.impact":
            return {"text": ChangePlanner(self.db).plan(**arguments)}
        if name == "code.symbols":
            return {
                "symbols": CodeIntelligence(self.db).search_symbols(
                    str(arguments["project"]),
                    query=arguments.get("query"),
                    limit=int(arguments.get("limit", 20)),
                )
            }
        if name == "code.references":
            return {
                "references": CodeIntelligence(self.db).references(
                    str(arguments["project"]),
                    symbol=str(arguments["symbol"]),
                    limit=int(arguments.get("limit", 50)),
                )
            }
        if name == "code.intelligence":
            project = str(arguments["project"])
            intent = str(arguments["intent"])
            limit = int(arguments.get("limit", 8))
            files = CodeMapBuilder(self.db).relevant_files(project, intent, limit=limit)
            lines = CodeIntelligence(self.db).impact_lines(project, intent, files, limit=limit)
            return {"text": "\n".join(lines), "lines": lines}
        if name == "code.diagnostics":
            project = str(arguments["project"])
            if not arguments.get("refresh"):
                return {
                    "diagnostics": CodeIntelligence(self.db).diagnostics(
                        project,
                        limit=int(arguments.get("limit", 100)),
                    )
                }
            root_path = arguments.get("path") or _project_root_path(self.db, project) or "."
            result = run_python_diagnostics(root_path, checker=arguments.get("checker"))
            if result["available"]:
                self.db.replace_code_diagnostics(
                    project=project,
                    diagnostics=result["diagnostics"],
                    source=Path(str(result["tool"])).name,
                )
            return {
                **result,
                "diagnostics": [asdict(diagnostic) for diagnostic in result["diagnostics"]],
            }
        if name == "context.bundle":
            arguments = self._with_default_project(arguments)
            turn = TurnOrchestrator(self.db).run_turn(
                str(arguments["intent"]),
                context={
                    **arguments,
                    "operation": "context_bundle",
                    "bundle_strategy": arguments.get("strategy", "auto"),
                    "writeback": False,
                },
                mode="preview",
            )
            return turn.runtime_payload()
        if name == "context.auto":
            arguments = self._with_default_project(arguments)
            turn = TurnOrchestrator(self.db).run_turn(
                str(arguments["intent"]),
                context={
                    **arguments,
                    "operation": "auto_context",
                    "track_token_savings": True,
                    "include_savings_in_text": True,
                },
                mode="preview",
            )
            return turn.runtime_payload()
        if name == "context.callback":
            arguments = self._with_default_project(arguments)
            turn = TurnOrchestrator(self.db).run_turn(
                str(arguments["intent"]),
                context={**arguments, "operation": "context_callback", "writeback": False},
                mode="preview",
            )
            return turn.runtime_payload()
        if name == "candidate.list":
            return {
                "candidates": self.db.list_candidates(
                    project=arguments.get("project"),
                    status=arguments.get("status", "pending"),
                )
            }
        if name == "conflict.list":
            return {
                "conflicts": self.db.list_conflicts(
                    project=arguments.get("project"),
                    limit=int(arguments.get("limit", 100)),
                )
            }
        raise ValueError(f"Unknown tool: {name}")

    def _with_default_project(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.get("project") or not self.default_project:
            return arguments
        return {**arguments, "project": self.default_project}


def serve_stdio(
    db: Database,
    *,
    safe_tool_names: bool = False,
    default_project: str | None = None,
) -> None:
    server = HippoMcpServer(
        db,
        safe_tool_names=safe_tool_names,
        default_project=default_project,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        response = server.handle(json.loads(line))
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        text = payload["text"]
    else:
        text = dumps_json(payload)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
    }


def _project_root_path(db: Database, project: str) -> str | None:
    record = db.get_project(project)
    if not record or not record.get("root_path"):
        return None
    return str(record["root_path"])
