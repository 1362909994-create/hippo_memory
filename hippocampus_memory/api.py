from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException

from hippocampus_memory.change_planner import ChangePlanner
from hippocampus_memory.code_intelligence import CodeIntelligence
from hippocampus_memory.code_map import CodeMapBuilder
from hippocampus_memory.config import Settings
from hippocampus_memory.consolidator import Consolidator
from hippocampus_memory.db import Database
from hippocampus_memory.lsp_diagnostics import run_python_diagnostics
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator import TurnOrchestrator
from hippocampus_memory.project_indexer import ProjectIndexer
from hippocampus_memory.project_profile import ProjectProfileBuilder
from hippocampus_memory.schemas import (
    AutoContextRequest,
    AutoStoreRequest,
    CandidateAcceptRequest,
    CodeDiagnosticsRequest,
    CodeIntelligenceRequest,
    CodeMapRequest,
    CodeReferenceRequest,
    CodeSymbolRequest,
    ConflictResolveRequest,
    ConsolidateRequest,
    ForgetRequest,
    HealthResponse,
    ImpactRequest,
    MemoryPackRequest,
    MemoryPackResponse,
    MemorySearchItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
    ProjectIndexRequest,
    SessionSummarizeRequest,
    TextPackResponse,
)
from hippocampus_memory.session_ingestor import (
    accept_candidate,
    discard_candidate,
    queue_summary_candidates,
    write_summary_candidates,
)
from hippocampus_memory.summarizer import summarize_session as build_session_summary
from hippocampus_memory.summarizer import summarize_session_file


def create_app(settings: Settings | None = None) -> FastAPI:
    db = Database(settings=settings)
    db.initialize()
    app = FastAPI(title="hippocampus-memory", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", db_path=str(db.path))

    @app.post("/memory/write", response_model=MemoryWriteResponse)
    def write_memory(payload: MemoryWriteRequest) -> MemoryWriteResponse:
        try:
            result = MemoryWriter(db).write(**payload.model_dump(by_alias=False))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MemoryWriteResponse(
            memory_id=result.memory_id,
            created=result.created,
            duplicate=result.duplicate,
        )

    @app.post("/memory/search", response_model=MemorySearchResponse)
    def search_memory(payload: MemorySearchRequest) -> MemorySearchResponse:
        turn = TurnOrchestrator(db).run_turn(
            payload.query,
            context={
                **payload.model_dump(),
                "operation": "memory_search",
                "writeback": False,
            },
            mode="preview",
        )
        runtime = turn.runtime_payload()
        results = [MemorySearchItem(**item) for item in runtime["results"]]
        retrieved = [MemorySearchItem(**item) for item in runtime["retrieved_memories"]]
        selected = [MemorySearchItem(**item) for item in runtime["selected_memories"]]
        return MemorySearchResponse(
            results=results,
            injected_context=runtime["injected_context"],
            execution_trace=runtime["execution_trace"],
            retrieved_memories=retrieved,
            selected_memories=selected,
            context_budget=runtime["context_budget"],
        )

    @app.post("/memory/pack", response_model=MemoryPackResponse)
    def pack_memory(payload: MemoryPackRequest) -> MemoryPackResponse:
        turn = TurnOrchestrator(db).run_turn(
            payload.query,
            context={
                **payload.model_dump(),
                "operation": "memory_pack",
                "writeback": False,
            },
            mode="preview",
        )
        runtime = turn.runtime_payload()
        retrieved = [MemorySearchItem(**item) for item in runtime["retrieved_memories"]]
        selected = [MemorySearchItem(**item) for item in runtime["selected_memories"]]
        return MemoryPackResponse(
            pack=turn.injected_context,
            injected_context=runtime["injected_context"],
            execution_trace=runtime["execution_trace"],
            retrieved_memories=retrieved,
            selected_memories=selected,
            context_budget=runtime["context_budget"],
        )

    @app.post("/memory/auto-store")
    def auto_store(payload: AutoStoreRequest) -> dict:
        try:
            turn = TurnOrchestrator(db).run_turn(
                payload.text,
                context={
                    **payload.model_dump(),
                    "operation": "memory_auto_store",
                    "store_mode": payload.mode,
                    "writeback": False,
                },
                mode="preview" if payload.dry_run or payload.mode == "preview" else "write",
            )
            return turn.runtime_payload()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/context/auto")
    def auto_context(payload: AutoContextRequest) -> dict:
        turn = TurnOrchestrator(db).run_turn(
            payload.intent,
            context={**payload.model_dump(), "operation": "auto_context"},
            mode="preview",
        )
        return turn.runtime_payload()

    @app.post("/memory/consolidate")
    def consolidate(payload: ConsolidateRequest) -> dict:
        return Consolidator(db).consolidate(project=payload.project)

    @app.post("/memory/forget")
    def forget(payload: ForgetRequest) -> dict:
        if payload.project:
            deleted = db.delete_project_memories(payload.project, hard=payload.hard)
            return {"deleted": deleted, "hard": payload.hard}
        if not payload.memory_id:
            raise HTTPException(status_code=400, detail="memory_id or project is required")
        ok = (
            db.delete_memory(payload.memory_id)
            if payload.hard
            else db.update_memory_status(payload.memory_id, "deleted")
        )
        return {"deleted": 1 if ok else 0, "hard": payload.hard}

    @app.post("/project/index")
    def index_project(payload: ProjectIndexRequest) -> dict:
        try:
            return ProjectIndexer(db).index_project(payload.path, payload.project)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/project/{project}/summary")
    def project_summary(project: str) -> dict:
        return {"project": project, "summary": Consolidator(db)._project_summary(project)}

    @app.get("/project/{project}/profile", response_model=TextPackResponse)
    def project_profile(project: str) -> TextPackResponse:
        return TextPackResponse(text=ProjectProfileBuilder(db).build(project))

    @app.post("/project/code-map", response_model=TextPackResponse)
    def code_map(payload: CodeMapRequest) -> TextPackResponse:
        text = CodeMapBuilder(db).build(
            project=payload.project,
            query=payload.query,
            max_files=payload.max_files,
        )
        return TextPackResponse(text=text)

    @app.post("/project/code-symbols")
    def code_symbols(payload: CodeSymbolRequest) -> dict:
        return {
            "symbols": CodeIntelligence(db).search_symbols(
                payload.project,
                query=payload.query,
                limit=payload.limit,
            )
        }

    @app.post("/project/code-references")
    def code_references(payload: CodeReferenceRequest) -> dict:
        return {
            "references": CodeIntelligence(db).references(
                payload.project,
                symbol=payload.symbol,
                limit=payload.limit,
            )
        }

    @app.post("/project/code-intelligence")
    def code_intelligence(payload: CodeIntelligenceRequest) -> dict:
        files = CodeMapBuilder(db).relevant_files(
            payload.project,
            payload.intent,
            limit=payload.limit,
        )
        lines = CodeIntelligence(db).impact_lines(
            payload.project,
            payload.intent,
            files,
            limit=payload.limit,
        )
        return {"lines": lines, "text": "\n".join(lines)}

    @app.post("/project/code-diagnostics")
    def code_diagnostics(payload: CodeDiagnosticsRequest) -> dict:
        intelligence = CodeIntelligence(db)
        if not payload.refresh:
            return {"diagnostics": intelligence.diagnostics(payload.project, limit=payload.limit)}
        root_path = payload.path or _project_root_path(db, payload.project) or "."
        result = run_python_diagnostics(root_path, checker=payload.checker)
        if result["available"]:
            db.replace_code_diagnostics(
                project=payload.project,
                diagnostics=result["diagnostics"],
                source=Path(str(result["tool"])).name,
            )
        return {
            **result,
            "diagnostics": [asdict(diagnostic) for diagnostic in result["diagnostics"]],
        }

    @app.post("/project/impact", response_model=TextPackResponse)
    def impact(payload: ImpactRequest) -> TextPackResponse:
        text = ChangePlanner(db).plan(
            intent=payload.intent,
            project=payload.project,
            max_tokens=payload.max_tokens,
        )
        return TextPackResponse(text=text)

    @app.get("/project/list")
    def project_list() -> dict:
        return {"projects": db.list_projects()}

    @app.get("/candidate/list")
    def candidate_list(project: str | None = None, status: str = "pending") -> dict:
        return {"candidates": db.list_candidates(project=project, status=status)}

    @app.post("/candidate/accept")
    def candidate_accept(payload: CandidateAcceptRequest) -> dict:
        try:
            return accept_candidate(db, payload.candidate_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/candidate/discard")
    def candidate_discard(payload: CandidateAcceptRequest) -> dict:
        return {"discarded": discard_candidate(db, payload.candidate_id)}

    @app.get("/conflict/list")
    def conflict_list(project: str | None = None, limit: int = 100) -> dict:
        return {"conflicts": db.list_conflicts(project=project, limit=limit)}

    @app.post("/conflict/resolve")
    def conflict_resolve(payload: ConflictResolveRequest) -> dict:
        ok = db.update_conflict(
            payload.conflict_id,
            resolution=payload.resolution,
            status=payload.status,
        )
        return {"updated": ok}

    @app.post("/session/summarize")
    def summarize_session(payload: SessionSummarizeRequest) -> dict:
        if payload.path:
            summary = summarize_session_file(payload.path, payload.project, use_llm=payload.use_llm)
        elif payload.text:
            summary = build_session_summary(payload.text, payload.project, use_llm=payload.use_llm)
        else:
            raise HTTPException(status_code=400, detail="path or text is required")
        if payload.write:
            if not payload.confirm_write:
                raise HTTPException(
                    status_code=400,
                    detail="confirm_write=true is required when write=true",
                )
            result = write_summary_candidates(db, summary, payload.project)
            return {"summary": summary, "write_result": result}
        return summary

    @app.post("/session/queue")
    def queue_session(payload: SessionSummarizeRequest) -> dict:
        if payload.path:
            summary = summarize_session_file(payload.path, payload.project, use_llm=payload.use_llm)
        elif payload.text:
            summary = build_session_summary(payload.text, payload.project, use_llm=payload.use_llm)
        else:
            raise HTTPException(status_code=400, detail="path or text is required")
        queue_result = queue_summary_candidates(db, summary, payload.project)
        return {"summary": summary, "queue_result": queue_result}

    @app.get("/stats")
    def stats() -> dict:
        return db.stats()

    return app


app = create_app()


def _project_root_path(db: Database, project: str) -> str | None:
    record = db.get_project(project)
    if not record or not record.get("root_path"):
        return None
    return str(record["root_path"])
