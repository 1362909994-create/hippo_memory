from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from hippocampus_memory.callback import reset_callback_pack
from hippocampus_memory.change_planner import ChangePlanner
from hippocampus_memory.code_graph import CodeGraphBuilder
from hippocampus_memory.code_intelligence import CodeIntelligence
from hippocampus_memory.code_map import CodeMapBuilder
from hippocampus_memory.codex_workspace import CodexWorkspaceResolver
from hippocampus_memory.consolidator import Consolidator
from hippocampus_memory.db import Database
from hippocampus_memory.deploy import (
    codex_doctor,
    deploy_codex,
    project_mcp_database,
    write_daemon_script,
    write_mcp_client_config,
)
from hippocampus_memory.evaluator import evaluate_retrieval
from hippocampus_memory.lsp_diagnostics import run_python_diagnostics
from hippocampus_memory.mcp_server import serve_stdio
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator import TurnOrchestrator
from hippocampus_memory.project_indexer import ProjectIndexer
from hippocampus_memory.project_profile import ProjectProfileBuilder
from hippocampus_memory.project_resolver import resolve_project_name, write_project_config
from hippocampus_memory.report import write_memory_browser
from hippocampus_memory.runner import run_with_context
from hippocampus_memory.session_ingestor import (
    accept_candidate,
    discard_candidate,
    queue_summary_candidates,
    write_summary_candidates,
)
from hippocampus_memory.session_recorder import record_run_session
from hippocampus_memory.summarizer import summarize_session_file
from hippocampus_memory.token_report import (
    format_savings_line,
    record_context_savings,
    token_ledger_report,
    token_savings_report,
)

app = typer.Typer(help="Local-first external memory for AI agents.")


def get_db() -> Database:
    db = Database()
    db.initialize()
    return db


@app.command()
def init() -> None:
    """Initialize the local SQLite database."""
    db = get_db()
    typer.echo(f"Initialized: {db.path}")


@app.command("project-init")
def project_init(
    project: str,
    root: Path = typer.Option(Path("."), "--root"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Write a .hippo.toml project config for automatic project detection."""
    path = write_project_config(root, project, force=force)
    db = get_db()
    db.insert_or_update_project(project, root_path=str(root.resolve()))
    typer.echo(str(path))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the local FastAPI service."""
    import uvicorn

    from hippocampus_memory.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def daemon(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the local long-lived HTTP daemon."""
    serve(host=host, port=port)


@app.command()
def write(
    content: str = typer.Option(..., "--content", "-c"),
    memory_type: str = typer.Option(..., "--type"),
    project: str | None = typer.Option(None, "--project"),
    confidence: float = typer.Option(0.8, "--confidence"),
    importance: float = typer.Option(0.5, "--importance"),
    visibility: str | None = typer.Option(None, "--visibility"),
    entity: list[str] = typer.Option([], "--entity"),
    tag: list[str] = typer.Option([], "--tag"),
) -> None:
    """Write one memory."""
    result = MemoryWriter(get_db()).write(
        content=content,
        memory_type=memory_type,
        project=project,
        confidence=confidence,
        importance=importance,
        visibility=visibility,
        entities=entity,
        tags=tag,
    )
    typer.echo(result.memory_id)


@app.command()
def search(
    query: str,
    project: str | None = typer.Option(None, "--project"),
    top_k: int = typer.Option(10, "--top-k"),
    mode: str = typer.Option("hybrid", "--mode"),
    entity: list[str] = typer.Option([], "--entity"),
    tag: list[str] = typer.Option([], "--tag"),
    dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe"),
) -> None:
    """Search memories."""
    turn = TurnOrchestrator(get_db()).run_turn(
        query,
        context={
            "operation": "memory_search",
            "project": project,
            "top_k": top_k,
            "search_mode": mode,
            "entities": entity or None,
            "tags": tag or None,
            "dedupe_results": dedupe,
            "writeback": False,
        },
        mode="preview",
    )
    for result in turn.turn_context.selected_memories:
        typer.echo(
            f"[{result.score:.3f}] {result.memory_type} {result.memory_id}: {result.content}"
        )


@app.command()
def explain(
    memory_id: str,
    project: str | None = typer.Option(None, "--project"),
    query: str | None = typer.Option(None, "--query"),
) -> None:
    """Explain why one memory would be recalled and how it is scored."""
    scoring_project = resolve_project_name(project) if project is not None else None
    try:
        result = TurnOrchestrator(get_db()).run_turn(
            query or memory_id,
            context={
                "operation": "memory_explain",
                "memory_id": memory_id,
                "project": scoring_project,
                "query": query,
                "writeback": False,
            },
            mode="preview",
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = dict(result.recall_payload)
    payload.pop("decision", None)
    typer.echo(payload)


@app.command()
def pack(
    query: str,
    project: str | None = typer.Option(None, "--project"),
    max_tokens: int = typer.Option(1500, "--max-tokens"),
    source_chunk_limit: int = typer.Option(2, "--source-chunk-limit"),
    compact: bool = typer.Option(False, "--compact"),
    exclude_memory_id: list[str] = typer.Option([], "--exclude-memory-id"),
    session_dedupe: bool = typer.Option(False, "--session-dedupe"),
    token_stats: bool = typer.Option(True, "--token-stats/--no-token-stats"),
    token_model: str | None = typer.Option(None, "--token-model"),
) -> None:
    """Generate a Memory Pack for an agent."""
    project = resolve_project_name(project)
    db = get_db()
    turn = TurnOrchestrator(db).run_turn(
        query,
        context={
            "operation": "memory_pack",
            "project": project,
            "max_tokens": max_tokens,
            "source_chunk_limit": source_chunk_limit,
            "compact": compact,
            "exclude_memory_ids": exclude_memory_id,
            "session_dedupe": session_dedupe,
            "writeback": False,
        },
        mode="preview",
    )
    text = turn.injected_context
    _echo_token_savings(
        db,
        project=project,
        intent=query,
        context_type="compact_pack" if compact else "memory_pack",
        output_text=text,
        enabled=token_stats,
        model=token_model,
    )
    typer.echo(text)


@app.command("auto-store")
def auto_store(
    text: str | None = typer.Option(None, "--text", "-t"),
    path: Path | None = typer.Option(None, "--path", "-p"),
    project: str | None = typer.Option(None, "--project"),
    source: str = typer.Option("auto_store", "--source"),
    mode: str = typer.Option("auto", "--mode", help="One of: auto, write, queue, preview."),
    max_candidates: int = typer.Option(12, "--max-candidates"),
    allow_sensitive: bool = typer.Option(False, "--allow-sensitive"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Automatically admit useful long-term memories from text."""
    if path is not None:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    elif text is not None:
        raw_text = text
    else:
        raise typer.BadParameter("--text or --path is required")
    resolved = resolve_project_name(project) if project is not None else None
    try:
        turn = TurnOrchestrator(get_db()).run_turn(
            raw_text,
            context={
                "operation": "memory_auto_store",
                "project": resolved,
                "source": source,
                "store_mode": mode,
                "max_candidates": max_candidates,
                "allow_sensitive": allow_sensitive,
                "dry_run": dry_run,
                "writeback": False,
            },
            mode="preview" if dry_run or mode == "preview" else "write",
        )
        typer.echo(turn.memory_writeback)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("auto-context")
def auto_context(
    intent: str,
    project: str | None = typer.Option(None, "--project"),
    session: str = typer.Option("default", "--session"),
    max_tokens: int = typer.Option(3500, "--max-tokens"),
    include_code_map: bool = typer.Option(True, "--code-map/--no-code-map"),
    metadata: bool = typer.Option(False, "--metadata"),
    token_stats: bool = typer.Option(True, "--token-stats/--no-token-stats"),
    token_model: str | None = typer.Option(None, "--token-model"),
) -> None:
    """Automatically decide whether and how to recall external memory."""
    resolved = resolve_project_name(project) if project is not None else None
    db = get_db()
    turn = TurnOrchestrator(db).run_turn(
        intent,
        context={
            "operation": "auto_context",
            "project": resolved,
            "session_key": session,
            "max_tokens": max_tokens,
            "include_code_map": include_code_map,
            "track_token_savings": token_stats and resolved is not None,
            "token_model": token_model,
        },
        mode="preview",
    )
    result = turn.runtime_payload()
    if result.get("token_savings_text"):
        typer.echo(result["token_savings_text"], err=True)
    typer.echo(result if metadata else result["text"])


@app.command()
def callback(
    intent: str,
    project: str | None = typer.Option(None, "--project"),
    session: str = typer.Option("default", "--session"),
    max_tokens: int = typer.Option(1500, "--max-tokens"),
    source_chunk_limit: int = typer.Option(2, "--source-chunk-limit"),
    compact: bool = typer.Option(True, "--compact/--full"),
    metadata: bool = typer.Option(False, "--metadata"),
) -> None:
    """Generate a project-scoped callback pack and remember injected memories."""
    project = resolve_project_name(project)
    turn = TurnOrchestrator(get_db()).run_turn(
        intent,
        context={
            "operation": "context_callback",
            "project": project,
            "session_key": session,
            "max_tokens": max_tokens,
            "source_chunk_limit": source_chunk_limit,
            "compact": compact,
            "writeback": False,
        },
        mode="preview",
    )
    result = turn.runtime_payload()
    typer.echo(result if metadata else result["pack"])


@app.command("callback-reset")
def callback_reset(
    project: str | None = typer.Option(None, "--project"),
    session: str = typer.Option("default", "--session"),
) -> None:
    """Reset remembered callback memory ids for one project session."""
    project = resolve_project_name(project)
    typer.echo(reset_callback_pack(get_db(), project=project, session_key=session))


@app.command("project-profile")
def project_profile(project: str | None = typer.Option(None, "--project")) -> None:
    """Generate a compact whole-project profile for AI coding context."""
    project = resolve_project_name(project)
    typer.echo(ProjectProfileBuilder(get_db()).build(project))


@app.command("code-map")
def code_map(
    project: str | None = typer.Option(None, "--project"),
    query: str | None = typer.Option(None, "--query"),
    max_files: int = typer.Option(12, "--max-files"),
) -> None:
    """Generate a compact map of indexed files, summaries, symbols and imports."""
    project = resolve_project_name(project)
    typer.echo(CodeMapBuilder(get_db()).build(project=project, query=query, max_files=max_files))


@app.command("code-graph")
def code_graph(
    project: str | None = typer.Option(None, "--project"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Generate a lightweight inferred cross-file call graph."""
    project = resolve_project_name(project)
    typer.echo(CodeGraphBuilder(get_db()).build(project=project, limit=limit))


@app.command("code-symbols")
def code_symbols(
    project: str | None = typer.Option(None, "--project"),
    query: str | None = typer.Option(None, "--query"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List indexed code symbols with definition locations."""
    project = resolve_project_name(project)
    typer.echo(CodeIntelligence(get_db()).search_symbols(project, query=query, limit=limit))


@app.command("code-references")
def code_references(
    symbol: str,
    project: str | None = typer.Option(None, "--project"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """List indexed references/calls for a symbol."""
    project = resolve_project_name(project)
    typer.echo(CodeIntelligence(get_db()).references(project, symbol=symbol, limit=limit))


@app.command("code-intelligence")
def code_intelligence(
    intent: str,
    project: str | None = typer.Option(None, "--project"),
    limit: int = typer.Option(8, "--limit"),
) -> None:
    """Generate a symbol-level impact summary for an intent."""
    project = resolve_project_name(project)
    db = get_db()
    files = CodeMapBuilder(db).relevant_files(project, intent, limit=limit)
    typer.echo(CodeIntelligence(db).impact_lines(project, intent, files, limit=limit))


@app.command("code-diagnostics")
def code_diagnostics(
    project: str | None = typer.Option(None, "--project"),
    path: Path | None = typer.Option(None, "--path"),
    checker: str | None = typer.Option(None, "--checker"),
    refresh: bool = typer.Option(False, "--refresh/--cached"),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    """List or refresh stored Python LSP diagnostics."""
    project = resolve_project_name(project, cwd=path)
    db = get_db()
    if refresh:
        root_path = path or _project_root_path(db, project) or Path(".")
        result = run_python_diagnostics(root_path, checker=checker)
        if result["available"]:
            db.replace_code_diagnostics(
                project=project,
                diagnostics=result["diagnostics"],
                source=Path(str(result["tool"])).name,
            )
        typer.echo(
            {
                **result,
                "diagnostics": [asdict(diagnostic) for diagnostic in result["diagnostics"]],
            }
        )
        return
    typer.echo(CodeIntelligence(db).diagnostics(project, limit=limit))


@app.command()
def impact(
    intent: str,
    project: str | None = typer.Option(None, "--project"),
    max_tokens: int = typer.Option(1200, "--max-tokens"),
) -> None:
    """Generate a minimal-change Code Impact Pack for a planned edit."""
    project = resolve_project_name(project)
    typer.echo(ChangePlanner(get_db()).plan(intent=intent, project=project, max_tokens=max_tokens))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
    intent: str = typer.Option("Continue the current coding task.", "--intent", "-i"),
    project: str | None = typer.Option(None, "--project"),
    inject: str = typer.Option(
        "print",
        "--inject",
        help="One of: print, file, env, stdin, arg.",
    ),
    cwd: Path | None = typer.Option(None, "--cwd"),
    context_file: Path | None = typer.Option(None, "--context-file"),
    max_tokens: int = typer.Option(3500, "--max-tokens"),
    include_code_map: bool = typer.Option(True, "--code-map/--no-code-map"),
    bundle_strategy: str = typer.Option(
        "auto",
        "--bundle-strategy",
        help="One of: auto, full, lean, pack.",
    ),
    record: bool = typer.Option(True, "--record/--no-record"),
    write_session_memory: bool = typer.Option(False, "--write-session-memory"),
    yes: bool = typer.Option(False, "--yes"),
    token_stats: bool = typer.Option(True, "--token-stats/--no-token-stats"),
    token_model: str | None = typer.Option(None, "--token-model"),
) -> None:
    """Generate context and optionally launch another AI coding command with it."""
    command = list(ctx.args)
    _validate_run_request(inject, command, bundle_strategy)
    db = get_db()
    project_name = resolve_project_name(project, cwd=cwd)
    turn = TurnOrchestrator(db).run_turn(
        intent,
        context={
            "operation": "context_bundle",
            "project": project_name,
            "max_tokens": max_tokens,
            "include_code_map": include_code_map,
            "bundle_strategy": bundle_strategy,
            "writeback": False,
        },
        mode="preview",
    )
    context = turn.injected_context
    _echo_token_savings(
        db,
        project=project_name,
        intent=intent,
        context_type=f"context_bundle:{bundle_strategy}",
        output_text=context,
        enabled=token_stats,
        model=token_model,
    )
    try:
        result = run_with_context(
            command=command,
            context=context,
            project=project_name,
            intent=intent,
            inject=inject,
            cwd=cwd,
            context_file=context_file,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if record:
        if write_session_memory and not yes:
            raise typer.BadParameter("--write-session-memory requires --yes")
        record_run_session(
            db,
            project=project_name,
            intent=intent,
            command=command,
            returncode=result.returncode,
            context_file=result.context_file,
            stdout=result.stdout,
            stderr=result.stderr,
            cwd=cwd,
            write_memory=write_session_memory,
        )
    typer.echo(result.stdout, nl=False)
    if result.stderr:
        typer.echo(result.stderr, err=True, nl=False)
    raise typer.Exit(result.returncode)


@app.command()
def mcp() -> None:
    """Run the lightweight JSON-RPC stdio memory server."""
    serve_stdio(get_db())


@app.command("mcp-project")
def mcp_project(root: Path = typer.Option(Path("."), "--root")) -> None:
    """Run MCP against the nearest project-local .hippo/hippo.db."""
    serve_stdio(
        project_mcp_database(root),
        safe_tool_names=True,
        default_project=resolve_project_name(cwd=root),
    )


@app.command("mcp-codex")
def mcp_codex(
    fallback_root: Path | None = typer.Option(
        None,
        "--fallback-root",
        help="Fallback project root when Codex does not expose a workspace cwd.",
    ),
    auto_create: bool = typer.Option(
        True,
        "--auto-create/--no-auto-create",
        help="Create a project-local .hippo database for the resolved Codex workspace.",
    ),
) -> None:
    """Run MCP for Codex App, resolving the active workspace per tool call."""
    serve_stdio(
        project_mcp_database(fallback_root) if fallback_root is not None else get_db(),
        safe_tool_names=True,
        project_resolver=CodexWorkspaceResolver(
            fallback_root=fallback_root,
            auto_create=auto_create,
        ),
    )


@app.command("codex-deploy")
def codex_deploy(
    root: Path = typer.Option(Path("."), "--root"),
    project: str | None = typer.Option(None, "--project"),
    force_project_config: bool = typer.Option(False, "--force-project-config"),
    index: bool = typer.Option(True, "--index/--no-index"),
    project_memory: bool = typer.Option(
        True,
        "--project-memory/--no-project-memory",
        help="Write a short AGENTS.md hint so Codex decides when to use hippo.",
    ),
) -> None:
    """One-command project-local memory deployment for Codex."""
    typer.echo(
        deploy_codex(
            root=root,
            project=project,
            force_project_config=force_project_config,
            index_project=index,
            project_memory=project_memory,
        )
    )


@app.command("doctor")
def doctor(
    root: Path = typer.Option(Path("."), "--root"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Diagnose Hippo's project-local Codex deployment without modifying files."""
    report = codex_doctor(root=root)
    if json_output:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        typer.echo(report)


@app.command("eval")
def eval_command(benchmark: Path) -> None:
    """Evaluate retrieval with a JSONL benchmark."""
    typer.echo(evaluate_retrieval(get_db(), benchmark))


@app.command("token-report")
def token_report(
    intent: str,
    project: str | None = typer.Option(None, "--project"),
    record: bool = typer.Option(True, "--record/--no-record"),
    model: str | None = typer.Option(None, "--model"),
    bundle_strategy: str = typer.Option("auto", "--bundle-strategy"),
) -> None:
    """Estimate token savings for the generated Context Bundle."""
    project = resolve_project_name(project)
    typer.echo(
        token_savings_report(
            get_db(),
            project,
            intent,
            record=record,
            model=model,
            strategy=bundle_strategy,
        )
    )


@app.command("token-ledger")
def token_ledger(
    project: str | None = typer.Option(None, "--project"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Show project-scoped historical token savings measurements."""
    project = resolve_project_name(project)
    typer.echo(token_ledger_report(get_db(), project, limit=limit))


@app.command("mcp-config")
def mcp_config(
    output: Path = typer.Option(Path("hippo-mcp-config.json"), "--output"),
    command: str | None = typer.Option(None, "--command"),
) -> None:
    """Write a starter MCP client config snippet."""
    typer.echo(str(write_mcp_client_config(output, command)))


@app.command("daemon-script")
def daemon_script(
    output: Path = typer.Option(Path("start-hippo-daemon.ps1"), "--output"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
) -> None:
    """Write a PowerShell script that starts the local daemon."""
    typer.echo(str(write_daemon_script(output, host=host, port=port)))


@app.command("browser")
def browser_report(
    output: Path = typer.Option(Path("hippo-memory-browser.html"), "--output"),
    project: str | None = typer.Option(None, "--project"),
) -> None:
    """Write a local HTML memory browser report."""
    path = write_memory_browser(get_db(), output, project)
    typer.echo(str(path))


@app.command("index-project")
def index_project(path: Path, project: str | None = typer.Option(None, "--project")) -> None:
    """Index a local project folder."""
    root = path.expanduser()
    if not root.exists() or not root.is_dir():
        raise typer.BadParameter(f"project path does not exist or is not a directory: {root}")
    project = resolve_project_name(project, cwd=root)
    try:
        result = ProjectIndexer(get_db()).index_project(root, project)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(result)


@app.command("summarize-session")
def summarize_session(
    path: Path,
    project: str | None = typer.Option(None, "--project"),
    write: bool = typer.Option(False, "--write"),
    yes: bool = typer.Option(False, "--yes"),
    use_llm: bool = typer.Option(False, "--llm"),
) -> None:
    """Extract memory candidates from a chat transcript."""
    db = get_db()
    project = resolve_project_name(project)
    summary = summarize_session_file(path, project, use_llm=use_llm)
    if write:
        if not yes:
            raise typer.BadParameter("--write requires --yes to avoid accidental memory pollution")
        typer.echo(
            {
                "summary": summary,
                "write_result": write_summary_candidates(db, summary, project),
            }
        )
        return
    typer.echo(summary)


@app.command("candidate-list")
def candidate_list(
    project: str | None = typer.Option(None, "--project"),
    status: str = typer.Option("pending", "--status"),
) -> None:
    """List queued memory candidates."""
    db = get_db()
    resolved = resolve_project_name(project) if project is not None else None
    typer.echo(db.list_candidates(project=resolved, status=status))


@app.command("candidate-accept")
def candidate_accept(candidate_id: str) -> None:
    """Accept one queued memory candidate into long-term memory."""
    typer.echo(accept_candidate(get_db(), candidate_id))


@app.command("candidate-discard")
def candidate_discard(candidate_id: str) -> None:
    """Discard one queued memory candidate."""
    typer.echo({"discarded": discard_candidate(get_db(), candidate_id)})


@app.command("conflict-list")
def conflict_list(
    project: str | None = typer.Option(None, "--project"),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    """List open conflict candidates."""
    resolved = resolve_project_name(project) if project is not None else None
    typer.echo(get_db().list_conflicts(project=resolved, limit=limit))


@app.command("conflict-resolve")
def conflict_resolve(
    conflict_id: str,
    resolution: str | None = typer.Option(None, "--resolution"),
    status: str = typer.Option("resolved", "--status"),
) -> None:
    """Resolve or update a conflict candidate."""
    ok = get_db().update_conflict(conflict_id, resolution=resolution, status=status)
    typer.echo({"updated": ok})


@app.command("queue-session")
def queue_session(
    path: Path,
    project: str | None = typer.Option(None, "--project"),
    use_llm: bool = typer.Option(False, "--llm"),
) -> None:
    """Queue session summary candidates for later confirmation."""
    db = get_db()
    project = resolve_project_name(project)
    summary = summarize_session_file(path, project, use_llm=use_llm)
    typer.echo({"summary": summary, "queue_result": queue_summary_candidates(db, summary, project)})


@app.command()
def consolidate(project: str | None = typer.Option(None, "--project")) -> None:
    """Consolidate duplicate, stale, and conflicting memories."""
    project = resolve_project_name(project) if project is not None else None
    typer.echo(Consolidator(get_db()).consolidate(project=project))


@app.command()
def forget(
    memory_id: str | None = typer.Argument(None),
    project: str | None = typer.Option(None, "--project"),
    hard: bool = typer.Option(False, "--hard"),
) -> None:
    """Forget one memory or all memories in a project."""
    db = get_db()
    if project:
        typer.echo({"deleted": db.delete_project_memories(project, hard=hard), "hard": hard})
        return
    if not memory_id:
        raise typer.BadParameter("memory_id or --project is required")
    ok = db.delete_memory(memory_id) if hard else db.update_memory_status(memory_id, "deleted")
    typer.echo({"deleted": 1 if ok else 0, "hard": hard})


@app.command("memory-supersede")
def memory_supersede(old_memory_id: str, new_memory_id: str) -> None:
    """Mark one memory as superseded by another memory."""
    typer.echo({"updated": get_db().supersede_memory(old_memory_id, new_memory_id)})


@app.command("project-summary")
def project_summary(project: str = typer.Option(..., "--project")) -> None:
    """Print a compact project summary."""
    project = resolve_project_name(project)
    typer.echo(Consolidator(get_db())._project_summary(project))


@app.command()
def stats() -> None:
    """Print database stats."""
    typer.echo(get_db().stats())


def _validate_run_request(inject: str, command: list[str], bundle_strategy: str) -> None:
    normalized = inject.strip().casefold()
    if normalized not in {"print", "file", "env", "stdin", "arg"}:
        raise typer.BadParameter(f"unsupported inject mode: {inject}")
    strategy = bundle_strategy.strip().casefold()
    if strategy not in {"auto", "full", "lean", "pack"}:
        raise typer.BadParameter("strategy must be one of: auto, full, lean, pack")
    if normalized != "print" and not command:
        raise typer.BadParameter("a command is required unless --inject print is used")


def _project_root_path(db: Database, project: str) -> Path | None:
    record = db.get_project(project)
    if not record or not record.get("root_path"):
        return None
    return Path(str(record["root_path"]))


def _echo_token_savings(
    db: Database,
    *,
    project: str | None,
    intent: str,
    context_type: str,
    output_text: str,
    enabled: bool,
    model: str | None,
) -> dict[str, object] | None:
    if not enabled or not project:
        return None
    report = record_context_savings(
        db,
        project=project,
        intent=intent,
        context_type=context_type,
        output_text=output_text,
        model=model,
        record=True,
    )
    typer.echo(format_savings_line(report), err=True)
    return report
