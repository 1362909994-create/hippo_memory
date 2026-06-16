from __future__ import annotations

from hippocampus_memory.context_bundle import ContextBundleBuilder
from hippocampus_memory.db import Database
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.utils import estimate_tokens, token_counter_name


def token_savings_report(
    db: Database,
    project: str,
    intent: str,
    *,
    record: bool = True,
    model: str | None = None,
    strategy: str = "auto",
) -> dict[str, object]:
    builder = ContextBundleBuilder(db)
    bundle = builder.build(project=project, intent=intent, strategy=strategy)
    full_bundle = builder.build(project=project, intent=intent, strategy="full")
    lean_bundle = builder.build(project=project, intent=intent, strategy="lean")
    bundle_tokens = estimate_tokens(bundle, model=model)
    full_bundle_tokens = estimate_tokens(full_bundle, model=model)
    lean_bundle_tokens = estimate_tokens(lean_bundle, model=model)
    packer = MemoryPacker(db)
    memory_pack = packer.pack(intent, project=project, max_tokens=1500)
    compact_pack = packer.pack(intent, project=project, max_tokens=1500, compact=True)
    memory_pack_tokens = estimate_tokens(memory_pack, model=model)
    compact_pack_tokens = estimate_tokens(compact_pack, model=model)
    memories = db.list_memories(
        project=project,
        include_archived=False,
        include_private=False,
        include_sensitive=False,
        limit=500,
    )
    memory_tokens = sum(estimate_tokens(memory.content, model=model) for memory in memories)
    files = db.list_files(project, limit=1000)
    file_summary_tokens = sum(
        estimate_tokens(str(row.get("summary") or ""), model=model) for row in files
    )
    naive_tokens = memory_tokens + file_summary_tokens
    bundle_saved, bundle_ratio = _savings(naive_tokens, bundle_tokens)
    pack_saved, pack_ratio = _savings(naive_tokens, memory_pack_tokens)
    compact_saved, compact_ratio = _savings(naive_tokens, compact_pack_tokens)
    counter = token_counter_name(model)
    report: dict[str, object] = {
        "project": project,
        "intent": intent,
        "token_counter": counter,
        "token_counter_exact": counter.startswith("tiktoken:"),
        "token_counter_note": _counter_note(counter),
        "context_bundle_strategy": _strategy_from_bundle(bundle),
        "context_bundle_tokens": bundle_tokens,
        "full_context_bundle_tokens": full_bundle_tokens,
        "lean_context_bundle_tokens": lean_bundle_tokens,
        "memory_pack_tokens": memory_pack_tokens,
        "compact_pack_tokens": compact_pack_tokens,
        "naive_index_summary_tokens": naive_tokens,
        "estimated_tokens_saved": bundle_saved,
        "estimated_savings_ratio": bundle_ratio,
        "memory_pack_tokens_saved": pack_saved,
        "memory_pack_savings_ratio": pack_ratio,
        "compact_pack_tokens_saved": compact_saved,
        "compact_pack_savings_ratio": compact_ratio,
        "memory_count": len(memories),
        "indexed_file_count": len(files),
    }
    if record:
        metadata = {
            "memory_count": len(memories),
            "indexed_file_count": len(files),
            "memory_tokens": memory_tokens,
            "file_summary_tokens": file_summary_tokens,
            "token_counter": counter,
            "token_counter_exact": counter.startswith("tiktoken:"),
            "context_bundle_strategy": _strategy_from_bundle(bundle),
            "full_context_bundle_tokens": full_bundle_tokens,
            "lean_context_bundle_tokens": lean_bundle_tokens,
        }
        report["ledger_ids"] = [
            db.insert_token_ledger(
                project=project,
                intent=intent,
                context_type="context_bundle",
                baseline_tokens=naive_tokens,
                output_tokens=bundle_tokens,
                metadata=metadata,
            ),
            db.insert_token_ledger(
                project=project,
                intent=intent,
                context_type="memory_pack",
                baseline_tokens=naive_tokens,
                output_tokens=memory_pack_tokens,
                metadata=metadata,
            ),
            db.insert_token_ledger(
                project=project,
                intent=intent,
                context_type="compact_pack",
                baseline_tokens=naive_tokens,
                output_tokens=compact_pack_tokens,
                metadata=metadata,
            ),
        ]
    return report


def token_ledger_report(db: Database, project: str, limit: int = 20) -> dict[str, object]:
    return {
        "project": project,
        "summary": db.token_ledger_summary(project),
        "recent_entries": db.list_token_ledger(project, limit=max(1, min(100, limit))),
    }


def _savings(baseline_tokens: int, output_tokens: int) -> tuple[int, float]:
    saved = max(0, baseline_tokens - output_tokens)
    ratio = saved / baseline_tokens if baseline_tokens else 0.0
    return saved, ratio


def _strategy_from_bundle(bundle: str) -> str:
    for line in bundle.splitlines():
        if line.startswith("Strategy:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _counter_note(counter: str) -> str:
    if counter.startswith("tiktoken:"):
        return "Exact for the selected tiktoken model mapping."
    if counter == "heuristic":
        return "Estimated with the local heuristic tokenizer."
    return "Estimated with the local heuristic tokenizer; install the tokens extra for tiktoken."
