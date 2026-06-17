from __future__ import annotations

from hippocampus_memory.context_bundle import ContextBundleBuilder
from hippocampus_memory.db import Database
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.utils import estimate_tokens, token_counter_name


def context_baseline_report(
    db: Database,
    project: str,
    *,
    model: str | None = None,
) -> dict[str, object]:
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
    return {
        "project": project,
        "baseline_tokens": memory_tokens + file_summary_tokens,
        "memory_tokens": memory_tokens,
        "file_summary_tokens": file_summary_tokens,
        "memory_count": len(memories),
        "indexed_file_count": len(files),
    }


def record_context_savings(
    db: Database,
    *,
    project: str,
    intent: str,
    context_type: str,
    output_text: str,
    model: str | None = None,
    record: bool = True,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    baseline = context_baseline_report(db, project, model=model)
    output_tokens = estimate_tokens(output_text, model=model)
    baseline_tokens = int(baseline["baseline_tokens"])
    saved_tokens, savings_ratio = _savings(baseline_tokens, output_tokens)
    net_tokens_saved = baseline_tokens - output_tokens
    net_savings_ratio = net_tokens_saved / baseline_tokens if baseline_tokens else 0.0
    counter = token_counter_name(model)
    ledger_id = None
    if record:
        ledger_id = db.insert_token_ledger(
            project=project,
            intent=intent,
            context_type=context_type,
            baseline_tokens=baseline_tokens,
            output_tokens=output_tokens,
            metadata={
                **baseline,
                "token_counter": counter,
                "token_counter_exact": counter.startswith("tiktoken:"),
                **(metadata or {}),
            },
        )
    summary = db.token_ledger_summary(project)
    return {
        "project": project,
        "intent": intent,
        "context_type": context_type,
        "token_counter": counter,
        "token_counter_exact": counter.startswith("tiktoken:"),
        "token_counter_note": _counter_note(counter),
        "baseline_tokens": baseline_tokens,
        "output_tokens": output_tokens,
        "saved_tokens": saved_tokens,
        "savings_ratio": savings_ratio,
        "net_tokens_saved": net_tokens_saved,
        "net_savings_ratio": net_savings_ratio,
        "ledger_id": ledger_id,
        "total_saved_tokens": int(summary["saved_tokens"]),
        "total_baseline_tokens": int(summary["baseline_tokens"]),
        "total_output_tokens": int(summary["output_tokens"]),
        "ledger_entry_count": int(summary["entry_count"]),
        "average_savings_ratio": float(summary["average_savings_ratio"]),
        "memory_count": int(baseline["memory_count"]),
        "indexed_file_count": int(baseline["indexed_file_count"]),
    }


def format_savings_line(report: dict[str, object]) -> str:
    saved = int(report["saved_tokens"])
    baseline = int(report["baseline_tokens"])
    output = int(report["output_tokens"])
    total = int(report["total_saved_tokens"])
    entry_count = int(report["ledger_entry_count"])
    net_saved = int(report.get("net_tokens_saved", saved))
    ratio = float(report.get("net_savings_ratio", report["savings_ratio"]))
    average = float(report["average_savings_ratio"])
    counter = str(report["token_counter"])
    if net_saved >= 0:
        run_text = f"this run saved {net_saved} tokens"
    else:
        run_text = f"this run used {-net_saved} more tokens"
    return (
        "Token savings: "
        f"{run_text} "
        f"({baseline} -> {output}, {ratio:.1%}); "
        f"project total saved {total} tokens across {entry_count} records "
        f"(avg {average:.1%}); counter={counter}"
    )


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
    baseline = context_baseline_report(db, project, model=model)
    memory_tokens = int(baseline["memory_tokens"])
    file_summary_tokens = int(baseline["file_summary_tokens"])
    naive_tokens = int(baseline["baseline_tokens"])
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
        "memory_count": int(baseline["memory_count"]),
        "indexed_file_count": int(baseline["indexed_file_count"]),
    }
    if record:
        metadata = {
            "memory_count": int(baseline["memory_count"]),
            "indexed_file_count": int(baseline["indexed_file_count"]),
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
