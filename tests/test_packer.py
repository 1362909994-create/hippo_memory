from __future__ import annotations

from hippocampus_memory.callback import callback_pack, reset_callback_pack
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import SearchResult
from hippocampus_memory.packer import MemoryPacker, _dedupe
from hippocampus_memory.utils import estimate_tokens


def test_sensitive_memory_not_in_default_pack(db):
    writer = MemoryWriter(db)
    writer.write(
        project="glasses",
        memory_type="task_state",
        content="Current goal is to light the TFT display.",
    )
    writer.write(
        project="glasses",
        memory_type="technical_fact",
        content="Sensitive serial number SECRET-123.",
        visibility="sensitive",
        importance=1.0,
    )
    pack = MemoryPacker(db).pack("SECRET TFT display", project="glasses")
    assert "SECRET-123" not in pack
    assert "TFT display" in pack


def test_pack_length_is_limited(db):
    writer = MemoryWriter(db)
    for idx in range(20):
        writer.write(
            project="glasses",
            memory_type="project_context",
            content=f"Display memory {idx} has repeated context for the screen project.",
        )
    pack = MemoryPacker(db).pack("screen project", project="glasses", max_tokens=80)
    assert estimate_tokens(pack) <= 80


def test_pack_prioritizes_structured_memory_over_source_chunks(db):
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="constraint",
        content="Do not widen the storage schema while changing search.",
        importance=0.5,
    )
    writer.write(
        project="demo",
        memory_type="decision",
        content="Search changes should stay inside Retriever and ranker.",
        importance=0.5,
    )
    for index in range(8):
        writer.write(
            project="demo",
            memory_type="source_chunk",
            content=(
                f"Generated source chunk {index}: search storage schema retriever "
                "ranker source context implementation details."
            ),
            importance=0.9,
        )

    pack = MemoryPacker(db).pack(
        "search storage schema retriever ranker",
        project="demo",
        max_tokens=220,
        source_chunk_limit=1,
    )

    assert "Constraints:" in pack
    assert "Do not widen the storage schema" in pack
    assert "Decisions:" in pack
    assert pack.count("Generated source chunk") <= 1


def test_pack_marks_low_confidence_memory(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="technical_fact",
        content="The migration tool might support YAML configuration.",
        confidence=0.5,
    )

    pack = MemoryPacker(db).pack("YAML migration configuration", project="demo")

    assert "[low confidence 0.50]" in pack


def test_compact_pack_reduces_small_task_overhead(db):
    writer = MemoryWriter(db)
    for idx in range(6):
        writer.write(
            project="demo",
            memory_type="technical_fact",
            content=f"Search quality fact {idx}: keep retrieval precise and concise.",
        )
    writer.write(
        project="demo",
        memory_type="constraint",
        content="Search changes must keep private and sensitive filters intact.",
    )

    standard = MemoryPacker(db).pack("search quality", project="demo", max_tokens=800)
    compact = MemoryPacker(db).pack(
        "search quality",
        project="demo",
        max_tokens=800,
        compact=True,
    )

    assert estimate_tokens(compact) < estimate_tokens(standard)
    assert "Open questions:" not in compact
    assert "Source context:" not in compact


def test_compact_pack_keeps_one_source_chunk_when_no_structured_memory(db):
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="source_chunk",
        content="callback.py defines callback_pack and uses MemoryPacker for project recall.",
        summary="callback.py: callback_pack uses MemoryPacker for project recall.",
        importance=0.8,
    )

    compact = MemoryPacker(db).pack(
        "callback project recall",
        project="demo",
        compact=True,
    )

    assert "Source context:" in compact
    assert "callback_pack uses MemoryPacker" in compact


def test_session_dedupe_skips_memories_seen_by_same_packer(db):
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="constraint",
        content="Search must preserve visibility filters.",
        importance=0.9,
    )
    writer.write(
        project="demo",
        memory_type="decision",
        content="Use compact packs for tiny tasks.",
        importance=0.8,
    )

    packer = MemoryPacker(db)
    first = packer.pack("search filters compact", project="demo", session_dedupe=True)
    second = packer.pack("search filters compact", project="demo", session_dedupe=True)

    assert "Search must preserve visibility filters." in first
    assert "Search must preserve visibility filters." not in second


def test_pack_can_exclude_seen_memory_ids(db):
    writer = MemoryWriter(db)
    result = writer.write(
        project="demo",
        memory_type="constraint",
        content="Do not return this already consumed memory.",
        importance=0.9,
    )
    writer.write(
        project="demo",
        memory_type="decision",
        content="Return this alternative memory instead.",
        importance=0.8,
    )

    pack = MemoryPacker(db).pack(
        "memory",
        project="demo",
        exclude_memory_ids=[result.memory_id],
    )

    assert "Do not return this already consumed memory." not in pack
    assert "Return this alternative memory instead." in pack


def test_callback_pack_remembers_seen_memory_ids(db):
    writer = MemoryWriter(db)
    first = writer.write(
        project="demo",
        memory_type="task_state",
        content="Callback first memory should only appear once.",
        importance=0.9,
    )
    writer.write(
        project="demo",
        memory_type="decision",
        content="Callback second memory can be used after the first is seen.",
        importance=0.8,
    )

    first_pack = callback_pack(
        db,
        project="demo",
        intent="callback memory",
        session_key="s1",
        compact=False,
    )
    second_pack = callback_pack(
        db,
        project="demo",
        intent="callback memory",
        session_key="s1",
        compact=False,
    )

    assert first.memory_id in first_pack["included_memory_ids"]
    assert first.memory_id in second_pack["excluded_memory_ids"]
    assert "Callback first memory should only appear once." not in second_pack["pack"]


def test_callback_reset_clears_seen_memory_ids(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Callback reset memory.",
        importance=0.9,
    )
    callback_pack(db, project="demo", intent="callback reset", session_key="s1")

    reset = reset_callback_pack(db, project="demo", session_key="s1")

    assert reset["reset"] is True
    assert db.get_callback_session("demo", "s1") is None


def test_dedupe_removes_near_duplicate_results():
    base = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
        "mu nu xi omicron"
    )
    near = base + " pi"
    different = "storage schema migration sqlite compatibility rollback"

    results = [
        _result("a", base),
        _result("b", near),
        _result("c", different),
    ]

    deduped = _dedupe(results)

    assert [item.memory_id for item in deduped] == ["a", "c"]


def _result(memory_id: str, content: str) -> SearchResult:
    return SearchResult(
        memory_id=memory_id,
        content=content,
        summary=None,
        memory_type="technical_fact",
        project="demo",
        importance=0.5,
        confidence=0.8,
        status="active",
        visibility="project",
        score=0.5,
        matched_reason="test",
    )
