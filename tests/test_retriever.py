from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import MemoryRecord
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.ranker import explain_memory_score, rank_memory
from hippocampus_memory.retriever import Retriever


class ExplodingEmbeddingBackend:
    dimensions = 3

    def embed(self, text: str) -> list[float]:
        raise AssertionError("semantic backend should not be called in keyword mode")


def test_keyword_search_finds_memory(db):
    MemoryWriter(db).write(
        project="glasses",
        memory_type="technical_fact",
        content="The TFT screen uses SPI for the first prototype.",
        entities=["TFT"],
    )
    results = Retriever(db).search("SPI screen", project="glasses", search_mode="keyword")
    assert results
    assert "SPI" in results[0].content


def test_keyword_search_works_when_semantic_backend_unavailable(db):
    MemoryWriter(db).write(
        project="glasses",
        memory_type="constraint",
        content="Keyword fallback must keep working.",
    )
    retriever = Retriever(db, embedding_backend=ExplodingEmbeddingBackend())
    results = retriever.search("fallback", project="glasses", search_mode="keyword")
    assert results
    assert results[0].memory_type == "constraint"


def test_project_filter_limits_results(db):
    writer = MemoryWriter(db)
    writer.write(project="alpha", memory_type="task_state", content="Alpha project display task.")
    writer.write(project="beta", memory_type="task_state", content="Beta project display task.")
    results = Retriever(db).search("display task", project="alpha", search_mode="keyword", top_k=10)
    assert results
    assert all(result.project == "alpha" for result in results)


def test_project_search_does_not_mix_global_memory(db):
    writer = MemoryWriter(db)
    writer.write(
        project=None,
        memory_type="technical_fact",
        content="Global display fact should not enter project search.",
        visibility="global",
    )
    writer.write(
        project="alpha",
        memory_type="task_state",
        content="Alpha project display task.",
    )

    results = Retriever(db).search("display", project="alpha", search_mode="keyword", top_k=10)

    assert results
    assert all("Global display fact" not in result.content for result in results)


def test_deleted_memory_is_not_recalled_by_default(db):
    result = MemoryWriter(db).write(
        project="glasses",
        memory_type="failure",
        content="This deleted failure should disappear.",
    )
    db.update_memory_status(result.memory_id, "deleted")
    results = Retriever(db).search("deleted failure", project="glasses", search_mode="keyword")
    assert results == []


def test_sensitive_memory_can_be_recalled_when_explicitly_included(db):
    writer = MemoryWriter(db)
    writer.write(
        project="glasses",
        memory_type="constraint",
        content="HUNTER2 sentinel should only appear when explicitly requested.",
        visibility="sensitive",
        importance=1.0,
    )
    writer.write(
        project="glasses",
        memory_type="constraint",
        content="General display constraint should not hide an explicit sensitive hit.",
        importance=1.0,
    )

    default_results = Retriever(db).search("HUNTER2", project="glasses", search_mode="hybrid")
    included_results = Retriever(db).search(
        "HUNTER2",
        project="glasses",
        search_mode="hybrid",
        include_sensitive=True,
    )

    assert all("HUNTER2" not in result.content for result in default_results)
    assert included_results
    assert "HUNTER2" in included_results[0].content


def test_search_can_filter_by_entities_and_tags(db):
    writer = MemoryWriter(db)
    writer.write(
        project="glasses",
        memory_type="technical_fact",
        content="The TFT display uses SPI.",
        entities=["TFT"],
        tags=["display"],
    )
    writer.write(
        project="glasses",
        memory_type="technical_fact",
        content="The UART bridge uses DMA.",
        entities=["UART"],
        tags=["serial"],
    )

    results = Retriever(db).search(
        "uses",
        project="glasses",
        entities=["tft"],
        tags=["DISPLAY"],
        search_mode="keyword",
    )

    assert len(results) == 1
    assert results[0].entities == ["TFT"]


def test_search_retrieves_chinese_memory(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="technical_fact",
        content="中文检索哨兵：接口鉴权失败时先检查本地令牌缓存，再检查重试策略和错误码映射。",
        entities=["鉴权", "令牌缓存"],
        tags=["中文检索"],
        importance=0.9,
    )

    results = Retriever(db).search("接口 鉴权 失败 令牌 缓存", project="demo", top_k=5)

    assert any("中文检索哨兵" in result.content for result in results)


def test_search_dedupes_near_duplicate_results(db):
    writer = MemoryWriter(db)
    base = (
        "Connection pool configuration uses dynamic sizing with minimum five "
        "connections maximum twenty connections timeout thirty seconds."
    )
    writer.write(project="demo", memory_type="technical_fact", content=base, importance=0.8)
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content=base + " Extra implementation note.",
        importance=0.7,
    )
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Retry policy uses exponential backoff for failed queue jobs.",
        importance=0.7,
    )

    deduped = Retriever(db).search("connection pool dynamic sizing", project="demo", top_k=5)
    raw = Retriever(db).search(
        "connection pool dynamic sizing",
        project="demo",
        top_k=5,
        dedupe_results=False,
    )

    assert len(raw) > len(deduped)
    assert sum("Connection pool configuration" in result.content for result in deduped) == 1


def test_pack_does_not_pull_low_value_project_bulk_without_query_overlap(db):
    writer = MemoryWriter(db)
    for index in range(30):
        writer.write(
            project="demo",
            memory_type="technical_fact",
            content=f"IRRELEVANT_BULK_{index}: stale unrelated visual theme note.",
            importance=0.3,
        )
    writer.write(
        project="demo",
        memory_type="decision",
        content="TOKEN_SENTINEL callback dedupe uses seen memory ids.",
        importance=0.9,
    )

    pack = MemoryPacker(db).pack("TOKEN_SENTINEL callback dedupe", project="demo")

    assert "TOKEN_SENTINEL" in pack
    assert "IRRELEVANT_BULK" not in pack


def test_ranker_boosts_structured_memory_over_source_chunks():
    common = {
        "project": "demo",
        "confidence": 0.8,
        "importance": 0.5,
        "status": "active",
        "visibility": "project",
    }
    constraint = MemoryRecord(
        id="mem_constraint",
        content="Keep the storage schema stable.",
        memory_type="constraint",
        **common,
    )
    source_chunk = MemoryRecord(
        id="mem_chunk",
        content="Keep the storage schema stable.",
        memory_type="source_chunk",
        **common,
    )

    constraint_score, _ = rank_memory(
        constraint,
        keyword_score=0.5,
        semantic_score=0.5,
        project="demo",
    )
    chunk_score, _ = rank_memory(
        source_chunk,
        keyword_score=0.5,
        semantic_score=0.5,
        project="demo",
    )

    assert constraint_score > chunk_score


def test_ranker_exposes_score_breakdown():
    memory = MemoryRecord(
        id="mem_constraint",
        content="Keep the storage schema stable.",
        memory_type="constraint",
        project="demo",
        confidence=0.8,
        importance=0.9,
        status="active",
        visibility="project",
        usage_count=3,
    )

    explanation = explain_memory_score(
        memory,
        keyword_score=0.5,
        semantic_score=0.25,
        project="demo",
    )

    assert explanation.score > 0
    assert "keyword=0.50" in explanation.reason
    assert "project_match" in explanation.reason
    assert explanation.factors["keyword"] == 0.5
    assert explanation.factors["project_match"] == 1.0
    assert explanation.factors["type_boost"] > 0


def test_retriever_results_include_score_details(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="constraint",
        content="Search must explain why a memory was recalled.",
        importance=0.9,
    )

    results = Retriever(db).search("explain recalled memory", project="demo")

    assert results
    assert results[0].score_details["keyword"] > 0
    assert results[0].score_details["project_match"] == 1.0
