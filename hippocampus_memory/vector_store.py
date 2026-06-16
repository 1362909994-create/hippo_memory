from __future__ import annotations

from typing import Protocol

from hippocampus_memory.config import Settings
from hippocampus_memory.db import Database
from hippocampus_memory.embedding import cosine_similarity


class VectorStore(Protocol):
    def upsert(self, memory_id: str, vector: list[float]) -> None:
        """Store or replace one vector."""

    def delete(self, memory_id: str) -> None:
        """Delete one vector."""

    def search(
        self,
        query_vector: list[float],
        allowed_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Return memory ids with similarity scores."""


class SQLiteVectorStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, memory_id: str, vector: list[float]) -> None:
        self.db.upsert_vector(memory_id, vector)

    def delete(self, memory_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))

    def search(
        self,
        query_vector: list[float],
        allowed_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        vectors = self.db.get_vectors(allowed_ids)
        scored = [
            (memory_id, cosine_similarity(query_vector, vector))
            for memory_id, vector in vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(memory_id, score) for memory_id, score in scored[:limit] if score > 0]


class ChromaVectorStore:
    def __init__(self, settings: Settings) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Install with: pip install -e .[chroma]"
            ) from exc
        path = settings.chroma_path or (settings.db_path.parent / "chroma")
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection("hippocampus_memories")

    def upsert(self, memory_id: str, vector: list[float]) -> None:
        self.collection.upsert(ids=[memory_id], embeddings=[vector])

    def delete(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def search(
        self,
        query_vector: list[float],
        allowed_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        where = {"id": {"$in": allowed_ids}} if allowed_ids else None
        try:
            result = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where,
            )
        except Exception:
            result = self.collection.query(query_embeddings=[query_vector], n_results=limit)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        scored = []
        for memory_id, distance in zip(ids, distances, strict=False):
            if allowed_ids and memory_id not in allowed_ids:
                continue
            score = 1.0 / (1.0 + float(distance))
            scored.append((memory_id, score))
        return scored


def create_vector_store(db: Database, settings: Settings | None = None) -> VectorStore:
    settings = settings or db.settings
    if settings.vector_backend.casefold() == "chroma":
        try:
            return ChromaVectorStore(settings)
        except Exception:
            return SQLiteVectorStore(db)
    return SQLiteVectorStore(db)
