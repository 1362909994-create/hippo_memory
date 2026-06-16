from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol

from hippocampus_memory.config import Settings
from hippocampus_memory.utils import tokenize


class EmbeddingBackend(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for text."""


class HashEmbeddingBackend:
    """Small deterministic local embedding fallback.

    This is intentionally simple. It keeps the MVP offline and gives the rest of
    the retrieval stack a stable interface that can later be backed by FAISS,
    Chroma, sentence-transformers, OpenAI, or a local model.
    """

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            idx = hash(token) % self.dimensions
            vector[idx] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerBackend:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install -e .[semantic]"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_sentence_embedding_dimension()
        self.dimensions = int(dimension or 384)

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]


def create_embedding_backend(settings: Settings) -> EmbeddingBackend:
    backend = settings.embedding_backend.casefold()
    if backend in {"sentence-transformers", "sentence_transformers", "sentence"}:
        try:
            return _cached_sentence_backend(settings.sentence_transformer_model)
        except Exception:
            return HashEmbeddingBackend(settings.vector_dimensions)
    return HashEmbeddingBackend(settings.vector_dimensions)


@lru_cache(maxsize=2)
def _cached_sentence_backend(model_name: str) -> SentenceTransformerBackend:
    return SentenceTransformerBackend(model_name)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))
