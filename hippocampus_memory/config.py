from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    max_file_size_bytes: int = 1_000_000
    vector_dimensions: int = 128
    embedding_backend: str = "hash"
    vector_backend: str = "sqlite"
    chroma_path: Path | None = None
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_pack_min_tokens: int = 300
    default_pack_max_tokens: int = 1500

    @classmethod
    def from_env(cls, db_path: str | Path | None = None) -> Settings:
        raw_path = db_path or os.getenv("HIPPO_DB_PATH")
        if raw_path:
            path = Path(raw_path).expanduser()
        else:
            path = Path.home() / ".hippocampus-memory" / "hippocampus.db"
        max_file_size = int(os.getenv("HIPPO_MAX_FILE_SIZE", "1000000"))
        return cls(
            db_path=path,
            max_file_size_bytes=max_file_size,
            embedding_backend=os.getenv("HIPPO_EMBEDDING_BACKEND", "hash"),
            vector_backend=os.getenv("HIPPO_VECTOR_BACKEND", "sqlite"),
            chroma_path=Path(os.getenv("HIPPO_CHROMA_PATH")).expanduser()
            if os.getenv("HIPPO_CHROMA_PATH")
            else None,
            sentence_transformer_model=os.getenv(
                "HIPPO_SENTENCE_TRANSFORMER_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        )

    def ensure_parent(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
