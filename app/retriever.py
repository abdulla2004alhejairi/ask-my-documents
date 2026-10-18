"""Load the FAISS index and search for the most relevant chunks."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss

from app.config import get_settings
from app.ingest import INDEX_FILE, META_FILE, embed_texts


class IndexNotBuiltError(RuntimeError):
    """Raised when the index files do not exist yet."""


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its similarity score."""

    source: str
    chunk_index: int
    text: str
    score: float


class Retriever:
    """Lazily loads index + metadata once (thread-safe) and answers searches."""

    def __init__(self, index_dir: str | Path, model_name: str) -> None:
        self._index_dir = Path(index_dir)
        self._model_name = model_name
        self._lock = threading.Lock()
        self._index: faiss.Index | None = None
        self._meta: list[dict] = []

    @property
    def available(self) -> bool:
        """Whether the index files exist on disk."""
        return (self._index_dir / INDEX_FILE).exists() and (self._index_dir / META_FILE).exists()

    def reset(self) -> None:
        """Drop the in-memory index so the next search reloads from disk."""
        with self._lock:
            self._index = None
            self._meta = []

    def _load(self) -> tuple[faiss.Index, list[dict]]:
        with self._lock:
            if self._index is None:
                if not self.available:
                    raise IndexNotBuiltError(
                        f"Index not found in '{self._index_dir}'. Run `python -m app.ingest`."
                    )
                self._index = faiss.read_index(str(self._index_dir / INDEX_FILE))
                self._meta = json.loads((self._index_dir / META_FILE).read_text(encoding="utf-8"))
            return self._index, self._meta

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Return the top-k chunks for the query, best first."""
        index, meta = self._load()
        k = min(k or get_settings().top_k, index.ntotal)
        scores, ids = index.search(embed_texts([query], self._model_name), k)
        return [
            Hit(
                source=meta[i]["source"],
                chunk_index=meta[i]["chunk_index"],
                text=meta[i]["text"],
                score=float(score),
            )
            for score, i in zip(scores[0], ids[0], strict=True)
            if i >= 0
        ]
