"""Vector search over the FAISS index built by `app.ingest`."""

import json
import threading
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings
from app.ingest import INDEX_FILE, META_FILE


class IndexNotBuiltError(RuntimeError):
    """Raised when the FAISS index files do not exist yet."""


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its similarity score."""

    source: str
    chunk_index: int
    text: str
    score: float


class Retriever:
    """Lazily loads the index and metadata, then serves top-k searches."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._index: faiss.Index | None = None
        self._meta: list[dict] = []
        self._model: SentenceTransformer | None = None

    @property
    def loaded(self) -> bool:
        """Whether the index has been loaded into memory."""
        return self._index is not None

    @property
    def index_exists(self) -> bool:
        """Whether the index files are present on disk."""
        directory = self._settings.index_dir
        return (directory / INDEX_FILE).exists() and (directory / META_FILE).exists()

    def _load(self) -> None:
        """Load index, metadata and embedding model once (thread-safe)."""
        with self._lock:
            if self._index is not None:
                return
            if not self.index_exists:
                raise IndexNotBuiltError(
                    "Index not built. Run `python -m app.ingest` or POST /ingest."
                )
            directory = self._settings.index_dir
            self._meta = json.loads((directory / META_FILE).read_text("utf-8"))
            self._model = SentenceTransformer(self._settings.embed_model)
            self._index = faiss.read_index(str(directory / INDEX_FILE))

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Return the top-k chunks most similar to the query."""
        self._load()
        assert self._index is not None and self._model is not None
        k = min(k or self._settings.top_k, self._index.ntotal)
        vector = self._model.encode([query], normalize_embeddings=True)
        scores, ids = self._index.search(vector.astype("float32"), k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            m = self._meta[idx]
            hits.append(Hit(m["source"], m["chunk_index"], m["text"], float(score)))
        return hits
