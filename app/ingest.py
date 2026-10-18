"""Load documents, chunk, embed, and build/save the FAISS index."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import get_settings

INDEX_FILE = "index.faiss"
META_FILE = "meta.json"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    """A source document: its relative path and extracted text."""

    source: str
    text: str


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> SentenceTransformer:
    """Load (once per model name) the sentence-transformers model."""
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Embed texts into L2-normalized float32 vectors."""
    vectors = get_embedder(model_name).encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    return np.asarray(vectors, dtype="float32")


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def load_documents(directory: str | Path) -> list[Document]:
    """Recursively read .txt/.md/.pdf files, skipping empty ones."""
    root = Path(directory)
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_file(path).strip()
        if text:
            docs.append(Document(source=path.relative_to(root).as_posix(), text=text))
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into word-based windows of `size` words sharing `overlap` words."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size (size > 0)")
    words = text.split()
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_index(
    docs: list[Document],
    index_dir: str | Path | None = None,
    model_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Embed all chunks, write index.faiss and meta.json; return the chunk count."""
    cfg = get_settings()
    index_dir = Path(index_dir or cfg.index_dir)
    model_name = model_name or cfg.embed_model
    size = chunk_size or cfg.chunk_size
    overlap = cfg.chunk_overlap if chunk_overlap is None else chunk_overlap

    meta: list[dict] = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc.text, size, overlap)):
            meta.append({"id": len(meta), "source": doc.source, "chunk_index": i, "text": chunk})
    if not meta:
        raise ValueError("No text to index")

    vectors = embed_texts([m["text"] for m in meta], model_name)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    (index_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return len(meta)


def ingest_directory(docs_dir: str | Path | None = None) -> tuple[int, int]:
    """Load documents from the docs dir and rebuild the index; return (docs, chunks)."""
    docs = load_documents(docs_dir or get_settings().docs_dir)
    return len(docs), build_index(docs)


def main() -> None:
    """CLI entry point: build the index and report counts."""
    n_docs, n_chunks = ingest_directory()
    print(f"Indexed {n_docs} documents into {n_chunks} chunks")


if __name__ == "__main__":
    main()
