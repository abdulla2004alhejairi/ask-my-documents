"""Document loading, chunking, embedding and FAISS index construction."""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings

INDEX_FILE = "index.faiss"
META_FILE = "meta.json"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    """A loaded source document."""

    source: str
    text: str


def _read_file(path: Path) -> str:
    """Read the text of a .txt, .md or .pdf file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(directory: Path) -> list[Document]:
    """Recursively load supported documents, skipping empty files."""
    docs: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_file(path).strip()
        if text:
            docs.append(Document(path.relative_to(directory).as_posix(), text))
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into word windows of `size` words sharing `overlap` words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if size <= 0 or overlap < 0:
        raise ValueError("size must be positive and overlap non-negative")
    words = text.split()
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_index(docs: list[Document], settings: Settings | None = None) -> int:
    """Embed all chunks, write the FAISS index and metadata; return chunk count."""
    settings = settings or get_settings()
    meta: list[dict] = []
    for doc in docs:
        pieces = chunk_text(doc.text, settings.chunk_size, settings.chunk_overlap)
        for i, piece in enumerate(pieces):
            meta.append(
                {"id": len(meta), "source": doc.source, "chunk_index": i, "text": piece}
            )
    if not meta:
        raise ValueError("No document chunks to index")

    model = SentenceTransformer(settings.embed_model)
    vectors = model.encode(
        [m["text"] for m in meta], normalize_embeddings=True, show_progress_bar=False
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.index_dir / INDEX_FILE))
    (settings.index_dir / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return len(meta)


def ingest(settings: Settings | None = None) -> tuple[int, int]:
    """Build the index from the configured docs dir; return (documents, chunks)."""
    settings = settings or get_settings()
    docs = load_documents(settings.docs_dir)
    return len(docs), build_index(docs, settings)


if __name__ == "__main__":
    n_docs, n_chunks = ingest()
    print(f"Indexed {n_docs} documents into {n_chunks} chunks")
