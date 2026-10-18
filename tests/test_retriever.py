"""Tests for ingestion + retrieval on a tiny index."""

from pathlib import Path

import pytest

from app.ingest import build_index, load_documents
from app.retriever import IndexNotBuiltError, Retriever

MODEL = "all-MiniLM-L6-v2"


def test_load_documents_skips_empty(docs_dir: Path) -> None:
    (docs_dir / "empty.md").write_text("  \n", encoding="utf-8")
    sources = {d.source for d in load_documents(docs_dir)}
    assert sources == {"cats.md", "volcanoes.txt", "bread.md"}


def test_top_hit_is_correct_source(docs_dir: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    build_index(load_documents(docs_dir), index_dir=index_dir, model_name=MODEL)
    retriever = Retriever(index_dir, MODEL)
    hits = retriever.search("How does lava erupt from a volcano?", k=2)
    assert hits[0].source == "volcanoes.txt"
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(IndexNotBuiltError):
        Retriever(tmp_path / "nope", MODEL).search("anything", k=1)
