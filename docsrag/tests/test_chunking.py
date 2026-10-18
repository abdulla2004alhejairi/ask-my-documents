"""Tests for word-based chunking."""

import pytest

from app.ingest import chunk_text


def test_window_size_and_overlap() -> None:
    text = " ".join(str(i) for i in range(25))
    chunks = chunk_text(text, size=10, overlap=2)
    assert [len(c.split()) for c in chunks] == [10, 10, 9]
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_short_text_single_chunk() -> None:
    assert chunk_text("a b c", size=10, overlap=2) == ["a b c"]


def test_exact_fit_has_no_trailing_duplicate() -> None:
    assert len(chunk_text(" ".join("w" * 10), size=10, overlap=2)) == 1


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_input(text: str) -> None:
    assert chunk_text(text, size=5, overlap=1) == []


def test_never_returns_empty_chunks() -> None:
    chunks = chunk_text(" ".join(str(i) for i in range(103)), size=7, overlap=3)
    assert all(c.strip() for c in chunks)


@pytest.mark.parametrize("overlap", [5, 6])
def test_invalid_overlap(overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=overlap)
