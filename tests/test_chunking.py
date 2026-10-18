"""Tests for chunk_text."""

import pytest

from app.ingest import chunk_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_window_size_and_overlap() -> None:
    chunks = chunk_text(words(10), size=4, overlap=1)
    assert chunks[0] == "w0 w1 w2 w3"
    assert chunks[1] == "w3 w4 w5 w6"
    assert chunks[-1].endswith("w9")
    assert all(len(c.split()) <= 4 for c in chunks)


def test_short_text_single_chunk() -> None:
    assert chunk_text("hello world", size=5, overlap=1) == ["hello world"]


def test_exact_fit_has_no_trailing_chunk() -> None:
    assert len(chunk_text(words(4), size=4, overlap=1)) == 1


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_input(text: str) -> None:
    assert chunk_text(text, size=4, overlap=1) == []


def test_no_empty_chunks() -> None:
    assert all(chunk_text(words(37), size=5, overlap=2))


@pytest.mark.parametrize(("size", "overlap"), [(4, 4), (4, 5), (0, 0), (4, -1)])
def test_invalid_overlap(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", size=size, overlap=overlap)
