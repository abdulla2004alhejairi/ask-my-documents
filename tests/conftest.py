"""Shared fixtures."""

from pathlib import Path

import pytest

FIXTURE_DOCS = {
    "cats.md": "Cats are small carnivorous mammals. A cat purrs when content and "
    "uses its whiskers to sense nearby objects. Domestic cats sleep most of the day.",
    "volcanoes.txt": "A volcano is a rupture in the crust of a planet through which "
    "molten lava, ash and gases escape from a magma chamber below the surface.",
    "bread.md": "Bread is baked from flour, water and yeast. Kneading develops gluten "
    "and the dough rises as yeast ferments sugars and releases carbon dioxide.",
}


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """A tmp directory with three small fixture documents."""
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in FIXTURE_DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d
