"""Measure top-k retrieval hit rate (no LLM involved)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.retriever import Retriever  # noqa: E402

QUESTIONS_FILE = Path(__file__).with_name("questions.json")
THRESHOLD = 0.70


def main() -> int:
    """Run all questions, print a table and return the process exit code."""
    cfg = get_settings()
    retriever = Retriever(cfg.index_dir, cfg.embed_model)
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    hits_found = 0
    print(f"{'Hit':<4} {'Expected':<22} Question")
    for item in questions:
        sources = [h.source for h in retriever.search(item["question"], cfg.top_k)]
        hit = item["expected_source"] in sources
        hits_found += hit
        print(f"{'yes' if hit else 'NO':<4} {item['expected_source']:<22} {item['question']}")

    rate = hits_found / len(questions)
    print(f"\nHit rate @ top-{cfg.top_k}: {hits_found}/{len(questions)} = {rate:.0%}")
    return 0 if rate >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
