"""Evaluate retrieval hit rate@k over eval/questions.json (no LLM calls)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.retriever import Retriever

THRESHOLD = 0.70


def main() -> int:
    """Print a hit table and return exit code 1 if hit rate is below threshold."""
    questions = json.loads((ROOT / "eval" / "questions.json").read_text("utf-8"))
    settings = get_settings()
    retriever = Retriever(settings)
    hits_count = 0
    print(f"{'hit':<4} {'expected_source':<32} question")
    for item in questions:
        sources = [h.source for h in retriever.search(item["question"], settings.top_k)]
        hit = item["expected_source"] in sources
        hits_count += hit
        print(
            f"{'yes' if hit else 'NO':<4} {item['expected_source']:<32} {item['question'][:70]}"
        )
    rate = hits_count / len(questions)
    print(f"\nHit rate@{settings.top_k}: {hits_count}/{len(questions)} = {rate:.0%}")
    return 0 if rate >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
