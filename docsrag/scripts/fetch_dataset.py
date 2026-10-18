"""Download sample documents and eval questions from the SQuAD dataset on Hugging Face.

SQuAD (rajpurkar/squad, CC BY-SA 4.0) contains Wikipedia paragraphs grouped by article
title, each with questions. We rebuild one Markdown document per article and derive
eval questions whose expected source is the article the question was written about.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "rajpurkar/squad"
PAGE = 100
MAX_ROWS = 2500  # rows are grouped by article, so the head covers many articles
NUM_ARTICLES = 12
NUM_QUESTIONS = 10
ROOT = Path(__file__).resolve().parent.parent


def fetch_page(offset: int) -> list[dict]:
    """Fetch one page of validation rows, backing off when rate limited."""
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "plain_text",
            "split": "validation",
            "offset": offset,
            "length": PAGE,
        }
    )
    for attempt in range(10):
        try:
            with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as resp:
                return [item["row"] for item in json.load(resp)["rows"]]
        except urllib.error.HTTPError as err:
            if err.code != 429:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Rate limited by the Hugging Face datasets-server")


def slugify(title: str) -> str:
    """Turn an article title into a safe file stem."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def main() -> None:
    """Download rows, write one .md per article and eval/questions.json."""
    contexts: dict[str, list[str]] = defaultdict(list)
    questions: dict[str, list[str]] = defaultdict(list)
    for offset in range(0, MAX_ROWS, PAGE):
        for row in fetch_page(offset):
            if row["context"] not in contexts[row["title"]]:
                contexts[row["title"]].append(row["context"])
            questions[row["title"]].append(row["question"])
        time.sleep(1)

    titles = list(contexts)[:NUM_ARTICLES]
    docs_dir = ROOT / "data" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for title in titles:
        body = "\n\n".join(contexts[title])
        name = title.replace("_", " ")
        (docs_dir / f"{slugify(title)}.md").write_text(
            f"# {name}\n\n{body}\n", encoding="utf-8"
        )

    items = [
        {"question": qs[len(qs) * frac // 3], "expected_source": f"{slugify(title)}.md"}
        for frac in (1, 2)
        for title in titles
        for qs in [questions[title]]
    ][:NUM_QUESTIONS]
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "questions.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(titles)} documents and {len(items)} questions")


if __name__ == "__main__":
    main()
