#!/usr/bin/env python3
"""A minimal BM25 baseline for Task 1 (Query-Product Ranking).

The benchmark gives each query a fixed candidate pool, so this script ranks the
candidates in that pool rather than searching the full catalog. That makes it a
smoke test and a worked scoring example, not a leaderboard baseline.

Predictions are written in the format expected by
``src/benchmark/fashion_rq_task1_compute_metrics.py``::

    {"<query_id>": ["<item_id>", ...], ...}

Usage:
    python examples/bm25_baseline.py
    python examples/bm25_baseline.py --limit 25 --output /tmp/preds.json
"""

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = ROOT / "benchmark/tasks/task1_ranking_benchmark.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "examples/predictions/task1_example.json"

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that "
    "the this to was were will with without".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def candidate_text(candidate: dict) -> str:
    fields = (
        "productDisplayName",
        "brandName",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "gender",
        "usage",
    )
    return " ".join(str(candidate.get(field, "")) for field in fields)


def bm25_rank(query: str, candidates: list[dict], k1: float = 1.2, b: float = 0.75) -> list[str]:
    query_terms = tokenize(query)
    docs = [tokenize(candidate_text(c)) for c in candidates]
    n_docs = len(docs)
    if not n_docs:
        return []

    avg_len = sum(len(doc) for doc in docs) / n_docs
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))

    scores = []
    for doc in docs:
        term_freq = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in term_freq:
                continue
            idf = math.log(1.0 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = term_freq[term] + k1 * (1.0 - b + b * len(doc) / avg_len if avg_len else 1.0)
            score += idf * term_freq[term] * (k1 + 1.0) / norm
        scores.append(score)

    order = sorted(range(n_docs), key=lambda i: (-scores[i], i))
    return [candidates[i]["item_id"] for i in order]


def read_records(path: Path, limit: int | None):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK,
                        help="Task 1 JSONL(.gz) file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the predictions JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N queries (for a quick smoke test)")
    args = parser.parse_args()

    predictions = {}
    for record in read_records(args.benchmark, args.limit):
        predictions[record["query_id"]] = bm25_rank(record["query"], record["candidates"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(predictions)} ranked queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
