"""
FASHION-RQ Task 1 — Query-Product Ranking metrics.

Scores a model's reranking of each query's fixed candidate set against the
Exact(3) > Sibling(2) > Related(1) > Irrelevant(0) grade produced by
fashion_rq_task1_build_ranking.py.

Predictions file (--predictions): JSON object
    {"<query_id>": ["<item_id>", "<item_id>", ...], ...}
one ranked list per query, most-relevant-first, over that query's candidate item_ids.
Any candidate item_id missing from a query's predicted list is treated as ranked last
(worst-case placement) before scoring — so a model must actually rank every candidate to
get full credit.

Metrics: nDCG@5, nDCG@10, Recall@5, Recall@10 (Recall@K = fraction of all "relevant"
candidates, i.e. grade >= 1 / not-Irrelevant, that appear in the top-K), averaged overall
and broken out per attribute tag (the query's explicit color/category/etc. constraints —
this is the attribute-level reporting dimension folded into Task 1 rather than a separate
task).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import fashion_rq_common as common


def ranked_ids_for_query(candidates, predicted_ids):
    grade_lookup = {c["item_id"]: c["grade"] for c in candidates}
    seen = set()
    ordered = []
    for item_id in predicted_ids or []:
        if item_id in grade_lookup and item_id not in seen:
            ordered.append(item_id)
            seen.add(item_id)
    for c in candidates:
        if c["item_id"] not in seen:
            ordered.append(c["item_id"])
            seen.add(c["item_id"])
    return ordered, grade_lookup


def score_query(record, predictions):
    candidates = record["candidates"]
    predicted_ids = predictions.get(record["query_id"], [])
    ordered_ids, grade_lookup = ranked_ids_for_query(candidates, predicted_ids)
    ranked_grades = [grade_lookup[i] for i in ordered_ids]
    relevant_ids = {c["item_id"] for c in candidates if c["grade"] >= 1}

    return {
        "query_id": record["query_id"],
        "attribute_tags": record.get("attribute_tags", []),
        "ndcg@5": common.ndcg_at_k(ranked_grades, 5),
        "ndcg@10": common.ndcg_at_k(ranked_grades, 10),
        "recall@5": common.recall_at_k(ordered_ids, relevant_ids, 5),
        "recall@10": common.recall_at_k(ordered_ids, relevant_ids, 10),
        "had_prediction": record["query_id"] in predictions,
    }


def summarize(per_query):
    def agg(keys, rows):
        return {k: common.mean([r[k] for r in rows]) for k in keys}

    keys = ["ndcg@5", "ndcg@10", "recall@5", "recall@10"]
    overall = agg(keys, per_query)
    overall["num_queries"] = len(per_query)
    overall["num_missing_predictions"] = sum(1 for r in per_query if not r["had_prediction"])

    by_attribute = {}
    all_tags = sorted({tag for r in per_query for tag in r["attribute_tags"]})
    for tag in all_tags:
        with_tag = [r for r in per_query if tag in r["attribute_tags"]]
        without_tag = [r for r in per_query if tag not in r["attribute_tags"]]
        by_attribute[tag] = {
            "with_constraint": {**agg(keys, with_tag), "num_queries": len(with_tag)},
            "without_constraint": {**agg(keys, without_tag), "num_queries": len(without_tag)},
        }

    return {"overall": overall, "by_attribute": by_attribute}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FASHION-RQ Task 1 predictions.")
    parser.add_argument(
        "--benchmark", default=str(common.SHIPPED_TASKS_DIR / "task1_ranking_benchmark.jsonl.gz")
    )
    parser.add_argument("--predictions", required=True, help="JSON: {query_id: [item_id, ...]}")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    records = common.read_jsonl(benchmark_path)
    predictions = common.load_json(Path(args.predictions))

    per_query = [score_query(record, predictions) for record in records]
    summary = summarize(per_query)

    output_path = Path(args.output) if args.output else benchmark_path.with_name(
        benchmark_path.stem + "_metrics.json"
    )
    common.write_json(output_path, {"summary": summary, "per_query": per_query})

    print(json.dumps(summary["overall"], indent=2))
    print(f"Wrote per-query + attribute breakdown metrics -> {output_path}")


if __name__ == "__main__":
    main()
