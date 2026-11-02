"""
FASHION-RQ Task 3 — Sibling Discrimination metrics.

Predictions file (--predictions): JSON object where each value is EITHER
    - a ranked list of item_ids over that query's candidates (ranking mode), or
    - a single selected item_id (selection mode)
Both formulations from the task spec are supported per-query, auto-detected by value type;
a query missing from the predictions file scores 0 on every metric.

Since each query's candidate set is small (1 target + a handful of siblings), ranking
metrics are reported at K=1 and K=3 rather than K=5/10 — Recall@K with K close to the
pool size is not discriminating. Ranking-mode relevance is binary (target=1, sibling=0).
Selection-mode metric is plain accuracy (predicted id in correct_item_ids).
"""

import argparse
import json
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
    correct_ids = set(record["correct_item_ids"])
    pred = predictions.get(record["query_id"])
    result = {"query_id": record["query_id"], "had_prediction": pred is not None}

    if isinstance(pred, list):
        ordered_ids, grade_lookup = ranked_ids_for_query(candidates, pred)
        ranked_grades = [grade_lookup[i] for i in ordered_ids]
        result.update(
            {
                "mode": "ranking",
                "ndcg@1": common.ndcg_at_k(ranked_grades, 1),
                "ndcg@3": common.ndcg_at_k(ranked_grades, 3),
                "recall@1": common.recall_at_k(ordered_ids, correct_ids, 1),
                "recall@3": common.recall_at_k(ordered_ids, correct_ids, 3),
            }
        )
    elif isinstance(pred, str):
        result.update({"mode": "selection", "correct": 1.0 if pred in correct_ids else 0.0})
    else:
        result.update(
            {
                "mode": "none",
                "ndcg@1": 0.0,
                "ndcg@3": 0.0,
                "recall@1": 0.0,
                "recall@3": 0.0,
                "correct": 0.0,
            }
        )
    return result


def summarize(per_query):
    ranking_rows = [r for r in per_query if r.get("mode") == "ranking"]
    selection_rows = [r for r in per_query if r.get("mode") == "selection"]
    missing = [r for r in per_query if not r["had_prediction"]]

    summary = {
        "num_queries": len(per_query),
        "num_missing_predictions": len(missing),
        "num_ranking_mode": len(ranking_rows),
        "num_selection_mode": len(selection_rows),
    }
    if ranking_rows:
        summary["ranking"] = {
            "ndcg@1": common.mean([r["ndcg@1"] for r in ranking_rows]),
            "ndcg@3": common.mean([r["ndcg@3"] for r in ranking_rows]),
            "recall@1": common.mean([r["recall@1"] for r in ranking_rows]),
            "recall@3": common.mean([r["recall@3"] for r in ranking_rows]),
        }
    if selection_rows:
        summary["selection"] = {"accuracy": common.mean([r["correct"] for r in selection_rows])}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FASHION-RQ Task 3 predictions.")
    parser.add_argument(
        "--benchmark",
        default=str(common.SHIPPED_TASKS_DIR / "task3_sibling_discrimination_benchmark.jsonl.gz"),
    )
    parser.add_argument(
        "--predictions", required=True, help="JSON: {query_id: [item_id,...] or item_id}"
    )
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

    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics -> {output_path}")


if __name__ == "__main__":
    main()
