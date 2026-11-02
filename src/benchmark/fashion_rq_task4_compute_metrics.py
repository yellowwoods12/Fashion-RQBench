"""
FASHION-RQ Task 4 — Query Paraphrase Robustness metrics.

Predictions file (--predictions): JSON object {"<query_id>": ["<item_id>", ...], ...},
same shape as Task 1 — one ranked list per individual paraphrase query_id. Missing
candidates are pushed to the end of a query's ranking (worst-case) before scoring,
same convention as fashion_rq_task1_compute_metrics.py.

Metrics:
  - Consistency@K (primary): mean pairwise Jaccard overlap of the top-K item sets
    across every pair of paraphrases within a group, averaged over all groups.
  - nDCG@10 per paraphrase, then mean/stdev *within* each group (a low stdev means the
    model retrieves equally well regardless of phrasing — a secondary stability signal
    alongside Consistency@K), averaged across groups.
"""

import argparse
import itertools
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


def score_group(record, predictions, ks=(5, 10)):
    candidates = record["candidates"]
    per_query = {}
    for q in record["queries"]:
        predicted_ids = predictions.get(q["query_id"], [])
        ordered_ids, grade_lookup = ranked_ids_for_query(candidates, predicted_ids)
        per_query[q["query_id"]] = {
            "intent": q.get("intent", ""),
            "ordered_ids": ordered_ids,
            "ndcg@10": common.ndcg_at_k([grade_lookup[i] for i in ordered_ids], 10),
            "had_prediction": q["query_id"] in predictions,
        }

    pair_jaccards = {k: [] for k in ks}
    query_ids = list(per_query.keys())
    for a, b in itertools.combinations(query_ids, 2):
        for k in ks:
            set_a = set(per_query[a]["ordered_ids"][:k])
            set_b = set(per_query[b]["ordered_ids"][:k])
            pair_jaccards[k].append(common.jaccard(set_a, set_b))

    ndcgs = [v["ndcg@10"] for v in per_query.values()]
    return {
        "group_id": record["group_id"],
        "num_paraphrases": len(query_ids),
        "consistency": {f"@{k}": common.mean(pair_jaccards[k]) for k in ks},
        "ndcg@10_mean": common.mean(ndcgs),
        "ndcg@10_stdev": common.stdev(ndcgs),
        "num_missing_predictions": sum(1 for v in per_query.values() if not v["had_prediction"]),
    }


def summarize(per_group, ks=(5, 10)):
    return {
        "num_groups": len(per_group),
        "consistency": {
            f"@{k}": common.mean([g["consistency"][f"@{k}"] for g in per_group]) for k in ks
        },
        "mean_ndcg@10_within_group": common.mean([g["ndcg@10_mean"] for g in per_group]),
        "mean_ndcg@10_stdev_within_group": common.mean([g["ndcg@10_stdev"] for g in per_group]),
        "num_missing_predictions": sum(g["num_missing_predictions"] for g in per_group),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FASHION-RQ Task 4 predictions.")
    parser.add_argument(
        "--benchmark", default=str(common.SHIPPED_TASKS_DIR / "task4_paraphrase_benchmark.jsonl.gz")
    )
    parser.add_argument("--predictions", required=True, help="JSON: {query_id: [item_id, ...]}")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    records = common.read_jsonl(benchmark_path)
    predictions = common.load_json(Path(args.predictions))

    per_group = [score_group(record, predictions) for record in records]
    summary = summarize(per_group)

    output_path = Path(args.output) if args.output else benchmark_path.with_name(
        benchmark_path.stem + "_metrics.json"
    )
    common.write_json(output_path, {"summary": summary, "per_group": per_group})

    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics -> {output_path}")


if __name__ == "__main__":
    main()
