"""
FASHION-RQ Task 4 — Query Paraphrase Robustness.

For each cluster that qualifies for Task 1, collects all of that cluster's usable enriched
queries (~2-12, avg ~6 distinct natural-language phrasings of the same cluster) as one
paraphrase group, all sharing a single candidate set — sharing the candidate set is what
makes the top-K overlap (Jaccard) between paraphrases meaningful, since it isolates "did
wording change what got retrieved" from "did the pool change."

Output: one JSONL record per paraphrase group:
{
  "group_id": str,
  "cluster_prefix": str,
  "queries": [{"query_id": str, "query": str}, ...],
  "candidates": [ ... same shape as fashion_rq_task1_build_ranking.py ... ]
}
"""

import argparse
import random
from pathlib import Path

import fashion_rq_common as common


def build_candidates(rng, pools, lookup, k_exact, k_sibling, k_related, k_irrelevant):
    picks = []
    for pool_key, relevance, k in (
        ("exact", "Exact", k_exact),
        ("sibling", "Sibling", k_sibling),
        ("related", "Related", k_related),
        ("irrelevant", "Irrelevant", k_irrelevant),
    ):
        for item_id in common.sample_with_seed(rng, pools[pool_key], k):
            picks.append((item_id, relevance))
    rng.shuffle(picks)

    candidates = []
    for item_id, relevance in picks:
        meta = lookup.get(item_id, {})
        candidates.append(
            {
                "item_id": item_id,
                "relevance": relevance,
                "grade": common.RELEVANCE_LABELS[relevance],
                **{k: v for k, v in meta.items() if k != "id"},
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the FASHION-RQ Task 4 (query paraphrase robustness) benchmark."
    )
    parser.add_argument("--parquet", default=str(common.PARQUET_PATH))
    parser.add_argument("--queries-json", default=str(common.ENRICHED_JSON_PATH))
    parser.add_argument(
        "--output", default=str(common.DEFAULT_OUTPUT_DIR / "task4_paraphrase_benchmark.jsonl")
    )
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument("--k-exact", type=int, default=3)
    parser.add_argument("--k-sibling", type=int, default=4)
    parser.add_argument("--k-related", type=int, default=4)
    parser.add_argument("--k-irrelevant", type=int, default=4)
    parser.add_argument("--min-paraphrases", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading catalog from {args.parquet} ...")
    df = common.load_items(Path(args.parquet))
    print(f"Loading cluster queries from {args.queries_json} ...")
    cluster_queries = common.load_cluster_queries(Path(args.queries_json))
    lookup = common.item_lookup(df)

    rng = random.Random(args.seed)
    records = []
    skipped_too_few_paraphrases = 0
    for prefix, record, pools in common.iter_qualifying_clusters(
        df, cluster_queries, min_cluster_size=args.min_cluster_size, require_related=True
    ):
        paraphrases = common.usable_enriched_queries(record)
        if len(paraphrases) < args.min_paraphrases:
            skipped_too_few_paraphrases += 1
            continue

        candidates = build_candidates(
            rng, pools, lookup, args.k_exact, args.k_sibling, args.k_related, args.k_irrelevant
        )
        if len(candidates) < 2:
            continue

        group_id = f"t4-g{len(records):05d}"
        queries = [
            {
                "query_id": f"{group_id}-{idx}",
                "query": q.get("natural_language_query", ""),
            }
            for idx, q in enumerate(paraphrases)
        ]
        records.append(
            {
                "group_id": group_id,
                "cluster_prefix": prefix,
                "queries": queries,
                "candidates": candidates,
            }
        )

    output_path = Path(args.output)
    common.write_jsonl(output_path, records)
    print(
        f"Wrote {len(records)} Task 4 paraphrase groups "
        f"({skipped_too_few_paraphrases} clusters skipped for < {args.min_paraphrases} paraphrases) -> {output_path}"
    )


if __name__ == "__main__":
    main()
