"""
FASHION-RQ Task 1 — Query-Product Ranking (primary task).

For each qualifying leaf cluster in the SID prefix tree, samples one fixed candidate set
(drawn from Exact / Sibling / Related / Irrelevant pools) and pairs it with *every* usable
enriched query for that cluster (~2-12, avg ~6) -- one output row per (cluster, enriched
query), all sharing the cluster's candidate set. A model under test reranks exactly this
candidate set; see fashion_rq_task1_compute_metrics.py for the scoring side.

Output: one JSONL record per (cluster, enriched query):
{
  "query_id": str,
  "cluster_prefix": str,
  "query": str,
  "attribute_tags": [str, ...],      # attribute keys this enriched query populates
  "candidates": [
    {"item_id": str, "relevance": "Exact|Sibling|Related|Irrelevant", "grade": int,
     "productDisplayName": ..., "brandName": ..., "articleType": ..., "baseColour": ...,
     "gender": ..., "usage": ..., "price": ..., "imageURL": ...},
    ...
  ]
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
    parser = argparse.ArgumentParser(description="Build the FASHION-RQ Task 1 (ranking) benchmark.")
    parser.add_argument("--parquet", default=str(common.PARQUET_PATH))
    parser.add_argument("--queries-json", default=str(common.ENRICHED_JSON_PATH))
    parser.add_argument(
        "--output", default=str(common.DEFAULT_OUTPUT_DIR / "task1_ranking_benchmark.jsonl")
    )
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument("--k-exact", type=int, default=3)
    parser.add_argument("--k-sibling", type=int, default=4)
    parser.add_argument("--k-related", type=int, default=4)
    parser.add_argument("--k-irrelevant", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading catalog from {args.parquet} ...")
    df = common.load_items(Path(args.parquet))
    print(f"Loading cluster queries from {args.queries_json} ...")
    cluster_queries = common.load_cluster_queries(Path(args.queries_json))
    lookup = common.item_lookup(df)

    rng = random.Random(args.seed)
    records = []
    skipped = 0
    for prefix, record, pools in common.iter_qualifying_clusters(
        df, cluster_queries, min_cluster_size=args.min_cluster_size, require_related=True
    ):
        candidates = build_candidates(
            rng, pools, lookup, args.k_exact, args.k_sibling, args.k_related, args.k_irrelevant
        )
        if len(candidates) < 2:
            skipped += 1
            continue
        for query in common.usable_enriched_queries(record):
            records.append(
                {
                    "query_id": f"t1-{len(records):05d}",
                    "cluster_prefix": prefix,
                    "query": query.get("natural_language_query", ""),
                    "attribute_tags": common.attribute_tags(query),
                    "candidates": candidates,
                }
            )

    output_path = Path(args.output)
    common.write_jsonl(output_path, records)
    print(
        f"Wrote {len(records)} Task 1 queries ({skipped} clusters skipped) -> {output_path}"
    )


if __name__ == "__main__":
    main()
