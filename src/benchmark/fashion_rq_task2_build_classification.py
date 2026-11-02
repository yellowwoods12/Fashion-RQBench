"""
FASHION-RQ Task 2 — Hierarchical Relevance Classification.

For each qualifying leaf cluster, samples one class-balanced set of query-product pairs
(2 Exact, 2 Sibling, 2 Related, 2 Irrelevant by default), then pairs that same 8-item set
with *every* usable enriched query for the cluster (~2-12, avg ~6) -- a fresh query_id (and
8 pair rows) per (cluster, enriched query), all reusing the cluster's sampled item ids. A
model under test predicts one of {E, S, R, I} per pair; see
fashion_rq_task2_compute_metrics.py for the scoring side.

Output: one JSONL record per pair:
{
  "pair_id": str,
  "query_id": str,
  "cluster_prefix": str,
  "query": str,
  "item_id": str,
  "label": "E"|"S"|"R"|"I",
  "productDisplayName": ..., "brandName": ..., "articleType": ..., "baseColour": ...,
  "gender": ..., "usage": ..., "price": ..., "imageURL": ...
}
"""

import argparse
import random
from pathlib import Path

import fashion_rq_common as common

LABEL_CODES = {"Exact": "E", "Sibling": "S", "Related": "R", "Irrelevant": "I"}


def build_pairs(rng, pools, lookup, per_class):
    pairs = []
    for pool_key, relevance in (
        ("exact", "Exact"),
        ("sibling", "Sibling"),
        ("related", "Related"),
        ("irrelevant", "Irrelevant"),
    ):
        for item_id in common.sample_with_seed(rng, pools[pool_key], per_class):
            pairs.append((item_id, relevance))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the FASHION-RQ Task 2 (hierarchical relevance classification) benchmark."
    )
    parser.add_argument("--parquet", default=str(common.PARQUET_PATH))
    parser.add_argument("--queries-json", default=str(common.ENRICHED_JSON_PATH))
    parser.add_argument(
        "--output",
        default=str(common.DEFAULT_OUTPUT_DIR / "task2_classification_benchmark.jsonl"),
    )
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument(
        "--per-class", type=int, default=2, help="Pairs sampled per relevance class per cluster."
    )
    args = parser.parse_args()

    print(f"Loading catalog from {args.parquet} ...")
    df = common.load_items(Path(args.parquet))
    print(f"Loading cluster queries from {args.queries_json} ...")
    cluster_queries = common.load_cluster_queries(Path(args.queries_json))
    lookup = common.item_lookup(df)

    rng = random.Random(args.seed)
    records = []
    query_counter = 0
    for prefix, record, pools in common.iter_qualifying_clusters(
        df, cluster_queries, min_cluster_size=args.min_cluster_size, require_related=True
    ):
        pairs = build_pairs(rng, pools, lookup, args.per_class)
        if len(pairs) < 2:
            continue
        for query in common.usable_enriched_queries(record):
            query_id = f"t2-q{query_counter:05d}"
            query_counter += 1
            for item_id, relevance in pairs:
                meta = lookup.get(item_id, {})
                records.append(
                    {
                        "pair_id": f"{query_id}-{len(records):06d}",
                        "query_id": query_id,
                        "cluster_prefix": prefix,
                        "query": query.get("natural_language_query", ""),
                        "item_id": item_id,
                        "label": LABEL_CODES[relevance],
                        **{k: v for k, v in meta.items() if k != "id"},
                    }
                )

    output_path = Path(args.output)
    common.write_jsonl(output_path, records)
    print(
        f"Wrote {len(records)} Task 2 pairs across {query_counter} queries -> {output_path}"
    )


if __name__ == "__main__":
    main()
