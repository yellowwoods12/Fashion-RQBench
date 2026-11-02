"""
FASHION-RQ Task 3 — Sibling Discrimination.

The user's most-valued task: for each qualifying leaf cluster, pits the target
(Exact) items against several Sibling items drawn from *distinct* neighboring leaf
clusters under the same parent (round-robin across sibling leaves), so distractors vary
on different single attributes rather than being repeats of one neighboring cluster —
mirrors the canonical example ("blue floral maxi dress" vs. "blue floral midi dress" /
"red floral maxi dress" / "blue solid maxi dress"). That target+sibling set is sampled once
per cluster and paired with *every* usable enriched query for the cluster (~2-12, avg ~6) --
one output record per (cluster, enriched query), all sharing the same candidates.

Only clusters with >= --min-sibling-leaves distinct sibling leaf clusters available are
used, since sibling *diversity* is the point of this task.

Output supports both formulations described in the task spec:
  - ranking: "candidates" (target + siblings, shuffled, binary grade Exact=1/Sibling=0)
  - selection: "correct_item_ids" (the target items a selection-style prediction should
    pick from the candidate set)

{
  "query_id": str,
  "cluster_prefix": str,
  "query": str,
  "correct_item_ids": [str, ...],
  "distractor_cluster_prefixes": [str, ...],
  "candidates": [
    {"item_id": str, "relevance": "Exact|Sibling", "grade": 0|1, "distractor_cluster_prefix": str|None,
     "productDisplayName": ..., ...},
    ...
  ]
}
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List

import fashion_rq_common as common


def round_robin_sample(rng: random.Random, groups: Dict[str, List[str]], k: int) -> List[tuple]:
    """Pick up to k (item_id, leaf_prefix) pairs, cycling across distinct leaf groups."""
    leaves = list(groups.keys())
    rng.shuffle(leaves)
    pools = {leaf: list(ids) for leaf, ids in groups.items()}
    for pool in pools.values():
        rng.shuffle(pool)

    picked = []
    idx = 0
    max_iters = k * max(len(leaves), 1) + len(leaves) + 10
    while len(picked) < k and idx < max_iters:
        leaf = leaves[idx % len(leaves)]
        if pools[leaf]:
            picked.append((pools[leaf].pop(), leaf))
        idx += 1
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the FASHION-RQ Task 3 (sibling discrimination) benchmark."
    )
    parser.add_argument("--parquet", default=str(common.PARQUET_PATH))
    parser.add_argument("--queries-json", default=str(common.ENRICHED_JSON_PATH))
    parser.add_argument(
        "--output",
        default=str(common.DEFAULT_OUTPUT_DIR / "task3_sibling_discrimination_benchmark.jsonl"),
    )
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument("--min-sibling-leaves", type=int, default=3)
    parser.add_argument("--k-target", type=int, default=1)
    parser.add_argument("--k-sibling", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading catalog from {args.parquet} ...")
    df = common.load_items(Path(args.parquet))
    print(f"Loading cluster queries from {args.queries_json} ...")
    cluster_queries = common.load_cluster_queries(Path(args.queries_json))
    lookup = common.item_lookup(df)

    rng = random.Random(args.seed)
    records = []
    skipped_low_diversity = 0
    for prefix, record, pools in common.iter_qualifying_clusters(
        df, cluster_queries, min_cluster_size=args.min_cluster_size, require_related=False
    ):
        if len(pools["sibling_groups"]) < args.min_sibling_leaves:
            skipped_low_diversity += 1
            continue

        queries = common.usable_enriched_queries(record)
        target_ids = common.sample_with_seed(rng, pools["exact"], args.k_target)
        sibling_picks = round_robin_sample(rng, pools["sibling_groups"], args.k_sibling)
        if not target_ids or not sibling_picks:
            continue

        candidates = []
        for item_id in target_ids:
            meta = lookup.get(item_id, {})
            candidates.append(
                {
                    "item_id": item_id,
                    "relevance": "Exact",
                    "grade": 1,
                    "distractor_cluster_prefix": None,
                    **{k: v for k, v in meta.items() if k != "id"},
                }
            )
        distractor_prefixes = set()
        for item_id, leaf_prefix in sibling_picks:
            meta = lookup.get(item_id, {})
            distractor_prefixes.add(leaf_prefix)
            candidates.append(
                {
                    "item_id": item_id,
                    "relevance": "Sibling",
                    "grade": 0,
                    "distractor_cluster_prefix": leaf_prefix,
                    **{k: v for k, v in meta.items() if k != "id"},
                }
            )
        rng.shuffle(candidates)

        for query in queries:
            records.append(
                {
                    "query_id": f"t3-{len(records):05d}",
                    "cluster_prefix": prefix,
                    "query": query.get("natural_language_query", ""),
                    "correct_item_ids": target_ids,
                    "distractor_cluster_prefixes": sorted(distractor_prefixes),
                    "candidates": candidates,
                }
            )

    output_path = Path(args.output)
    common.write_jsonl(output_path, records)
    print(
        f"Wrote {len(records)} Task 3 queries "
        f"({skipped_low_diversity} clusters skipped for insufficient sibling diversity) -> {output_path}"
    )


if __name__ == "__main__":
    main()
