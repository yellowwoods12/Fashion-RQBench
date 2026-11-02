"""Report the FASHION-RQ benchmark statistics from a stage-2 enriched artifact.

Compares the corpus against the target specification it was generated to hit, so
the "target" column below is the Gemini-authored spec, not a gold standard: the
Qwen3-8B release deliberately deviates on volume and query length (see
docs/BENCHMARK.md §3).

Usage:
    python benchmark_stats.py <enriched.json[.gz]>
"""

import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fashion_rq_common import load_json

TARGET = {
    "clusters_total": 588,
    "clusters_by_level": {"1": 17, "2": 135, "3": 436},
    "enriched_queries": 5380,
    "queries_per_cluster": (2, 12),
    "unique_queries": 5305,
    "unique_pct_of_queries": 98.7,
    "mean_query_words": 10.0,
    "query_word_range": (3, 26),
    "cluster_size_mean": {"1": 823.9, "2": 219.2, "3": 79.5},
    "cluster_size_range": {"1": (6, 5349), "2": (5, 4896), "3": (5, 4888)},
}


def load(path: Path):
    raw = load_json(path)
    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    return [v for v in raw.values() if isinstance(v, dict)]


def main():
    records = load(Path(sys.argv[1]))
    print(f"records: {len(records):,}")

    by_level = Counter(str(r.get("level")) for r in records)
    print(f"\nclusters total: {len(records)}  (target {TARGET['clusters_total']})")
    for level in sorted(by_level):
        tgt = TARGET["clusters_by_level"].get(level)
        print(f"  level {level}: {by_level[level]:>4}  (target {tgt})")

    sizes = [int(r["group_size"]) for r in records if r.get("group_size") is not None]
    print("\ncluster sizes (group_size):")
    for level in sorted(by_level):
        vals = [int(r["group_size"]) for r in records if str(r.get("level")) == level and r.get("group_size") is not None]
        if vals:
            tgt = TARGET["cluster_size_mean"].get(level)
            rng = TARGET["cluster_size_range"].get(level)
            print(f"  level {level}: mean={statistics.mean(vals):.1f} (target {tgt}) "
                  f"min={min(vals)} max={max(vals)} (target range {rng})")

    per_cluster = [len(r.get("enriched_queries") or []) for r in records]
    total_queries = sum(per_cluster)
    nonempty = [n for n in per_cluster if n]

    print(f"\nenriched queries: {total_queries:,}  (target {TARGET['enriched_queries']:,})")
    print(f"queries per cluster: min={min(per_cluster)} max={max(per_cluster)} "
          f"mean={statistics.mean(per_cluster):.1f}  (target range {TARGET['queries_per_cluster']})")
    if nonempty:
        print(f"  clusters with >=1 query: {len(nonempty)}/{len(records)}; "
              f"median={statistics.median(nonempty)}")
    print(f"  distribution: {sorted(Counter(per_cluster).items())}")

    texts = [
        (eq.get("natural_language_query") or "").strip()
        for r in records
        for eq in (r.get("enriched_queries") or [])
        if isinstance(eq, dict)
    ]
    stripped = [t for t in texts if t]
    unique = set(stripped)
    pct = 100.0 * len(unique) / len(stripped) if stripped else 0.0
    lens = [len(t.split()) for t in stripped]

    print(f"\nunique query strings: {len(unique):,} / {len(stripped):,} = {pct:.1f}% "
          f"(target {TARGET['unique_queries']:,} / {TARGET['unique_pct_of_queries']}%)")
    if lens:
        print(f"query words: mean={statistics.mean(lens):.1f} (target {TARGET['mean_query_words']}) "
              f"min={min(lens)} max={max(lens)} (target range {TARGET['query_word_range']})")

    missing_attrs = sum(
        1
        for r in records
        for eq in (r.get("enriched_queries") or [])
        if not (isinstance(eq, dict) and eq.get("attributes"))
    )
    print(f"\nenriched entries missing attributes: {missing_attrs:,}")


if __name__ == "__main__":
    main()