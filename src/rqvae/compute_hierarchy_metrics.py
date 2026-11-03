#!/usr/bin/env python3
"""
Hierarchy-quality metrics for the Fashion-RQBench semantic-ID set.

Computes the columns of the exclusivity-loss ablation table for the SID set that
was actually generated *with* L_excl (the purity_v3 lineage):

  Purity L1/L2/L3   per-cluster dominant-attribute share, item-weighted, using the
                    attribute each level's SupCon term supervises
  Cov.              catalog coverage (see PRIMARY CHOICE OF DEFINITION below)
  Frag.             attribute fragmentation: how many leaf clusters a supervised
                    attribute value is spread across

Definitions that the repository does not fix are chosen here and printed with the
run so the numbers are reproducible and re-cuttable. Two conventions are taken from
the repo rather than invented:
  * MIN_CLUSTER_SIZE = 2  (reports/rqvae_purity_v3_report.py) -- clusters smaller
    than this are excluded from the aggregate
  * purity aggregation = item-weighted mean of the dominant share, matching
    summarize()'s weighted_mean in the same report
The per-level supervision is read from src_3token/train_rqvae_purity.py:
  L0 (prefix_1, sid_0)            masterCategory
  L1 (prefix_2, sid_0+sid_1)      articleType
  L2 (prefix_3, full semantic_id) all articleType x <attr> combos at once

PRIMARY CHOICE OF DEFINITION
  Cov. is reported as the share of catalog items that fall in a leaf cluster with
  >= MIN_CLUSTER_SIZE items, i.e. items the hierarchy actually groups rather than
  leaves isolated. Variants (>= 5 items; codebook utilisation) are printed too,
  because "catalog coverage" has no definition in the repo and the headline number
  depends on which one the table means.
  Frag. is reported as the mean number of distinct leaf clusters per supervised
  attribute value, averaged over the 7 fine-level combos (label-unweighted, with
  the item-weighted figure printed alongside); lower is better.

Requires data/items_with_sids.parquet (scripts/download_data.py).

Usage:
  python compute_hierarchy_metrics.py            # table + secondary metrics
  python compute_hierarchy_metrics.py --json out.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PARQUET = Path(
    os.environ.get("SID_ITEMS_PARQUET", str(REPO_ROOT / "data" / "items_with_sids.parquet"))
)

MIN_CLUSTER_SIZE = 2          # reports/rqvae_purity_v3_report.py
COVERAGE_VARIANTS = (2, 5)    # >=2 = this repo's cluster floor; >=5 = benchmark's usability floor

# Fine-level SupCon labels: articleType x each attribute (src_3token/train_rqvae_purity.py)
COMBO_TARGETS = ["baseColour", "gender", "usage", "season", "ageGroup", "price_tier", "fashionType"]

# Levels as numbered in docs/HIERARCHY_QUALITY.md, with the attribute each level supervises
LEVELS = [
    ("L1", "prefix_1", "masterCategory"),
    ("L2", "prefix_2", "articleType"),
    ("L3", "prefix_3", None),   # the 7 combos, handled separately
]


def price_tier(price) -> Optional[str]:
    """Exact thresholds from train_rqvae_purity.py::_price_tier."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p < 711:
        return "budget"
    if p <= 1595:
        return "mid-range"
    return "premium"


def load() -> pl.DataFrame:
    cols = ["id", "discountedPrice", "masterCategory", "subCategory", "articleType",
            "baseColour", "gender", "usage", "season", "ageGroup", "fashionType",
            "sid_0", "sid_1", "sid_2"]
    df = pl.read_parquet(PARQUET, columns=cols)
    df = df.with_columns([
        pl.format("<|sid_{}|>", pl.col("sid_0")).alias("prefix_1"),
        pl.format("<|sid_{}|><|sid_{}|>", pl.col("sid_0"), pl.col("sid_1")).alias("prefix_2"),
        pl.format("<|sid_{}|><|sid_{}|><|sid_{}|>", pl.col("sid_0"), pl.col("sid_1"),
                  pl.col("sid_2")).alias("prefix_3"),
        pl.col("discountedPrice").map_elements(price_tier, return_dtype=pl.Utf8).alias("price_tier"),
    ])
    # Combo labels: f"{a}__{b}", null when either parent is null (mirrors the dataset builder)
    for target in COMBO_TARGETS:
        df = df.with_columns(
            pl.when(pl.col("articleType").is_not_null() & pl.col(target).is_not_null())
            .then(pl.col("articleType") + pl.lit("__") + pl.col(target).cast(pl.Utf8))
            .otherwise(None)
            .alias(f"combo::{target}")
        )
    return df


def dominance(df: pl.DataFrame, group_col: str, label_col: str
              ) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    (per-group dominance, per-label dominance) over non-null labels.

    per-group: size, and the dominant label's share (max class count / size)
    per-label: size, and the dominant group's share -- this second frame is what the
               repo calls exclusivity, the quantity L_excl is aimed at
    """
    sub = df.select([group_col, label_col]).drop_nulls()
    by_group = (sub.group_by([group_col, label_col]).len()
                .group_by(group_col)
                .agg(pl.col("len").max().alias("dom_n"), pl.col("len").sum().alias("size"))
                .with_columns((pl.col("dom_n") / pl.col("size")).alias("dom_share")))
    by_label = (sub.group_by([label_col, group_col]).len()
                .group_by(label_col)
                .agg(pl.col("len").max().alias("dom_n"), pl.col("len").sum().alias("size"),
                     pl.col(group_col).n_unique().alias("n_groups"))
                .with_columns((pl.col("dom_n") / pl.col("size")).alias("dom_share")))
    return by_group, by_label


def aggregate(frame: pl.DataFrame, min_size: int = MIN_CLUSTER_SIZE) -> Dict[str, float]:
    """Item-weighted dominant share over groups with >= min_size items."""
    q = frame.filter(pl.col("size") >= min_size)
    if q.height == 0:
        return {"n_groups": 0, "weighted_mean": float("nan"), "unweighted_mean": float("nan")}
    return {
        "n_groups": q.height,
        "weighted_mean": float((q["dom_share"] * q["size"]).sum() / q["size"].sum()),
        "unweighted_mean": float(q["dom_share"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, default=PARQUET,
                    help="Catalog + SIDs to score (default: data/items_with_sids.parquet)")
    ap.add_argument("--json", type=Path, default=None, help="Also write results as JSON")
    args = ap.parse_args()

    global PARQUET
    PARQUET = args.parquet
    print(f"Loading {PARQUET.name} …")
    df = load()
    n = df.height
    print(f"  {n:,} items\n")

    results: Dict[str, object] = {"n_items": n, "min_cluster_size": MIN_CLUSTER_SIZE}

    print("=" * 78)
    print("PURITY  (per-cluster dominant share of the level's supervised attribute,")
    print("        item-weighted over clusters with >= %d items)" % MIN_CLUSTER_SIZE)
    print("=" * 78)
    purity: Dict[str, float] = {}
    for name, group_col, label_col in LEVELS:
        if label_col is None:
            per_combo, shares = {}, []
            for target in COMBO_TARGETS:
                frame, _ = dominance(df, group_col, f"combo::{target}")
                agg = aggregate(frame)
                per_combo[target] = agg["weighted_mean"]
                shares.append(agg["weighted_mean"])
            val = sum(shares) / len(shares)
            purity[name] = val
            print(f"{name} ({group_col}, all {len(COMBO_TARGETS)} combos, mean): {100*val:.1f}%")
            for target, v in per_combo.items():
                print(f"      {target:<12} {100*v:.1f}%  ({df[f'combo::{target}'].n_unique()} values)")
            results["purity_per_combo"] = {k: 100 * v for k, v in per_combo.items()}
        else:
            frame, _ = dominance(df, group_col, label_col)
            agg = aggregate(frame)
            val = agg["weighted_mean"]
            purity[name] = val
            all_sizes = df[group_col].n_unique()
            print(f"{name} ({group_col}, {label_col}): {100*val:.1f}%  "
                  f"[unweighted {100*agg['unweighted_mean']:.1f}%, "
                  f"{agg['n_groups']} of {all_sizes} clusters have >= {MIN_CLUSTER_SIZE} items]")
            results[f"purity_{name.lower()}_unweighted"] = 100 * agg["unweighted_mean"]
    results["purity"] = {k: 100 * v for k, v in purity.items()}

    print()
    print("=" * 78)
    print("COV.  catalog coverage")
    print("=" * 78)
    leaf = df.group_by("prefix_3").len().rename({"len": "size"})
    cov = {}
    for k in COVERAGE_VARIANTS:
        items = int(leaf.filter(pl.col("size") >= k)["size"].sum())
        cov[k] = 100 * items / n
        print(f"  items in leaf clusters with >= {k} items: {items:,} / {n:,}  "
              f"= {cov[k]:.1f}%   [{'PRIMARY' if k == 2 else 'variant'}]")
    codebook = {f"L{i+1}": 100 * df[f"sid_{i}"].n_unique() / 64 for i in range(3)}
    print("  codebook utilisation (% of 64 codes used per level): "
          + ", ".join(f"{k} {v:.0f}%" for k, v in codebook.items())
          + f"  [variant, mean {sum(codebook.values())/3:.1f}%]")
    singleton_items = int(leaf.filter(pl.col("size") == 1)["size"].sum())
    print(f"  singleton leaves: {leaf.filter(pl.col('size') == 1).height} "
          f"holding {singleton_items:,} items ({100*singleton_items/n:.1f}%)")
    results["coverage"] = {f"ge{k}": cov[k] for k in COVERAGE_VARIANTS}
    results["codebook_utilisation"] = codebook

    print()
    print("=" * 78)
    print("FRAG.  attribute fragmentation — distinct leaf clusters per attribute value")
    print("=" * 78)
    frag_unweighted, frag_weighted = [], []
    for target in COMBO_TARGETS:
        _, by_label = dominance(df, "prefix_3", f"combo::{target}")
        q = by_label.filter(pl.col("size") >= MIN_CLUSTER_SIZE)
        frag_unweighted.append(float(q["n_groups"].mean()))
        frag_weighted.append(float((q["n_groups"] * q["size"]).sum() / q["size"].sum()))
    frag = sum(frag_unweighted) / len(frag_unweighted)
    frag_w = sum(frag_weighted) / len(frag_weighted)
    print(f"  mean leaf clusters per combo value: {frag:.1f}   [PRIMARY, label-unweighted]")
    print(f"  item-weighted:                      {frag_w:.1f}   [variant]")
    _, by_label = dominance(df, "prefix_3", "articleType")
    q = by_label.filter(pl.col("size") >= MIN_CLUSTER_SIZE)
    print(f"  articleType alone:                  {q['n_groups'].mean():.1f} "
          f"leaf clusters per articleType value   [variant]")
    results["fragmentation"] = {"combo_unweighted": frag, "combo_weighted": frag_w,
                                "articletype": float(q["n_groups"].mean())}

    print()
    print("=" * 78)
    print("SECONDARY — exclusivity (per-label dominant-cluster share), the quantity")
    print("            L_excl directly optimises; >= 0.5 share is the repo's threshold")
    print("=" * 78)
    excl = {}
    for name, group_col, label_col in LEVELS:
        if label_col is None:
            vals = []
            for target in COMBO_TARGETS:
                _, by_label = dominance(df, group_col, f"combo::{target}")
                q = by_label.filter(pl.col("size") >= MIN_CLUSTER_SIZE)
                vals.append(float((q["dom_share"] * q["size"]).sum() / q["size"].sum()))
            excl[name] = sum(vals) / len(vals)
            print(f"  {name} (7 combos, mean): {100*excl[name]:.1f}%")
        else:
            _, by_label = dominance(df, group_col, label_col)
            q = by_label.filter(pl.col("size") >= MIN_CLUSTER_SIZE)
            excl[name] = float((q["dom_share"] * q["size"]).sum() / q["size"].sum())
            ge = 100 * float((q["dom_share"] >= 0.5).mean())
            print(f"  {name} ({label_col}): {100*excl[name]:.1f}%   "
                  f"[{ge:.0f}% of values sit >50% inside one cluster]")
    results["exclusivity"] = {k: 100 * v for k, v in excl.items()}

    print()
    print("=" * 78)
    print("VARIANTS — for the readings of 'L1/L2/L3 purity' and the cluster set that")
    print("           the benchmark actually anchors on")
    print("=" * 78)
    print("  raw tree size (distinct prefixes): "
          + ", ".join(f"L{i+1} {df[f'prefix_{i+1}'].n_unique()}" for i in range(3)))
    # The benchmark keeps only prefixes whose articleType purity >= 0.80 with >= 5 items
    # (CLUSTER_PURITY_THRESHOLD / MIN_GROUP_SIZE in the generator). Purity measured on
    # that subset is inflated by construction -- it is the filter's own criterion.
    benchmark_kept = {}
    for i in range(3):
        frame, _ = dominance(df, f"prefix_{i+1}", "articleType")
        keep = frame.filter((pl.col("dom_share") >= 0.80) & (pl.col("size") >= 5))
        benchmark_kept[f"L{i+1}"] = keep.height
    print(f"  benchmark-filtered clusters (articleType purity >= 0.80 and >= 5 items): "
          + ", ".join(f"{k} {v}" for k, v in benchmark_kept.items())
          + "   [generator reports 17/135/436 for its prefix universe]")
    results["benchmark_filtered_counts"] = benchmark_kept

    print("  purity restricted to that filtered subset (inflated by construction):")
    for name, group_col, label_col in LEVELS:
        frame, _ = dominance(df, group_col, "articleType")
        keep = frame.filter((pl.col("dom_share") >= 0.80) & (pl.col("size") >= 5))
        if label_col is None:
            vals = []
            for target in COMBO_TARGETS:
                fr, _ = dominance(df, group_col, f"combo::{target}")
                q = fr.join(keep.select(group_col), on=group_col, how="inner")
                vals.append(float((q["dom_share"] * q["size"]).sum() / q["size"].sum()))
            print(f"      {name}: {100*sum(vals)/len(vals):.1f}%")
        else:
            q = frame.join(keep.select(group_col), on=group_col, how="inner")
            print(f"      {name}: {100*float((q['dom_share']*q['size']).sum()/q['size'].sum()):.1f}%")

    print("  taxonomy reading (supervision replaced by the catalog taxonomy):")
    for name, group_col, label_col in [("L1", "prefix_1", "masterCategory"),
                                       ("L2", "prefix_2", "subCategory"),
                                       ("L3", "prefix_3", "articleType")]:
        frame, _ = dominance(df, group_col, label_col)
        print(f"      {name} ({label_col}): {100*aggregate(frame)['weighted_mean']:.1f}%")

    print()
    print("=" * 78)
    print("TABLE ROW — with L_excl (the currently generated SID set)")
    print("=" * 78)
    print(f"  Purity L1 {100*purity['L1']:.1f} | L2 {100*purity['L2']:.1f} | "
          f"L3 {100*purity['L3']:.1f} | Cov. {cov[2]:.1f} | Frag. {frag:.1f}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
