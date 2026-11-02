"""
Shared hierarchy / candidate-pool / metric helpers for the FASHION-RQ benchmark suite.

FASHION-RQ is a 4-task benchmark (Query-Product Ranking, Hierarchical Relevance
Classification, Sibling Discrimination, Query Paraphrase Robustness) built on top of
the public Myntra/Kaggle fashion dataset's existing RQ-VAE semantic-ID prefix tree.

Relevance mapping used by every task (derived directly from the SID prefix tree):
    Exact      = same level-3 prefix (semantic_id)         — the target leaf cluster
    Sibling    = same level-2 prefix, different level-3     — a different leaf under
                                                               the same parent
    Related    = same level-1 prefix, different level-2     — a different parent
                                                               branch under the same
                                                               grandparent
    Irrelevant = different level-1 prefix entirely

Query text/attributes for each leaf cluster are reused as-is from the released
`enriched_queries` corpus (`ENRICHED_JSON_PATH`) — no new LLM calls are made here. Each qualifying cluster
carries ~4-12 enriched queries, each with its own natural-language text and a per-query
`attributes` dict; T1/T2/T3 pair every one of a cluster's usable enriched queries with
that cluster's single sampled candidate pool, and T4 treats the full set as one
paraphrase group.
"""

import gzip
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import polars as pl

# Everything resolves relative to the repository root so the release is relocatable;
# each path is env-overridable for use outside it. The catalog + semantic IDs are
# fetched by scripts/download_data.py (see data/README.md), the query corpus ships
# in-repo under benchmark/queries/.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("FASHION_RQ_DATA_DIR", str(REPO_ROOT / "data")))
PARQUET_PATH = Path(
    os.environ.get("SID_ITEMS_PARQUET", str(DATA_DIR / "items_with_sids.parquet"))
)
ENRICHED_JSON_PATH = Path(
    os.environ.get(
        "FASHION_RQ_ENRICHED_JSON",
        str(REPO_ROOT / "benchmark" / "queries" / "cluster_queries_enriched.json.gz"),
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("FASHION_RQ_TASKS_DIR", str(REPO_ROOT / "benchmark" / "tasks_generated"))
)

# The released task files, which the scorers default to. Regenerated output goes to
# DEFAULT_OUTPUT_DIR instead, so a reproduction run never overwrites the release.
SHIPPED_TASKS_DIR = Path(
    os.environ.get("FASHION_RQ_SHIPPED_TASKS_DIR", str(REPO_ROOT / "benchmark" / "tasks"))
)

RELEVANCE_LABELS: Dict[str, int] = {"Exact": 3, "Sibling": 2, "Related": 1, "Irrelevant": 0}
DEFAULT_SEED = 42

ITEM_METADATA_COLS = [
    "id",
    "productDisplayName",
    "brandName",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "gender",
    "usage",
    "price",
    "imageURL",
]

_SID_TOKEN_RE = re.compile(r"<\|sid_\d+\|>")


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_text(path: Path, mode: str = "rt"):
    """
    Open plain or gzip-compressed text transparently.

    The released benchmark ships compressed (a 57 MB task set is ~7 MB gzipped), so
    every reader goes through here rather than assuming a plain file.
    """
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def load_json(path: Path) -> Any:
    with open_text(path) as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SID prefix helpers
# ─────────────────────────────────────────────────────────────────────────────

def sid_tokens(prefix: str) -> List[str]:
    return _SID_TOKEN_RE.findall(prefix or "")


def token_prefix(prefix: str, n: int) -> str:
    return "".join(sid_tokens(prefix)[:n])


# ─────────────────────────────────────────────────────────────────────────────
# Catalog / cluster-query loading
# ─────────────────────────────────────────────────────────────────────────────

def load_items(parquet_path: Path = PARQUET_PATH) -> pl.DataFrame:
    """Load the purity_v3 catalog and derive prefix_1tok/2tok/3tok columns."""
    df = pl.read_parquet(parquet_path)
    if "embedding" in df.columns:
        df = df.drop("embedding")
    df = df.with_columns(pl.col("id").cast(pl.Utf8))
    df = df.with_columns(
        pl.format("<|sid_{}|>", pl.col("sid_0")).alias("prefix_1tok"),
        pl.format("<|sid_{}|><|sid_{}|>", pl.col("sid_0"), pl.col("sid_1")).alias("prefix_2tok"),
        pl.col("semantic_id").alias("prefix_3tok"),
    )
    return df


def load_cluster_queries(json_path: Path = ENRICHED_JSON_PATH) -> Dict[str, dict]:
    """Load the enriched Gemini-anchor records, keyed by SID prefix string."""
    records = load_json(json_path)
    return {r["prefix"]: r for r in records if isinstance(r, dict) and r.get("prefix")}


def item_lookup(df: pl.DataFrame) -> Dict[str, dict]:
    cols = [c for c in ITEM_METADATA_COLS if c in df.columns]
    return {row["id"]: row for row in df.select(cols).to_dicts()}


def usable_enriched_queries(record: dict) -> List[dict]:
    """
    All of a cluster's `enriched_queries` entries usable as a query anchor: non-empty
    `natural_language_query` and a dict-shaped `attributes` (17 of 3541 entries across the
    whole file carry a list-shaped `attributes` instead -- multi-product broad queries --
    and are skipped rather than crashing `attribute_tags` below).
    """
    entries = record.get("enriched_queries") or []
    return [
        q
        for q in entries
        if isinstance(q, dict)
        and q.get("natural_language_query")
        and isinstance(q.get("attributes"), dict)
    ]


def attribute_tags(query: dict) -> List[str]:
    """Attribute keys this enriched query populates (for T1's reporting breakdown)."""
    attributes = (query or {}).get("attributes") or {}
    return sorted(k for k, v in attributes.items() if v)


# ─────────────────────────────────────────────────────────────────────────────
# Audience (gender) consistency
# ─────────────────────────────────────────────────────────────────────────────

# Audience terms a query can use, mapped to the catalog's gender vocabulary
# (men / women / boys / girls / unisex). "kids" is not a catalog value, so it is
# judged against boys+girls combined.
AUDIENCE_TERMS = {
    "men": ("men", "men's", "mens", "male"),
    "women": ("women", "women's", "womens", "female", "ladies"),
    "boys": ("boys", "boy's", "boys'"),
    "girls": ("girls", "girl's", "girls'"),
}
_KID_TERMS = ("kids", "kid's", "kids'", "children", "child")
AUDIENCE_MIN_SHARE = float(os.environ.get("AUDIENCE_MIN_SHARE", "0.05"))
_WORD_RE = re.compile(r"[a-z']+")


def gender_shares(df: pl.DataFrame, prefix: str, level: int) -> Dict[str, float]:
    """Share of each catalog gender inside `prefix`, e.g. {"men": 0.87, "unisex": 0.10}."""
    col = f"prefix_{level}tok"
    counts = (
        df.filter(pl.col(col) == prefix)["gender"]
        .drop_nulls()
        .str.to_lowercase()
        .value_counts()
    )
    total = int(counts["count"].sum()) if counts.height else 0
    if not total:
        return {}
    return {row["gender"]: row["count"] / total for row in counts.iter_rows(named=True)}


def absent_audiences(shares: Dict[str, float], query: str) -> List[str]:
    """
    Audience labels `query` names that cover under AUDIENCE_MIN_SHARE of the cluster.

    A query carries the audience it is written for, so "sandals for girls" inside an
    87%-men cluster describes items the cluster does not contain and no retrieval system
    can satisfy. An empty list means the query is consistent with its cluster.
    """
    if not shares:
        return []
    allowed = {g for g, s in shares.items() if s >= AUDIENCE_MIN_SHARE}
    # A unisex cluster legitimately answers to both men and women.
    if shares.get("unisex", 0.0) >= AUDIENCE_MIN_SHARE:
        allowed |= {"men", "women"}
    words = set(_WORD_RE.findall(str(query).lower()))
    absent = [
        label
        for label, terms in AUDIENCE_TERMS.items()
        if label not in allowed and words & set(terms)
    ]
    if "kids" not in allowed and words & set(_KID_TERMS):
        if shares.get("boys", 0.0) + shares.get("girls", 0.0) < AUDIENCE_MIN_SHARE:
            absent.append("kids")
    return absent


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pools (Exact / Sibling / Related / Irrelevant)
# ─────────────────────────────────────────────────────────────────────────────

def sibling_leaf_groups(df: pl.DataFrame, leaf_prefix: str) -> Dict[str, List[str]]:
    """Item ids under the same parent as leaf_prefix, grouped by their own leaf prefix."""
    parent = token_prefix(leaf_prefix, 2)
    sib_df = df.filter(
        (pl.col("prefix_2tok") == parent) & (pl.col("prefix_3tok") != leaf_prefix)
    )
    groups: Dict[str, List[str]] = {}
    for leaf, item_id in sib_df.select(["prefix_3tok", "id"]).iter_rows():
        groups.setdefault(leaf, []).append(item_id)
    return groups


def build_cluster_pools(df: pl.DataFrame, leaf_prefix: str) -> Dict[str, Any]:
    """Exact/Sibling/Related/Irrelevant item-id pools for one level-3 leaf cluster."""
    parent = token_prefix(leaf_prefix, 2)
    grandparent = token_prefix(leaf_prefix, 1)

    exact = df.filter(pl.col("prefix_3tok") == leaf_prefix)["id"].to_list()
    groups = sibling_leaf_groups(df, leaf_prefix)
    sibling = [item_id for ids in groups.values() for item_id in ids]
    related = df.filter(
        (pl.col("prefix_1tok") == grandparent) & (pl.col("prefix_2tok") != parent)
    )["id"].to_list()
    irrelevant = df.filter(pl.col("prefix_1tok") != grandparent)["id"].to_list()

    return {
        "exact": exact,
        "sibling": sibling,
        "sibling_groups": groups,
        "related": related,
        "irrelevant": irrelevant,
        "parent": parent,
        "grandparent": grandparent,
    }


def iter_qualifying_clusters(
    df: pl.DataFrame,
    cluster_queries: Dict[str, dict],
    min_cluster_size: int = 5,
    require_related: bool = True,
) -> Iterator[Tuple[str, dict, Dict[str, Any]]]:
    """
    Yield (leaf_prefix, cluster_record, pools) for every level-3 anchor cluster that
    has a usable canonical query and non-empty Sibling (and, unless disabled, Related)
    pools to sample negatives from.
    """
    for prefix, record in cluster_queries.items():
        if record.get("level") != "3":
            continue
        try:
            group_size = int(record.get("group_size", 0))
        except (TypeError, ValueError):
            group_size = 0
        if group_size < min_cluster_size:
            continue
        if not usable_enriched_queries(record):
            continue
        pools = build_cluster_pools(df, prefix)
        if not pools["exact"] or not pools["sibling"]:
            continue
        if require_related and not pools["related"]:
            continue
        yield prefix, record, pools


def sample_with_seed(rng, pool: Sequence[str], k: int, exclude: Optional[Set[str]] = None) -> List[str]:
    """Deduplicated, seeded sampling of up to k items from pool."""
    candidates = [x for x in pool if not exclude or x not in exclude]
    if len(candidates) <= k:
        return list(candidates)
    return rng.sample(candidates, k)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking / classification metrics (shared by all four *_compute_metrics.py scripts)
# ─────────────────────────────────────────────────────────────────────────────

def dcg(grades: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_grades: Sequence[float], k: int) -> float:
    """ranked_grades: relevance grades in the order a model ranked its candidates."""
    top = list(ranked_grades[:k])
    ideal = sorted(ranked_grades, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(top) / ideal_dcg


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of all relevant_ids that appear in the top-k of ranked_ids."""
    if not relevant_ids:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for item_id in top if item_id in relevant_ids)
    return hits / len(relevant_ids)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
