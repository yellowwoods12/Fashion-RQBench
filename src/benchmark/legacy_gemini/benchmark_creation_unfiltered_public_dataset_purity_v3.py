# %% [markdown]
# # Semantic ID Prefix Analysis
#
# This notebook analyzes whether similar products share common prefixes in their semantic IDs.
#
# ## Background:
# Semantic IDs (Rajput et al., 2023; Singh et al., 2023) are hierarchical representations that encode items into a sequence of tokens, replacing embeddings or hash-based IDs. Unlike a conventional item ID (B0040JHNQG) that has no inherent meaning, a semantic ID (`<|sid_0|><|sid_256|><|sid_512|><|sid_768|>`) encodes item information.
#
# **Format:** `<|sid_start|><|sid_104|><|sid_365|><|sid_621|><|sid_768|><|sid_end|>`
# - Start and end tokens are special markers
# - Middle tokens (e.g., `<|sid_104|><|sid_365|><|sid_621|><|sid_768|>`) encode hierarchical item information
# - Similar items naturally share common **prefix tokens**, forming a tree-like structure
# - Each level represents increasingly fine-grained information
#
# ## Goals:
# 1. Extract and analyze semantic ID token sequences (excluding start/end tokens)
# 2. Identify common prefix patterns (e.g., items sharing `<|sid_104|><|sid_365|>`)
# 3. Visualize items that share common prefixes with their images and metadata
# 4. Quantify whether similar products (by category, title, description, image) share common prefix tokens
# 5. Analyze the hierarchical structure encoded in semantic IDs
#
# %%
# Import necessary libraries
import polars as pl
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')

# # Set plotting style
# plt.rcParams['figure.figsize'] = (14, 8)

# Configuration — public Amazon Fashion dataset
DATA_DIR = Path("/home/tripti.shukla/tripti-workspace/semantic-id-llms/data")

print("Libraries imported successfully")

# %%
# Load public dataset (already contains semantic IDs: sid_0, sid_1, sid_2, semantic_id)
print("Loading data files...")

df = pl.read_parquet(
    DATA_DIR / "output" / "public_dataset_items_with_embeddings_1024dim_with_sids_purity_v3.parquet"
)
# Drop the embedding column — not needed for benchmark generation
if "embedding" in df.columns:
    df = df.drop("embedding")
print(f"Loaded {len(df):,} items")
print(f"Columns: {df.columns}")
print(df.select(["id", "semantic_id", "productDisplayName", "masterCategory"]).head())

# %%
# Parse semantic ID tokens
print("Parsing semantic ID token sequences...")

def parse_semantic_id(sid_string):
    """
    Parse semantic ID string and extract token sequence.
    Example (public dataset): '<|sid_195|><|sid_122|><|sid_232|>'
    Returns: ['<|sid_195|>', '<|sid_122|>', '<|sid_232|>']
    """
    import re
    tokens = re.findall(r'<\|sid_\d+\|>', sid_string)
    # Remove legacy start/end tokens if present (not used in the public dataset)
    tokens = [t for t in tokens if t not in ['<|sid_start|>', '<|sid_end|>']]
    return tokens

def get_token_prefix(tokens, length):
    """Get prefix of specified length from token list."""
    return tokens[:length] if len(tokens) >= length else tokens

def tokens_to_string(tokens):
    """Convert token list to string representation."""
    return ''.join(tokens)

# Parse all semantic IDs and extract token sequences
print("\nParsing all semantic IDs...")
df = df.with_columns(
    pl.col("semantic_id").map_elements(
        lambda x: parse_semantic_id(x),
        return_dtype=pl.List(pl.Utf8)
    ).alias("sid_tokens")
)

# Count tokens per semantic ID
df = df.with_columns(
    pl.col("sid_tokens").list.len().alias("num_tokens")
)

max_tokens = df.select("num_tokens").max().item()
prefix_lengths = list(range(1, max_tokens + 1))  # [1, 2, 3] for the 3-token public dataset

for length in prefix_lengths:
    df = df.with_columns(
        pl.col("sid_tokens").map_elements(
            lambda tokens: tokens_to_string(get_token_prefix(tokens, length)),
            return_dtype=pl.Utf8
        ).alias(f"prefix_{length}tok")
    )

# Build the top-prefix list used by candidate_prefixes in the batch config below
analysis_prefix_length = 3
prefix_col = f"prefix_{analysis_prefix_length}tok"
top_prefixes = (
    df.group_by(prefix_col)
    .agg(pl.count().alias("count"))
    .sort("count", descending=True)
    .head(20)
)
top_prefix_values = top_prefixes[prefix_col].to_list()

# %%
def summarize_prefix_semantics(
    df,
    prefix_string,
    categorical_cols=None,
    numeric_cols=None,
    text_cols=None,
    top_k=5,
    min_term_len=4,
    lowercase=True,
    stop_words=None,
    return_summary=False,
    quiet=False,
    ):
    """Generate an exhaustive, human-readable description for a semantic ID prefix."""
    if stop_words is None:
        stop_words = {
            "the","and","for","with","that","have","from","this","your","pack","set","over","made","only","when","into","about","more"
        }
    tokens = re.findall(r'<\|sid_\d+\|>', str(prefix_string))
    prefix_level = len(tokens)
    if prefix_level == 0:
        print("No valid semantic ID tokens detected in the prefix string.")
        return None
    prefix_col = f"prefix_{prefix_level}tok"
    if prefix_col not in df.columns:
        print(f"Column '{prefix_col}' not found. Run the prefix extraction cells first.")
        return None
    prefix_items = df.filter(pl.col(prefix_col) == prefix_string)
    group_size = prefix_items.height
    if group_size == 0:
        print("No items share this prefix.")
        return None
    total_items = len(df)
    coverage = group_size / total_items * 100 if total_items else 0.0
    if categorical_cols is None:
        categorical_cols = [col for col in (
            "masterCategory", "subCategory", "articleType", "brandName",
            "baseColour", "gender", "usage",
        ) if col in df.columns]
    if numeric_cols is None:
        numeric_cols = [col for col in ("price", "discountedPrice") if col in df.columns]
    if text_cols is None:
        text_cols = [col for col in ("productDisplayName", "description") if col in df.columns]
    summary = {
        "prefix": prefix_string,
        "level": prefix_level,
        "group_size": group_size,
        "coverage_pct": coverage,
        "categorical": {},
        "numeric": {},
        "text_terms": {},
    }
    if not quiet:
        print(f"\n{'='*100}")
        print(f"SEMANTIC SUMMARY FOR PREFIX {prefix_string} (level {prefix_level})")
        print(f"Items in group: {group_size:,} ({coverage:.2f}% of dataset)")
        print(f"{'='*100}\n")
    for col in categorical_cols:
        if col not in df.columns:
            if not quiet:
                print(f"Column '{col}' not found. Skipping categorical summary.\n")
            summary["categorical"][col] = {}
            continue
        base_counts = (
            df.filter(pl.col(col).is_not_null())
            .group_by(col)
            .agg(pl.count().alias("global_count"))
        )
        grp_counts = (
            prefix_items.filter(pl.col(col).is_not_null())
            .group_by(col)
            .agg(pl.count().alias("group_count"))
            .sort("group_count", descending=True)
        )
        if not grp_counts.height:
            summary["categorical"][col] = {}
            continue
        merged = grp_counts.join(base_counts, on=col, how="left")
        merged = merged.with_columns(
            (pl.col("group_count") / group_size * 100).alias("group_pct"),
            (pl.col("global_count") / total_items * 100).alias("global_pct"),
            pl.when(pl.col("global_count") > 0)
            .then((pl.col("group_count") / group_size) / (pl.col("global_count") / total_items))
            .otherwise(None)
            .alias("lift"),
        )
        top_values = merged.head(top_k).to_dict(as_series=False)
        summary["categorical"][col] = top_values
        if not quiet:
            print(f"Top {min(top_k, len(grp_counts))} values for {col}:")
            for val, gcount, gpct, globpct, lift in zip(
                top_values.get(col, []),
                top_values.get("group_count", []),
                top_values.get("group_pct", []),
                top_values.get("global_pct", []),
                top_values.get("lift", [])
                ):
                lift_str = f"x{lift:.2f}" if lift and lift > 0 else "n/a"
                print(
                    f"  {str(val)[:400]:<40} | {gcount:6,d} items | group {gpct:5.2f}% | global {globpct:5.2f}% | lift {lift_str}",
                )
            exclusive = merged.filter(pl.col("group_count") == group_size).height > 0
            if exclusive:
                print(f"  -> All items share the same {col}.")
            print()
    for col in numeric_cols:
        if col not in df.columns:
            if not quiet:
                print(f"Column '{col}' not found. Skipping numeric summary.\n")
            summary["numeric"][col] = {}
            continue
        numeric_expr = pl.col(col).cast(pl.Float64, strict=False)
        stats_df = prefix_items.select(
            numeric_expr.mean().alias("mean"),
            numeric_expr.std().alias("std"),
            numeric_expr.median().alias("median"),
            numeric_expr.quantile(0.25).alias("q25"),
            numeric_expr.quantile(0.75).alias("q75"),
            numeric_expr.min().alias("min"),
            numeric_expr.max().alias("max"),
        )
        if not stats_df.height:
            summary["numeric"][col] = {}
            continue
        stats_row = stats_df.row(0, named=True)
        summary["numeric"][col] = stats_row
        if not quiet:
            def fmt_num(value):
                return f"{value:.2f}" if isinstance(value, (int, float)) and value is not None else "n/a"
            print(f"{col} stats:")
            print(
                f"  mean={fmt_num(stats_row['mean'])}, median={fmt_num(stats_row['median'])}, "
                f"min={fmt_num(stats_row['min'])}, max={fmt_num(stats_row['max'])}, "
                f"IQR=({fmt_num(stats_row['q25'])}, {fmt_num(stats_row['q75'])})",
            )
    def top_terms(series):
        tokens = []
        for value in series:
            if not value:
                continue
            text = value.lower() if lowercase else value
            words = re.findall(r"[a-zA-Z0-9]+", text)
            tokens.extend(
                [w for w in words if len(w) >= min_term_len and w not in stop_words]
            )
        return Counter(tokens).most_common(top_k)
    for col in text_cols:
        if col not in df.columns:
            if not quiet:
                print(f"Column '{col}' not found. Skipping text summary.\n")
            summary["text_terms"][col] = []
            continue
        values = prefix_items.select(col).to_series().to_list()
        terms = top_terms(values)
        summary["text_terms"][col] = terms
        if not quiet and terms:
            print(f"Most informative terms for {col}:")
            for term, count in terms:
                pct = count / group_size * 100
                print(f"  {term:<20} {count:5d} occurrences (~{pct:4.1f}% of items)")
            print()
    if return_summary:
        return summary
    return None

# %% [markdown]
# ## 6. Automated Workflow: Prefix Summaries → Gemini Prompts
# This section operationalizes the earlier plan:
#
# 1. Build compact, information-dense summaries for any prefix (coverage, entropy, exclusivity, sibling contrast).
# 2. Convert each summary into a structured blueprint that can seed LLM prompts.
# 3. Auto-generate exhaustive (what to include) and exclusive (what to filter) natural-language queries for Gemini along with a ready-to-send payload template.
# 4. Keep everything modular so we can slot in alternative stats or downstream endpoints later.
#
# %%
# Workflow helpers: summary → blueprint → prompt → Gemini payload
import json
import math
import os
import re
from itertools import combinations
# Common attribute mining utilities
from functools import wraps
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from InstructorEmbedding import INSTRUCTOR
except ImportError:
    INSTRUCTOR = None

SEMANTIC_SIM_THRESHOLD = float(os.environ.get("SEMANTIC_SIM_THRESHOLD", 0.82))
SEMANTIC_EMBED_MODEL = os.environ.get(
    "SEMANTIC_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
SEMANTIC_EMBED_BACKEND = os.environ.get(
    "SEMANTIC_EMBED_BACKEND",
    "sentence-transformers",
)

_SEMANTIC_ENCODER = None
_SEMANTIC_ENCODER_BACKEND = None
_SEMANTIC_ENCODER_ERROR = False


def _load_semantic_encoder():
    """Lazily load the contextual embedding encoder (Sentence-BERT or Instructor)."""
    global _SEMANTIC_ENCODER, _SEMANTIC_ENCODER_BACKEND, _SEMANTIC_ENCODER_ERROR
    if _SEMANTIC_ENCODER is not None:
        return _SEMANTIC_ENCODER_BACKEND, _SEMANTIC_ENCODER

    backend_choice = (SEMANTIC_EMBED_BACKEND or "").lower()
    model_name = SEMANTIC_EMBED_MODEL
    wants_instructor = backend_choice.startswith("instructor") or "instructor" in model_name.lower()

    if wants_instructor:
        if INSTRUCTOR is None:
            if not _SEMANTIC_ENCODER_ERROR:
                print("InstructorEmbedding not installed; semantic collapse disabled.")
                _SEMANTIC_ENCODER_ERROR = True
            return None, None
        _SEMANTIC_ENCODER = INSTRUCTOR(model_name)
        _SEMANTIC_ENCODER_BACKEND = "instructor"
        return _SEMANTIC_ENCODER_BACKEND, _SEMANTIC_ENCODER

    if SentenceTransformer is None:
        if not _SEMANTIC_ENCODER_ERROR:
            print("sentence-transformers not installed; semantic collapse disabled.")
            _SEMANTIC_ENCODER_ERROR = True
        return None, None

    _SEMANTIC_ENCODER = SentenceTransformer(model_name)
    _SEMANTIC_ENCODER_BACKEND = "sentence-transformers"
    return _SEMANTIC_ENCODER_BACKEND, _SEMANTIC_ENCODER


def _encode_texts(texts):
    """Return normalized embeddings for the provided phrases."""
    backend, encoder = _load_semantic_encoder()
    if encoder is None or not texts:
        return None

    if backend == "instructor":
        payload = [["Represent the attribute phrase for deduplication", text] for text in texts]
        embeddings = encoder.encode(payload)
    else:
        embeddings = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if backend == "instructor":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    return embeddings


def _collapse_semantic_entries(
    entries,
    text_key="value",
    sum_keys=None,
    threshold=None,
    derived_fn=None,
    sort_key=None,
):
    """
    Merge entries whose phrases are paraphrases (cosine similarity >= threshold).

    Args:
        entries (list[dict]): Records with a textual key to compare.
        text_key (str): Field containing the phrase/value.
        sum_keys (list[str]): Numeric fields to sum when clusters merge.
        threshold (float): Cosine similarity threshold.
        derived_fn (callable): Optional callback to recompute derived metrics.
        sort_key (str): Field to sort on after collapsing.
    """
    if not entries or len(entries) <= 1 or SEMANTIC_SIM_THRESHOLD <= 0:
        return entries

    normalized_texts = [str(entry.get(text_key, "")).strip().lower() for entry in entries]
    embeddings = _encode_texts(normalized_texts)
    if embeddings is None:
        return entries

    threshold = threshold or SEMANTIC_SIM_THRESHOLD
    n = len(entries)
    cluster_labels = np.full(n, -1, dtype=int)
    current_cluster = 0

    for idx in range(n):
        if cluster_labels[idx] != -1:
            continue
        cluster_labels[idx] = current_cluster
        sims = embeddings @ embeddings[idx]
        for jdx in range(idx + 1, n):
            if cluster_labels[jdx] != -1:
                continue
            if sims[jdx] >= threshold:
                cluster_labels[jdx] = current_cluster
        current_cluster += 1

    collapsed = []
    for cid in range(current_cluster):
        member_indices = [i for i, label in enumerate(cluster_labels) if label == cid]
        cluster_entries = [entries[i] for i in member_indices]
        merged = cluster_entries[0].copy()
        alias_values = {entry.get(text_key) for entry in cluster_entries if entry.get(text_key)}
        if len(alias_values) > 1:
            merged["aliases"] = sorted(alias_values)
        if sum_keys:
            for key in sum_keys:
                merged[key] = float(
                    sum(entry.get(key, 0.0) or 0.0 for entry in cluster_entries)
                )
        if derived_fn:
            derived_fn(merged, cluster_entries)
        collapsed.append(merged)

    if sort_key:
        collapsed.sort(key=lambda item: item.get(sort_key) or 0.0, reverse=True)
    elif sum_keys:
        collapsed.sort(key=lambda item: item.get(sum_keys[0]) or 0.0, reverse=True)
    return collapsed


def _entropy_from_counts(counts):
    """Shannon entropy (bits) from a list of counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p, 2) for p in probs)


# Ordered from most specific to least specific. The first column whose dominant
# value covers ≥80% of the cluster becomes the query-generation axis.
# gender is excluded — it is too coarse to be a useful query discriminator on its own.
_PURITY_AXIS_PRIORITY = [
    "articleType",
    "subCategory",
    "brandName",
    "baseColour",
    "usage",
    "masterCategory",
]


def _best_purity_axis(cat_profiles, threshold=80.0):
    """Return the most-specific attribute column whose dominant value covers ≥threshold%.

    Checks columns in _PURITY_AXIS_PRIORITY order (most specific first) so that
    e.g. articleType is preferred over masterCategory when both qualify.
    Returns (column_name, dominant_pct) or (None, 0.0) if none qualify.
    """
    for col in _PURITY_AXIS_PRIORITY:
        profile = cat_profiles.get(col)
        if not profile:
            continue
        top_values = profile.get("top_values") or []
        if not top_values:
            continue
        dom_pct = top_values[0].get("pct") or 0.0
        if dom_pct >= threshold:
            return col, dom_pct
    return None, 0.0


def _format_top_values(count_df, value_col, top_k, total):
    rows = []
    if not count_df.height:
        return rows
    for row in count_df.head(top_k).iter_rows(named=True):
        count = int(row["count"])
        pct = count / total * 100 if total else 0.0
        rows.append({
            "value": row[value_col],
            "count": count,
            "pct": pct,
        })
    return rows


def _categorical_profile(prefix_items, col, top_k):
    if col not in prefix_items.columns:
        return None
    counts = (
        prefix_items
        .filter(pl.col(col).is_not_null())
        .group_by(col)
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    if not counts.height:
        return None
    total = counts["count"].sum()
    rows = _format_top_values(counts, col, top_k, total)
    rows = _collapse_semantic_entries(
        rows,
        text_key="value",
        sum_keys=["count", "pct"],
        sort_key="count",
    )
    return {
        "top_values": rows,
        "entropy_bits": _entropy_from_counts(counts["count"].to_list()),
        "exclusive": counts["count"].max() == total,
        "unique_values": counts.height,
    }


def _sibling_context(df, prefix_col, tokens):
    if len(tokens) == 0:
        return {}
    if len(tokens) == 1:
        # Level-1 prefixes have no parent; their "siblings" are all other level-1
        # prefixes in the corpus (children of the virtual root).
        if prefix_col not in df.columns:
            return {}
        siblings = (
            df.group_by(prefix_col)
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        if not siblings.height:
            return {}
        total = siblings["count"].sum()
        return {
            "parent_prefix": "<root>",
            "top_siblings": _format_top_values(siblings, prefix_col, 5, total),
            "total_siblings": siblings.height,
        }
    parent_tokens = tokens[:-1]
    parent_prefix = ''.join(parent_tokens)
    parent_col = f"prefix_{len(parent_tokens)}tok"
    if parent_col not in df.columns:
        return {}
    siblings = (
        df.filter(pl.col(parent_col) == parent_prefix)
        .group_by(prefix_col)
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    if not siblings.height:
        return {}
    total = siblings["count"].sum()
    return {
        "parent_prefix": parent_prefix or "<root>",
        "top_siblings": _format_top_values(siblings, prefix_col, 5, total),
        "total_siblings": siblings.height,
    }


def derive_attribute_signatures(
    df,
    prefix_items,
    prefix_string,
    candidate_cols=None,
    max_combo=3,
    min_group_pct=0.50,
    min_precision=0.85,
    top_signatures=8,
):
    """Identify attribute combinations that cover the prefix items exhaustively/exclusively."""
    if candidate_cols is None:
        candidate_cols = [
            col
            for col in (
                "masterCategory",
                "subCategory",
                "articleType",
                "brandName",
                "baseColour",
                "gender",
                "usage",
                "price",
                "productDisplayName",
            )
            if col in df.columns
        ]
    total_items = len(df)
    group_size = prefix_items.height
    signatures = []
    if group_size == 0 or not candidate_cols:
        return signatures
    for depth in range(1, max_combo + 1):
        for cols in combinations(candidate_cols, depth):
            if any(col not in prefix_items.columns for col in cols):
                continue
            subset_prefix = prefix_items.drop_nulls(list(cols))
            if subset_prefix.height == 0:
                continue
            prefix_counts = (
                subset_prefix
                .group_by(list(cols))
                .agg(pl.count().alias("group_count"))
                .sort("group_count", descending=True)
                .with_columns((pl.col("group_count") / group_size * 100).alias("group_pct"))
            )
            prefix_counts = prefix_counts.filter(pl.col("group_pct") >= min_group_pct * 100)
            if not prefix_counts.height:
                continue
            global_counts = (
                df.drop_nulls(list(cols))
                .group_by(list(cols))
                .agg(pl.count().alias("global_count"))
            )
            merged = prefix_counts.join(global_counts, on=list(cols), how="left")
            merged = merged.with_columns(
                pl.col("global_count").fill_null(0),
                pl.when(pl.col("global_count") > 0)
                .then(pl.col("group_count") / pl.col("global_count"))
                .otherwise(0.0)
                .alias("precision"),
                (pl.col("global_count") == pl.col("group_count")).alias("exclusive"),
                (pl.col("global_count") / total_items * 100).alias("global_pct"),
            )
            merged = merged.filter(pl.col("precision") >= min_precision)
            if not merged.height:
                continue
            for row in merged.iter_rows(named=True):
                value_map = {col: row[col] for col in cols}
                signatures.append(
                    {
                        "prefix": prefix_string,
                        "columns": list(cols),
                        "values": value_map,
                        "group_count": int(row["group_count"]),
                        "group_pct": float(row["group_pct"]),
                        "global_count": int(row["global_count"]),
                        "global_pct": float(row["global_pct"]),
                        "precision": float(row["precision"]),
                        "exclusive": bool(row["exclusive"]),
                    }
                )
    if not signatures:
        return []
    signatures.sort(key=lambda s: (s["exclusive"], s["precision"], s["group_pct"]), reverse=True)
    return signatures[:top_signatures]


def build_prefix_workflow_record(
    df,
    prefix_string,
    categorical_cols=None,
    numeric_cols=None,
    text_cols=None,
    top_k=20,
    min_group_size=10,
    signature_cols=None,
    max_signature_depth=3,
    # --- legacy single-threshold params (kept for backward compat) ---
    min_signature_pct=0.25,
    min_signature_precision=0.85,
    top_signature_results=8,
    # --- coverage signatures: enumerate every sub-type for exhaustiveness ---
    coverage_min_pct=0.05,
    coverage_min_precision=0.50,
    coverage_top=20,
    # --- discriminating signatures: tight anchors for exclusivity ---
    discriminating_min_pct=0.10,
    discriminating_min_precision=0.90,
    discriminating_top=8,
):
    """Package all stats required to drive Gemini prompts for a prefix.

    Two separate signature sets are produced:

    ``coverage_signatures``
        Loose thresholds (min_group_pct=5%, min_precision=50%).  These map to
        every sub-type present at ≥5% share so downstream query generation can
        guarantee exhaustive coverage.

    ``discriminating_signatures``
        Tight thresholds (min_group_pct=10%, min_precision=90%).  These are
        attribute combos that strongly identify *this* prefix and minimise
        cross-group leakage, driving exclusive queries.

    ``attribute_signatures`` is an alias for ``discriminating_signatures`` so
    callers that rely on the old key continue to work.
    """
    tokens = re.findall(r'<\|sid_\d+\|>', str(prefix_string))
    prefix_level = len(tokens)
    if prefix_level == 0:
        return None
    prefix_col = f"prefix_{prefix_level}tok"
    if prefix_col not in df.columns:
        raise ValueError(f"Column {prefix_col} missing; generate prefix columns first.")
    prefix_items = df.filter(pl.col(prefix_col) == prefix_string)
    group_size = prefix_items.height
    if group_size < min_group_size:
        return None
    summary = summarize_prefix_semantics(
        df,
        prefix_string,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        text_cols=text_cols,
        top_k=top_k,
        return_summary=True,
        quiet=True,
    )
    if summary is None:
        return None
    if categorical_cols is None:
        categorical_cols = [col for col in (
            "masterCategory", "subCategory", "articleType", "brandName",
            "baseColour", "gender", "usage",
        ) if col in df.columns]
    if numeric_cols is None:
        numeric_cols = [col for col in ("price", "discountedPrice") if col in df.columns]
    if text_cols is None:
        text_cols = [col for col in ("productDisplayName", "description") if col in df.columns]
    if signature_cols is None:
        signature_cols = categorical_cols
    cat_profiles = {}
    exclusive_attrs = []
    for col in categorical_cols:
        profile = _categorical_profile(prefix_items, col, top_k)
        cat_profiles[col] = profile
        if profile and profile.get("exclusive"):
            exclusive_attrs.append(col)
    sibling_info = _sibling_context(df, prefix_col, tokens)

    # Coverage signatures — loose, depth-2 sufficient, enumerates all sub-types
    coverage_signatures = derive_attribute_signatures(
        df,
        prefix_items,
        prefix_string,
        candidate_cols=signature_cols,
        max_combo=2,
        min_group_pct=coverage_min_pct,
        min_precision=coverage_min_precision,
        top_signatures=coverage_top,
    )

    # Discriminating signatures — tight, depth-3, uniquely identifies this group
    discriminating_signatures = derive_attribute_signatures(
        df,
        prefix_items,
        prefix_string,
        candidate_cols=signature_cols,
        max_combo=max_signature_depth,
        min_group_pct=discriminating_min_pct,
        min_precision=discriminating_min_precision,
        top_signatures=discriminating_top,
    )

    record = {
        "prefix": prefix_string,
        "level": prefix_level,
        "group_size": group_size,
        "coverage_pct": summary["coverage_pct"],
        "categorical_profiles": cat_profiles,
        "numeric_profiles": summary["numeric"],
        "text_terms": summary["text_terms"],
        "exclusive_attributes": exclusive_attrs,
        "sibling_info": sibling_info,
        # Both signature sets stored explicitly
        "coverage_signatures": coverage_signatures,
        "discriminating_signatures": discriminating_signatures,
        # Backward-compat alias: callers using attribute_signatures get the discriminating set
        "attribute_signatures": discriminating_signatures,
        "raw_summary": summary,
    }
    return record


def build_prompt_blueprint(record):
    """Convert the workflow record into a compact briefing for Gemini."""
    lines = [
        f"Prefix: {record['prefix']} (level {record['level']})",
        f"Items: {record['group_size']:,} ({record['coverage_pct']:.3f}% coverage)",
    ]

    # Show full category distribution (not just top-3) so Gemini sees minority types
    cat_profile = (
        (record.get("categorical_profiles") or {}).get("masterCategory")
        or (record.get("categorical_profiles") or {}).get("articleType")
    )
    if cat_profile and cat_profile.get("top_values"):
        lines.append(f"Category distribution ({len(cat_profile['top_values'])} types):")
        for entry in cat_profile["top_values"]:
            marker = " ← minority" if entry["pct"] < 5.0 else ""
            lines.append(f"  - {entry['value']}: {entry['pct']:.1f}%{marker}")

    for attr in record.get("exclusive_attributes") or []:
        lines.append(f"Exclusive signal: 100% of items share {attr}.")

    # Discriminating signatures — high precision, use as exclusivity anchors
    disc_sigs = record.get("discriminating_signatures") or record.get("attribute_signatures") or []
    if disc_sigs:
        lines.append("Discriminating attribute combos (exclusivity anchors, precision ≥90%):")
        for sig in disc_sigs[:4]:
            combo = ", ".join(f"{k}={v}" for k, v in sig["values"].items())
            qualifier = "exclusive" if sig["exclusive"] else f"{sig['precision'] * 100:.1f}% precision"
            lines.append(f"  - {combo} | covers {sig['group_pct']:.1f}% of group ({qualifier})")

    # Coverage signatures — low precision, enumerate all sub-types
    cov_sigs = record.get("coverage_signatures") or []
    if cov_sigs:
        lines.append("Coverage signatures (exhaustiveness map, include every sub-type):")
        for sig in cov_sigs[:6]:
            combo = ", ".join(f"{k}={v}" for k, v in sig["values"].items())
            lines.append(f"  - {combo} | {sig['group_pct']:.1f}% of group")

    sibling_info = record.get("sibling_info") or {}
    if sibling_info.get("top_siblings"):
        lines.append(f"Sibling scope under {sibling_info['parent_prefix']}:")
        for sib in sibling_info["top_siblings"][:3]:
            marker = " (target)" if sib["value"] == record["prefix"] else ""
            lines.append(f"  - {sib['value']} :: {sib['pct']:.1f}%{marker}")

    key_terms = [term for term, _ in record.get("text_terms", {}).get("name", [])][:5]
    if key_terms:
        lines.append("Representative title terms: " + ", ".join(key_terms))
    return "\n".join(lines)


def _diff_category_profiles(child_profile, reference_profile):
    """Return delta between child and reference categorical percentages."""
    if not child_profile:
        return []
    ref_lookup = {}
    if reference_profile:
        for entry in reference_profile.get("top_values", []):
            ref_lookup[entry["value"]] = entry["pct"]
    deltas = []
    for entry in child_profile.get("top_values", []):
        ref_pct = ref_lookup.get(entry["value"], 0.0)
        deltas.append({
            "value": entry["value"],
            "child_pct": entry["pct"],
            "reference_pct": ref_pct,
            "delta_pct": entry["pct"] - ref_pct,
        })

    def _update_delta(merged, _cluster):
        merged["delta_pct"] = (
            (merged.get("child_pct") or 0.0)
            - (merged.get("reference_pct") or 0.0)
        )

    return _collapse_semantic_entries(
        deltas,
        text_key="value",
        sum_keys=["child_pct", "reference_pct"],
        derived_fn=_update_delta,
        sort_key="delta_pct",
    )


def _diff_numeric_profiles(child_stats, reference_stats):
    if not child_stats:
        return {}
    reference_stats = reference_stats or {}
    fields = ["mean", "median", "min", "max", "q25", "q75"]
    diff = {}
    for field in fields:
        child_val = child_stats.get(field)
        ref_val = reference_stats.get(field)
        delta = None
        if isinstance(child_val, (int, float)) and isinstance(ref_val, (int, float)):
            delta = child_val - ref_val
        diff[field] = {
            "child": child_val,
            "reference": ref_val,
            "delta": delta,
        }
    return diff


def build_relational_context(
    df,
    record,
    categorical_cols=None,
    numeric_cols=None,
    text_cols=None,
    top_k=5,
    sibling_limit=3,
    record_cache=None,
):
    """Gather parent and sibling comparisons for a target prefix record.

    *record_cache* is an optional dict keyed by prefix string.  When provided,
    built records are stored in it and re-used instead of being recomputed.
    Pass the same dict across all calls in a batch to avoid redundant work —
    especially for the shared level-1 parent of many level-2 siblings.
    """
    if record_cache is None:
        record_cache = {}

    def _get_record(prefix, **kwargs):
        if prefix not in record_cache:
            result = build_prefix_workflow_record(df, prefix, **kwargs)
            if result:
                record_cache[prefix] = result
            return result
        return record_cache[prefix]

    tokens = re.findall(r'<\|sid_\d+\|>', record["prefix"])
    parent_record = None
    if len(tokens) > 1:
        parent_prefix = ''.join(tokens[:-1])
        parent_record = _get_record(
            parent_prefix,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            text_cols=text_cols,
            top_k=top_k,
            min_group_size=3,
        )
    sibling_context = []
    sibling_info = record.get("sibling_info", {})
    for sibling_entry in (sibling_info.get("top_siblings") or [])[:sibling_limit]:
        sibling_prefix = sibling_entry["value"]
        if sibling_prefix == record["prefix"]:
            continue
        sibling_record = _get_record(
            sibling_prefix,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            text_cols=text_cols,
            top_k=top_k,
            min_group_size=3,
        )
        if sibling_record:
            sibling_context.append({
                "summary": sibling_record,
                "share_pct": sibling_entry["pct"],
            })
    category_deltas = {}
    numeric_deltas = {}
    if parent_record:
        for col, child_profile in record["categorical_profiles"].items():
            category_deltas[col] = _diff_category_profiles(
                child_profile,
                parent_record["categorical_profiles"].get(col) if parent_record else None,
            )
        for col, child_stats in record["numeric_profiles"].items():
            numeric_deltas[col] = _diff_numeric_profiles(
                child_stats,
                parent_record["numeric_profiles"].get(col) if parent_record else None,
            )
    sibling_deltas = []
    for sibling in sibling_context:
        sibling_record = sibling["summary"]
        sibling_diff = {
            "categorical": {},
            "numeric": {},
        }
        for col, child_profile in record["categorical_profiles"].items():
            sibling_diff["categorical"][col] = _diff_category_profiles(
                child_profile,
                sibling_record["categorical_profiles"].get(col),
            )
        for col, child_stats in record["numeric_profiles"].items():
            sibling_diff["numeric"][col] = _diff_numeric_profiles(
                child_stats,
                sibling_record["numeric_profiles"].get(col),
            )
        # Materialise dominant categories so _build_exclusivity_constraints can surface
        # them without needing to traverse the full sibling record later.
        _sib_cat_profiles = sibling_record.get("categorical_profiles") or {}
        sib_cat_profile = _sib_cat_profiles.get("masterCategory") or _sib_cat_profiles.get("articleType") or {}
        sib_top_categories = [
            f"{e['value']} ({e['pct']:.0f}%)"
            for e in (sib_cat_profile.get("top_values") or [])[:4]
            if e.get("value") and e.get("pct") is not None
        ]
        sibling_deltas.append({
            "prefix": sibling_record["prefix"],
            "level": sibling_record["level"],
            "group_size": sibling_record["group_size"],
            "share_pct": sibling["share_pct"],
            "diffs": sibling_diff,
            "attribute_signatures": sibling_record.get("attribute_signatures", []),
            "key_terms": sibling_record["text_terms"].get("name", []),
            "top_categories": sib_top_categories,
        })
    return {
        "parent_record": parent_record,
        "category_deltas_vs_parent": category_deltas,
        "numeric_deltas_vs_parent": numeric_deltas,
        "sibling_records": sibling_context,
        "sibling_deltas": sibling_deltas,
    }


def _build_full_coverage_context(df, prefix_string, record, top_k_full=None):
    """
    Return a dict that maps each categorical column to its COMPLETE value distribution
    (not just top-k) so the Gemini prompt can require exhaustive query coverage.

    Args:
        df: main Polars dataframe
        prefix_string: target prefix, e.g. '<|sid_104|><|sid_365|>'
        record: workflow record for the prefix
        top_k_full: cap how many distinct values to return per column (None = all)
    """
    tokens = re.findall(r'<\|sid_\d+\|>', str(prefix_string))
    prefix_col = f"prefix_{len(tokens)}tok"
    if prefix_col not in df.columns:
        return {}
    prefix_items = df.filter(pl.col(prefix_col) == prefix_string)
    group_size = prefix_items.height
    if group_size == 0:
        return {}

    coverage_ctx = {}
    candidate_cols = [c for c in (
        "masterCategory", "subCategory", "articleType", "brandName",
        "baseColour", "gender", "usage",
    ) if c in df.columns]
    for col in candidate_cols:
        counts = (
            prefix_items
            .filter(pl.col(col).is_not_null())
            .group_by(col)
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        if top_k_full is not None:
            counts = counts.head(top_k_full)
        rows = []
        for row in counts.iter_rows(named=True):
            pct = row["count"] / group_size * 100
            rows.append({"value": row[col], "count": int(row["count"]), "pct": round(pct, 1)})
        if rows:
            coverage_ctx[col] = {
                "total_distinct": counts.height,
                "values": rows,
            }
    return coverage_ctx


def _build_exclusivity_constraints(record, relational_context):
    """
    Return a human-readable string listing sibling prefixes and the attribute-value
    combinations that discriminate them from the target prefix.

    For each sibling we report:
    - The sibling's dominant categories/brands
    - The attributes where the target and sibling diverge most (delta_pct > 10pp)
    These facts should be passed to Gemini so it can produce queries that cannot
    accidentally retrieve sibling-group items.
    """
    sibling_deltas = (relational_context or {}).get("sibling_deltas") or []
    if not sibling_deltas:
        return ""

    lines = [
        "EXCLUSIVITY CONSTRAINTS — your queries must NOT retrieve items from these sibling prefixes:",
    ]
    for sib in sibling_deltas:
        sib_prefix = sib.get("prefix", "?")
        sib_size = sib.get("group_size", "?")

        # Dominant categories materialised directly on the sibling_delta entry
        sib_types = sib.get("top_categories") or []

        # Attributes where target DIFFERS from sibling strongly (|delta| >= 20pp).
        # 10pp was too loose — a 10pp spread still means heavy overlap; 20pp
        # represents a genuinely dominant shift that anchors an exclusive query.
        categorical_diffs = (sib.get("diffs") or {}).get("categorical") or {}
        target_exclusive = []   # high in target, low in sibling
        sibling_exclusive = []  # high in sibling, low in target
        for col, deltas in categorical_diffs.items():
            for d in (deltas or []):
                dpct = d.get("delta_pct")
                if dpct is None:
                    continue
                val = d.get("value", "?")
                if dpct >= 20:
                    target_exclusive.append(f"{col}={val} (+{dpct:.0f}pp in target)")
                elif dpct <= -20:
                    sibling_exclusive.append(f"{col}={val} (+{abs(dpct):.0f}pp in sibling)")

        line = f"  Sibling {sib_prefix} ({sib_size} items)"
        if sib_types:
            line += f"\n    Contains: {', '.join(sib_types)}"
        if sibling_exclusive:
            line += f"\n    Sibling-only attributes (avoid these in target queries): {'; '.join(sibling_exclusive[:3])}"
        if target_exclusive:
            line += f"\n    Target-only attributes (safe anchors for exclusivity): {'; '.join(target_exclusive[:3])}"
        lines.append(line)

    lines.append(
        "\nFor EVERY query you write, verify it would NOT match the dominant attributes of the sibling prefixes above."
    )
    return "\n".join(lines)


def compose_gemini_prompt(record, relational_context, max_terms=6, df=None):
    """
    Create a text prompt describing the prefix, parent deltas, and sibling contrasts.

    Args:
        record: workflow record for the target prefix.
        relational_context: parent + sibling comparison dicts.
        max_terms: max title terms to include.
        df: optional Polars dataframe; when provided, the full attribute distribution
            (not just top-k) is included so Gemini can write exhaustive queries.
    """
    # --- Full attribute distribution (for exhaustiveness) ---
    full_coverage = {}
    if df is not None:
        full_coverage = _build_full_coverage_context(df, record["prefix"], record, top_k_full=None)

    # Build a human-readable coverage requirement for the prompt.
    # Use the most specific attribute whose dominant value covers ≥80% of the cluster
    # so Gemini's coverage_map keys are meaningful (e.g. "Tshirts", not "Apparel").
    # Falls back to masterCategory when no attribute is that pure.
    _fc_profiles = {
        col: {"top_values": (full_coverage.get(col) or {}).get("values") or []}
        for col in _PURITY_AXIS_PRIORITY
    }
    primary_col, _ = _best_purity_axis(_fc_profiles)
    if primary_col is None:
        primary_col = "masterCategory"

    coverage_lines = []
    cat_full = full_coverage.get(primary_col) or {}
    if cat_full.get("values"):
        total_items = record.get("group_size", 1)
        major = [e for e in cat_full["values"] if e["pct"] >= 5.0]
        minor = [e for e in cat_full["values"] if e["pct"] < 5.0]
        coverage_lines.append(
            f"EXHAUSTIVE COVERAGE REQUIREMENT — this prefix contains {cat_full['total_distinct']} distinct product types:"
        )
        for e in major:
            coverage_lines.append(f"  • {e['value']}: {e['pct']:.1f}% ({e['count']} items)  ← MUST be reachable by at least one query")
        if minor:
            minor_names = ", ".join(e["value"] for e in minor[:5])
            if len(minor) > 5:
                minor_names += f", … ({len(minor) - 5} more)"
            coverage_lines.append(f"  • Minority types (<5% each): {minor_names}  ← cover collectively with a broad query")

    # --- Exclusivity constraints (for exclusivity) ---
    exclusivity_text = _build_exclusivity_constraints(record, relational_context)

    context_doc = {
        "target_prefix": {
            "metadata": record,
            "blueprint": build_prompt_blueprint(record),
            # Discriminating sigs: high-precision anchors for exclusive queries
            "discriminating_signatures": record.get("discriminating_signatures") or record.get("attribute_signatures", []),
            # Coverage sigs: enumerate every sub-type for exhaustive queries
            "coverage_signatures": record.get("coverage_signatures", []),
            "common_attributes": record.get("common_attributes"),
            "full_attribute_distribution": full_coverage,
        },
        "parent": {
            "record": relational_context.get("parent_record"),
            "category_deltas": relational_context.get("category_deltas_vs_parent"),
            "numeric_deltas": relational_context.get("numeric_deltas_vs_parent"),
        },
        "sibling_comparisons": relational_context.get("sibling_deltas"),
    }

    coverage_block = ("\n\n" + "\n".join(coverage_lines)) if coverage_lines else ""
    exclusivity_block = ("\n\n" + exclusivity_text) if exclusivity_text else ""

    instructions = (
        "You are an expert analyst of hierarchical fashion semantic ID clusters and a specialist in e-commerce search behavior.\n\n"
        "Your task is to understand what a given semantic prefix represents by analyzing:\n"
        "- High-precision attribute signatures (combinations of attributes strongly associated with the prefix)\n"
        "- Differences from its parent cluster (what makes it more specific)\n"
        "- Differences from sibling clusters (what makes it unique and non-overlapping)\n\n"
        "Based on this, infer the underlying product concept that this prefix encodes.\n\n"
        "Then generate natural-language user queries that a real shopper would type into a fashion recommendation system "
        "(e.g., Myntra, Amazon, Flipkart). These queries must:\n\n"
        "1. Sound natural and human-like (NOT attribute lists like 'color=black AND material=leather').\n"
        "2. Reflect real shopping intent (style, occasion, product type, etc.).\n"
        "3. Be phrased as proper, complete sentences.\n"
        "4. Be EXHAUSTIVE: the set of queries must collectively retrieve EVERY item in the prefix group.\n"
        "   - Generate at least one distinct query for each major product type (>5% share) listed in the coverage section.\n"
        "   - Add a broad catch-all query to cover minor types (<5% share) that are not individually addressed.\n"
        "5. Be EXCLUSIVE: no query should retrieve items from sibling prefixes or from outside this cluster.\n"
        "   - For every query, verify it cannot be satisfied by the dominant attributes of each sibling listed in the exclusivity section.\n"
        "   - Use target-only discriminating attributes as anchors wherever possible.\n"
        "6. Prefer commonly used fashion terminology over internal attribute names.\n"
        "7. Ground each query in the dominant phrases from description/productDisplayName to make the intent explicit.\n"
        "8. Never use a single attribute value unless 100% of items share it; describe multi-value ranges collectively.\n\n"
        "Additional guidelines:\n"
        "- Use attribute signatures to ground the meaning.\n"
        "- Use parent deltas to explain what makes this cluster more specific than its parent.\n"
        "- Use sibling contrasts and the exclusivity constraints below to sharpen boundaries.\n\n"
        "Return ONLY a valid JSON object (no markdown fences, no extra text) with these keys:\n"
        "{\n"
        '  "insights": "<concise explanation of what this prefix represents>",\n'
        '  "user_queries": ["<query 1>", "<query 2>", ...],   // 6–12 queries, covering all major AND minor product types\n'
        '  "delta_vs_parent": "<key ways this prefix is more specific than its parent>",\n'
        '  "delta_vs_siblings": "<key distinguishing factors vs sibling prefixes>",\n'
        '  "validation_checks": ["<rule 1>", ...],   // one rule per major product type confirming coverage; one rule per sibling confirming exclusivity\n'
        '  "coverage_map": {"<product_type>": "<which query covers it>", ...}  // explicit mapping from each major product type to the query index that covers it\n'
        "}\n"
    )

    prompt = (
        instructions
        + coverage_block
        + exclusivity_block
        + "\n\nDATA CONTEXT:\n"
        + json.dumps(context_doc, indent=2)
    )
    return prompt

# %%
def craft_gemini_queries(record, num_queries=4, relational_context=None):
    """Generate shopper-style queries as full sentences while enforcing exclusivity.

    Uses two separate signature sets:
    - ``discriminating_signatures`` (high precision ≥90%) anchor the strict /
      contrast queries so they don't bleed into sibling groups.
    - ``coverage_signatures`` (low precision ≥50%) enumerate every sub-type so
      the per-subcategory queries guarantee full group coverage.
    Falls back to the legacy ``attribute_signatures`` key when either dedicated
    set is absent (e.g. records built with the old schema).
    """
    # Discriminating sigs → used for exclusivity anchors
    disc_sigs = (
        record.get("discriminating_signatures")
        or record.get("attribute_signatures")
        or []
    )
    # Coverage sigs → used to enumerate sub-types for exhaustiveness
    cov_sigs = (
        record.get("coverage_signatures")
        or record.get("attribute_signatures")
        or []
    )
    # Need at least one signature to build any query
    if not disc_sigs and not cov_sigs:
        return []
    # Use whichever non-empty set is available as the primary driver
    attr_sigs = disc_sigs or cov_sigs

    text_terms = record.get("text_terms") or {}
    desc_terms = _extract_text_terms(text_terms.get("description") or text_terms.get("productDisplayName"), limit=4)
    desc_phrase = ", ".join(desc_terms)

    # Relational context gives the authoritative parent + sibling picture.
    # Build human-readable sibling descriptions (never expose raw prefix token IDs).
    sibling_descs = []
    if relational_context:
        for sib in (relational_context.get("sibling_deltas") or []):
            top_cats = sib.get("top_categories") or []
            key_terms_raw = sib.get("key_terms") or []
            if top_cats:
                # e.g. "Tops (40%)" → "Tops"
                label = top_cats[0].split("(")[0].strip()
            elif key_terms_raw:
                terms = _extract_text_terms(key_terms_raw, limit=2)
                label = " ".join(terms) if terms else None
            else:
                label = None
            if label:
                sibling_descs.append(label)

    sibling_info = record.get("sibling_info") or {}
    # Fall back to sibling_info.top_siblings key_terms if relational_context gave nothing
    if not sibling_descs:
        for sib in (sibling_info.get("top_siblings") or []):
            if sib.get("value") and sib.get("value") != record["prefix"]:
                sibling_descs.append(sib["value"])
        sibling_descs = sibling_descs[:2]

    universal_attrs = _collect_universal_attributes(record)
    # Prefer a high-precision (discriminating) sig that spans the whole group as the
    # universals anchor; fall back to any full-coverage sig regardless of precision.
    full_span_sig = next(
        (sig for sig in disc_sigs if sig.get("group_pct", 0) >= 99.5), None
    ) or next(
        (sig for sig in attr_sigs if sig.get("group_pct", 0) >= 99.5), None
    )
    if full_span_sig:
        for key, value in (full_span_sig.get("values") or {}).items():
            universal_attrs.setdefault(key, value)

    palette_sets = {
        col: vals[:5]
        for col, vals in _collect_complete_value_sets(record).items()
        if col not in universal_attrs
    }
    palette_snippets = []
    for col, values in palette_sets.items():
        if not values:
            continue
        palette_snippets.append(
            f"{_humanize_attribute_label(col)} limited to {_natural_join(values)}"
        )
    palette_sentence = ""
    if palette_snippets:
        palette_sentence = " They only appear in " + "; ".join(palette_snippets) + "."

    def _render_signature(values_dict):
        parts = []
        for key, value in values_dict.items():
            if value is None:
                continue
            parts.append(f"{_humanize_attribute_label(key)} {value}")
        return ", ".join(parts)

    signature_sentence = _render_signature(universal_attrs)
    if desc_phrase and signature_sentence:
        core_clause = f"{desc_phrase} {signature_sentence}"
    elif desc_phrase:
        core_clause = desc_phrase
    elif signature_sentence:
        core_clause = signature_sentence
    else:
        core_clause = "this exact cluster"

    # sibling_guard: human-readable exclusion hint used in query sentences.
    # Uses category names, never raw prefix token IDs.
    sibling_guard = ""
    if sibling_descs:
        sibling_guard = " (excluding " + " and ".join(sibling_descs[:2]) + " styles)"

    negative_constraints = []
    for col, profile in (record.get("categorical_profiles") or {}).items():
        if not profile or not profile.get("exclusive"):
            continue
        top_val = profile["top_values"][0]["value"]
        negative_constraints.append(f"{col}!={top_val}")

    include_clause = " AND ".join(f"{k}={v}" for k, v in universal_attrs.items())

    queries = []

    # --- strict_retrieval: anchored to the best discriminating signature ---
    # The top discriminating sig has precision ≥90%, meaning ≤10% false positives
    # from other groups; this is the exclusivity anchor for the query.
    best_disc_sig = disc_sigs[0] if disc_sigs else (attr_sigs[0] if attr_sigs else None)
    disc_precision_note = ""
    if best_disc_sig:
        prec = best_disc_sig.get("precision", 0)
        disc_precision_note = f" (exclusivity anchor: {prec * 100:.0f}% precision)"

    strict_sentence = (
        "I'm shopping for "
        f"{core_clause} pieces.{palette_sentence}"
        f" Show me the complete collection — every available style and type{sibling_guard}."
    )
    queries.append(
        {
            "intent": "strict_retrieval",
            "natural_language_query": strict_sentence.strip(),
            "constraints": {
                "include": universal_attrs,
                "exclude": negative_constraints,
            },
            "discriminating_precision": best_disc_sig.get("precision") if best_disc_sig else None,
        }
    )

    palette_remention = palette_sentence or ""
    sibling_clause = (
        f" without overlapping with {' or '.join(sibling_descs[:2])} items"
        if sibling_descs
        else ""
    )
    user_sentence = (
        "Show me "
        f"{core_clause} options — I want to browse the full range of styles and types."
        f"{palette_remention}"
        f" Keep results focused on this exact product cluster{sibling_clause}."
    )
    queries.append(
        {
            "intent": "user_query",
            "natural_language_query": user_sentence.strip(),
        }
    )

    breadth_sentence = (
        "Find everything that fits "
        f"{core_clause} — include all variations and sub-types in one complete result set."
        f"{palette_remention}"
        " The results should be unique to this cluster and not spill into related categories."
    )
    queries.append(
        {
            "intent": "broad_recall",
            "natural_language_query": breadth_sentence.strip(),
        }
    )

    if sibling_descs:
        focus_sibling_desc = sibling_descs[0]
        palette_support = "the colour and material palette" if palette_sentence else "the exclusive style cues"
        contrast_sentence = (
            f"What makes these {desc_phrase or 'items'} distinct from {focus_sibling_desc} items?"
            f" Describe the specific attributes that separate this product group from {focus_sibling_desc},"
            f" using {palette_support} to show the description cannot apply elsewhere."
        )
        queries.append(
            {
                "intent": "contrast",
                "natural_language_query": contrast_sentence.strip(),
            }
        )

    # --- Per-subcategory exhaustiveness queries ---
    # Compute available budget: base slots are strict_retrieval + user_query +
    # broad_recall (3) plus contrast if there are siblings (4).  Give the remaining
    # slots to per-subcategory queries so a diverse group gets full coverage.
    base_slots = 3 + (1 if sibling_descs else 0)
    available_for_subcats = max(num_queries - base_slots, 2)

    if cov_sigs or _category_entropy(record) >= 1.0:
        subcat_queries = _per_subcategory_queries(
            record,
            sibling_guard=sibling_guard,
            desc_phrase=desc_phrase,
            max_subcats=available_for_subcats,
        )
        queries.extend(subcat_queries)

    return queries[:num_queries]


def _category_entropy(record):
    """Shannon entropy (bits) of the masterCategory/articleType distribution within the prefix."""
    profiles = record.get("categorical_profiles") or {}
    cat_profile = profiles.get("masterCategory") or profiles.get("articleType")
    if not cat_profile:
        return 0.0
    counts = [e.get("count", 0) for e in (cat_profile.get("top_values") or []) if e.get("count")]
    return _entropy_from_counts(counts)


def _per_subcategory_queries(record, sibling_guard, desc_phrase, max_subcats=6):
    """
    Generate one focused query per major product sub-type inside the prefix.

    When a prefix group spans multiple distinct categories (high entropy), a single
    broad query cannot reach all of them.  This function emits one targeted query per
    major sub-type (≥5% share, capped at *max_subcats*), plus a catch-all for the rest.

    *max_subcats* is normally passed in from craft_gemini_queries as the remaining
    budget after the 3-4 base queries, so the cap adapts to the caller's num_queries.

    Returns a list of query dicts.
    """
    profiles = record.get("categorical_profiles") or {}
    cat_profile = profiles.get("masterCategory") or profiles.get("articleType")
    if not cat_profile:
        return []

    top_values = cat_profile.get("top_values") or []
    major = [e for e in top_values if (e.get("pct") or 0) >= 5.0][:max_subcats]
    if len(major) < 2:
        # Not enough sub-type diversity to warrant per-subcat queries
        return []

    covered_cats = set()
    queries = []
    for entry in major:
        cat_name = entry.get("value")
        pct = entry.get("pct", 0)
        if not cat_name:
            continue
        covered_cats.add(cat_name)
        sentence = (
            f"Show me {cat_name} items"
            + (f" with {desc_phrase} aesthetic" if desc_phrase else "")
            + (sibling_guard or "")
            + "."
        )
        queries.append({
            "intent": f"subcategory_{cat_name.lower().replace(' ', '_')}",
            "natural_language_query": sentence.strip(),
            "covers_category": cat_name,
        })

    # Catch-all for minority types
    minor = [e["value"] for e in top_values if (e.get("pct") or 0) < 5.0 and e.get("value")]
    if minor:
        minority_list = _natural_join(minor[:5])
        catchall = (
            f"Find all {minority_list} and related less-common styles"
            + (f" with {desc_phrase} character" if desc_phrase else "")
            + (sibling_guard or "")
            + "."
        )
        queries.append({
            "intent": "subcategory_catchall",
            "natural_language_query": catchall.strip(),
            "covers_category": "minority_types",
        })

    return queries


# %%
def _extract_text_terms(term_entries, limit=5):
    """Normalize free-text term payloads into bare strings."""
    terms = []
    if not term_entries:
        return terms
    for entry in term_entries:
        term = None
        if isinstance(entry, dict):
            term = entry.get("term") or entry.get("value")
        elif isinstance(entry, (list, tuple)) and entry:
            term = entry[0]
        else:
            term = entry
        if not term:
            continue
        terms.append(str(term))
        if len(terms) >= limit:
            break
    return terms
 

def _humanize_attribute_label(label):
    if label is None:
        return ""
    return str(label).replace("_", " ").strip()


def _natural_join(values):
    cleaned = [str(val) for val in values if val]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f", or {cleaned[-1]}"


def _collect_universal_attributes(record, coverage_threshold=0.85):
    universals = {}
    profiles = record.get("categorical_profiles") or {}
    for attr in record.get("exclusive_attributes") or []:
        profile = profiles.get(attr)
        if profile and profile.get("top_values"):
            universals[attr] = profile["top_values"][0]["value"]
    for col, profile in profiles.items():
        if not profile:
            continue
        top_values = profile.get("top_values") or []
        if not top_values:
            continue
        top_entry = top_values[0]
        pct = top_entry.get("pct")
        if pct is None:
            continue
        if pct >= coverage_threshold * 100:
            universals.setdefault(col, top_entry.get("value"))
    return universals


def _collect_complete_value_sets(record, coverage_threshold=0.98):
    sets_map = {}
    group_size = record.get("group_size") or 0
    profiles = record.get("categorical_profiles") or {}
    for col, profile in profiles.items():
        if not profile:
            continue
        values = profile.get("top_values") or []
        if not values or len(values) == 1:
            continue
        coverage = 0.0
        collected = []
        for entry in values:
            count = entry.get("count")
            pct = entry.get("pct")
            if group_size and isinstance(count, (int, float)):
                coverage += (count or 0) / group_size
            elif isinstance(pct, (int, float)):
                coverage += pct / 100.0
            collected.append(entry.get("value"))
        if coverage >= coverage_threshold:
            sets_map[col] = [val for val in collected if val]
    return sets_map


def _summarize_common_attributes(record, categorical_threshold=0.60, text_top_n=8):
    """Return attributes that most items inside the prefix share."""
    if not record:
        return {}
    summary = {}
    threshold_pct = categorical_threshold * 100
    group_size = record.get("group_size") or 0
    categorical_payload = {}
    for col, profile in (record.get("categorical_profiles") or {}).items():
        if not profile:
            continue
        selected = []
        for entry in profile.get("top_values", []):
            pct = entry.get("pct")
            value = entry.get("value")
            if value is None or pct is None:
                continue
            try:
                pct_val = float(pct)
            except (TypeError, ValueError):
                continue
            if pct_val < threshold_pct:
                break
            payload = {
                "value": value,
                "pct": pct_val,
                "count": entry.get("count"),
            }
            if group_size and isinstance(entry.get("count"), (int, float)):
                payload["overlap_ratio"] = (entry.get("count") or 0) / group_size
            selected.append(payload)
        if selected:
            categorical_payload[col] = selected
    if categorical_payload:
        summary["categorical"] = categorical_payload

    numeric_payload = {}
    for col, stats in (record.get("numeric_profiles") or {}).items():
        if not stats:
            continue
        payload = {
            "median": stats.get("median"),
            "iqr": (
                [stats.get("q25"), stats.get("q75")] if stats.get("q25") is not None and stats.get("q75") is not None else None
            ),
            "mean": stats.get("mean"),
            "min": stats.get("min"),
            "max": stats.get("max"),
        }
        if any(val is not None for val in payload.values()):
            numeric_payload[col] = payload
    if numeric_payload:
        summary["numeric"] = numeric_payload

    text_payload = {}
    for col, terms in (record.get("text_terms") or {}).items():
        if not terms:
            continue
        collected = []
        for entry in terms[:text_top_n]:
            if isinstance(entry, (list, tuple)):
                term = entry[0]
                score = entry[1] if len(entry) > 1 else None
            elif isinstance(entry, dict):
                term = entry.get("term")
                score = entry.get("score")
            else:
                term = entry
                score = None
            if not term:
                continue
            collected.append({"term": term, "score": score})
        if collected:
            text_payload[col] = collected
    if text_payload:
        summary["text"] = text_payload

    return summary


def _format_common_attribute_snippet(common_attrs, group_size=None, per_type_limit=3):
    if not common_attrs:
        return []
    lines = []
    categorical_group_size = group_size if isinstance(group_size, (int, float)) and group_size else None
    for col, entries in (common_attrs.get("categorical") or {}).items():
        formatted = []
        for entry in entries[:per_type_limit]:
            value = entry.get("value")
            pct = entry.get("pct")
            if value is None or not isinstance(pct, (int, float)):
                continue
            fragments = [f"{pct:.1f}% overlap"]
            count_val = entry.get("count")
            if categorical_group_size and isinstance(count_val, (int, float)):
                fragments.append(f"{int(count_val)}/{int(categorical_group_size)} items")
            elif isinstance(count_val, (int, float)):
                fragments.append(f"{int(count_val)} items")
            formatted.append(f"{value} ({' | '.join(fragments)})")
        if formatted:
            lines.append(f"{col}: {', '.join(formatted)}")
    for col, stats in (common_attrs.get("numeric") or {}).items():
        median = stats.get("median")
        if median is None:
            continue
        median_str = f"{median:.2f}" if isinstance(median, (int, float)) else str(median)
        line = f"{col}: median {median_str}"
        iqr = stats.get("iqr")
        if (
            isinstance(iqr, (list, tuple))
            and len(iqr) == 2
            and all(isinstance(val, (int, float)) for val in iqr if val is not None)
        ):
            line += f", IQR {iqr[0]:.2f}-{iqr[1]:.2f}"
        lines.append(line)
    for col, terms in (common_attrs.get("text") or {}).items():
        names = []
        for term in terms[:per_type_limit]:
            if not isinstance(term, dict):
                continue
            label = term.get("term")
            score = term.get("score")
            if not label:
                continue
            if isinstance(score, (int, float)):
                names.append(f"{label} (score {score:.2f})")
            else:
                names.append(label)
        if names:
            lines.append(f"{col} terms: {', '.join(names)}")
    return lines


if "_ORIGINAL_BUILD_PREFIX_WORKFLOW_RECORD" not in globals():
    _ORIGINAL_BUILD_PREFIX_WORKFLOW_RECORD = build_prefix_workflow_record
else:
    build_prefix_workflow_record = _ORIGINAL_BUILD_PREFIX_WORKFLOW_RECORD


@wraps(_ORIGINAL_BUILD_PREFIX_WORKFLOW_RECORD)
def build_prefix_workflow_record(*args, **kwargs):
    record = _ORIGINAL_BUILD_PREFIX_WORKFLOW_RECORD(*args, **kwargs)
    if not record:
        return record
    common_attrs = _summarize_common_attributes(record)
    if common_attrs:
        record["common_attributes"] = common_attrs
    else:
        record.pop("common_attributes", None)
    return record


if "_ORIGINAL_BUILD_PROMPT_BLUEPRINT" not in globals():
    _ORIGINAL_BUILD_PROMPT_BLUEPRINT = build_prompt_blueprint
else:
    build_prompt_blueprint = _ORIGINAL_BUILD_PROMPT_BLUEPRINT


@wraps(_ORIGINAL_BUILD_PROMPT_BLUEPRINT)
def build_prompt_blueprint(record):
    base = _ORIGINAL_BUILD_PROMPT_BLUEPRINT(record)
    tail_lines = []
    description_terms = _extract_text_terms(
        (record.get("text_terms") or {}).get("description")
        or (record.get("text_terms") or {}).get("productDisplayName"),
        limit=5,
    )
    if description_terms:
        tail_lines.append("Description cues: " + ", ".join(description_terms))
    additions = _format_common_attribute_snippet(
        record.get("common_attributes"),
        group_size=record.get("group_size"),
    )
    if additions:
        tail_lines.append("Common attribute highlights:")
        tail_lines.extend(f"  - {line}" for line in additions)
    if not tail_lines:
        return base
    return base + "\n" + "\n".join(tail_lines)

# %%
# Gemini client configuration and helpers
import os
from google import genai
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(vertexai=True, project="ailooks-non-prod-dcd1", location="global")


def _response_text(response):
    text_chunks = []
    if not response or not getattr(response, "candidates", None):
        return ""
    for candidate in response.candidates:
        candidate_content = getattr(candidate, "content", None)
        if not candidate_content or not getattr(candidate_content, "parts", None):
            continue
        for part in candidate_content.parts:
            if hasattr(part, "text"):
                text_chunks.append(part.text)
    return "\n".join(text_chunks).strip()


def call_gemini_for_prefix(record, relational_context, temperature=0.2, top_p=0.8, df=None):
    prompt = compose_gemini_prompt(record, relational_context, df=df)
    parts = prompt

    resp = client.models.generate_content(
        model=MODEL,
        contents=parts,
        config=types.GenerateContentConfig(temperature=temperature, top_p=top_p),
    )
    return resp, _response_text(resp)

# %%
# Batch data generation for level-1, level-2, and level-3 common prefixes
from pathlib import Path
from datetime import datetime
import json

OUTPUT_STREAM_PATH = Path("./data/output/data_samples_public_dataset_unfiltered_purity_v3.json")
OUTPUT_STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)

LEVEL_CONFIG = {
    1: {"top_n": None},  # process every qualifying 1-token (top-level) prefix
    2: {"top_n": None},  # process every qualifying 2-token prefix
    3: {"top_n": None},  # process every qualifying 3-token prefix
}

MIN_GROUP_SIZE = 5
NAME_SAMPLE_LIMIT = 25

# Purity filter: skip clusters where the dominant articleType covers < this fraction
# of items — they are mixed-category buckets that make poor benchmark anchors.
# E.g. 64-cb sid_0=41 has Apparel/Accessories/Footwear mixed → purity ≈ 0.64 → filtered.
CLUSTER_PURITY_THRESHOLD = 0.80

# Singleton filter: at the full SID depth (level == max_tokens) a cluster of size 1
# has no shared-prefix neighbours, making it useless for retrieval evaluation.
SINGLETON_MIN_CLUSTER_SIZE = 2

GEMINI_USER_QUERY_KEYS = {"user_queries", "queries", "top_queries", "key_queries"}
GEMINI_PARENT_KEYS = {"parent_delta", "parent_deltas", "parent_highlights", "parent_shift", "parent_insights"}
GEMINI_SIBLING_KEYS = {"sibling_delta", "sibling_deltas", "sibling_contrasts", "sibling_highlights", "sibling_insights"}


def _json_safe(value):
    """Recursively convert numpy / polars scalars to Python types for JSON serialization."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    for attr_name in ("to_dict", "model_dump"):
        method = getattr(value, attr_name, None)
        if callable(method):
            try:
                return _json_safe(method())
            except Exception:
                pass
    for attr_name in ("to_json", "model_dump_json"):
        method = getattr(value, attr_name, None)
        if callable(method):
            try:
                serialized = method()
                if isinstance(serialized, str):
                    return _json_safe(json.loads(serialized))
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _collect_item_samples(df, prefix_string, sample_limit=NAME_SAMPLE_LIMIT):
    tokens = re.findall(r'<\|sid_\d+\|>', prefix_string)
    prefix_col = f"prefix_{len(tokens)}tok"
    if prefix_col not in df.columns:
        return [], [], []
    candidate_cols = [col for col in (
        "id", "productDisplayName", "brandName", "masterCategory",
        "subCategory", "articleType", "baseColour", "gender", "imageURL",
    ) if col in df.columns]
    if not candidate_cols:
        return [], [], []
    subset = (
        df.filter(pl.col(prefix_col) == prefix_string)
        .select(candidate_cols)
        .head(sample_limit)
        .to_dicts()
    )
    names = [row.get("productDisplayName") for row in subset if row.get("productDisplayName")]
    image_urls = []
    for row in subset:
        url = row.get("imageURL")
        if url:
            image_urls.append(url)
    return names, image_urls, subset


def _summarize_parent_deltas(delta_map, max_items=2):
    highlights = []
    if not delta_map:
        return highlights
    for column, deltas in delta_map.items():
        if not deltas:
            continue
        snippets = []
        for delta in deltas:
            value = delta.get("value")
            delta_pct = delta.get("delta_pct")
            if value is None or delta_pct is None:
                continue
            snippets.append(f"{value} ({delta_pct:+.1f}pp)")
            if len(snippets) >= max_items:
                break
        if snippets:
            highlights.append({"column": column, "shifts": snippets})
    return highlights


def _summarize_sibling_deltas(sibling_deltas, max_items=2):
    contrasts = []
    if not sibling_deltas:
        return contrasts
    for sibling in sibling_deltas:
        highlights = []
        categorical_diffs = (sibling.get("diffs") or {}).get("categorical") or {}
        for column, deltas in categorical_diffs.items():
            if not deltas:
                continue
            first_delta = next(
                (entry for entry in deltas if entry.get("delta_pct") not in (None, 0.0)),
                None,
            )
            if first_delta:
                highlights.append(
                    f"{column}: {first_delta['value']} ({first_delta['delta_pct']:+.1f}pp)"
                )
            if len(highlights) >= max_items:
                break
        contrasts.append(
            {
                "prefix": sibling.get("prefix"),
                "share_pct": sibling.get("share_pct"),
                "highlights": highlights,
            }
        )
    return contrasts


def _build_structured_insights(record, relational_context, sample_names, delta_limit=2):
    cat_profiles = record.get("categorical_profiles") or {}
    text_terms = record.get("text_terms") or {}

    def _top_values(profile, limit=3):
        if not profile:
            return []
        return [
            entry.get("value")
            for entry in (profile.get("top_values") or [])[:limit]
            if entry.get("value")
        ]

    insights = {
        "summary": build_prompt_blueprint(record),
        "top_categories": _top_values(cat_profiles.get("masterCategory") or cat_profiles.get("articleType")),
        "top_brands": _top_values(cat_profiles.get("brandName")),
        "description_terms": [
            term
            for term, _ in (text_terms.get("description") or text_terms.get("productDisplayName") or [])[:5]
        ],
        "exclusive_attributes": record.get("exclusive_attributes") or [],
        "attribute_signatures": (record.get("attribute_signatures") or [])[:3],
        "sample_names": sample_names[:5],
    }
    parent_record = (relational_context or {}).get("parent_record")
    if parent_record:
        insights["parent_prefix"] = parent_record.get("prefix")
    insights["parent_delta_highlights"] = _summarize_parent_deltas(
        (relational_context or {}).get("category_deltas_vs_parent"),
        max_items=delta_limit,
    )
    insights["sibling_contrasts"] = _summarize_sibling_deltas(
        (relational_context or {}).get("sibling_deltas"),
        max_items=delta_limit,
    )
    return insights


def _normalize_section_key(label):
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return normalized or "section"


def _collect_sections_by_keys(sections, candidate_keys):
    aggregated = []
    for key in candidate_keys:
        aggregated.extend(sections.get(key, []))
    return aggregated


def _extract_json_from_blob(text_blob):
    """Try to extract a JSON object from a text blob (handles markdown code blocks)."""
    if not text_blob:
        return None
    # 1. Try markdown code block: ```json ... ``` or ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text_blob)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    # 2. Try the entire blob as JSON
    try:
        return json.loads(text_blob.strip())
    except json.JSONDecodeError:
        pass
    # 3. Find the first { ... } span and try that
    brace_start = text_blob.find('{')
    brace_end = text_blob.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text_blob[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _queries_from_json(json_data):
    """Extract a flat list of query strings from a Gemini JSON response."""
    queries = []
    for key in ("user_queries", "queries", "top_queries", "key_queries"):
        val = json_data.get(key)
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip())
            elif isinstance(item, dict):
                text = (
                    item.get("query")
                    or item.get("text")
                    or item.get("natural_language_query")
                    or item.get("description")
                )
                if text and str(text).strip():
                    queries.append(str(text).strip())
        if queries:
            return queries
    return queries


def _parse_gemini_text_blob(text_blob):
    """Parse a Gemini response, handling JSON (with or without code fence) and free-form text."""
    sections = {}
    key_value_pairs = []
    empty_result = {
        "sections": sections,
        "key_value_pairs": key_value_pairs,
        "key_value_map": {},
        "flat_key_value_map": {},
        "user_queries": [],
        "parent_deltas": [],
        "sibling_deltas": [],
        "coverage_map": {},
        "validation_checks": [],
    }
    if not text_blob:
        return empty_result

    # --- 1. Try structured JSON parsing first ---
    json_data = _extract_json_from_blob(text_blob)
    if json_data and isinstance(json_data, dict):
        user_queries = _queries_from_json(json_data)

        parent_deltas = []
        for key in ("delta_vs_parent", "parent_delta", "parent_deltas", "parent_highlights", "parent_insights"):
            val = json_data.get(key)
            if not val:
                continue
            if isinstance(val, list):
                parent_deltas.extend(str(v) for v in val if v)
            elif isinstance(val, str):
                parent_deltas.append(val)
            if parent_deltas:
                break

        sibling_deltas = []
        for key in ("delta_vs_siblings", "sibling_delta", "sibling_deltas", "sibling_contrasts", "sibling_insights"):
            val = json_data.get(key)
            if not val:
                continue
            if isinstance(val, list):
                sibling_deltas.extend(str(v) for v in val if v)
            elif isinstance(val, str):
                sibling_deltas.append(val)
            if sibling_deltas:
                break

        # Build flat sections from all JSON keys for downstream compatibility
        for k, v in json_data.items():
            norm_k = _normalize_section_key(k)
            if not norm_k:
                continue
            if isinstance(v, list):
                sections[norm_k] = [str(item) for item in v if item is not None]
            elif isinstance(v, str):
                sections[norm_k] = [v]
            elif v is not None:
                sections[norm_k] = [str(v)]

        key_value_map = {
            k: {"label": k, "values": v}
            for k, v in sections.items()
        }
        flat_key_value_map = {
            k: v[0] if len(v) == 1 else v
            for k, v in sections.items()
            if v
        }
        # Coverage map: {"product_type": "query_index_or_description"}
        coverage_map = json_data.get("coverage_map") or {}
        if not isinstance(coverage_map, dict):
            coverage_map = {}

        # Validation checks: list of strings Gemini used to self-verify coverage/exclusivity
        validation_checks = json_data.get("validation_checks") or []
        if isinstance(validation_checks, list):
            validation_checks = [str(v) for v in validation_checks if v]
        else:
            validation_checks = []

        return {
            "sections": sections,
            "key_value_pairs": key_value_pairs,
            "key_value_map": key_value_map,
            "flat_key_value_map": flat_key_value_map,
            "user_queries": user_queries,
            "parent_deltas": parent_deltas,
            "sibling_deltas": sibling_deltas,
            "coverage_map": coverage_map,
            "validation_checks": validation_checks,
        }

    # --- 2. Fall back to line-by-line key:value parsing ---
    current_key = None
    for raw_line in text_blob.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            current_key = None
            continue
        candidate = stripped.lstrip("-*• ")
        colon_idx = candidate.find(":")
        if 0 < colon_idx < 80:
            raw_key = candidate[:colon_idx].strip()
            value = candidate[colon_idx + 1 :].strip()
            normalized_key = _normalize_section_key(raw_key)
            if normalized_key:
                sections.setdefault(normalized_key, []).append(value)
                key_value_pairs.append(
                    {
                        "key": raw_key,
                        "normalized_key": normalized_key,
                        "value": value,
                    }
                )
                current_key = normalized_key
            continue
        if current_key:
            sections.setdefault(current_key, []).append(candidate)
    label_lookup = {}
    for pair in key_value_pairs:
        label_lookup.setdefault(pair["normalized_key"], pair["key"])
    key_value_map = {
        normalized: {
            "label": label_lookup.get(normalized, normalized),
            "values": values,
        }
        for normalized, values in sections.items()
    }
    flat_key_value_map = {
        normalized: values[0] if len(values) == 1 else values
        for normalized, values in sections.items()
        if values
    }
    return {
        "sections": sections,
        "key_value_pairs": key_value_pairs,
        "key_value_map": key_value_map,
        "flat_key_value_map": flat_key_value_map,
        "user_queries": _collect_sections_by_keys(sections, GEMINI_USER_QUERY_KEYS),
        "parent_deltas": _collect_sections_by_keys(sections, GEMINI_PARENT_KEYS),
        "sibling_deltas": _collect_sections_by_keys(sections, GEMINI_SIBLING_KEYS),
    }


def _extract_prefix_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    stored_prefix = payload.get("prefix")
    if isinstance(stored_prefix, str):
        return stored_prefix
    if len(payload) == 1:
        prefix_candidate, nested_value = next(iter(payload.items()))
        if isinstance(prefix_candidate, str) and isinstance(nested_value, dict):
            return prefix_candidate
    return None


def _load_existing_entries(path):
    if not path.exists():
        return {}
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return {}
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        print(f"Detected legacy JSONL format in {path}; converting in-memory to JSON.")
        entries = {}
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            record_line = line.strip()
            if not record_line:
                continue
            try:
                payload = json.loads(record_line)
            except json.JSONDecodeError:
                print(f"Warning: could not parse line {line_number} in {path}; skipping.")
                continue
            prefix = _extract_prefix_from_payload(payload)
            if not prefix:
                continue
            nested_value = payload.get(prefix) if isinstance(payload, dict) else None
            if isinstance(nested_value, dict):
                entries[prefix] = nested_value
            else:
                cleaned_payload = dict(payload) if isinstance(payload, dict) else {}
                cleaned_payload.pop("prefix", None)
                entries[prefix] = cleaned_payload
        return entries
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    if isinstance(data, list):
        entries = {}
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            prefix = entry.get("prefix")
            if not prefix:
                record_digest = entry.get("record_digest") or {}
                prefix = record_digest.get("prefix")
            if not prefix:
                record_payload = entry.get("record") or entry.get("payload") or {}
                prefix = record_payload.get("prefix")
            if not prefix:
                continue
            entries[str(prefix)] = entry
        return entries
    print(f"Warning: expected JSON object/list in {path}; resetting file contents.")
    return {}


def _write_entries(path, entries):
    """Full rewrite — kept for one-off exports; the batch loop uses _append_entry."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")


def _ensure_jsonl_format(path, entries):
    """
    If *path* holds a pretty-printed JSON dict (written by a previous _write_entries
    call) convert it to JSONL in-place so incremental _append_entry calls work.
    A JSONL file already starts with '{' on each line — detect the dict case by
    checking whether the very first character is '{' AND the file has only one JSON
    object (i.e. the whole file parses as a dict rather than multiple lines).
    """
    if not path.exists() or not entries:
        return
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return
    # A multi-entry JSONL file has multiple newlines each starting a JSON object.
    # A single JSON dict starts with '{' and ends with '}' with no top-level newline
    # separating independent objects.  The cheapest check: try parsing the whole
    # file as one JSON value — if it succeeds, it's the dict format that needs migrating.
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return  # already JSONL (or empty) — nothing to migrate
    # Rewrite as JSONL (one compact line per prefix)
    with path.open("w", encoding="utf-8") as fh:
        for prefix, entry in entries.items():
            fh.write(json.dumps({prefix: entry}, ensure_ascii=False) + "\n")


def _append_entry(path, prefix, entry):
    """
    Append a single completed entry as one JSONL line and flush immediately.

    Writing one line per entry means:
    - Each completed prefix is durable the moment this call returns.
    - A crash mid-write corrupts at most the current (in-progress) line; all
      prior lines remain intact and are loaded correctly on the next run.
    - Writing is O(1) — no full-file rewrite as the output grows.
    """
    line = json.dumps({prefix: entry}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _top_prefixes_for_level(df, level, top_n, min_group_size):
    col_name = f"prefix_{level}tok"
    if col_name not in df.columns:
        print(f"Skipping level {level}: column {col_name} not found in dataframe.")
        return []
    counts = (
        df.filter(pl.col(col_name).is_not_null())
        .group_by(col_name)
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .filter(pl.col("count") >= min_group_size)
    )
    if top_n is not None:
        counts = counts.head(top_n)
    return counts[col_name].to_list()


def _cluster_articletype_purity(df, prefix_col, prefix_string):
    """Return (dominant_count / total_count) for articleType within the cluster.

    Uses only the fastest possible Polars group-by — no Python loops.
    Returns 1.0 for clusters where articleType is missing (not filterable).
    Returns 0.0 if the cluster is empty (should not reach this function).
    """
    if "articleType" not in df.columns:
        return 1.0
    counts = (
        df.filter(pl.col(prefix_col) == prefix_string)
        .filter(pl.col("articleType").is_not_null())
        .group_by("articleType")
        .agg(pl.count().alias("cnt"))
    )
    if counts.is_empty():
        return 1.0
    total = counts["cnt"].sum()
    dominant = counts["cnt"].max()
    return float(dominant) / float(total)


def _record_digest(record):
    cat_profiles = record.get("categorical_profiles") or {}

    def _safe_top_values(profile):
        if not isinstance(profile, dict):
            return []
        top_values = profile.get("top_values")
        return top_values if isinstance(top_values, list) else []

    return {
        "prefix": record.get("prefix"),
        "level": record.get("level"),
        "group_size": record.get("group_size"),
        "coverage_pct": record.get("coverage_pct"),
        "top_categories": _safe_top_values(cat_profiles.get("masterCategory") or cat_profiles.get("articleType")),
        "top_brands": _safe_top_values(cat_profiles.get("brandName")),
    }


existing_entries = _load_existing_entries(OUTPUT_STREAM_PATH)
existing_prefixes = set(existing_entries.keys())
# Migrate a legacy pretty-printed JSON dict to JSONL so all subsequent writes are
# safe append operations (crash-proof, O(1) per entry).
_ensure_jsonl_format(OUTPUT_STREAM_PATH, existing_entries)
processed = 0
skipped = 0
start_time = datetime.utcnow()
# Shared record cache across all iterations — avoids rebuilding the same parent /
# sibling records for every prefix processed in this batch run.
_batch_record_cache: dict = {}
if existing_prefixes:
    print(
        f"Resuming: {len(existing_prefixes)} prefixes already done, continuing from next unprocessed prefix."
    )
else:
    print(f"Starting fresh batch generation → {OUTPUT_STREAM_PATH}")
print(f"Start time: {start_time.isoformat()}Z")

for level, config in LEVEL_CONFIG.items():
    candidate_prefixes = _top_prefixes_for_level(df, level, config.get("top_n"), MIN_GROUP_SIZE)
    if not candidate_prefixes:
        continue
    for prefix in candidate_prefixes:
        if not prefix:
            continue
        if prefix in existing_prefixes:
            skipped += 1
            continue

        # --- Purity filter ---
        # Skip mixed-category clusters: when the dominant articleType covers less
        # than CLUSTER_PURITY_THRESHOLD of the cluster, the SID prefix conflates
        # semantically distinct product types (e.g. 64-cb sid_0=41 mixes Apparel +
        # Accessories + Footwear).  Such clusters produce low-quality benchmark tasks
        # because a correct "same-SID" retrieval would surface unrelated products.
        _prefix_col = f"prefix_{level}tok"
        _purity = _cluster_articletype_purity(df, _prefix_col, prefix)
        if _purity < CLUSTER_PURITY_THRESHOLD:
            print(f"  [purity skip] {prefix} (level {level}): articleType purity={_purity:.2f} < {CLUSTER_PURITY_THRESHOLD}")
            skipped += 1
            continue

        # --- Singleton filter (full-depth clusters only) ---
        # At the maximum SID depth every item ideally shares its full SID with at
        # least one neighbour.  A cluster of size 1 has no shared-prefix partner, so
        # it cannot be used as a retrieval benchmark anchor.
        if level == max_tokens:
            _cluster_size = df.filter(pl.col(_prefix_col) == prefix).height
            if _cluster_size < SINGLETON_MIN_CLUSTER_SIZE:
                skipped += 1
                continue

        _PUBLIC_CAT_COLS = ["masterCategory", "subCategory", "articleType", "brandName", "baseColour", "gender", "usage"]
        _PUBLIC_NUM_COLS = ["price", "discountedPrice"]
        _PUBLIC_TXT_COLS = ["productDisplayName", "description"]

        record = _batch_record_cache.get(prefix) or build_prefix_workflow_record(
            df,
            prefix,
            categorical_cols=_PUBLIC_CAT_COLS,
            numeric_cols=_PUBLIC_NUM_COLS,
            text_cols=_PUBLIC_TXT_COLS,
            top_k=20,
            min_group_size=MIN_GROUP_SIZE,
        )
        if not record:
            skipped += 1
            continue
        _batch_record_cache[prefix] = record
        relational_context = build_relational_context(
            df,
            record,
            categorical_cols=_PUBLIC_CAT_COLS,
            numeric_cols=_PUBLIC_NUM_COLS,
            text_cols=_PUBLIC_TXT_COLS,
            top_k=20,
            sibling_limit=3,
            record_cache=_batch_record_cache,
        )
        queries = craft_gemini_queries(
            record,
            num_queries=10,    # was 5; extra budget for per-subcategory queries
            relational_context=relational_context,
        )
        names, image_urls, sample_items = _collect_item_samples(df, prefix, sample_limit=NAME_SAMPLE_LIMIT)
        try:
            gemini_response, gemini_text = call_gemini_for_prefix(record, relational_context, df=df)
        except Exception as exc:
            print(f"Gemini call failed for prefix {prefix}: {exc}")
            skipped += 1
            continue
        gemini_structured = _parse_gemini_text_blob(gemini_text)
        gemini_user_queries = gemini_structured["user_queries"]
        gemini_parent_highlights = gemini_structured["parent_deltas"]
        gemini_sibling_highlights = gemini_structured["sibling_deltas"]
        gemini_coverage_map = gemini_structured.get("coverage_map") or {}
        gemini_validation_checks = gemini_structured.get("validation_checks") or []

        # Coverage gap check: warn if any major product type (≥5%) is absent from
        # Gemini's coverage_map.  Use the same axis that compose_gemini_prompt showed
        # to Gemini — the most-specific attribute with dominant purity ≥80%.
        _cat_profiles = record.get("categorical_profiles") or {}
        _gap_col, _ = _best_purity_axis(_cat_profiles)
        if _gap_col is None:
            _gap_col = "masterCategory"
        _primary_cat_profile = _cat_profiles.get(_gap_col) or {}
        major_cats = [
            e["value"]
            for e in (_primary_cat_profile.get("top_values") or [])
            if (e.get("pct") or 0) >= 5.0 and e.get("value")
        ]
        uncovered_cats = [cat for cat in major_cats if cat not in gemini_coverage_map]
        if uncovered_cats and gemini_coverage_map:
            print(f"  [coverage gap] {prefix}: {uncovered_cats} not in Gemini coverage_map")
        insights = _build_structured_insights(
            record,
            relational_context,
            names,
        )
        if gemini_user_queries:
            insights["gemini_user_queries"] = gemini_user_queries
        if gemini_parent_highlights:
            insights["relational_parent_delta_highlights"] = insights.get("parent_delta_highlights", [])
            insights["parent_delta_highlights"] = gemini_parent_highlights
        if gemini_sibling_highlights:
            insights["relational_sibling_contrasts"] = insights.get("sibling_contrasts", [])
            insights["sibling_contrasts"] = gemini_sibling_highlights
        gemini_analysis = {
            "response_text": gemini_text,
            "sections": gemini_structured["sections"],
            "key_value_pairs": gemini_structured["key_value_pairs"],
            "key_value_map": gemini_structured["key_value_map"],
            "flat_key_value_map": gemini_structured["flat_key_value_map"],
            "coverage_map": gemini_coverage_map,
            "validation_checks": gemini_validation_checks,
            "coverage_gaps": uncovered_cats,
        }
        parent_deltas_struct = {
            "categorical": relational_context.get("category_deltas_vs_parent"),
            "numeric": relational_context.get("numeric_deltas_vs_parent"),
        }
        if gemini_parent_highlights:
            parent_deltas_struct["gemini_highlights"] = gemini_parent_highlights
        sibling_deltas_struct = {
            "relational": relational_context.get("sibling_deltas"),
        }
        if gemini_sibling_highlights:
            sibling_deltas_struct["gemini_highlights"] = gemini_sibling_highlights
        entry_details = _json_safe(
            {
                "prefix": prefix,
                "level": level,
                "group_size": record["group_size"],
                "coverage_pct": record.get("coverage_pct"),
                "record_digest": _record_digest(record),
                "queries": queries,
                "gemini_analysis": gemini_analysis,
                "insights": insights,
                "parent_deltas": parent_deltas_struct,
                "sibling_deltas": sibling_deltas_struct,
                "sample_names": names,
                "sample_image_urls": image_urls,
                "sample_items": sample_items,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        # Append this entry as a single JSONL line — durable immediately.
        # If the process is killed after this line, the entry survives and will
        # be skipped on the next run via existing_prefixes.
        _append_entry(OUTPUT_STREAM_PATH, prefix, entry_details)
        existing_prefixes.add(prefix)
        processed += 1
        print(
            f"[{processed}] Stored prefix {prefix} (level {level}, {record['group_size']} items)"
        )

elapsed = datetime.utcnow() - start_time
print(
    f"Completed batch generation: {processed} prefixes stored, {skipped} skipped.",
    f" Elapsed {elapsed}. Output file: {OUTPUT_STREAM_PATH}",
)
