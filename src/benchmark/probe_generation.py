"""Probe the retuned prompt on a few clusters through the real stage-1 code path.

Executes the stage-1 script up to the batch loop (no side effects), then composes
and generates for a size-spread sample of prefixes, reporting query counts,
word counts and uniqueness — the metrics the target table pins down.

Usage:  python probe_generation.py [n_prefixes]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRIPT = Path(__file__).resolve().parent / "generate_cluster_queries.py"
MARKER = "existing_entries = _load_existing_entries(OUTPUT_STREAM_PATH)"

head, sep, _ = SCRIPT.read_text().partition(MARKER)
if not sep:
    raise SystemExit("marker not found")

ns = {"__name__": "__probe__", "__file__": str(SCRIPT)}
exec(compile(head, str(SCRIPT), "exec"), ns)

import statistics

df, pl = ns["df"], ns["pl"]
_CAT = ["masterCategory", "subCategory", "articleType", "brandName", "baseColour", "gender", "usage"]
_NUM = ["price", "discountedPrice"]
_TXT = ["productDisplayName", "description"]

print(f"TARGET_QUERY_RANGE={ns['TARGET_QUERY_RANGE']} max={ns['TARGET_QUERY_MAX']} "
      f"max_new_tokens={ns['QWEN_MAX_NEW_TOKENS']} yarn={os.environ.get('QWEN_YARN_FACTOR')}", flush=True)

# 4 prefixes per level spanning the size distribution, so the sample crosses
# product families instead of landing on four T-shirt clusters.
picks = []
for level in sorted(ns["LEVEL_CONFIG"]):
    candidates = ns["_top_prefixes_for_level"](df, level, None, ns["MIN_GROUP_SIZE"])
    col = f"prefix_{level}tok"
    sizes = {p: df.filter(pl.col(col) == p).height for p in candidates}
    ordered = sorted(candidates, key=lambda p: sizes[p])
    for frac in (0.99, 0.7, 0.45, 0.2):
        picks.append((level, ordered[min(len(ordered) - 1, int(frac * len(ordered)))], sizes))

RANGES = os.environ.get("PROBE_RANGES", ns["TARGET_QUERY_RANGE"]).split(",")
print(f"probing ranges: {RANGES} over {len(picks)} prefixes", flush=True)

# Resolve each prefix to (record, relational_context) once, reuse across ranges.
cases = []
for level, prefix, sizes in picks:
    col = f"prefix_{level}tok"
    if ns["_cluster_articletype_purity"](df, col, prefix) < ns["CLUSTER_PURITY_THRESHOLD"]:
        print(f"skip {prefix} (purity)")
        continue
    record = ns["build_prefix_workflow_record"](
        df, prefix, categorical_cols=_CAT, numeric_cols=_NUM, text_cols=_TXT,
        top_k=20, min_group_size=ns["MIN_GROUP_SIZE"],
    )
    if not record:
        print(f"skip {prefix} (no record)")
        continue
    rctx = ns["build_relational_context"](
        df, record, categorical_cols=_CAT, numeric_cols=_NUM, text_cols=_TXT,
        top_k=20, sibling_limit=3,
    )
    cases.append((level, prefix, sizes[prefix], record, rctx))

for rng in RANGES:
    ns["TARGET_QUERY_RANGE"] = rng.strip()
    ns["TARGET_QUERY_MAX"] = int(rng.strip().split("-")[-1])
    rows = []
    print("\n" + "#" * 78)
    print(f"# RANGE {ns['TARGET_QUERY_RANGE']}  (cap {ns['TARGET_QUERY_MAX']})")
    print("#" * 78, flush=True)
    for level, prefix, size, record, rctx in cases:
        text = ns["call_llm_for_prefix"](record, rctx, df=df)
        q = ns["_parse_gemini_text_blob"](text)["user_queries"]
        words = [len(x.split()) for x in q]
        print("=" * 78)
        print(f"L{level} {prefix} size={size} -> {len(q)} queries, "
              f"words mean={statistics.mean(words):.1f}" if words else
              f"L{level} {prefix} -> 0 queries", flush=True)
        for x in q:
            print(f"   [{len(x.split()):>2}w] {x}")
        rows.append((level, size, q, words))

    allq = [x for *_, q, _ in rows for x in q]
    allw = [w for *_, ws in rows for w in ws]
    counts = [len(q) for *_, q, _ in rows]
    if not counts:
        continue
    print("\n=== summary for range", ns["TARGET_QUERY_RANGE"], "===")
    print(f"clusters={len(counts)} queries={sum(counts)} per-cluster "
          f"{min(counts)}-{max(counts)} mean={statistics.mean(counts):.1f} (target 9.15)")
    print(f"words mean={statistics.mean(allw):.1f} range={min(allw)}-{max(allw)} (target 10.0, 3-26)")
    uniq = len(set(allq))
    print(f"unique={uniq}/{len(allq)} = {100*uniq/len(allq):.1f}% (target 98.7%)")
    within = sum(len(set(q)) for *_, q, _ in rows) / sum(len(q) for *_, q, _ in rows)
    print(f"within-cluster uniqueness={100*within:.1f}%")
    print(f"projected full-run queries: {statistics.mean(counts)*588:.0f} (target 5,380)")

