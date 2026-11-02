"""Rewrite benchmark queries that name an audience their cluster does not contain.

Stage 1 occasionally writes "sandals for girls" into an 87%-men cluster. Such a query
describes items the cluster does not hold, so no retrieval system can ever satisfy it —
the defect is in the query text, not in the items.

Offending queries are rewritten by the same Qwen3-8B backend that authored them, with the
cluster's true gender mix stated in the prompt. The edit is applied to the **stage-1
artifact**, which is the single source of truth downstream stages derive from, and is
recorded on the record under `audience_repairs`. That way a later stage-2 re-run cannot
silently revert it, and the change stays auditable. Attributes for the rewritten text are
filled by re-running stage 2 afterwards.

Usage:
  python repair_audience_queries.py --dry-run     # list offenders only
  python repair_audience_queries.py               # rewrite and save
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fashion_rq_common import (  # noqa: E402
    AUDIENCE_TERMS,
    PARQUET_PATH,
    absent_audiences,
    gender_shares,
)
from qwen_local_client import generate as _qwen_generate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCHMARK = Path(
    os.environ.get(
        "BENCH_OUT", str(REPO_ROOT / "benchmark" / "queries" / "cluster_queries_raw.jsonl")
    )
)

# Every place the stage-1 record keeps the query list. Stage 2 reads the first; the rest
# are kept in step so the artifact does not carry two versions of the same query. Note the
# shapes differ -- key_value_map nests the list under "values", and insights sits outside
# gemini_analysis -- so each view is addressed by its full path.
QUERY_VIEWS = (
    ("gemini_analysis", "flat_key_value_map", "user_queries"),
    ("gemini_analysis", "key_value_map", "user_queries", "values"),
    ("gemini_analysis", "sections", "user_queries"),
    ("insights", "gemini_user_queries"),
)
_AUDIENCE_WORDS = {t: label for label, terms in AUDIENCE_TERMS.items() for t in terms}
_AUDIENCE_WORDS.update({"kids": "kids", "kid's": "kids", "children": "kids", "child": "kids"})
_WORD_RE = re.compile(r"[a-z']+")

REPAIR_PROMPT = """You are repairing one search query in a fashion-catalog benchmark.

The cluster {prefix} (level {level}, {size} items) has this gender mix:
{mix}

A query for this cluster may only name an audience the cluster actually contains, and
"unisex" is always acceptable. The query below names {bad}, which this cluster does not
contain, so the query describes items that are not there.

Other queries already written for this cluster (yours must stay distinct from them):
{others}

Rewrite the query so it keeps the same product intent and styling but names only
audiences this cluster contains, or names no audience at all. Keep it a conversational
first-person sentence of 8-14 words, like a shopper talking to a search assistant.

Original query: {original}

Return ONLY a JSON object: {{"natural_language_query": "<rewrite>"}}"""


def load_rows(path: Path) -> List[dict]:
    """Load the stage-1 JSONL as a list of one-key {prefix: entry} objects."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve(entry: dict, path: Tuple[str, ...]) -> Optional[List[str]]:
    """The list at `path`, or None when that view is absent or shaped differently."""
    node: Any = entry
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, list) else None


def get_queries(entry: dict) -> List[str]:
    return list(resolve(entry, QUERY_VIEWS[0]) or [])


def set_queries(entry: dict, queries: List[str]) -> List[str]:
    """Write the list into every view of matching length; return the views written."""
    written = []
    for path in QUERY_VIEWS:
        target = resolve(entry, path)
        if target is None:
            continue
        if len(target) != len(queries):
            written.append("!.".join(path) + f" (length {len(target)} != {len(queries)})")
            continue
        for idx, text in enumerate(queries):
            target[idx] = text
        written.append(".".join(path))
    return written


def is_provenance(path: str) -> bool:
    """
    Whether `path` holds a record of what the model said rather than the query list.

    `response_text` is the model's verbatim response and `validation_checks` are its own
    self-check strings. Both quote the queries as authored, so repairing them would mean
    fabricating model output; they are left as-is and reported separately. They are inert
    -- stage 2 reads only flat_key_value_map.user_queries, and stage 3 reads the enriched
    file -- so the artifact does not actually carry two live versions of a query.
    """
    return path.startswith("audience_repairs") or path.endswith("response_text") \
        or ".validation_checks" in path or path.endswith("validation_checks")


def stale_leaves(entry: dict, superseded: List[str]) -> Tuple[List[str], List[str]]:
    """
    (live stale paths, provenance paths) whose strings still contain a superseded query.

    Substring, not equality: a view can embed the query in a longer sentence, and a view
    that is missed here would otherwise keep the contradiction silently -- which is how
    key_value_map's `values` list and insights were missed on the first pass.
    """
    found: List[str] = []
    provenance: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                walk(value, f"{path}[{idx}]")
        elif isinstance(node, str):
            if any(old in node for old in superseded):
                (provenance if is_provenance(path) else found).append(path)

    walk(entry, "")
    return found, provenance


def apply_recorded_repairs(entry: dict) -> Tuple[int, List[str], List[str], List[str]]:
    """
    Re-apply this record's recorded repairs to every view; return (views updated, views
    written, live stale paths, provenance paths).

    Detection only sees the text as it stands, so once a rewrite has been applied the
    detector reports nothing left to do -- but a view that was missed on the first pass
    would stay stale forever. Driving the update from the recorded repairs instead makes
    this self-healing: run it again and any view added later is brought into line.
    """
    repairs = entry.get("audience_repairs") or []
    if not repairs:
        return 0, [], [], []
    queries = get_queries(entry)
    if not queries:
        return 0, [], [], []
    for repair in repairs:
        idx = repair.get("index")
        if isinstance(idx, int) and 0 <= idx < len(queries):
            queries[idx] = repair["new"]

    # Count views whose contents actually change, not queries changed: the first view is
    # normally already correct, so counting that way would report a no-op pass.
    updated = 0
    for path in QUERY_VIEWS:
        target = resolve(entry, path)
        if target is not None and len(target) == len(queries):
            if any(a != b for a, b in zip(target, queries)):
                updated += 1

    written = set_queries(entry, queries)
    stale, provenance = stale_leaves(entry, [r["old"] for r in repairs if r.get("old")])
    entry["audience_repair_note"] = (
        "response_text and validation_checks keep the pre-repair wording: they record what "
        "the model authored, not the live query list."
    )
    return updated, written, stale, provenance


def parse_rewrite(text: str) -> Optional[str]:
    """Pull natural_language_query out of a completion, tolerating fences/prose."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and payload.get("natural_language_query"):
            return str(payload["natural_language_query"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r'"natural_language_query"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"').strip()
        except json.JSONDecodeError:
            return match.group(1).strip()
    return None


def swap_audience(text: str, absent: List[str], target: str) -> str:
    """Last-resort deterministic repair: replace each offending audience word."""
    replacement = target
    def _sub(match: re.Match) -> str:
        word = match.group(0).lower()
        label = _AUDIENCE_WORDS.get(word)
        if label in absent:
            # keep the possessive/capitalisation shape of the original
            if match.group(0)[0].isupper():
                replacement_ = replacement.capitalize()
            else:
                replacement_ = replacement
            if word.endswith("'s"):
                replacement_ = f"{replacement_}'s"
            return replacement_
        return match.group(0)

    fixed = _WORD_RE.sub(_sub, text)
    return re.sub(r"\s{2,}", " ", fixed).strip()


def rewrite_query(shares: Dict[str, float], original: str, absent: List[str],
                  others: List[str], context: dict, max_attempts: int) -> Tuple[Optional[str], str]:
    """Return (new_query, how) where how is 'model', 'fallback' or 'failed'."""
    mix = ", ".join(f"{g} {100*s:.0f}%" for g, s in sorted(shares.items(), key=lambda kv: -kv[1]))
    other_block = "\n".join(f"- {q}" for q in others[:12]) or "- (none)"
    prompt = REPAIR_PROMPT.format(
        prefix=context["prefix"], level=context["level"], size=context["size"], mix=mix,
        bad=" and ".join(absent), others=other_block, original=f'"{original}"',
    )
    for _ in range(max_attempts):
        completion = _qwen_generate([prompt], max_new_tokens=160, temperature=0.3,
                                    top_p=0.9, batch_size=1)[0]
        candidate = parse_rewrite(completion)
        if not candidate or candidate.strip('"').strip() == original.strip():
            continue
        if absent_audiences(shares, candidate):
            continue
        if candidate.lower() in {q.lower() for q in others}:
            continue
        return candidate, "model"

    # Deterministic last resort so the artifact cannot keep a contradiction.
    target = "unisex"
    if shares:
        dom = max(shares, key=shares.get)
        if dom != "unisex":
            target = dom
    candidate = swap_audience(original, absent, target)
    if not absent_audiences(shares, candidate) and candidate.lower() not in {q.lower() for q in others}:
        return candidate, "fallback"
    return None, "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK,
                        help="Stage-1 JSONL artifact to repair in place")
    parser.add_argument("--parquet", type=Path, default=PARQUET_PATH)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Detect and report only")
    args = parser.parse_args()

    if not args.benchmark.exists():
        raise SystemExit(f"benchmark not found: {args.benchmark}")

    print(f"Loading catalog from {args.parquet} …")
    tokens = pl.col("semantic_id").str.extract_all(r"<\|sid_\d+\|>")
    df = (
        pl.read_parquet(args.parquet, columns=["semantic_id", "gender"])
        .with_columns(
            [
                tokens.list.slice(0, n).list.join("").alias(f"prefix_{n}tok")
                for n in (1, 2, 3)
            ]
        )
    )

    print(f"Loading stage-1 artifact from {args.benchmark} …")
    rows = load_rows(args.benchmark)
    print(f"  {len(rows):,} prefix records")

    # Push any repairs already recorded on the artifact out to every query view before
    # looking for new offenders -- a view missed on an earlier pass would otherwise stay
    # stale forever, because the detector sees only the text it already fixed.
    print("\nreconciling recorded repairs into every query view …")
    reconciled, mismatched, stale_all, prov_all = 0, [], [], []
    for row in rows:
        prefix, entry = next(iter(row.items()))
        changed, written, stale, provenance = apply_recorded_repairs(entry)
        reconciled += changed
        mismatched += [(prefix, w) for w in written if w.startswith("!")]
        stale_all += [(prefix, path) for path in stale]
        prov_all += provenance
    print(f"  {reconciled} views brought into line with the recorded repairs")
    for prefix, note in mismatched:
        print(f"  [VIEW] {prefix}: {note}")
    for prefix, path in stale_all:
        print(f"  [STALE] {prefix}: superseded text still in {path}")
    if prov_all:
        by_path: Dict[str, int] = {}
        for path in prov_all:
            if path.startswith("audience_repairs"):
                key = "audience_repairs (the change record)"
            elif ".validation_checks" in path:
                key = "validation_checks"
            else:
                key = "response_text"
            by_path[key] = by_path.get(key, 0) + 1
        summary = ", ".join(f"{k} ({v})" for k, v in sorted(by_path.items()))
        print(f"  provenance kept as authored (not query views): {summary}")
    if not mismatched and not stale_all:
        print("  all live views agree; no superseded text left in any query list")

    repairs = []  # (row, prefix, level, size, shares, index, text, absent)
    for row in rows:
        prefix, entry = next(iter(row.items()))
        level = str(entry.get("level") or len(re.findall(r"<\|sid_\d+\|>", prefix)))
        queries = get_queries(entry)
        if not queries:
            continue
        shares = gender_shares(df, prefix, int(level))
        for idx, text in enumerate(queries):
            absent = absent_audiences(shares, text)
            if absent:
                repairs.append((row, prefix, int(level), entry.get("group_size"), shares, idx, text, absent))

    print(f"\noffending queries: {len(repairs)} across {len({r[1] for r in repairs})} clusters")
    for _, prefix, level, size, shares, idx, text, absent in repairs:
        mix = ", ".join(f"{g} {100*s:.0f}%" for g, s in sorted(shares.items(), key=lambda kv: -kv[1]))
        print(f"  L{level} {prefix} (n={size}) [{mix}] #{idx} names {absent}")
        print(f"      {text}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    if not repairs and not reconciled:
        return

    # Apply per cluster so each rewrite can see its cluster's other queries.
    by_prefix: Dict[str, List[tuple]] = {}
    for item in repairs:
        by_prefix.setdefault(item[1], []).append(item)

    applied = failed = 0
    for prefix, items in by_prefix.items():
        row, _, level, size, shares, _, _, _ = items[0]
        entry = next(iter(row.values()))
        queries = get_queries(entry)
        others = [q for q in queries]
        record_repairs = []
        for item in items:
            _, _, _, _, _, idx, old, absent = item
            new, how = rewrite_query(
                shares, old, absent, [q for q in others if q != old],
                {"prefix": prefix, "level": level, "size": size}, args.max_attempts,
            )
            if not new:
                print(f"  [FAILED] {prefix} #{idx}: no valid rewrite found; left unchanged")
                failed += 1
                continue
            queries[idx] = new
            record_repairs.append({"index": idx, "absent": absent, "old": old, "new": new, "how": how})
            applied += 1
            print(f"  [{how}] {prefix} #{idx}")
            print(f"      old: {old}")
            print(f"      new: {new}")
        if record_repairs:
            set_queries(entry, queries)
            entry.setdefault("audience_repairs", []).extend(record_repairs)
            entry["audience_repairs_updated_at"] = datetime.now(timezone.utc).isoformat()

    if applied or reconciled:
        backup = args.benchmark.with_suffix(args.benchmark.suffix + ".pre_audience_repair")
        if not backup.exists():
            backup.write_bytes(args.benchmark.read_bytes())
        with args.benchmark.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\napplied {applied} rewrites ({failed} failed), reconciled {reconciled} "
              f"into views; backup at {backup}")
        if stale_all or mismatched:
            raise SystemExit("STALE: some view still holds superseded text — see [STALE]/[VIEW] above")
        print("Re-run stage 2 (to fill attributes for the new text) and stage 3 next.")
    else:
        print(f"\nno rewrites applied ({failed} failed)")


if __name__ == "__main__":
    main()
