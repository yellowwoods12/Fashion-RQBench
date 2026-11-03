#!/usr/bin/env python3
"""
Check that this release is intact, and that the shipped benchmark matches the
counts published with this release.

Runs without a GPU and without the data files for the parts that do not need
them, so it works as a post-clone smoke test and as a release gate.

Checks
  artifacts  every shipped .gz opens, and each task file's record count and the
             query corpus's statistics match the published figures
  schema     task records carry the fields the scorers expect
  data       if data/ is populated: the catalog parquet has the expected row
             count, the SID columns are present and well-formed, and the
             checksums match the pinned manifest in scripts/download_data.py
  hygiene    no absolute local paths leaked into the source

Usage:
  python scripts/validate_release.py
  python scripts/validate_release.py --strict     # treat warnings as failures
"""

import argparse
import ast
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from download_data import FILES, sha256  # noqa: E402

# Published figures for the Qwen3-8B release (see docs/BENCHMARK.md §3)
EXPECTED = {
    "clusters": 588,
    "levels": {"1": 17, "2": 135, "3": 436},
    "queries": 5807,
    "unique_queries": 5752,
    "task1": 4103,
    "task2": 32688,
    "task3": 3262,
    "task4": 416,
}
TASK_SCHEMA = {
    "task1": {"query_id", "cluster_prefix", "query", "candidates"},
    "task2": {"pair_id", "query_id", "query", "item_id", "label"},
    "task3": {"query_id", "cluster_prefix", "query", "correct_item_ids", "candidates"},
    "task4": {"group_id", "cluster_prefix", "queries", "candidates"},
}
# the field holding each task's ground truth
TASK_TRUTH = {"task1": "candidates", "task2": "label",
              "task3": "correct_item_ids", "task4": "queries"}

problems: List[str] = []
warnings: List[str] = []


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def bad(msg: str) -> None:
    problems.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  WARN  {msg}")


def read_gz_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_artifacts() -> None:
    print("\n[artifacts]")
    queries_dir, tasks_dir = REPO_ROOT / "benchmark/queries", REPO_ROOT / "benchmark/tasks"

    enr_path = queries_dir / "cluster_queries_enriched.json.gz"
    if not enr_path.exists():
        bad(f"missing {enr_path.relative_to(REPO_ROOT)}")
        return
    with gzip.open(enr_path, "rt", encoding="utf-8") as fh:
        enriched = json.load(fh)
    queries = [q for r in enriched for q in (r.get("enriched_queries") or [])]
    levels = {}
    for r in enriched:
        levels[r.get("level")] = levels.get(r.get("level"), 0) + 1

    for label, got, want in [
        ("query corpus records", len(enriched), EXPECTED["clusters"]),
        ("query corpus queries", len(queries), EXPECTED["queries"]),
        ("unique query strings", len({q["natural_language_query"] for q in queries}),
         EXPECTED["unique_queries"]),
    ]:
        ok(f"{label}: {got}") if got == want else bad(f"{label}: {got} (expected {want})")
    if levels == EXPECTED["levels"]:
        ok(f"clusters per level: {levels}")
    else:
        bad(f"clusters per level: {levels} (expected {EXPECTED['levels']})")

    missing_attr = [q for q in queries if not isinstance(q.get("attributes"), dict)
                    or not any(q["attributes"].values())]
    ok("every query carries attributes") if not missing_attr else \
        bad(f"{len(missing_attr)} queries have no filled attributes")

    raw_path = queries_dir / "cluster_queries_raw.jsonl.gz"
    if raw_path.exists():
        n = sum(1 for _ in read_gz_jsonl(raw_path))
        ok(f"raw generation stream: {n} records") if n == EXPECTED["clusters"] else \
            bad(f"raw generation stream: {n} records (expected {EXPECTED['clusters']})")

    for task, want in (("task1", EXPECTED["task1"]), ("task2", EXPECTED["task2"]),
                       ("task3", EXPECTED["task3"]), ("task4", EXPECTED["task4"])):
        path = next(tasks_dir.glob(f"{task}_*.jsonl.gz"), None)
        if path is None:
            bad(f"missing {task} file")
            continue
        records = list(read_gz_jsonl(path))
        ok(f"{path.name}: {len(records)} records") if len(records) == want else \
            bad(f"{path.name}: {len(records)} records (expected {want})")
        if records:
            extra = TASK_SCHEMA[task] - set(records[0])
            if extra:
                bad(f"{path.name}: record missing fields {sorted(extra)}")
            else:
                ok(f"{path.name}: record schema complete")
            # ground truth must be present: every record carries its relevance labels
            gt_field = TASK_TRUTH[task]
            empty = sum(1 for r in records if not r.get(gt_field))
            ok(f"{path.name}: ground truth present on all records") if not empty else \
                bad(f"{path.name}: {empty} records lack {gt_field}")


def check_data() -> None:
    print("\n[data]")
    for key in ("sids", "embeddings", "checkpoint"):
        spec = FILES[key]
        path = REPO_ROOT / spec["dest"]
        if not path.exists():
            warn(f"{spec['dest']} not present — run scripts/download_data.py --only {key}")
            continue
        actual = sha256(path)
        ok(f"{path.name} checksum {actual[:12]}…") if actual == spec["sha256"] else \
            bad(f"{path.name} checksum {actual[:12]}… (expected {spec['sha256'][:12]}…)")

    sids = REPO_ROOT / FILES["sids"]["dest"]
    if not sids.exists():
        return
    try:
        import polars as pl
    except ImportError:
        warn("polars not installed — skipping parquet schema checks")
        return

    df = pl.read_parquet(sids)
    ok(f"catalog: {df.height} items") if df.height == 44446 else \
        bad(f"catalog: {df.height} items (expected 44446)")
    for col in ("sid_0", "sid_1", "sid_2", "semantic_id"):
        if col not in df.columns:
            bad(f"catalog missing column {col}")

    # semantic_id must agree with the three code columns
    rebuilt = pl.format("<|sid_{}|><|sid_{}|><|sid_{}|>",
                        pl.col("sid_0"), pl.col("sid_1"), pl.col("sid_2"))
    mismatched = df.select(
        (rebuilt != pl.col("semantic_id")).sum().alias("n")
    ).item()
    ok("semantic_id matches sid_0/sid_1/sid_2 for every item") if mismatched == 0 else \
        bad(f"{mismatched} items where semantic_id disagrees with the code columns")

    n_leaves = df["semantic_id"].n_unique()
    print(f"  INFO  distinct leaf clusters: {n_leaves}, "
          f"level-1: {df['sid_0'].n_unique()}, level-2: {df.select(['sid_0','sid_1']).n_unique()}")


def check_hygiene() -> None:
    print("\n[hygiene]")
    pat = re.compile(r"/(?:home|Users|mnt)/[a-z_.-]+")
    hits = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if pat.search(line) and "pathlib" not in line:
                hits.append(f"{path.relative_to(REPO_ROOT)}:{i}")
    ok("no absolute local paths in src/") if not hits else \
        bad(f"absolute local paths remain: {hits[:6]}")

    unparseable = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            unparseable.append(f"{path.name}: {exc}")
    ok("every python file in src/ parses") if not unparseable else \
        bad(f"unparseable: {unparseable}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = ap.parse_args()

    print(f"Validating release at {REPO_ROOT}")
    check_artifacts()
    check_data()
    check_hygiene()

    print()
    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    if warnings:
        print(f"OK with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
        return 1 if args.strict else 0
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
