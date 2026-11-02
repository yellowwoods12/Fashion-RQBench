"""
Enriches a benchmark JSON with per-query structured attributes extracted by an LLM.

Queries are sourced exclusively from gemini_analysis.flat_key_value_map.user_queries
in each record (that key name is legacy: the stage-1 generator now uses Qwen3-8B,
but the emitted artifact schema is unchanged). Qwen3-8B extracts product attributes
(product_type, gender, color, etc.) and stores them under record["enriched_queries"]
as a uniform list of {natural_language_query, attributes} dicts.

Heavy/redundant fields (gemini_analysis, raw_response, analysis_steps, etc.) are
stripped from the output to keep the enriched file compact.

Usage:
    python enrich_query_attributes.py \
        --benchmark benchmark/queries/cluster_queries_raw.jsonl.gz \
        --output    benchmark/queries/cluster_queries_enriched.json \
        [--batch-size 16] \
        [--max-new-tokens 320] \
        [--cache .query_attr_cache_qwen3.json]

The benchmark input may be plain or gzip-compressed.
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qwen_local_client import generate as _qwen_generate

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


# ── Qwen3 extraction prompt ───────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are extracting structured product attributes from a user's shopping query.
Return ONLY a valid JSON object — no explanation, no markdown fences.

Fields:
- "product_type": string — primary product type (e.g. "shirt", "dress", "footmuff"). REQUIRED.
- "gender": "men" | "women" | "boys" | "girls" | "unisex" | null
- "age_group": "baby" | "toddler" | "kids" | "adult" | null
- "color": list of color strings | null          (e.g. ["blue", "white"])
- "pattern": list of pattern strings | null      (e.g. ["striped", "floral"])
- "material": list of material strings | null    (e.g. ["cotton", "wool"])
- "occasion": list of occasion/use-case strings | null  (e.g. ["casual", "outdoor", "gym"])
- "brand": string | null                         (only if explicitly named in the query, carefully look for this)
- "price_hint": "budget" | "mid-range" | "premium" | null
- "style_tags": list of style descriptor strings | null  (e.g. ["slim fit", "oversized"])

Rules:
- Use null for any attribute not mentioned or clearly implied by the query.
- Lower-case all string values.
- For product_type, use the most specific product name mentioned (e.g. "running shoes", not "shoes").""".strip()


# ── JSON parsing helpers ───────────────────────────────────────────────────────

def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        start = cleaned.find("```")
        end = cleaned.rfind("```")
        if end > start:
            cleaned = cleaned[start + 3 : end].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
    return cleaned


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_markdown_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if repair_json:
            try:
                return json.loads(repair_json(cleaned))
            except Exception:
                pass
    return None


# ── Qwen3 batch extraction ─────────────────────────────────────────────────────

def _query_prompt(query: str) -> str:
    return f'Extract product attributes from this shopping query:\n\n"{query}"'


def _extract_batch(
    queries: List[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float = 0.1,
    top_p: float = 0.8,
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """Return (query, attributes) for each query; attributes is None on failure."""
    try:
        completions = _qwen_generate(
            [_query_prompt(q) for q in queries],
            system=EXTRACTION_SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
    except Exception as exc:
        print(f"  [WARN] Qwen batch of {len(queries)} failed: {exc}")
        return [(q, None) for q in queries]

    results: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for query, text in zip(queries, completions):
        attrs = _parse_json(text) if text and text.strip() else None
        if attrs is None:
            print(f"  [WARN] unparseable Qwen output for query {query!r:.80}")
        results.append((query, attrs))
    return results


# ── Fields stripped from output records ───────────────────────────────────────

_STRIP_FIELDS = {
    "gemini_analysis", "raw_response", "analysis_steps",
    "parent_deltas", "sibling_deltas", "record_digest",
    "sample_names", "sample_image_urls", "updated_at",
    "used_queries", "insights",
}


# ── Query extraction from benchmark records ────────────────────────────────────

def _iter_query_texts(record: Dict[str, Any]):
    """Yield user query strings from gemini_analysis.flat_key_value_map.user_queries."""
    fkv = (record.get("gemini_analysis") or {}).get("flat_key_value_map") or {}
    for q in fkv.get("user_queries") or []:
        if isinstance(q, str) and q.strip():
            yield q.strip()


# ── Write attributes back into benchmark ──────────────────────────────────────

def enrich_record(record: Dict[str, Any], cache: Dict[str, Dict]) -> int:
    """Populate enriched_queries and strip clutter fields. Returns count of enriched queries."""
    seen: set = set()
    enriched_queries: List[Dict] = []

    for text in _iter_query_texts(record):
        if text not in seen and text in cache:
            seen.add(text)
            enriched_queries.append({
                "natural_language_query": text,
                "attributes": cache[text],
            })

    record["enriched_queries"] = enriched_queries
    for field in _STRIP_FIELDS:
        record.pop(field, None)
    return len(enriched_queries)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enrich benchmark JSON with per-query structured attributes via Qwen3-8B."
    )
    parser.add_argument("--benchmark",   required=True, help="Input benchmark JSON path")
    parser.add_argument("--output",      required=True, help="Output enriched benchmark JSON path")
    parser.add_argument("--batch-size",  type=int, default=16,
                        help="Queries pushed through Qwen per generate() call")
    parser.add_argument("--max-new-tokens", type=int, default=320,
                        help="Token budget per attribute-extraction completion")
    parser.add_argument("--cache",       default=".query_attr_cache_qwen3.json",
                        help="Cache file for extracted attributes (supports resume)")
    args = parser.parse_args()

    # ── Load benchmark ────────────────────────────────────────────────────────
    print(f"Loading benchmark from {args.benchmark} …")
    _opener = gzip.open if str(args.benchmark).endswith(".gz") else open
    with _opener(args.benchmark, "rt", encoding="utf-8") as f:
        raw_text = f.read()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        # Resumable pipelines (e.g. benchmark_creation_unfiltered_public_dataset*.py)
        # append one JSON object per line — {prefix: record} — instead of a single
        # JSON document. Merge those lines into one dict.
        raw = {}
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            raw.update(json.loads(line))

    # Support both a bare dict {prefix: record} and a wrapped {"results": [...]} format
    if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list):
        benchmark_meta = {k: v for k, v in raw.items() if k != "results"}
        records: List[Dict] = raw["results"]
    elif isinstance(raw, list):
        benchmark_meta = {}
        records = raw
    else:
        benchmark_meta = {}
        records = [v for v in raw.values() if isinstance(v, dict)]
    print(f"  {len(records):,} prefix records loaded.")

    # ── Load cache ────────────────────────────────────────────────────────────
    cache_path = Path(args.cache)
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Cache hit: {len(cache):,} queries already attributed.")

    # ── Resume from existing output if present ────────────────────────────────
    # Only the extraction cache is resumed from. The output is a derived artifact —
    # the heavy source fields (gemini_analysis, which is where the queries live) are
    # stripped from it — so adopting its records wholesale would leave this stage
    # with nothing to attribute. Records always come from --benchmark.
    out_path = Path(args.output)
    if out_path.exists():
        print(f"  Output file found — reusing its attributes via the cache …")
        with out_path.open(encoding="utf-8") as f:
            existing_output = json.load(f)
        if isinstance(existing_output, dict) and "results" in existing_output and isinstance(existing_output["results"], list):
            existing_records = existing_output["results"]
        elif isinstance(existing_output, list):
            existing_records = existing_output
        else:
            existing_records = [v for v in existing_output.values() if isinstance(v, dict)]
        print(f"  {len(existing_records):,} records loaded from existing output.")

        # Pre-populate cache from already-enriched queries so Qwen is only
        # called for queries that have no attributes yet.
        pre_loaded = 0
        for record in existing_records:
            for eq in record.get("enriched_queries") or []:
                if isinstance(eq, dict):
                    text = (eq.get("natural_language_query") or "").strip()
                    attrs = eq.get("attributes")
                    if text and attrs and text not in cache:
                        cache[text] = attrs
                        pre_loaded += 1
        print(f"  Pre-loaded {pre_loaded:,} attributes from existing output into cache.")

    # ── Collect and deduplicate all queries ───────────────────────────────────
    all_queries: List[str] = []
    seen_q: set = set()
    for record in records:
        for text in _iter_query_texts(record):
            if text not in seen_q:
                seen_q.add(text)
                all_queries.append(text)
    missing = [q for q in all_queries if q not in cache]
    print(f"  Total unique queries: {len(all_queries):,} | To extract: {len(missing):,}")

    # ── Batch extract with Qwen3 ──────────────────────────────────────────────
    if missing:
        processed = errors = 0
        batches_since_save = 0
        for start in range(0, len(missing), args.batch_size):
            chunk = missing[start : start + args.batch_size]
            for q, attrs in _extract_batch(
                chunk,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            ):
                if attrs is not None:
                    cache[q] = attrs
                else:
                    errors += 1
                processed += 1

            if processed % 160 == 0 or processed == len(missing):
                print(f"  Extracted {processed}/{len(missing)} "
                      f"(errors: {errors}, cached: {len(cache)}) …")

            batches_since_save += 1
            if batches_since_save >= 10:
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
                batches_since_save = 0

        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"  Cache saved → {cache_path} ({len(cache):,} entries)")

    # ── Write attributes back into benchmark records ──────────────────────────
    total_enriched = 0
    for record in records:
        total_enriched += enrich_record(record, cache)

    print(f"  Enriched {total_enriched:,} query entries across {len(records):,} records.")

    # ── Save enriched benchmark (preserve original wrapper structure) ──────────
    if benchmark_meta:
        output_data = {**benchmark_meta, "results": records}
    else:
        output_data = records
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved enriched benchmark → {out_path}")


if __name__ == "__main__":
    main()
