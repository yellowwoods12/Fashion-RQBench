"""
Enriches a benchmark JSON with per-query structured attributes extracted by an LLM.

Queries are sourced exclusively from gemini_analysis.flat_key_value_map.user_queries
in each record. Gemini extracts product attributes (product_type, gender, color, etc.)
and stores them under record["enriched_queries"] as a uniform list of
{natural_language_query, attributes} dicts.

Heavy/redundant fields (gemini_analysis, raw_response, analysis_steps, etc.) are
stripped from the output to keep the enriched file compact.

Usage:
    python enrich_benchmark_with_query_attributes.py \
        --benchmark ./results/evaluation_outputs/benchmark_54k_unfiltered.json \
        --output    ./results/semantic-id-llms/benchmark_suite/data/benchmark_54k_unfiltered_enriched_new.json \
        [--model gemini-2.5-flash] \
        [--concurrency 8] \
        [--cache .query_attr_cache.json]
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
from google import genai
from google.genai import types

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


# ── Gemini extraction prompt ───────────────────────────────────────────────────

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


# ── Gemini API call ────────────────────────────────────────────────────────────

def _call_gemini(client, model: str, query: str) -> Optional[Dict[str, Any]]:
    prompt = f'Extract product attributes from this shopping query:\n\n"{query}"'
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.1,
                top_p=0.8,
                response_mime_type="application/json",
            ),
        )
        text = getattr(resp, "text", "") or ""
        return _parse_json(text) if text.strip() else None
    except Exception as exc:
        print(f"  [WARN] Gemini error for query {query!r:.80}: {exc}")
        return None


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
        description="Enrich benchmark JSON with per-query structured attributes via Gemini."
    )
    parser.add_argument("--benchmark",   required=True, help="Input benchmark JSON path")
    parser.add_argument("--output",      required=True, help="Output enriched benchmark JSON path")
    parser.add_argument("--model",       default="gemini-2.5-flash", help="Gemini model ID")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel Gemini requests")
    parser.add_argument("--cache",       default=".query_attr_cache.json",
                        help="Cache file for extracted attributes (supports resume)")
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project="ailooks-non-prod-dcd1", location="global")

    # ── Load benchmark ────────────────────────────────────────────────────────
    print(f"Loading benchmark from {args.benchmark} …")
    with open(args.benchmark, encoding="utf-8") as f:
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
    out_path = Path(args.output)
    if out_path.exists():
        print(f"  Output file found — resuming from {out_path} …")
        with out_path.open(encoding="utf-8") as f:
            existing_output = json.load(f)
        if isinstance(existing_output, dict) and "results" in existing_output and isinstance(existing_output["results"], list):
            benchmark_meta = {k: v for k, v in existing_output.items() if k != "results"}
            records = existing_output["results"]
        elif isinstance(existing_output, list):
            records = existing_output
        else:
            records = [v for v in existing_output.values() if isinstance(v, dict)]
        print(f"  {len(records):,} records loaded from existing output.")

        # Pre-populate cache from already-enriched queries so Gemini is only
        # called for queries that failed (i.e. still have no attributes).
        pre_loaded = 0
        for record in records:
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

    # ── Batch extract with Gemini ─────────────────────────────────────────────
    if missing:
        def _task(q: str) -> Tuple[str, Optional[Dict]]:
            time.sleep(0.05)  # light throttle to stay within API rate limits
            return q, _call_gemini(client, args.model, q)

        processed = errors = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_task, q): q for q in missing}
            for future in as_completed(futures):
                q, attrs = future.result()
                if attrs is not None:
                    cache[q] = attrs
                else:
                    errors += 1
                processed += 1

                if processed % 50 == 0 or processed == len(missing):
                    print(f"  Extracted {processed}/{len(missing)} "
                          f"(errors: {errors}, cached: {len(cache)}) …")

                # Incremental cache save every 100 queries
                if processed % 100 == 0:
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(cache, f, indent=2)

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
