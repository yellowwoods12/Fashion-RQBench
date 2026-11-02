# Query generation, task construction and scoring

The pipeline that produced `benchmark/queries/` and `benchmark/tasks/`. Process narrative,
prompts and the corpus statistics are in [../../docs/BENCHMARK.md](../../docs/BENCHMARK.md);
task definitions are in [../../docs/TASKS.md](../../docs/TASKS.md). This file is the script
reference.

```
scripts/reproduce.sh   the whole pipeline, with the settings the release used
```

## Scripts

| Script | Stage | Purpose |
|---|---|---|
| `generate_cluster_queries.py` | 1 | walks the SID tree, builds a per-cluster attribute profile, prompts Qwen3-8B for natural-language queries; writes one JSONL line per cluster (resumable, skips prefixes already present) |
| `enrich_query_attributes.py` | 2 | second Qwen3-8B pass over every generated query, filling its `attributes` dict; cached by query text (resumable) |
| `benchmark_stats.py` | 2→ | corpus statistics vs. the target spec |
| `probe_generation.py` | 1 | runs stage 1 over a handful of size-spread clusters to measure a prompt/budget change in ~20 min instead of ~3 h 20 m |
| `repair_audience_queries.py` | 2→ | finds and rewrites queries that name an audience their cluster does not contain; reconciles recorded repairs across every view of the query list |
| `fashion_rq_common.py` | all | shared paths, SID-prefix helpers, candidate pools, relevance grades, metrics; `SHIPPED_TASKS_DIR` is what the scorers default to |
| `fashion_rq_task{1..4}_build_*.py` | 3 | build the four task files + ground truth (GPU-free, seeded, deterministic) |
| `fashion_rq_task{1..4}_compute_metrics.py` | — | score predictions against them |
| `qwen_local_client.py` | 1,2 | in-process Qwen3-8B batching client (thinking disabled, optional YaRN) |
| `legacy_gemini/` | — | the original Gemini-authored generator and enricher, verbatim, for reference — see its README |

## Stage contract

- **Stage 1** reads `data/items_with_sids.parquet` and appends `{prefix: record}` JSONL to
  `BENCH_OUT` (default `benchmark/queries/cluster_queries_raw.jsonl`). Re-running skips
  prefixes already present, so an interrupted run resumes by relaunching the same command.
  It writes **plain** JSONL — appending to gzip is not supported, so gzip afterwards if you
  want a compressed artifact.
- **Stage 2** reads that file (plain or `.gz`), writes the enriched corpus, and resumes from
  `--cache` only — never from its own output, which has the source fields stripped. It also
  pre-populates its cache from an existing output file, so re-running after a change to the
  *inputs* re-extracts only the queries whose text changed.
- **Stage 3** reads the enriched corpus and writes four task files. No GPU, no network,
  `seed=42` — same inputs, same files.

## Environment

| Variable | Default | Effect |
|---|---|---|
| `TARGET_QUERY_RANGE` | `9-10` (in `reproduce.sh`; `7-9` in the script) | queries requested per cluster. Qwen3-8B pins to the **ceiling** of the range, so widening it inflates the corpus: 9-10 → 9.9/cluster, 9-12 → 11.7. A floor of 8 collapses to exactly 8, apparently colliding with the prompt's "roughly 8-14 words" |
| `QWEN_MODEL_ID` | `Qwen/Qwen3-8B` | authoring model |
| `QWEN_YARN_FACTOR` | `4` (stage 1), `0` (stage 2) | RoPE scaling. Stage-1 prompts run 16.9k–55.6k tokens against a 40,960-token native window, so stage 1 needs it; stage 2 does not |
| `QWEN_MAX_NEW_TOKENS` | `4096` | stage-1 response budget |
| `QWEN_BATCH_SIZE` | `1` | stage-1 batch (prompts vary wildly in length; 1 avoids padding waste) |
| `HF_HUB_OFFLINE` | `1` | skips hub revalidation; without it, unauthenticated 429s stall ~30 s per metadata file |
| `SID_ITEMS_PARQUET` | `data/items_with_sids.parquet` | catalog path |
| `FASHION_RQ_ENRICHED_JSON` | `benchmark/queries/cluster_queries_enriched.json.gz` | corpus the task builders read |
| `FASHION_RQ_TASKS_DIR` | `benchmark/tasks_generated` | where task builders write |
| `BENCH_OUT` | `benchmark/queries/cluster_queries_raw.jsonl` | stage-1 output / stage-2 input |
| `SEMANTIC_EMBED_*`, `SEMANTIC_SIM_THRESHOLD` | MiniLM, `0.82` | in-cluster near-duplicate query removal. Optional: if `sentence-transformers` is not installed the step is skipped and a few more near-duplicates survive |
| `AUDIENCE_MIN_SHARE` | `0.05` | minimum share for a query to be allowed to name an audience (see `fashion_rq_common.absent_audiences`) |
| `PROBE_RANGES` | — | comma-separated ranges for `probe_generation.py` |

Two known traps, both hit during development and worth knowing before you edit the prompt:
Qwen3-8B over-delivers on quantity (it reads "one query per major product type" literally and
enumerates colours), and it does not sample a stated range but pins to one of its bounds.
Measure any prompt change with `probe_generation.py` on the largest clusters, not on a
convenient small one.
