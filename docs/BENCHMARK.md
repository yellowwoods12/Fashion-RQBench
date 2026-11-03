# How the benchmark was generated

This document describes the pipeline that produced the released corpus: a **local
Qwen3-8B** authored the queries from the semantic-ID hierarchy, replacing the hosted
Gemini used for the original Fashion-RQ suite. Task definitions, record schemas and
scoring live in [TASKS.md](TASKS.md); data provenance in [DATA_CARD.md](DATA_CARD.md);
the hierarchy being queried in [HIERARCHY_QUALITY.md](HIERARCHY_QUALITY.md).

```
src/benchmark/
  generate_cluster_queries.py            # stage 1: cluster analysis + query authoring
  enrich_query_attributes.py             # stage 2: per-query attribute extraction
  benchmark_stats.py                     # corpus statistics vs. the target spec
  fashion_rq_common.py                   # shared paths, candidate pools, relevance grades
  fashion_rq_task{1..4}_build_*.py       # stage 3: the four tasks (see TASKS.md)
  fashion_rq_task{1..4}_compute_metrics.py  # scoring (this doc, §5)
  qwen_local_client.py                   # in-process Qwen3-8B client (YaRN, thinking off)
  repair_audience_queries.py             # data-quality pass over audience mentions (§3)
  probe_generation.py                    # cheap prompt-budget probe before a full stage-1 run
benchmark/queries/
  cluster_queries_raw.jsonl.gz           # stage 1 output: per-cluster analysis + queries
  cluster_queries_enriched.json.gz       # stage 2 output: the query corpus with attributes
benchmark/tasks/                         # stage 3 output: 4 task files + ground truth
benchmark/stats.txt                      # stats report for the released corpus
data/                                    # catalog + SIDs, fetched by scripts/download_data.py
scripts/reproduce.sh                     # stages: data → 1 → 2 → 3 → validate
```

---

## 1. What the benchmark is

Fashion-RQ is built on the public Myntra/Kaggle fashion catalog, whose items already carry
RQ-VAE **semantic IDs** (SIDs) of the form `<|sid_a|><|sid_b|><|sid_c|>`. The SID codes
form a 3-level prefix tree, and each level-3 (leaf) cluster is a group of visually and
semantically coherent products. For every qualifying leaf cluster, an LLM writes
natural-language search queries that describe the cluster, and each query carries
structured attributes (gender, color, product type, …).

Relevance between a query's cluster and any catalog item is then read **directly off the
SID tree** — no human labeling:

| Relation    | Definition                                   | Grade (T1/T4) | Code (T2) |
|-------------|----------------------------------------------|---------------|-----------|
| Exact       | same level-3 prefix (the target cluster)     | 3             | E         |
| Sibling     | same level-2 prefix, different level-3       | 2             | S         |
| Related     | same level-1 prefix, different level-2       | 1             | R         |
| Irrelevant  | different level-1 prefix entirely            | 0             | I         |

All random sampling (candidate pools, distractor picks, candidate shuffling) uses
`random.Random(seed=42)`, so every task file is byte-for-byte reproducible.

## 2. Inputs

- **Catalog**: `data/items_with_sids.parquet` — 44,446 products, fetched by
  `scripts/download_data.py`. One row per product: `id`, `productDisplayName`,
  `brandName`, the taxonomy columns, `baseColour`, `gender`, `usage`, `price`,
  `imageURL`, and the SID columns (`sid_0`, `sid_1`, `sid_2`, `semantic_id`). The
  `embedding` column is not needed here and is not shipped in this file.
- **Model**: `Qwen/Qwen3-8B` from the local HF cache, run in-process on `cuda:0`,
  thinking disabled (`enable_thinking=False`), YaRN RoPE factor 4 (window 40,960 →
  163,840 tokens; stage-1 prompts run 16.9k–55.6k tokens, median 34.5k). Transformers
  5.x note: RoPE config key is `rope_parameters` (the old `rope_scaling` key raises
  `TypeError`). `HF_HUB_OFFLINE=1` avoids ~30 s 429 stalls per metadata file.

## 3. Query generation (stages 1–2)

### Stage 1 — cluster analysis + query authoring
`generate_cluster_queries.py` walks the SID tree:

1. Selects level-3 leaf clusters of **≥ 5 items** whose **articleType purity ≥ 0.8**
   (354 clusters were skipped by the purity filter; 588 stored: 17 / 135 / 436 at
   levels 1 / 2 / 3 — exactly the target spec).
2. For each cluster, builds a prompt containing the cluster's item inventory and asks
   for **9–10 queries** ("one query per major product type (>5% share); coverage comes
   from merging, not from adding more queries"). The code also truncates to the stated
   maximum as a postcondition, so the budget cannot be violated.
   Two quirks of Qwen3-8B set this budget (measured with `scripts/probe_generation.py`):
   without a stated maximum it returns 31 queries for the largest cluster, and with one
   it **pins to a bound of the stated range rather than sampling it** — 9–12 yields 11.7
   queries/cluster, 9–11 yields 10.2, 9–10 yields 9.8. A range whose floor is 8 collapses
   to exactly 8, apparently because it collides with the same prompt's "roughly 8–14
   words" instruction; 9 is therefore the lowest usable floor. 9–10 is the setting that
   lands closest to the target spec's 5,380 queries.
3. Emits one JSON line per prefix: the cluster record with `enriched_queries`.

Runtime: ~3 h 20 m on one L4 (`logs/stage1.log`).

### Stage 2 — per-query attribute extraction
`enrich_query_attributes.py` makes a second Qwen3-8B pass over
every generated query to fill its `attributes` dict (gender / color / product_type /
occasion / …), cached in `.query_attr_cache_qwen3.json` (11,774 entries). Zero queries
ended up without attributes.

Stage 2 resumes only its **cache**, never its records: the enriched output is a derived
artifact with the source fields (`gemini_analysis`) stripped, so re-reading it as input
yields zero queries and silently produces an empty benchmark.

### Resulting query corpus (vs. the Gemini-authored target)

| Stat                        | Qwen3-8B (this run) | Gemini (target spec) |
|-----------------------------|---------------------|----------------------|
| clusters                    | 588 (17/135/436)    | 588 (17/135/436)     |
| enriched queries            | 5,807               | 5,380                |
| queries per cluster         | 6–10, mean 9.9      | 2–12                 |
| unique query strings        | 5,752 (99.1%)       | 5,305 (98.7%)        |
| mean query length           | 11.2 words          | 10.0 words           |
| query length range          | 5–23 words          | 3–26 words           |

Full report: `benchmark/stats.txt`. Cluster counts and cluster-size distributions
match the target exactly. Known differences: total query volume is 7.9% above target and
query length 12% above, because Qwen3-8B pins to the ceiling of its stated budget (523 of
588 clusters land on 10) rather than spreading across it; the narrower 6–10 spread is a
subset of the spec's 2–12. Both are consequences of the same behaviour, and 9–10 is the
setting that comes closest — see §3 step 2.

### Audience consistency (repaired 2026-11-01)

A query carries the audience it is written for, so a query naming a gender its cluster does
not contain describes items that are not there and no retrieval system can satisfy. Stage 1
produced **5 such queries across 5 clusters** (0.09% of 5,827; 5 of 588 clusters):

| cluster | cluster gender mix | query |
|---|---|---|
| `<\|sid_23\|><\|sid_10\|>` | women 52%, men 38% | Do you have any free gifts that are watches **for girls**? |
| `<\|sid_17\|><\|sid_24\|><\|sid_36\|>` | men 50%, women 38% | Are there T-shirts **for girls** in a jersey fabric and casual style? |
| `<\|sid_17\|><\|sid_24\|><\|sid_60\|>` | women 50%, men 34% | Show me casual Tshirts **for girls** in kidswear styles |
| `<\|sid_13\|><\|sid_10\|><\|sid_6\|>` | men 87%, unisex 10% | Do you have casual sandals **for girls** with a unisex fit? |
| `<\|sid_13\|><\|sid_10\|><\|sid_0\|>` | unisex 52%, men 29% | Find me casual sandals **for girls** with a comfortable and stylish design. |

The rule applied: a query may name an audience only if that audience is ≥ 5%
(`AUDIENCE_MIN_SHARE`) of the cluster; `unisex` clusters also answer to `men` and `women`,
and `kids` is judged against boys + girls combined. Three things followed from this, all in
`fashion_rq_common.py`:

- **Prompt rule 10** in `compose_gemini_prompt` tells the author to check
  `gender` in `full_attribute_distribution` before naming an audience, and the prompt is
  built from the catalog — so the fix prevents recurrence rather than only patching output.
- **A `[GENDER]` postcondition** in stage 1 prints any query naming an absent audience, so
  the next regeneration reports it instead of shipping it.
- **`scripts/repair_audience_queries.py`** rewrote the five existing queries with the same
  Qwen3-8B backend, stating the cluster's true mix in the prompt (all 5 by the model, 0 by
  the deterministic fallback). Each rewrite is recorded on the record under
  `audience_repairs` (index, old text, new text, and how it was produced), so the change is
  auditable and a later stage-2 re-run cannot silently revert it.

Two rewrites dropped the audience, one named the dominant one, one named
`unisex`, and one (`<\|sid_17\|><\|sid_24\|><\|sid_60\|>`) named **boys** — permitted at 8% of
that cluster, so the query stays satisfiable, but it is the one that moved the audience to a
minority rather than removing it. Lowering `AUDIENCE_MIN_SHARE` would push that case (and
others) toward dropping the audience instead.

Auditing the repair turned up a **shape assumption worth remembering**: the record keeps the
query list in four views, not three — `flat_key_value_map.user_queries`,
`key_value_map.user_queries.values` (a dict-nested list), `sections.user_queries`, and
`insights.gemini_user_queries` — so an edit that assumes plain lists at each path silently
leaves views stale, and a verification written against the same assumption agrees with the
bug. The script now reconciles every recorded repair into all four views idempotently and
scans the whole record for superseded text as a *substring*, which is what exposed
`response_text` and `validation_checks`. Those two keep the pre-repair wording by design:
they record what the model authored, not the live query list, and nothing downstream reads
them (stage 2 reads only `flat_key_value_map.user_queries`; stage 3 reads the enriched file).

Only the five texts changed — 5,807 queries, 6–10 per cluster, **5,752 unique (99.1%)**,
and all four task files (`4,103 / 32,688 / 3,262 / 416`) are unchanged in size.

A separate pre-existing quirk, untouched: on 4 clusters the query cap trimmed the canonical
list to 10 while `key_value_map.user_queries.values` still holds the model's 11. Nothing
reads that view.

## 4. Task generation (stage 3) — the 4 tasks and their ground truth

Driver: `scripts/fashion_rq_task{1..4}_build_*.py`, all sharing `fashion_rq_common.py`.
Common selection rule for every task: a cluster qualifies if it is a level-3 record with
`group_size ≥ 5`, has ≥ 1 usable enriched query (non-empty text + dict-shaped
attributes — all 5,807 qualify), and has non-empty Exact and Sibling pools (tasks 1, 2, 4
additionally require a non-empty Related pool). **416 of the 436 level-3 clusters** qualify (level-1 and level-2 anchor
records are never task candidates); the 20 exclusions all have an empty Exact or
Sibling pool.

Every task reuses the cluster's queries as-is and samples **one candidate set per
cluster**, shared by all of that cluster's queries — so per-query differences come only
from the query text, never from a different pool.

### Task 1 — Query-Product Ranking (primary task)
`task1_ranking_benchmark.jsonl` — **4,103 queries** over the 416 clusters.

Per cluster, samples **3 Exact + 4 Sibling + 4 Related + 4 Irrelevant** items (15
candidates, shuffled; 3,829 of 4,103 queries have the full 15 — small pools yield
smaller sets). Each cluster query becomes one record pairing the query with the shared
candidate set.

Ground truth per candidate: `relevance` ∈ {Exact, Sibling, Related, Irrelevant} and the
numeric `grade` (3/2/1/0, table above). Record shape:

```json
{"query_id": "t1-00000", "cluster_prefix": "<|sid_..|><|sid_..|><|sid_..|>",
 "query": "...", "attribute_tags": ["gender", "occasion", "product_type"],
 "candidates": [{"item_id": "...", "relevance": "Exact", "grade": 3,
                 "productDisplayName": "...", "brandName": "...", ...}, ...]}
```

`attribute_tags` lists which attributes the query's enrichment populates — the scoring
side breaks metrics out per tag (§5).

### Task 2 — Hierarchical Relevance Classification
`task2_classification_benchmark.jsonl` — **32,688 pairs** across 4,103 query ids.

Per cluster, samples a class-balanced set of **2 items per class** (E/S/R/I = 8 pairs)
and pairs the same 8 item ids with every cluster query — a fresh `query_id` and 8
`pair_id` rows per query. Label codes: `E`/`S`/`R`/`I`. Observed distribution:
9,654 E / 9,493 S / 9,654 R / 9,654 I (161 S pairs short because a few clusters have
< 2 sibling items in their pool — `sample_with_seed` returns fewer when the pool runs
out rather than padding).

```json
{"pair_id": "t2-q00000-000000", "query_id": "t2-q00000", "cluster_prefix": "...",
 "query": "...", "item_id": "...", "label": "E", "productDisplayName": "...", ...}
```

### Task 3 — Sibling Discrimination (the hardest distinction)
`task3_sibling_discrimination_benchmark.jsonl` — **3,262 queries** over **331 clusters**
(85 of the 416 skipped: fewer than 3 distinct sibling leaf clusters, and sibling
*diversity* is the point of this task).

Per cluster: 1 Exact **target** + 4 **Sibling distractors** drawn round-robin from
*distinct* neighboring leaf clusters under the same parent (so each distractor differs
from the target on a different single attribute — the "blue floral maxi dress" vs.
"blue floral midi" / "red floral maxi" / "blue solid maxi" pattern). Candidates are
shuffled; distractors keep their `distractor_cluster_prefix` for analysis.

Ground truth is **binary** and supports two prediction formulations:

- ranking: `grade` 1 (Exact target) / 0 (Sibling)
- selection: `correct_item_ids` — the target item ids a selection-style model must pick

Verified internally consistent: every grade-1 candidate is in `correct_item_ids`, every
grade-0 candidate is not.

```json
{"query_id": "t3-00000", "cluster_prefix": "...", "query": "...",
 "correct_item_ids": ["..."], "distractor_cluster_prefixes": ["...", ...],
 "candidates": [{"item_id": "...", "relevance": "Exact", "grade": 1,
                 "distractor_cluster_prefix": null, ...}, ...]}
```

### Task 4 — Query Paraphrase Robustness
`task4_paraphrase_benchmark.jsonl` — **416 paraphrase groups** covering all 4,103
queries (groups of 4–12 paraphrases; 352 groups have 12).

Each group is one cluster's full usable-query set, sharing a single candidate set
sampled exactly like Task 1 (3/4/4/4, grades 3/2/1/0). Sharing the pool is what makes
the top-K overlap between paraphrases meaningful: it isolates "did the wording change
what got retrieved" from "did the pool change."

```json
{"group_id": "t4-g00000", "cluster_prefix": "...",
 "queries": [{"query_id": "t4-g00000-0", "query": "..."}, ...],
 "candidates": [ ...same shape as Task 1... ]}
```

### Cross-task summary

| Task | Records | Clusters | Ground truth |
|------|---------|----------|--------------|
| 1 Ranking           | 4,103 queries (× ~15 candidates)  | 416 | graded relevance 3/2/1/0 |
| 2 Classification    | 32,688 pairs                      | 416 | E/S/R/I per pair |
| 3 Sibling Discrim.  | 3,262 queries (× 5 candidates)    | 331 | binary grade + `correct_item_ids` |
| 4 Paraphrase Robust.| 416 groups / 4,103 queries        | 416 | graded relevance 3/2/1/0 |

Rebuild at any time (no GPU needed, deterministic):

```bash
FASHION_RQ_TASKS_DIR=benchmark/tasks_generated \
  RUN_STAGE=3 bash scripts/reproduce.sh
```

Or invoke the four builders directly — they read
`FASHION_RQ_ENRICHED_JSON` and write to `FASHION_RQ_TASKS_DIR`, both defaulting to
release paths.

## 5. Scoring a model under test

`src/benchmark/fashion_rq_task{1..4}_compute_metrics.py` score predictions against the ground
truth above. All take `--benchmark` (default: the file built in §4), `--predictions`,
and an optional `--output` (default: `<benchmark>.metrics.json`).

Predictions formats:

- **T1 / T4**: `{"<query_id>": ["<item_id>", ...]}` — one ranked list per query,
  most-relevant-first. Unranked candidates count as ranked last (worst-case), so a
  model must rank every candidate for full credit.
- **T2**: `{"<pair_id>": "E"|"S"|"R"|"I"}`. Missing pairs score as an incorrect guess
  (a placeholder class), never silently dropped.
- **T3**: per query, either a ranked id list (ranking mode) or a single selected id
  (selection mode) — auto-detected; a missing query scores 0.

Metrics:

| Task | Metrics |
|------|---------|
| 1 | nDCG@5/10, Recall@5/10 (relevant = grade ≥ 1), overall + per attribute tag |
| 2 | accuracy, macro-F1, per-class P/R/F1/support, confusion matrix (sklearn) |
| 3 | ranking mode: Recall@1/@3 (binary grades); selection mode: accuracy |
| 4 | Consistency@K (primary; mean pairwise top-K Jaccard within a group), plus per-paraphrase nDCG@10 mean/stdev within groups |

## 6. Full reproduction

```bash
bash scripts/reproduce.sh                 # data -> stage 1 -> stage 2 -> stage 3 -> validate
RUN_STAGE=3 bash scripts/reproduce.sh     # a single stage
```

Stage 1 (~3 h 20 m GPU on one L4) and stage 2 (~15 min GPU) are resumable: stage 1
skips prefixes already present in its JSONL output, stage 2 skips queries already in
its attribute cache. Stage 3 is GPU-free and rebuilds the four task files
deterministically from the enriched corpus (seed 42).

Regenerated output goes to `benchmark/queries/*_regenerated_*` and
`benchmark/tasks_generated/`, so a reproduction run never overwrites the released
corpus. Paths are env-overridable: `SID_ITEMS_PARQUET`, `FASHION_RQ_ENRICHED_JSON`,
`FASHION_RQ_TASKS_DIR`, `BENCH_OUT` (see `fashion_rq_common.py` and each script's
`--help`).

## 7. Provenance

- The original Gemini-authored suite is included under
  [`src/benchmark/legacy_gemini/`](../src/benchmark/legacy_gemini/) for methodological
  reference, with its local paths replaced by placeholders; it is not wired into this
  release. The task and metrics scripts here are unmodified ports apart from path
  handling, gzip support and the scorer defaults.
- Qwen3-8B porting notes (YaRN, the query-budget cap, HF offline mode) are in §3 and in
  the header comments of `scripts/reproduce.sh`.
- The released corpus, tasks and ground truth were produced by this pipeline at the
  settings in `scripts/reproduce.sh`; `scripts/validate_release.py` re-checks the
  published counts against the shipped files.
