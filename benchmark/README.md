# The released benchmark

Everything here is self-contained: item metadata travels inside each task record, so the
four task files can be scored straight after cloning — no data download needed. Only
*regenerating* the corpus or building your own tasks requires the catalog.

Sizes are gzipped; every loader in `src/benchmark/` opens `.gz` transparently
(`fashion_rq_common.load_json` / `read_jsonl`), and so do `pandas.read_json`, `polars` and
`jq` if you decompress first.

## `queries/`

| File | Size | Contents |
|---|---|---|
| `cluster_queries_raw.jsonl.gz` | 3.0 MB | stage-1 output, one line per cluster (588): the cluster's attribute distribution, the model's analysis, and its `user_queries` |
| `cluster_queries_enriched.json.gz` | 0.8 MB | stage-2 output: the same 588 clusters, each with `enriched_queries` — 5,807 queries, every one carrying an `attributes` dict |

The enriched file is the corpus to use; the raw file is the generation record, and is the
input to stage 2 if you re-run attribute extraction.

```python
import gzip, json
with gzip.open("benchmark/queries/cluster_queries_enriched.json.gz", "rt") as fh:
    clusters = json.load(fh)
cluster = clusters[0]
print(cluster["prefix"], cluster["group_size"], len(cluster["enriched_queries"]))
q = cluster["enriched_queries"][0]
print(q["natural_language_query"])
print(q["attributes"])
```

## `tasks/`

Four files, described in full in [../docs/TASKS.md](../docs/TASKS.md):

| File | Records | Ground truth |
|---|---|---|
| `task1_ranking_benchmark.jsonl.gz` | 4,103 queries | per-candidate `relevance` + `grade` |
| `task2_classification_benchmark.jsonl.gz` | 32,688 pairs | `label` ∈ {E, S, R, I} |
| `task3_sibling_discrimination_benchmark.jsonl.gz` | 3,262 queries | `correct_item_ids` + graded candidates |
| `task4_paraphrase_benchmark.jsonl.gz` | 416 groups (4,103 queries) | shared candidate pool + grades |

```python
import gzip, json
with gzip.open("benchmark/tasks/task1_ranking_benchmark.jsonl.gz", "rt") as fh:
    record = json.loads(fh.readline())
print(record["query"])
for c in record["candidates"]:
    print(f"  {c['grade']}  {c['relevance']:<10} {c['productDisplayName']}")
```

Score a submission with `--predictions` (formats in [../docs/TASKS.md](../docs/TASKS.md)):

```bash
python src/benchmark/fashion_rq_task1_compute_metrics.py --predictions preds.json
```

The scorers default to these files; a regeneration run writes to `tasks_generated/` instead,
so nothing here is overwritten.

## `stats.txt`

`src/benchmark/benchmark_stats.py` output for the corpus: cluster counts per level, cluster
sizes, queries per cluster, uniqueness and length statistics — each against the target
specification the corpus was generated to hit. The "target" column is the original
Gemini-authored spec, not a gold standard; the deliberate deviations (volume +7.9%, length
+12%) are explained in [../docs/BENCHMARK.md](../docs/BENCHMARK.md).

## Integrity

`python scripts/validate_release.py` re-derives every count above from the shipped files and
checks record schemas and ground-truth presence. Run it after any modification — for example
if you subset the benchmark for a faster evaluation run, to confirm what changed.
