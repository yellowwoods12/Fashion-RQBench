# Task specifications

Four tasks over one semantic-ID hierarchy. All share a single relevance mapping and all are
scored by scripts in `src/benchmark/`.

> **Scope:** this document and the shipped scorers cover the four task-level metrics. The
> paper's weighted query relevance and attribute-match scores are not part of this release; see
> [EVALUATION_SCOPE.md](EVALUATION_SCOPE.md) for the exact boundary.

## Relevance mapping

Derived from the SID prefix tree, not from human annotation:

| Label | Code | Grade | Condition on two items' semantic IDs |
|---|---|---|---|
| Exact | `E` | 3 | same level-3 prefix (`semantic_id`) — the target leaf cluster |
| Sibling | `S` | 2 | same level-2 prefix, different level-3 |
| Related | `R` | 1 | same level-1 prefix, different level-2 |
| Irrelevant | `I` | 0 | different level-1 prefix |

`Sibling` is the interesting one: siblings sit under the same parent, so they are close in
the learned space and hard to tell apart — which is what task 3 isolates.

## Cluster selection (all tasks)

A cluster becomes a task anchor if it is a level-3 record with ≥ 5 items, at least one usable
query, and non-empty Exact and Sibling pools; tasks 1, 2 and 4 additionally require a
non-empty Related pool. **416 of the 436 level-3 clusters qualify** — all 20 exclusions have
an empty Exact or Sibling pool (tiny leaves). Level-1 and level-2 records are never anchors.

Each cluster's candidate pool is sampled **once** (seed 42) and shared by all of that
cluster's queries, so differences between queries reflect the query text, never a different
candidate set. Candidate sets are 3 Exact + 4 Sibling + 4 Related + 4 Irrelevant = 15 items,
shuffled; 3,829 of 4,103 task-1 queries get the full 15, the rest have smaller pools.

---

## Task 1 — Query–Product Ranking

`benchmark/tasks/task1_ranking_benchmark.jsonl.gz` — **4,103 queries** over 416 clusters.

**Record fields**

| Field | Type | Meaning |
|---|---|---|
| `query_id` | str | `t1-00000` |
| `cluster_prefix` | str | the anchor cluster's full semantic ID |
| `query` | str | the natural-language query |
| `attribute_tags` | list[str] | which attribute constraints the query carries (`color`, `gender`, …) — the reporting dimension |
| `candidates` | list[obj] | 15 items: `item_id`, `relevance`, `grade`, `productDisplayName`, `brandName`, … |

**Prediction format** — `{"<query_id>": ["<item_id>", ...]}`, ranked most-relevant first.
Candidates omitted from a list are ranked last before scoring, so a model must rank every
candidate to get full credit.

**Metrics** — nDCG@5, nDCG@10, Recall@5, Recall@10, overall and broken out per attribute tag
(Recall@K here = the fraction of all non-Irrelevant candidates appearing in the top K).

```bash
python src/benchmark/fashion_rq_task1_compute_metrics.py --predictions preds.json --output t1.json
```

---

## Task 2 — Hierarchical Relevance Classification

`benchmark/tasks/task2_classification_benchmark.jsonl.gz` — **32,688 pairs** over 4,103 queries.

**Record fields** — `pair_id`, `query_id`, `cluster_prefix`, `query`, `item_id`, `label`
(one of `E`/`S`/`R`/`I`), plus the item's full metadata (`productDisplayName`, `brandName`,
`masterCategory`, `subCategory`, `articleType`, `baseColour`, `gender`, `usage`, `price`,
`imageURL`), so the task is answerable from the record alone without the catalog.

**Prediction format** — `{"<pair_id>": "<E|S|R|I>"}`. Pairs absent from the file are scored
as an incorrect guess (a distinct placeholder label), not dropped — an incomplete submission
is penalised rather than rewarded with a smaller denominator.

**Metrics** — accuracy, macro-F1, and per-class precision/recall/F1/support.

```bash
python src/benchmark/fashion_rq_task2_compute_metrics.py --predictions preds.json --output t2.json
```

---

## Task 3 — Sibling Discrimination

`benchmark/tasks/task3_sibling_discrimination_benchmark.jsonl.gz` — **3,262 queries** over 331
clusters (85 clusters were skipped for insufficient sibling diversity; ≥ 3 distinct sibling
leaves are required).

**Record fields** — `query_id`, `cluster_prefix`, `query`, `correct_item_ids`,
`distractor_cluster_prefixes`, `candidates` (`item_id`, `relevance`, `grade`,
`distractor_cluster_prefix`). Candidates are the target plus siblings drawn from other leaf
clusters under the same parent: the target is labelled `Exact` with grade 1, every distractor
`Sibling` with grade 0, and each distractor records which sibling cluster it came from. All
candidates share the query's parent, so the task cannot be solved by coarse category matching.

**Prediction format** — `{"<query_id>": ["<item_id>", ...]}` (ranking mode) **or**
`{"<query_id>": "<item_id>"}` (selection mode). The two are auto-detected per query by value
type and can be mixed in one file. Queries missing from the file score 0 on every metric.

**Metrics** — nDCG@1, nDCG@3, Recall@1, Recall@3 (ranking mode; relevance is binary,
target = 1), plus plain accuracy (selection mode). K is small because the pool is small.

```bash
python src/benchmark/fashion_rq_task3_compute_metrics.py --predictions preds.json --output t3.json
```

---

## Task 4 — Query Paraphrase Robustness

`benchmark/tasks/task4_paraphrase_benchmark.jsonl.gz` — **416 groups** covering **4,103
paraphrase queries** (6–10 per group); each group is one cluster's full query set, treated as
paraphrases of a single intent, scored against the cluster's shared candidate pool. The
paraphrase queries are the same 4,103 query instances task 1 ranks — task 4 regroups them by
cluster and scores consistency across the group instead of quality within one query.

**Record fields** — `group_id`, `cluster_prefix`, `queries` (list of `{query_id, query}`),
`candidates` (same shape as task 1).

**Prediction format** — `{"<query_id>": ["<item_id>", ...]}`, one ranked list per individual
paraphrase (same shape as task 1; missing candidates are pushed last).

**Metrics**
- **Consistency@K (primary)** — mean pairwise Jaccard overlap of the top-K item sets across
  every pair of paraphrases in a group, averaged over groups. High means the model retrieves
  the same items however the query is worded.
- **Per-paraphrase nDCG@10**, with the mean and stdev *within* each group averaged across
  groups. Low within-group stdev is a second stability signal.

```bash
python src/benchmark/fashion_rq_task4_compute_metrics.py --predictions preds.json --output t4.json
```

---

## Design notes

- **The four tasks share a cluster universe but not a query universe.** Task 2 and 3 expand
  each query into pairs; tasks 1 and 4 rank a pool. A model that wins task 1 does not
  automatically win task 3 — sibling discrimination is a finer distinction than ranking a
  pool that is 80% non-Exact.
- **Task 4's paraphrase groups are one model's variations**, not human paraphrases. The task
  measures self-consistency of retrieval under rewording, which is a real robustness
  property, but read it as that and not as agreement with human intent.
- **Ground truth follows the learned hierarchy.** If the SID hierarchy mis-groups items, the
  ground truth inherits that error. See
  [docs/HIERARCHY_QUALITY.md](HIERARCHY_QUALITY.md) for how pure the clusters actually are,
  and [docs/DATA_CARD.md](DATA_CARD.md#limitations-and-known-issues) for the caveats.
- Every task file is self-contained: item metadata travels with each record, so no task
  requires the catalog parquet to be scored.
