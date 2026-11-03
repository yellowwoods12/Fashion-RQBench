# Fashion-RQBench

A four-task retrieval benchmark over a **hierarchical semantic-ID item space**, with
natural-language queries authored by an LLM from the hierarchy itself.

Items are tokenized into three-level semantic IDs (`<|sid_104|><|sid_365|><|sid_621|>`) by
an RQ-VAE trained with a per-level **exclusivity loss**. Each SID prefix defines a cluster;
queries are generated per cluster from that cluster's actual attribute distribution, so
every query has a well-defined ground-truth item set that follows from the hierarchy rather
than from human annotation.

**Release contents:** 588 clusters (17 / 135 / 436 at levels 1 / 2 / 3), 5,807 queries
(99.1% unique strings, mean 11.2 words), and four tasks totalling 44,156 scored instances
with ground truth (4,103 ranking queries · 32,688 classification pairs · 3,262 sibling
queries · 4,103 paraphrase queries in 416 groups). Code, queries, tasks and ground truth
ship in this repository; the item catalog, embeddings and RQ-VAE checkpoint are published
as release assets (see [Data](#data)).

---

## The four tasks

All four share one relevance mapping derived from the SID tree: two items are **Exact** if
they share a level-3 prefix, **Sibling** if they share a level-2 prefix, **Related** if they
share only a level-1 prefix, and **Irrelevant** otherwise.

| Task | What it tests | Instances | Ground truth |
|---|---|---|---|
| **1. Query–Product Ranking** | Rank a fixed 15-item candidate pool (3 Exact, 4 Sibling, 4 Related, 4 Irrelevant) for a query | 4,103 queries | Per-candidate `relevance` + `grade` (3/2/1/0) |
| **2. Hierarchical Relevance Classification** | Assign the relevance label for a (query, item) pair, each pair shown with item metadata | 32,688 pairs | `label` ∈ {E, S, R, I} |
| **3. Sibling Discrimination** | Pick the correct item from its SID siblings — the hardest distinction, since siblings are near-identical by construction | 3,262 queries | `correct_item_ids` + sibling candidates |
| **4. Query Paraphrase Robustness** | Retrieve the same cluster from every paraphrase of one cluster query, i.e. be invariant to wording | 416 groups | Shared candidate pool + relevance grades |

Tasks 1 and 4 reuse one sampled candidate pool per cluster; tasks 2 and 3 are pair-level.
Full construction details, field-by-field schemas, and scoring commands are in
[docs/TASKS.md](docs/TASKS.md).

## Hierarchy quality of the semantic-ID set

Metrics for the released SID set (the exclusivity-loss lineage), as reported in the paper.
[`src/rqvae/compute_hierarchy_metrics.py`](src/rqvae/compute_hierarchy_metrics.py) is included
for recomputation; see [docs/HIERARCHY_QUALITY.md](docs/HIERARCHY_QUALITY.md) for the exact
definitions, which the metric names alone do not pin down.

| Purity L1 | Purity L2 | Purity L3 | Coverage | Fragmentation |
|---|---|---|---|---|
| 94.7% | 89.2% | 83.6% | 99.3% | 3.76 |

Purity is the item-weighted per-cluster share of the dominant value of each level's
supervised attribute (L1 = `masterCategory`, L2 = `articleType`, L3 = the seven
`articleType × {colour, gender, usage, season, ageGroup, priceTier, fashionType}` combinations).
Coverage is the share of catalog items in a leaf cluster of ≥ 2 items; fragmentation is the mean
number of leaf clusters per supervised attribute value. These describe the **raw** SID tree
(49 / 304 / 1,270 prefixes); the 588 benchmark clusters are that tree after a purity ≥ 0.80 and
size ≥ 5 filter. Measuring purity on the filtered set reads 95.7 / 96.7 / 70.9, which restates
the benchmark's selection rule rather than the hierarchy, so the raw-tree figures above are the
ones to quote.

## Quickstart

```bash
git clone https://github.com/yellowwoods12/Fashion-RQBench.git
cd Fashion-RQBench
pip install -r requirements.txt   # training/embedding extras: requirements-{train,embed}.txt

# 1. (optional) fetch the catalog + semantic IDs (~16 MB); drop --only to also
#    get the embeddings and the RQ-VAE checkpoint (~200 MB total)
python scripts/download_data.py --only sids

# 2. confirm the release is intact (checksums, counts, schemas)
python scripts/validate_release.py

# 3. load a task and have a look
python - <<'PY'
import json, gzip
with gzip.open("benchmark/tasks/task1_ranking_benchmark.jsonl.gz", "rt") as f:
    record = json.loads(f.readline())
print(record["query"])
for c in record["candidates"][:5]:
    print(f"  {c['grade']}  {c['relevance']:<10} {c['productDisplayName']}")
PY
```

To generate queries and tasks from scratch instead, see
[scripts/reproduce.sh](scripts/reproduce.sh) and [docs/BENCHMARK.md](docs/BENCHMARK.md).
Stage 1 needs a GPU (~3 h 20 m on one L4 for 588 clusters); everything else is quick and
stage 3 is GPU-free.

## Repository layout

```
benchmark/            the released benchmark — self-contained, no data download needed
  queries/            raw generation stream + enriched query corpus (gzip JSON)
  tasks/              4 task files with ground truth (gzip JSONL)
docs/                 data card, task specs, generation process, hierarchy metrics
src/benchmark/        query generation + task construction + scoring (Qwen3-8B)
src/benchmark/legacy_gemini/   the earlier Gemini-authored pipeline, for reference
src/rqvae/            RQ-VAE training with the exclusivity loss + SID generation
scripts/              download_data.py, validate_release.py, reproduce.sh
data/                 populated by download_data.py (not in version control)
```

## Documentation

- [docs/DATA_CARD.md](docs/DATA_CARD.md) — contents, schema, provenance, bias/limitations, licensing
- [docs/TASKS.md](docs/TASKS.md) — task schemas, prediction formats, metrics
- [docs/BENCHMARK.md](docs/BENCHMARK.md) — generation pipeline and prompts
- [docs/HIERARCHY_QUALITY.md](docs/HIERARCHY_QUALITY.md) — SID hierarchy metrics
- [docs/CATALOG_PROVENANCE.md](docs/CATALOG_PROVENANCE.md) — catalog derivation and source of record
- [docs/REPRODUCE_SIDS.md](docs/REPRODUCE_SIDS.md) — checkpoint config and SID regeneration
- [docs/HUMAN_STUDY_ETHICS.md](docs/HUMAN_STUDY_ETHICS.md) — human-study protocols, ethics/consent, aggregate results
- [docs/EVALUATION_SCOPE.md](docs/EVALUATION_SCOPE.md) — which evaluation code is included
- [examples/](examples/) — runnable BM25 baseline and scoring walkthrough

## Data

The item catalog, embeddings and RQ-VAE checkpoint are published as assets of the
[v1.0.0 release](https://github.com/yellowwoods12/Fashion-RQBench/releases/tag/v1.0.0)
because of size. `scripts/download_data.py` fetches them from the release by default,
verifying SHA-256 against hashes pinned in the script:

| File | Size | Needed for |
|---|---|---|
| `items_with_sids.parquet` | 16 MB | regenerating queries or building tasks — catalog + `sid_0/sid_1/sid_2` + `semantic_id` for all 44,446 items |
| `items_embeddings_1024d.parquet` | 169 MB | retraining or re-encoding with the RQ-VAE (`src/rqvae/`) |
| `rqvae_purity_best_model.pth` | 17 MB | re-encoding items with the released SIDs' checkpoint (`src/rqvae/generate_sids.py`) |

Queries, tasks and ground truth **do** ship in-repo (6 MB gzipped), so the benchmark is
usable immediately after cloning. See [docs/DATA_CARD.md](docs/DATA_CARD.md) for provenance,
schema, licensing, checksums and hosting details for the three downloads.

## Reproducing

```bash
bash scripts/reproduce.sh              # data -> queries -> attributes -> tasks -> validate
RUN_STAGE=3 bash scripts/reproduce.sh  # or one stage at a time
```

The released corpus was produced with the defaults in that script (`TARGET_QUERY_RANGE=9-10`,
Qwen3-8B, 3-stage pipeline). Stage 1 and 2 are resumable; the checked-in corpus is never
overwritten — regenerated output goes to `benchmark/queries/*_regenerated_*` and
`benchmark/tasks_generated/`. For the SID/embedding side, see
[docs/REPRODUCE_SIDS.md](docs/REPRODUCE_SIDS.md); for what evaluation code is and is not
included, see [docs/EVALUATION_SCOPE.md](docs/EVALUATION_SCOPE.md).

## Citation

If you use this benchmark, please cite it — see [CITATION.cff](CITATION.cff).

## Ethics

Queries are LLM-generated and contain no personal data. The two human studies
(query naturalness/clarity and SID relevance-label validation) used voluntary
participants with informed consent; only aggregate results are released. See
[docs/HUMAN_STUDY_ETHICS.md](docs/HUMAN_STUDY_ETHICS.md) for the protocols,
aggregate tables, and ethics/consent statement.

## Licensing

- **Code** — MIT ([LICENSE](LICENSE)).
- **Benchmark data** (queries, tasks, ground truth, semantic IDs) — CC BY 4.0
  ([LICENSE-DATA](LICENSE-DATA)).
- **RQ-VAE checkpoint** (`rqvae_purity_best_model.pth`) — MIT, like the code.
- **Underlying catalog** — MIT (upstream), details in [NOTICE](NOTICE) and
  [docs/DATA_CARD.md](docs/DATA_CARD.md#licence).
