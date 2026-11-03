# Data card

What the release contains, where it came from, what its limits are, and how to host the two
data files that are not in version control.

## What ships where

| Artifact | Location | Size | Licence |
|---|---|---|---|
| Query corpus (588 clusters, 5,807 queries + attributes) | `benchmark/queries/` | 3.8 MB gz | CC BY 4.0 |
| Four task files + ground truth | `benchmark/tasks/` | 2.1 MB gz | CC BY 4.0 |
| Catalog + semantic IDs, 44,446 items | separate download — `items_with_sids.parquet` | 16 MB | upstream MIT + CC BY 4.0 for the SID columns (needed to regenerate queries or build tasks) |
| Item embeddings, 1024-d | separate download — `items_embeddings_1024d.parquet` | 169 MB | upstream |
| RQ-VAE checkpoint | separate download — `rqvae_purity_best_model.pth` | 17 MB | MIT |

Fetched with `python scripts/download_data.py` (the default base URL is the v1.0.0
release); each file is SHA-256-verified against hashes pinned in that script. See
[Data availability](#data-availability).

## Source catalog

44,446 fashion products. The taxonomy columns, the image host, and the row count all match
the public **Myntra product catalog distributed as the Kaggle "Fashion Product Images
Dataset"**, whose `styles.csv` holds 44,441 rows:

- 44,441 of the 44,446 rows carry an `imageURL` on `assets.myntassets.com` — the Myntra CDN.
- The 5 remaining rows have a null `imageURL` and no image; they are retained because their
  attributes are complete.
- `id`, `gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`,
  `year`, `usage`, `productDisplayName` are that dataset's `styles.csv` columns.

**Upstream licence: MIT**, as stated by the dataset's distributor and confirmed by us. MIT permits redistribution and
derivative works, so shipping the catalog (or the semantically-ID'd version of it) alongside
the benchmark is fine; keep the upstream copyright notice with any copy. Our own artifacts —
queries, attributes, tasks, ground truth, SID assignments — are licensed CC BY 4.0 as stated
below, which is compatible with the MIT-licensed input.

One caveat worth keeping in mind even though the licence is permissive: the product images are
hotlinked rather than redistributed, and brands named in `productDisplayName`/`brandName` remain
their owners' trademarks. The licence covers the catalogue data, not any right to use the marks.

### Derived columns

Added by this project on top of the source catalog:

| Column | How it was derived |
|---|---|
| `brandName`, `ageGroup`, `price`, `discountedPrice`, `fashionType`, `displayCategories`, `articleAttributes`, `description`, `imageURL` | present in the working catalog prior to this release; treated as given |
| `item_context` | the text blob embedded to produce item vectors (median 954 characters, max 52,860) |
| `embedding` | 1024-d vector from **Qwen3-Embedding-0.6B**, pooled and projected to 1024 dims (`src/rqvae/embed_items_public_dataset.py`) — shipped separately as `items_embeddings_1024d.parquet` |
| `sid_0`, `sid_1`, `sid_2` | RQ-VAE codebook indices at each of the three quantization levels (64-entry codebooks) |
| `semantic_id` | `"<|sid_{sid_0}|><|sid_{sid_1}|><|sid_{sid_2}|>"` — the item's semantic ID |

## Schema of `items_with_sids.parquet`

44,446 rows. `id` is a string; `sid_*` are int32; `year` is float64; `price` and
`discountedPrice` are int64; every other column is a string.

| Column | Distinct values | Nulls | Notes |
|---|---|---|---|
| `id` | 44,446 | 0 | unique product id |
| `gender` | 5 | 0 | Men, Women, Boys, Girls, Unisex |
| `masterCategory` | 7 | 0 | level-1 supervised attribute |
| `subCategory` | 45 | 0 | catalog taxonomy |
| `articleType` | 143 | 0 | level-2 supervised attribute; the benchmark's purity filter uses this |
| `baseColour` | 47 | 0 | |
| `season` | 5 | 0 | |
| `year` | — | 0 | catalog year |
| `usage` | 10 | 0 | Casual dominates (34,414 of 44,446) |
| `productDisplayName` | — | 0 | |
| `brandName` | 424 | 0 | Nike largest (2,203 items) |
| `ageGroup` | 7 | 0 | |
| `price`, `discountedPrice` | — | 0 | `discountedPrice` drives the derived `price_tier` |
| `fashionType` | 3 | 0 | Fashion dominates (39,973) |
| `displayCategories` | — | 0 | |
| `articleAttributes` | — | 0 | free-text attribute blob |
| `description` | — | 0 | |
| `imageURL` | — | **5** | hotlinked to Myntra's CDN; see limitations |
| `item_context` | — | 0 | text used for embedding |
| `sid_0`, `sid_1`, `sid_2` | 49 / — / — | 0 | 64-entry codebooks; 49 of 64 root codes are used |
| `semantic_id` | 1,270 | 0 | leaf cluster label; `sid_0/sid_1/sid_2` rebuild it exactly |

**There are no missing values in any taxonomy attribute**, which is what lets purity and the
query-generation prompt treat the catalog as fully labelled.

## Hierarchy shape

| Level | Distinct prefixes | With ≥ 2 items | Benchmark clusters (purity ≥ 0.80, ≥ 5 items) |
|---|---|---|---|
| 1 (`sid_0`) | 49 | 48 | 17 |
| 2 (`sid_0·sid_1`) | 304 | 277 | 135 |
| 3 (full `semantic_id`) | 1,270 | 969 | 436 |

99.3% of items fall in a leaf cluster of ≥ 2 items; 301 leaves are singletons (0.7% of items).
Only 436 of the 1,270 leaves become benchmark clusters, because a cluster must be both
attribute-pure and large enough to sample negatives from. See
[docs/HIERARCHY_QUALITY.md](HIERARCHY_QUALITY.md).

## Query corpus

5,807 queries over 588 clusters: 6–10 per cluster (mean 9.9), 99.1% unique strings, mean 11.2
words, range 5–23. Every query carries a filled `attributes` dict
(`product_type`, `gender`, `age_group`, `color`, `pattern`, `material`, `occasion`, `brand`,
`price_hint`, `style_tags`). Authored by Qwen3-8B from each cluster's attribute distribution;
see [docs/BENCHMARK.md](BENCHMARK.md) for the prompts and the one post-hoc repair that was
applied (`audience_repairs`).

## Limitations and known issues

- **Images are hotlinked, not redistributed.** `imageURL` points at Myntra's CDN. The
  benchmark does not require images (every task is text + item metadata), but if you plan to
  use them, fetch them yourself and expect link rot. 5 items have no image.
- **One catalog, one embedder, one taxonomy.** All numbers are from a single 44k-item fashion
  catalog embedded with Qwen3-Embedding-0.6B. Nothing here establishes that the hierarchy
  quality figures transfer to another catalog or embedder.
- **The catalog is not relevance-labelled by humans.** Relevance comes from the SID tree
  (Exact / Sibling / Related / Irrelevant). The benchmark therefore measures agreement with a
  *learned* hierarchy, not with human judgements of relevance.
- **Queries are LLM-authored and template-shaped.** They are grounded in each cluster's real
  attribute distribution, but they are not real user queries. Paraphrase groups (task 4) in
  particular are one model's variations on a theme.
- **`usage` is heavily imbalanced** (Casual is 77% of the catalog), so attribute-conditioned
  conclusions about non-casual usage are weakly supported.
- **`item_context` has a long tail** (median 954 characters, max 52,860), so embedding text
  length is not uniform across items.
- **Five queries were rewritten post-hoc** to stop them naming a gender their cluster lacks;
  each is recorded in `audience_repairs` on its cluster. 21 further queries (0.36%) carry an
  extracted `gender` attribute outside their cluster's audience, almost all the extractor
  inferring `unisex` where the query names no audience. Attributes feed only task 1's
  per-attribute reporting breakdown, not any ground truth.

## Data availability

The artifacts are not bundled with the git repository. They are published as assets of the
[v1.0.0 release](https://github.com/yellowwoods12/Fashion-RQBench/releases/tag/v1.0.0), and
`scripts/download_data.py` downloads and checksum-verifies them by default:

| File | Bytes | SHA-256 | Local path |
|---|---|---|---|
| `items_with_sids.parquet` | 16,228,222 | `ce3431c0…d5b8eb` | `data/` |
| `items_embeddings_1024d.parquet` | 169,157,398 | `e680f6ac…6322fe` | `data/` |
| `rqvae_purity_best_model.pth` | 16,750,453 | `7ef5abc0…43ea25` | `checkpoints/rqvae_purity/` |

Override the host with `--base-url` or `$FASHION_RQ_DATA_BASE_URL` to use a mirror. Only the
catalog is needed to regenerate queries or build new tasks; none of the three is needed to
score the shipped tasks.

## Licence

- Benchmark artifacts produced here (queries, attributes, tasks, ground truth, SID
  assignments): **CC BY 4.0** — see [../LICENSE-DATA](../LICENSE-DATA).
- Code and the RQ-VAE checkpoint: **MIT** — see [../LICENSE](../LICENSE).
- Underlying product catalog: **MIT** (upstream, confirmed) — redistributable; retain the
  upstream copyright notice. Our derived columns and artifacts are CC BY 4.0.
