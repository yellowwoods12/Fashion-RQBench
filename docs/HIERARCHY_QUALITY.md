# Semantic-ID hierarchy: training and quality

How the semantic IDs were produced, and how good the resulting hierarchy is. The quality
numbers are the ones to quote when describing the SID set; the definitions matter as much as
the values, so they are spelled out below rather than left to the metric names.

## Training

`src/rqvae/train_rqvae_purity.py` trains a residual-quantized VAE over 1024-d item embeddings
(Qwen3-Embedding-0.6B) with three quantization levels, each a 64-entry codebook. An item's
semantic ID is the three chosen code indices.

Beyond the usual reconstruction (`L_recon`) and codebook (`L_vq`) terms, training adds a
**per-level exclusivity loss**:

```
L = L_recon + L_vq + λ_excl · Σ_levels  L_exclusivity(level)
L_exclusivity(level) = SupCon(normalize(quantized_out_at_level), labels(level))
λ_excl = 0.05   # the script default; the released run's value is in its checkpoint config
```

Supervision labels per level:

| Level | Supervised on | Classes |
|---|---|---|
| L0 (`sid_0`) | `masterCategory` | 7 |
| L1 (`sid_1`) | `articleType` | 143 |
| L2 (`sid_2`) | all seven `articleType × {baseColour, gender, usage, season, ageGroup, price_tier, fashionType}` combinations at once | 203–1,886 per combo |

A supervised-contrastive term pulls items sharing an attribute combination together and
pushes different combinations apart, so an attribute combination ends up **owned by one
cluster** instead of scattered across many. That property — exclusivity — is what makes a
cluster-specific query well-defined: without it, "men's casual blue T-shirts" would match
items spread over dozens of clusters, and a cluster-scoped ground truth would be a fiction.

The rationale for putting the loss here rather than on a dedicated purity head: vanilla
reconstruction + VQ already gets individual attributes to ≥ 90% cluster purity on its own;
the gap is combinations. Reconstructions dominate the objective, so the exclusivity term is
small (0.05) and shapes the codebook assignment rather than the encoder.

`price_tier` is derived from `discountedPrice` at training time (`< 711` budget,
`≤ 1595` mid-range, else premium); items with a null label are masked out of the loss.

**Reproducing:** you need `data/items_embeddings_1024d.parquet` (169 MB) and a GPU. See
[../src/rqvae/README.md](../src/rqvae/README.md) for the command and hyperparameters. The
released SID set was generated from that lineage's `best_model.pth` by
`src/rqvae/generate_sids.py`, which writes `sid_0/sid_1/sid_2/semantic_id` onto the catalog.

## Quality metrics

The figures below are the with-exclusivity ablation values reported in the paper.
`src/rqvae/compute_hierarchy_metrics.py` is included for recomputation over all 44,446 items.
Clusters are prefix groups: L1 = `sid_0`, L2 = `sid_0·sid_1`, L3 = full `semantic_id`. Clusters
with fewer than 2 items are excluded from the aggregates (`MIN_CLUSTER_SIZE = 2`, matching the
repository's own convention), and dominance is item-weighted.

| Metric | L1 | L2 | L3 |
|---|---|---|---|
| Purity | 94.7% | 89.2% | 83.6% |
| Exclusivity | 24.4% | 52.2% | 40.8% |

| Metric | Value |
|---|---|
| Catalog coverage (items in a leaf cluster ≥ 2 items) | 99.3% |
| Attribute fragmentation (leaf clusters per combo value) | 3.76 |

**Definitions**, because "purity"/"coverage"/"fragmentation" do not have single agreed
meanings:

- **Purity** — for each cluster, the share held by the most common value of that level's
  supervised attribute; averaged over clusters weighted by cluster size. L3 is the mean across
  the seven combos, since L3's loss term pulls on all of them simultaneously.
- **Exclusivity** — the transpose: for each attribute *value*, the share held by its most
  common cluster. Weighted the same way. This is the quantity the exclusivity loss targets,
  and the one that matters for query specificity.
- **Coverage** — the share of catalog items that land in a leaf cluster with ≥ 2 items, i.e.
  items the hierarchy actually groups rather than isolates. (A 5-item variant, matching the
  benchmark's own cluster floor, reads 97.4%.)
- **Fragmentation** — the mean number of distinct leaf clusters each supervised attribute
  value is spread across; lower is better. The headline value is label-unweighted (3.76).

### Two things to be careful with when quoting these

**Use the raw tree, not the benchmark's clusters.** The raw hierarchy has 49 / 304 / 1,270
prefixes (48 / 277 / 969 with ≥ 2 items). The benchmark's 588 clusters are that tree after a
`articleType`-purity ≥ 0.80 and size ≥ 5 filter. Measuring purity on the filtered set returns
95.7 / 96.7 / 70.9 — mostly a restatement of the filter, not a hierarchy property. The filter
is a benchmark-construction choice; the numbers above describe the hierarchy.

**L3 aggregates seven articleType-anchored combinations.** `baseColour` is the hardest of
them: there are 1,886 distinct `articleType × baseColour` combinations but only 1,270 leaf
clusters, so a cluster cannot be dominated by one colour combination. The headline L3 value
above is the with-exclusivity ablation figure reported in the paper. If a single-attribute
reading of the levels is wanted instead, purity against the catalog taxonomy
(`masterCategory` / `subCategory` / `articleType`) is 88.8 / 88.9 / 88.9.

### Sanity check

Reconstructing the cluster universe with the benchmark's own filter (`articleType` purity
≥ 0.80, ≥ 5 items) reproduces its published cluster counts exactly — **17 / 135 / 436** —
which is what ties these metrics to the benchmark's clusters rather than to some parallel
accounting of them.
