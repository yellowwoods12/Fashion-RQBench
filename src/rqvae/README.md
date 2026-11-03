# RQ-VAE with an attribute exclusivity loss

Trains the residual-quantized VAE that turns item embeddings into three-level semantic IDs,
and turns a trained checkpoint into the `sid_0/sid_1/sid_2/semantic_id` columns the benchmark
consumes. Background and quality figures: [../../docs/HIERARCHY_QUALITY.md](../../docs/HIERARCHY_QUALITY.md).

| File | Role |
|---|---|
| `train_rqvae_purity.py` | training, including the per-level exclusivity (SupCon) loss |
| `generate_sids.py` | encodes the catalog with a checkpoint and writes the SID parquet |
| `compute_hierarchy_metrics.py` | purity / exclusivity / coverage / fragmentation, plus the ablation variants |
| `embed_items_public_dataset.py` | produces the 1024-d item embeddings (Qwen3-Embedding-0.6B) |
| `device_manager.py`, `logger.py` | small shared utilities imported by the trainer |

## Requirements

A CUDA GPU (training on 44k items × 1024-d at 3 levels fits comfortably on a single 24 GB
card). Install the pinned training extras:

```bash
pip install -r requirements-train.txt
```

The trainer logs to Weights & Biases; set `WANDB_MODE=offline` to skip the network.

## Data you need

```bash
python scripts/download_data.py --only embeddings   # data/items_embeddings_1024d.parquet (169 MB)
python scripts/download_data.py --only checkpoint   # checkpoints/rqvae_purity/best_model.pth (17 MB)
```

Regenerating the embeddings is **not supported** in this release.
`embed_items_public_dataset.py` is retained for provenance, but it expects a
tokenizer step (`src/tokenize_items.py`) and a CSV/NPZ layout that are not
shipped. Download the checksummed `items_embeddings_1024d.parquet` instead.

## Train

```bash
python src/rqvae/train_rqvae_purity.py \
    --embeddings-path data/items_embeddings_1024d.parquet \
    --output-dir checkpoints/rqvae_purity
```

Key hyperparameters (all in `RQVAEPurityConfig`, overridable on the command line):

| Setting | Default | Notes |
|---|---|---|
| `codebook_quantization_levels` | 3 | one codebook per level → 3-token semantic IDs |
| `codebook_size` | 64 | so each level contributes one of 64 codes |
| `lambda_exclusivity` | 0.05 | weight on the per-level SupCon term |
| `SUPCON_LABELS_PER_LEVEL` | `masterCategory` / `articleType` / all 7 combos | the supervision per level |

The checkpoint directory accumulates step checkpoints plus `best_model.pth` (lowest
validation loss). The released SID set came from a run of this script.

## Generate semantic IDs

```bash
python src/rqvae/generate_sids.py \
    --checkpoint checkpoints/rqvae_purity/best_model.pth \
    --embeddings data/items_embeddings_1024d.parquet \
    --catalog    data/items_with_sids.parquet \
    --out        data/items_with_sids_regenerated.parquet
```

The encoder reads embeddings and writes SIDs onto the catalog, matching rows on `id` — so the
two released files can be used as-is. Pass the same path to `--embeddings` and `--catalog` if
you have a single parquet carrying both. The exact checkpoint configuration and expected
hashes are in [../../docs/REPRODUCE_SIDS.md](../../docs/REPRODUCE_SIDS.md).

> `generate_sids.py` is adapted from the script that produced the released SID parquet:
> same checkpoint format and same output columns, with paths parameterised and the
> training-only attribute labels no longer loaded. If you only need to *use* the released
> semantic IDs, download `items_with_sids.parquet` and skip this step entirely.

## Measure the hierarchy

```bash
python src/rqvae/compute_hierarchy_metrics.py            # table + secondary metrics
python src/rqvae/compute_hierarchy_metrics.py --json results.json
```

Prints purity per level, coverage, fragmentation, and exclusivity, plus the variant readings
(raw tree vs. benchmark-filtered clusters, per-combo breakdown, taxonomy-based purity). The
choices behind the headline numbers, and why the filtered variant is inflated by
construction, are explained in [../../docs/HIERARCHY_QUALITY.md](../../docs/HIERARCHY_QUALITY.md).
