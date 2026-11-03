# Reproducing the semantic IDs

The released `sid_0`, `sid_1`, `sid_2` and `semantic_id` columns were produced
by the checkpoint in `checkpoints/rqvae_purity/best_model.pth`. You do **not**
need to retrain the model to reproduce the released SIDs; run the released
checkpoint through `src/rqvae/generate_sids.py`.

## Released checkpoint

| Field | Value |
|---|---|
| File | `checkpoints/rqvae_purity/best_model.pth` |
| SHA-256 | `7ef5abc07e58c832aba2699d390505dd9286044eb03825bc9849e60d4243ea25` |
| Epoch | 9 |
| Step | 44,000 |
| Validation loss | 4.266479 |

## Training configuration (from the checkpoint)

| Setting | Value |
|---|---|
| `item_embedding_dim` | 1024 |
| `encoder_hidden_dims` | `[512, 256, 128]` |
| `codebook_embedding_dim` | 32 |
| `codebook_quantization_levels` | 3 |
| `codebook_size` | 64 |
| `commitment_weight` | 0.25 |
| `use_rotation_trick` | `True` |
| `supcon_temperature` | 0.07 |
| `lambda_exclusivity` | 0.05 |
| `batch_size` | 512 |
| `gradient_accumulation_steps` | 4 |
| `num_epochs` | 50,000 (best checkpoint at epoch 9) |
| Optimizer / schedule | AdamW, `cosine_with_warmup` |
| `max_lr` / `min_lr` | `3e-4` / `1e-6` |
| `warmup_steps` | 200 |
| `use_gradient_clipping` | `True`, norm 1.0 |
| `use_kmeans_init` | `True` |
| `val_split` | 0.05 |
| Split / torch seed | 42 (`--seed`) |

The per-level supervision is `masterCategory` at L1, `articleType` at L2, and
the seven `articleType × {colour, gender, usage, season, ageGroup, price_tier,
fashionType}` combinations at L3.

## Generate SIDs from the released checkpoint

```bash
python scripts/download_data.py --only embeddings
python scripts/download_data.py --only sids

python src/rqvae/generate_sids.py \
    --checkpoint checkpoints/rqvae_purity/best_model.pth \
    --embeddings data/items_embeddings_1024d.parquet \
    --catalog    data/items_with_sids.parquet \
    --out        data/items_with_sids_regenerated.parquet
```

The encoder reads the 1024-d embeddings and writes the SID columns onto the
catalog, matching rows on `id`.

## Optional: train from scratch

The released checkpoint is the supported path. To train a new model with the
same configuration:

```bash
python src/rqvae/train_rqvae_purity.py \
    --embeddings-path data/items_embeddings_1024d.parquet \
    --output-dir checkpoints/rqvae_purity_new \
    --seed 42
```

Training used a 95/5 train/validation split with seed 42. Exact bitwise
reproduction of the released checkpoint also depends on the GPU, CUDA,
PyTorch and Transformers versions; `requirements-train.txt` pins the versions
used for this release.

## Expected artifacts

| File | SHA-256 |
|---|---|
| `items_with_sids.parquet` | `ce3431c09daf158b16b3cb9456e6ffa8ddf114e7fc488d6a4f2d62683ed5b8eb` |
| `items_embeddings_1024d.parquet` | `e680f6ac4fe62be183e06ce769018d681be7ad2e2dfb67430c860b23866322fe` |
| `rqvae_purity_best_model.pth` | `7ef5abc07e58c832aba2699d390505dd9286044eb03825bc9849e60d4243ea25` |

A regenerated parquet may not be byte-identical because parquet metadata and
row ordering can differ. Validate the result by checking that
`semantic_id` equals the concatenation of `sid_0/sid_1/sid_2` for every item
(`python scripts/validate_release.py` performs this check on `data/`), and
compare the SID columns against the released file.
