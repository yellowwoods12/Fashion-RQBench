# data/

Empty until you fetch it. Nothing here is versioned (see `../.gitignore`). The two files
are distributed separately from the repository; see
[../docs/DATA_CARD.md#data-availability](../docs/DATA_CARD.md#data-availability).

```bash
python scripts/download_data.py                    # all three artifacts
python scripts/download_data.py --only sids        # catalog + SIDs (16 MB)
python scripts/download_data.py --verify-only      # check what you have
```

| File | Local path | Size | SHA-256 | Needed for |
|---|---|---|---|---|
| `items_with_sids.parquet` | `data/` | 16 MB | `ce3431c0…d5b8eb` | regenerating queries or building tasks |
| `items_embeddings_1024d.parquet` | `data/` | 169 MB | `e680f6ac…6322fe` | retraining the RQ-VAE |
| `rqvae_purity_best_model.pth` | `checkpoints/rqvae_purity/` | 17 MB | `7ef5abc0…43ea25` | re-encoding items with the released checkpoint |

Checksums are pinned in `scripts/download_data.py`; a file that does not match is deleted
rather than left in place.

**`items_with_sids.parquet`** — 44,446 products with the catalog attributes plus
`sid_0`, `sid_1`, `sid_2` and `semantic_id`. Schema, provenance and limitations:
[../docs/DATA_CARD.md](../docs/DATA_CARD.md). You do not need this file to *score* a task —
each task record carries its own item metadata; you need it to regenerate queries or build
new tasks.

**`items_embeddings_1024d.parquet`** — `id` + 1024-d `embedding` per item
(Qwen3-Embedding-0.6B over `item_context`). Only the RQ-VAE trainer and
`src/rqvae/generate_sids.py` read it.

Override the location with `SID_ITEMS_PARQUET`, or point the whole tree elsewhere with
`FASHION_RQ_DATA_DIR`.
