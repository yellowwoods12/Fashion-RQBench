# data/

Empty until you fetch it. Nothing here is versioned (see `../.gitignore`).

```bash
python scripts/download_data.py --base-url <DATA_BASE_URL>          # catalog + SIDs (16 MB)
python scripts/download_data.py --only embeddings --base-url <...>  # + embeddings (169 MB)
python scripts/download_data.py --verify-only                       # check what you have
```

| File | Size | SHA-256 | Needed for |
|---|---|---|---|
| `items_with_sids.parquet` | 16 MB | `ce3431c0…d5b8eb` | building or scoring any task |
| `items_embeddings_1024d.parquet` | 169 MB | `e680f6ac…6322fe` | retraining the RQ-VAE only |

Checksums are pinned in `scripts/download_data.py`; a file that does not match is deleted
rather than left in place. Full hashes are in `release_data/SHA256SUMS`.

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
