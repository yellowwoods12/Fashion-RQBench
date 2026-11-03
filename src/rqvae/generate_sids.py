#!/usr/bin/env python3
"""
Encode a catalog with a trained RQ-VAE checkpoint and attach its semantic IDs.

Reads item embeddings, runs the encoder + residual quantization, and writes the
catalog back out with four extra columns:

    sid_0, sid_1, sid_2   the codebook index chosen at each quantization level
    semantic_id           "<|sid_{sid_0}|><|sid_{sid_1}|><|sid_{sid_2}|>"

This is the same transformation that produced the released
`items_with_sids.parquet`, included so the SID set can be regenerated (for a
different checkpoint, or a retrained model) rather than only downloaded.

Two inputs, because the release ships them separately:

    --embeddings  parquet with `id` + `embedding`   (169 MB, the encoder's input)
    --catalog     parquet with item attributes      (16 MB, where SIDs are written)

If your embeddings file also carries the attribute columns, pass the same path to
both. Rows are matched on `id`, so the two files need not be in the same order;
they must cover the same id set.

Adapted from the script that produced the released SID parquet: same
checkpoint format, same output columns, with paths parameterised and the attribute
labels -- needed only to compute the SupCon loss during training -- no longer
loaded, since inference needs embeddings alone.

Usage:
  python generate_sids.py --checkpoint checkpoints/rqvae_purity/best_model.pth
"""

import argparse
import sys
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# -- pathlib._local compat shim (checkpoints pickled under a newer Python) -------
_pathlib_local = types.ModuleType("pathlib._local")
_pathlib_local.Path = Path
_pathlib_local.PosixPath = Path
_pathlib_local.WindowsPath = Path
sys.modules.setdefault("pathlib._local", _pathlib_local)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_rqvae_purity import RQVAEPurity, RQVAEPurityConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SID_COLUMNS = ("sid_0", "sid_1", "sid_2", "semantic_id")


def load_embeddings(path: Path) -> pl.DataFrame:
    """(id, embedding) as float32, in file order -- the row order the encoder sees."""
    df = pl.read_parquet(path, columns=["id", "embedding"])
    return df.with_columns(pl.col("id").cast(pl.Utf8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to best_model.pth (or any step checkpoint)")
    ap.add_argument("--embeddings", type=Path,
                    default=REPO_ROOT / "data" / "items_embeddings_1024d.parquet")
    ap.add_argument("--catalog", type=Path,
                    default=REPO_ROOT / "data" / "items_with_sids.parquet")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output parquet (default: <catalog>_regenerated.parquet)")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    for label, path in (("checkpoint", args.checkpoint), ("embeddings", args.embeddings),
                        ("catalog", args.catalog)):
        if not path.exists():
            raise SystemExit(
                f"{label} not found: {path}\n"
                "Fetch the released files with `python scripts/download_data.py` "
                "(use --only embeddings for the 169 MB embeddings parquet)."
            )
    out_path = args.out or args.catalog.with_name(args.catalog.stem + "_regenerated.parquet")

    print(f"[1/4] Loading embeddings from {args.embeddings} …")
    emb_df = load_embeddings(args.embeddings)
    n_items = emb_df.height
    matrix = torch.tensor(np.asarray(emb_df["embedding"].to_list(), dtype=np.float32),
                          dtype=torch.float32)
    print(f"    {n_items:,} items, embedding dim {matrix.shape[1]}")

    print(f"[2/4] Loading checkpoint {args.checkpoint} …")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = RQVAEPurityConfig(**{k: v for k, v in raw["config"].items()
                               if k != "embeddings_path"})
    model = RQVAEPurity(cfg)
    state = raw["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):   # torch.compile wrapper
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    print(f"    step {raw['step']:,}, val_loss {raw['val_loss']:.4e}, "
          f"levels {cfg.codebook_quantization_levels} x {cfg.codebook_size} codes")

    print(f"[3/4] Encoding on {args.device} …")
    batches = []
    with torch.no_grad():
        for start in range(0, n_items, args.batch_size):
            batch = matrix[start:start + args.batch_size].to(args.device)
            batches.append(model.encode_to_semantic_ids(batch).cpu().numpy())
    sids = np.concatenate(batches, axis=0).astype(np.int32)

    print(f"[4/4] Joining onto the catalog and writing {out_path} …")
    codes = emb_df.select("id").with_columns([
        pl.Series("sid_0", sids[:, 0], dtype=pl.Int32),
        pl.Series("sid_1", sids[:, 1], dtype=pl.Int32),
        pl.Series("sid_2", sids[:, 2], dtype=pl.Int32),
    ])

    catalog = pl.read_parquet(args.catalog)
    if catalog["id"].dtype != pl.Utf8:
        catalog = catalog.with_columns(pl.col("id").cast(pl.Utf8))
    catalog = catalog.drop([c for c in SID_COLUMNS if c in catalog.columns])
    before = catalog.height
    merged = catalog.join(codes, on="id", how="left")
    if merged.height != before:
        raise SystemExit(f"join changed the row count ({before} -> {merged.height}); "
                         "the catalog has duplicate ids")
    unmatched = int(merged["sid_0"].null_count())
    if unmatched:
        raise SystemExit(f"{unmatched} catalog items have no embedding "
                         f"(ids present in {args.catalog.name} but not {args.embeddings.name})")

    merged = merged.with_columns(
        pl.format("<|sid_{}|><|sid_{}|><|sid_{}|>",
                  pl.col("sid_0"), pl.col("sid_1"), pl.col("sid_2")).alias("semantic_id")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path)

    used = [int(np.unique(sids[:, lvl]).shape[0]) for lvl in range(3)]
    distinct = len(set(map(tuple, sids)))
    print(f"    codebook utilisation (of {cfg.codebook_size} per level): {used}")
    print(f"    distinct semantic IDs: {distinct:,} across {n_items:,} items "
          f"({100 * distinct / n_items:.1f}%)")
    print("    Score it with `python src/rqvae/compute_hierarchy_metrics.py "
          f"--parquet {out_path}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
