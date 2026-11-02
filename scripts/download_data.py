#!/usr/bin/env python3
"""
Fetch the catalog + semantic-ID data for Fashion-RQBench.

The code and the query/task benchmark ship inside this repository; the item data
is hosted separately because it is too large for version control (the embeddings
alone are 169 MB). This script downloads the files into `data/`, verifies their
SHA-256 against a pinned manifest, and refuses to leave a partial download in
place.

  items_with_sids.parquet        16 MB   REQUIRED -- catalog + sid_0/sid_1/sid_2
                                          + semantic_id for all 44,446 items.
                                          Needed to build or score any task.
  items_embeddings_1024d.parquet 169 MB  OPTIONAL -- Qwen3-Embedding-0.6B item
                                          embeddings. Only needed to retrain the
                                          RQ-VAE from scratch (src/rqvae/).

Usage:
  python scripts/download_data.py                       # both files
  python scripts/download_data.py --only sids           # just the 16 MB catalog
  python scripts/download_data.py --base-url https://... --out-dir /somewhere

Hosting: point --base-url (or $FASHION_RQ_DATA_BASE_URL) at wherever the two
files live -- a HuggingFace dataset repo, a Zenodo record, or any static host.
`release_data/` in the authors' working copy contains the exact files to upload
plus SHA256SUMS and manifest.json. See docs/DATA_CARD.md.
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned so a download cannot silently change under a release. Update these when
# the hosted files are re-cut.
FILES = {
    "sids": {
        "name": "items_with_sids.parquet",
        "sha256": "ce3431c09daf158b16b3cb9456e6ffa8ddf114e7fc488d6a4f2d62683ed5b8eb",
        "bytes": 16228222,
        "required": True,
    },
    "embeddings": {
        "name": "items_embeddings_1024d.parquet",
        "sha256": "e680f6ac4fe62be183e06ce769018d681be7ad2e2dfb67430c860b23866322fe",
        "bytes": 169157398,
        "required": False,
    },
}

DEFAULT_BASE_URL = os.environ.get("FASHION_RQ_DATA_BASE_URL", "")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def fetch(url: str, dest: Path, expected_bytes: int) -> None:
    """Stream to a .part file, then move into place, so a failure leaves nothing half-written."""
    part = dest.with_suffix(dest.suffix + ".part")
    downloaded = 0
    shown = -1
    with urllib.request.urlopen(url) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or expected_bytes)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            # repaint only when the whole percentage changes, so a slow link does not
            # produce a line per megabyte in a captured log
            pct = int(100 * downloaded / total) if total else 0
            if pct != shown:
                shown = pct
                sys.stdout.write(f"\r    {human(downloaded)} / {human(total)}  ({pct:3d}%)")
                sys.stdout.flush()
    sys.stdout.write(f"\r    {human(downloaded)} / {human(total)}  (100%)\n")
    shutil.move(str(part), str(dest))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="Base URL holding the data files (or $FASHION_RQ_DATA_BASE_URL)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--only", choices=["sids", "embeddings", "all"], default="all")
    ap.add_argument("--verify-only", action="store_true",
                    help="Check existing files against the pinned hashes; download nothing")
    ap.add_argument("--force", action="store_true", help="Re-download even if present and valid")
    args = ap.parse_args()

    targets = list(FILES.values()) if args.only == "all" else [FILES[args.only]]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.verify_only and not args.base_url:
        print("No --base-url given and $FASHION_RQ_DATA_BASE_URL is unset.\n")
        print("Point --base-url at wherever the data files are hosted, e.g.")
        print("  python scripts/download_data.py \\")
        print("      --base-url https://huggingface.co/datasets/<user>/fashion-rqbench/resolve/main")
        print("\nSee docs/DATA_CARD.md for hosting options and how to build the bundle.")
        return 2

    failures = []
    for spec in targets:
        dest = args.out_dir / spec["name"]
        label = "required" if spec["required"] else "optional"
        print(f"[{spec['name']}] ({label}, {human(spec['bytes'])})")

        if dest.exists() and not args.force:
            actual = sha256(dest)
            if actual == spec["sha256"]:
                print("    already present, checksum OK")
                continue
            print(f"    present but checksum MISMATCH (have {actual[:12]}…, want {spec['sha256'][:12]}…)")

        if args.verify_only:
            failures.append(spec["name"])
            continue

        url = f"{args.base_url.rstrip('/')}/{spec['name']}"
        print(f"    fetching {url}")
        try:
            fetch(url, dest, spec["bytes"])
        except Exception as exc:  # noqa: BLE001 - report and continue to the next file
            print(f"    FAILED: {exc}")
            failures.append(spec["name"])
            continue

        actual = sha256(dest)
        if actual == spec["sha256"]:
            print("    checksum OK")
        else:
            print(f"    checksum MISMATCH — removing {dest}")
            print(f"      expected {spec['sha256']}")
            print(f"      got      {actual}")
            dest.unlink(missing_ok=True)
            failures.append(spec["name"])

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll requested files present and verified in", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
