# Changelog

## 1.0.0 — 2026-11-03

First public release.

- **Semantic IDs** — RQ-VAE with a per-level attribute exclusivity loss (3 levels × 64 codes),
  producing 1,270 leaf clusters over 44,446 items. Hierarchy quality: purity 88.8 / 81.1 /
  64.9% (L1/L2/L3), coverage 99.3%, fragmentation 12.5. See `docs/HIERARCHY_QUALITY.md`.
- **Query corpus** — 5,807 queries over 588 clusters (17 / 135 / 436), authored by Qwen3-8B
  from each cluster's attribute distribution; 99.1% unique strings, mean 11.2 words. Five
  queries were rewritten before release to remove audience mentions their cluster does not
  contain; each rewrite is recorded under `audience_repairs`.
- **Tasks** — four tasks with ground truth: ranking (4,103 queries), classification
  (32,688 pairs), sibling discrimination (3,262 queries), paraphrase robustness (416 groups /
  4,103 queries), plus a scorer for each.
- **Tooling** — `scripts/download_data.py` (checksum-pinned data fetch),
  `scripts/reproduce.sh` (end-to-end regeneration), `scripts/validate_release.py`
  (release gate), `src/rqvae/compute_hierarchy_metrics.py` (hierarchy quality).
- **Packaging** — the catalog, embeddings and RQ-VAE checkpoint are published as v1.0.0
  release assets (16 MB / 169 MB / 17 MB); the benchmark artifacts ship in-repo gzipped
  (90 MB raw → 6 MB). The four task files rebuild byte-identically from the shipped corpus.
