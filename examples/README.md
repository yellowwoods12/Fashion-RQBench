# Examples

A runnable end-to-end example: load Task 1, rank each query's fixed candidate
pool with BM25, write predictions, and score them with the shipped metric
script.

## BM25 baseline (Task 1)

The baseline is pure Python (no retrieval library). The scorer uses
`scikit-learn`, so install the core requirements first:

```bash
pip install -r requirements.txt

# 1. produce predictions for all 4,103 Task-1 queries
python examples/bm25_baseline.py

# 2. score them
python src/benchmark/fashion_rq_task1_compute_metrics.py \
    --predictions examples/predictions/task1_example.json \
    --output /tmp/task1_metrics.json
```

Expected output for the checked-in predictions file:

```json
{
  "ndcg@5": 0.7707459195595936,
  "ndcg@10": 0.8186902118748394,
  "recall@5": 0.41803847640431024,
  "recall@10": 0.7676138056164824,
  "num_queries": 4103,
  "num_missing_predictions": 0
}
```

`--limit N` is available for a quick syntax/runtime check, but the scorer assumes
full coverage: queries missing from the predictions file are scored as zero. Do
not use a limited predictions file for metrics.

```bash
python examples/bm25_baseline.py --limit 25 --output /tmp/task1_preds_25.json
```

## What this example does and does not show

BM25 ranks **the candidates already in each query's pool**; it does not search
the full 44,446-item catalog. That is exactly what Task 1 asks a model to do,
and it makes the example fast and dependency-free. The scores above are a
sanity check, not a leaderboard result and not directly comparable to the
paper's baseline table, which was computed on a different corpus version.
