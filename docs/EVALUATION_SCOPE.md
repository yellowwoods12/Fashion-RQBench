# What evaluation code is included

The repository ships the **task-level** evaluation code described in
[TASKS.md](TASKS.md). It does not ship every analysis used in the paper. This
page makes the boundary explicit so the release is not over-claimed.

## Included

| Component | Where |
|---|---|
| Four task files with ground truth | `benchmark/tasks/` |
| Task 1 scorer — nDCG@5/10, Recall@5/10, overall + per attribute tag | `src/benchmark/fashion_rq_task1_compute_metrics.py` |
| Task 2 scorer — accuracy, macro-F1, per-class precision/recall/F1 | `src/benchmark/fashion_rq_task2_compute_metrics.py` |
| Task 3 scorer — nDCG@1/3, Recall@1/3, selection accuracy | `src/benchmark/fashion_rq_task3_compute_metrics.py` |
| Task 4 scorer — Consistency@K, per-paraphrase nDCG@10 mean/stdev | `src/benchmark/fashion_rq_task4_compute_metrics.py` |
| Task builders (GPU-free, seeded) | `src/benchmark/fashion_rq_task*_build_*.py` |
| Release integrity checks | `scripts/validate_release.py` |
| Worked example | `examples/` |

Prediction formats and metric definitions are in [TASKS.md](TASKS.md).

## Not included

- **Weighted query relevance `R(q,p)`** and the per-attribute match functions
  used in the paper's evaluation framework.
- **Cross-encoder-derived attribute weights** (`w_a`).
- **Baseline model configurations and result files** for the retrieval models in
  the paper's comparison tables.

The task-level scorers are sufficient to evaluate a model on the four released
tasks. The additional analyses require code that is not part of this release;
cite the paper for their definitions and contact the authors if you need them.

## Running the shipped evaluation

```bash
# Task 1
python src/benchmark/fashion_rq_task1_compute_metrics.py \
    --predictions preds.json --output t1.metrics.json

# Task 2
python src/benchmark/fashion_rq_task2_compute_metrics.py \
    --predictions preds.json --output t2.metrics.json

# Task 3
python src/benchmark/fashion_rq_task3_compute_metrics.py \
    --predictions preds.json --output t3.metrics.json

# Task 4
python src/benchmark/fashion_rq_task4_compute_metrics.py \
    --predictions preds.json --output t4.metrics.json
```

See [examples/README.md](../examples/README.md) for a complete load → predict →
score walkthrough on Task 1.
