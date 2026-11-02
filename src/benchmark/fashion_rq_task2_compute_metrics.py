"""
FASHION-RQ Task 2 — Hierarchical Relevance Classification metrics.

Predictions file (--predictions): JSON object {"<pair_id>": "<E|S|R|I>", ...}.
Pairs missing from the predictions file are scored as an incorrect guess (a placeholder
label distinct from all four classes) rather than being silently dropped, so an
incomplete submission is penalized instead of getting a smaller, easier denominator.

Metrics: accuracy, Macro-F1, and per-class precision/recall/F1/support, computed with
scikit-learn (already a project dependency).
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

import fashion_rq_common as common

LABELS = ["E", "S", "R", "I"]
MISSING_LABEL = "MISSING"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FASHION-RQ Task 2 predictions.")
    parser.add_argument(
        "--benchmark",
        default=str(common.SHIPPED_TASKS_DIR / "task2_classification_benchmark.jsonl.gz"),
    )
    parser.add_argument("--predictions", required=True, help="JSON: {pair_id: label}")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    records = common.read_jsonl(benchmark_path)
    predictions = common.load_json(Path(args.predictions))

    y_true = [r["label"] for r in records]
    y_pred = [predictions.get(r["pair_id"], MISSING_LABEL) for r in records]
    num_missing = sum(1 for p in y_pred if p == MISSING_LABEL)

    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    summary = {
        "num_pairs": len(records),
        "num_missing_predictions": num_missing,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in LABELS
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
    }

    output_path = Path(args.output) if args.output else benchmark_path.with_name(
        benchmark_path.stem + "_metrics.json"
    )
    common.write_json(output_path, summary)

    print(json.dumps({k: v for k, v in summary.items() if k != "confusion_matrix"}, indent=2))
    print(f"Wrote classification metrics -> {output_path}")


if __name__ == "__main__":
    main()
