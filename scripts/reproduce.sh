#!/usr/bin/env bash
# End-to-end reproduction of the Fashion-RQBench query corpus and tasks.
#
#   data      catalog + semantic IDs       (scripts/download_data.py)
#   stage 1   cluster analysis + queries   (src/benchmark/generate_cluster_queries.py)
#   stage 2   per-query attributes         (src/benchmark/enrich_query_attributes.py)
#   stage 3   4 tasks + ground truth       (src/benchmark/fashion_rq_task*_build_*.py)
#   check     counts vs. the published figures (scripts/validate_release.py)
#
# The released corpus in benchmark/queries/ was produced with exactly these settings.
# Stage 1 is the only GPU-heavy step (~3 h 20 m for 588 clusters on one L4); stage 2
# takes ~15 min; stage 3 is GPU-free. Both LLM stages are resumable: stage 1 skips
# prefixes already in its output file, stage 2 skips queries already in its cache.
#
# Usage:
#   bash scripts/reproduce.sh                 # everything, reusing existing outputs
#   RUN_STAGE=3 bash scripts/reproduce.sh     # only one stage
#   DATA_BASE_URL=https://... bash scripts/reproduce.sh
#
# Environment
#   PYTHON          interpreter to use (default: python3 on PATH)
#   DATA_BASE_URL   where the catalog/embeddings live (see docs/DATA_CARD.md)
#   RUN_STAGE       one of: data, 1, 2, 3, validate, all (default: all)
#   See src/benchmark/README.md for the model/context env knobs (QWEN_*, TARGET_QUERY_RANGE).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SRC="$ROOT/src/benchmark"
OUT_QUERIES="$ROOT/benchmark/queries"
OUT_TASKS="$ROOT/benchmark/tasks_generated"
LOGS="$ROOT/logs"
RUN_STAGE="${RUN_STAGE:-all}"
mkdir -p "$OUT_QUERIES" "$OUT_TASKS" "$LOGS"

# Weights are expected in the local HF cache. Without this, an unauthenticated hub
# revalidation returns 429 and stalls ~30 s per metadata file.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-8B}"

# The released corpus used 9-10 queries per cluster. Qwen3-8B pins to the ceiling of
# whatever range it is given, so a wider range inflates the corpus (9-12 -> 11.7/cluster).
export TARGET_QUERY_RANGE="${TARGET_QUERY_RANGE:-9-10}"
export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-4096}"
export QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-1}"

BENCH_OUT="${BENCH_OUT:-$OUT_QUERIES/cluster_queries_raw.jsonl}"
ENRICHED_OUT="${ENRICHED_OUT:-$OUT_QUERIES/cluster_queries_regenerated_enriched.json}"
CACHE="$ROOT/.query_attr_cache_qwen3.json"

run() { echo; echo "=== $* ($(date -Is)) ==="; }

want() { [[ "$RUN_STAGE" == "all" || "$RUN_STAGE" == "$1" ]]; }

if want data; then
  # Only the catalog+SIDs are needed here; add items_embeddings_1024d.parquet with
  # `python scripts/download_data.py --only embeddings` if you intend to retrain the RQ-VAE.
  run "data: catalog + semantic IDs"
  if [[ -n "${DATA_BASE_URL:-}" ]]; then
    "$PYTHON" "$ROOT/scripts/download_data.py" --only sids --base-url "$DATA_BASE_URL"
  else
    "$PYTHON" "$ROOT/scripts/download_data.py" --only sids
  fi
fi

if want 1; then
  run "stage 1: cluster analysis + query generation"
  # YaRN factor 4 lifts Qwen3-8B's 40,960-token window to 163,840; the largest SID
  # cluster's prompt is ~56k tokens. Loading it x4 costs more VRAM but is required.
  QWEN_YARN_FACTOR="${QWEN_YARN_FACTOR:-4}" BENCH_OUT="$BENCH_OUT" \
    "$PYTHON" -u "$SRC/generate_cluster_queries.py" 2>&1 | tee "$LOGS/stage1.log"
fi

if want 2; then
  run "stage 2: per-query attribute extraction"
  QWEN_YARN_FACTOR=0 "$PYTHON" -u "$SRC/enrich_query_attributes.py" \
    --benchmark "$BENCH_OUT" --output "$ENRICHED_OUT" --cache "$CACHE" \
    --batch-size 16 --max-new-tokens 320 2>&1 | tee "$LOGS/stage2.log"
fi

if want 3; then
  run "stage 3: tasks + ground truth"
  export FASHION_RQ_ENRICHED_JSON="$ENRICHED_OUT"
  export FASHION_RQ_TASKS_DIR="$OUT_TASKS"
  for t in 1 2 3 4; do
    case $t in
      1) s=fashion_rq_task1_build_ranking.py;;
      2) s=fashion_rq_task2_build_classification.py;;
      3) s=fashion_rq_task3_build_sibling_discrimination.py;;
      4) s=fashion_rq_task4_build_paraphrase_robustness.py;;
    esac
    "$PYTHON" -u "$SRC/$s"
  done 2>&1 | tee "$LOGS/stage3.log"
fi

if want validate; then
  run "validate"
  "$PYTHON" "$ROOT/scripts/validate_release.py"
fi

echo
echo "Done. Regenerated corpus: $ENRICHED_OUT"
echo "Regenerated tasks:        $OUT_TASKS"
echo "Note: the checked-in corpus under benchmark/queries/ is left untouched."
