#!/usr/bin/env bash
# The sizing sweep, deliberately small.
#
#   bash bench/sweep.sh <served-model-name> <label>
#
# Run this ONCE PER SERVED CONFIGURATION, with the server already up. It scores
# 60 documents and appends a row to bench/tier_selection.md.
#
# It scores 60 rather than 20 because 20 documents is roughly 100 line-item
# numeric fields, and a Wilson interval on 100 is about +/-6 points, which is
# the width at which two configurations become distinguishable. Below that you
# are choosing on noise. Above ~200 documents you are spending instance time to
# narrow an interval that is already narrow enough to decide with.
set -eu
#
# VRAM and UTILISATION are not guessable from inside this process. Get them
# from bench/throughput.sh (which samples rocm-smi alongside a run) and pass
# them in, or the row lands saying so. A throughput figure without the
# utilisation it was measured at is not a systems result, and the row prints
# UTILISATION NOT RECORDED rather than quietly omitting it.
#
#   VRAM_GB=21.4 UTIL=82 bash bench/sweep.sh <model> <label>
MODEL="${1:?model}"; LABEL="${2:-$1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
OUT="$ROOT/runs/${LABEL// /_}"
mkdir -p "$OUT"

python "$ROOT/src/extraction/run.py" \
  --records "$ROOT/data/generated/records.jsonl" \
  --out "$OUT/extracted.jsonl" \
  --model "$MODEL" --limit 60 --layout layout_a --concurrency 8 \
  --base-url "${BASE_URL:-http://localhost:8000/v1}" \
  --prompt-id "$LABEL" | tee "$OUT/run.log"

# --append-row emits ONE comparable row: config, guided mode, scoring mode, n,
# line-item numeric, identifier, doc numeric, precision loss, cache, docs/s at
# utilisation, peak VRAM, run health. A free-form code block per configuration
# reads fine and cannot be compared row to row, which is how a tier gets picked
# on a point estimate whose interval overlaps the alternative.
python "$ROOT/bench/score.py" \
  --truth "$ROOT/data/generated/records.jsonl" \
  --got "$OUT/extracted.jsonl" --layout layout_a \
  --label "$LABEL" \
  --append-row "$ROOT/bench/tier_selection.md" \
  ${VRAM_GB:+--vram-gb "$VRAM_GB"} ${UTIL:+--utilisation "$UTIL"} \
  | tee "$OUT/score.md"
