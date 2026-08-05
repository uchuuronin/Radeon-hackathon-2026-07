#!/usr/bin/env bash
# B5 — the sizing sweep, deliberately small.
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
MODEL="${1:?model}"; LABEL="${2:-$1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
OUT="$ROOT/runs/${LABEL// /_}"
mkdir -p "$OUT"

python "$ROOT/src/extraction/run.py" \
  --records "$ROOT/data/generated/records.jsonl" \
  --out "$OUT/extracted.jsonl" \
  --model "$MODEL" --limit 60 --layout layout_a --concurrency 8 \
  --prompt-id "$LABEL" | tee "$OUT/run.log"

python "$ROOT/bench/score.py" \
  --truth "$ROOT/data/generated/records.jsonl" \
  --got "$OUT/extracted.jsonl" --layout layout_a \
  --label "$LABEL" | tee "$OUT/score.md"

{ echo; echo "### $LABEL"; echo '```'
  cat "$OUT/score.md"; echo; grep -E 'throughput|prefix cache|latency' "$OUT/run.log" || true
  echo '```'; } >> "$ROOT/bench/tier_selection.md"
echo "appended -> bench/tier_selection.md"
