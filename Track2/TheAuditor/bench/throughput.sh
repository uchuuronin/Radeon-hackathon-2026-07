#!/usr/bin/env bash
# The 20-point optimisation number: docs/sec AT A STATED GPU UTILISATION.
#
#   bash bench/throughput.sh <served-model-name> <label>
#
# Run with the server already up. Sweeps --concurrency and samples rocm-smi
# alongside each run, so every throughput figure arrives with the utilisation
# it was measured at and the VRAM it cost.
#
# WHY SWEEP RATHER THAN PICK
# --------------------------
# vLLM's continuous batching has nothing to batch unless requests are in flight
# together. At concurrency 1 the card is idle between decodes and the resulting
# docs/sec describes our client's round-trip time, not the hardware. Throughput
# climbs with concurrency until the batch saturates, then flattens while p95
# latency keeps climbing. The interesting number is the knee, and the knee is
# specific to the model, the quantisation and the KV budget, so it cannot be
# assumed from anyone else's measurement.
#
# WHY THE UTILISATION MATTERS AS MUCH AS THE THROUGHPUT
# ----------------------------------------------------
# "N docs/sec" at 40% utilisation is not a systems result, it is an unfinished
# one: it says most of the card was idle and a bigger batch was available. The
# claim being made is efficient use of a single Radeon, and that claim needs
# both halves of the pair.
set -u
MODEL="${1:?usage: throughput.sh <served-model-name> <label>}"
LABEL="${2:-$1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
CONCURRENCIES="${CONCURRENCIES:-1 4 8 16 32}"
LIMIT="${LIMIT:-60}"
OUT="$ROOT/runs/throughput_${LABEL// /_}"
mkdir -p "$OUT"

# rocm-smi sampled at 1 Hz for the duration of one run. Mean, not peak:
# a peak is one instant and every run touches 100% briefly during a decode
# burst. The mean is what "we kept the card busy" actually means.
sample_gpu() {
  local out="$1"
  : > "$out"
  while :; do
    rocm-smi --showuse --showmemuse 2>/dev/null \
      | grep -oE '[0-9]+' | paste -sd' ' >> "$out"
    sleep 1
  done
}

echo "| concurrency | docs/s | mean util % | peak util % | p50 ms | p95 ms | cache | ok/sent |"
echo "|---|---|---|---|---|---|---|---|"

for C in $CONCURRENCIES; do
  SAMPLES="$OUT/gpu_c${C}.txt"
  sample_gpu "$SAMPLES" & SAMPLER=$!

  python "$ROOT/src/extraction/run.py" \
    --records "$ROOT/data/generated/records.jsonl" \
    --out "$OUT/c${C}.jsonl" --model "$MODEL" \
    --limit "$LIMIT" --layout layout_a --concurrency "$C" \
    --prompt-id "$LABEL-c$C" > "$OUT/c${C}.log" 2>&1

  kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null

  python - "$OUT/c${C}.jsonl.manifest.json" "$SAMPLES" "$C" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
util = []
for line in Path(sys.argv[2]).read_text().splitlines():
    parts = [int(x) for x in line.split() if x.isdigit()]
    if parts:
        util.append(parts[0])
mean = sum(util) / len(util) if util else 0
peak = max(util) if util else 0
print(f"| {sys.argv[3]} | {m['docs_per_s']:.2f} | {mean:.0f} | {peak} | "
      f"{m['latency_p50_ms']:.0f} | {m['latency_p95_ms']:.0f} | "
      f"{m['cache_hit_rate']:.0%} | {m['ok']}/{m['documents']} |")
PY
done

cat <<'NOTE'

### Reading this table

- Pick the concurrency where docs/s stops improving. Past the knee you are
  buying latency with no throughput, and p95 is where that shows up first.
- Mean utilisation below roughly 70% at the knee means the batch is not the
  limit: check --max-num-seqs and --max-num-batched-tokens on the server
  before concluding anything about the model.
- A cache hit rate that FALLS as concurrency rises means the prefix is being
  evicted; raise --gpu-memory-utilization or lower --max-model-len, since KV
  cache is allocated against the context window whether it is used or not.
- Report the chosen row, not the best cell from different rows.
NOTE
