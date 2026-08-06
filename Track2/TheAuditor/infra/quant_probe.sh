#!/usr/bin/env bash
# Which quantisation methods INITIALISE AT ALL on this card.
#
#   bash infra/quant_probe.sh Qwen/Qwen3-8B | tee bench/quant_probe.md
#   bash infra/quant_probe.sh <model> awq gptq            # subset
#
# This is an hour that saves a day. It does not measure quality, only whether a
# method loads and emits twenty non-garbage tokens, which eliminates half the
# sizing-sweep matrix before any of it is scheduled. The published caution is
# specific: AWQ has historically been weak on ROCm with no Marlin kernels and
# was for a period unsupported outright, and consumer RDNA3 (gfx1100) has no
# FP8 weight quantisation at all, which is why the plan is AWQ INT4 + BF16 and
# not INT4 + FP8. None of that is a substitute for finding out on this card.
#
# A NEGATIVE RESULT IS THE DELIVERABLE.
# "AWQ INT4 would not initialise on gfx1100 at vLLM x.y.z, here is the error"
# is worth more in the spec than one more successful row: it is evidence the
# quantisation choice was measured rather than assumed, which is exactly what
# the optimisation bonus is asking for. So every attempt is recorded, and the
# failures are recorded with their error text.
#
# Deliberately no `set -e`. A method that fails to load is the RESULT.

set -u
MODEL="${1:?usage: quant_probe.sh <model> [methods...]}"; shift || true
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8000}"
METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then METHODS=(none awq gptq bitsandbytes fp8); fi

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_USE_TRITON_FLASH_ATTN=1

echo "# Quantisation probe"
echo
echo "Model \`$MODEL\`, captured $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo "Target: \`$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-z]*' || echo unknown)\`"
echo "vLLM:   \`$(pip show vllm 2>/dev/null | awk '/^Version/{print $2}' || echo unknown)\`"
echo
echo "Twenty tokens per method. Quality is NOT measured here; the sizing sweep does that."
echo
echo '| method | loads | 20 tokens | VRAM | note |'
echo '|---|---|---|---|---|'

probe_one() {
  local method="$1" args="" log tag
  tag="${method}"
  [ "$method" = "none" ] && tag="bf16 (control)" || args="--quantization $method"
  log="$(mktemp)"

  # shellcheck disable=SC2086
  vllm serve "$MODEL" --port "$PORT" --max-model-len 2048 \
      --gpu-memory-utilization 0.85 $args > "$log" 2>&1 &
  local pid=$!

  # vLLM prints a startup banner when the HTTP server is live. Poll for the
  # endpoint rather than sleeping a fixed time: a cold weight download and a
  # warm cache differ by minutes, and a fixed sleep would report a slow load
  # as a failure.
  local ok="no" deadline=$((SECONDS + 900))
  while [ $SECONDS -lt $deadline ]; do
    if ! kill -0 $pid 2>/dev/null; then break; fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      ok="yes"; break
    fi
    sleep 5
  done

  local gen="-" vram="-" note=""
  if [ "$ok" = "yes" ]; then
    # bench/gpu_stats.py, not `grep -oE '[0-9]+' | tail -1`: tail -1 assumes
    # exactly one GPU line in the output and silently returns the WRONG
    # device's reading the moment rocm-smi reports a second device (a
    # workstation iGPU alongside the discrete card is common). Explicit
    # device 0 instead of "whichever line printed last".
    vram="$(rocm-smi --showmemuse 2>/dev/null \
            | python3 "$ROOT/bench/gpu_stats.py" memuse 0 2>/dev/null || echo '-')"
    gen="$(curl -sf "http://127.0.0.1:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"List three colours:\",\"max_tokens\":20,\"temperature\":0}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"].replace("|"," ").replace("\n"," ")[:60])' \
        2>/dev/null || echo 'EMPTY')"
    [ -z "$gen" ] && gen="EMPTY"
    case "$gen" in EMPTY|*"$(printf '\uFFFD')"*) note="loaded but produced nothing usable" ;; esac
  else
    # The last error line is the finding. Keep it: it is what goes in the spec.
    note="$(grep -iE 'error|not supported|no kernel|assert|Traceback' "$log" \
            | tail -1 | cut -c1-160 | tr '|' '/')"
    [ -z "$note" ] && note="did not come up within 900s"
  fi

  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 5                       # let VRAM actually come back before the next
  echo "| $tag | $ok | ${gen:--} | ${vram:--} | ${note:-} |"
  cp "$log" "$ROOT/bench/quant_probe_${method}.log" 2>/dev/null
}

for m in "${METHODS[@]}"; do probe_one "$m"; done

echo
cat <<'NOTE'
### Reading this table

- `none` is the control. If BF16 does not load, the problem is the environment,
  not the quantisation, and the Checkpoint 0 fallback fires: one model, BF16,
  stop sweeping, take the loss on the bonus rather than the 40 points beside it.
- `fp8` is expected to fail on gfx1100. Recording the failure is the point.
- A method that loads but emits `EMPTY` or replacement characters is a FAILURE
  even though it initialised, and is more dangerous than one that refuses to
  load, because a sweep would score it as a very bad model.
- Methods that survive here, and only those, go into the sizing sweep.
NOTE
