#!/usr/bin/env bash
# Serve a model. THE gate.
#
#   bash infra/serve.sh Qwen/Qwen3-8B            # BF16, the safe first move
#   bash infra/serve.sh <model> --quantization awq
#
# Get ANY model answering before you get the RIGHT model answering. A served
# small model is 40 points of "core inference on AMD Radeon GPU"; a perfect
# quantisation plan that never served is zero.

set -u
MODEL="${1:?usage: serve.sh <model> [extra vllm args]}"; shift || true

# Consumer RDNA3 needs the Triton attention path; without these vLLM either
# falls back to something slow or fails to start, and the error does not
# obviously point here.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_USE_TRITON_FLASH_ATTN=1

# --enable-prefix-caching is the single highest-value flag we set. Our frozen
# prefix is ~3.5k tokens against a ~400 token document, so on a cache hit we
# skip roughly 90% of prefill on every document after the first.
#
# --max-model-len 8192 is deliberate, not lazy: KV cache is allocated against
# it, so a needlessly large window buys nothing and costs the batch size that
# actually keeps the card busy. Our longest prompt is well under 5k.
exec vllm serve "$MODEL" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-seqs 32 \
  --disable-log-requests \
  "$@"
