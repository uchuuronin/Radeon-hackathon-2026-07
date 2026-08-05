#!/usr/bin/env bash
# Characterise the card and pin every version, in one pass.
#
# Run this FIRST on the Radeon instance, before serving anything. It writes
# infra/versions.md, which becomes the README environment section and is a
# submission requirement (reproducibility). It also answers the two questions
# that decide the whole quantisation plan: which gfx target, and how much VRAM.
#
#   bash infra/characterise.sh | tee infra/versions.md
#
# Deliberately no `set -e`: a missing tool is DATA. If rocm-smi is absent that
# is the single most important line in the file, and aborting would hide it.

echo "# Environment"
echo
echo "Captured $(date -u +%Y-%m-%dT%H:%M:%SZ) on \`$(hostname)\`."
echo
echo '## Card'
echo '```'
rocm-smi --showproductname --showmeminfo vram 2>&1 | sed 's/^/  /'
echo '```'
echo
echo '## Compute target (decides quantisation options)'
echo '```'
rocminfo 2>/dev/null | grep -E 'gfx|Marketing Name' | sort -u | sed 's/^/  /'
echo '```'
echo
echo '## Torch / ROCm'
echo '```'
python -c "
import torch, sys
print('python  ', sys.version.split()[0])
print('torch   ', torch.__version__)
print('hip     ', getattr(torch.version, 'hip', None))
print('cuda_ok ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device  ', torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print('vram_gb ', round(total / 1024**3, 1), 'free', round(free / 1024**3, 1))
" 2>&1 | sed 's/^/  /'
echo '```'
echo
echo '## Pinned versions'
echo '```'
pip list 2>/dev/null | grep -iE 'vllm|torch|transformers|triton|rocm|xgrammar|openai' | sed 's/^/  /'
echo '```'
echo
echo '## Serving environment'
echo '```'
echo "  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
echo "  VLLM_USE_TRITON_FLASH_ATTN=1"
echo '```'
