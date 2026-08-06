"""Parse rocm-smi text output without grep -oE '[0-9]+'.

WHY THIS EXISTS
----------------
Two of infra/quant_probe.sh and bench/throughput.sh's rocm-smi parsers were
blind digit-grabs over the whole output:

    rocm-smi --showmemuse | grep -oE '[0-9]+' | tail -1
    rocm-smi --showuse --showmemuse | grep -oE '[0-9]+' | paste -sd' '

Both are wrong on real rocm-smi output, not hypothetically:

  - Every device line is prefixed "GPU[0]", "GPU[1]", ... The index itself
    is a digit, so it enters the same match stream as the percentage you
    actually wanted. In the combined --showuse --showmemuse case this puts
    the GPU *index* (0) in front of the GPU *use %* value on the line, and
    throughput.sh's `parts[0]` records the index as "mean utilisation" --
    silently, every run, regardless of the real number. See
    tests/test_gpu_stats.py::test_reproduces_throughput_sh_index_bug for the
    reproduction.
  - `tail -1` assumes exactly one GPU line in the output. The moment
    rocm-smi reports a second device (a workstation iGPU alongside the
    discrete card is common), tail -1 silently reports the wrong device.

A wrong number is worse than a crash here, because it goes straight into
tier_selection.md / the throughput table with no error to notice. This
module replaces the digit-grab with a parser that matches on the label text
rocm-smi actually prints ("GPU use (%)", "GPU memory use (%)"), keyed by
device index, so:
  - header/footer noise ("====", "ROCm System Management Interface") never
    contributes a spurious digit,
  - the value returned is always the one from the line that names it, not
    positional guesswork,
  - multi-device output resolves to the requested device index instead of
    "whatever line happened to be last".

No dependencies. Runs identically off a real rocm-smi or the mock in
tests/fixtures/mock_rocm_smi.py.
"""

from __future__ import annotations

import re
import sys
from typing import Optional

# rocm-smi has changed its label wording across versions. Match loosely
# ("gpu" + "use" + "%" in some order on the same line) rather than pinning
# one exact string, so a wording change degrades to "no match found" (loud)
# instead of "matched the wrong thing" (silent).
_USE_RE = re.compile(r"GPU\[(\d+)\]\s*:\s*GPU use \(%\):\s*(\d+)", re.IGNORECASE)
_MEMUSE_RE = re.compile(
    r"GPU\[(\d+)\]\s*:\s*GPU memory use \(%\):\s*(\d+)", re.IGNORECASE
)
# Newer rocm-smi wording seen in the wild: "GPU Memory Allocated (VRAM%)"
_MEMUSE_ALT_RE = re.compile(
    r"GPU\[(\d+)\]\s*:.*Memory Allocated \(VRAM%\):\s*(\d+)", re.IGNORECASE
)


def _by_device(pattern: re.Pattern, text: str) -> dict[int, int]:
    return {int(idx): int(val) for idx, val in pattern.findall(text)}


def parse_use(text: str) -> dict[int, int]:
    """{device_index: GPU use %} from `rocm-smi --showuse` output."""
    return _by_device(_USE_RE, text)


def parse_memuse(text: str) -> dict[int, int]:
    """{device_index: GPU memory use %} from `rocm-smi --showmemuse` output."""
    result = _by_device(_MEMUSE_RE, text)
    if not result:
        result = _by_device(_MEMUSE_ALT_RE, text)
    return result


def device_use(text: str, device: int = 0) -> Optional[int]:
    return parse_use(text).get(device)


def device_memuse(text: str, device: int = 0) -> Optional[int]:
    return parse_memuse(text).get(device)


def _main(argv: list[str]) -> int:
    """CLI: `python3 gpu_stats.py use|memuse [device] < rocm-smi-output`.

    Prints a single integer, or exits 1 with nothing on stdout if the
    requested device/metric was not found -- an empty result must be a
    visible failure to the calling shell script, not a "0" that looks like
    a real 0% reading.
    """
    if not argv or argv[0] not in ("use", "memuse"):
        print("usage: gpu_stats.py use|memuse [device_index]", file=sys.stderr)
        return 2
    device = int(argv[1]) if len(argv) > 1 else 0
    text = sys.stdin.read()
    value = device_use(text, device) if argv[0] == "use" else device_memuse(text, device)
    if value is None:
        print(f"gpu_stats: no '{argv[0]}' reading for device {device} in input",
              file=sys.stderr)
        return 1
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
