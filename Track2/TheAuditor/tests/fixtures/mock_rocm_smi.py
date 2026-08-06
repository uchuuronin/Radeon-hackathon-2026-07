#!/usr/bin/env python3
"""A fake rocm-smi that prints real-format output, for testing the parsers
in bench/gpu_stats.py and the shell scripts that call them, without a card.

Usage (put this on PATH as `rocm-smi`, e.g. `ln -sf mock_rocm_smi.py
/tmp/fakebin/rocm-smi && chmod +x` and prepend /tmp/fakebin to PATH):

    ROCM_SMI_USE=87 ROCM_SMI_MEMUSE=34 rocm-smi --showuse --showmemuse
    ROCM_SMI_DEVICES=2 ROCM_SMI_MEMUSE=34,2 rocm-smi --showmemuse

Values come from env vars so a test can drive many scenarios (idle card,
saturated card, a second reporting device) without many fixture files.
Output format matches real rocm-smi text mode (confirmed against
ROCm/ROCm#1562 and the rocm-smi CLI docs): a banner, one line per device per
metric shaped "GPU[N]\t\t: <label>: <value>", and a footer -- all of which
carry digits themselves (the [N] index, dates if --showtime is added later),
which is exactly what makes the naive `grep -oE '[0-9]+'` parsing unsafe.
"""
from __future__ import annotations

import os
import sys


def _values(var: str, n: int, default: int) -> list[int]:
    raw = os.environ.get(var, "")
    if not raw:
        return [default] * n
    parts = [int(x) for x in raw.split(",")]
    if len(parts) < n:
        parts += [default] * (n - len(parts))
    return parts[:n]


def main() -> int:
    args = sys.argv[1:]
    n = int(os.environ.get("ROCM_SMI_DEVICES", "1"))
    print("======================= ROCm System Management Interface =======================")

    if "--showuse" in args:
        use = _values("ROCM_SMI_USE", n, 0)
        print("================================================================================")
        for i, v in enumerate(use):
            print(f"GPU[{i}]\t\t: GPU use (%): {v}")

    if "--showmemuse" in args:
        memuse = _values("ROCM_SMI_MEMUSE", n, 0)
        print("================================================================================")
        print("============================== Current Memory Use ==============================")
        for i, v in enumerate(memuse):
            print(f"GPU[{i}]\t\t: GPU memory use (%): {v}")

    print("================================================================================")
    print("============================= End of ROCm SMI Log ==============================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
