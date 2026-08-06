"""Tests for bench/gpu_stats.py. No GPU, no network -- this is exactly the
"debug small functionality first" pass: prove the parser against realistic
rocm-smi text now, so infra/quant_probe.sh and bench/throughput.sh are
correct on the first real run instead of the first debugging session.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import gpu_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MOCK = ROOT / "tests" / "fixtures" / "mock_rocm_smi.py"

SHOWUSE_SHOWMEMUSE_1GPU = """\
======================= ROCm System Management Interface =======================
================================================================================
GPU[0]\t\t: GPU use (%): 87
================================================================================
============================== Current Memory Use ==============================
GPU[0]\t\t: GPU memory use (%): 34
================================================================================
============================= End of ROCm SMI Log ==============================
"""

SHOWMEMUSE_2GPU = """\
======================= ROCm System Management Interface =======================
============================== Current Memory Use ==============================
GPU[0]\t\t: GPU memory use (%): 34
GPU[1]\t\t: GPU memory use (%): 2
================================================================================
============================= End of ROCm SMI Log ==============================
"""

SHOWMEMUSE_NEWER_WORDING = """\
======================= ROCm System Management Interface =======================
============================== Current Memory Use ==============================
GPU[0]\t\t: GPU Memory Allocated (VRAM%): 41
================================================================================
============================= End of ROCm SMI Log ==============================
"""


def test_parses_combined_use_and_memuse_correctly():
    assert gpu_stats.device_use(SHOWUSE_SHOWMEMUSE_1GPU) == 87
    assert gpu_stats.device_memuse(SHOWUSE_SHOWMEMUSE_1GPU) == 34


def test_resolves_correct_device_on_multi_gpu_output():
    # tail -1 on the raw grep would return device 1's reading (2), not
    # device 0's (34) -- the card we are actually serving on.
    assert gpu_stats.device_memuse(SHOWMEMUSE_2GPU, device=0) == 34
    assert gpu_stats.device_memuse(SHOWMEMUSE_2GPU, device=1) == 2


def test_newer_label_wording_still_parses():
    assert gpu_stats.device_memuse(SHOWMEMUSE_NEWER_WORDING) == 41


def test_missing_device_returns_none_not_a_wrong_number():
    assert gpu_stats.device_use(SHOWUSE_SHOWMEMUSE_1GPU, device=3) is None


def test_header_and_footer_never_contribute_a_spurious_reading():
    banner_only = "======================= ROCm System Management Interface =======================\n"
    assert gpu_stats.parse_use(banner_only) == {}
    assert gpu_stats.parse_memuse(banner_only) == {}


# ---------------------------------------------------------------------------
# Reproduce the two bugs the old grep/tail parsing had, so a regression back
# to that approach is caught even if nobody rereads the working brief.
# ---------------------------------------------------------------------------

def test_reproduces_throughput_sh_index_bug():
    """The line throughput.sh actually built from real output ("0 87 0 34")
    and the value it recorded as utilisation (parts[0] = 0, the GPU index,
    not 87, the real use %). Locks the failure mode in so nobody re-adds the
    blind digit-grab."""
    import re
    digits = re.findall(r"[0-9]+", SHOWUSE_SHOWMEMUSE_1GPU)
    old_parts_line = " ".join(digits)
    old_parts = [int(x) for x in old_parts_line.split() if x.isdigit()]
    assert old_parts[0] == 0  # the bug: this is what the old code recorded
    assert old_parts[0] != 87  # 87 was the real GPU use %
    # the new parser gets it right on the same input
    assert gpu_stats.device_use(SHOWUSE_SHOWMEMUSE_1GPU) == 87


def test_reproduces_quant_probe_sh_tail_bug_on_multi_gpu():
    import re
    digits = re.findall(r"[0-9]+", SHOWMEMUSE_2GPU)
    old_tail_minus_1 = int(digits[-1])
    assert old_tail_minus_1 == 2       # the bug: device 1's reading
    assert old_tail_minus_1 != 34      # 34 was device 0's, the served card
    assert gpu_stats.device_memuse(SHOWMEMUSE_2GPU, device=0) == 34


# ---------------------------------------------------------------------------
# End-to-end: the real mock binary through the real CLI entrypoint, the same
# way the shell scripts will call it. This is the part that runs "before the
# instance does".
# ---------------------------------------------------------------------------

def _run(cmd, env):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)


def test_cli_end_to_end_against_mock_binary(tmp_path):
    env = dict(os.environ)
    env["ROCM_SMI_USE"] = "87"
    env["ROCM_SMI_MEMUSE"] = "34"

    smi = _run([sys.executable, str(MOCK), "--showuse", "--showmemuse"], env)
    # feed smi.stdout into gpu_stats.py via stdin
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "gpu_stats.py"), "use"],
        input=smi.stdout, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "87"

    proc2 = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "gpu_stats.py"), "memuse"],
        input=smi.stdout, capture_output=True, text=True, env=env,
    )
    assert proc2.returncode == 0
    assert proc2.stdout.strip() == "34"


def test_cli_missing_reading_exits_nonzero_not_zero(tmp_path):
    """A device that never reports must fail loudly (nonzero exit, no
    stdout) -- never silently print '0', which is indistinguishable from a
    real idle reading."""
    env = dict(os.environ)
    env["ROCM_SMI_USE"] = "87"
    smi = _run([sys.executable, str(MOCK), "--showuse"], env)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "gpu_stats.py"), "use", "5"],
        input=smi.stdout, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 1
    assert proc.stdout.strip() == ""
