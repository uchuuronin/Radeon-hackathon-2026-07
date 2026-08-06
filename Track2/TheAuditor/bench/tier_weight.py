"""Open question 3 — measure TIER_WEIGHT, the one placeholder left in the
cost model.

    python bench/tier_weight.py runs/fast/c16.jsonl.manifest.json \\
                                runs/precise/c16.jsonl.manifest.json

TIER_WEIGHT is currently `{FAST: 1.0, PRECISE: 3.0}` in src/ladder.py, and the
3.0 is a guess. Two headline metrics are computed through it, "% of cases
resolved before the slow path" and "cost per document across a run", and the
~50% escalation kill-metric has nothing to sit against until it is real. There
is no published universal break-even for a cascade: it depends entirely on the
cost ratio between the two tiers and the accuracy gap between them, which is
why the ratio has to be measured on this card rather than borrowed.

WHAT "COST" MEANS HERE, AND WHY NOT LATENCY
------------------------------------------
The cost of a call is the CARD TIME it consumes, not the wall clock a caller
waits. Those diverge badly under continuous batching: at concurrency 1 a call
looks slow because the card is idle between decodes, and that latency figure
describes our client's round trip rather than the hardware. Two calls that each
take 400 ms wall clock can occupy the card for very different amounts of time.

So the estimator is inverse throughput at saturation:

    cost_per_call  =  1 / (docs per second at the knee)
    TIER_WEIGHT    =  throughput_fast / throughput_precise

Both runs must use the same documents, the same limit and the same concurrency,
at or above the concurrency knee from bench/throughput.sh. Below the knee the
card is not the bottleneck in either run and the ratio measures the client.

WHAT THIS IS NOT
----------------
It is not the cost of the expensive RUNG. That is

    expensive rung  =  escalation_rate x N x TIER_WEIGHT x cost_per_fast_call

N is the self-consistency sample count and is counted separately, in
inference_calls: an N-sample pass is N calls, not one, even though it batches
into a single request and finishes in roughly one generation of wall clock.
Folding N into TIER_WEIGHT would double-count it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    if "docs_per_s" not in m:
        raise SystemExit(f"{path} predates the manifest fix and has no "
                         f"docs_per_s. Re-run through src/extraction/run.py.")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("fast", type=Path, help="manifest from the FAST tier run")
    p.add_argument("precise", type=Path, help="manifest from the PRECISE run")
    a = p.parse_args()

    f, q = load(a.fast), load(a.precise)
    problems: list[str] = []

    if f["concurrency"] != q["concurrency"]:
        problems.append(
            f"different concurrency ({f['concurrency']} vs {q['concurrency']}). "
            f"The ratio would measure the batch schedule, not the tiers.")
    if set(f.get("doc_ids", [])) != set(q.get("doc_ids", [])):
        problems.append(
            "different document sets. Document length drives decode length, "
            "so a ratio across different corpora is not a tier ratio.")
    for label, m in (("fast", f), ("precise", q)):
        if m["ok"] < 0.9 * m["documents"]:
            problems.append(
                f"{label}: only {m['ok']}/{m['documents']} calls succeeded. "
                f"Wall clock includes the failures, so throughput is "
                f"understated and the ratio is not trustworthy.")
        if m.get("truncated"):
            problems.append(
                f"{label}: {m['truncated']} responses hit max_tokens, which "
                f"cuts decode short and flatters that tier's throughput.")
        if m["ok"] < 30:
            problems.append(
                f"{label}: {m['ok']} documents is thin for a ratio that the "
                f"whole cost model multiplies through. Prefer 60.")

    if not f["docs_per_s"] or not q["docs_per_s"]:
        raise SystemExit("a run recorded zero throughput; nothing to divide.")

    weight = f["docs_per_s"] / q["docs_per_s"]

    print(f"\n  fast    {f['model']:<40} {f['docs_per_s']:.3f} docs/s "
          f"({1000 / f['docs_per_s']:.0f} ms of card per call)")
    print(f"  precise {q['model']:<40} {q['docs_per_s']:.3f} docs/s "
          f"({1000 / q['docs_per_s']:.0f} ms of card per call)")
    print(f"  both at concurrency {f['concurrency']} on "
          f"{f['ok']}/{f['documents']} and {q['ok']}/{q['documents']} documents")
    print(f"\n  TIER_WEIGHT = {weight:.2f}   (placeholder in ladder.py is 3.0)")

    if weight < 1.0:
        print("\n  The precise tier is FASTER than the fast tier. That is a "
              "\n  real finding and it collapses the cascade: if the better "
              "\n  model is also cheaper there is no tier to route between. "
              "\n  Check the quantisation actually engaged before believing it "
              "\n  — an unsupported method that silently fell back to BF16 "
              "\n  produces exactly this number.")
    elif weight < 1.3:
        print("\n  The tiers are within 30% of each other. A cascade cannot "
              "\n  save much when the rungs cost nearly the same, and the "
              "\n  routing overhead is not free. Collapsing to the precise "
              "\n  model and reporting that is a legitimate, reportable "
              "\n  outcome — cascades have been measured scoring BELOW "
              "\n  always-using-the-larger-model while costing more.")
    else:
        break_even = 1.0 / weight
        print(f"\n  At this ratio the cascade stops saving compute once the "
              f"escalation\n  rate passes roughly {break_even:.0%} for a "
              f"single-sample precise pass,")
        for n in (3, 5):
            print(f"    or roughly {break_even / n:.0%} at N={n} "
                  f"self-consistency samples.")
        print("  That is the kill-metric, derived rather than assumed. The "
              "~50%\n  figure in the plan is a target, not an industry "
              "constant.")

    if problems:
        print("\n  BEFORE USING THIS NUMBER:")
        for x in problems:
            print(f"    - {x}")

    print(f"\n  To adopt it, edit src/ladder.py:\n"
          f"      TIER_WEIGHT[Rung.PRECISE_TIER] = {weight:.2f}\n"
          f"  and re-run anything that reported cost per document: every "
          f"such figure\n  already recorded was computed against 3.0 and is "
          f"void.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
