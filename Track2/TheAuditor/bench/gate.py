"""CHECKPOINT 0, as one command.

    python bench/gate.py --model <served-model-name>

Runs the whole gate end to end and prints a verdict per condition. It exists
because the only way to know the gate is closed is to run every condition
against the real instance; a checklist someone ticks by hand is not evidence.

Everything it needs is already in the repo. It spends about twenty extraction
calls, which on the free shared endpoint is free and on a served model is
seconds. Run it before the sweep, not after.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Everything below this line in bench/tier_selection.md is machine-appended
#: measurement. Everything above it is prose about what the columns mean.
ROW_MARKER = "<!-- rows are appended below this line by bench/sweep.sh -->"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OK, NO = "  GREEN  ", "   RED   "


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--n", type=int, default=20, help="documents to score")
    p.add_argument("--records", type=Path,
                   default=ROOT / "data/generated/records.jsonl")
    a = p.parse_args()

    verdicts: list[tuple[str, bool, str]] = []

    # --- labelled dataset in >= 2 layouts ---------------------------------
    if not a.records.exists():
        verdicts.append(("labelled dataset, >=2 layouts", False,
                         "run data/generator/gen.py first"))
    else:
        import json
        layouts, n = set(), 0
        for line in a.records.read_text(encoding="utf-8").splitlines():
            if line.strip():
                layouts.add(json.loads(line)["meta"].get("layout"))
                n += 1
        key = a.records.parent / "answer_key.json"
        verdicts.append(("labelled dataset, >=2 layouts", len(layouts) >= 2 and key.exists(),
                         f"{n} documents, layouts {sorted(x for x in layouts if x)}, "
                         f"answer key {'present' if key.exists() else 'MISSING'}"))

    # --- the harness itself runs ------------------------------------------
    # Cheapest condition and the one that used to be assumed. Every artefact
    # on the GPU side was written and unit-tested but had never executed against a
    # live socket, and "written and tested" is not "run once". bench/selftest.py
    # closes that on loopback with no GPU, so a failure here is a harness bug
    # being found for free rather than on metered instance time.
    selftest = ROOT / "runs/selftest/perfect/extracted.jsonl"
    verdicts.append(("harness exercised end to end", selftest.exists(),
                     "bench/selftest.py has run" if selftest.exists()
                     else "run: python bench/selftest.py   (no GPU needed)"))

    # --- card characterised -----------------------------------------------
    versions = ROOT / "infra/versions.md"
    ok = versions.exists() and "gfx" in versions.read_text(encoding="utf-8", errors="ignore")
    verdicts.append(("card characterised", ok,
                     "infra/versions.md present with a gfx target" if ok
                     else "run: bash infra/characterise.sh | tee infra/versions.md"))

    # --- serving endpoint answers -----------------------------------------
    from extraction.client import LocalVLLM
    llm = LocalVLLM(model=a.model, base_url=a.base_url)
    serving, msg = llm.health()
    verdicts.append(("vLLM serving (THE gate)", serving, msg))

    # --- quantisation methods probed --------------------------------------
    # Not a Checkpoint 0 condition in its own right, but the condition that IS
    # one ("a model/quantisation pair selected on measured accuracy") cannot be
    # honestly met without it, and a method that never initialised is a real
    # finding that belongs in the spec rather than a gap in the table.
    probe = ROOT / "bench/quant_probe.md"
    verdicts.append(("quantisation probed on this card", probe.exists(),
                     "bench/quant_probe.md present" if probe.exists()
                     else "run: bash infra/quant_probe.sh <model> | "
                          "tee bench/quant_probe.md"))

    # --- extraction produces scoreable records ----------------------------
    if serving and a.records.exists():
        from bench.score import score_files
        from extraction.run import extract_many, load_sources
        from schemas import Tier
        docs = load_sources(a.records, a.n, layout="layout_a")
        recs, stats = extract_many(docs, llm, Tier.FAST, concurrency=4,
                                   prompt_id="gate")
        out = ROOT / "runs/gate/extracted.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(r.model_dump_json() for r in recs) + "\n",
                       encoding="utf-8")
        s = score_files(a.records, out, layout="layout_a",
                        attempted=set(stats.doc_ids))
        acc, lo, hi = s.rate("line_numeric", "relaxed")
        verdicts.append(("line-item numeric accuracy", s.n["line_numeric"] > 0,
                         f"{acc:.1%} [{lo:.1%},{hi:.1%}] on n={s.n['line_numeric']} fields, "
                         f"{stats.cache_hit_rate:.0%} prefix cache"))
        print(s.render(f"{a.model} — gate sample"))
        print(s.top_misses(5))
        print(stats.render())
    else:
        verdicts.append(("line-item numeric accuracy", False,
                         "blocked on the serving gate"))

    # --- a PAIR was selected, not just one model measured -----------------
    # The gate's own wording is "a model/quantisation pair is selected on
    # measured extraction accuracy". One configuration measured is not a
    # selection: there is nothing it was chosen over. Two rows whose Wilson
    # intervals overlap are also not a selection, because the measurement did
    # not separate them, but that judgement is a human one and the gate only
    # insists the rows exist.
    # Count ONLY below the marker. The seeded file documents its own column
    # contract in a markdown table, and a naive "lines starting with |" count
    # reads those 13 explanatory rows as 13 measured configurations: the gate
    # would go green on a file where no sweep had ever run. A gate that can
    # pass without the work is worse than no gate, because it is trusted.
    table = ROOT / "bench/tier_selection.md"
    rows = 0
    if table.exists():
        text = table.read_text(encoding="utf-8")
        appended = text.split(ROW_MARKER, 1)[1] if ROW_MARKER in text else ""
        rows = sum(1 for line in appended.splitlines()
                   if line.startswith("| ") and "| config |" not in line
                   and not line.startswith("|---"))
    verdicts.append(("tier PAIR selected on measurement", rows >= 2,
                     f"bench/tier_selection.md has {rows} comparable rows"
                     if rows else
                     "bench/tier_selection.md has no rows: run bench/sweep.sh "
                     "once per configuration"))
    verdicts.append(("line-item numeric accuracy", False,"blocked on the serving gate"))

    # --- a PAIR was selected, not just one model measured -----------------
    # The gate's own wording is "a model/quantisation pair is selected on
    # measured extraction accuracy". One configuration measured is not a
    # selection: there is nothing it was chosen over. Two rows whose Wilson
    # intervals overlap are also not a selection, because the measurement did
    # not separate them, but that judgement is a human one and the gate only
    # insists the rows exist.
    table = ROOT / "bench/tier_selection.md"
    rows = 0
    if table.exists():
        rows = sum(1 for line in table.read_text(encoding="utf-8").splitlines()
                   if line.startswith("| ") and "| config |" not in line
                   and not line.startswith("|---"))
    verdicts.append(("tier PAIR selected on measurement", rows >= 2,
                     f"bench/tier_selection.md has {rows} comparable rows"
                     if rows else
                     "bench/tier_selection.md has no rows: run bench/sweep.sh "
                     "once per configuration"))

    print("\n" + "=" * 78)
    print("CHECKPOINT 0")
    print("=" * 78)
    for i, (name, ok, note) in enumerate(verdicts, 1):
        print(f"[{OK if ok else NO}] {i}. {name:<32} {note}")
    closed = all(v[1] for v in verdicts)
    print("=" * 78)
    if closed:
        print("GATE CLOSED — proceed to Stage 1.")
    else:
        print("GATE OPEN.")
        if not serving:
            print("Serving is red, so the plan's own fallback fires NOW, not after "
                  "a sweep:\n"
                  "  one model, BF16, no quantisation matrix. Take the loss on "
                  "the 20-point\n"
                  "  bonus rather than the 40 points for local inference and "
                  "optimisation that\n"
                  "  sit beside it. A late fallback costs both.")
    return 0 if closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
