"""CHECKPOINT 0, as one command.

    python bench/gate.py --model <served-model-name>

Runs the whole gate end to end and prints a verdict per condition. It exists
because the gate has three conditions owned by two people and the only way to
know it is closed is to run all three against the real instance; a checklist
someone ticks by hand is not evidence.

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

    # --- A: labelled dataset in >= 2 layouts ------------------------------
    if not a.records.exists():
        verdicts.append(("A  labelled dataset, >=2 layouts", False,
                         "run data/generator/gen.py first"))
    else:
        import json
        layouts, n = set(), 0
        for line in a.records.read_text(encoding="utf-8").splitlines():
            if line.strip():
                layouts.add(json.loads(line)["meta"].get("layout"))
                n += 1
        key = a.records.parent / "answer_key.json"
        verdicts.append(("A  labelled dataset, >=2 layouts", len(layouts) >= 2 and key.exists(),
                         f"{n} documents, layouts {sorted(x for x in layouts if x)}, "
                         f"answer key {'present' if key.exists() else 'MISSING'}"))

    # --- B1: card characterised -------------------------------------------
    versions = ROOT / "infra/versions.md"
    ok = versions.exists() and "gfx" in versions.read_text(encoding="utf-8", errors="ignore")
    verdicts.append(("B1 card characterised", ok,
                     "infra/versions.md present with a gfx target" if ok
                     else "run: bash infra/characterise.sh | tee infra/versions.md"))

    # --- B2: serving endpoint answers -------------------------------------
    from extraction.client import LocalVLLM
    llm = LocalVLLM(model=a.model, base_url=a.base_url)
    serving, msg = llm.health()
    verdicts.append(("B2 vLLM serving (THE gate)", serving, msg))

    # --- B4/B5: extraction produces scoreable records ----------------------
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
        s = score_files(a.records, out, layout="layout_a")
        acc, lo, hi = s.rate("line_numeric", "relaxed")
        verdicts.append(("B5 line-item numeric accuracy", s.n["line_numeric"] > 0,
                         f"{acc:.1%} [{lo:.1%},{hi:.1%}] on n={s.n['line_numeric']} fields, "
                         f"{stats.cache_hit_rate:.0%} prefix cache"))
        print(s.render(f"{a.model} — gate sample"))
        print(s.top_misses(5))
        print(stats.render())
    else:
        verdicts.append(("B5 line-item numeric accuracy", False,
                         "blocked on B2"))

    print("\n" + "=" * 78)
    print("CHECKPOINT 0")
    print("=" * 78)
    for name, ok, note in verdicts:
        print(f"[{OK if ok else NO}] {name:<34} {note}")
    closed = all(v[1] for v in verdicts)
    print("=" * 78)
    print("GATE CLOSED — proceed to Stage 1." if closed else
          "GATE OPEN. If B2 is red, fall back to BF16 on one model NOW and "
          "stop sweeping.")
    return 0 if closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
