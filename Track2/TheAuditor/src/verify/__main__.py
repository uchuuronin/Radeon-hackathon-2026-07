"""`python -m verify <files>` — the standalone validator CLI.

bean-check parity: a checker that runs with NO model anywhere in the picture.
It reads ExtractedRecord JSON files or JSONL batches, verifies every document,
prints a human-readable report, writes an optional verified.jsonl audit file,
and exits non-zero if anything FAILED.

This is the end-of-Day-2 deliverable from the plan:
    python -m verify data/fixtures/records/*.json
    python -m verify data/generated/records.jsonl --out runs/dev/verified.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from schemas import CheckOutcome, ExtractedRecord
from ladder import CaseTrace, Rung, record, summarise

from .engine import verify_doc


def _iter_records(paths: list[str]):
    for raw in paths:
        # Windows shells do not expand globs; do it ourselves. Split the
        # pattern from its (possibly absolute) parent — Path.glob refuses
        # absolute patterns.
        if any(c in raw for c in "*?["):
            p = Path(raw)
            matched = sorted(p.parent.glob(p.name))
        else:
            matched = [Path(raw)]
        if not matched:
            print(f"warning: no files match {raw}", file=sys.stderr)
        for p in matched:
            text = p.read_text(encoding="utf-8").strip()
            if p.suffix == ".jsonl":
                for line in text.splitlines():
                    if line.strip():
                        yield p, ExtractedRecord.model_validate_json(line)
            else:
                yield p, ExtractedRecord.model_validate_json(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="verify",
        description="Deterministic document verifier (no GPU, no model).")
    ap.add_argument("paths", nargs="+",
                    help="ExtractedRecord .json files or .jsonl batches; "
                         "globs allowed")
    ap.add_argument("--out", type=Path, default=None,
                    help="write VerificationReports as JSONL (the audit file)")
    ap.add_argument("--trace", type=Path, default=None,
                    help="write CaseTraces as JSONL (exit rung, timing, "
                         "inference accounting)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="summary line only")
    args = ap.parse_args(argv)

    reports = []
    traces: list[CaseTrace] = []
    n_docs = 0
    outcomes: Counter[str] = Counter()
    failed_docs = 0

    for path, rec in _iter_records(args.paths):
        # Rung 0. A case that is STRICT here is resolved for free — it never
        # reaches the GPU, and that is what the headline metric counts.
        trace = CaseTrace(case_id=rec.doc.doc_id)
        with record(trace, Rung.DETERMINISTIC) as ev:
            report = verify_doc(rec.doc)
            ev.resolved = report.strict_pass
            ev.note = "strict pass" if report.strict_pass else (
                "escalates: within-tolerance or unresolved"
                if report.verify_pass else "escalates: failed check")
        traces.append(trace)
        reports.append(report)
        n_docs += 1
        for c in report.checks:
            outcomes[c.outcome] += 1
        ok = report.verify_pass
        strict = report.strict_pass
        if not ok:
            failed_docs += 1
        if not args.quiet:
            flag = "STRICT" if strict else ("PASS" if ok else "FAIL")
            print(f"[{flag:>6}] {rec.doc.doc_id:<12} {rec.doc.doc_type:<16} "
                  f"{rec.doc.doc_number:<14} ({path.name})")
            if not ok:
                for c in report.checks:
                    if c.outcome == CheckOutcome.FAIL:
                        loc = f" at {c.field_path}" if c.field_path else ""
                        print(f"         FAIL {c.check}{loc}: "
                              f"expected {c.expected}, got {c.actual}  "
                              f"{c.message}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(r.model_dump_json() for r in reports) + "\n",
            encoding="utf-8")

    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(
            "\n".join(t.model_dump_json() for t in traces) + "\n",
            encoding="utf-8")

    print()
    print(summarise(traces).render())
    print(f"\n{n_docs} documents · "
          f"pass {outcomes[CheckOutcome.PASS]} · "
          f"within-tolerance {outcomes[CheckOutcome.WITHIN_TOLERANCE]} · "
          f"fail {outcomes[CheckOutcome.FAIL]} · "
          f"skipped {outcomes[CheckOutcome.SKIPPED]} checks · "
          f"{failed_docs} document(s) failed")
    return 1 if failed_docs else 0


if __name__ == "__main__":
    raise SystemExit(main())
