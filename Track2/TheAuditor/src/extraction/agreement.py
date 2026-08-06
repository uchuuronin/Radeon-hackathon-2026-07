"""Field-level agreement across sampled extractions. No GPU.

WHAT THIS IS FOR
----------------
`ladder.route()` takes an `agreement` score and nothing produced one. This does.

Agreement is the learned half of the confidence signal. The other half,
deterministic verification, needs no calibration and cannot be miscalibrated:
the identities hold or they do not. This half is different. Its ORDERING is
robust (documents whose samples disagree really are the ones more likely to be
wrong) but its absolute value is not, which is why the thresholds it feeds are
placed on a risk-coverage curve rather than picked.

WHY IT RUNS OFF DISK
--------------------
The N samples are recorded once, on the card, by `run.py --samples N`. Every
question after that (agreement, risk-coverage, calibration, threshold
selection, cost) is answered here, on a laptop, forever. GPU hours buy a frozen
dataset, not results.

WHAT COUNTS AS AGREEMENT
------------------------
Field by field across the samples, then averaged over fields. NOT
whole-record equality: one differing character in a party name would make an
otherwise unanimous extraction score zero, which throws away exactly the
gradation the routing thresholds need.

Numeric fields compare by VALUE, so "4500" and "4500.00" agree. That is
deliberate and it is the opposite of what the extraction scorer does, because
the two are answering different questions. The scorer asks "did the model obey
the output contract", where stated precision matters because the verifier
derives its tolerance band from it. This asks "do the samples mean the same
thing", where they plainly do.

Line items are keyed by line_id, never by position. A sample that drops a line
shifts every subsequent index, and positional comparison would report total
disagreement for one missing row.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional, Sequence

from schemas import CanonicalDoc, ExtractedRecord

#: Compared for agreement. Deliberately the fields a reconciliation decision
#: actually rests on. `source_text` and `doc_id` are supplied by the harness
#: and identical by construction, so including them would inflate every score
#: towards 1.0 and flatten the signal the thresholds are placed on.
DOC_FIELDS: tuple[str, ...] = (
    "doc_number", "doc_type", "party_name", "doc_date", "currency",
    "subtotal", "allowance_total", "charge_total", "total_excl_tax", "tax",
    "total", "paid_amount", "rounding_amount", "amount_due", "payment_terms",
)
LINE_FIELDS: tuple[str, ...] = ("description", "quantity", "unit_price",
                                "line_total")


def _norm(value) -> object:
    """Compare numbers by value, everything else by normalised string.

    None is its own category rather than being folded into "" or 0: a sample
    that says a field is ABSENT and one that says it is ZERO disagree, and that
    disagreement is exactly the open question about goods receipts.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value.normalize()
    if isinstance(value, str):
        s = value.strip()
        try:
            return Decimal(s).normalize()
        except (InvalidOperation, ValueError):
            return " ".join(s.split()).casefold()
    return value


def _fields(doc: CanonicalDoc) -> dict[str, object]:
    """Flatten a record to comparable (path -> value) pairs."""
    out: dict[str, object] = {f: _norm(getattr(doc, f, None)) for f in DOC_FIELDS}
    for li in (doc.line_items or []):
        for f in LINE_FIELDS:
            out[f"line_items[{li.line_id}].{f}"] = _norm(getattr(li, f, None))
    return out


def agreement(samples: Sequence[CanonicalDoc]) -> float:
    """Mean per-field modal agreement across N samples, in [0, 1].

    For each field, the share of samples holding the most common value. A field
    only one sample mentions counts as disagreement on the others, because a
    hallucinated line is a disagreement about whether the line exists at all.

    One sample returns 1.0. That is honest rather than convenient: with N=1
    there is no self-consistency evidence, and the routing decision then rests
    entirely on deterministic verification, which is exactly the intended
    fallback when sampling proves too expensive.
    """
    live = [s for s in samples if s is not None]
    if len(live) <= 1:
        return 1.0
    per_sample = [_fields(s) for s in live]
    paths = sorted({p for f in per_sample for p in f})
    if not paths:
        return 1.0

    total = 0.0
    for path in paths:
        counts: dict[object, int] = defaultdict(int)
        for f in per_sample:
            counts[f.get(path, "__absent__")] += 1
        total += max(counts.values()) / len(per_sample)
    return total / len(paths)


def disagreeing_fields(samples: Sequence[CanonicalDoc]) -> list[str]:
    """Which fields the samples did not agree on, worst first.

    The routing score says a document is uncertain; this says WHERE, which is
    what makes an escalation actionable rather than just expensive.
    """
    live = [s for s in samples if s is not None]
    if len(live) <= 1:
        return []
    per_sample = [_fields(s) for s in live]
    scored: list[tuple[float, str]] = []
    for path in sorted({p for f in per_sample for p in f}):
        counts: dict[object, int] = defaultdict(int)
        for f in per_sample:
            counts[f.get(path, "__absent__")] += 1
        share = max(counts.values()) / len(per_sample)
        if share < 1.0:
            scored.append((share, path))
    return [p for _, p in sorted(scored)]


def group_samples(records: Iterable[ExtractedRecord]
                  ) -> dict[str, list[CanonicalDoc]]:
    """Regroup a flat JSONL of samples back into per-document sets.

    `run.py --samples N` writes one line per sample, so a 60-document run at
    N=5 is 300 lines. The grouping key is doc_id; `meta.n_sample_index` only
    orders them within a document.
    """
    out: dict[str, list[tuple[Optional[int], CanonicalDoc]]] = defaultdict(list)
    for r in records:
        out[r.doc.doc_id].append((r.meta.n_sample_index, r.doc))
    return {k: [d for _, d in sorted(v, key=lambda x: (x[0] is None, x[0]))]
            for k, v in out.items()}
