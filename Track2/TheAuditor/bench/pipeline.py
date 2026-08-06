"""Frozen extraction JSONL -> link -> reconcile -> ChainVerdict JSONL. No GPU.

    python bench/pipeline.py --extracted runs/frozen/fast_layout_a.jsonl \\
                             --out runs/frozen/verdicts_fast_a.jsonl
    python bench/detect_score.py --verdicts runs/frozen/verdicts_fast_a.jsonl

THE PIECE THAT MAKES THE RECORDING WORTH ANYTHING
-------------------------------------------------
GPU hours buy a frozen dataset, not results. This is what turns the dataset
back into results, and it is the reason the recording session can be short: the
model is asked once, its answers are written to disk, and every question after
that is answered here, on a laptop, as many times as anyone likes.

It also closes the loop the corpus was built for. Up to now the deterministic
pipeline has only ever been scored on GROUND-TRUTH records, where extraction is
perfect by construction. That measures reconciliation in isolation, which is a
real result and not the claimed one. Running the same pipeline on records a
model actually produced measures the thing the spec promises, and the gap
between the two numbers is the extraction tier's contribution to end-to-end
error, stated rather than hidden.

N SAMPLES COLLAPSE TO ONE RECORD, DELIBERATELY
----------------------------------------------
`run.py --samples 5` writes five lines per document. Reconciliation needs one
record per document, so the samples are collapsed by majority vote per field
(see `--vote`), and the agreement score is kept alongside as the routing
signal. Voting is not an accuracy trick: it is what self-consistency IS. The
alternative, reconciling five times and voting on verdicts, costs five times
the work and throws away which FIELD the samples disagreed about, which is the
thing that makes an escalation actionable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extraction.agreement import (agreement, disagreeing_fields,  # noqa: E402
                                  group_samples)
from harmonize.reconcile import DEFAULT_POLICY, reconcile_all  # noqa: E402
from linker.link import link                                    # noqa: E402
from schemas import (CanonicalDoc, ChainVerdict, ExtractedRecord,  # noqa: E402
                     LineItem)


def load_extracted(path: Path) -> list[ExtractedRecord]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(ExtractedRecord.model_validate_json(line))
        except Exception as exc:                                # noqa: BLE001
            # A malformed line is a finding about the recording, not something
            # to skip silently: losing documents quietly makes every rate
            # downstream wrong in a direction nobody can see.
            print(f"  [pipeline] {path.name} line {i} did not validate: "
                  f"{type(exc).__name__}. Skipped, and the count below is "
                  f"short by one.")
    return out


def _vote_field(values: Sequence) -> object:
    """Modal value, ties broken towards the first sample.

    First rather than "most precise" or "largest": the first sample is the one
    a single-sample run would have produced, so a tie resolves to the answer
    the cheap path would have given. Any other rule would make N=5 and N=1
    disagree on documents where the samples were evenly split, which is exactly
    where the comparison between them needs to be honest.
    """
    counts = Counter(str(v) for v in values)
    top = max(counts.values())
    for v in values:
        if counts[str(v)] == top:
            return v
    return values[0]


def collapse(samples: Sequence[CanonicalDoc]) -> CanonicalDoc:
    """N sampled extractions -> one record, by per-field majority.

    Per FIELD, not per record. Picking the modal whole record throws away every
    field the majority agreed on whenever one field differed, which on a
    fifteen-field document is most of the evidence.

    Line items are voted per line_id, and a line only a minority saw is
    dropped: a line three of five samples never mentioned is more likely
    hallucinated than missed, and inventing a line item creates money that does
    not exist.
    """
    if len(samples) == 1:
        return samples[0]
    base = samples[0].model_copy(deep=True)

    for fname in type(base).model_fields:
        if fname in ("doc_id", "source_text", "line_items"):
            continue
        vals = [getattr(s, fname, None) for s in samples]
        setattr(base, fname, _vote_field(vals))

    seen: dict[str, list[LineItem]] = defaultdict(list)
    for s in samples:
        for li in (s.line_items or []):
            seen[li.line_id].append(li)

    voted: list[LineItem] = []
    for lid in sorted(seen):
        if len(seen[lid]) * 2 <= len(samples):
            continue                    # a minority of samples saw this line
        parts = seen[lid]
        merged = parts[0].model_copy(deep=True)
        for fname in type(merged).model_fields:
            if fname == "line_id":
                continue
            merged_val = _vote_field([getattr(p, fname, None) for p in parts])
            setattr(merged, fname, merged_val)
        voted.append(merged)
    base.line_items = voted
    return base


def build_verdicts(records: Iterable[ExtractedRecord], vote: bool = True
                   ) -> tuple[list[ChainVerdict], dict[str, float], dict]:
    """The whole deterministic pipeline, in one place.

    Returns (verdicts, agreement per doc_id, run stats).
    """
    grouped = group_samples(records)
    n_samples = max((len(v) for v in grouped.values()), default=1)

    agreements: dict[str, float] = {}
    docs: dict[str, CanonicalDoc] = {}
    for doc_id, samples in grouped.items():
        agreements[doc_id] = agreement(samples) if n_samples > 1 else 1.0
        docs[doc_id] = collapse(samples) if vote else samples[0]

    chains = link(list(docs.values()))
    verdicts = reconcile_all(chains, docs, DEFAULT_POLICY)

    # Agreement is per DOCUMENT; routing decides per CASE. The weakest document
    # in a chain sets the chain's confidence, because one badly-read invoice is
    # enough to make the whole verdict wrong, and averaging would let four
    # clean documents hide it.
    for v in verdicts:
        low = min((agreements.get(i, 1.0) for i in v.doc_ids), default=1.0)
        worst = min(v.doc_ids, key=lambda i: agreements.get(i, 1.0)) \
            if v.doc_ids else ""
        v.trace.append(f"chain agreement {low:.4f} (weakest document {worst}; "
                       f"the weakest sets the chain, an average would let "
                       f"clean documents hide a badly-read one)")

    stats = {
        "documents": len(docs),
        "samples_per_document": n_samples,
        "chains": len(chains),
        "singletons": sum(1 for c in chains if c.is_singleton),
        "verdicts": len(verdicts),
        "clean_chains": sum(1 for v in verdicts if v.is_clean),
        "discrepancies": sum(len(v.discrepancies) for v in verdicts),
        "mean_agreement": (sum(agreements.values()) / len(agreements)
                           if agreements else 1.0),
    }
    return verdicts, agreements, stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--extracted", type=Path, required=True,
                   help="JSONL of ExtractedRecord from src/extraction/run.py")
    p.add_argument("--out", type=Path, required=True,
                   help="JSONL of ChainVerdict, for bench/detect_score.py")
    p.add_argument("--agreement-out", type=Path, default=None,
                   help="JSON of doc_id -> agreement, for src/calibrate.py")
    p.add_argument("--no-vote", action="store_true",
                   help="use only the first sample; the honest baseline that "
                        "shows what self-consistency actually bought")
    a = p.parse_args()

    records = load_extracted(a.extracted)
    if not records:
        print(f"no valid records in {a.extracted}")
        return 2

    verdicts, agreements, stats = build_verdicts(records, vote=not a.no_vote)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(v.model_dump_json() for v in verdicts) + "\n",
                     encoding="utf-8")
    if a.agreement_out:
        a.agreement_out.write_text(json.dumps(agreements, indent=2),
                                   encoding="utf-8")

    print(f"\n  {a.extracted.name} -> {a.out.name}")
    for k, v in stats.items():
        print(f"    {k:<22} {v:.4f}" if isinstance(v, float)
              else f"    {k:<22} {v}")
    if stats["samples_per_document"] > 1:
        print(f"    voting                 per-field majority across "
              f"{stats['samples_per_document']} samples"
              if not a.no_vote else
              "    voting                 OFF (first sample only)")
    if stats["singletons"]:
        print(f"\n    {stats['singletons']} singleton chain(s): documents that "
              f"could not be linked.\n    On extracted records that is usually "
              f"a misread identifier, which is\n    the failure the linker's "
              f"repair rung exists for. Compare against\n    the ground-truth "
              f"run to separate extraction error from linking error.")
    print(f"\n  next: python bench/detect_score.py --verdicts {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
