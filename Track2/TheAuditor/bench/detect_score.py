"""Score reconciler output against the planted-anomaly answer key.

    python bench/detect_score.py --demo      # runs the naive baseline below

WRITTEN BEFORE THE RECONCILER, DELIBERATELY
-------------------------------------------
The corpus is built to punish false positives, not to reward recall:

    price_drift          210    of which 130 are BELOW-TOLERANCE DECOYS
    quantity_mismatch     30    near_duplicate       30
    unapplied_discount    30    term_change          30   (non-numeric)
    partial_shipment      30
    chains with no anomaly       150

230 true positives to find; 130 decoys and 150 clean chains to leave alone.

And the decoys are not weak versions of the real thing. A decoy has a genuinely
different unit price on a line, sized so the CHAIN TOTAL stays inside the
cross-document band:

    REAL   line delta  42.00   totals differ by 2016.00   band 100.00  OUTSIDE
    DECOY  line delta   0.13   totals differ by    0.42   band 100.00  INSIDE

So a field-by-field comparator scores perfect recall on price drift and eats
130 false positives, and the arithmetic works out to a headline number that
looks respectable. Detection belongs on the AGGREGATE, attribution belongs at
the LINE, and that constraint falls out of the answer key rather than out of
anyone's opinion. Building the reconciler first and discovering it afterwards
means rebuilding it.

Hence the order: the oracle, then a deliberately naive baseline to prove the
oracle discriminates, then the reconciler written against a target that is
already measurable. The baseline is the same move as the replay server on the
extraction side: a claim about a scorer is untested until something wrong has
been fed to it.

WHAT COUNTS AS A MATCH
----------------------
A reported Discrepancy matches a PlantedAnomaly when the document PAIR agrees
and the FIELD PATH agrees. Type is reported separately rather than required,
because misclassifying a real drift is a different and lesser failure than
missing it, and collapsing the two would hide which is happening. Delta is
compared where both are numeric, but a wrong magnitude on a correctly located
drift is again reported, not counted as a miss.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemas import (AnswerKey, CanonicalDoc, ChainVerdict,  # noqa: E402
                     CROSS_DOC_TOLERANCE, Discrepancy, ExtractedRecord,
                     PlantedAnomaly, allowed_delta)

#: Ratio of aggregate delta to the tolerance band, bucketed. The corpus plants
#: hard negatives that straddle the boundary, so an overall rate hides the only
#: region that is actually difficult: detection at 1.20x is arithmetic,
#: detection at 1.05x is the product.
TIERS: tuple[tuple[str, float, float], ...] = (
    ("<=0.50x  (deep decoy)", 0.0, 0.50),
    ("0.50-0.80x", 0.50, 0.80),
    ("0.80-0.95x", 0.80, 0.95),
    ("0.95-1.05x (the boundary)", 0.95, 1.05),
    ("1.05-1.20x", 1.05, 1.20),
    (">1.20x   (easy)", 1.20, float("inf")),
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate with a Wilson score interval.

    Never a bare percentage. Two configurations whose intervals overlap have
    not been separated by the measurement, however different their point
    estimates look, and 3 correct out of 3 is not 100%.
    """
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class DetectionScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    #: Located correctly but typed wrongly. NOT a miss: the drift was found and
    #: an analyst is looking at the right two documents and the right field.
    mistyped: int = 0
    #: Located correctly, magnitude wrong beyond a cent.
    wrong_delta: int = 0
    #: The decoys specifically. Flagging one is a false positive, and this is
    #: the number the corpus was built to produce.
    decoys_flagged: int = 0
    decoys_total: int = 0
    #: Chains with no planted anomaly that were nonetheless flagged.
    clean_chains_flagged: int = 0
    clean_chains_total: int = 0
    by_type: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    by_tier: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    @property
    def precision(self) -> tuple[float, float, float]:
        return wilson(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> tuple[float, float, float]:
        return wilson(self.tp, self.tp + self.fn)

    def render(self, label: str = "") -> str:
        p, plo, phi = self.precision
        r, rlo, rhi = self.recall
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        L = [f"## {label or 'anomaly detection'}", ""]
        L.append(f"  precision  {p:6.1%}  [{plo:.1%}, {phi:.1%}]   "
                 f"tp={self.tp} fp={self.fp}")
        L.append(f"  recall     {r:6.1%}  [{rlo:.1%}, {rhi:.1%}]   "
                 f"tp={self.tp} fn={self.fn}")
        L.append(f"  F1         {f1:6.1%}")
        L.append("")
        L.append(f"  decoys flagged        {self.decoys_flagged}/{self.decoys_total}"
                 f"   <- every one is a false positive by construction")
        L.append(f"  clean chains flagged  {self.clean_chains_flagged}/"
                 f"{self.clean_chains_total}")
        if self.mistyped:
            L.append(f"  located but mistyped  {self.mistyped}"
                     f"   <- found, classified wrongly. Not a miss.")
        if self.wrong_delta:
            L.append(f"  located, delta wrong  {self.wrong_delta}")

        L += ["", "  by anomaly type (recall):"]
        for t in sorted(self.by_type):
            k, n = self.by_type[t]
            rate, lo, hi = wilson(k, n)
            L.append(f"    {t:<22} {k:>3}/{n:<4} {rate:6.1%} [{lo:.0%},{hi:.0%}]")
        L.append("  ^ never averaged. A system that catches every price drift and")
        L.append("    no duplicate is not 83% correct, it has a hole.")

        L += ["", "  by aggregate delta / band (recall):"]
        for name, _lo, _hi in TIERS:
            if name not in self.by_tier:
                continue
            k, n = self.by_tier[name]
            rate, lo, hi = wilson(k, n)
            L.append(f"    {name:<28} {k:>3}/{n:<4} {rate:6.1%} [{lo:.0%},{hi:.0%}]")
        L.append("  ^ the 0.95-1.05x row is the product. The >1.20x row is arithmetic.")
        return "\n".join(L)


def _pair(ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(ids))


def _tier_of(ratio: Optional[float]) -> Optional[str]:
    if ratio is None:
        return None
    for name, lo, hi in TIERS:
        if lo <= ratio < hi:
            return name
    return TIERS[-1][0]


def chain_totals_ratio(anom: PlantedAnomaly,
                       docs: dict[str, CanonicalDoc]) -> Optional[float]:
    """Aggregate delta over the band, for the two documents involved.

    This is the quantity the FLAG decision should be made on, so it is also the
    right axis to stratify difficulty along.
    """
    pair = [docs.get(i) for i in anom.doc_ids_involved]
    if any(d is None or d.total is None for d in pair):
        return None
    a, b = pair[0].total, pair[1].total
    band = allowed_delta(b, CROSS_DOC_TOLERANCE, [str(a), str(b)])
    if band == 0:
        return None
    return float(abs(a - b) / band)


def score(verdicts: Sequence[ChainVerdict], key: AnswerKey,
          docs: dict[str, CanonicalDoc]) -> DetectionScore:
    s = DetectionScore()
    reported: dict[str, list[Discrepancy]] = {v.chain_id: list(v.discrepancies)
                                              for v in verdicts}
    # Chains are keyed by the answer key's own ids; the linker derives its own
    # stable ids, so match on membership rather than on the string.
    by_members: dict[tuple[str, ...], list[Discrepancy]] = {}
    for v in verdicts:
        by_members[_pair(v.doc_ids) if False else tuple(sorted(v.doc_ids))] = \
            list(v.discrepancies)

    for chain in key.chains:
        members = tuple(sorted(chain.doc_ids))
        found = by_members.get(members)
        if found is None:                      # fall back to id match
            found = reported.get(chain.chain_id, [])
        unmatched = list(found)

        if not chain.anomalies:
            s.clean_chains_total += 1
            if found:
                s.clean_chains_flagged += 1

        for anom in chain.anomalies:
            ratio = chain_totals_ratio(anom, docs)
            tier = _tier_of(ratio)
            must_flag = not anom.is_within_tolerance
            if anom.is_within_tolerance:
                s.decoys_total += 1

            hit = None
            for d in unmatched:
                if _pair(d.doc_ids_involved) == _pair(anom.doc_ids_involved) \
                        and d.field_path == anom.field_path:
                    hit = d
                    break

            if hit is not None:
                unmatched.remove(hit)

            if must_flag:
                s.by_type[str(anom.anomaly_type)][1] += 1
                if tier:
                    s.by_tier[tier][1] += 1
                if hit is not None:
                    s.tp += 1
                    s.by_type[str(anom.anomaly_type)][0] += 1
                    if tier:
                        s.by_tier[tier][0] += 1
                    if str(hit.anomaly_type) != str(anom.anomaly_type):
                        s.mistyped += 1
                    if (hit.delta is not None and anom.expected_delta is not None
                            and abs(hit.delta - anom.expected_delta) > Decimal("0.01")):
                        s.wrong_delta += 1
                else:
                    s.fn += 1
            else:
                if hit is not None:
                    s.fp += 1
                    s.decoys_flagged += 1

        s.fp += len(unmatched)                 # reported, nothing planted there
    return s


# ---------------------------------------------------------------------------
# The deliberately naive baseline
# ---------------------------------------------------------------------------

def naive_field_comparator(chain_docs: Sequence[CanonicalDoc]) -> ChainVerdict:
    """Compare every line field between every document pair, flag any
    difference. THE WRONG ANSWER, implemented so the oracle can be shown to
    catch it.

    It is not a straw man. It is what a reasonable person writes first, and
    what the answer key is specifically built to expose: the decoys have
    genuinely different line values, so this scores well on recall and is
    unusable in production.
    """
    from schemas import AnomalyType
    out: list[Discrepancy] = []
    trace: list[str] = []
    for i, a in enumerate(chain_docs):
        for b in chain_docs[i + 1:]:
            la = {li.line_id: li for li in (a.line_items or [])}
            lb = {li.line_id: li for li in (b.line_items or [])}
            for lid in sorted(set(la) & set(lb)):
                for fld, kind in (("unit_price", AnomalyType.PRICE_DRIFT),
                                  ("quantity", AnomalyType.QUANTITY_MISMATCH)):
                    va, vb = getattr(la[lid], fld), getattr(lb[lid], fld)
                    if va is None or vb is None or va == vb:
                        continue
                    out.append(Discrepancy(
                        anomaly_type=kind,
                        doc_ids_involved=sorted([a.doc_id, b.doc_id]),
                        field_path=f"line_items[{lid}].{fld}",
                        delta=vb - va,
                        evidence=f"{fld} {va} vs {vb}"))
            trace.append(f"compared {a.doc_id} <-> {b.doc_id}")
    return ChainVerdict(chain_id="CH-" + min(d.doc_id for d in chain_docs),
                        doc_ids=sorted(d.doc_id for d in chain_docs),
                        discrepancies=out, trace=trace)


def load_corpus(records: Path, key_path: Path, layout: str = "layout_a"):
    docs: dict[str, CanonicalDoc] = {}
    for line in records.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw["meta"].get("layout") != layout:
            continue
        d = ExtractedRecord.model_validate(raw).doc
        docs[d.doc_id] = d
    key = AnswerKey.model_validate(json.loads(key_path.read_text(encoding="utf-8")))
    return docs, key


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--records", type=Path,
                   default=ROOT / "data/generated/records.jsonl")
    p.add_argument("--key", type=Path,
                   default=ROOT / "data/generated/answer_key.json")
    p.add_argument("--verdicts", type=Path, default=None,
                   help="JSONL of ChainVerdict. Omit with --demo.")
    p.add_argument("--demo", action="store_true",
                   help="score the naive field comparator instead")
    p.add_argument("--layout", default="layout_a")
    a = p.parse_args()

    docs, key = load_corpus(a.records, a.key, a.layout)

    if a.demo:
        verdicts = [naive_field_comparator([docs[i] for i in c.doc_ids
                                            if i in docs])
                    for c in key.chains if any(i in docs for i in c.doc_ids)]
        label = "NAIVE field-by-field comparator (the wrong answer)"
    else:
        if not a.verdicts:
            p.error("pass --verdicts or --demo")
        verdicts = [ChainVerdict.model_validate_json(l)
                    for l in a.verdicts.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
        label = a.verdicts.stem

    print(score(verdicts, key, docs).render(label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
