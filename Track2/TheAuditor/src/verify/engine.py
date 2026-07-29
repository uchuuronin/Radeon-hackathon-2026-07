"""A5 — Mechanism A. The deterministic verifier.

Pure Python, zero GPU, zero model. This module is simultaneously three things
the plan asks for: the accuracy safeguard (a mis-read is caught before it
poisons reconciliation), the audit trail (every check logged with expected /
actual / delta / band), and the free component of the S2 confidence signal
(verify_pass and strict_pass need no calibration — they are ground-truth
correct by construction).

SCOPE
-----
verify_doc(doc)            — the five DOC-LEVEL checks. What the CLI runs.
verify_pair(earlier,later) — the two PAIR-LEVEL checks (party names match,
                             dates in order). The reconciler calls this when
                             walking a chain; the verifier does not invent
                             chain context it does not have.

Cross-document AMOUNT comparison is deliberately NOT here — that is the
reconciler's job, with CROSS_DOC_TOLERANCE. Keeping that boundary is what
keeps the escalation rate honest: within-document arithmetic is near-exact,
commercial variance lives between documents.

TOLERANCE
---------
Bands come from allowed_delta() with the operands' own stated precision.
Extraction preserves the printed precision in the Decimal exponent
("2,477.00" -> Decimal("2477.00"), "11100" -> Decimal("11100")), so the
rendered forms passed to allowed_delta are simply str() of the operands —
no source_text lookup needed, and a whole-unit document automatically gets
a wider band than a cent-precise one.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Optional

from normalise import compare_parties, read_date
from precision import unexplained_claims
from schemas import (
    DEFAULT_TOLERANCES,
    CanonicalDoc,
    CheckName,
    CheckOutcome,
    CheckResult,
    Tolerance,
    VerificationReport,
    allowed_delta,
)


def _skip(check: CheckName, why: str) -> CheckResult:
    return CheckResult(check=check, outcome=CheckOutcome.SKIPPED, message=why)


def _compare(
    check: CheckName,
    expected: Decimal,
    actual: Decimal,
    operands: Iterable[Decimal],
    field_path: Optional[str] = None,
    tol: Optional[Tolerance] = None,
) -> CheckResult:
    """Three-outcome comparison: exact PASS, WITHIN_TOLERANCE inside the
    precision-inferred band, FAIL beyond it."""
    tol = tol or DEFAULT_TOLERANCES[check]
    delta = actual - expected
    band = allowed_delta(expected, tol, [str(v) for v in operands])
    if delta == 0:
        outcome = CheckOutcome.PASS
    elif abs(delta) <= band:
        outcome = CheckOutcome.WITHIN_TOLERANCE
    else:
        outcome = CheckOutcome.FAIL
    return CheckResult(
        check=check, outcome=outcome, field_path=field_path,
        expected=str(expected), actual=str(actual),
        delta=delta, tolerance_applied=band,
        message="" if outcome == CheckOutcome.PASS else
                f"delta {delta} vs band ±{band}")


# ---------------------------------------------------------------------------
# Doc-level checks
# ---------------------------------------------------------------------------

def check_br_co_10(doc: CanonicalDoc) -> CheckResult:
    """Sum of line net amounts equals subtotal (BT-131 -> BT-106)."""
    priced = [li for li in doc.line_items if li.line_total is not None]
    if doc.subtotal is None or not priced:
        return _skip(CheckName.BR_CO_10, "no subtotal or no priced lines")
    lines_sum = sum((li.line_total for li in priced), Decimal(0))
    return _compare(CheckName.BR_CO_10, expected=lines_sum,
                    actual=doc.subtotal, field_path="subtotal",
                    operands=[li.line_total for li in priced] + [doc.subtotal])


def check_br_co_13(doc: CanonicalDoc) -> CheckResult:
    """total_excl_tax = subtotal - allowances + charges (BT-109 identity).
    The identity the naive `subtotal + tax = total` version gets wrong on
    every discounted or freighted document."""
    if doc.subtotal is None or doc.total_excl_tax is None:
        return _skip(CheckName.BR_CO_13, "subtotal or total_excl_tax absent")
    allowance = doc.allowance_total or Decimal(0)
    charge = doc.charge_total or Decimal(0)
    expected = doc.subtotal - allowance + charge
    operands = [v for v in (doc.subtotal, doc.allowance_total,
                            doc.charge_total, doc.total_excl_tax)
                if v is not None]
    return _compare(CheckName.BR_CO_13, expected=expected,
                    actual=doc.total_excl_tax, field_path="total_excl_tax",
                    operands=operands)


def check_br_co_15(doc: CanonicalDoc) -> CheckResult:
    """total = total_excl_tax + tax (+ rounding_amount) — the BT-112 identity.
    A document with no tax line is checked as total == total_excl_tax and the
    message says so: absence of tax is information, not an error."""
    if doc.total is None or doc.total_excl_tax is None:
        return _skip(CheckName.BR_CO_15, "total or total_excl_tax absent")
    tax = doc.tax if doc.tax is not None else Decimal(0)
    rounding = doc.rounding_amount or Decimal(0)
    expected = doc.total_excl_tax + tax + rounding
    operands = [v for v in (doc.total_excl_tax, doc.tax,
                            doc.rounding_amount, doc.total) if v is not None]
    r = _compare(CheckName.BR_CO_15, expected=expected, actual=doc.total,
                 field_path="total", operands=operands)
    if doc.tax is None:
        r = r.model_copy(update={
            "message": (r.message + " (no tax line stated)").strip()})
    return r


def check_line_net_amounts(doc: CanonicalDoc) -> list[CheckResult]:
    """quantity x unit_price - line_allowance = line_total, per priced line.
    One CheckResult per line so field_path names the exact line that broke —
    'this field failed' beats 'this document failed'."""
    priced = [li for li in doc.line_items
              if li.unit_price is not None and li.line_total is not None]
    if not priced:
        return [_skip(CheckName.LINE_NET_AMOUNT, "no priced lines")]
    out = []
    for li in priced:
        expected = li.quantity * li.unit_price - (li.line_allowance or Decimal(0))
        # Compare at the precision the line total itself states.
        expected_q = expected.quantize(li.line_total)
        out.append(_compare(
            CheckName.LINE_NET_AMOUNT, expected=expected_q,
            actual=li.line_total,
            field_path=f"line_items[{li.line_id}].line_total",
            operands=[li.quantity, li.unit_price, li.line_total]))
    return out


def check_amounts_appear_in_source(doc: CanonicalDoc) -> CheckResult:
    """Every stated monetary field has INDEPENDENT evidence in source_text.

    This is the anti-hallucination check, and the one the arithmetic cannot
    replace: an extraction that invented a number and kept itself CONSISTENT
    passes every identity. Roughly two-thirds of financial extraction errors
    are invented numeric values, and only the source exposes the consistent
    ones.

    Membership alone is too weak — a hallucinated total of 500.00 "appears"
    in a document printing `Qty 500`. So we count: a monetary value must
    occur MORE often than the non-monetary fields (line quantities, date
    components) already account for. See precision.unexplained_claims.

    Comparison is by VALUE, never by raw string: one layout prints 2,477.00
    and another prints 2477 for the same figure, both extractions are
    correct, and a substring search would punish the second.
    """
    if not doc.source_text:
        return _skip(CheckName.AMOUNTS_APPEAR_IN_SOURCE, "no source_text")

    claimed: list[tuple[str, Decimal]] = []
    for name in ("subtotal", "allowance_total", "charge_total",
                 "total_excl_tax", "tax", "total", "rounding_amount",
                 "amount_due"):
        v = getattr(doc, name)
        if v is not None:
            claimed.append((name, v))
    for li in doc.line_items:
        if li.unit_price is not None:
            claimed.append((f"line_items[{li.line_id}].unit_price", li.unit_price))
        if li.line_total is not None:
            claimed.append((f"line_items[{li.line_id}].line_total", li.line_total))
    if not claimed:
        return _skip(CheckName.AMOUNTS_APPEAR_IN_SOURCE, "no amounts stated")

    # Numbers we already know are on the page for NON-monetary reasons.
    non_monetary: list[Decimal] = [li.quantity for li in doc.line_items]
    non_monetary += [Decimal(doc.doc_date.year), Decimal(doc.doc_date.month),
                     Decimal(doc.doc_date.day)]

    missing = unexplained_claims(claimed, non_monetary, doc.source_text)
    if not missing:
        return CheckResult(check=CheckName.AMOUNTS_APPEAR_IN_SOURCE,
                           outcome=CheckOutcome.PASS)
    path, value = missing[0]
    return CheckResult(
        check=CheckName.AMOUNTS_APPEAR_IN_SOURCE, outcome=CheckOutcome.FAIL,
        field_path=path, expected=f"{value} present in source", actual="not found",
        message=f"{len(missing)} amount(s) without independent evidence in "
                f"source; first: {path}={value}")


#: Per-party date order, exactly like tolerance: a vendor's format is stable
#: even though the format space is not. Absent an entry, genuine ambiguity is
#: REPORTED rather than guessed — dateparser's own default of MDY for English
#: would silently misread a UK or EU document.
PARTY_DATE_ORDER: dict[str, str] = {}


def check_date_parses(doc: CanonicalDoc,
                      date_order: Optional[str] = None) -> CheckResult:
    """Re-parse the printed date in code and compare with the model's
    normalised date. Mechanism A applied properly: deterministic code checks
    the model, never the model checking itself.

    Ambiguity is not failure. 02/03/2026 is genuinely undecidable without
    knowing the vendor, so a claim matching EITHER reading is
    WITHIN_TOLERANCE — the same principle that stops rounding manufacturing
    exceptions.
    """
    if doc.doc_date_raw is None:
        return CheckResult(check=CheckName.DATES_PARSE_AND_ORDER,
                           outcome=CheckOutcome.PASS,
                           message="normalised date only")

    import config as _config
    order = (date_order or PARTY_DATE_ORDER.get(doc.party_name)
             or _config.load().date_order_for(doc.party_name))
    reading = read_date(doc.doc_date_raw, order)

    if not reading.candidates:
        return CheckResult(
            check=CheckName.DATES_PARSE_AND_ORDER, outcome=CheckOutcome.SKIPPED,
            field_path="doc_date_raw", actual=doc.doc_date_raw,
            message="raw date format not recognised — cannot verify")

    if doc.doc_date not in reading.candidates:
        return CheckResult(
            check=CheckName.DATES_PARSE_AND_ORDER, outcome=CheckOutcome.FAIL,
            field_path="doc_date", expected=" or ".join(
                c.isoformat() for c in reading.candidates),
            actual=doc.doc_date.isoformat(),
            message="normalised date does not match the printed date")

    if reading.ambiguous:
        return CheckResult(
            check=CheckName.DATES_PARSE_AND_ORDER,
            outcome=CheckOutcome.WITHIN_TOLERANCE, field_path="doc_date",
            expected=" or ".join(c.isoformat() for c in reading.candidates),
            actual=doc.doc_date.isoformat(),
            message=f"{doc.doc_date_raw!r} is ambiguous; claim matches one "
                    f"reading. Set PARTY_DATE_ORDER for this vendor to resolve.")

    return CheckResult(check=CheckName.DATES_PARSE_AND_ORDER,
                       outcome=CheckOutcome.PASS)


def verify_doc(doc: CanonicalDoc) -> VerificationReport:
    """All doc-level checks, one report — the audit record for this doc."""
    checks: list[CheckResult] = [
        check_br_co_10(doc),
        check_br_co_13(doc),
        check_br_co_15(doc),
        *check_line_net_amounts(doc),
        check_amounts_appear_in_source(doc),
        check_date_parses(doc),
    ]
    return VerificationReport(doc_id=doc.doc_id, checks=checks)


# ---------------------------------------------------------------------------
# Pair-level checks — called by the reconciler while walking a chain
# ---------------------------------------------------------------------------

def verify_pair(earlier: CanonicalDoc, later: CanonicalDoc) -> list[CheckResult]:
    out: list[CheckResult] = []
    m = compare_parties(earlier.party_name, later.party_name)
    if m.exact:
        out.append(CheckResult(check=CheckName.PARTY_NAMES_MATCH,
                               outcome=CheckOutcome.PASS))
    elif m.same_base:
        # Same base name, different legal form. Standard guidance strips
        # suffixes for CRM matching but carves out an exception for financial
        # and compliance contexts, which is exactly where we sit: "Acme Inc"
        # and "Acme LLC" may be different legal entities, and lookalike
        # vendor names are a live invoice-fraud vector. So this SURFACES with
        # both raw names intact and is never silently passed.
        out.append(CheckResult(
            check=CheckName.PARTY_NAMES_MATCH,
            outcome=CheckOutcome.WITHIN_TOLERANCE, field_path="party_name",
            expected=earlier.party_name, actual=later.party_name,
            message="same base name, different legal form — review: these may "
                    "be distinct legal entities"))
    else:
        out.append(CheckResult(
            check=CheckName.PARTY_NAMES_MATCH, outcome=CheckOutcome.FAIL,
            field_path="party_name",
            expected=earlier.party_name, actual=later.party_name,
            message=f"{earlier.doc_id} vs {later.doc_id}"))
    if earlier.doc_date <= later.doc_date:
        out.append(CheckResult(check=CheckName.DATES_PARSE_AND_ORDER,
                               outcome=CheckOutcome.PASS))
    else:
        out.append(CheckResult(
            check=CheckName.DATES_PARSE_AND_ORDER, outcome=CheckOutcome.FAIL,
            field_path="doc_date",
            expected=f"{earlier.doc_id} ({earlier.doc_date}) on or before "
                     f"{later.doc_id}",
            actual=str(later.doc_date),
            message="documents out of chronological order"))
    return out
