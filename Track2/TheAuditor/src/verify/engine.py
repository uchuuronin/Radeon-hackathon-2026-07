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

import re
from decimal import Decimal
from typing import Iterable, Optional

from normalise import compare_parties, read_date
from precision import unexplained_claims
from schemas import (
    DEFAULT_TOLERANCES,
    CanonicalDoc,
    DocType,
    CheckName,
    CheckOutcome,
    CheckResult,
    Tolerance,
    VerificationReport,
    allowed_delta,
)


def _not_applicable(check: CheckName, why: str) -> CheckResult:
    """The check cannot mean anything on this document type. Neutral, and
    neutral is correct: a payment has no line items to sum."""
    return CheckResult(check=check, outcome=CheckOutcome.SKIPPED, message=why)


def _inputs_missing(check: CheckName, why: str,
                    field_path: Optional[str] = None) -> CheckResult:
    """The check WOULD have applied and a field it needs was not supplied.

    Still neutral for pass/fail, because we cannot assert a document is wrong
    using evidence we do not have. But it is counted against coverage, so an
    extractor cannot buy a clean report by emitting less.
    """
    return CheckResult(check=check, outcome=CheckOutcome.INPUTS_MISSING,
                       field_path=field_path, message=why)


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
    if not priced:
        return _not_applicable(CheckName.BR_CO_10, "no priced lines")
    if doc.doc_type not in _CARRIES_TOTALS:
        return _not_applicable(CheckName.BR_CO_10,
                               "document type states no subtotal")
    if doc.subtotal is None:
        return _inputs_missing(
            CheckName.BR_CO_10,
            "priced lines present but subtotal not extracted", "subtotal")
    lines_sum = sum((li.line_total for li in priced), Decimal(0))
    return _compare(CheckName.BR_CO_10, expected=lines_sum,
                    actual=doc.subtotal, field_path="subtotal",
                    operands=[li.line_total for li in priced] + [doc.subtotal])


def check_br_co_13(doc: CanonicalDoc) -> CheckResult:
    """total_excl_tax = subtotal - allowances + charges (BT-109 identity).
    The identity the naive `subtotal + tax = total` version gets wrong on
    every discounted or freighted document."""
    if doc.doc_type not in _CARRIES_TOTALS or (
            doc.subtotal is None and doc.total_excl_tax is None):
        return _not_applicable(CheckName.BR_CO_13,
                               "document type states no net totals")
    if doc.subtotal is None or doc.total_excl_tax is None:
        absent = "subtotal" if doc.subtotal is None else "total_excl_tax"
        return _inputs_missing(CheckName.BR_CO_13,
                               f"{absent} not extracted", absent)
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
    if doc.doc_type not in _CARRIES_TOTALS or (
            doc.total is None and doc.total_excl_tax is None):
        return _not_applicable(CheckName.BR_CO_15,
                               "document type states no gross totals")
    if doc.total is None or doc.total_excl_tax is None:
        absent = "total" if doc.total is None else "total_excl_tax"
        return _inputs_missing(CheckName.BR_CO_15,
                               f"{absent} not extracted", absent)
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
        if any(li.unit_price is not None for li in doc.line_items):
            return [_inputs_missing(CheckName.LINE_NET_AMOUNT,
                                    "lines carry prices but no line totals")]
        return [_not_applicable(CheckName.LINE_NET_AMOUNT, "no priced lines")]
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
        return _inputs_missing(CheckName.AMOUNTS_APPEAR_IN_SOURCE,
                               "no source_text supplied", "source_text")

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
        return _not_applicable(CheckName.AMOUNTS_APPEAR_IN_SOURCE,
                               "document states no amounts")

    # Numbers we already know are on the page for NON-monetary reasons.
    non_monetary: list[Decimal] = [li.quantity for li in doc.line_items]
    non_monetary += [Decimal(doc.doc_date.year), Decimal(doc.doc_date.month),
                     Decimal(doc.doc_date.day)]
    # Identifiers print digits too. 'PO-4021' puts 4021 on the page, and
    # without this a hallucinated total of 4021.00 would find itself
    # conveniently "explained" by the document's own reference number. The
    # counting argument only works if EVERY known non-monetary source of
    # digits is subtracted, and identifiers are the one we had missed.
    non_monetary += _identifier_numbers(doc)

    # PASS 1 — which claims have verbatim evidence of their own?
    missing = unexplained_claims(claimed, non_monetary, doc.source_text)
    if missing:
        # PASS 2 — a claim with no verbatim evidence is still warranted if it
        # follows arithmetically from claims that DO have it. Grounding is
        # computed first precisely so derivation cannot bootstrap itself.
        ungrounded = {v for _, v in missing}
        grounded = {v for _, v in claimed if v not in ungrounded}
        derivable = _derivable_values(doc, grounded)
        missing = [(path, v) for path, v in missing if v not in derivable]
    if not missing:
        return CheckResult(check=CheckName.AMOUNTS_APPEAR_IN_SOURCE,
                           outcome=CheckOutcome.PASS)
    path, value = missing[0]
    return CheckResult(
        check=CheckName.AMOUNTS_APPEAR_IN_SOURCE, outcome=CheckOutcome.FAIL,
        field_path=path, expected=f"{value} present in source", actual="not found",
        message=f"{len(missing)} amount(s) without independent evidence in "
                f"source; first: {path}={value}")


#: Document types that state a net/tax/gross breakdown. A goods receipt
#: records what arrived and a remittance settles a figure; neither omits the
#: breakdown through extraction failure, so reporting a missing input there
#: would be counting a document's nature against the extractor.
_CARRIES_TOTALS = {DocType.QUOTE, DocType.SALES_ORDER,
                   DocType.PURCHASE_ORDER, DocType.INVOICE}

_DIGITS = re.compile(r"\d+")


def _identifier_numbers(doc: CanonicalDoc) -> list[Decimal]:
    """Digit runs printed as part of identifiers rather than as money."""
    out: list[Decimal] = []
    for ident in [doc.doc_number, *doc.references]:
        for run in _DIGITS.findall(ident):
            out.append(Decimal(run))
    return out


def _derivable_values(doc: CanonicalDoc,
                      grounded: set[Decimal]) -> set[Decimal]:
    """Values the document need not print because they follow from ones it did.

    The counting check can only ground a VERBATIM amount. Plenty of real
    layouts omit an intermediate figure that is nonetheless correct: a
    Stripe-style invoice prints line items and one grand total and no
    subtotal line at all. Failing the extractor for correctly computing that
    subtotal would punish the exact behaviour we want and would make the
    check fire hardest on the cleanest layouts.

    THE CONSTRAINT THAT MAKES THIS SAFE: an identity only confers evidence if
    EVERY operand it consumes is itself grounded in the source. Derivability
    must chain back to something actually printed, never float free. Without
    that rule the exemption would destroy the check's most valuable case, the
    CONSISTENT hallucination where every amount is shifted in lockstep: those
    documents satisfy every identity perfectly and are detectable only
    because nothing is grounded. With the rule, nothing is grounded, so
    nothing is derivable, and the hallucination still fails. Absent operands
    (no allowance stated) contribute zero and need no grounding, because
    absence is not a claim.
    """
    z = Decimal(0)
    ok = lambda v: v is None or v in grounded                   # noqa: E731
    d: set[Decimal] = set()

    lines = [li.line_total for li in doc.line_items if li.line_total is not None]
    if lines and all(v in grounded for v in lines):
        d.add(sum(lines, z))                                    # BR-CO-10
    allowance, charge = doc.allowance_total or z, doc.charge_total or z
    if ok(doc.allowance_total) and ok(doc.charge_total):
        if doc.subtotal in grounded:                            # BR-CO-13 fwd
            d.add(doc.subtotal - allowance + charge)
        if doc.total_excl_tax in grounded:                      # BR-CO-13 rev
            d.add(doc.total_excl_tax + allowance - charge)
    if ok(doc.tax) and ok(doc.rounding_amount):
        tax, rnd = doc.tax or z, doc.rounding_amount or z
        if doc.total_excl_tax in grounded:                      # BR-CO-15 fwd
            d.add(doc.total_excl_tax + tax + rnd)
        if doc.total in grounded:                               # BR-CO-15 rev
            d.add(doc.total - tax - rnd)
    if doc.total in grounded:
        d.add(doc.total)              # amount_due mirrors it on most invoices
    return d
    # DELIBERATELY NOT DERIVED: the line identity quantity x unit_price.
    # It looks like the same kind of warrant and is not. Quantity is a
    # non-monetary field, so that derivation needs only ONE grounded money
    # value to manufacture a whole money pyramid: a document printing
    # "Qty 500 EA @ 1.00" would confer evidence on a line total of 500.00, a
    # subtotal of 500.00 and a total of 500.00, none of which it prints. That
    # is exactly the confirmed false positive this check was built to catch.
    # The document-level identities above are safe because every operand they
    # consume is itself a grounded monetary claim. The cost of the omission
    # is one escalation on the rare layout that prints unit prices without
    # line amounts, which is the right side to be wrong on.


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


def check_br_co_16(doc: CanonicalDoc) -> CheckResult:
    """BR-CO-16: amount due = total incl. VAT - paid amount + rounding.

    The settlement identity. Distinct from BR-CO-15 in what it catches: BR-CO-15
    asks whether the invoice adds up, BR-CO-16 asks whether what is still owed
    follows from what was billed and what has already been paid. A part-payment
    recorded against the wrong invoice satisfies every other identity on this
    document and breaks only this one.

    Absent paid_amount and rounding_amount contribute zero, which matches the
    standard: they are optional business terms, and their absence is a claim of
    nothing rather than a gap in the evidence.
    """
    if doc.amount_due is None or doc.total is None:
        if doc.doc_type not in _CARRIES_TOTALS:
            return _not_applicable(CheckName.BR_CO_16,
                                   "document type states no amount due")
        absent = "amount_due" if doc.amount_due is None else "total"
        return _inputs_missing(CheckName.BR_CO_16,
                               f"{absent} not extracted", absent)

    z = Decimal(0)
    paid, rnd = doc.paid_amount or z, doc.rounding_amount or z
    expected = doc.total - paid + rnd
    operands = [v for v in (doc.total, doc.paid_amount, doc.rounding_amount,
                            doc.amount_due) if v is not None]
    return _compare(CheckName.BR_CO_16, expected=expected,
                    actual=doc.amount_due, field_path="amount_due",
                    operands=operands)


#: Monetary business terms and the BR-DEC rule that caps each at two
#: decimals. Named individually rather than looped generically so the audit
#: line can cite the actual rule number a validator would cite.
_BR_DEC_FIELDS: tuple[tuple[str, str], ...] = (
    ("subtotal", "BR-DEC-09 (BT-106)"),
    ("allowance_total", "BR-DEC-10 (BT-107)"),
    ("charge_total", "BR-DEC-11 (BT-108)"),
    ("total_excl_tax", "BR-DEC-12 (BT-109)"),
    ("tax", "BR-DEC-13 (BT-110)"),
    ("total", "BR-DEC-14 (BT-112)"),
    ("paid_amount", "BR-DEC-16 (BT-113)"),
    ("rounding_amount", "BR-DEC-17 (BT-114)"),
    ("amount_due", "BR-DEC-18 (BT-115)"),
)


def check_br_dec_max_2(doc: CanonicalDoc) -> CheckResult:
    """Every monetary amount carries at most two decimals.

    EN 16931 caps each monetary BT through its own BR-DEC rule, and real
    validators reject a third decimal outright. It is worth having for a
    reason that has nothing to do with conformance: THREE DECIMALS IN A MONEY
    FIELD IS AN EXTRACTION TELL. A model that computed rather than read a
    figure leaks the division  -  a 3-line split of 100.00 emitted as 33.333
    is arithmetically reasonable and textually impossible, because no invoice
    prints it. This catches a class of hallucination the identity checks
    cannot, since 33.333 x 3 balances perfectly.

    Free: the Decimal exponent already carries the printed precision, so
    there is no wire change and no source scan. Deliberately NOT applied to
    quantity or unit_price - EN 16931 constrains neither, and unit prices
    genuinely carry four decimals in real catalogues.
    """
    offenders: list[str] = []
    for name, rule in _BR_DEC_FIELDS:
        v = getattr(doc, name)
        if v is not None and -v.as_tuple().exponent > 2:
            offenders.append(f"{name}={v} violates {rule}")
    if not offenders:
        return CheckResult(check=CheckName.BR_DEC_MAX_2,
                           outcome=CheckOutcome.PASS)
    first = offenders[0].split("=")[0]
    return CheckResult(
        check=CheckName.BR_DEC_MAX_2, outcome=CheckOutcome.FAIL,
        field_path=first, expected="at most 2 decimals",
        actual=offenders[0].split(" ")[0].split("=")[1],
        message="; ".join(offenders))


def verify_doc(doc: CanonicalDoc) -> VerificationReport:
    """All doc-level checks, one report — the audit record for this doc."""
    checks: list[CheckResult] = [
        check_br_co_10(doc),
        check_br_co_13(doc),
        check_br_co_15(doc),
        *check_line_net_amounts(doc),
        check_br_co_16(doc),
        check_br_dec_max_2(doc),
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
