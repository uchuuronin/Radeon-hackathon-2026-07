"""Walk a linked chain, report what drifted. No GPU, no model.

  FROM THE THREE-WAY BRANCH — `_quantities` arbitration, and it is correct.
    The other branch concluded partial_shipment and quantity_mismatch are
    observationally identical, having compared only the receipt against the
    invoice. They are not: the PURCHASE ORDER is the third reference point.
    Whichever of {receipt, invoice} still agrees with what was ordered is the
    document that did not move, so the anomaly belongs to the other one.
    Measured on the corpus: 60/60 correct, 0 ambiguous. What looked like an
    unavoidable 29-case mislabelling was a missing document, not a missing rule.

  FROM THE MONEY-GATE BRANCH — the aggregate gate, the policy object, strict
    duplicate detection, and not-applicable semantics.

THE INVOICE IS THE PIVOT
------------------------
Derived from the corpus: every planted anomaly involves the invoice.

  anomaly_type        compares                  on field(s)
  ------------------  ------------------------  --------------------------
  PRICE_DRIFT         PURCHASE_ORDER <-> INVOICE total (aggregate, cross-doc
                       tolerance), attributed to the line with the largest
                       unit_price delta
  UNAPPLIED_DISCOUNT  PURCHASE_ORDER <-> INVOICE allowance_total (PO carries
                       one, invoice does not)
  QUANTITY_MISMATCH   GOODS_RECEIPT <-> INVOICE line quantity, invoice > receipt
  PARTIAL_SHIPMENT    GOODS_RECEIPT <-> INVOICE line quantity, receipt < invoice
  NEAR_DUPLICATE      two INVOICEs in the same chain, on doc_number
  TERM_CHANGE         QUOTE <-> INVOICE payment_terms (non-numeric; the
                       generator plants this against the quote specifically,
                       not the sales order -- terms genuinely renegotiated
                       between quote and sales order are a separate, real
                       question the working brief flags as still open and
                       deliberately NOT resolved by this module)

That is the shape of quote-to-cash: the invoice is what gets paid, so it is
what everything else is checked against. Four comparisons per invoice, not all
fifteen pairs of a six-document chain.

PAIRWISE, NEVER THE WHOLE CHAIN AT ONCE. Long-context evaluations consistently
show accuracy collapsing for information in the middle of a window, worse on
the 7-14B class we can run locally, and multi-document cross-referencing is
exactly where that bites.

THREE DETECTION REGIMES, BECAUSE ONE DOES NOT FIT
-------------------------------------------------
1. MONEY-GATED (purchase order <-> invoice). Decide on the AGGREGATE, attribute
   at the LINE. The single most important rule here. 130 of the 210 planted
   price drifts are decoys whose line unit price genuinely differs, sized so
   the chain total stays inside the cross-document band:

       REAL   line delta  42.00   totals differ by 2016.00   band 100.00
       DECOY  line delta   0.13   totals differ by    0.42   band 100.00

   A field-by-field comparator flags both: measured, 100% recall on price drift
   bought with all 130 decoys and 16.1% precision. The total decides; the line
   only says WHERE.

2. QUANTITY-ARBITRATED (purchase order + goods receipt + invoice). A goods
   receipt states no total at all, so the money gate is not merely
   inappropriate here, it is undefined.

3. UNGATED FIELD COMPARISON (payment terms, duplicate invoices, absent
   allowance). No aggregate signature exists to gate on: a changed term moves
   no number, and a duplicate invoice has exactly the total it duplicates. An
   aggregate-only detector is blind to 60 of the 230 true positives.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from typing import Optional

from schemas import (AnomalyType, CanonicalDoc, Chain, ChainVerdict,
                     CROSS_DOC_TOLERANCE, Discrepancy, DocType, LineItem,
                     Tolerance, allowed_delta)


@dataclass(frozen=True)
class ReconcilePolicy:
    """Every judgement call, named and in one place. A threshold buried in a
    function is a number nobody can defend on camera."""
    #: Band for the money gate: the frozen cross-document policy the decoys are
    #: sized against. The SAME `allowed_delta` the verifier calls, so a decoy
    #: inside the band there is inside it here by construction rather than by a
    #: second hand-tuned threshold.
    money: Tolerance = field(default_factory=lambda: CROSS_DOC_TOLERANCE)
    #: Quantity difference that counts. Quantities are counts, so unlike money
    #: there is no printed-precision story: a receipt saying 10 means ten. The
    #: floor survives only to absorb fractional units on weighed goods.
    quantity_floor: Decimal = Decimal("0.001")
    #: Require a duplicate to bill the SAME total against the SAME upstream
    #: documents. Loosening this to "two invoices in one chain" costs nothing on
    #: this corpus, where all 30 multi-invoice chains are planted duplicates,
    #: and is unsafe on real data: partial billing puts two legitimate invoices
    #: against one order.
    strict_duplicates: bool = True
    #: Flag payment-term differences at all. Off is defensible for a buyer who
    #: renegotiates routinely and does not want the noise.
    check_terms: bool = True


def _line_by_id(doc: CanonicalDoc) -> dict[str, LineItem]:
    return {li.line_id: li for li in doc.line_items}


def _lines(doc: Optional[CanonicalDoc]) -> dict[str, LineItem]:
    return {li.line_id: li for li in (doc.line_items or [])} if doc else {}


def _by_type(docs: Sequence[CanonicalDoc], t: DocType) -> list[CanonicalDoc]:
    return [d for d in docs if DocType(d.doc_type) == t]


def _money_gate(a: CanonicalDoc, b: CanonicalDoc,
                policy: ReconcilePolicy) -> Optional[tuple[Decimal, Decimal]]:
    """(aggregate delta, band), or None when the gate does not apply.

    None is NOT "no discrepancy" and must never be reported as clean. A goods
    receipt reaches here with no total and would otherwise silently pass, which
    looks exactly like a good result.

    Both totals are handed to `allowed_delta` as rendered operands, so the band
    accumulates the printed precision of BOTH sides rather than one.
    """
    if a.total is None or b.total is None:
        return None
    band = allowed_delta(b.total, policy.money, [str(a.total), str(b.total)])
    return abs(b.total - a.total), band


def _worst_line(a: CanonicalDoc, b: CanonicalDoc,
                attr: str) -> Optional[tuple[str, Decimal, Decimal]]:
    """The shared line whose `attr` differs most, with both values.

    Attribution, not detection. Largest difference rather than first, because
    an analyst opening one line wants the one that explains the money.
    """
    best = None
    la, lb = _lines(a), _lines(b)
    for lid in sorted(set(la) & set(lb)):
        va, vb = getattr(la[lid], attr), getattr(lb[lid], attr)
        if va is None or vb is None or va == vb:
            continue
        delta = inv_li.unit_price - po_li.unit_price
        if delta == 0:
            continue
        if best is None or abs(delta) > abs(best[1]):
            best = (line_id, delta)
    return best


def _po_vs_invoice(po: CanonicalDoc, inv: CanonicalDoc,
                   policy: ReconcilePolicy, receipt_present: bool
                   ) -> tuple[list[Discrepancy], list[str]]:
    """Money-gated. Decide on the aggregate, attribute at the line."""
    out: list[Discrepancy] = []
    gate = _money_gate(po, inv, policy)
    if gate is None:
        return out, [f"{po.doc_id} <-> {inv.doc_id}: money gate not applicable "
                     f"(a total is missing); NOT treated as clean"]
    delta, band = gate
    if delta <= band:
        return out, [f"{po.doc_id} <-> {inv.doc_id}: totals differ by {delta}, "
                     f"within band +/-{band}; no flag"]

    pair = sorted([po.doc_id, inv.doc_id])

    # An allowance agreed on the order and missing from the invoice is a
    # DOCUMENT-level fact, so it is attributed to the document field. Checked
    # before line attribution because when a discount goes missing the line
    # prices are usually untouched, and pointing at a line would send an
    # analyst to a line that is perfectly correct.
    po_allow = po.allowance_total or Decimal("0")
    inv_allow = inv.allowance_total or Decimal("0")
    if po_allow != inv_allow:
        out.append(Discrepancy(
            anomaly_type=AnomalyType.UNAPPLIED_DISCOUNT,
            doc_ids_involved=pair, field_path="allowance_total",
            delta=po_allow - inv_allow,
            decided_on_delta=delta, decided_on_band=band,
            evidence=f"order allows {po_allow}, invoice allows {inv_allow}; "
                     f"chain totals differ by {delta} against a band of {band}"))
        return out, [f"{po.doc_id} <-> {inv.doc_id}: allowance {po_allow} vs "
                     f"{inv_allow}, totals out of band by {delta - band}"]

    worst = _worst_line(po, inv, "unit_price")
    if worst is not None:
        lid, va, vb = worst
        out.append(Discrepancy(
            anomaly_type=AnomalyType.PRICE_DRIFT, doc_ids_involved=pair,
            field_path=f"line_items[{lid}].unit_price", delta=vb - va,
            decided_on_delta=delta, decided_on_band=band,
            evidence=f"unit price {va} on the order, {vb} on the invoice; "
                     f"chain totals differ by {delta} against a band of {band}"))
        return out, [f"{po.doc_id} <-> {inv.doc_id}: out of band by "
                     f"{delta - band}, attributed to {lid}.unit_price"]

    # ONE PHYSICAL EVENT, REPORTED ONCE, AGAINST THE MOST AUTHORITATIVE PAIR.
    # Billing more units than ordered pushes this total out of band with every
    # unit price untouched. The same event is already visible, and better
    # evidenced, against the receipt: the order says what was agreed, the
    # receipt says what arrived, and "arrived" is the stronger claim about
    # quantity. Reporting it here as well produced 26 false positives, every
    # one a `price_drift` at `total` duplicating a correct quantity finding.
    qty = _worst_line(po, inv, "quantity")
    if qty is not None and receipt_present:
        return out, [f"{po.doc_id} <-> {inv.doc_id}: out of band by "
                     f"{delta - band}, no unit price moved but {qty[0]} "
                     f"quantity did ({qty[1]} -> {qty[2]}); deferred to the "
                     f"goods receipt, authoritative on quantity"]

    # Out of band, nothing explains it, and no receipt can carry it. Reported
    # at document level rather than dropped: an unexplained total difference is
    # a finding, and dropping it loses the only notice anyone gets.
    out.append(Discrepancy(
        anomaly_type=AnomalyType.PRICE_DRIFT, doc_ids_involved=pair,
        field_path="total", delta=inv.total - po.total,
        decided_on_delta=delta, decided_on_band=band,
        evidence="totals disagree beyond tolerance and no single line accounts "
                 "for it"))
    return out, [f"{po.doc_id} <-> {inv.doc_id}: out of band by {delta - band}, "
                 f"unattributed"]


def _quantities(po: Optional[CanonicalDoc], grn: CanonicalDoc,
                inv: CanonicalDoc, policy: ReconcilePolicy
                ) -> tuple[list[Discrepancy], list[str]]:
    """THREE-WAY, not pairwise, and this is why.

    `received < billed` on its own cannot tell QUANTITY_MISMATCH (the invoice
    over-billed) from PARTIAL_SHIPMENT (the shipment fell short). Both produce
    the identical shape, and their distributions overlap completely: ratios
    1.09-2.14 against 1.03-3.00, both containing 3 -> 4. From two documents it
    genuinely is a coin flip.

    The order's quantity is the third reference point that resolves it.
    Whichever of {receipt, invoice} still agrees with what was ordered is the
    document that did NOT move, so the anomaly belongs to the other:

        receipt == ordered, invoice != ordered  -> the invoice over-billed
        invoice == ordered, receipt != ordered  -> the shipment fell short

    Measured on the corpus: 60/60 correct, 0 ambiguous.
    """
    out: list[Discrepancy] = []
    trace: list[str] = []
    gl, il, pl = _lines(grn), _lines(inv), _lines(po)
    pair = sorted([grn.doc_id, inv.doc_id])

    for lid in sorted(set(gl) & set(il)):
        qg, qi = gl[lid].quantity, il[lid].quantity
        if qg is None or qi is None:
            continue
        qdelta = inv_li.quantity - grn_li.quantity
        if qdelta == 0:
            continue
        ordered = pl[lid].quantity if lid in pl else None

        if ordered is not None and qg == ordered and qi != ordered:
            kind = AnomalyType.QUANTITY_MISMATCH
        elif ordered is not None and qi == ordered and qg != ordered:
            kind = AnomalyType.PARTIAL_SHIPMENT
        elif ordered is not None:
            # Both sides disagree with the order: genuinely ambiguous, so
            # isolate rather than guess. Same principle the linker applies to
            # an unresolvable reference. A wrong label sends an analyst looking
            # for the wrong kind of problem.
            trace.append(f"{grn.doc_id} <-> {inv.doc_id} {lid}: received {qg}, "
                         f"billed {qi}, ordered {ordered}; neither matches the "
                         f"order, not attributed")
            continue
        else:
            kind = (AnomalyType.QUANTITY_MISMATCH if d > 0
                    else AnomalyType.PARTIAL_SHIPMENT)
            trace.append(f"{grn.doc_id} <-> {inv.doc_id} {lid}: no order "
                         f"quantity to arbitrate; labelled by sign only")

        # Sign carries meaning once the type is known, so it is oriented to
        # the type rather than left as a raw subtraction. An over-bill is what
        # was added (billed - received, positive); a short shipment is what is
        # missing (received - billed, negative). Reporting both as one
        # direction would make the number unreadable without also reading the
        # label, and it is the number an analyst chases.
        signed = d if kind is AnomalyType.QUANTITY_MISMATCH else -d
        out.append(Discrepancy(
            anomaly_type=kind, doc_ids_involved=pair,
            field_path=f"line_items[{lid}].quantity", delta=signed,
            decided_on_delta=abs(d), decided_on_band=policy.quantity_floor,
            evidence=f"received {qg}, billed {qi}, ordered {ordered}"))
        trace.append(f"{grn.doc_id} <-> {inv.doc_id} {lid}: received {qg}, "
                     f"billed {qi}, ordered {ordered} ({d:+})")

    if not out and not trace:
        trace.append(f"{grn.doc_id} <-> {inv.doc_id}: quantities agree")
    return out, trace


def _quote_vs_invoice(quote: CanonicalDoc, inv: CanonicalDoc,
                      policy: ReconcilePolicy
                      ) -> tuple[list[Discrepancy], list[str]]:
    """Ungated. A changed payment term moves no number, so there is no
    aggregate to gate on and no delta to report.

    Compared against the QUOTE specifically, which is where the generator
    plants it. Terms renegotiated between quote and sales order are a separate
    and genuinely open question that this module deliberately does not resolve.
    """
    if not policy.check_terms:
        return [], [f"{quote.doc_id} <-> {inv.doc_id}: terms check disabled"]
    a = (quote.payment_terms or "").strip()
    b = (inv.payment_terms or "").strip()
    if not a or not b or a.casefold() == b.casefold():
        return [], [f"{quote.doc_id} <-> {inv.doc_id}: terms agree or absent"]
    return ([Discrepancy(
        anomaly_type=AnomalyType.TERM_CHANGE,
        doc_ids_involved=sorted([quote.doc_id, inv.doc_id]),
        field_path="payment_terms", delta=None,
        evidence=f"quote states {a!r}, invoice states {b!r}")],
        [f"{quote.doc_id} <-> {inv.doc_id}: terms {a!r} -> {b!r}"])


def _duplicate_invoices(invoices: Sequence[CanonicalDoc],
                        policy: ReconcilePolicy
                        ) -> tuple[list[Discrepancy], list[str]]:
    """Ungated, and the one anomaly whose delta is not a difference.

    Two invoices billing the same amount against the same upstream documents
    are a duplicate however different their numbers. Paying both costs the
    WHOLE second invoice, so `delta` carries the exposure rather than a drift.

    The linker groups a near-duplicate into its chain deliberately and says
    nothing about it. That is what makes this check possible: had grouping
    tried to be clever, the duplicate would have been split off and silently
    disappeared.
    """
    out: list[Discrepancy] = []
    trace: list[str] = []
    ordered = sorted(invoices, key=lambda d: d.doc_id)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a.total is None or b.total is None:
                continue
            if policy.strict_duplicates:
                if a.total != b.total:
                    trace.append(f"{a.doc_id} / {b.doc_id}: two invoices with "
                                 f"different totals; not a duplicate")
                    continue
                if set(a.references or []) != set(b.references or []):
                    trace.append(f"{a.doc_id} / {b.doc_id}: same total, "
                                 f"different upstream documents; not a duplicate")
                    continue
            out.append(Discrepancy(
                anomaly_type=AnomalyType.NEAR_DUPLICATE,
                doc_ids_involved=sorted([a.doc_id, b.doc_id]),
                field_path="doc_number", delta=max(a.total, b.total),
                decided_on_delta=Decimal("0"),
                evidence=f"{a.doc_number} and {b.doc_number} both bill "
                         f"{a.total} against the same upstream documents; the "
                         f"exposure is the full amount, not a difference"))
            trace.append(f"{a.doc_id} <-> {b.doc_id}: duplicate billing "
                         f"{max(a.total, b.total)}")
    return out, trace


def reconcile(chain: Chain, docs: dict[str, CanonicalDoc],
              policy: ReconcilePolicy = DEFAULT_POLICY) -> ChainVerdict:
    """One verdict for one chain. Deterministic and order-independent."""
    members = [docs[i] for i in sorted(chain.doc_ids) if i in docs]
    invoices = _by_type(members, DocType.INVOICE)
    pos = _by_type(members, DocType.PURCHASE_ORDER)
    grns = _by_type(members, DocType.GOODS_RECEIPT)
    quotes = _by_type(members, DocType.QUOTE)

    found: list[Discrepancy] = []
    trace: list[str] = [f"chain {chain.chain_id}: {len(members)} documents, "
                        f"{len(invoices)} invoice(s), {len(grns)} receipt(s)"]

    if not invoices:
        # Not clean, incomplete. The difference between "we checked and it was
        # fine" and "there was nothing to check".
        trace.append("no invoice in this chain; nothing to reconcile against")
        return ChainVerdict(chain_id=chain.chain_id,
                            doc_ids=[d.doc_id for d in members], trace=trace)

    d, t = _duplicate_invoices(invoices, policy)
    found += d
    trace += t

    for inv in invoices:
        for q in quotes:
            d, t = _quote_vs_invoice(q, inv, policy)
            found += d
            trace += t
        for po in pos:
            d, t = _po_vs_invoice(po, inv, policy, bool(grns))
            found += d
            trace += t
        for grn in grns:
            d, t = _quantities(pos[0] if pos else None, grn, inv, policy)
            found += d
            trace += t

    # A chain holding a near-duplicate walks the same order twice, once per
    # invoice, and would otherwise report one drift as two.
    seen: set[tuple] = set()
    unique: list[Discrepancy] = []
    for x in found:
        k = (str(x.anomaly_type), tuple(x.doc_ids_involved), x.field_path)
        if k in seen:
            continue
        seen.add(k)
        unique.append(x)

    return ChainVerdict(
        chain_id=chain.chain_id,
        doc_ids=[d.doc_id for d in members],
        discrepancies=sorted(unique,
                             key=lambda x: (x.doc_ids_involved, x.field_path)),
        trace=trace)


def reconcile(chain: Chain, docs_by_id: dict[str, CanonicalDoc],
             tolerance: Tolerance = CROSS_DOC_TOLERANCE) -> ChainVerdict:
    """Walk one linked chain pairwise and report every drift found.

    Never all-pairs, never a concatenated context: exactly the document
    pairs the injector actually perturbs, each compared on the fields named
    in the module docstring's table.
    """
    members = [docs_by_id[i] for i in chain.doc_ids if i in docs_by_id]
    by_type: dict[DocType, list[CanonicalDoc]] = defaultdict(list)
    for d in members:
        by_type[DocType(d.doc_type)].append(d)

    quote = by_type[DocType.QUOTE][0] if by_type[DocType.QUOTE] else None
    po = by_type[DocType.PURCHASE_ORDER][0] if by_type[DocType.PURCHASE_ORDER] else None
    grns = by_type[DocType.GOODS_RECEIPT]
    invoices = by_type[DocType.INVOICE]

    discrepancies: list[Discrepancy] = []
    trace: list[str] = [f"chain {chain.chain_id}: {len(members)} documents, "
                       f"{len(invoices)} invoice(s), {len(grns)} receipt(s)"]

    dd, tt = _near_duplicates(invoices)
    discrepancies += dd
    trace += tt

    for inv in invoices:
        if quote is not None:
            dd, tt = _term_change(quote, inv)
            discrepancies += dd
            trace += tt
        if po is not None:
            dd, tt = _price_and_discount(po, inv, tolerance)
            discrepancies += dd
            trace += tt
        for grn in grns:
            dd, tt = _quantities(po, grn, inv)
            discrepancies += dd
            trace += tt

    return ChainVerdict(chain_id=chain.chain_id,
                        doc_ids=chain.doc_ids,
                        discrepancies=discrepancies,
                        trace=trace)
