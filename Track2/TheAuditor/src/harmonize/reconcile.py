"""Walk a deal chain and report what drifted. No GPU, no model.

THE INVOICE IS THE PIVOT
------------------------
Derived from the corpus, not assumed: every planted anomaly involves the
invoice.

    price_drift          purchase_order <-> invoice      210
    unapplied_discount   purchase_order <-> invoice       30
    partial_shipment     goods_receipt  <-> invoice       30
    quantity_mismatch    goods_receipt  <-> invoice       30
    term_change          quote          <-> invoice       30
    near_duplicate       invoice        <-> invoice       30

That is the shape of quote-to-cash: the invoice is the document that gets paid,
so it is the one everything else is checked against. So the walk is four
comparisons per invoice rather than all fifteen pairs of a six-document chain,
which is both cheaper and the reason each comparison can be given rules that
actually fit it.

PAIRWISE, NEVER THE WHOLE CHAIN AT ONCE. Long-context evaluations consistently
show accuracy collapsing for information in the middle of a window, worse on
the 7-14B class we can run locally, and multi-document cross-referencing is
exactly where that bites. The unit of comparison is a pair.

THREE DETECTION REGIMES, BECAUSE ONE DOES NOT FIT
-------------------------------------------------
1. MONEY-GATED (purchase order <-> invoice). Decide on the AGGREGATE, attribute
   at the LINE. This is the single most important rule in the file. 130 of the
   210 planted price drifts are decoys whose line unit price genuinely differs,
   sized so the chain total stays inside the cross-document band:

       REAL   line delta  42.00   totals differ by 2016.00   band 100.00
       DECOY  line delta   0.13   totals differ by    0.42   band 100.00

   A field-by-field comparator flags both. Measured, that is 100% recall on
   price drift bought with all 130 decoys and 16.1% precision. The flag
   decision has to be made on the total; the line is only how we say WHERE.

2. QUANTITY-GATED (goods receipt <-> invoice). A goods receipt states no total
   at all, so the money gate is not merely inappropriate here, it is
   undefined. Quantities are the only comparable aggregate.

3. UNGATED FIELD COMPARISON (payment terms, duplicate invoices, and a
   purchase-order allowance absent from the invoice). These have no aggregate
   signature to gate on: a changed payment term moves no number, and a
   duplicate invoice has exactly the same total as the one it duplicates. An
   aggregate-only detector is blind to 60 of the 230 true positives.

A KNOWN LIMIT, STATED RATHER THAN HIDDEN
----------------------------------------
`partial_shipment` and `quantity_mismatch` are the same observable event: the
invoice bills more units than the receipt records. Their distributions overlap
completely (ratio 1.09-2.14 against 1.03-3.00, both containing 3 -> 4), so no
rule reads one from the documents. We detect the event once and emit one type,
which mistypes the other half. The oracle counts that as located-but-mislabelled
rather than a miss, because an analyst sent to the right two documents and the
right line has been served even if the label is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Optional, Sequence

from normalise import compare_parties
from schemas import (AnomalyType, CanonicalDoc, Chain, ChainVerdict,
                     CROSS_DOC_TOLERANCE, Discrepancy, DocType, LineItem,
                     Tolerance, allowed_delta)


@dataclass(frozen=True)
class ReconcilePolicy:
    """Every judgement call, named and in one place.

    A threshold buried in a function is a number nobody can defend on camera,
    and these are the ones an analyst will ask about.
    """
    #: Band for the money gate. The default is the frozen cross-document
    #: policy, which the below-tolerance decoys are sized against.
    money: Tolerance = field(default_factory=lambda: CROSS_DOC_TOLERANCE)
    #: Units of difference between received and invoiced quantity that counts.
    #: Quantities are counts, so unlike money there is no printed-precision
    #: story: a receipt saying 10 means ten. A small floor survives only to
    #: absorb fractional units on weighed or measured goods.
    quantity_floor: Decimal = Decimal("0.001")
    #: Flag payment-term differences at all. Off is defensible for a buyer who
    #: renegotiates routinely and does not want the noise.
    check_terms: bool = True


DEFAULT_POLICY = ReconcilePolicy()


def _lines(doc: Optional[CanonicalDoc]) -> dict[str, LineItem]:
    return {li.line_id: li for li in (doc.line_items or [])} if doc else {}


def _by_type(docs: Sequence[CanonicalDoc], t: DocType) -> list[CanonicalDoc]:
    return [d for d in docs if d.doc_type == t]


def _money_gate(a: CanonicalDoc, b: CanonicalDoc,
                policy: ReconcilePolicy) -> Optional[tuple[Decimal, Decimal]]:
    """(aggregate delta, band) for two money-bearing documents, or None.

    None means the gate is not applicable, which is NOT the same as "no
    discrepancy" and must never be reported as a clean result. A goods receipt
    reaches here with no total and would otherwise silently pass.
    """
    if a.total is None or b.total is None:
        return None
    band = allowed_delta(b.total, policy.money, [str(a.total), str(b.total)])
    return abs(b.total - a.total), band


def _worst_line(a: CanonicalDoc, b: CanonicalDoc, field: str
                ) -> Optional[tuple[str, Decimal, Decimal]]:
    """The shared line whose `field` differs most, with both values.

    Attribution, not detection. Once the aggregate says something is wrong,
    this says where to look. Largest difference rather than first difference,
    because an analyst opening one line wants the one that explains the money.
    """
    best = None
    for lid in sorted(set(_lines(a)) & set(_lines(b))):
        va = getattr(_lines(a)[lid], field)
        vb = getattr(_lines(b)[lid], field)
        if va is None or vb is None or va == vb:
            continue
        if best is None or abs(vb - va) > abs(best[2] - best[1]):
            best = (lid, va, vb)
    return best


def _po_vs_invoice(po: CanonicalDoc, inv: CanonicalDoc,
                   policy: ReconcilePolicy,
                   receipt_covers_quantity: bool = False
                   ) -> tuple[list[Discrepancy], str]:
    """Money-gated. Decide on the aggregate, attribute at the line.

    ONE PHYSICAL EVENT, REPORTED ONCE, AGAINST THE MOST AUTHORITATIVE PAIR.
    Billing more units than were ordered pushes the order-to-invoice total out
    of band with every unit price untouched, so this comparison sees a total it
    cannot explain. But the same event is already visible, and better evidenced,
    against the goods receipt: the order says what was agreed, the receipt says
    what actually arrived, and "arrived" is the stronger claim about quantity.
    Reporting it here as well produced 26 false positives, all of them
    `price_drift` at `total`, all of them the same events already correctly
    reported at the receipt.

    So when a goods receipt is present to carry the quantity finding, this
    comparison stays on price and says in the trace what it deferred. When
    there is no receipt, nothing else can carry it and it is reported here.
    """
    out: list[Discrepancy] = []
    gate = _money_gate(po, inv, policy)
    if gate is None:
        return out, (f"{po.doc_id} <-> {inv.doc_id}: money gate not applicable "
                     f"(a total is missing); NOT treated as clean")
    delta, band = gate
    if delta <= band:
        return out, (f"{po.doc_id} <-> {inv.doc_id}: totals differ by {delta} "
                     f"within band +/-{band}; no flag")

    pair = sorted([po.doc_id, inv.doc_id])

    # An allowance agreed on the order and absent from the invoice is a
    # DOCUMENT-level fact, so it is attributed to the document field rather
    # than to a line. Checked first: when a discount goes missing the line
    # prices are usually untouched, and attributing that to a line would send
    # an analyst to a line that is perfectly correct.
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
        return out, (f"{po.doc_id} <-> {inv.doc_id}: allowance {po_allow} vs "
                     f"{inv_allow}, totals out of band by {delta - band}")

    worst = _worst_line(po, inv, "unit_price")
    if worst is None and receipt_covers_quantity \
            and _worst_line(po, inv, "quantity") is not None:
        qty = _worst_line(po, inv, "quantity")
        return out, (f"{po.doc_id} <-> {inv.doc_id}: totals out of band by "
                     f"{delta - band}, but no unit price moved and {qty[0]} "
                     f"quantity did ({qty[1]} -> {qty[2]}); deferred to the "
                     f"goods receipt, which is authoritative on quantity")
    if worst is not None:
        lid, va, vb = worst
        out.append(Discrepancy(
            anomaly_type=AnomalyType.PRICE_DRIFT,
            doc_ids_involved=pair,
            field_path=f"line_items[{lid}].unit_price",
            delta=vb - va,
            decided_on_delta=delta, decided_on_band=band,
            evidence=f"unit price {va} on the order, {vb} on the invoice; "
                     f"chain totals differ by {delta} against a band of {band}"))
        return out, (f"{po.doc_id} <-> {inv.doc_id}: out of band by "
                     f"{delta - band}, attributed to {lid}.unit_price")

    # Out of band and no line explains it. Reported at document level rather
    # than dropped: an unexplained total difference is a finding.
    out.append(Discrepancy(
        anomaly_type=AnomalyType.PRICE_DRIFT, doc_ids_involved=pair,
        field_path="total", delta=inv.total - po.total,
        decided_on_delta=delta, decided_on_band=band,
        evidence="totals disagree beyond tolerance and no single line "
                 "accounts for it"))
    return out, (f"{po.doc_id} <-> {inv.doc_id}: out of band by {delta - band}, "
                 f"unattributed")


def _grn_vs_invoice(grn: CanonicalDoc, inv: CanonicalDoc,
                    policy: ReconcilePolicy) -> tuple[list[Discrepancy], str]:
    """Quantity-gated. A goods receipt states no total, so money is undefined
    here and quantities are the only comparable aggregate."""
    out: list[Discrepancy] = []
    gl, il = _lines(grn), _lines(inv)
    pair = sorted([grn.doc_id, inv.doc_id])
    worst = None
    for lid in sorted(set(gl) & set(il)):
        qg, qi = gl[lid].quantity, il[lid].quantity
        if qg is None or qi is None:
            continue
        d = qi - qg
        if abs(d) <= policy.quantity_floor:
            continue
        if worst is None or abs(d) > abs(worst[1]):
            worst = (lid, d, qg, qi)

    if worst is None:
        return out, f"{grn.doc_id} <-> {inv.doc_id}: quantities agree"

    lid, d, qg, qi = worst
    # See the module docstring: partial_shipment and quantity_mismatch are the
    # same observable event and their distributions overlap completely. One
    # type is emitted for both.
    out.append(Discrepancy(
        anomaly_type=AnomalyType.QUANTITY_MISMATCH,
        doc_ids_involved=pair, field_path=f"line_items[{lid}].quantity",
        delta=d, decided_on_delta=abs(d), decided_on_band=policy.quantity_floor,
        evidence=f"receipt records {qg}, invoice bills {qi} on {lid}"))
    return out, (f"{grn.doc_id} <-> {inv.doc_id}: billed {qi} against {qg} "
                 f"received on {lid}")


def _quote_vs_invoice(quote: CanonicalDoc, inv: CanonicalDoc,
                      policy: ReconcilePolicy) -> tuple[list[Discrepancy], str]:
    """Ungated. A changed payment term moves no number, so there is no
    aggregate to gate on and no delta to report."""
    if not policy.check_terms:
        return [], f"{quote.doc_id} <-> {inv.doc_id}: terms check disabled"
    a = (quote.payment_terms or "").strip()
    b = (inv.payment_terms or "").strip()
    if not a or not b or a.casefold() == b.casefold():
        return [], f"{quote.doc_id} <-> {inv.doc_id}: terms agree or absent"
    return ([Discrepancy(
        anomaly_type=AnomalyType.TERM_CHANGE,
        doc_ids_involved=sorted([quote.doc_id, inv.doc_id]),
        field_path="payment_terms", delta=None,
        evidence=f"quote states {a!r}, invoice states {b!r}")],
        f"{quote.doc_id} <-> {inv.doc_id}: terms {a!r} -> {b!r}")


def _duplicate_invoices(invoices: Sequence[CanonicalDoc]
                        ) -> tuple[list[Discrepancy], list[str]]:
    """Ungated, and the one anomaly with no delta in the usual sense.

    Two invoices citing the same upstream documents for the same amount is a
    duplicate however different their numbers are, and the amount at risk is
    the WHOLE total rather than a drift: paying both costs the full second
    invoice. So `delta` carries the exposure.

    The linker groups a near-duplicate into its chain deliberately and says
    nothing about it, which is what makes this check possible: had grouping
    tried to be clever the duplicate would have been split off and silently
    disappeared.
    """
    out: list[Discrepancy] = []
    trace: list[str] = []
    for i, a in enumerate(invoices):
        for b in invoices[i + 1:]:
            if a.total is None or a.total != b.total:
                continue
            if set(a.references or []) != set(b.references or []):
                continue
            out.append(Discrepancy(
                anomaly_type=AnomalyType.NEAR_DUPLICATE,
                doc_ids_involved=sorted([a.doc_id, b.doc_id]),
                field_path="doc_number", delta=a.total,
                decided_on_delta=Decimal("0"),
                evidence=f"{a.doc_number} and {b.doc_number} both bill "
                         f"{a.total} against the same upstream documents; "
                         f"the exposure is the full amount, not a difference"))
            trace.append(f"{a.doc_id} <-> {b.doc_id}: duplicate billing "
                         f"{a.total}")
    return out, trace


def reconcile(chain: Chain, docs: dict[str, CanonicalDoc],
              policy: ReconcilePolicy = DEFAULT_POLICY) -> ChainVerdict:
    """Produce one verdict for one chain. Deterministic, order-independent."""
    members = [docs[i] for i in sorted(chain.doc_ids) if i in docs]
    invoices = _by_type(members, DocType.INVOICE)
    found: list[Discrepancy] = []
    trace: list[str] = []

    if not invoices:
        # A chain with nothing to bill against is not clean, it is incomplete.
        # Saying so is the difference between "we checked and it was fine" and
        # "there was nothing to check".
        trace.append("no invoice in this chain; nothing to reconcile against")
        return ChainVerdict(chain_id=chain.chain_id,
                            doc_ids=[d.doc_id for d in members], trace=trace)

    dups, dup_trace = _duplicate_invoices(invoices)
    found += dups
    trace += dup_trace

    has_receipt = bool(_by_type(members, DocType.GOODS_RECEIPT))
    for inv in invoices:
        for po in _by_type(members, DocType.PURCHASE_ORDER):
            d, t = _po_vs_invoice(po, inv, policy, has_receipt)
            found += d
            trace.append(t)
        for grn in _by_type(members, DocType.GOODS_RECEIPT):
            d, t = _grn_vs_invoice(grn, inv, policy)
            found += d
            trace.append(t)
        for q in _by_type(members, DocType.QUOTE):
            d, t = _quote_vs_invoice(q, inv, policy)
            found += d
            trace.append(t)

    # Deduplicate: a chain holding a near-duplicate walks the same purchase
    # order twice and would otherwise report the same drift once per invoice.
    seen: set[tuple] = set()
    unique: list[Discrepancy] = []
    for d in found:
        k = (str(d.anomaly_type), tuple(d.doc_ids_involved), d.field_path)
        if k in seen:
            continue
        seen.add(k)
        unique.append(d)

    return ChainVerdict(
        chain_id=chain.chain_id,
        doc_ids=[d.doc_id for d in members],
        discrepancies=sorted(unique,
                             key=lambda d: (d.doc_ids_involved, d.field_path)),
        trace=trace)


def reconcile_all(chains: Iterable[Chain], docs: dict[str, CanonicalDoc],
                  policy: ReconcilePolicy = DEFAULT_POLICY) -> list[ChainVerdict]:
    return [reconcile(c, docs, policy) for c in chains]
