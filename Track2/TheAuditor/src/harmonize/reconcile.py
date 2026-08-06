"""Reconcile a linked chain: walk it pairwise, find where money, quantity and
terms drift, emit Discrepancy/ChainVerdict. No GPU, no model -- this is the
deterministic reconciliation pass; extraction-side judgement calls (like
resolving partial-shipment against a null vs. a stated zero) are explicitly
out of scope here and stay with Mechanism A / the extraction review, per the
working brief's §4.4.

GROUNDED IN THE INJECTOR, NOT GUESSED
--------------------------------------
Every comparison this module makes mirrors exactly what
data/generator/inject.py perturbs, so the reconciler is answerable against
ground truth rather than against an assumed shape of the problem:

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

PRICE_DRIFT vs. UNAPPLIED_DISCOUNT is decided by which field moved: if the PO
carried an allowance_total the invoice has silently dropped, that is the
more specific, more actionable finding and takes precedence over reporting
the same aggregate delta as an undifferentiated price drift.

DECOYS ARE SUPPOSED TO SURVIVE
-------------------------------
`allowed_delta(po.total, tolerance, rendered=...)` is the SAME function the
verifier calls -- one implementation, so a decoy sized inside the
cross-document band by the generator is inside it here too, by construction,
not by a second hand-tuned threshold.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from typing import Optional

from schemas import (
    AnomalyType,
    CanonicalDoc,
    Chain,
    ChainVerdict,
    CROSS_DOC_TOLERANCE,
    Discrepancy,
    DocType,
    LineItem,
    Tolerance,
    allowed_delta,
)


def _rendered_totals(doc: CanonicalDoc) -> list[str]:
    """Amount strings as printed, for allowed_delta's precision inference.
    source_text is byte-preserved, but the reconciler works off CanonicalDoc
    fields only (never re-parses source_text), so it renders the total the
    same way the document's own currency formatting would, at cent
    precision -- the same convention allowed_delta's other callers use when
    no better evidence is available."""
    return [str(doc.total)] if doc.total is not None else []


def _line_by_id(doc: CanonicalDoc) -> dict[str, LineItem]:
    return {li.line_id: li for li in doc.line_items}


def _biggest_price_delta_line(po: CanonicalDoc,
                              inv: CanonicalDoc) -> Optional[tuple[str, Decimal]]:
    """The line_id and signed unit_price delta with the largest magnitude,
    for attributing an aggregate total drift to a specific line. None if no
    line's price actually moved (the drift is elsewhere, e.g. tax)."""
    po_lines = _line_by_id(po)
    best: Optional[tuple[str, Decimal]] = None
    for line_id, inv_li in _line_by_id(inv).items():
        po_li = po_lines.get(line_id)
        if po_li is None or po_li.unit_price is None or inv_li.unit_price is None:
            continue
        delta = inv_li.unit_price - po_li.unit_price
        if delta == 0:
            continue
        if best is None or abs(delta) > abs(best[1]):
            best = (line_id, delta)
    return best


def _price_and_discount(po: CanonicalDoc, inv: CanonicalDoc,
                        tolerance: Tolerance) -> tuple[list[Discrepancy], list[str]]:
    discrepancies: list[Discrepancy] = []
    trace: list[str] = []
    if po.total is None or inv.total is None:
        return discrepancies, trace

    band = allowed_delta(po.total, tolerance, rendered=_rendered_totals(po))
    delta = inv.total - po.total
    trace.append(f"{po.doc_id} total {po.total} vs {inv.doc_id} total "
                f"{inv.total}: delta {delta}, band \u00b1{band}")

    if abs(delta) <= band:
        return discrepancies, trace     # inside tolerance: must not flag

    if po.allowance_total is not None and inv.allowance_total is None:
        discrepancies.append(Discrepancy(
            anomaly_type=AnomalyType.UNAPPLIED_DISCOUNT,
            doc_ids_involved=sorted([po.doc_id, inv.doc_id]),
            field_path="allowance_total",
            delta=po.allowance_total,
            decided_on_delta=delta,
            decided_on_band=band,
            evidence=f"PO carries an allowance of {po.allowance_total}; "
                    f"absent on the invoice"))
        return discrepancies, trace

    attribution = _biggest_price_delta_line(po, inv)
    if attribution is None:
        # The aggregate total moved outside the band, but no line's
        # unit_price actually changed -- the cause is something this pair
        # doesn't explain (most commonly a quantity difference the GRN<->
        # invoice comparison already reports on its own terms). Reporting
        # PRICE_DRIFT here with no price evidence would double-count a
        # quantity anomaly under the wrong label, which corrupts per-type
        # precision. Log it and stop; do not guess a cause.
        trace.append(f"{po.doc_id}/{inv.doc_id}: total delta outside band "
                    f"but no line unit_price changed -- not attributed to "
                    f"price drift (see quantity comparison for this pair)")
        return discrepancies, trace

    field_path = f"line_items[{attribution[0]}].unit_price"
    line_delta = attribution[1]
    discrepancies.append(Discrepancy(
        anomaly_type=AnomalyType.PRICE_DRIFT,
        doc_ids_involved=sorted([po.doc_id, inv.doc_id]),
        field_path=field_path,
        delta=line_delta,
        decided_on_delta=delta,
        decided_on_band=band,
        evidence=f"invoice total {inv.total} vs PO total {po.total} exceeds "
                f"the cross-document band ({abs(delta)} > {band})"))
    return discrepancies, trace


def _quantities(po: Optional[CanonicalDoc], grn: CanonicalDoc,
                inv: CanonicalDoc) -> tuple[list[Discrepancy], list[str]]:
    """THREE-WAY, not pairwise, despite the pair in the return type.

    grn_qty < inv_qty on its own is NOT enough to tell QUANTITY_MISMATCH
    (the invoice over-bills) from PARTIAL_SHIPMENT (the shipment fell
    short): both produce the identical inv_qty > grn_qty shape, and a
    two-document comparison cannot distinguish "invoice added extra units"
    from "receipt is short of what was ordered" -- they're the same
    arithmetic fact seen from two different documents having moved.

    The PO's ordered quantity is the third reference point that resolves
    it: whichever of {grn, inv} still agrees with what was ordered is the
    document that DIDN'T move, so the anomaly belongs to the other one.
      - grn == ordered, inv > ordered  -> invoice over-billed (MISMATCH)
      - inv == ordered, grn < ordered  -> shipment fell short (PARTIAL)
    Falls back to the old two-document heuristic only when no PO line is
    available to arbitrate, and marks that case in the trace so a reader
    knows the label is a guess.
    """
    discrepancies: list[Discrepancy] = []
    trace: list[str] = []
    grn_lines = _line_by_id(grn)
    po_lines = _line_by_id(po) if po is not None else {}
    for line_id, inv_li in _line_by_id(inv).items():
        grn_li = grn_lines.get(line_id)
        if grn_li is None:
            continue
        qdelta = inv_li.quantity - grn_li.quantity
        if qdelta == 0:
            continue
        po_li = po_lines.get(line_id)
        ordered = po_li.quantity if po_li is not None else None

        anomaly: Optional[AnomalyType]
        if ordered is not None and grn_li.quantity == ordered and inv_li.quantity != ordered:
            anomaly = AnomalyType.QUANTITY_MISMATCH
        elif ordered is not None and inv_li.quantity == ordered and grn_li.quantity != ordered:
            anomaly = AnomalyType.PARTIAL_SHIPMENT
        elif ordered is not None:
            # Both sides disagree with the order -- genuinely ambiguous.
            # Isolate rather than guess, same principle the linker uses.
            trace.append(f"{grn.doc_id}/{inv.doc_id} line {line_id}: "
                        f"receipt {grn_li.quantity}, invoice "
                        f"{inv_li.quantity}, PO ordered {ordered} -- "
                        f"neither matches the order, not attributed")
            continue
        else:
            # No PO line to arbitrate with: fall back to sign, and say so.
            anomaly = (AnomalyType.QUANTITY_MISMATCH if qdelta > 0
                      else AnomalyType.PARTIAL_SHIPMENT)
            trace.append(f"{grn.doc_id}/{inv.doc_id} line {line_id}: "
                        f"no PO quantity to arbitrate; labelled by sign only")

        trace.append(f"{grn.doc_id} line {line_id} received "
                    f"{grn_li.quantity}, {inv.doc_id} billed "
                    f"{inv_li.quantity}, PO ordered {ordered} ({qdelta:+})")
        discrepancies.append(Discrepancy(
            anomaly_type=anomaly,
            doc_ids_involved=sorted([grn.doc_id, inv.doc_id]),
            field_path=f"line_items[{line_id}].quantity",
            delta=qdelta,
            evidence=f"received {grn_li.quantity}, billed {inv_li.quantity}, "
                    f"ordered {ordered}"))
    return discrepancies, trace


def _term_change(quote: CanonicalDoc, inv: CanonicalDoc
                ) -> tuple[list[Discrepancy], list[str]]:
    if not quote.payment_terms or not inv.payment_terms:
        return [], []
    if quote.payment_terms == inv.payment_terms:
        return [], []
    d = Discrepancy(
        anomaly_type=AnomalyType.TERM_CHANGE,
        doc_ids_involved=sorted([quote.doc_id, inv.doc_id]),
        field_path="payment_terms",
        delta=None,
        evidence=f"quote states {quote.payment_terms!r}, invoice states "
                f"{inv.payment_terms!r}")
    return [d], [f"{quote.doc_id} terms {quote.payment_terms!r} vs "
                f"{inv.doc_id} terms {inv.payment_terms!r}"]


def _near_duplicates(invoices: list[CanonicalDoc]
                     ) -> tuple[list[Discrepancy], list[str]]:
    discrepancies: list[Discrepancy] = []
    trace: list[str] = []
    for a, b in combinations(sorted(invoices, key=lambda d: d.doc_id), 2):
        if a.total is None or b.total is None:
            continue
        exposure = max(a.total, b.total)
        discrepancies.append(Discrepancy(
            anomaly_type=AnomalyType.NEAR_DUPLICATE,
            doc_ids_involved=sorted([a.doc_id, b.doc_id]),
            field_path="doc_number",
            delta=exposure,
            evidence=f"{a.doc_number!r} and {b.doc_number!r} both present "
                    f"in one chain; exposure {exposure}"))
        trace.append(f"duplicate billing: {a.doc_id} ({a.doc_number}) and "
                    f"{b.doc_id} ({b.doc_number})")
    return discrepancies, trace


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
