"""A3 — anomaly injection. Chain-level drift on top of clean chains.

PaySim's model, applied to document chains: generate a clean baseline, then
inject known bad behaviour so ground truth exists by construction. One
deliberate divergence: PaySim mirrors a realistic fraud rate (~0.13%); an
evaluation corpus needs measurable per-type recall, so anomaly types are
STRATIFIED across chains rather than sampled. Realism belongs to the base
data; coverage belongs to the labels.

THE PROPERTY THAT MATTERS
-------------------------
Every injected document remains INTERNALLY CONSISTENT — it passes the whole
verifier. The drift lives BETWEEN documents: PO says one thing, invoice says
another, and each is a perfectly plausible document on its own. That is
precisely why the reconciler must exist: rung 0 cannot see these by design,
and tests/test_inject.py asserts it.

The exception is nothing: even the near-duplicate is a coherent invoice.
Doc-level corruption (extraction errors) is A6's job, already done.

DECOYS
------
Some price drifts are planted BELOW the cross-document tolerance band and
recorded with is_within_tolerance=True. The system is scored on NOT flagging
them — precision measured honestly, not "flag everything and claim recall".
"""

from __future__ import annotations

import random
from decimal import Decimal as D

from schemas import (
    AnomalyType,
    CanonicalDoc,
    CROSS_DOC_TOLERANCE,
    PlantedAnomaly,
    allowed_delta,
)

CENT = D("0.01")
q = lambda x: x.quantize(CENT)                                  # noqa: E731

# docs list order from gen.generate_chain
QUOTE, SO, PO, GRN, INVOICE, PAYMENT = range(6)


def _recompute(doc: CanonicalDoc) -> CanonicalDoc:
    """Rebuild the money pyramid from the line items, preserving the doc's
    own allowance/charge/tax-rate structure. Keeps injected documents
    internally consistent — the whole point."""
    subtotal = q(sum((li.line_total for li in doc.line_items
                      if li.line_total is not None), D(0)))
    # Preserve the effective tax rate rather than the absolute tax.
    rate = None
    if doc.tax is not None and doc.total_excl_tax not in (None, D(0)):
        rate = doc.tax / doc.total_excl_tax
    allowance = doc.allowance_total
    charge = doc.charge_total
    net = q(subtotal - (allowance or D(0)) + (charge or D(0)))
    tax = q(net * rate) if rate is not None else doc.tax
    total = q(net + (tax or D(0)))
    updates = dict(subtotal=subtotal, total_excl_tax=net, tax=tax, total=total)
    if doc.amount_due is not None:
        updates["amount_due"] = total
    return doc.model_copy(update=updates)


def _reprice_line(doc: CanonicalDoc, line_id: str, *,
                  unit_price: D | None = None,
                  quantity: D | None = None) -> CanonicalDoc:
    lines = []
    for li in doc.line_items:
        if li.line_id == line_id:
            price = unit_price if unit_price is not None else li.unit_price
            qty = quantity if quantity is not None else li.quantity
            li = li.model_copy(update={
                "unit_price": price, "quantity": qty,
                "line_total": q(qty * price) if price is not None else None})
        lines.append(li)
    return _recompute(doc.model_copy(update={"line_items": lines}))


def _priced_line(doc: CanonicalDoc, rng: random.Random):
    return rng.choice([li for li in doc.line_items
                       if li.unit_price is not None])


# --- injectors: (docs, rng) -> (docs, PlantedAnomaly) -----------------------

def inject_price_drift(docs, rng, decoy=False):
    inv = docs[INVOICE]
    li = _priced_line(inv, rng)
    if decoy:
        # Stay strictly inside the cross-document band on the TOTAL.
        band = allowed_delta(inv.total, CROSS_DOC_TOLERANCE)
        per_unit = q(min(band / 2, D("0.40")) / li.quantity) or D("0.01")
    else:
        pct = D(rng.randint(3, 9)) / 100
        per_unit = q(li.unit_price * pct) or D("0.05")
    new_price = q(li.unit_price + per_unit)
    docs = list(docs)
    docs[INVOICE] = _reprice_line(inv, li.line_id, unit_price=new_price)
    # Payment settles the invoice AS BILLED — the drift is PO vs invoice.
    docs[PAYMENT] = docs[PAYMENT].model_copy(
        update={"total": docs[INVOICE].total})
    return docs, PlantedAnomaly(
        anomaly_type=AnomalyType.PRICE_DRIFT,
        doc_ids_involved=[docs[PO].doc_id, docs[INVOICE].doc_id],
        field_path=f"line_items[{li.line_id}].unit_price",
        expected_delta=per_unit,
        is_within_tolerance=decoy,
        note=("decoy: total delta inside the cross-doc band; must NOT flag"
              if decoy else
              f"invoice bills {new_price} vs agreed {li.unit_price}"))


def inject_quantity_mismatch(docs, rng):
    inv = docs[INVOICE]
    li = _priced_line(inv, rng)
    extra = D(rng.randint(1, 3))
    docs = list(docs)
    docs[INVOICE] = _reprice_line(inv, li.line_id, quantity=li.quantity + extra)
    docs[PAYMENT] = docs[PAYMENT].model_copy(
        update={"total": docs[INVOICE].total})
    return docs, PlantedAnomaly(
        anomaly_type=AnomalyType.QUANTITY_MISMATCH,
        doc_ids_involved=[docs[GRN].doc_id, docs[INVOICE].doc_id],
        field_path=f"line_items[{li.line_id}].quantity",
        expected_delta=extra,
        note=f"invoice bills {li.quantity + extra}, receipt shows {li.quantity}")


def inject_partial_shipment(docs, rng):
    """The canonical hard case: ship 480 of 500, invoice all 500."""
    grn = docs[GRN]
    li = rng.choice([l for l in grn.line_items if l.quantity >= 2])
    short = max(D(1), q(li.quantity * D("0.10")).quantize(D("1")))
    docs = list(docs)
    lines = [l.model_copy(update={"quantity": l.quantity - short})
             if l.line_id == li.line_id else l for l in grn.line_items]
    docs[GRN] = grn.model_copy(update={"line_items": lines})
    return docs, PlantedAnomaly(
        anomaly_type=AnomalyType.PARTIAL_SHIPMENT,
        doc_ids_involved=[docs[GRN].doc_id, docs[INVOICE].doc_id],
        field_path=f"line_items[{li.line_id}].quantity",
        expected_delta=-short,
        note=f"received {li.quantity - short}, invoiced {li.quantity}")


def inject_near_duplicate(docs, rng):
    """A second invoice, one digit changed in the number — double billing."""
    inv = docs[INVOICE]
    num = list(inv.doc_number)
    for i in range(len(num) - 1, -1, -1):
        if num[i].isdigit():
            num[i] = str((int(num[i]) + 1) % 10)
            break
    dup = inv.model_copy(update={
        "doc_id": inv.doc_id + "-DUP",
        "doc_number": "".join(num),
        "doc_date": inv.doc_date.replace(
            day=min(inv.doc_date.day + rng.randint(2, 5), 28)),
    })
    docs = list(docs) + [dup]
    return docs, PlantedAnomaly(
        anomaly_type=AnomalyType.NEAR_DUPLICATE,
        doc_ids_involved=[inv.doc_id, dup.doc_id],
        field_path="doc_number",
        expected_delta=inv.total,       # the double-billed exposure
        note=f"{inv.doc_number} resubmitted as {dup.doc_number}")


def inject_unapplied_discount(docs, rng):
    """Agreed discount silently dropped from the invoice. Requires the chain
    to carry an allowance — the stratifier guarantees it."""
    inv = docs[INVOICE]
    assert inv.allowance_total is not None, "stratifier must force allowance"
    dropped = inv.allowance_total
    docs = list(docs)
    docs[INVOICE] = _recompute(inv.model_copy(
        update={"allowance_total": None}))
    docs[PAYMENT] = docs[PAYMENT].model_copy(
        update={"total": docs[INVOICE].total})
    return docs, PlantedAnomaly(
        anomaly_type=AnomalyType.UNAPPLIED_DISCOUNT,
        doc_ids_involved=[docs[PO].doc_id, docs[INVOICE].doc_id],
        field_path="allowance_total",
        expected_delta=dropped,
        note=f"agreed discount {dropped} absent from the invoice")


def inject_term_change(docs, rng):
    inv = docs[INVOICE]
    others = [t for t in ("Net 60", "Net 90", "Due on receipt", "Net 15")
              if t != inv.payment_terms]
    new_terms = rng.choice(others)
    docs = list(docs)
    docs[INVOICE] = inv.model_copy(update={"payment_terms": new_terms})
    return docs, PlantedAnomaly(
        anomaly_type=AnomalyType.TERM_CHANGE,
        doc_ids_involved=[docs[QUOTE].doc_id, docs[INVOICE].doc_id],
        field_path="payment_terms",
        expected_delta=None,
        note=f"quote says {docs[QUOTE].payment_terms!r}, "
             f"invoice says {new_terms!r}")


# --- stratification ----------------------------------------------------------

#: Slot table, cycled by chain index. 10 slots: 3 clean, 1 decoy, 6 anomalous
#: (one per type). With --n 20 every category appears exactly twice —
#: coverage by construction, independent of seed.
SLOTS = ("clean", "clean", "clean", "price_drift_decoy",
         "price_drift", "quantity_mismatch", "near_duplicate",
         "unapplied_discount", "term_change", "partial_shipment")


def slot_for(chain_index: int) -> str:
    return SLOTS[chain_index % len(SLOTS)]


def needs_allowance(chain_index: int) -> bool:
    return slot_for(chain_index) == "unapplied_discount"


def inject_for_slot(slot: str, docs, rng):
    """Returns (docs, anomalies) — empty list for clean chains."""
    if slot == "clean":
        return list(docs), []
    if slot == "price_drift_decoy":
        d, a = inject_price_drift(docs, rng, decoy=True)
        return d, [a]
    d, a = {
        "price_drift": inject_price_drift,
        "quantity_mismatch": inject_quantity_mismatch,
        "near_duplicate": inject_near_duplicate,
        "unapplied_discount": inject_unapplied_discount,
        "term_change": inject_term_change,
        "partial_shipment": inject_partial_shipment,
    }[slot](docs, rng)
    return d, [a]
