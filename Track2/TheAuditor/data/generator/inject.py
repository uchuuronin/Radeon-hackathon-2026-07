"""Anomaly injection — chain-level drift on top of clean chains.

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
Doc-level corruption (extraction errors) belongs to the corrupted fixtures,
already done.

DECOYS, AND WHY THEY ARE PLACED WHERE THEY ARE
---------------------------------------------
Some price drifts are planted BELOW the cross-document tolerance band and
recorded with is_within_tolerance=True. The system is scored on NOT flagging
them: precision measured honestly, not "flag everything and claim recall".

The FIRST version of this got the idea right and the magnitudes wrong.
Measured, its decoys sat at roughly 0.4% of the band while genuine drifts
sat at 1.5x to 20x it, so the whole region next to the decision boundary was
empty. Nothing in the corpus could distinguish a threshold at 0.5x from one
at 1.4x, every such threshold scored perfectly, and the risk-coverage curve
those scores produce is a straight line through empty space. Easy negatives
measure nothing; the informative ones sit where the decision is actually
hard.

So decoys are now STRATIFIED ACROSS THE BOUNDARY: negatives at 0.50, 0.80
and 0.95 of the band (must not flag) and near-positives at 1.05 and 1.20
(must flag). The 0.95 and 1.05 pair differ by a tenth of the band and demand
opposite verdicts, which is the case that decides whether the tolerance model
is real.

Placement is by ACHIEVED delta, never by intent: the per-unit price is
quantised to cents, so the realised total can land either side of the
requested fraction. is_within_tolerance is set from what the document
actually says after injection, and tests/test_inject.py asserts the achieved
fraction is on the intended side of 1.0.
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

def _tax_multiplier(doc) -> D:
    """How a change to the subtotal propagates to the total on this document.

    A line change of X moves the subtotal by X, and tax is recomputed off the
    new net, so the TOTAL moves by X * (1 + rate). Targeting a band fraction
    on the total without this lands every decoy roughly one tax rate away
    from where it was aimed, which for a 20% VAT chain is enough to push a
    0.95-of-band negative over the line and invert its label.
    """
    if doc.tax is None or not doc.total_excl_tax:
        return D(1)
    return D(1) + (doc.tax / doc.total_excl_tax)


def inject_price_drift(docs, rng, decoy=False, band_fraction=None):
    """Price drift, optionally aimed at a chosen multiple of the tolerance band.

    band_fraction=None  gross drift, 3-9% of unit price (the obvious case).
    band_fraction=f     solve for the per-unit change whose effect on the
                        invoice TOTAL is f x the cross-document band. f < 1
                        is a negative the system must not flag; f > 1 is a
                        near-positive it must.
    """
    inv = docs[INVOICE]
    li = _priced_line(inv, rng)
    po_total = docs[PO].total
    band = allowed_delta(po_total, CROSS_DOC_TOLERANCE)

    if band_fraction is not None:
        target = band * D(str(band_fraction))
        per_unit = q(target / (li.quantity * _tax_multiplier(inv)))
        if per_unit == 0:                 # band too tight for cent precision
            per_unit = D("0.01")
    elif decoy:
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

    # Label from what the corpus ACTUALLY contains, not from what was asked
    # for. Cent quantisation moves the realised delta, and a decoy whose
    # achieved value crossed the band would be an answer key that lies.
    achieved = abs(docs[INVOICE].total - po_total)
    fraction = achieved / band if band else D(0)
    within = fraction <= 1

    if band_fraction is not None:
        note = (f"aimed {band_fraction:.2f}x band, achieved {fraction:.3f}x "
                f"({achieved} of ±{band}); "
                + ("must NOT flag" if within else "must flag"))
    elif decoy:
        note = "decoy: total delta inside the cross-doc band; must NOT flag"
    else:
        note = f"invoice bills {new_price} vs agreed {li.unit_price}"

    return docs, PlantedAnomaly(
        anomaly_type=AnomalyType.PRICE_DRIFT,
        doc_ids_involved=[docs[PO].doc_id, docs[INVOICE].doc_id],
        field_path=f"line_items[{li.line_id}].unit_price",
        expected_delta=per_unit,
        is_within_tolerance=within,
        note=note)


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
    """The canonical hard case: ship 480 of 500, invoice all 500.

    Line choice is deliberately not a plain filter on quantity >= 2. It was,
    and at 20 chains it always found one; at 480 chains a chain eventually
    turns up whose every line is a single unit, and the filter returned an
    empty list and took the whole generator down. A generator that works
    until the corpus is big enough to be worth measuring is worse than one
    that fails immediately, so: prefer a shippable-in-part line, fall back to
    the largest line and short it by half, and never emit a receipt for a
    negative or zero quantity.
    """
    grn = docs[GRN]
    divisible = [l for l in grn.line_items if l.quantity >= 2]
    if divisible:
        li = rng.choice(divisible)
        short = max(D(1), q(li.quantity * D("0.10")).quantize(D("1")))
    else:
        li = max(grn.line_items, key=lambda l: l.quantity)
        short = (li.quantity / 2).quantize(D("0.1"))
    short = min(short, li.quantity - D("0.1"))     # something must arrive
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

#: Slot table, cycled by chain index. Coverage by construction, independent
#: of seed: at --n 480 every slot appears exactly 30 times, which is the size
#: calibrate.required_n_for_halfwidth says a per-type recall claim needs to
#: carry a Wilson half-width under ten points.
#:
#: THE FIRST TEN ENTRIES ARE FROZEN AND MUST NOT BE REORDERED. Chains 0-4 are
#: the sealed holdout in data/fixtures/holdout/docs/, rendered and committed
#: on Day 2. Those files were generated from slots 0-4, so touching any of
#: them silently invalidates the seal: the holdout documents would no longer
#: correspond to any answer key the generator can produce, and the one
#: artifact whose value depends entirely on never having been reopened would
#: be quietly worthless. New behaviour is APPENDED, never inserted.
SLOTS = (
    # --- FROZEN: chains 0-9, and the holdout depends on 0-4 ----------------
    "clean", "clean", "clean", "price_drift_decoy",
    "price_drift", "quantity_mismatch", "near_duplicate",
    "unapplied_discount", "term_change", "partial_shipment",
    # --- APPENDED: boundary-straddling hard cases --------------------------
    # Negatives, ascending towards the band. The system must flag NONE.
    "decoy_050", "decoy_080", "decoy_095",
    # Near-positives, just past it. The system must flag ALL.
    # decoy_095 and drift_105 differ by a tenth of the band and demand
    # opposite verdicts: that pair is the corpus's real discriminating power.
    "drift_105", "drift_120",
    # Keeps clean chains at a quarter of the corpus so precision has enough
    # true negatives to be a meaningful denominator.
    "clean", "clean",
)

#: Slot name -> band multiple, for the stratified price-drift slots.
BAND_FRACTIONS = {"decoy_050": 0.50, "decoy_080": 0.80, "decoy_095": 0.95,
                  "drift_105": 1.05, "drift_120": 1.20}


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
    if slot in BAND_FRACTIONS:
        d, a = inject_price_drift(docs, rng,
                                  band_fraction=BAND_FRACTIONS[slot])
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
