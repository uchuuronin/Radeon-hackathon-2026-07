"""Anomaly injection tests.

The property that matters most: every injected document is INTERNALLY
consistent and passes the whole verifier. The drift lives BETWEEN documents.
If an anomalous doc failed rung 0, the reconciler would have nothing to
prove — the deterministic layer would already have caught it.
"""

import random
from decimal import Decimal

import pytest

from gen import generate_chain, materialise
from inject import SLOTS, inject_for_slot, needs_allowance, slot_for
from schemas import (
    AnomalyType,
    CROSS_DOC_TOLERANCE,
    CheckOutcome,
    Layout,
    allowed_delta,
)
from verify.engine import verify_doc

SEED = 1337


def chain(i, slot=None):
    b = generate_chain(i, SEED, force_allowance=needs_allowance(i))
    rng = random.Random(SEED * 7_000_003 + i)
    docs, anomalies = inject_for_slot(slot or slot_for(i), b.docs, rng)
    return b, docs, anomalies


ALL = [chain(i) for i in range(20)]


def test_stratification_covers_every_type_deterministically():
    """One full cycle of the slot table gives one of everything, whatever the
    seed. Coverage is a property of the table, not of luck."""
    n = len(SLOTS)
    slots = [slot_for(i) for i in range(n)]
    assert slots.count("clean") == SLOTS.count("clean")
    for s in set(SLOTS) - {"clean"}:
        assert slots.count(s) == 1, s
    # Three cycles, three of each: the canonical corpus is a whole number of
    # cycles so per-type counts never depend on where the corpus was cut.
    triple = [slot_for(i) for i in range(3 * n)]
    for s in set(SLOTS):
        assert triple.count(s) == 3 * SLOTS.count(s), s


def test_holdout_slots_are_frozen():
    """Chains 0-4 are rendered into the sealed holdout. Reordering the head of
    the slot table would change what those committed files contain and there
    is no way to notice from inside the holdout, so the guard lives here."""
    assert SLOTS[:10] == (
        "clean", "clean", "clean", "price_drift_decoy",
        "price_drift", "quantity_mismatch", "near_duplicate",
        "unapplied_discount", "term_change", "partial_shipment")


def test_decoys_straddle_the_tolerance_boundary():
    """The corpus must contain cases on BOTH sides of the band and close to
    it. Without these every threshold between the easy negatives and the
    gross positives scores identically and the risk-coverage curve measures
    nothing."""
    from inject import BAND_FRACTIONS
    inside = [f for f in BAND_FRACTIONS.values() if f < 1]
    outside = [f for f in BAND_FRACTIONS.values() if f > 1]
    assert inside and outside
    # The decisive pair: nearest negative and nearest positive within a tenth
    # of the band of each other, demanding opposite verdicts.
    assert min(outside) - max(inside) <= 0.15


@pytest.mark.parametrize("slot,fraction", sorted(
    __import__("inject").BAND_FRACTIONS.items()))
def test_band_targeted_drift_lands_on_the_intended_side(slot, fraction):
    """Labels come from the ACHIEVED delta, so the corpus cannot contain an
    answer key that disagrees with its own documents. Cent quantisation moves
    the realised value; this asserts it never moves it across the band."""
    idx = SLOTS.index(slot)
    _, docs, anomalies = chain(idx)
    a = anomalies[0]
    assert a.is_within_tolerance == (fraction < 1), (
        f"{slot} aimed {fraction} but landed the other side: {a.note}")


def test_injection_is_deterministic():
    _, d1, a1 = chain(4)
    _, d2, a2 = chain(4)
    assert [x.model_dump_json() for x in d1] == [x.model_dump_json() for x in d2]
    assert [x.model_dump_json() for x in a1] == [x.model_dump_json() for x in a2]


@pytest.mark.parametrize("i", range(20))
def test_every_injected_document_still_passes_the_verifier(i):
    """THE property. Anomalies are invisible to rung 0 by construction —
    that is the argument for the reconciler existing."""
    _, docs, _ = ALL[i]
    for layout in (Layout.A, Layout.B):
        for doc in (d.model_copy(update={"source_text": r.source_text})
                    for d, r in zip(docs, _materialised(docs, layout))):
            report = verify_doc(doc)
            fails = [c for c in report.checks
                     if c.outcome == CheckOutcome.FAIL]
            assert not fails, (slot_for(i), doc.doc_id,
                               [(c.check, c.message) for c in fails])


def _materialised(docs, layout):
    from gen import RENDERERS
    render = RENDERERS[layout]
    return [d.model_copy(update={"source_text": render(d)}) for d in docs]


def test_price_drift_is_visible_across_documents_and_beyond_tolerance():
    _, docs, [a] = chain(4, "price_drift")
    po, inv = docs[2], docs[4]
    delta = abs(inv.total - po.total)
    assert delta > allowed_delta(po.total, CROSS_DOC_TOLERANCE), (
        "a real drift must exceed the cross-doc band or recall is untestable")


def test_decoy_drift_stays_inside_the_band():
    """Precision measured honestly: the system is scored on NOT flagging
    these."""
    _, docs, [a] = chain(3, "price_drift_decoy")
    assert a.is_within_tolerance
    po, inv = docs[2], docs[4]
    delta = abs(inv.total - po.total)
    assert Decimal(0) < delta <= allowed_delta(po.total, CROSS_DOC_TOLERANCE)


def test_unapplied_discount_drops_allowance_only_on_the_invoice():
    _, docs, [a] = chain(7, "unapplied_discount")
    assert docs[2].allowance_total is not None      # PO keeps the agreement
    assert docs[4].allowance_total is None          # invoice lost it
    assert a.expected_delta == docs[2].allowance_total
    assert docs[4].total > docs[2].total            # customer overbilled


def test_partial_shipment_receipt_short_of_invoice():
    _, docs, [a] = chain(9, "partial_shipment")
    grn_qty = {li.line_id: li.quantity for li in docs[3].line_items}
    inv_qty = {li.line_id: li.quantity for li in docs[4].line_items}
    lid = a.field_path.split("[")[1].split("]")[0]
    assert grn_qty[lid] < inv_qty[lid]
    assert a.expected_delta < 0


def test_near_duplicate_adds_a_seventh_coherent_invoice():
    key, docs, [a] = chain(6, "near_duplicate")
    assert len(docs) == 7
    orig, dup = docs[4], docs[6]
    assert dup.doc_number != orig.doc_number
    assert dup.total == orig.total                  # double-billed exposure
    assert a.expected_delta == orig.total
    assert not [c for c in verify_doc(dup).checks
                if c.outcome == CheckOutcome.FAIL]


def test_term_change_is_non_numeric():
    _, docs, [a] = chain(8, "term_change")
    assert a.expected_delta is None
    assert docs[0].payment_terms != docs[4].payment_terms


def test_payment_settles_the_invoice_as_billed():
    """The drifted amount flows through to payment — the fraud/mistake
    SUCCEEDED, which is what makes the case worth catching."""
    for i, slot in ((4, "price_drift"), (5, "quantity_mismatch"),
                    (7, "unapplied_discount")):
        _, docs, _ = chain(i, slot)
        assert docs[5].total == docs[4].total


def test_answer_key_field_paths_use_line_ids():
    for _, _, anomalies in ALL:
        for a in anomalies:
            if "line_items[" in a.field_path:
                assert "[LI-" in a.field_path
