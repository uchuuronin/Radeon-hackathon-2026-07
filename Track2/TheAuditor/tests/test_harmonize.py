"""Reconciler tests. No GPU, no network.

Grounded in the real generator + injectors rather than hand-guessed fixtures:
each test takes ONE clean generated chain, applies the SAME injector
data/generator/inject.py uses to build the corpus, and asserts
src/harmonize/reconcile.py recovers the PlantedAnomaly it produced. This
ties the reconciler directly to ground truth semantics instead of to an
assumed shape of the problem -- if inject.py's convention for a field_path
or a doc pairing ever changes, these tests fail for the right reason.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonize.reconcile import reconcile  # noqa: E402
from schemas import AnomalyType, Chain  # noqa: E402

from gen import generate_chain  # noqa: E402
from inject import (  # noqa: E402
    QUOTE, SO, PO, GRN, INVOICE, PAYMENT,
    inject_near_duplicate,
    inject_partial_shipment,
    inject_price_drift,
    inject_quantity_mismatch,
    inject_term_change,
    inject_unapplied_discount,
    needs_allowance,
)

SEED = 4242


def _clean(i: int, force_allowance=False):
    return generate_chain(i, SEED, force_allowance=force_allowance).docs


def _run(docs) -> "ChainVerdict":
    docs_by_id = {d.doc_id: d for d in docs}
    chain = Chain(chain_id="CH-TEST", doc_ids=[d.doc_id for d in docs])
    return reconcile(chain, docs_by_id)


def _only(verdict, anomaly_type):
    return [d for d in verdict.discrepancies if d.anomaly_type == anomaly_type]


class TestCleanChainIsClean:

    def test_a_clean_chain_has_no_discrepancies(self):
        docs = _clean(0)
        verdict = _run(docs)
        assert verdict.is_clean
        assert verdict.discrepancies == []

    def test_clean_chain_still_produces_a_trace(self):
        """A clean verdict is a verdict, not silence -- 150 of 510 corpus
        chains are clean by construction and must be distinguishable from a
        reconciler that never ran."""
        docs = _clean(1)
        verdict = _run(docs)
        assert len(verdict.trace) >= 1


class TestPriceDrift:

    def test_gross_price_drift_is_detected_when_outside_the_band(self):
        """The generator's 'gross' drift is 3-9% OF ONE LINE'S unit price,
        which is not always enough to move a large invoice's TOTAL outside
        the 2%-capped-at-$100 cross-document band -- is_within_tolerance is
        the generator's own, honest verdict on the achieved delta, so assert
        against it rather than assuming 'gross' always means 'flagged'."""
        rng = random.Random(SEED)
        docs, anomaly = inject_price_drift(_clean(2), rng)
        verdict = _run(docs)
        found = _only(verdict, AnomalyType.PRICE_DRIFT)
        if anomaly.is_within_tolerance:
            assert found == []
        else:
            assert len(found) == 1
            d = found[0]
            assert set(d.doc_ids_involved) == set(anomaly.doc_ids_involved)
            assert d.field_path == anomaly.field_path

    def test_below_tolerance_decoy_is_not_flagged(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_price_drift(_clean(3), rng, decoy=True)
        assert anomaly.is_within_tolerance
        verdict = _run(docs)
        assert _only(verdict, AnomalyType.PRICE_DRIFT) == []

    def test_near_positive_just_above_band_is_flagged(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_price_drift(_clean(4), rng, band_fraction=1.20)
        assert not anomaly.is_within_tolerance
        verdict = _run(docs)
        assert len(_only(verdict, AnomalyType.PRICE_DRIFT)) == 1

    def test_near_negative_just_below_band_is_not_flagged(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_price_drift(_clean(5), rng, band_fraction=0.95)
        assert anomaly.is_within_tolerance
        verdict = _run(docs)
        assert _only(verdict, AnomalyType.PRICE_DRIFT) == []


class TestUnappliedDiscount:

    def test_dropped_allowance_is_reported_as_discount_not_generic_price_drift(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_unapplied_discount(
            _clean(6, force_allowance=True), rng)
        verdict = _run(docs)
        found = _only(verdict, AnomalyType.UNAPPLIED_DISCOUNT)
        assert len(found) == 1
        assert found[0].delta == anomaly.expected_delta
        # must NOT also double-report as an undifferentiated price drift
        assert _only(verdict, AnomalyType.PRICE_DRIFT) == []


class TestQuantity:

    def test_over_billed_quantity_is_a_mismatch_not_a_partial_shipment(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_quantity_mismatch(_clean(7), rng)
        verdict = _run(docs)
        found = _only(verdict, AnomalyType.QUANTITY_MISMATCH)
        assert len(found) == 1
        assert set(found[0].doc_ids_involved) == set(anomaly.doc_ids_involved)
        assert found[0].field_path == anomaly.field_path
        assert found[0].delta == anomaly.expected_delta
        assert _only(verdict, AnomalyType.PARTIAL_SHIPMENT) == []

    def test_short_shipment_is_a_partial_shipment_not_a_mismatch(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_partial_shipment(_clean(8), rng)
        verdict = _run(docs)
        found = _only(verdict, AnomalyType.PARTIAL_SHIPMENT)
        assert len(found) == 1
        assert set(found[0].doc_ids_involved) == set(anomaly.doc_ids_involved)
        assert found[0].field_path == anomaly.field_path
        assert _only(verdict, AnomalyType.QUANTITY_MISMATCH) == []


class TestNearDuplicate:

    def test_a_resubmitted_invoice_is_flagged_with_the_full_exposure(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_near_duplicate(_clean(9), rng)
        verdict = _run(docs)
        found = _only(verdict, AnomalyType.NEAR_DUPLICATE)
        assert len(found) == 1
        assert set(found[0].doc_ids_involved) == set(anomaly.doc_ids_involved)
        assert found[0].delta == anomaly.expected_delta


class TestTermChange:

    def test_a_changed_payment_term_is_flagged_quote_to_invoice(self):
        rng = random.Random(SEED)
        docs, anomaly = inject_term_change(_clean(10), rng)
        verdict = _run(docs)
        found = _only(verdict, AnomalyType.TERM_CHANGE)
        assert len(found) == 1
        assert set(found[0].doc_ids_involved) == set(anomaly.doc_ids_involved)
        assert found[0].delta is None       # non-numeric, per the schema


class TestOneAnomalyPerChainNoCrossTalk:

    def test_injecting_one_type_does_not_spuriously_flag_others(self):
        """A price-drift chain should not also light up quantity or term
        discrepancies -- cross-talk here would mean the comparisons aren't
        actually independent, which would corrupt per-type precision."""
        rng = random.Random(SEED)
        docs, _ = inject_quantity_mismatch(_clean(11), rng)
        verdict = _run(docs)
        assert _only(verdict, AnomalyType.PRICE_DRIFT) == []
        assert _only(verdict, AnomalyType.TERM_CHANGE) == []
        assert _only(verdict, AnomalyType.UNAPPLIED_DISCOUNT) == []
        assert _only(verdict, AnomalyType.NEAR_DUPLICATE) == []
