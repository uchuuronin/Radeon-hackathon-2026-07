"""Reconciler tests. No GPU, no network, no model.

The corpus punishes false positives rather than rewarding recall, so the tests
that matter most are the ones asserting the reconciler stays SILENT: on decoys
sized just inside the band, and on the 150 chains that are clean by
construction.
"""
from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from harmonize.reconcile import ReconcilePolicy, reconcile        # noqa: E402
from schemas import (AnomalyType, CanonicalDoc, Chain, DocType,   # noqa: E402
                     LineItem)


def _li(lid: str, qty: str, price: str) -> LineItem:
    return LineItem(line_id=lid, description="widget", quantity=Decimal(qty),
                    unit_price=Decimal(price),
                    line_total=Decimal(qty) * Decimal(price))


def _doc(doc_id, number, dtype, lines=(), total=None, **kw) -> CanonicalDoc:
    lines = list(lines)
    sub = sum((li.line_total for li in lines), Decimal("0"))
    return CanonicalDoc(
        doc_id=doc_id, doc_number=number, doc_type=dtype,
        party_name="Averill Fastener GmbH", doc_date=date(2026, 6, 1),
        currency="EUR", line_items=lines, subtotal=sub or None,
        total=Decimal(total) if total is not None else (sub or None),
        source_text=f"{number}", **kw)


def _chain(*docs) -> tuple[Chain, dict]:
    by_id = {d.doc_id: d for d in docs}
    return (Chain(chain_id="CH-T", doc_ids=sorted(by_id)), by_id)


class TestTheAggregateGate:
    """The single most important rule: decide on the total, attribute at the
    line. A decoy's line value genuinely differs, so field-by-field comparison
    cannot separate it from the real thing."""

    def _pair(self, po_price: str, inv_price: str):
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER, [_li("LI-001", "100", po_price)])
        inv = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "100", inv_price)])
        return _chain(po, inv)

    def test_a_drift_beyond_the_band_is_flagged_and_attributed_to_the_line(self):
        chain, docs = self._pair("50.00", "70.00")     # totals differ by 2000
        v = reconcile(chain, docs)
        assert len(v.discrepancies) == 1
        d = v.discrepancies[0]
        assert d.anomaly_type == AnomalyType.PRICE_DRIFT
        assert d.field_path == "line_items[LI-001].unit_price"
        assert d.delta == Decimal("20.00")
        # It must be able to show WHAT the flag was decided on, not just where.
        assert d.decided_on_delta and d.decided_on_band

    def test_a_line_that_differs_inside_the_band_is_NOT_flagged(self):
        """The decoy shape: the unit price really is different, and flagging it
        is the failure the corpus exists to provoke. 130 of 210 planted price
        drifts are built this way."""
        chain, docs = self._pair("50.00", "50.10")     # totals differ by 10.00
        v = reconcile(chain, docs)
        assert v.is_clean
        assert any("within band" in t for t in v.trace)

    def test_identical_documents_produce_a_clean_verdict_not_silence(self):
        """150 chains are clean by construction. A reconciler that emits
        nothing for them is indistinguishable from one that crashed."""
        chain, docs = self._pair("50.00", "50.00")
        v = reconcile(chain, docs)
        assert v.is_clean and v.trace


class TestTheMoneyGateIsNotAppliedWhereItIsUndefined:

    def test_a_goods_receipt_is_compared_on_quantity_not_money(self):
        """A goods receipt states no total, so the money gate is not
        inappropriate here, it is undefined."""
        grn = _doc("D-4", "GRN-1", DocType.GOODS_RECEIPT, [_li("LI-001", "10", "0")])
        grn.total = grn.subtotal = None
        inv = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "11", "50.00")])
        chain, docs = _chain(grn, inv)
        v = reconcile(chain, docs)
        assert len(v.discrepancies) == 1
        d = v.discrepancies[0]
        assert d.field_path == "line_items[LI-001].quantity"
        assert d.delta == Decimal("1")
        # No order to arbitrate with, so the label is a guess and says so.
        assert any("labelled by sign only" in t for t in v.trace)


class TestTheOrderArbitratesWhichDocumentMoved:
    """`received < billed` cannot tell an over-bill from a short shipment: both
    produce the identical shape and their distributions overlap completely. The
    ORDER is the third reference point. Whichever of receipt and invoice still
    agrees with what was ordered is the document that did not move."""

    def _three(self, ordered, received, billed):
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER,
                  [_li("LI-001", ordered, "50.00")])
        grn = _doc("D-4", "GRN-1", DocType.GOODS_RECEIPT,
                   [_li("LI-001", received, "0")])
        grn.total = grn.subtotal = None
        inv = _doc("D-5", "INV-1", DocType.INVOICE,
                   [_li("LI-001", billed, "50.00")])
        return _chain(po, grn, inv)

    def test_receipt_matches_the_order_so_the_invoice_over_billed(self):
        chain, docs = self._three("10", "10", "12")
        d = next(x for x in reconcile(chain, docs).discrepancies
                 if "quantity" in x.field_path)
        assert d.anomaly_type == AnomalyType.QUANTITY_MISMATCH
        # An over-bill reports what was ADDED.
        assert d.delta == Decimal("2")

    def test_invoice_matches_the_order_so_the_shipment_fell_short(self):
        chain, docs = self._three("10", "8", "10")
        d = next(x for x in reconcile(chain, docs).discrepancies
                 if "quantity" in x.field_path)
        assert d.anomaly_type == AnomalyType.PARTIAL_SHIPMENT
        # A short shipment reports what is MISSING, so the sign flips. The
        # number is unreadable without the label otherwise.
        assert d.delta == Decimal("-2")

    def test_when_neither_matches_the_order_it_is_not_attributed(self):
        """Genuinely ambiguous. Isolate rather than guess, the same principle
        the linker applies to an unresolvable reference: a wrong label sends an
        analyst looking for the wrong kind of problem."""
        chain, docs = self._three("10", "8", "12")
        v = reconcile(chain, docs)
        assert not [x for x in v.discrepancies if "quantity" in x.field_path]
        assert any("neither matches the order" in t for t in v.trace)

    def test_a_missing_total_is_never_reported_as_clean(self):
        """Silently passing a document we could not check is the worst
        available outcome: it looks exactly like a clean result."""
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER, [_li("LI-001", "1", "5.00")])
        inv = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "1", "5.00")])
        inv.total = None
        chain, docs = _chain(po, inv)
        v = reconcile(chain, docs)
        assert any("not applicable" in t and "NOT treated as clean" in t
                   for t in v.trace)


class TestOneEventReportedOnce:

    def test_a_quantity_finding_defers_to_the_goods_receipt(self):
        """Billing more units than ordered pushes the order-to-invoice total
        out of band with every unit price untouched. The same event is already
        visible, and better evidenced, against the receipt: the order says what
        was agreed, the receipt says what arrived. Reporting both produced 26
        false positives, every one of them `price_drift` at `total`."""
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER, [_li("LI-001", "10", "50.00")])
        grn = _doc("D-4", "GRN-1", DocType.GOODS_RECEIPT, [_li("LI-001", "10", "0")])
        grn.total = grn.subtotal = None
        inv = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "20", "50.00")])
        chain, docs = _chain(po, grn, inv)
        v = reconcile(chain, docs)
        paths = {d.field_path for d in v.discrepancies}
        assert paths == {"line_items[LI-001].quantity"}
        assert any("deferred to the goods receipt" in t for t in v.trace)

    def test_with_no_receipt_the_order_carries_it(self):
        """Nothing else can, so suppressing it would lose the finding."""
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER, [_li("LI-001", "10", "50.00")])
        inv = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "20", "50.00")])
        chain, docs = _chain(po, inv)
        assert reconcile(chain, docs).discrepancies


class TestAnomaliesWithNoAggregateSignature:
    """60 of the 230 true positives move no total at all. An aggregate-only
    detector is blind to every one of them."""

    def test_a_duplicate_invoice_reports_the_full_exposure_not_a_difference(self):
        """Two invoices citing the same upstream documents for the same amount
        is a duplicate however different their numbers. Paying both costs the
        whole second invoice, so `delta` is the amount at risk."""
        a = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "1", "500.00")],
                 references=["PO-1"])
        b = _doc("D-5-DUP", "INV-2", DocType.INVOICE, [_li("LI-001", "1", "500.00")],
                 references=["PO-1"])
        chain, docs = _chain(a, b)
        v = reconcile(chain, docs)
        assert len(v.discrepancies) == 1
        d = next(x for x in v.discrepancies
                 if x.anomaly_type == AnomalyType.NEAR_DUPLICATE)
        assert d.field_path == "doc_number"
        assert d.delta == Decimal("500.00")

    def test_a_changed_payment_term_is_flagged_with_no_delta(self):
        q = _doc("D-1", "Q-1", DocType.QUOTE, [_li("LI-001", "1", "5.00")],
                 payment_terms="Due on receipt")
        inv = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "1", "5.00")],
                   payment_terms="Net 15")
        chain, docs = _chain(q, inv)
        d = reconcile(chain, docs).discrepancies[0]
        assert d.anomaly_type == AnomalyType.TERM_CHANGE
        assert d.delta is None

    def test_an_absent_allowance_is_attributed_to_the_document_not_a_line(self):
        """When a discount goes missing the line prices are usually untouched,
        so attributing it to a line sends an analyst to a correct line."""
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER,
                  [_li("LI-001", "100", "50.00")], total="4000.00")
        po.allowance_total = Decimal("1000.00")
        inv = _doc("D-5", "INV-1", DocType.INVOICE,
                   [_li("LI-001", "100", "50.00")], total="5000.00")
        chain, docs = _chain(po, inv)
        d = reconcile(chain, docs).discrepancies[0]
        assert d.anomaly_type == AnomalyType.UNAPPLIED_DISCOUNT
        assert d.field_path == "allowance_total"


class TestVerdictShape:

    def test_a_chain_with_no_invoice_is_incomplete_not_clean(self):
        """The difference between "we checked and it was fine" and "there was
        nothing to check"."""
        q = _doc("D-1", "Q-1", DocType.QUOTE, [_li("LI-001", "1", "5.00")])
        chain, docs = _chain(q)
        v = reconcile(chain, docs)
        assert v.is_clean
        assert any("nothing to reconcile" in t for t in v.trace)

    def test_a_duplicate_does_not_double_report_the_upstream_drift(self):
        """A chain holding a near-duplicate walks the same purchase order
        twice, once per invoice."""
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER,
                  [_li("LI-001", "100", "50.00")], references=[])
        a = _doc("D-5", "INV-1", DocType.INVOICE,
                 [_li("LI-001", "100", "70.00")], references=["PO-1"])
        b = _doc("D-5-DUP", "INV-2", DocType.INVOICE,
                 [_li("LI-001", "100", "70.00")], references=["PO-1"])
        chain, docs = _chain(po, a, b)
        v = reconcile(chain, docs)
        drifts = [d for d in v.discrepancies
                  if d.anomaly_type == AnomalyType.PRICE_DRIFT]
        assert len({(tuple(d.doc_ids_involved), d.field_path) for d in drifts}) \
            == len(drifts)

    def test_every_pair_walked_leaves_a_trace_line(self):
        """The audit record, wired in from the start rather than retrofitted."""
        po = _doc("D-3", "PO-1", DocType.PURCHASE_ORDER, [_li("LI-001", "1", "5.00")])
        inv = _doc("D-5", "INV-1", DocType.INVOICE, [_li("LI-001", "1", "5.00")])
        chain, docs = _chain(po, inv)
        assert any("D-3" in t and "D-5" in t for t in reconcile(chain, docs).trace)


class TestAgainstTheCorpus:

    def test_the_full_pipeline_holds_precision_on_the_decoys(self):
        import json
        from bench.detect_score import load_corpus, score
        from linker.link import link
        from harmonize.reconcile import reconcile_all
        recs = ROOT / "data/generated/records.jsonl"
        if not recs.exists():
            pytest.skip("run data/generator/gen.py first")
        docs, key = load_corpus(recs, ROOT / "data/generated/answer_key.json")
        s = score(reconcile_all(link(list(docs.values())), docs), key, docs)
        # The corpus punishes false positives. These are the two numbers that
        # decide whether the system is usable at all.
        assert s.decoys_flagged == 0, "a below-tolerance decoy was flagged"
        assert s.clean_chains_flagged == 0, "a clean chain was flagged"
        assert s.precision[0] == 1.0
        assert s.recall[0] > 0.95


class TestStrictDuplicateDetection:
    """Two invoices in one chain is not enough. On this corpus all 30
    multi-invoice chains are planted duplicates so the loose rule costs
    nothing, but partial billing puts two legitimate invoices against one
    order and the loose rule flags it."""

    def _two(self, total_a, total_b, refs_a, refs_b):
        a = _doc("D-5", "INV-1", DocType.INVOICE,
                 [_li("LI-001", "1", total_a)], references=list(refs_a))
        b = _doc("D-6", "INV-2", DocType.INVOICE,
                 [_li("LI-001", "1", total_b)], references=list(refs_b))
        return _chain(a, b)

    def test_different_totals_are_not_a_duplicate(self):
        chain, docs = self._two("500.00", "300.00", ["PO-1"], ["PO-1"])
        v = reconcile(chain, docs)
        assert not v.discrepancies
        assert any("different totals" in t for t in v.trace)

    def test_same_total_against_different_orders_is_not_a_duplicate(self):
        chain, docs = self._two("500.00", "500.00", ["PO-1"], ["PO-9"])
        v = reconcile(chain, docs)
        assert not v.discrepancies
        assert any("different upstream documents" in t for t in v.trace)
