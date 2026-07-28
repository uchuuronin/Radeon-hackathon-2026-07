"""A4 fixture integrity — the independent re-check of hand-written arithmetic.

The fixtures exist to be the oracle that gen.py cannot contaminate; this file
exists so a hand-computation typo cannot contaminate the oracle. Every
identity is re-derived here from the line items, not read back from the file.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from precision import amount_in_source
from schemas import DocType, ExtractedRecord, Tier

RECORDS = Path(__file__).parent.parent / "data" / "fixtures" / "records"
FILES = sorted(RECORDS.glob("*.json"))


@pytest.fixture(scope="module")
def fixtures() -> list[ExtractedRecord]:
    assert len(FILES) == 10, f"expected 10 fixtures, found {len(FILES)}"
    return [ExtractedRecord.model_validate_json(f.read_text(encoding="utf-8"))
            for f in FILES]


def priced(fixtures):
    return [r.doc for r in fixtures if r.doc.subtotal is not None]


# --- arithmetic re-derived from line items, never trusted from the field ----

def test_br_co_10_rederived(fixtures):
    for d in priced(fixtures):
        assert sum((li.line_total for li in d.line_items), Decimal(0)) \
            == d.subtotal, d.doc_id


def test_br_co_13_rederived(fixtures):
    for d in priced(fixtures):
        expected = (d.subtotal - (d.allowance_total or Decimal(0))
                    + (d.charge_total or Decimal(0)))
        assert d.total_excl_tax == expected, d.doc_id


def test_br_co_15_rederived(fixtures):
    for d in priced(fixtures):
        if d.tax is None:
            assert d.total == d.total_excl_tax, (
                f"{d.doc_id}: no tax line, total must equal net")
        else:
            assert d.total == d.total_excl_tax + d.tax, d.doc_id


def test_line_identity_rederived(fixtures):
    for r in fixtures:
        for li in r.doc.line_items:
            if li.unit_price is None:
                assert li.line_total is None, r.doc.doc_id
                continue
            expected = (li.quantity * li.unit_price
                        - (li.line_allowance or Decimal(0)))
            assert li.line_total == expected.quantize(Decimal("0.01")) \
                or li.line_total == expected, r.doc.doc_id


# --- every stated amount is really in the printed document ------------------

def test_amounts_present_in_source_by_value(fixtures):
    for r in fixtures:
        d = r.doc
        amounts = [d.subtotal, d.allowance_total, d.charge_total,
                   d.total_excl_tax, d.tax, d.total, d.amount_due]
        amounts += [li.unit_price for li in d.line_items]
        amounts += [li.line_total for li in d.line_items]
        for v in [a for a in amounts if a is not None]:
            assert amount_in_source(v, d.source_text), (
                f"{d.doc_id}: {v} not found in source_text")


def test_doc_number_present_in_source(fixtures):
    for r in fixtures:
        assert r.doc.doc_number in r.doc.source_text, r.doc.doc_id


def test_references_present_in_source(fixtures):
    for r in fixtures:
        for ref in r.doc.references:
            assert ref in r.doc.source_text, f"{r.doc.doc_id}: {ref}"


# --- coverage the set was designed to provide -------------------------------

def test_doc_type_coverage(fixtures):
    types = [r.doc.doc_type for r in fixtures]
    assert types.count(DocType.INVOICE) == 4
    assert types.count(DocType.PURCHASE_ORDER) == 2
    for t in (DocType.SALES_ORDER, DocType.QUOTE, DocType.PAYMENT,
              DocType.GOODS_RECEIPT):
        assert types.count(t) == 1, t


def test_all_four_allowance_charge_combos_present(fixtures):
    combos = {(d.allowance_total is not None, d.charge_total is not None)
              for d in priced(fixtures)}
    assert combos == {(False, False), (True, False), (False, True),
                      (True, True)}


def test_whole_unit_precision_fixture_exists(fixtures):
    """The Coupa PO states unit precision everywhere — the precision
    inference target. If every fixture printed cents, that path is untested
    against realistic input."""
    coupa = next(r.doc for r in fixtures if r.doc.doc_number == "PO-7731")
    assert ".00" not in coupa.source_text
    assert coupa.subtotal == Decimal("11100")


def test_goods_receipt_has_no_money(fixtures):
    grn = next(r.doc for r in fixtures
               if r.doc.doc_type == DocType.GOODS_RECEIPT)
    assert grn.line_items and all(
        li.unit_price is None and li.line_total is None
        for li in grn.line_items)
    assert grn.subtotal is None and grn.total is None


def test_cross_fixture_settlement(fixtures):
    """The payment settles the NetSuite order's total and the GRN receives
    the SAP PO's quantities — two ready-made linking/reconciliation cases."""
    by_num = {r.doc.doc_number: r.doc for r in fixtures}
    assert by_num["PMT-118764"].total == by_num["SO-2201"].total
    grn, po = by_num["GR-5000221"], by_num["4500012345"]
    assert po.doc_number in grn.references
    assert [li.quantity for li in grn.line_items] \
        == [li.quantity for li in po.line_items]


def test_provenance_is_hand_written(fixtures):
    for r in fixtures:
        assert r.meta.tier == Tier.NONE
        assert r.meta.model_id == "hand-written"
