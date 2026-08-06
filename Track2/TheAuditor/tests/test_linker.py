"""Linker tests. No GPU, no network, no model.

The linker's failures are asymmetric, and the tests are shaped around that
rather than around coverage. A MISSED link leaves a singleton, which is visible
and recoverable. A WRONG link welds two deals together and every comparison
afterwards runs against the wrong baseline, so it does not present as a linking
failure at all: it presents as a discrepancy that is not there. So the negative
tests (refuses to guess) matter more here than the positive ones.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linker.link import LinkPolicy, link                       # noqa: E402
from schemas import CanonicalDoc, DocType, LinkMethod          # noqa: E402


def _doc(doc_id: str, number: str, dtype: DocType, refs=(),
         party="Averill Fastener GmbH", day=1, total="1000.00") -> CanonicalDoc:
    return CanonicalDoc(
        doc_id=doc_id, doc_number=number, doc_type=dtype, party_name=party,
        doc_date=date(2026, 1, 1) + timedelta(days=day), currency="EUR",
        references=list(refs),
        total=Decimal(total), source_text=f"{number} {total}")


def _chain(prefix: str, party="Averill Fastener GmbH", base="7000", day=1):
    """The real topology: a DAG with a diamond at the invoice."""
    return [
        _doc(f"{prefix}-1", f"Q-{base}", DocType.QUOTE, (), party, day),
        _doc(f"{prefix}-2", f"SO-{base}", DocType.SALES_ORDER,
             (f"Q-{base}",), party, day + 1),
        _doc(f"{prefix}-3", f"PO-{base}-A", DocType.PURCHASE_ORDER,
             (f"Q-{base}",), party, day + 2),
        _doc(f"{prefix}-4", f"GRN-{base}", DocType.GOODS_RECEIPT,
             (f"PO-{base}-A",), party, day + 3),
        _doc(f"{prefix}-5", f"INV-{base}", DocType.INVOICE,
             (f"PO-{base}-A", f"GRN-{base}"), party, day + 4),
        _doc(f"{prefix}-6", f"PAY-{base}", DocType.PAYMENT,
             (f"INV-{base}",), party, day + 5),
    ]


class TestGroupingOnReferences:

    def test_the_diamond_does_not_split_the_chain(self):
        """The invoice cites BOTH the purchase order and the goods receipt, so
        the reference graph is a DAG, not a tree. A parent-walk would visit the
        invoice twice and a tree structure could not hold it; connected
        components can, because direction is not part of the question."""
        chains = link(_chain("D"))
        assert len(chains) == 1
        assert len(chains[0].doc_ids) == 6
        assert {str(e.method) for e in chains[0].edges} == {
            LinkMethod.EXACT_REFERENCE}

    def test_two_deals_stay_two_deals(self):
        chains = link(_chain("A", base="7000") + _chain("B", base="8000"))
        assert len(chains) == 2
        assert all(len(c.doc_ids) == 6 for c in chains)

    def test_chain_id_is_stable_under_reordering(self):
        """The same corpus linked twice, or in a different order, or split
        across two runs, must produce the same chain_id or nothing downstream
        can be joined back to it."""
        docs = _chain("D")
        a = link(docs)
        b = link(list(reversed(docs)))
        assert [c.chain_id for c in a] == [c.chain_id for c in b]
        assert [sorted(c.doc_ids) for c in a] == [sorted(c.doc_ids) for c in b]

    def test_a_reference_to_a_document_we_never_saw_is_recorded_not_dropped(self):
        """A corpus sliced by date or party legitimately contains half a chain.
        That is not an error, but a rising count is the signal that the linker
        is being fed a partial view, which would otherwise present downstream
        as unexplained discrepancies."""
        docs = [d for d in _chain("D") if d.doc_id != "D-3"]   # remove the PO
        chains = link(docs)
        dangling = [r for c in chains for r in c.dangling_references]
        assert "PO-7000-A" in dangling


class TestTheLinkerGroupsAndDoesNotJudge:

    def test_a_near_duplicate_stays_inside_its_chain(self):
        """The duplicate carries IDENTICAL references and an identical total;
        only its number differs. Grouping it is correct. Whether a chain
        holding two invoices against one purchase order is a duplicate is the
        reconciler's question, and keeping the concerns apart is what stops a
        linking bug from presenting as a false discrepancy, and stops a real
        duplicate from being silently repaired by dropping it from the chain.
        """
        docs = _chain("D")
        docs.append(_doc("D-5-DUP", "INV-7001", DocType.INVOICE,
                         ("PO-7000-A", "GRN-7000"), day=5))
        chains = link(docs)
        assert len(chains) == 1
        assert "D-5-DUP" in chains[0].doc_ids


class TestRepairTargetsTheDocumentedFailure:

    def test_a_single_glyph_confusion_is_repaired_with_its_rationale(self):
        """0/O in an identifier is the hardest field class in every published
        extraction comparison, and the one that silently detaches a document
        from its deal."""
        docs = _chain("D")
        docs[5].references = ["INV-70O0"]              # zero read as letter O
        chains = link(docs)
        assert len(chains) == 1
        edge = next(e for e in chains[0].edges
                    if str(e.method) == LinkMethod.REPAIRED_REFERENCE)
        assert edge.stated_reference == "INV-70O0"
        assert "INV-7000" in edge.rationale

    def test_repair_is_glyph_confusion_not_edit_distance(self):
        """"PO-1001" and "PO-1002" are one edit apart and are different
        purchase orders. A plain Levenshtein-1 rule would merge unrelated
        deals, so repair is restricted to confusions that actually occur."""
        a = _doc("A-1", "PO-1001", DocType.PURCHASE_ORDER)
        b = _doc("B-1", "PO-1002", DocType.PURCHASE_ORDER)
        grn = _doc("A-2", "GRN-1", DocType.GOODS_RECEIPT, ("PO-1003",))
        chains = link([a, b, grn])
        assert len(chains) == 3                      # no guess, three isolates

    def test_an_ambiguous_repair_is_refused(self):
        """Two repair candidates is the dangerous case: a guess would be a coin
        flip between real documents."""
        # Both known numbers are ONE confusion from the stated reference:
        # "INV-100" reaches "INV-1O0" and "INV-10O" alike.
        a = _doc("A-1", "INV-1O0", DocType.INVOICE, party="Alpha Ltd")
        b = _doc("B-1", "INV-10O", DocType.INVOICE, party="Beta Ltd")
        pay = _doc("C-1", "PAY-9", DocType.PAYMENT, ("INV-100",),
                   party="Gamma Ltd")
        chains = link([a, b, pay])
        assert len(chains) == 3


class TestSharedDocumentNumbers:
    """Document numbers are not globally unique in practice: real systems run
    per-vendor or per-year sequences. The corpus reproduces it, with 92 numbers
    held by two documents each."""

    def test_a_shared_number_is_disambiguated_by_party(self):
        left = _chain("A", party="Alpha Ltd", base="7000")
        right = _chain("B", party="Beta Ltd", base="7000")   # SAME numbers
        chains = link(left + right)
        assert len(chains) == 2
        for c in chains:
            assert len(c.doc_ids) == 6
            assert not {d for d in c.doc_ids if d.startswith("A")} or \
                   not {d for d in c.doc_ids if d.startswith("B")}

    def test_an_undisambiguable_reference_isolates_rather_than_merging(self):
        """Same party, same number, same week. Guessing here does not merely
        fail to link, it welds two real deals together."""
        a = _doc("A-1", "INV-7000", DocType.INVOICE, day=1)
        b = _doc("B-1", "INV-7000", DocType.INVOICE, day=2)
        pay = _doc("C-1", "PAY-1", DocType.PAYMENT, ("INV-7000",), day=3)
        chains = link([a, b, pay])
        assert len(chains) == 3
        assert any("INV-7000" in c.dangling_references for c in chains)


class TestAttributeFallback:

    def _orphan(self, **kw):
        return _doc("X-1", "PAY-ZZZ", DocType.PAYMENT, (), **kw)

    def test_all_three_attributes_agreeing_attaches_the_document(self):
        docs = _chain("D")
        chains = link(docs + [self._orphan(day=6, total="1000.00")])
        assert len(chains) == 1
        edge = next(e for e in chains[0].edges
                    if str(e.method) == LinkMethod.ATTRIBUTE_MATCH)
        assert "no other chain qualified" in edge.rationale

    @pytest.mark.parametrize("kw", [
        {"party": "Totally Different Ltd", "day": 6, "total": "1000.00"},
        {"day": 250, "total": "1000.00"},              # outside the date window
        {"day": 6, "total": "99999.00"},               # no amount agrees
    ])
    def test_any_one_attribute_disagreeing_isolates(self, kw):
        """Necessary conditions, not a score. Two out of three is how a foreign
        document gets welded into a deal."""
        chains = link(_chain("D") + [self._orphan(**kw)])
        assert len(chains) == 2
        assert any(c.is_singleton for c in chains)

    def test_two_qualifying_chains_isolate_rather_than_coin_flip(self):
        """A missed link is visible and recoverable. A wrong one is neither."""
        docs = _chain("A", base="7000") + _chain("B", base="8000")
        chains = link(docs + [self._orphan(day=6, total="1000.00")])
        assert len(chains) == 3
        assert any(c.is_singleton for c in chains)

    def test_the_fallback_can_be_switched_off_for_measurement(self):
        """The exact-reference path has to be measurable in isolation, or a
        headline number cannot say which mechanism earned it."""
        docs = _chain("D") + [self._orphan(day=6, total="1000.00")]
        chains = link(docs, LinkPolicy(attribute_fallback=False))
        assert len(chains) == 2
