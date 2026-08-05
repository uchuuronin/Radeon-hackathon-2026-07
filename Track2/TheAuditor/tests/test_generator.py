"""A1 generator correctness.

The answer key is the deliverable, so the generator IS the ground truth. If it
emits an internally inconsistent document by accident, every precision/recall
number downstream is measuring the wrong thing and nothing else in the project
can detect it. These tests are the only thing standing between us and that.
"""

from decimal import Decimal

import pytest

from gen import RENDERERS, generate_chain, materialise
from precision import amount_in_source
from schemas import DocType, Layout

SEED = 1337
CHAINS = [generate_chain(i, SEED) for i in range(25)]
ALL_DOCS = [d for c in CHAINS for d in c.docs]
PRICED = [d for d in ALL_DOCS if d.subtotal is not None]


# --- determinism -------------------------------------------------------------

def test_same_seed_reproduces_byte_identical_output():
    """Benchmarks on Day 2 and Day 13 must be comparable, and the demo set has
    to survive the platform stress test killing the instance."""
    a = materialise(generate_chain(7, SEED), Layout.A)
    b = materialise(generate_chain(7, SEED), Layout.A)
    assert [d.model_dump_json() for d in a] == [d.model_dump_json() for d in b]


def test_different_seed_produces_different_data():
    a = generate_chain(7, SEED).docs[4]
    b = generate_chain(7, SEED + 1).docs[4]
    assert (a.total, a.party_name) != (b.total, b.party_name)


# --- EN 16931 identities hold EXACTLY on clean data --------------------------

@pytest.mark.parametrize("doc", PRICED, ids=lambda d: d.doc_id)
def test_br_co_10_lines_sum_to_subtotal(doc):
    assert sum((li.line_total for li in doc.line_items), Decimal(0)) == doc.subtotal


@pytest.mark.parametrize("doc", PRICED, ids=lambda d: d.doc_id)
def test_br_co_13_net_identity(doc):
    expected = (doc.subtotal
                - (doc.allowance_total or Decimal(0))
                + (doc.charge_total or Decimal(0)))
    assert doc.total_excl_tax == expected


@pytest.mark.parametrize("doc", PRICED, ids=lambda d: d.doc_id)
def test_br_co_15_gross_identity(doc):
    assert doc.total == doc.total_excl_tax + doc.tax


@pytest.mark.parametrize("doc", PRICED, ids=lambda d: d.doc_id)
def test_line_net_amount_identity(doc):
    for li in doc.line_items:
        assert li.line_total == (li.quantity * li.unit_price).quantize(Decimal("0.01"))


# --- document semantics ------------------------------------------------------

def test_goods_receipt_has_quantities_but_no_money():
    """The case that forces LineItem prices to be Optional. A GRN records
    what arrived, not what it cost."""
    grns = [d for d in ALL_DOCS if d.doc_type == DocType.GOODS_RECEIPT]
    assert grns
    for g in grns:
        assert g.line_items
        for li in g.line_items:
            assert li.quantity > 0
            assert li.unit_price is None
            assert li.line_total is None
        assert g.subtotal is None and g.total is None


def test_payment_has_no_line_items():
    pays = [d for d in ALL_DOCS if d.doc_type == DocType.PAYMENT]
    assert pays
    assert all(p.line_items == [] and p.total is not None for p in pays)


def test_absent_values_are_none_never_zero():
    for d in ALL_DOCS:
        for v in (d.subtotal, d.allowance_total, d.charge_total, d.tax,
                  d.total, d.amount_due):
            assert v != Decimal(0), f"{d.doc_id}: zero used for an absent value"


def test_doc_id_is_opaque():
    """doc_id must encode nothing. If it leaked doc_type, the linker could
    cheat and the linking metric would be meaningless."""
    for d in ALL_DOCS:
        assert d.doc_type.lower() not in d.doc_id.lower()
        assert d.doc_number not in d.doc_id


def test_references_resolve_within_the_chain():
    for c in CHAINS:
        numbers = {d.doc_number for d in c.docs}
        for d in c.docs:
            for ref in d.references:
                assert ref in numbers, f"{d.doc_id} cites unknown {ref}"


def test_chain_is_six_documents_in_date_order():
    for c in CHAINS:
        assert len(c.docs) == 6
        dates = [d.doc_date for d in c.docs]
        assert dates == sorted(dates)


# --- the verbatim-amount check, across both layouts --------------------------

@pytest.mark.parametrize("layout", [Layout.A, Layout.B])
def test_every_extracted_amount_appears_in_source_text(layout):
    """Resolves the open DECIDE in schemas.py.

    A raw substring search fails here: Layout A prints '2,477.00' and Layout B
    prints '2477' for the same value, and BOTH extractions are correct. The
    check has to compare VALUES, not strings. This test is what proves the
    verifier's amounts_appear_in_source can be layout-robust without being
    layout-specific.
    """
    for bundle in CHAINS[:10]:
        for doc in materialise(bundle, layout):
            amounts = [doc.subtotal, doc.allowance_total, doc.charge_total,
                       doc.total_excl_tax, doc.tax, doc.total, doc.amount_due]
            amounts += [li.unit_price for li in doc.line_items]
            amounts += [li.line_total for li in doc.line_items]
            for v in [a for a in amounts if a is not None]:
                assert amount_in_source(v, doc.source_text), (
                    f"{doc.doc_id} [{layout.value}]: {v} not found in source")


def test_raw_substring_search_would_fail_on_layout_b():
    """Guards the reasoning above. If this ever passes, the layouts have
    drifted into cosmetic variants and Leg 1 has lost its evidence."""
    doc = materialise(CHAINS[0], Layout.B)[4]
    canonical = [f"{v:,.2f}" for v in [doc.total] if v is not None]
    assert any(c not in doc.source_text for c in canonical)


# --- layouts are genuinely divergent ----------------------------------------

def test_layouts_share_no_structural_vocabulary():
    """'Not cosmetic variants — if a regex could translate one to the other,
    they're too similar.'"""
    a = render_lines(Layout.A)
    b = render_lines(Layout.B)
    assert not (a & b), f"layouts share label tokens: {a & b}"


def render_lines(layout: Layout) -> set[str]:
    doc = CHAINS[0].docs[4]
    text = RENDERERS[layout](doc)
    tokens = set()
    for line in text.splitlines():
        for sep in (":", "=", " "):
            if sep in line:
                head = line.split(sep)[0].strip().lower()
                if head.isalpha() and len(head) > 3:
                    tokens.add(head)
                break
    return tokens


def test_layouts_yield_identical_records_apart_from_rendering():
    """Same deal, two layouts, one verdict — the MAIN evidence for Leg 1.
    If the SEMANTIC content differed here, any downstream difference would be
    the generator's fault rather than the model's.

    source_text and doc_date_raw are excluded because they ARE the rendering:
    one layout prints '02 May 2026' and the other '2026-05-02' for the same
    doc_date. Everything that carries meaning must be byte-identical.
    """
    rendering = {"source_text", "doc_date_raw"}
    a = materialise(CHAINS[3], Layout.A)
    b = materialise(CHAINS[3], Layout.B)
    for da, db in zip(a, b):
        assert (da.model_dump(exclude=rendering)
                == db.model_dump(exclude=rendering))
        assert da.doc_date == db.doc_date        # meaning is identical
        assert da.doc_date_raw != db.doc_date_raw  # rendering is not


def test_layouts_state_different_precision():
    """The point of the B renderer. Same value, different committed precision,
    therefore different inferred tolerance."""
    a = "\n".join(d.source_text for d in materialise(CHAINS[0], Layout.A))
    b = "\n".join(d.source_text for d in materialise(CHAINS[0], Layout.B))
    # Value equality across layouts is proven by the two tests above; here we
    # only assert the RENDERING differs. (A whole-document find_amounts()
    # comparison would also pick up dates and part numbers, which is exactly
    # the false-positive surface noted in the A5 write-up.)
    assert a.count(".00") > 0, "Layout A should state cent precision"
    assert b.count(".00") == 0, "Layout B should strip trailing zeros"


# --- the answer key ----------------------------------------------------------

def test_generator_emits_clean_chains_only():
    """A1 is clean chains. Anomaly injection is A3, layered on top. A
    generator that breaks documents by accident is unusable as ground truth."""
    for c in CHAINS:
        assert c.key.anomalies == []
        assert c.key.generator_seed == SEED
        assert set(c.key.doc_ids) == {d.doc_id for d in c.docs}


def test_doc_date_raw_matches_what_the_layout_prints():
    """The page shows a date, so ground truth must say so. Otherwise the date
    check is dead on this corpus and extraction is penalised for reading it."""
    from gen import RAW_DATE
    for layout in (Layout.A, Layout.B):
        for doc in materialise(CHAINS[0], layout):
            assert doc.doc_date_raw == RAW_DATE[layout](doc.doc_date)
            assert doc.doc_date_raw in doc.source_text


def test_generated_dates_are_unambiguous_in_both_layouts():
    """'02 May 2026' and '2026-05-02' both have exactly one reading, so the
    corpus exercises the date check without manufacturing ambiguity noise."""
    from normalise import read_date
    for layout in (Layout.A, Layout.B):
        for doc in materialise(CHAINS[0], layout):
            r = read_date(doc.doc_date_raw)
            assert r.unique == doc.doc_date and not r.ambiguous
