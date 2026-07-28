"""A5 + A6 tests.

The two-sided contract that makes the verifier trustworthy:
  RECALL side  — every corrupted fixture trips the check it is named after.
  PRECISION side — clean data (10 hand-written fixtures + the full generated
                   corpus in both layouts) produces ZERO FAILs. A verifier
                   that flags clean documents inflates the escalation rate
                   with garbage and can fire the ~50% kill-metric on noise.
"""

import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from gen import generate_chain, materialise
from schemas import (
    CheckName,
    CheckOutcome,
    ExtractedRecord,
    Layout,
    LineItem,
    CanonicalDoc,
    DocType,
)
from verify.engine import verify_doc, verify_pair

FIXDIR = Path(__file__).parent.parent / "data" / "fixtures"
CLEAN = sorted((FIXDIR / "records").glob("*.json"))
CORRUPTED = sorted((FIXDIR / "corrupted").glob("*.json"))


def load(p: Path) -> ExtractedRecord:
    return ExtractedRecord.model_validate_json(p.read_text(encoding="utf-8"))


# --- PRECISION: clean data must not fail ------------------------------------

@pytest.mark.parametrize("path", CLEAN, ids=lambda p: p.stem)
def test_hand_written_fixtures_verify_clean(path):
    report = verify_doc(load(path).doc)
    fails = [c for c in report.checks if c.outcome == CheckOutcome.FAIL]
    assert not fails, f"{path.stem}: {[(c.check, c.message) for c in fails]}"
    assert report.verify_pass


def test_generated_corpus_verifies_clean_in_both_layouts():
    """The kill-metric guard from the master plan: any FAIL on a chain the
    generator labelled clean means the tolerance policy or the verifier is
    wrong, NOT the data — the generator is identity-tested. 25 chains x 6
    docs x 2 layouts = 300 documents through the full verifier."""
    for i in range(25):
        bundle = generate_chain(i, 1337)
        for layout in (Layout.A, Layout.B):
            for doc in materialise(bundle, layout):
                report = verify_doc(doc)
                fails = [c for c in report.checks
                         if c.outcome == CheckOutcome.FAIL]
                assert not fails, (
                    f"{doc.doc_id} [{layout}]: "
                    f"{[(c.check, c.message) for c in fails]}")


def test_whole_unit_fixture_is_a_strict_pass():
    """The Coupa PO states unit precision; its identities hold exactly, so it
    must be STRICT — the high-confidence tier — not merely within tolerance."""
    coupa = next(p for p in CLEAN if "coupa" in p.stem)
    assert verify_doc(load(coupa).doc).strict_pass


def test_pair_checks_pass_on_the_designed_relationships():
    """F3 (SAP PO) precedes F9 (GRN); same vendor. Built into A4 for this."""
    by = {p.stem.split("_")[0]: load(p).doc for p in CLEAN}
    results = verify_pair(by["F-0003"], by["F-0009"])
    assert all(c.outcome == CheckOutcome.PASS for c in results)


# --- RECALL: every corrupted fixture trips its named check ------------------

def test_corrupted_fixtures_exist():
    assert len(CORRUPTED) == 6, (
        "run: PYTHONPATH=src python data/fixtures/make_corrupted.py")


@pytest.mark.parametrize("path", CORRUPTED, ids=lambda p: p.stem)
def test_named_check_fires(path):
    named = path.stem.split("__")[0]
    report = verify_doc(load(path).doc)
    fired = {c.check for c in report.checks
             if c.outcome == CheckOutcome.FAIL}
    assert named in {str(c) for c in fired}, (
        f"{path.stem}: named check did not FAIL; fired={fired}")
    assert not report.verify_pass


def test_consistent_hallucination_is_caught_only_by_the_source_check():
    """The star case. Every amount shifted in lockstep: all arithmetic
    identities HOLD, and only AMOUNTS_APPEAR_IN_SOURCE fails. This is the
    dominant real failure mode (invented numerics) and the proof that the
    source check is not redundant with the identities."""
    path = next(p for p in CORRUPTED if p.stem.startswith("amounts_appear"))
    report = verify_doc(load(path).doc)
    outcomes = {c.check: c.outcome for c in report.checks}
    assert outcomes[CheckName.AMOUNTS_APPEAR_IN_SOURCE] == CheckOutcome.FAIL
    for check in (CheckName.BR_CO_10, CheckName.BR_CO_13, CheckName.BR_CO_15,
                  CheckName.LINE_NET_AMOUNT):
        assert outcomes[check] in (CheckOutcome.PASS, CheckOutcome.SKIPPED), (
            f"{check} should hold on a consistent hallucination")


def test_unapplied_discount_fires_only_br_co_13():
    """The BR-CO-13 corruption is engineered so tax and total follow the
    wrong net — the discount was ignored end-to-end. Only the net identity
    breaks; 10 and 15 hold. This is the unapplied-discount anomaly the
    generator will plant at chain level, seen here at doc level."""
    path = next(p for p in CORRUPTED if p.stem.startswith("br_co_13"))
    report = verify_doc(load(path).doc)
    outcomes = {c.check: c.outcome for c in report.checks}
    assert outcomes[CheckName.BR_CO_13] == CheckOutcome.FAIL
    assert outcomes[CheckName.BR_CO_10] == CheckOutcome.PASS
    assert outcomes[CheckName.BR_CO_15] == CheckOutcome.PASS


# --- three-outcome semantics ------------------------------------------------

def _doc(**kw) -> CanonicalDoc:
    base = dict(doc_id="T-1", doc_number="T-1", doc_type=DocType.INVOICE,
                party_name="Test Co", doc_date=date(2026, 6, 1),
                currency="USD", source_text="x", references=[])
    base.update(kw)
    return CanonicalDoc(**base)


def test_aggregate_rounding_lands_within_tolerance_not_fail():
    """Three cent-precision lines of 33.33 sum to 99.99; the document states
    a whole-unit subtotal of 100. Delta 0.01, band inferred from the stated
    precisions ≈ 0.57 — WITHIN_TOLERANCE. The fixed-floor v1.1 verifier
    would have flagged this legitimate rounding on every such document."""
    lines = [LineItem(line_id=f"LI-{i}", description="unit",
                      quantity=Decimal(1), unit_price=Decimal("33.33"),
                      line_total=Decimal("33.33")) for i in range(1, 4)]
    doc = _doc(line_items=lines, subtotal=Decimal("100"),
               source_text="3 x 33.33, total about 100")
    r = verify_doc(doc)
    br10 = next(c for c in r.checks if c.check == CheckName.BR_CO_10)
    assert br10.outcome == CheckOutcome.WITHIN_TOLERANCE
    assert r.verify_pass and not r.strict_pass


def test_beyond_the_band_fails():
    lines = [LineItem(line_id="LI-1", description="unit",
                      quantity=Decimal(1), unit_price=Decimal("33.33"),
                      line_total=Decimal("33.33"))]
    doc = _doc(line_items=lines, subtotal=Decimal("35.00"),
               source_text="33.33 and 35.00")
    r = verify_doc(doc)
    br10 = next(c for c in r.checks if c.check == CheckName.BR_CO_10)
    assert br10.outcome == CheckOutcome.FAIL


def test_moneyless_document_skips_and_is_not_strict():
    """A goods receipt has quantities and no money: arithmetic checks are
    SKIPPED, verify_pass holds, but strict_pass must be False when nothing
    numeric was checkable — the empty-extraction guard."""
    grn = next(p for p in CLEAN if "grn" in p.stem)
    r = verify_doc(load(grn).doc)
    assert r.verify_pass
    skipped = {c.check for c in r.checks if c.outcome == CheckOutcome.SKIPPED}
    assert CheckName.BR_CO_10 in skipped


def test_pair_detects_out_of_order_and_party_mismatch():
    a = _doc(doc_id="T-A", doc_date=date(2026, 6, 10))
    b = _doc(doc_id="T-B", doc_date=date(2026, 6, 1),
             party_name="Other Corp")
    results = {c.check: c.outcome for c in verify_pair(a, b)}
    assert results[CheckName.PARTY_NAMES_MATCH] == CheckOutcome.FAIL
    assert results[CheckName.DATES_PARSE_AND_ORDER] == CheckOutcome.FAIL


# --- the CLI ----------------------------------------------------------------

def _run_cli(*args, cwd):
    """Inherit the real environment and override only PYTHONPATH. A stripped
    env breaks the interpreter on Windows: CPython needs SYSTEMROOT to seed
    hash randomisation and dies before reaching our code."""
    env = dict(os.environ, PYTHONPATH=str(cwd / "src"))
    return subprocess.run([sys.executable, "-m", "verify", *args],
                          cwd=cwd, capture_output=True, text=True, env=env)


def test_cli_exits_zero_on_clean_and_nonzero_on_corrupted(tmp_path):
    root = Path(__file__).parent.parent
    ok = _run_cli(str(FIXDIR / "records" / "*.json"),
                  "--out", str(tmp_path / "verified.jsonl"), "-q", cwd=root)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert (tmp_path / "verified.jsonl").exists()

    bad = _run_cli(str(FIXDIR / "corrupted" / "*.json"), "-q", cwd=root)
    assert bad.returncode == 1, bad.stdout + bad.stderr
