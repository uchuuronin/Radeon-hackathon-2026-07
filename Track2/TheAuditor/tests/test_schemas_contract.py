"""Contract tests for schemas.py — run with `pytest`, no GPU, no network.

These are the automated version of Sync 1, question 3 ("does B's extraction
JSONL deserialise into A's CanonicalDoc without errors?"). They run on every
push instead of once at the sync.
"""

import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from schemas import (
    CROSS_DOC_TOLERANCE,
    DEFAULT_TOLERANCES,
    AnomalyType,
    AnswerKey,
    CanonicalDoc,
    ChainKey,
    CheckName,
    CheckOutcome,
    CheckResult,
    DocType,
    ExtractedRecord,
    ExtractionMeta,
    LineItem,
    PlantedAnomaly,
    ScoringMode,
    Tier,
    Tolerance,
    VerificationReport,
    allowed_delta,
)


def make_fixture_doc() -> CanonicalDoc:
    """A discounted invoice with freight — the case the naive identity gets wrong.

    lines            4500.00 + 300.00 = 4800.00      (subtotal, BT-106)
    allowance                          -  480.00     (BT-107, 10% discount)
    charge (freight)                   +   75.00     (BT-108)
    total excl tax                     = 4395.00     (BT-109)
    tax @10%                           +  439.50     (BT-110)
    total                              = 4834.50     (BT-112)
    """
    return CanonicalDoc(
        doc_id="D-000123",
        doc_number="INV-4021",
        doc_type=DocType.INVOICE,
        party_name="Northwind Traders",
        doc_date=date(2026, 7, 20),
        doc_date_raw="20 Jul 2026",
        currency="USD",
        line_items=[
            LineItem(line_id="LI-001", description="Widget, blue",
                     quantity=Decimal("3"), unit_of_measure="EA",
                     unit_price=Decimal("1500.00"),
                     line_total=Decimal("4500.00")),
            LineItem(line_id="LI-002", description="Install labour",
                     quantity=Decimal("1.5"), unit_of_measure="HUR",
                     unit_price=Decimal("200.00"),
                     line_total=Decimal("300.00")),
        ],
        subtotal=Decimal("4800.00"),
        allowance_total=Decimal("480.00"),
        charge_total=Decimal("75.00"),
        total_excl_tax=Decimal("4395.00"),
        tax=Decimal("439.50"),
        total=Decimal("4834.50"),
        amount_due=Decimal("4834.50"),
        payment_terms="Net 30",
        references=["PO#4021-A"],
        source_text="INVOICE ... 4,500.00 ... 300.00 ... total 4,834.50 ...",
    )


# --- round-trip / wire format ------------------------------------------------

def test_round_trip_preserves_equality():
    rec = ExtractedRecord(doc=make_fixture_doc(),
                          meta=ExtractionMeta(tier=Tier.NONE))
    back = ExtractedRecord.model_validate_json(rec.model_dump_json())
    assert back == rec


def test_money_survives_as_decimal_not_float():
    rec = ExtractedRecord(doc=make_fixture_doc(), meta=ExtractionMeta())
    wire = json.loads(rec.model_dump_json())
    assert isinstance(wire["doc"]["subtotal"], str)
    assert isinstance(wire["doc"]["line_items"][0]["unit_price"], str)
    back = ExtractedRecord.model_validate_json(rec.model_dump_json())
    assert back.doc.subtotal == Decimal("4800.00")
    assert isinstance(back.doc.subtotal, Decimal)


def test_floats_are_rejected():
    doc = make_fixture_doc().model_dump()
    doc["subtotal"] = 4800.0
    with pytest.raises(ValidationError):
        CanonicalDoc.model_validate(doc)


def test_extra_fields_are_rejected():
    doc = json.loads(make_fixture_doc().model_dump_json())
    doc["grand_total"] = "4834.50"
    with pytest.raises(ValidationError):
        CanonicalDoc.model_validate(doc)


def test_json_schema_exports_for_guided_decoding():
    """B feeds this to XGrammar guided_json. Must exist and be strict."""
    schema = CanonicalDoc.model_json_schema()
    assert schema.get("additionalProperties") is False
    assert "source_text" in schema["properties"]
    assert "allowance_total" in schema["properties"]


# --- EN 16931 identities: the naive-identity bug ----------------------------

def test_naive_subtotal_plus_tax_identity_is_wrong_on_discounted_doc():
    """Guards the naive identity. A discounted+freighted document is LEGITIMATE but
    fails subtotal + tax == total. If this assertion ever flips, someone has
    reverted to the naive identity and the verifier will false-positive on
    every discounted invoice."""
    d = make_fixture_doc()
    assert d.subtotal + d.tax != d.total


def test_br_co_13_and_15_hold_on_the_same_doc():
    d = make_fixture_doc()
    # BR-CO-13: BT-109 = BT-106 - BT-107 + BT-108
    assert d.total_excl_tax == d.subtotal - d.allowance_total + d.charge_total
    # BR-CO-15: BT-112 = BT-109 + BT-110
    assert d.total == d.total_excl_tax + d.tax


def test_br_co_10_line_totals_sum_to_subtotal():
    d = make_fixture_doc()
    assert sum(li.line_total for li in d.line_items) == d.subtotal


# --- tolerance model ---------------------------------------------------------

def test_absolute_cap_beats_percentage_on_large_amounts():
    tol = Tolerance(pct=Decimal("0.02"), abs_cap=Decimal("100.00"))
    assert allowed_delta(Decimal("1000000.00"), tol) == Decimal("100.00")


def test_percentage_applies_below_the_cap():
    assert allowed_delta(Decimal("50.00"), CROSS_DOC_TOLERANCE) == Decimal("1.00")


def test_floor_is_below_one_cent_so_one_cent_is_detectable():
    """This test previously asserted the opposite, and was wrong.

    It read "floor prevents flagging a cent" and locked the floor at 0.01,
    which made a one-cent discrepancy undetectable on every within-document
    identity: the comparison is inclusive, so delta 0.01 <= band 0.01 passed.
    One cent is the smallest meaningful financial error and the floor sat
    exactly where it blinded us.

    Nothing in this suite could have caught that, because the suite and the
    verifier were written from the same assumption. It surfaced when CEN's own
    BR-CO-10 fixture (subtotal 200.01 against lines of 110.00 + 90.00, declared
    an error by the standard's maintainers) came back green from our verifier.

    0.005 is a half-ULP at cent precision, so the floor now agrees with the
    inference model instead of being a round number.
    """
    tol = DEFAULT_TOLERANCES[CheckName.BR_CO_10]
    assert allowed_delta(Decimal("0.10"), tol) == Decimal("0.005")
    assert Decimal("0.01") > allowed_delta(Decimal("0.10"), tol)


def test_uncapped_tolerance_is_allowed():
    tol = Tolerance(pct=Decimal("0.05"), abs_cap=None)
    assert allowed_delta(Decimal("10000.00"), tol) == Decimal("500.00")


# --- verification outcomes ---------------------------------------------------

def _cr(check, outcome, **kw):
    return CheckResult(check=check, outcome=outcome, **kw)


def test_within_tolerance_passes_but_is_not_strict():
    report = VerificationReport(doc_id="D-000123", checks=[
        _cr(CheckName.BR_CO_10, CheckOutcome.PASS),
        _cr(CheckName.BR_CO_15, CheckOutcome.WITHIN_TOLERANCE,
            delta=Decimal("0.02"), tolerance_applied=Decimal("0.05"),
            message="rounding"),
    ])
    assert report.verify_pass is True
    assert report.strict_pass is False


def test_skipped_checks_are_neutral():
    report = VerificationReport(doc_id="D-000900", checks=[
        _cr(CheckName.BR_CO_10, CheckOutcome.PASS),
        _cr(CheckName.BR_CO_15, CheckOutcome.SKIPPED,
            message="no tax field on this doc"),
    ])
    assert report.verify_pass is True
    assert report.strict_pass is True


def test_a_single_fail_sinks_both_signals():
    report = VerificationReport(doc_id="D-000123", checks=[
        _cr(CheckName.BR_CO_10, CheckOutcome.PASS),
        _cr(CheckName.AMOUNTS_APPEAR_IN_SOURCE, CheckOutcome.FAIL,
            expected="4500.00 in source", actual="not found"),
    ])
    assert report.verify_pass is False
    assert report.strict_pass is False


def test_all_skipped_is_not_a_strict_pass():
    """A document where nothing was checkable must not read as high
    confidence — that is exactly the empty-extraction failure mode."""
    report = VerificationReport(doc_id="D-000901", checks=[
        _cr(CheckName.BR_CO_10, CheckOutcome.SKIPPED),
    ])
    assert report.strict_pass is False


def test_rounding_policy_is_recorded():
    report = VerificationReport(doc_id="D-1", checks=[])
    assert report.rounding_policy == "aggregate"


# --- documents with legitimately absent fields -------------------------------

def test_absent_amounts_are_none_not_zero():
    payment = CanonicalDoc(
        doc_id="D-000900",
        doc_number="PAY-4021",
        doc_type=DocType.PAYMENT,
        party_name="Northwind Traders",
        doc_date=date(2026, 7, 25),
        currency="USD",
        line_items=[],                 # legitimately empty for a payment
        amount_due=Decimal("4834.50"),
        references=["INV-77"],
        source_text="Payment received 4,834.50 ref INV-77",
    )
    assert payment.subtotal is None
    assert payment.tax is None
    assert payment.allowance_total is None


def test_the_contract_carries_no_version_of_its_own():
    """The commit is the version.

    A version string beside a git history is a second answer to a question
    that already has one, and it is the answer that rots, because nothing
    fails when you forget to bump it. This test is the guard on that: if a
    schema_version, contract_version or similar reappears, it will be because
    someone added it by reflex rather than because the project grew a second
    consumer that needs it.
    """
    import schemas
    banned = {"SCHEMA_VERSION", "WIRE_COMPATIBLE_WITH", "CONTRACT_VERSION"}
    assert not (banned & set(vars(schemas))), (
        "the contract grew a version constant; the commit is the version")
    for model in (ExtractedRecord, ExtractionMeta, CanonicalDoc, LineItem):
        fields = set(model.model_fields)
        assert not {f for f in fields if "schema_version" in f}, model


def test_check_result_can_locate_the_failure():
    """Beancount's validation errors carry a source location; ours carry a
    field_path. 'This field failed' beats 'this document failed'."""
    r = CheckResult(check=CheckName.LINE_NET_AMOUNT, outcome=CheckOutcome.FAIL,
                    field_path="line_items[LI-002].line_total",
                    delta=Decimal("0.40"))
    assert r.field_path.endswith(".line_total")


def test_goods_receipt_is_distinct_from_despatch_advice():
    """Three-way match uses the GOODS RECEIPT. Collapsing the two loses the
    document that matters."""
    assert DocType.GOODS_RECEIPT != DocType.DESPATCH_ADVICE


# --- answer key --------------------------------------------------------------

def test_answer_key_allows_clean_chains():
    key = AnswerKey(chains=[
        ChainKey(chain_id="C-1", doc_ids=["D-1", "D-2"],
                 anomalies=[], generator_seed=42),
    ])
    back = AnswerKey.model_validate_json(key.model_dump_json())
    assert back.chains[0].anomalies == []


def test_answer_key_carries_below_tolerance_decoys():
    """A drift planted below tolerance must be recorded as such — the system
    is scored on NOT flagging it."""
    key = AnswerKey(chains=[ChainKey(
        chain_id="C-2", doc_ids=["D-3", "D-4"], generator_seed=42,
        anomalies=[
            PlantedAnomaly(anomaly_type=AnomalyType.PRICE_DRIFT,
                           doc_ids_involved=["D-3", "D-4"],
                           field_path="total",
                           expected_delta=Decimal("-1500.00")),
            PlantedAnomaly(anomaly_type=AnomalyType.PRICE_DRIFT,
                           doc_ids_involved=["D-3", "D-4"],
                           field_path="line_items[LI-002].unit_price",
                           expected_delta=Decimal("-0.40"),
                           is_within_tolerance=True,
                           note="decoy: below the 2%/$100 band"),
        ])])
    back = AnswerKey.model_validate_json(key.model_dump_json())
    decoys = [a for a in back.chains[0].anomalies if a.is_within_tolerance]
    assert len(decoys) == 1
    assert decoys[0].expected_delta == Decimal("-0.40")


def test_field_paths_use_line_ids_not_indices():
    """Positional paths break when a layout reorders rows."""
    a = PlantedAnomaly(anomaly_type=AnomalyType.QUANTITY_MISMATCH,
                       doc_ids_involved=["D-3"],
                       field_path="line_items[LI-002].quantity")
    assert "LI-" in a.field_path


def test_partial_shipment_anomaly_exists():
    assert AnomalyType.PARTIAL_SHIPMENT.value == "partial_shipment"


# --- misc --------------------------------------------------------------------

def test_provenance_travels_without_a_schema_version():
    """meta still has to say WHERE a record came from.

    Dropping the schema version does not mean dropping provenance. model_id,
    tier, n_sample_index and layout are what cost accounting and agreement
    scoring are computed from, and they describe the RUN, which genuinely
    varies. The schema did not.
    """
    rec = ExtractedRecord(doc=make_fixture_doc(), meta=ExtractionMeta())
    assert set(ExtractionMeta.model_fields) >= {
        "model_id", "tier", "n_sample_index", "layout"}
    assert rec.model_validate_json(rec.model_dump_json()).doc.doc_id == rec.doc.doc_id


def test_scoring_modes_are_declarable():
    assert {m.value for m in ScoringMode} == {"exact", "relaxed"}
