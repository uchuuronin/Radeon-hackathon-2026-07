"""TheAuditor — the frozen wire contract between the deterministic
pipeline and the extraction tier.

Track 2, AMD AI DevMaster Hackathon 2026.

RULES OF THIS FILE
------------------
1. This is the ONLY shared interface in the project. Everything imports it;
   it imports nothing from the project.
2. THE COMMIT IS THE VERSION. There is no version string to bump and no
   compatibility list to maintain: one branch, one consumer of
   this file. A number carried alongside the git history is a second answer
   to a question that already has one, and it is the answer that goes stale,
   because nothing fails when you forget to bump it. If you need to know what
   the contract said at some point, read the commit.
3. Changes are not unilateral. The distinction that matters is COST,
   not numbering:
     WIRE-BREAKING — any change to CanonicalDoc / LineItem / the answer-key
       types. These alter the JSON Schema in the extraction prompt, so the
       prefix cache is invalidated and affected benchmarks must be re-run.
       Say so in the commit message. It is the only warning anyone gets.
     POLICY-ONLY — tolerance bands, helper functions, verifier-internal
       shapes. No wire impact; nothing needs re-running.
4. Wire format is JSON (JSONL for batches). Money and dates travel as
   STRINGS on the wire and are parsed to Decimal / date on load.
   JSON numbers are FORBIDDEN for money — floats hallucinate cents.
5. `model_json_schema()` on these models is the single source of truth for
   guided decoding (XGrammar guided_json). Do not hand-write a second
   JSON Schema anywhere.

STANDARDS ALIGNMENT
-------------------
Field semantics follow EN 16931 (CEN/TC 434), the European semantic data
model for e-invoicing. We borrow the *meaning and the arithmetic invariants*,
not the XML syntax — we are not claiming EN 16931 conformance. BT-xxx
references in comments map our fields to EN 16931 business terms so the
verifier's checks can cite named rules (BR-CO-10 / 13 / 15) in the spec.

Document types follow OASIS UBL 2.1 naming, which covers the whole
quote-to-cash chain rather than the invoice alone.

Field NAMES are deliberately plain English rather than standards jargon
(`line_total`, not `line_net_amount`) because these names appear verbatim in
the extraction prompt, and natural wording extracts better. The comment
carries the standards mapping; the field name carries the readability.

"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from contextvars import ContextVar
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from precision import inferred_tolerance

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocType(str, Enum):
    """Chain document types, named after their UBL 2.1 equivalents.

    NOTE the despatch/receipt split: UBL separates DespatchAdvice (the SELLER
    says it shipped) from ReceiptAdvice (the BUYER says it arrived). Three-way
    match uses the GOODS RECEIPT, not the despatch note. Collapsing them into
    one 'delivery' type loses the document that actually matters.
    """
    QUOTE = "quote"                        # UBL Quotation
    SALES_ORDER = "sales_order"            # UBL Order (seller side)
    PURCHASE_ORDER = "purchase_order"      # UBL Order (buyer side)
    DESPATCH_ADVICE = "despatch_advice"    # UBL DespatchAdvice — seller shipped
    GOODS_RECEIPT = "goods_receipt"        # UBL ReceiptAdvice / GRN — buyer got it
    INVOICE = "invoice"                    # UBL Invoice
    PAYMENT = "payment"                    # UBL RemittanceAdvice


class AnomalyType(str, Enum):
    """The planted anomaly types. The answer key uses these verbatim; the
    scorer matches on them; do not add types without a version bump."""
    PRICE_DRIFT = "price_drift"
    QUANTITY_MISMATCH = "quantity_mismatch"
    NEAR_DUPLICATE = "near_duplicate"          # one digit changed
    UNAPPLIED_DISCOUNT = "unapplied_discount"  # allowance on PO, absent on invoice
    TERM_CHANGE = "term_change"                # non-numeric
    PARTIAL_SHIPMENT = "partial_shipment"      # invoice > goods receipt.
    # PARTIAL_SHIPMENT is the canonical hard case in real AP (ship 480 of 500,
    # invoice all 500). Worth more than a sixth synthetic variant.


class Layout(str, Enum):
    """Generator serialisation layouts. C is sealed until demo day."""
    A = "layout_a"
    B = "layout_b"
    C = "layout_c"  # holdout — data/fixtures/holdout/, do not open


class Tier(str, Enum):
    """Which inference tier produced an extraction."""
    FAST = "fast"        # quantised — triage/routing only
    PRECISE = "precise"  # numeric extraction + reconciliation reasoning
    NONE = "none"        # hand-written fixture / deterministic path


class ScoringMode(str, Enum):
    """How a numeric/field accuracy figure was computed. Every accuracy
    number in bench/ MUST declare one — exact-match and relaxed-match
    accuracy are different numbers and a judge will ask which we report.
    (Convention borrowed from the DocILE benchmark's evaluation protocol.)"""
    EXACT = "exact"        # string-identical after canonical normalisation
    RELAXED = "relaxed"    # numeric equality within the field's tolerance


# ---------------------------------------------------------------------------
# Tolerance model
# ---------------------------------------------------------------------------
# Real AP systems do not use exact equality. Standard industry practice is a
# PERCENTAGE combined with an ABSOLUTE CAP (e.g. 2% up to $100), because a
# bare percentage lets a large invoice hide a large absolute variance. We also
# carry an absolute FLOOR so sub-cent rounding never generates an exception.
#
# Legitimate small variances come from rounding, unit-of-measure conversion,
# freight estimates and tax calculation — a verifier that flags all of them
# inflates the escalation rate and would trip our own ~50% kill-metric on
# clean data.

class _Base(BaseModel):
    """Common config: no invented fields, enums serialise as values.
    pydantic v2 already emits Decimal as string and date/datetime as ISO-8601
    in JSON mode, so no custom encoders are needed."""
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
    )


class Tolerance(_Base):
    pct: Decimal = Field(default=Decimal("0.02"),
                         description="Fractional, e.g. 0.02 = 2%.")
    abs_cap: Optional[Decimal] = Field(
        default=Decimal("100.00"),
        description="Maximum absolute variance allowed regardless of pct. "
                    "None = uncapped. Stops a 2% pass on a $1M line.")
    abs_floor: Decimal = Field(
        default=Decimal("0.01"),
        description="Minimum allowance when nothing better can be inferred. "
                    "This is the FALLBACK, not the primary rule — see "
                    "precision.py for why a fixed floor is wrong in both "
                    "directions.")
    inferred_multiplier: Decimal = Field(
        default=Decimal("1.1"),
        description="Buffer applied to precision-inferred tolerance. Mirrors "
                    "Beancount's inferred_tolerance_multiplier; 1.1 is its "
                    "documented recommendation.")
    abs_ceiling: Optional[Decimal] = Field(
        default=None,
        description="Upper bound on the PRECISION-INFERRED widening. None = "
                    "uncapped, which is correct wherever the operands carry "
                    "genuinely unknown precision. Set it on the document-total "
                    "identities, where EN 16931 defines a fixed rounding "
                    "allowance and inference would otherwise be looser than "
                    "the standard. Does not bound `pct` or `abs_floor`: those "
                    "answer different questions.")
    mode: Literal["rss", "linear"] = Field(
        default="rss",
        description="How half-ULPs accumulate across the operands of an "
                    "identity. 'rss' is the independent-error bound "
                    "(sqrt of sum of squares, grows as sqrt(N)); 'linear' is "
                    "the worst-case aligned-error ceiling (grows as N) and is "
                    "twice as wide at four operands. Default rss: linear "
                    "would absorb a genuine one-cent-per-line error across "
                    "four lines. Every tolerance figure reported must name "
                    "the mode that produced it.")


#: When True, `allowed_delta` ignores stated precision and allows only the
#: absolute floor. See the note on allowed_delta.
STRICT_PRECISION = ContextVar("STRICT_PRECISION", default=False)


def allowed_delta(
    reference: Decimal,
    tol: Tolerance,
    rendered: Sequence[str] | None = None,
) -> Decimal:
    """Single implementation of the tolerance rule. BOTH the verifier and the
    reconciler call this — two implementations would drift, and the drift
    would surface as an unreproducible precision/recall number.

    `rendered` is the list of amount strings AS THE DOCUMENT PRINTED THEM,
    for every operand in the identity being checked. When supplied, the
    floor becomes the precision the document actually committed to rather
    than a fixed constant. Omit it and behaviour falls back to the fixed floor,
    so existing callers are unaffected.
    """
    band = abs(reference) * tol.pct
    if tol.abs_cap is not None:
        band = min(band, tol.abs_cap)

    # TWO DEFENSIBLE READINGS OF A STATED AMOUNT, AND THEY DISAGREE
    # ------------------------------------------------------------
    # Default (inference): "250" is a value someone rounded, so it stands for
    #   anything in [249.5, 250.5) and the band widens accordingly. Correct for
    #   documents of unknown provenance, which is what an extraction pipeline
    #   handles: the precision a document prints is the only evidence we have
    #   about the precision it kept.
    # Strict: "250" is exactly 250. Correct for a conformant e-invoice, where
    #   the standard already constrains amounts to two decimals and treats what
    #   is printed as the value.
    #
    # This is not a hedge. Running CEN's own rule fixtures against us showed the
    # two readings disagree on real cases: their BR-CO-15 fixture states tax as
    # "250" and a gross total one cent off, and calls it an error, while
    # inference gives a band of 0.55 because "250" could have been 249.5. Both
    # verdicts are right for their own context, so the context is now selectable
    # rather than assumed, and bench/conformance.py reports both.
    if STRICT_PRECISION.get():
        return max(band, tol.abs_floor)

    floor = tol.abs_floor
    if rendered:
        inferred = inferred_tolerance(rendered, tol.inferred_multiplier,
                                      mode=tol.mode)
        # WHY THE INFERRED WIDENING IS BOUNDED ON THE TOTAL IDENTITIES
        # -----------------------------------------------------------
        # Inference reads a document's printed precision as evidence of the
        # precision it kept, which is right when provenance is unknown. But it
        # accumulates: three operands printed without decimals give a band of
        # 0.78, and a one-cent discrepancy disappears inside it. Cent-level
        # discrepancy is not a rounding artefact to a reconciliation engine, it
        # is the thing the engine exists to find.
        #
        # Measured against CEN's own labelled rule fixtures, unbounded
        # inference agreed on 43 of 48 (89.6%). Every one of the five misses
        # was a one-cent break that CEN requires to fail, on operands it prints
        # as bare integers. EN 16931 answers the same question with a fixed
        # +/-0.01 rounding allowance regardless of how an operand is written,
        # so the ceiling is not a fudge factor: it is the standard's own
        # number, applied where the standard applies it.
        #
        # Deliberately NOT global. It is set per check, on the document-total
        # identities only. A line total is qty x unit price, and a unit price
        # printed to fewer decimals than it was kept to makes a genuinely wider
        # product; capping that would manufacture failures rather than find
        # them. Uncapped inference stays the default everywhere else.
        if tol.abs_ceiling is not None:
            inferred = min(inferred, tol.abs_ceiling)
        floor = max(floor, inferred)
    return max(band, floor)


# ---------------------------------------------------------------------------
# Core extracted record
# ---------------------------------------------------------------------------

class LineItem(_Base):
    line_id: str = Field(
        description="Stable identity for this line, assigned by the generator "
                    "and preserved across ALL layouts. Positional indices "
                    "break the moment a layout reorders rows, which would "
                    "silently corrupt answer-key field_paths.")
    description: str
    quantity: Decimal                     # BT-129
    unit_of_measure: Optional[str] = Field(
        default=None,
        description="EA / KG / HUR etc. (UN/ECE Rec 20 codes or free text). "
                    "UoM conversion is a real cause of quantity variance — "
                    "without this field a legitimate variance is unexplainable.")
    unit_price: Optional[Decimal] = Field(
        default=None,
        description="BT-146. None on documents that record quantities but no "
                    "money — a goods receipt or despatch advice. NEVER 0 for "
                    "absent; a zero price is a hallucinated number.")
    line_allowance: Optional[Decimal] = Field(
        default=None,
        description="Line-level discount. BT-136.")
    line_total: Optional[Decimal] = Field(
        default=None,
        description="Line net amount (BT-131) = quantity x unit_price "
                    "- line_allowance. None on quantity-only documents.")

    @field_validator("quantity", "unit_price", "line_allowance", "line_total",
                     mode="before")
    @classmethod
    def _no_floats(cls, v):
        if isinstance(v, float):
            raise ValueError("floats are forbidden for money/quantity; "
                             "send strings on the wire")
        return v


class CanonicalDoc(_Base):
    """The extracted record. The ONLY thing the verifier, linker, and
    reconciler ever see. If a field is not here, downstream cannot reason
    about it."""

    doc_id: str = Field(description="Generator-assigned, globally unique, "
                                    "semantically opaque. Encodes NOTHING — "
                                    "no doc_type, no chain membership.")
    doc_number: str = Field(
        description="The identifier PRINTED ON THE DOCUMENT, e.g. 'PO-4021'. "
                    "This is what other documents cite in `references`, and "
                    "it is what the linker matches on. It is NOT doc_id: "
                    "doc_id is opaque ground truth so the linker cannot "
                    "cheat, doc_number is visible evidence it must reason "
                    "from. Both are required and they are different things.")
    doc_type: DocType
    party_name: str
    doc_date: date = Field(description="Normalised ISO-8601.")
    doc_date_raw: Optional[str] = Field(
        default=None,
        description="Date string exactly as it appeared in the source, "
                    "for the audit trail. None for hand-written fixtures.")
    currency: str = Field(description="ISO 4217 code, e.g. 'USD'. "
                                      "v1 assumes single currency per chain — "
                                      "known scope cut, not an accident.")
    line_items: list[LineItem] = Field(
        default_factory=list,
        description="May legitimately be empty (payments, goods receipts).")

    # --- Monetary totals, EN 16931 semantics -------------------------------
    # All Optional: None means ABSENT FROM THE DOCUMENT. Never 0 for
    # 'not found' — an unextracted zero is a hallucinated number.
    subtotal: Optional[Decimal] = Field(
        default=None,
        description="Sum of line net amounts (BT-106).")
    allowance_total: Optional[Decimal] = Field(
        default=None,
        description="Document-level allowances / discounts (BT-107). "
                    "REQUIRED for BR-CO-13 — without it the totals identity "
                    "is wrong on every discounted document.")
    charge_total: Optional[Decimal] = Field(
        default=None,
        description="Document-level charges, e.g. freight billed separately "
                    "(BT-108). Same reason as allowance_total.")
    total_excl_tax: Optional[Decimal] = Field(
        default=None,
        description="Total amount without VAT (BT-109).")
    tax: Optional[Decimal] = Field(
        default=None, description="Total VAT amount (BT-110).")
    total: Optional[Decimal] = Field(
        default=None, description="Total amount with VAT (BT-112).")
    paid_amount: Optional[Decimal] = Field(
        default=None,
        description="BT-113. Already-settled amount on this invoice. Absent "
                    "means nothing has been paid, which is NOT the same as a "
                    "stated zero: a document that prints 'Paid: 0.00' is "
                    "asserting it, and one that says nothing is not.")
    rounding_amount: Optional[Decimal] = Field(
        default=None,
        description="Explicit rounding adjustment (BT-114). Some vendors "
                    "state it; capturing it prevents a false arithmetic fail.")
    amount_due: Optional[Decimal] = Field(
        default=None,
        description="Amount due for payment (BT-115). The field a PAYMENT "
                    "document actually reconciles against.")

    references: list[str] = Field(
        default_factory=list,
        description="Doc numbers this document points at, RAW as found "
                    "(e.g. 'PO#4021-A'). Extraction never interprets; "
                    "normalisation is the linker's job.")
    payment_terms: Optional[str] = Field(
        default=None,
        description="Raw terms string, e.g. 'Net 30, 2% 10'. The non-numeric "
                    "field the term_change anomaly perturbs.")
    source_text: str = Field(
        description="Raw document text, byte-preserved. No whitespace "
                    "normalisation, no re-encoding. Required by the "
                    "verbatim-amount check.")

    @field_validator("subtotal", "allowance_total", "charge_total",
                     "total_excl_tax", "tax", "total", "rounding_amount",
                     "amount_due", mode="before")
    @classmethod
    def _no_floats(cls, v):
        if isinstance(v, float):
            raise ValueError("floats are forbidden for money; "
                             "send strings on the wire")
        return v


class ExtractionMeta(_Base):
    """Envelope recorded alongside every model-produced CanonicalDoc.
    S2's self-consistency agreement and cost-per-doc accounting need this;
    retrofitting it into a frozen schema on Day 6 is forbidden by the freeze,
    so it exists now, even while mostly unused."""
    model_id: str = ""                # e.g. "Qwen3-14B-GPTQ-int8"
    tier: Tier = Tier.NONE
    prompt_id: str = ""
    # A NAME, not a number. "v3" tells you nothing about what changed and
    # invites the same rot as a schema version; "grounded-anchors-fewshot-4"
    # tells you what you were testing when the sweep row was recorded. B sets
    # it freely and it only has to be unique within a run.
    extraction_ts: Optional[datetime] = None
    n_sample_index: Optional[int] = Field(
        default=None,
        description="Which of the N self-consistency samples this is (0-based). "
                    "None for single-pass extraction.")
    layout: Optional[Layout] = None


class ExtractedRecord(_Base):
    """What the extraction pipeline emits, one per JSONL line:
    the doc plus its provenance."""
    doc: CanonicalDoc
    meta: ExtractionMeta


# ---------------------------------------------------------------------------
# Answer key (ground truth) — the deliverable of anomaly injection
# ---------------------------------------------------------------------------

class PlantedAnomaly(_Base):
    anomaly_type: AnomalyType
    doc_ids_involved: list[str] = Field(min_length=1)
    field_path: str = Field(
        description="Which field drifted. Dotted path into CanonicalDoc using "
                    "LINE IDs, not indices: 'total' or "
                    "'line_items[LI-003].unit_price'. Enables field-level "
                    "precision/recall, not just chain-level.")
    expected_delta: Optional[Decimal] = Field(
        default=None,
        description="Signed magnitude of the drift where numeric. "
                    "None for non-numeric anomalies (term_change).")
    is_within_tolerance: bool = Field(
        default=False,
        description="TRUE for drifts planted deliberately BELOW the "
                    "tolerance band. The system must NOT flag these. This is "
                    "how we measure precision honestly — a system that flags "
                    "everything scores perfect recall and useless precision.")
    note: str = ""                    # human-readable, for debugging

    @field_validator("expected_delta", mode="before")
    @classmethod
    def _no_floats(cls, v):
        if isinstance(v, float):
            raise ValueError("floats are forbidden; send strings on the wire")
        return v


class ChainKey(_Base):
    """Ground truth for one deal chain. CLEAN CHAINS APPEAR TOO, with an
    empty anomalies list — false positives on clean chains are half of
    precision."""
    chain_id: str
    doc_ids: list[str] = Field(min_length=1)
    anomalies: list[PlantedAnomaly] = Field(default_factory=list)
    generator_seed: int = Field(
        description="Seed that produced this chain. Reproducibility: "
                    "benchmarks on Day 2 and Day 13 must be comparable, and "
                    "the demo set must survive the platform stress test.")
    layouts_emitted: list[Layout] = Field(default_factory=list)


class AnswerKey(_Base):
    chains: list[ChainKey]


# ---------------------------------------------------------------------------
# Verifier output — Mechanism A's report, the audit trail, and the
# S2 confidence anchor. One shape consumed by routing, console, and audit log.
# ---------------------------------------------------------------------------

class CheckName(str, Enum):
    """Named invariants.

    The BR_CO_* checks implement calculation rules from EN 16931 and are
    cited by ID in the spec. The remaining checks are OURS — they have no
    standards backing and must not be presented as if they did.
    """
    # --- EN 16931 calculation rules ---
    BR_CO_10 = "br_co_10_line_totals_sum_to_subtotal"
    # sum(line_items.line_total) == subtotal            (BT-131 -> BT-106)
    BR_CO_13 = "br_co_13_total_excl_tax_identity"
    # total_excl_tax == subtotal - allowance_total + charge_total
    #                                        (BT-109 = BT-106 - 107 + 108)
    BR_CO_15 = "br_co_15_total_incl_tax_identity"
    # total == total_excl_tax + tax                (BT-112 = BT-109 + BT-110)
    LINE_NET_AMOUNT = "line_net_amount_identity"
    # quantity * unit_price - line_allowance == line_total   (BT-131 defn)

    BR_CO_16 = "br_co_16_amount_due_identity"
    # BT-115 = BT-112 - BT-113 + BT-114. The settlement identity, and the one
    # a payment chain actually turns on: it is what says "this invoice was
    # part-paid and this is what remains", which is exactly the state a
    # reconciliation engine exists to reason about. We had amount_due and
    # rounding_amount already; only the paid amount was missing.

    BR_DEC_MAX_2 = "br_dec_amounts_max_2_decimals"
    # Every monetary BT is capped at two decimals by its own BR-DEC rule
    # (BR-DEC-09 BT-106, -12 BT-109, -13 BT-110, -14 BT-112, -17 BT-114,
    #  -23 BT-131, and so on). One check covers the family because the
    #  constraint is identical and the Decimal exponent already carries it,
    #  so this costs no wire change. NOTE the family constrains AMOUNTS only:
    #  there is no BR-DEC rule on quantities or VAT rates, so quantity is
    #  deliberately not checked here.

    # --- Ours, no standards backing ---
    AMOUNTS_APPEAR_IN_SOURCE = "amounts_appear_in_source"
    DATES_PARSE_AND_ORDER = "dates_parse_and_order"
    PARTY_NAMES_MATCH = "party_names_match"


class CheckOutcome(str, Enum):
    """Three states, not a bool. WITHIN_TOLERANCE is the state that stops the
    verifier generating exceptions on legitimate rounding/freight/UoM
    variance — and it feeds the confidence signal as a distinct level."""
    PASS = "pass"                          # exact
    WITHIN_TOLERANCE = "within_tolerance"  # differs, but inside the band
    FAIL = "fail"
    SKIPPED = "skipped"
    # NOT APPLICABLE to this document. A payment has no line items, so
    # BR-CO-10 cannot mean anything: neutral, and correctly so.
    INPUTS_MISSING = "inputs_missing"
    # The check COULD have applied and the field was not supplied. Neutral
    # for pass/fail (we cannot assert a document is wrong on evidence we do
    # not have) but it counts against COVERAGE, because otherwise an
    # extractor that emits fewer fields buys itself a cheaper verification.
    # That is the unevaluable-assertion gaming hole, and it is a real one:
    # the cheapest way to a clean report would be to extract nothing.


class CheckResult(_Base):
    check: CheckName
    outcome: CheckOutcome
    field_path: Optional[str] = Field(
        default=None,
        description="Where the failure is, e.g. 'line_items[LI-002].line_total'. "
                    "Beancount's validation errors carry a source location and "
                    "it is the difference between 'this document failed' and "
                    "'this field failed'. The console answers 'which invariant "
                    "failed and by how much' from this.")
    expected: Optional[str] = None    # stringified — human-readable audit line
    actual: Optional[str] = None
    delta: Optional[Decimal] = None
    tolerance_applied: Optional[Decimal] = Field(
        default=None,
        description="The allowed_delta() band used, so an auditor can see "
                    "WHY something passed within tolerance.")
    message: str = ""

    @field_validator("delta", "tolerance_applied", mode="before")
    @classmethod
    def _no_floats(cls, v):
        if isinstance(v, float):
            raise ValueError("floats are forbidden; send strings on the wire")
        return v


class VerificationReport(_Base):
    doc_id: str
    checks: list[CheckResult]
    rounding_policy: str = Field(
        default="aggregate",
        description="EN 16931 expects arithmetic on raw values with rounding "
                    "at the AGGREGATE level, not per line. Rounding each line "
                    "then summing produces discrepancies that fail BR-CO-10 "
                    "at large quantities. Recorded so the policy is auditable.")
    verified_ts: Optional[datetime] = None

    @property
    def verify_pass(self) -> bool:
        """The deterministic component of Mechanism B's confidence signal.
        Within-tolerance counts as a pass; neither neutral state fails.

        DELIBERATELY UNCHANGED by 1.3. INPUTS_MISSING does not fail a
        document, because the routing threshold was measured against this
        definition and silently moving it would invalidate every number
        already recorded. Coverage is reported ALONGSIDE, not folded in.
        """
        return not any(c.outcome == CheckOutcome.FAIL for c in self.checks)

    @property
    def field_coverage(self) -> tuple[int, int]:
        """(evaluated, evaluable) — the anti-gaming counter.

        Evaluable = every check except the not-applicable ones. Evaluated =
        those that actually ran. A document whose extractor dropped fields
        scores a high verify_pass and a LOW coverage, and the gap is exactly
        the thing a reviewer should look at.
        """
        evaluable = [c for c in self.checks
                     if c.outcome != CheckOutcome.SKIPPED]
        evaluated = [c for c in evaluable
                     if c.outcome != CheckOutcome.INPUTS_MISSING]
        return len(evaluated), len(evaluable)

    @property
    def evaluable_pct(self) -> float:
        done, total = self.field_coverage
        return 100.0 * done / total if total else 100.0

    @property
    def strict_pass(self) -> bool:
        """Stronger signal: every applicable check was EXACT. Use as the
        high-confidence tier of the routing rule; verify_pass alone is the
        low bar."""
        applicable = [c for c in self.checks
                      if c.outcome not in (CheckOutcome.SKIPPED,
                                           CheckOutcome.INPUTS_MISSING)]
        return bool(applicable) and all(
            c.outcome == CheckOutcome.PASS for c in applicable)


# ---------------------------------------------------------------------------
# Default tolerance policy — frozen with the schema so the verifier and the
# accuracy scoring band the same way.
# ---------------------------------------------------------------------------
# DECIDE: confirm these two numbers in the session. Industry practice is a
#         low single-digit percentage with an absolute cap; we default to
#         2% / $100 and tighten the pure-arithmetic identities, which should
#         hold to the cent on a well-formed document.

#: The smallest band that is never wider than the smallest real error.
#:
#: This was 0.01 and that was a bug, found by running the standard
#: maintainers' own rule fixtures against our verifier. A band of exactly one
#: cent makes a one-cent discrepancy UNDETECTABLE, because the comparison is
#: inclusive: delta 0.01 <= band 0.01 passes. One cent is the smallest
#: meaningful unit of financial error, so the floor was set precisely where it
#: blinded us. CEN's BR-CO-10 fixture states a subtotal of 200.01 against lines
#: of 110.00 + 90.00 and declares it an error; we called it within tolerance.
#:
#: 0.005 is a half-ULP at cent precision, which is the same quantity
#: precision.py infers for a single amount stated to two decimals, so the floor
#: is now consistent with the inference model rather than a round number
#: someone liked.
HALF_CENT = Decimal("0.005")

#: Ceiling on inferred widening for the document-total identities, i.e. the
#: point at which we adopt cent-precision semantics instead of inferring.
#:
#: HALF_CENT, not 0.01, and the difference is the whole point. CEN implements
#: BR-CO-13 as `round(sum * 100) div 100` compared to the stated amount, so ANY
#: difference at cent resolution is an error there. Our comparison is
#: `abs(delta) <= band`, so a ceiling of 0.01 would put a one-cent break
#: exactly ON the boundary and pass it: the bug would survive the fix. At
#: HALF_CENT a one-cent delta fails and a genuine sub-cent rounding artefact
#: still passes, which is the same partition CEN draws.
#:
#: Note what this means honestly: on these four rules the band collapses to the
#: floor, so precision inference is switched OFF for document totals rather
#: than merely bounded. That is the intended reading. A total is stated in the
#: document's currency at cent resolution; treating "250" as evidence the
#: author only knew the figure to the nearest unit is the wrong inference for
#: this one field class, however right it is elsewhere.
EN16931_TOTALS_CEILING = HALF_CENT

DEFAULT_TOLERANCES: dict[CheckName, Tolerance] = {
    # Internal arithmetic must be near-exact — a document that does not add
    # up internally is a genuine extraction or authoring error.
    CheckName.BR_CO_10:  Tolerance(pct=Decimal("0"), abs_cap=Decimal("0.05"),
                                   abs_floor=HALF_CENT,
                                   abs_ceiling=EN16931_TOTALS_CEILING),
    CheckName.BR_CO_13:  Tolerance(pct=Decimal("0"), abs_cap=Decimal("0.05"),
                                   abs_floor=HALF_CENT,
                                   abs_ceiling=EN16931_TOTALS_CEILING),
    CheckName.BR_CO_15:  Tolerance(pct=Decimal("0"), abs_cap=Decimal("0.05"),
                                   abs_floor=HALF_CENT,
                                   abs_ceiling=EN16931_TOTALS_CEILING),
    CheckName.LINE_NET_AMOUNT: Tolerance(pct=Decimal("0"),
                                         abs_cap=Decimal("0.05"),
                                         abs_floor=HALF_CENT),
    # BR-DEC is a digit-count rule, not a magnitude rule. It never calls
    # allowed_delta; the entry exists so every CheckName has a policy row.
    CheckName.BR_CO_16: Tolerance(pct=Decimal("0"),
                                  abs_cap=Decimal("0.05"),
                                  abs_floor=HALF_CENT,
                                  abs_ceiling=EN16931_TOTALS_CEILING),
    CheckName.BR_DEC_MAX_2: Tolerance(pct=Decimal("0"),
                                      abs_cap=Decimal("0"),
                                      abs_floor=Decimal("0")),
}

# Cross-document comparison is where real tolerance lives (PO vs invoice vs
# goods receipt). The reconciler, not the verifier, applies these.
CROSS_DOC_TOLERANCE = Tolerance(pct=Decimal("0.02"),
                                abs_cap=Decimal("100.00"),
                                abs_floor=HALF_CENT)


# ---------------------------------------------------------------------------
# Linking — documents grouped into the deal they belong to
# ---------------------------------------------------------------------------
#
# ADDITIVE, NOT WIRE-BREAKING. CanonicalDoc does not reference these types, so
# they never enter its $defs, the extraction schema is unchanged, and the
# frozen prompt prefix is byte-identical. Verified: prefix sha256 prefix
# 9d84942d7b81c742 before and after. No benchmark needs re-running.


class LinkMethod(str, Enum):
    """HOW an edge was established, carried on the edge itself.

    Not decoration. The product claim is that links are INFERRED rather than
    read from a fixed key, and a claim like that is only worth what the audit
    trail behind it is worth. An analyst looking at a flagged deal needs to
    know whether these documents were joined because one literally cited the
    other, or because we decided a mistyped identifier probably meant this one,
    or because the party and dates and amounts happened to agree.

    Ordered cheapest and safest first, mirroring the escalation ladder: every
    document exits at the earliest method that resolves it.
    """
    #: The reference resolved verbatim to a known document number.
    EXACT_REFERENCE = "exact_reference"
    #: The reference resolved to SEVERAL documents, and an independent
    #: attribute picked one. Document numbers are not globally unique in
    #: practice: real systems run per-vendor or per-year sequences, and the
    #: corpus reproduces that (92 numbers held by two documents each). Refusing
    #: every ambiguous reference costs real recall; guessing merges unrelated
    #: deals. Resolving on a second, independent field does neither.
    DISAMBIGUATED_REFERENCE = "disambiguated_reference"
    #: The reference did not resolve, but exactly one known document number is
    #: one character-confusion away from it (0/O, 1/l). Identifiers are the
    #: hardest field class in every published extraction comparison, and this
    #: is the failure they actually exhibit.
    REPAIRED_REFERENCE = "repaired_reference"
    #: No usable reference. Joined on party AND date proximity AND amount.
    ATTRIBUTE_MATCH = "attribute_match"


class LinkEdge(_Base):
    """One inferred join between two documents, with its provenance."""
    from_doc_id: str
    to_doc_id: str
    method: LinkMethod
    #: The reference string as printed, before any repair. Kept so a repair is
    #: always reversible and always visible: "we read PO-9567-A where the
    #: document said PO-9S67-A" is a claim an analyst can overrule.
    stated_reference: Optional[str] = None
    #: Populated for REPAIRED_REFERENCE and ATTRIBUTE_MATCH. Free text, human
    #: readable, part of the audit record.
    rationale: str = ""


class Chain(_Base):
    """A set of documents believed to belong to one deal.

    The linker GROUPS and nothing else. It does not judge, does not compare
    amounts, and does not flag duplicates. A chain containing two invoices
    against the same purchase order is a correctly linked chain; whether that
    is a duplicate is the reconciler's question, and keeping the two concerns
    apart is what stops a grouping bug from presenting as a false discrepancy.
    """
    chain_id: str
    doc_ids: list[str] = Field(default_factory=list)
    edges: list[LinkEdge] = Field(default_factory=list)
    #: References that named a document we never saw. NOT an error: a corpus
    #: sliced by date or party will legitimately contain half a chain. Recorded
    #: because a rising count is the signal that the linker is being fed a
    #: partial view, which would otherwise present as unexplained discrepancies
    #: further down the pipeline.
    dangling_references: list[str] = Field(default_factory=list)

    @property
    def is_singleton(self) -> bool:
        """A document we could not attach to anything. Worth surfacing rather
        than hiding: it is either a genuinely standalone document or a link we
        failed to make, and those need different responses."""
        return len(self.doc_ids) <= 1


# ---------------------------------------------------------------------------
# Reconciliation — what drifted between two documents in a chain
# ---------------------------------------------------------------------------
#
# ADDITIVE, NOT WIRE-BREAKING. Same argument as Chain: CanonicalDoc does not
# reference these, so the extraction schema and the frozen prefix are unchanged.
#
# THE SHAPE IS DERIVED FROM THE ANSWER KEY, NOT INVENTED
# -----------------------------------------------------
# Every planted anomaly in the corpus is exactly a two-document, one-field
# claim with a signed delta (or no delta, for the one non-numeric type). A
# Discrepancy that did not carry those four things could not be matched against
# ground truth, so precision and recall would be undefined. Designing the
# output shape before knowing how it is scored is how a system ends up
# measurable only by the metric that flatters it.


class Discrepancy(_Base):
    """One drift, between exactly two documents, at one field path.

    PAIRWISE ON PURPOSE. Long-context evaluations consistently show accuracy
    collapsing for information in the middle of a window, worse on the 7-14B
    class we can run locally, and multi-document cross-referencing is exactly
    where that bites. Whole chains are never concatenated into one prompt, so
    the unit of reasoning is a pair and the unit of reporting matches it.
    """
    anomaly_type: AnomalyType
    #: Unordered in meaning, sorted on construction so two reconcilers that
    #: walked the chain in opposite directions produce comparable output.
    doc_ids_involved: list[str] = Field(min_length=2, max_length=2)
    #: Same dotted convention as the answer key, LINE IDs not indices:
    #: 'total', 'payment_terms', 'line_items[LI-003].unit_price'. Indices would
    #: break the moment extraction dropped or reordered a line, which is
    #: precisely when attribution matters most.
    field_path: str
    #: Signed, in the direction (later document - earlier document). None for
    #: non-numeric drift such as a change of payment terms.
    delta: Optional[Decimal] = None
    #: The aggregate difference the FLAG decision was actually made on, and the
    #: band it was judged against.
    #:
    #: These two are not decoration and not the same as `delta`. A decoy in the
    #: corpus has a genuinely different unit price on a line, so a field-by-
    #: field comparator flags it and is wrong: 130 of the 210 planted price
    #: drifts are sized so the CHAIN total stays inside the cross-document
    #: band. Detection happens on the aggregate, attribution happens at the
    #: line, and a Discrepancy that cannot show both cannot be defended.
    decided_on_delta: Optional[Decimal] = None
    decided_on_band: Optional[Decimal] = None
    evidence: str = ""


class ChainVerdict(_Base):
    """The reconciler's output for one chain: what drifted, and what it cost.

    A chain with no discrepancies is a verdict, not an absence of one. 150 of
    the 510 chains are clean by construction, and a reconciler that emits
    nothing for them is indistinguishable from a reconciler that crashed.
    """
    chain_id: str
    doc_ids: list[str] = Field(default_factory=list)
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    #: Free-text, human-readable, one line per pair walked. The audit record,
    #: wired in from the start rather than retrofitted.
    trace: list[str] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.discrepancies


# ---------------------------------------------------------------------------
# Memory — resolutions worth remembering (Mechanism C)
# ---------------------------------------------------------------------------
#
# ADDITIVE, NOT WIRE-BREAKING. Same argument as Chain and Discrepancy:
# CanonicalDoc does not reference these, so the extraction schema and the
# frozen prompt prefix are unchanged.


class ResolutionKind(str, Enum):
    """WHAT an analyst decided, as a closed set.

    The signature migrates a pattern from the slow path to the fast one once
    enough matching resolutions agree, and matching requires collision. Free
    text does not collide: two analysts write "freight billed separately" and
    "freight is on a separate invoice" for the same decision, the signature
    never accumulates its k confirmations, and the mechanism silently never
    fires. So the enum is what the signature hashes.

    The free-text `pattern` on the episode carries how a human said it, because
    an audit line reading WITHIN_AGREED_TERMS alone is unreadable at review
    time. One collides, the other explains; pointing one at the other gets both.
    """
    #: The billed figure is correct as it stands.
    ACCEPT_AS_BILLED = "accept_as_billed"
    #: The discrepancy is real and money is owed back.
    REQUEST_CREDIT = "request_credit"
    #: Real difference, but covered by something agreed outside this chain.
    WITHIN_AGREED_TERMS = "within_agreed_terms"
    #: Our own extraction or linking was wrong, not the documents.
    EXTRACTION_ERROR = "extraction_error"
    #: Needs the counterparty. NOT a resolution, and deliberately in the enum
    #: so it can be recorded WITHOUT ever being counted towards migration: a
    #: pattern nobody has actually settled must not become an auto-resolve.
    ESCALATE_TO_VENDOR = "escalate_to_vendor"

    @property
    def settles(self) -> bool:
        return self is not ResolutionKind.ESCALATE_TO_VENDOR


class MemoryEpisode(_Base):
    """One human-approved resolution, with the provenance that makes it
    defensible.

    GATED ENTRY IS THE POINT. Unfiltered agent memory is a documented poisoning
    vector: small poisoned sets reliably outcompete benign experiences in
    retrieval, and goal-adjacent poisoned memories are used in most agentic
    evaluations across frontier models. There is also little peer-reviewed
    evidence that naive episodic memory improves accuracy on its own without
    curation. So this is not "the agent learns". It is curated precedent
    lookup, every entry traceable to a named human, and that framing is both
    honest and the stronger security claim.
    """
    signature: str
    party_name: str
    anomaly_type: AnomalyType
    resolution_kind: ResolutionKind
    #: How the approver described it. Never hashed, always shown.
    pattern: str = ""
    #: Named. An audit line reading "approved by someone" is not an audit line.
    approver: str
    approved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    chain_id: str = ""
    field_path: str = ""


# ---------------------------------------------------------------------------
# Interchange conventions (not code, but part of the contract)
# ---------------------------------------------------------------------------
# - Batches: JSONL, one ExtractedRecord per line, UTF-8.
# - Raw generated documents:  data/generated/{chain_id}/{doc_id}.{layout}.txt
# - Extraction output:        runs/{run_id}/extracted.jsonl
# - Verification output:      runs/{run_id}/verified.jsonl
# - Answer keys:              data/generated/answer_key.json
# - Hand-written fixtures:    data/fixtures/records/*.json  (ExtractedRecord,
#                             meta.tier = "none")
# - Corrupted fixtures:       data/fixtures/corrupted/{check_name}__*.json —
#                             filename names the check that must fire.
# - Holdout layout C:         data/fixtures/holdout/ — README inside says
#                             do not open until demo day.
# - Every accuracy figure in bench/ declares its ScoringMode.
#
# Signed off by: ______ (A)   ______ (B)   Date: ______
