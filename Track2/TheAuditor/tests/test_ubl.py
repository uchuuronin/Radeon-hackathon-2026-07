"""UBL serialiser tests. No network, no artefacts, no Saxon required.

The conformance harness itself needs a 20 MB download and an XSLT processor, so
it cannot run in the normal suite. These tests guard the properties that make
its result MEANINGFUL, and they run everywhere:

  - the numbers in the XML are the numbers in the record, unrounded
  - no placeholder is ever a monetary value
  - the document under test is not silently repaired on the way out

That last one is the whole risk. A serialiser that normalised precision, or
recomputed a total to make the XML tidy, would produce a conformance run that
validated our serialiser rather than our verifier, and it would pass beautifully
while proving nothing.
"""

import re
from datetime import date
from decimal import Decimal

import pytest

from schemas import CanonicalDoc, DocType, LineItem
from ubl import PLACEHOLDER, _vat_rate, to_ubl_invoice


def _inv(**kw) -> CanonicalDoc:
    base = dict(
        doc_id="D-1", doc_number="INV-3987", doc_type=DocType.INVOICE,
        party_name="Adventure Works Supply", doc_date=date(2026, 5, 2),
        currency="USD", source_text="",
        line_items=[LineItem(line_id="LI-001", description="Widget",
                             quantity=Decimal("2"), unit_of_measure="H87",
                             unit_price=Decimal("100.00"),
                             line_total=Decimal("200.00"))],
        subtotal=Decimal("200.00"), total_excl_tax=Decimal("200.00"),
        tax=Decimal("40.00"), total=Decimal("240.00"),
        amount_due=Decimal("240.00"))
    base.update(kw)
    return CanonicalDoc(**base)


class TestTheDocumentIsNotRepairedOnTheWayOut:
    """If the serialiser tidies anything, the conformance run tests the
    serialiser instead of the verifier and passes for the wrong reason."""

    def test_stated_precision_survives(self):
        """4500 and 4500.00 state different precision and the verifier derives
        its tolerance band from it. Normalising here would hide exactly the
        defect BR-DEC exists to catch."""
        xml = to_ubl_invoice(_inv(subtotal=Decimal("200"),
                                  total_excl_tax=Decimal("200")))
        assert ">200<" in xml and ">200.00<" not in xml.split("TaxAmount")[0]

    def test_a_broken_identity_is_transmitted_broken(self):
        """The mutation sweep depends on this completely."""
        xml = to_ubl_invoice(_inv(subtotal=Decimal("999.00")))
        assert ">999.00<" in xml            # not silently recomputed to 200.00

    def test_a_third_decimal_is_transmitted(self):
        xml = to_ubl_invoice(_inv(total=Decimal("240.001"), amount_due=None))
        assert ">240.001<" in xml

    def test_no_amount_is_rounded(self):
        odd = Decimal("77988.20")
        xml = to_ubl_invoice(_inv(subtotal=odd, total_excl_tax=odd,
                                  tax=Decimal("15597.64"),
                                  total=Decimal("93585.84"),
                                  amount_due=Decimal("93585.84")))
        for v in ("77988.20", "15597.64", "93585.84"):
            assert f">{v}<" in xml


class TestPlaceholdersAreNeverArithmetic:
    """Placeholders are what make the mapping possible; they must never touch a
    value a calculation rule computes over, or the comparison is meaningless."""

    def test_no_placeholder_is_a_number(self):
        for key, val in PLACEHOLDER.items():
            if key in ("postcode", "vat_id", "buyer_vat_id"):
                continue                    # identifiers, not amounts
            assert not re.fullmatch(r"-?\d+(\.\d+)?", val), key

    def test_monetary_elements_all_carry_the_document_currency(self):
        xml = to_ubl_invoice(_inv(currency="GBP"))
        for m in re.finditer(r'<cbc:(\w*Amount)\b([^>]*)>', xml):
            assert 'currencyID="GBP"' in m.group(2), m.group(1)


class TestVatRateDerivation:
    def test_rate_recovered_from_amounts(self):
        assert _vat_rate(_inv()) == Decimal("20.00")

    def test_rate_is_rounded_so_br_co_17_is_not_manufactured(self):
        """BR-CO-17 recomputes tax from the rate and compares within a cent, so
        an unrounded repeating rate would create a failure the source document
        does not contain."""
        r = _vat_rate(_inv(tax=Decimal("16.50"), total_excl_tax=Decimal("200.00"),
                           subtotal=Decimal("200.00"), total=Decimal("216.50"),
                           amount_due=Decimal("216.50")))
        assert -r.as_tuple().exponent <= 2

    def test_zero_tax_uses_a_zero_rated_category(self):
        xml = to_ubl_invoice(_inv(tax=None, total=Decimal("200.00"),
                                  amount_due=Decimal("200.00")))
        assert "<cbc:ID>Z</cbc:ID>" in xml


class TestScopeIsHonest:
    def test_non_invoices_are_refused_rather_than_approximated(self):
        """EN 16931 is an invoice standard. Emitting a goods receipt as an
        Invoice to raise the document count would be padding the result."""
        with pytest.raises(ValueError, match="invoices only"):
            to_ubl_invoice(_inv(doc_type=DocType.GOODS_RECEIPT))

    def test_generator_unit_codes_are_un_ece_rec20(self):
        """Regression on the first defect the differential run found: "EA" is
        the obvious code for "each" and is not in the list EN 16931 accepts
        (BR-CL-23). The correct code is H87. Our verifier does not check
        codelists, so our code and our fixtures were wrong together and only an
        independent implementation could tell us."""
        from gen import PRODUCTS
        valid = {"H87", "HUR", "LTR", "MTR", "KGM", "C62"}
        for name, uom, *_ in PRODUCTS:
            assert uom in valid, f"{name} uses {uom!r}, not a Rec 20 code"

    def test_wellformed_xml(self):
        import xml.etree.ElementTree as ET
        ET.fromstring(to_ubl_invoice(_inv()))


# ---------------------------------------------------------------------------
# The reverse direction, and what CEN's own fixtures taught us
# ---------------------------------------------------------------------------

class TestUblReader:
    """The reader exists to let the standard maintainers' fixtures act as an
    oracle for our verifier. These guard the two things that reading taught us,
    both of which were wrong before a fixture said so."""

    def _inv(self, body: str):
        import xml.etree.ElementTree as ET
        from ubl import NS, from_ubl_invoice
        xml = (f'<Invoice xmlns="{NS[""]}" xmlns:cac="{NS["cac"]}" '
               f'xmlns:cbc="{NS["cbc"]}">{body}</Invoice>')
        return from_ubl_invoice(ET.fromstring(xml))

    def test_second_currency_tax_total_is_ignored(self):
        """A DKK invoice may restate its VAT in EUR for a tax authority. Only
        the document-currency one belongs in BR-CO-15; taking whichever came
        first would compare a DKK total against an EUR tax amount."""
        d = self._inv(
            '<cbc:DocumentCurrencyCode>DKK</cbc:DocumentCurrencyCode>'
            '<cac:TaxTotal><cbc:TaxAmount currencyID="DKK">675.00</cbc:TaxAmount></cac:TaxTotal>'
            '<cac:TaxTotal><cbc:TaxAmount currencyID="EUR">628.62</cbc:TaxAmount></cac:TaxTotal>'
            '<cac:LegalMonetaryTotal>'
            '<cbc:TaxExclusiveAmount currencyID="DKK">4000.00</cbc:TaxExclusiveAmount>'
            '<cbc:TaxInclusiveAmount currencyID="DKK">4675.00</cbc:TaxInclusiveAmount>'
            '</cac:LegalMonetaryTotal>')
        assert d.tax == Decimal("675.00")

    def test_contradictory_tax_totals_yield_none_not_the_first(self):
        """Two different amounts in the same currency is an error because there
        is then no single total VAT amount. Taking the first would resolve the
        contradiction on the document's behalf and hide it."""
        d = self._inv(
            '<cbc:DocumentCurrencyCode>EUR</cbc:DocumentCurrencyCode>'
            '<cac:TaxTotal><cbc:TaxAmount currencyID="EUR">700.00</cbc:TaxAmount></cac:TaxTotal>'
            '<cac:TaxTotal><cbc:TaxAmount currencyID="EUR">715</cbc:TaxAmount></cac:TaxTotal>')
        assert d.tax is None

    def test_stated_precision_survives_the_read(self):
        d = self._inv(
            '<cbc:DocumentCurrencyCode>EUR</cbc:DocumentCurrencyCode>'
            '<cac:LegalMonetaryTotal>'
            '<cbc:TaxInclusiveAmount currencyID="EUR">1250.010</cbc:TaxInclusiveAmount>'
            '</cac:LegalMonetaryTotal>')
        assert str(d.total) == "1250.010"


class TestPrecisionSemanticsAreSelectable:
    """Document totals use cent-precision semantics; everything else infers.

    CEN's fixtures state a tax amount as "250" and a gross total one cent off,
    and call it an error. Unbounded inference said "250" could have been 249.5,
    so a cent was nothing, and we disagreed with the standard on 5 of its 48
    labelled fixtures. Every miss was a one-cent break.

    That is the wrong trade for this field class. A cent-level discrepancy on a
    document total is not a rounding artefact to a reconciliation engine, it is
    the thing the engine exists to find. So the total identities carry an
    abs_ceiling and inference is bounded there, while the line-level and
    cross-document checks keep it.
    """

    def test_total_identities_do_not_widen_past_cent_resolution(self):
        from schemas import DEFAULT_TOLERANCES, CheckName, allowed_delta
        tol = DEFAULT_TOLERANCES[CheckName.BR_CO_15]
        band = allowed_delta(Decimal("1250.00"), tol, ["1000.00", "250"])
        # A one-cent break must FAIL. Note the strict inequality: the check
        # compares `abs(delta) <= band`, so a ceiling of 0.01 would sit a cent
        # exactly ON the boundary and pass it. The half-cent is what actually
        # draws the partition CEN draws.
        assert Decimal("0.01") > band
        # ...while a genuine sub-cent rounding artefact still passes, so this
        # is a narrowing, not a switch to exact equality.
        assert Decimal("0.004") <= band

    def test_inference_still_widens_where_precision_is_genuinely_unknown(self):
        """The ceiling is deliberately NOT global.

        A line total is qty x unit price. A unit price printed to fewer
        decimals than it was kept to makes the product genuinely wider, so
        capping it would manufacture failures rather than find them.
        """
        from schemas import DEFAULT_TOLERANCES, CheckName, allowed_delta
        tol = DEFAULT_TOLERANCES[CheckName.LINE_NET_AMOUNT]
        assert tol.abs_ceiling is None
        band = allowed_delta(Decimal("6792.50"), tol, ["50", "135.85"])
        assert band > Decimal("0.5")

    def test_strict_mode_allows_only_the_floor(self):
        from schemas import (DEFAULT_TOLERANCES, STRICT_PRECISION, CheckName,
                             allowed_delta)
        tol = DEFAULT_TOLERANCES[CheckName.BR_CO_15]
        token = STRICT_PRECISION.set(True)
        try:
            tight = allowed_delta(Decimal("1250.00"), tol, ["1000.00", "250"])
        finally:
            STRICT_PRECISION.reset(token)
        assert tight == tol.abs_floor
        assert Decimal("0.01") > tight          # a cent now fails

    def test_the_default_is_inference(self):
        from schemas import STRICT_PRECISION
        assert STRICT_PRECISION.get() is False


class TestOneCentIsDetectable:
    def test_the_floor_no_longer_swallows_a_cent(self):
        """The defect CEN's BR-CO-10 fixture found: subtotal 200.01 against
        lines of 110.00 + 90.00, which the standard calls an error and we
        called within tolerance, because the floor was exactly one cent and the
        comparison is inclusive."""
        from schemas import CheckName, CheckOutcome
        from verify.engine import verify_doc
        doc = _inv(subtotal=Decimal("200.01"), total_excl_tax=Decimal("200.01"),
                   tax=None, total=Decimal("200.01"),
                   amount_due=Decimal("200.01"),
                   line_items=[
                       LineItem(line_id="LI-001", description="A",
                                quantity=Decimal("1"),
                                unit_price=Decimal("110.00"),
                                line_total=Decimal("110.00")),
                       LineItem(line_id="LI-002", description="B",
                                quantity=Decimal("1"),
                                unit_price=Decimal("90.00"),
                                line_total=Decimal("90.00"))])
        r = next(c for c in verify_doc(doc).checks
                 if c.check == CheckName.BR_CO_10)
        assert r.outcome == CheckOutcome.FAIL


class TestBrCo16:
    def test_settlement_identity(self):
        from schemas import CheckName, CheckOutcome
        from verify.engine import verify_doc
        ok = _inv(total=Decimal("1200.78"), paid_amount=Decimal("100.00"),
                  rounding_amount=Decimal("0.22"),
                  amount_due=Decimal("1101.00"))
        r = next(c for c in verify_doc(ok).checks
                 if c.check == CheckName.BR_CO_16)
        assert r.outcome == CheckOutcome.PASS

    def test_a_part_payment_against_the_wrong_invoice_breaks_only_this_rule(self):
        """BR-CO-15 asks whether the invoice adds up; BR-CO-16 asks whether
        what is still owed follows from what was billed and already paid."""
        from schemas import CheckName, CheckOutcome
        from verify.engine import verify_doc
        # Built so the invoice itself is internally consistent: 200 net + 40
        # VAT = 240 gross, which BR-CO-15 accepts. Only the settlement is
        # wrong, and only BR-CO-16 sees it.
        bad = _inv(subtotal=Decimal("200.00"), total_excl_tax=Decimal("200.00"),
                   tax=Decimal("40.00"), total=Decimal("240.00"),
                   paid_amount=Decimal("100.00"),
                   amount_due=Decimal("240.00"))
        checks = {c.check: c.outcome for c in verify_doc(bad).checks}
        assert checks[CheckName.BR_CO_16] == CheckOutcome.FAIL
        assert checks[CheckName.BR_CO_15] == CheckOutcome.PASS

    def test_absent_paid_amount_means_nothing_paid(self):
        from schemas import CheckName, CheckOutcome
        from verify.engine import verify_doc
        d = _inv(total=Decimal("500.00"), amount_due=Decimal("500.00"))
        r = next(c for c in verify_doc(d).checks
                 if c.check == CheckName.BR_CO_16)
        assert r.outcome == CheckOutcome.PASS
