"""CanonicalDoc -> UBL 2.1 Invoice XML, for validation against the CEN reference.

WHY THIS EXISTS
---------------
Our verifier claims to implement EN 16931 calculation rules. Until now that
claim rested on us having read the standard and written Python we believed
matched it. That is the weakest form of a standards claim: it is unfalsifiable
by anyone who does not re-read the standard themselves, which is nobody.

CEN publishes the rules as machine-readable Schematron, and the EU publishes a
compiled XSLT of it under EUPL v1.2. So the claim can be upgraded from "we
implemented the rules" to "our implementation agrees with the CEN reference
implementation, on documents we generate, including ones we deliberately
broke". That is checkable by a judge in one command, and it is the difference
between asserting conformance and demonstrating it.

This module is the adapter that makes the comparison possible. It is NOT part
of the agent pipeline: nothing in the escalation ladder imports it, and it adds
no runtime dependency to the product. It exists so the test harness can put our
documents in front of somebody else's validator.

SCOPE, STATED HONESTLY
----------------------
UBL Invoice only. Quotes, orders, despatch and receipt advices and payments are
different UBL document types with different rule sets, and EN 16931 is an
invoice standard. The invoice is where the arithmetic identities live, so it is
the document worth cross-validating; claiming more would be padding.

The mapping is lossy in one direction on purpose. EN 16931 requires fields a
reconciliation record has no reason to carry (a seller postal address, a VAT
scheme identifier, a country code), because it is an e-invoicing transmission
standard and we are not transmitting anything. Those are filled with declared
placeholders, marked below, and they are all NON-ARITHMETIC. No placeholder
touches a value any BR-CO or BR-DEC rule computes over, which is what keeps the
comparison meaningful: every number the reference validator checks is a number
that came out of our generator.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional
from xml.sax.saxutils import escape

from schemas import CanonicalDoc, DocType, LineItem

NS = {
    "": "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2",
    "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
}

#: The CIUS identifier that tells a validator to apply the EN 16931 core rules
#: and nothing else. Using a Peppol identifier instead would pull in Peppol's
#: additional rules, which we do not claim to satisfy.
CUSTOMIZATION_ID = "urn:cen.eu:en16931:2017"
PROFILE_ID = "urn:fdc:peppol.eu:2017:poacc:billing:01:1.0"

#: Placeholders. Every one is non-arithmetic: required by the transmission
#: standard, irrelevant to the calculation rules under test. Listed as a
#: constant so the test can assert none of them is a monetary value.
PLACEHOLDER = {
    "country": "DE",
    "street": "Not modelled",
    "city": "Not modelled",
    "postcode": "00000",
    "buyer_name": "Buyer Not Modelled",
    "vat_id": "DE123456789",
    "buyer_vat_id": "DE987654321",
}


def _d(v: Optional[Decimal]) -> str:
    """Serialise preserving the PRINTED precision.

    This is the whole point. If this normalised to two decimals it would hide
    exactly the class of defect BR-DEC exists to catch, and the comparison
    would be against a document we had silently repaired on the way out.
    """
    return "0" if v is None else str(v)


def _amt(tag: str, v: Optional[Decimal], cur: str, ind: str = "    ") -> str:
    if v is None:
        return ""
    return f'{ind}<cbc:{tag} currencyID="{cur}">{_d(v)}</cbc:{tag}>\n'


def _vat_rate(doc: CanonicalDoc) -> Decimal:
    """Recover the VAT percentage from the amounts the document states.

    Derived rather than stored because CanonicalDoc has no rate field: our
    records come from documents that print amounts, not tax configurations.
    Rounded to two decimals because BR-CO-17 recomputes the tax amount from
    this rate and compares within a cent, so an unrounded repeating rate would
    manufacture a failure the source document does not contain.
    """
    net = doc.total_excl_tax if doc.total_excl_tax is not None else doc.subtotal
    if not net or doc.tax is None:
        return Decimal("0")
    return (doc.tax / net * 100).quantize(Decimal("0.01"))


def to_ubl_invoice(doc: CanonicalDoc) -> str:
    """Render an invoice as EN 16931-conformant UBL 2.1."""
    if doc.doc_type != DocType.INVOICE:
        raise ValueError(f"UBL mapping covers invoices only, got {doc.doc_type}")

    cur = doc.currency
    rate = _vat_rate(doc)
    category = "S" if rate > 0 else "Z"
    net = doc.total_excl_tax if doc.total_excl_tax is not None else doc.subtotal

    lines = []
    for i, li in enumerate(doc.line_items, start=1):
        # BT-129/130: quantity and its unit. BT-131: the line net amount, the
        # value BR-CO-10 sums. BT-146: the unit price.
        lines.append(
            f"""  <cac:InvoiceLine>
    <cbc:ID>{i}</cbc:ID>
    <cbc:InvoicedQuantity unitCode="{escape(li.unit_of_measure or 'EA')}">{_d(li.quantity)}</cbc:InvoicedQuantity>
    <cbc:LineExtensionAmount currencyID="{cur}">{_d(li.line_total)}</cbc:LineExtensionAmount>
    <cac:Item>
      <cbc:Name>{escape(li.description or 'Item')}</cbc:Name>
      <cac:ClassifiedTaxCategory>
        <cbc:ID>{category}</cbc:ID>
        <cbc:Percent>{_d(rate)}</cbc:Percent>
        <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
      </cac:ClassifiedTaxCategory>
    </cac:Item>
    <cac:Price>
      <cbc:PriceAmount currencyID="{cur}">{_d(li.unit_price)}</cbc:PriceAmount>
    </cac:Price>
  </cac:InvoiceLine>""")

    exemption = ("" if category == "S" else
                 "\n        <cbc:TaxExemptionReasonCode>VATEX-EU-O</cbc:TaxExemptionReasonCode>")

    doc_allowance = ""
    if doc.allowance_total is not None:
        doc_allowance = f"""  <cac:AllowanceCharge>
    <cbc:ChargeIndicator>false</cbc:ChargeIndicator>
    <cbc:AllowanceChargeReason>Discount</cbc:AllowanceChargeReason>
    <cbc:Amount currencyID="{cur}">{_d(doc.allowance_total)}</cbc:Amount>
    <cac:TaxCategory>
      <cbc:ID>{category}</cbc:ID>
      <cbc:Percent>{_d(rate)}</cbc:Percent>
      <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
    </cac:TaxCategory>
  </cac:AllowanceCharge>
"""
    doc_charge = ""
    if doc.charge_total is not None:
        doc_charge = f"""  <cac:AllowanceCharge>
    <cbc:ChargeIndicator>true</cbc:ChargeIndicator>
    <cbc:AllowanceChargeReason>Freight</cbc:AllowanceChargeReason>
    <cbc:Amount currencyID="{cur}">{_d(doc.charge_total)}</cbc:Amount>
    <cac:TaxCategory>
      <cbc:ID>{category}</cbc:ID>
      <cbc:Percent>{_d(rate)}</cbc:Percent>
      <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
    </cac:TaxCategory>
  </cac:AllowanceCharge>
"""

    refs = "".join(
        f"""  <cac:OrderReference><cbc:ID>{escape(r)}</cbc:ID></cac:OrderReference>\n"""
        for r in doc.references[:1])

    p = PLACEHOLDER
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Invoice xmlns="{NS['']}"
         xmlns:cac="{NS['cac']}"
         xmlns:cbc="{NS['cbc']}">
  <cbc:CustomizationID>{CUSTOMIZATION_ID}</cbc:CustomizationID>
  <cbc:ProfileID>{PROFILE_ID}</cbc:ProfileID>
  <cbc:ID>{escape(doc.doc_number)}</cbc:ID>
  <cbc:IssueDate>{doc.doc_date.isoformat()}</cbc:IssueDate>
  <cbc:DueDate>{doc.doc_date.isoformat()}</cbc:DueDate>
  <cbc:InvoiceTypeCode>380</cbc:InvoiceTypeCode>
  <cbc:DocumentCurrencyCode>{cur}</cbc:DocumentCurrencyCode>
{refs}  <cac:AccountingSupplierParty>
    <cac:Party>
      <cac:PartyName><cbc:Name>{escape(doc.party_name)}</cbc:Name></cac:PartyName>
      <cac:PostalAddress>
        <cbc:StreetName>{p['street']}</cbc:StreetName>
        <cbc:CityName>{p['city']}</cbc:CityName>
        <cbc:PostalZone>{p['postcode']}</cbc:PostalZone>
        <cac:Country><cbc:IdentificationCode>{p['country']}</cbc:IdentificationCode></cac:Country>
      </cac:PostalAddress>
      <cac:PartyTaxScheme>
        <cbc:CompanyID>{p['vat_id']}</cbc:CompanyID>
        <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
      </cac:PartyTaxScheme>
      <cac:PartyLegalEntity>
        <cbc:RegistrationName>{escape(doc.party_name)}</cbc:RegistrationName>
      </cac:PartyLegalEntity>
    </cac:Party>
  </cac:AccountingSupplierParty>
  <cac:AccountingCustomerParty>
    <cac:Party>
      <cac:PartyName><cbc:Name>{p['buyer_name']}</cbc:Name></cac:PartyName>
      <cac:PostalAddress>
        <cbc:StreetName>{p['street']}</cbc:StreetName>
        <cbc:CityName>{p['city']}</cbc:CityName>
        <cbc:PostalZone>{p['postcode']}</cbc:PostalZone>
        <cac:Country><cbc:IdentificationCode>{p['country']}</cbc:IdentificationCode></cac:Country>
      </cac:PostalAddress>
      <cac:PartyLegalEntity>
        <cbc:RegistrationName>{p['buyer_name']}</cbc:RegistrationName>
      </cac:PartyLegalEntity>
    </cac:Party>
  </cac:AccountingCustomerParty>
  <cac:PaymentMeans>
    <!-- Code 1, "instrument not defined". Deliberate: code 30 (credit
         transfer) triggers BR-61, which requires a payee account identifier,
         and a reconciliation record has no bank details because it never
         needed them. Inventing an IBAN to satisfy a transmission rule would
         put fabricated data into a document we are using as evidence. -->
    <cbc:PaymentMeansCode>1</cbc:PaymentMeansCode>
  </cac:PaymentMeans>
{doc_allowance}{doc_charge}  <cac:TaxTotal>
    <cbc:TaxAmount currencyID="{cur}">{_d(doc.tax)}</cbc:TaxAmount>
    <cac:TaxSubtotal>
      <cbc:TaxableAmount currencyID="{cur}">{_d(net)}</cbc:TaxableAmount>
      <cbc:TaxAmount currencyID="{cur}">{_d(doc.tax)}</cbc:TaxAmount>
      <cac:TaxCategory>
        <cbc:ID>{category}</cbc:ID>
        <cbc:Percent>{_d(rate)}</cbc:Percent>{exemption}
        <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
      </cac:TaxCategory>
    </cac:TaxSubtotal>
  </cac:TaxTotal>
  <cac:LegalMonetaryTotal>
{_amt('LineExtensionAmount', doc.subtotal, cur)}{_amt('TaxExclusiveAmount', net, cur)}{_amt('TaxInclusiveAmount', doc.total, cur)}{_amt('AllowanceTotalAmount', doc.allowance_total, cur)}{_amt('ChargeTotalAmount', doc.charge_total, cur)}{_amt('PrepaidAmount', doc.paid_amount, cur)}{_amt('PayableRoundingAmount', doc.rounding_amount, cur)}{_amt('PayableAmount', doc.amount_due if doc.amount_due is not None else doc.total, cur)}  </cac:LegalMonetaryTotal>
{chr(10).join(lines)}
</Invoice>
"""


# ---------------------------------------------------------------------------
# The reverse direction: UBL -> CanonicalDoc
# ---------------------------------------------------------------------------
#
# WHY BOTHER READING UBL WHEN THE PRODUCT ONLY EVER WRITES IT
# -----------------------------------------------------------
# Because it turns CEN's own test corpus into an oracle for OUR verifier.
#
# The artefacts we already fetch ship a directory of rule unit tests
# (test/Invoice-unit-UBL/BR-*.xml). Each file is a testSet of small invoice
# fragments, and each fragment carries a DECLARED EXPECTED OUTCOME written by
# the standard's own maintainers: <success>BR-CO-15</success> for a document
# that must pass, <error>BR-CO-15</error> for one that must fail.
#
# That is the corpus our mutation sweep cannot be: documents we did not write,
# breaks we did not choose, expectations we did not set. Our own mutants are
# large, systematic and, being ours, share whatever assumptions we made. CEN's
# are small, adversarial and independent, and several sit exactly on the
# one-cent boundary that our sweep never probes: BR-CO-15's error case states a
# gross total of 1250.01 where 1250.00 is required, which is a single cent of
# violation and precisely the region where our tolerance model and the
# standard's fixed +/-0.01 could legitimately disagree.
#
# So this reader exists to point our verifier at somebody else's labelled
# tests. It is not part of the agent pipeline and nothing in the ladder imports
# it.
#
# SCOPE. These fragments are deliberately minimal: most carry no party, no
# date, sometimes no currency, because they exist to isolate one rule. The
# reader therefore synthesises the non-arithmetic fields CanonicalDoc requires
# and takes every monetary value verbatim. Same discipline as the writer: no
# placeholder is ever an amount.

_UBL_TESTSET_NS = "http://difi.no/xsd/vefa/validator/1.0"
_CBC = f"{{{NS['cbc']}}}"
_CAC = f"{{{NS['cac']}}}"

#: Non-arithmetic stand-ins for fields a rule-isolation fragment omits.
_READER_DEFAULTS = {
    "doc_number": "CEN-UNIT-TEST",
    "party_name": "CEN Unit Test",
    "currency": "EUR",
    "doc_date": date(2026, 1, 1),
}


def _text(node: Optional[ET.Element]) -> Optional[str]:
    return node.text.strip() if node is not None and node.text else None


def _dec(node: Optional[ET.Element]) -> Optional[Decimal]:
    """Verbatim, including stated precision. Never normalised.

    "1250.01" and "1250.010" state different precision and our tolerance model
    is derived from stated precision, so quantising here would silently repair
    the very cases these fixtures exist to expose.
    """
    t = _text(node)
    if t is None:
        return None
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def from_ubl_invoice(inv: ET.Element, doc_id: str = "CEN-0000") -> CanonicalDoc:
    """Parse a UBL Invoice element into a CanonicalDoc.

    Only the elements our checks consume are read. Anything else in the
    document is ignored rather than approximated, because a half-read field
    would be indistinguishable from an absent one downstream and absence is
    load-bearing in this schema.
    """
    tot = inv.find(f"{_CAC}LegalMonetaryTotal")

    # A UBL invoice may carry SEVERAL TaxTotal elements and they mean two
    # different things, which took two CEN fixtures to learn:
    #
    #   - A SECOND CURRENCY. An invoice in DKK may restate its VAT in EUR for a
    #     tax authority. Both are correct and only the one in the document
    #     currency belongs in BR-CO-15, so filter by currencyID first. Reading
    #     whichever came first here would compare a DKK total against an EUR
    #     tax amount and report a spurious break.
    #   - A CONTRADICTION. Two amounts in the SAME currency that disagree is an
    #     error precisely because there is then no single total VAT amount for
    #     the identity to use. Taking the first would resolve the contradiction
    #     on the document's behalf and make it look consistent, so we take None:
    #     the document states no usable tax total, and absence is a fact this
    #     schema carries.
    doc_currency = _text(inv.find(f"{_CBC}DocumentCurrencyCode"))
    tax_totals = [t for t in inv.findall(f"{_CAC}TaxTotal")
                  if (t.find(f"{_CBC}TaxAmount") is not None
                      and (doc_currency is None
                           or t.find(f"{_CBC}TaxAmount").get("currencyID")
                           in (None, doc_currency)))]
    tax_amounts = {_dec(t.find(f"{_CBC}TaxAmount")) for t in tax_totals}
    tax_amounts.discard(None)
    tax_total = tax_totals[0] if len(tax_amounts) == 1 else None

    def m(parent, tag):
        return _dec(parent.find(f"{_CBC}{tag}")) if parent is not None else None

    lines = []
    for i, ln in enumerate(inv.findall(f"{_CAC}InvoiceLine"), start=1):
        price = ln.find(f"{_CAC}Price")
        item = ln.find(f"{_CAC}Item")
        qty_node = ln.find(f"{_CBC}InvoicedQuantity")
        qty = _dec(qty_node)
        lines.append(LineItem(
            line_id=_text(ln.find(f"{_CBC}ID")) or f"LI-{i:03d}",
            description=(_text(item.find(f"{_CBC}Name")) if item is not None
                         else None) or "Item",
            quantity=qty if qty is not None else Decimal(1),
            unit_of_measure=(qty_node.get("unitCode")
                             if qty_node is not None else None),
            unit_price=m(price, "PriceAmount"),
            line_total=m(ln, "LineExtensionAmount")))

    raw_date = _text(inv.find(f"{_CBC}IssueDate"))
    try:
        issued = date.fromisoformat(raw_date) if raw_date else _READER_DEFAULTS["doc_date"]
    except ValueError:
        issued = _READER_DEFAULTS["doc_date"]

    return CanonicalDoc(
        doc_id=doc_id,
        doc_number=_text(inv.find(f"{_CBC}ID")) or _READER_DEFAULTS["doc_number"],
        doc_type=DocType.INVOICE,
        party_name=_READER_DEFAULTS["party_name"],
        doc_date=issued,
        doc_date_raw=raw_date,
        currency=(_text(inv.find(f"{_CBC}DocumentCurrencyCode"))
                  or _READER_DEFAULTS["currency"]),
        line_items=lines,
        subtotal=m(tot, "LineExtensionAmount"),
        allowance_total=m(tot, "AllowanceTotalAmount"),
        charge_total=m(tot, "ChargeTotalAmount"),
        total_excl_tax=m(tot, "TaxExclusiveAmount"),
        tax=m(tax_total, "TaxAmount"),
        total=m(tot, "TaxInclusiveAmount"),
        rounding_amount=m(tot, "PayableRoundingAmount"),
        paid_amount=m(tot, "PrepaidAmount"),
        amount_due=m(tot, "PayableAmount"),
        source_text="")


def read_cen_testset(path) -> list[tuple[str, str, bool, CanonicalDoc]]:
    """Parse one CEN rule unit-test file.

    Returns (rule, description, expected_to_fail, doc) per embedded test. A
    <success> assertion means the rule must NOT fire; an <error> assertion means
    it must. Those declarations are the standard maintainers' own, which is the
    entire point of using them.
    """
    root = ET.parse(str(path)).getroot()
    out = []
    for i, test in enumerate(root.findall(f"{{{_UBL_TESTSET_NS}}}test")):
        assertion = test.find(f"{{{_UBL_TESTSET_NS}}}assert")
        if assertion is None:
            continue
        err = assertion.find(f"{{{_UBL_TESTSET_NS}}}error")
        ok = assertion.find(f"{{{_UBL_TESTSET_NS}}}success")
        node = err if err is not None else ok
        if node is None:
            continue                      # <warning> and friends: not a verdict
        inv = test.find(f"{{{NS['']}}}Invoice")
        if inv is None:
            continue
        out.append((
            (node.text or "").strip(),
            (_text(assertion.find(f"{{{_UBL_TESTSET_NS}}}description")) or "")[:80],
            err is not None,
            from_ubl_invoice(inv, doc_id=f"{path.stem}#{i}")))
    return out
