"""A4 — ten HAND-WRITTEN CanonicalDoc fixtures.

WHY HAND-WRITTEN AND NOT DERIVED FROM gen.py
--------------------------------------------
These fixtures are the INDEPENDENT ORACLE. If they were derived from the
generator, a bug in the generator would propagate into them and nothing left
in the project could catch it. Every number below was chosen and multiplied
by hand; this file contains literals and serialisation only — no generation
logic. tests/test_fixtures.py re-checks the arithmetic independently, so a
typo here fails loudly instead of poisoning Person B's extraction target.

WHY THE LAYOUTS LOOK LIKE REAL SOFTWARE
---------------------------------------
Each source_text imitates the print layout of a top market tool — QuickBooks,
Xero, SAP, Coupa, NetSuite, Zoho, Stripe, Bill.com, a WMS goods receipt, and
Wave. The generator's Layout A/B are OUR OWN inventions, so extraction that
works on them proves nothing about documents we did not design. These ten are
the closest thing to market reality available without scraping real
(confidential) documents, and they are exactly what B's prompt should be
developed against first.

Coverage deliberately spans:
- doc types: invoice x4, PO x2, sales order, quote, payment, goods receipt
- allowance/charge combos: none/none, allowance-only, charge-only, both
- precision: cent-precision, whole-unit (Coupa), fractional quantities
- currencies: USD and GBP; date formats: US, UK, ISO, SAP dotted
- BR-CO-15 skip path: documents with no tax line (POs, Stripe-style)

Run once, commit the JSON output:
    PYTHONPATH=src python data/fixtures/make_records.py
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal as D
from pathlib import Path

from schemas import (
    CanonicalDoc,
    DocType,
    ExtractedRecord,
    ExtractionMeta,
    LineItem,
    Tier,
)

OUT = Path(__file__).parent / "records"


def li(n, desc, qty, uom, price=None, total=None, allow=None):
    return LineItem(line_id=f"LI-{n:03d}", description=desc, quantity=D(qty),
                    unit_of_measure=uom,
                    unit_price=None if price is None else D(price),
                    line_allowance=None if allow is None else D(allow),
                    line_total=None if total is None else D(total))


FIXTURES: list[tuple[str, CanonicalDoc]] = []


def fixture(name):
    def add(doc: CanonicalDoc):
        FIXTURES.append((name, doc))
    return add


# ---------------------------------------------------------------------------
# F1 · QuickBooks Online invoice — the most common US SMB layout.
# 4 x 385.00 = 1540.00 ; 6 x 112.50 = 675.00 ; subtotal 2215.00
# tax 8.25% of 2215.00 = 182.7375 -> 182.74 ; total 2397.74
# ---------------------------------------------------------------------------
fixture("F-0001_qbo_invoice")(CanonicalDoc(
    doc_id="F-0001", doc_number="1042", doc_type=DocType.INVOICE,
    party_name="Ridgeline HVAC Services", currency="USD",
    doc_date=date(2026, 5, 14), doc_date_raw="05/14/2026",
    line_items=[
        li(1, "Condenser coil replacement", "4", "EA", "385.00", "1540.00"),
        li(2, "Service labor", "6", "HUR", "112.50", "675.00"),
    ],
    subtotal=D("2215.00"), total_excl_tax=D("2215.00"),
    tax=D("182.74"), total=D("2397.74"), amount_due=D("2397.74"),
    payment_terms="Net 30", references=["EST-0977"],
    source_text="""\
Ridgeline HVAC Services
INVOICE

BILL TO                             INVOICE #  1042
Meridian Property Group             DATE       05/14/2026
882 Fulton Ave                      TERMS      Net 30
                                    DUE DATE   06/13/2026

ACTIVITY                        QTY      RATE       AMOUNT
Condenser coil replacement        4    385.00     1,540.00
Service labor                     6    112.50       675.00

                                SUBTOTAL           2,215.00
                                TAX (8.25%)          182.74
                                TOTAL              2,397.74
                                BALANCE DUE      $2,397.74

Reference: Estimate EST-0977
"""))

# ---------------------------------------------------------------------------
# F2 · Xero tax invoice (GBP, VAT, allowance-only) — UK/AU/NZ standard.
# 10 x 89.00 = 890.00 ; 2 x 1250.00 = 2500.00 ; subtotal 3390.00
# discount 10% = 339.00 ; net 3051.00 ; VAT 20% = 610.20 ; total 3661.20
# ---------------------------------------------------------------------------
fixture("F-0002_xero_invoice")(CanonicalDoc(
    doc_id="F-0002", doc_number="INV-0087", doc_type=DocType.INVOICE,
    party_name="Harrow & Finch Ltd", currency="GBP",
    doc_date=date(2026, 4, 2), doc_date_raw="2 Apr 2026",
    line_items=[
        li(1, "Monthly retainer - April", "10", "HUR", "89.00", "890.00"),
        li(2, "Brand guidelines document", "2", "EA", "1250.00", "2500.00"),
    ],
    subtotal=D("3390.00"), allowance_total=D("339.00"),
    total_excl_tax=D("3051.00"), tax=D("610.20"), total=D("3661.20"),
    amount_due=D("3661.20"), payment_terms="Due 1 May 2026",
    references=["PO-HF-2214"],
    source_text="""\
TAX INVOICE

Harrow & Finch Ltd                       Invoice Number   INV-0087
                                         Invoice Date     2 Apr 2026
                                         Reference        PO-HF-2214
                                         VAT Number       GB 442 8765 11

Description                     Quantity  Unit Price  Amount GBP
Monthly retainer - April           10.00       89.00      890.00
Brand guidelines document           2.00    1,250.00    2,500.00

                                   Subtotal              3,390.00
                                   Discount (10%)          339.00
                                   Total excl. VAT       3,051.00
                                   VAT 20%                 610.20
                                   TOTAL GBP             3,661.20

Payment due 1 May 2026.
"""))

# ---------------------------------------------------------------------------
# F3 · SAP purchase order (charge-only, no tax -> BR-CO-15 skip path).
# 500 x 12.35 = 6175.00 ; 120 x 48.00 = 5760.00 ; subtotal 11935.00
# freight 250.00 ; net value incl. delivery 12185.00
# ---------------------------------------------------------------------------
fixture("F-0003_sap_po")(CanonicalDoc(
    doc_id="F-0003", doc_number="4500012345", doc_type=DocType.PURCHASE_ORDER,
    party_name="Averill Fastener GmbH", currency="USD",
    doc_date=date(2026, 3, 27), doc_date_raw="27.03.2026",
    line_items=[
        li(1, "Hex bolt M10x60 zinc", "500", "EA", "12.35", "6175.00"),
        li(2, "Locking washer M10", "120", "EA", "48.00", "5760.00"),
    ],
    subtotal=D("11935.00"), charge_total=D("250.00"),
    total_excl_tax=D("12185.00"), total=D("12185.00"),
    payment_terms="ZB01 - Net 45", references=["QT-88113"],
    source_text="""\
Purchase Order
Document No. 4500012345                      Date: 27.03.2026
Vendor: 100447 Averill Fastener GmbH
Purch. Org.: 1000   Purch. Group: 001   Company Code: 1000
Terms of Payment: ZB01 - Net 45
Your quotation: QT-88113

Item  Material Description         Order Qty  Un   Net Price  Per  Net Value
00010 Hex bolt M10x60 zinc              500   EA       12.35    1   6,175.00
00020 Locking washer M10                120   EA       48.00    1   5,760.00

                                        Total net item value   11,935.00
                                        Freight (FRB1)            250.00
                                        Net value incl. delivery costs
                                                               12,185.00
"""))

# ---------------------------------------------------------------------------
# F4 · Coupa purchase order — WHOLE-UNIT precision throughout.
# 25 x 340 = 8500 ; 40 x 65 = 2600 ; total 11100. No tax on PO.
# The precision-inference target: every amount states unit precision.
# ---------------------------------------------------------------------------
fixture("F-0004_coupa_po")(CanonicalDoc(
    doc_id="F-0004", doc_number="PO-7731", doc_type=DocType.PURCHASE_ORDER,
    party_name="Cobalt Office Interiors", currency="USD",
    doc_date=date(2026, 6, 9), doc_date_raw="06/09/2026",
    line_items=[
        li(1, "Task chair, mesh back", "25", "EA", "340", "8500"),
        li(2, "Monitor arm, dual", "40", "EA", "65", "2600"),
    ],
    subtotal=D("11100"), total_excl_tax=D("11100"), total=D("11100"),
    payment_terms="Net 60", references=[],
    source_text="""\
Purchase Order PO-7731                            Status: Issued
Supplier: Cobalt Office Interiors                 Created: 06/09/2026
Ship To: 400 Commerce Park, Dock 4
Payment Term: Net 60

Line  Item                        Qty   Price   Total
1     Task chair, mesh back        25     340    8500
2     Monitor arm, dual            40      65    2600

                                  Order Total   11100 USD
"""))

# ---------------------------------------------------------------------------
# F5 · NetSuite sales order — BOTH allowance and charge (full BR-CO-13).
# 3 x 2150.00 = 6450.00 ; 1.5 x 180.00 = 270.00 ; subtotal 6720.00
# discount 5% = 336.00 ; shipping 95.00 ; net 6479.00
# tax 6% of 6479.00 = 388.74 ; total 6867.74
# ---------------------------------------------------------------------------
fixture("F-0005_netsuite_so")(CanonicalDoc(
    doc_id="F-0005", doc_number="SO-2201", doc_type=DocType.SALES_ORDER,
    party_name="Brightwell Labs Inc.", currency="USD",
    doc_date=date(2026, 5, 22), doc_date_raw="5/22/2026",
    line_items=[
        li(1, "Peristaltic pump PX-300", "3", "EA", "2150.00", "6450.00"),
        li(2, "On-site calibration", "1.5", "HUR", "180.00", "270.00"),
    ],
    subtotal=D("6720.00"), allowance_total=D("336.00"),
    charge_total=D("95.00"), total_excl_tax=D("6479.00"),
    tax=D("388.74"), total=D("6867.74"),
    payment_terms="Net 30", references=["Q-4410"],
    source_text="""\
Sales Order  #SO-2201
Date 5/22/2026            Customer  Brightwell Labs Inc.
Memo: converts quote Q-4410
Terms: Net 30

Item                      Qty    Rate       Amount
Peristaltic pump PX-300     3    2,150.00   6,450.00
On-site calibration       1.5      180.00     270.00

Subtotal                                    6,720.00
Discount Item (5%)                            336.00
Shipping (Ground)                              95.00
Total Before Tax                            6,479.00
Tax (6%)                                      388.74
Total                                       6,867.74
"""))

# ---------------------------------------------------------------------------
# F6 · Zoho Books quote.
# 12 x 74.50 = 894.00 ; 5 x 210.00 = 1050.00 ; subtotal 1944.00
# tax 10% = 194.40 ; total 2138.40
# ---------------------------------------------------------------------------
fixture("F-0006_zoho_quote")(CanonicalDoc(
    doc_id="F-0006", doc_number="QT-000341", doc_type=DocType.QUOTE,
    party_name="Sable Point Media", currency="USD",
    doc_date=date(2026, 2, 18), doc_date_raw="18/02/2026",
    line_items=[
        li(1, "Podcast episode edit", "12", "EA", "74.50", "894.00"),
        li(2, "Studio session", "5", "HUR", "210.00", "1050.00"),
    ],
    subtotal=D("1944.00"), total_excl_tax=D("1944.00"),
    tax=D("194.40"), total=D("2138.40"),
    payment_terms="Valid until 20/03/2026", references=[],
    source_text="""\
QUOTE
Quote#       QT-000341
Quote Date   18/02/2026
Expiry Date  20/03/2026
Customer     Sable Point Media

#  Item & Description        Qty    Rate      Amount
1  Podcast episode edit      12     74.50     894.00
2  Studio session             5    210.00   1,050.00

                             Sub Total       1,944.00
                             Tax Rate (10%)    194.40
                             Total           2,138.40
"""))

# ---------------------------------------------------------------------------
# F7 · Stripe-style invoice — minimal SaaS layout, no tax line.
# 1 x 2400.00 = 2400.00
# ---------------------------------------------------------------------------
fixture("F-0007_stripe_invoice")(CanonicalDoc(
    doc_id="F-0007", doc_number="A7C2E1-0042", doc_type=DocType.INVOICE,
    party_name="Lanternfish Software", currency="USD",
    doc_date=date(2026, 7, 1), doc_date_raw="July 1, 2026",
    line_items=[
        li(1, "Team plan - annual (Jul 2026 - Jun 2027)", "1", "EA",
           "2400.00", "2400.00"),
    ],
    subtotal=D("2400.00"), total_excl_tax=D("2400.00"),
    total=D("2400.00"), amount_due=D("2400.00"),
    payment_terms="Due July 15, 2026", references=[],
    source_text="""\
Invoice
Invoice number   A7C2E1-0042
Date of issue    July 1, 2026
Date due         July 15, 2026

Lanternfish Software

$2,400.00 USD due July 15, 2026

Description                                Qty   Unit price   Amount
Team plan - annual (Jul 2026 - Jun 2027)     1    $2,400.00   $2,400.00

Subtotal                                                      $2,400.00
Amount due                                                    $2,400.00
"""))

# ---------------------------------------------------------------------------
# F8 · Bill.com-style remittance advice (payment) — settles F5's total.
# ---------------------------------------------------------------------------
fixture("F-0008_billcom_payment")(CanonicalDoc(
    doc_id="F-0008", doc_number="PMT-118764", doc_type=DocType.PAYMENT,
    party_name="Brightwell Labs Inc.", currency="USD",
    doc_date=date(2026, 6, 24), doc_date_raw="06/24/2026",
    line_items=[],
    total=D("6867.74"),
    references=["INV-2201"],
    source_text="""\
Payment Confirmation
Payment ID     PMT-118764
Process date   06/24/2026
Payment method ACH - ePayment
Payer          Brightwell Labs Inc.

Payment amount   $6,867.74

Invoice(s) paid
Invoice #    Invoice amount    Amount paid
INV-2201         6,867.74        6,867.74
"""))

# ---------------------------------------------------------------------------
# F9 · Goods receipt note (WMS / SAP MIGO style) — quantities, NO prices.
# Receives F3's PO in full.
# ---------------------------------------------------------------------------
fixture("F-0009_grn")(CanonicalDoc(
    doc_id="F-0009", doc_number="GR-5000221", doc_type=DocType.GOODS_RECEIPT,
    party_name="Averill Fastener GmbH", currency="USD",
    doc_date=date(2026, 4, 8), doc_date_raw="08.04.2026",
    line_items=[
        li(1, "Hex bolt M10x60 zinc", "500", "EA"),
        li(2, "Locking washer M10", "120", "EA"),
    ],
    references=["4500012345"],
    source_text="""\
Goods Receipt Slip
Material Document  GR-5000221        Posting Date 08.04.2026
Vendor             Averill Fastener GmbH
Purchase Order     4500012345
Movement Type      101 - GR goods receipt

Item  Material Description       Qty Received  UoM  Storage Loc
0001  Hex bolt M10x60 zinc               500   EA   0001
0002  Locking washer M10                 120   EA   0001

Received by: R. Okafor   Dock 2
"""))

# ---------------------------------------------------------------------------
# F10 · Wave invoice — allowance AND charge, fractional quantity.
# 7.5 x 120.00 = 900.00 ; 3 x 415.80 = 1247.40 ; subtotal 2147.40
# discount 5% = 107.37 ; delivery 65.00 ; net 2105.03
# tax 8% of 2105.03 = 168.4024 -> 168.40 ; total 2273.43
# ---------------------------------------------------------------------------
fixture("F-0010_wave_invoice")(CanonicalDoc(
    doc_id="F-0010", doc_number="2026-0119", doc_type=DocType.INVOICE,
    party_name="Foxglove Joinery", currency="USD",
    doc_date=date(2026, 1, 30), doc_date_raw="January 30, 2026",
    line_items=[
        li(1, "Site carpentry", "7.5", "HUR", "120.00", "900.00"),
        li(2, "Oak shelving unit", "3", "EA", "415.80", "1247.40"),
    ],
    subtotal=D("2147.40"), allowance_total=D("107.37"),
    charge_total=D("65.00"), total_excl_tax=D("2105.03"),
    tax=D("168.40"), total=D("2273.43"), amount_due=D("2273.43"),
    payment_terms="Due on receipt", references=["EST-2026-014"],
    source_text="""\
INVOICE 2026-0119
Foxglove Joinery
Date: January 30, 2026        Payment due: On receipt
Estimate ref: EST-2026-014

Items                     Quantity    Price      Amount
Site carpentry                 7.5    120.00     900.00
Oak shelving unit                3    415.80   1,247.40

Subtotal                                       2,147.40
Discount - returning client (5%)                 107.37
Delivery                                          65.00
Total before tax                               2,105.03
Sales tax (8%)                                   168.40
Amount due                                    $2,273.43
"""))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, doc in FIXTURES:
        rec = ExtractedRecord(
            doc=doc,
            meta=ExtractionMeta(tier=Tier.NONE, model_id="hand-written"))
        (OUT / f"{name}.json").write_text(rec.model_dump_json(indent=2),
                                          encoding="utf-8")
    print(f"{len(FIXTURES)} fixtures -> {OUT}")


if __name__ == "__main__":
    main()
