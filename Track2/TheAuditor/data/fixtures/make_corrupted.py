"""Corrupted fixtures: one broken invariant each, named after the check
that must fire.

Pattern per the plan (and independently used by the K-1 document-intelligence
pipeline's test profiles, e.g. "profile 14, invalid 85%, triggers rule A1"):
take a known-good fixture, apply ONE named perturbation aimed at ONE check,
save it as corrupted/{check_name}__{source}.json. tests/test_verifier.py then
asserts the named check FAILS on each file.

A perturbation may trip secondary checks too (changing a total also removes
it from source-by-value) — that is realistic, not a defect. The contract is:
the NAMED check fires, and every clean fixture produces zero FAILs.

The star fixture is the last one: a CONSISTENT hallucination. Every amount
shifted together, all identities still hold, only the source check can see
it. That is the dominant real-world failure mode (invented numerics, ~2/3 of
extraction errors) and the whole reason AMOUNTS_APPEAR_IN_SOURCE exists.

Run once, commit the output:
    PYTHONPATH=src python data/fixtures/make_corrupted.py
"""

from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

from schemas import CanonicalDoc, ExtractedRecord

RECORDS = Path(__file__).parent / "records"
OUT = Path(__file__).parent / "corrupted"


def load(name: str) -> ExtractedRecord:
    return ExtractedRecord.model_validate_json(
        (RECORDS / f"{name}.json").read_text(encoding="utf-8"))


def save(check: str, source: str, rec: ExtractedRecord) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{check}__{source}.json"
    path.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    print(f"  {path.name}")


def corrupt(rec: ExtractedRecord, **doc_updates) -> ExtractedRecord:
    doc = rec.doc.model_copy(update=doc_updates)
    CanonicalDoc.model_validate(doc.model_dump())   # stay wire-valid
    return rec.model_copy(update={"doc": doc})


def main() -> None:
    print("corrupted fixtures -> data/fixtures/corrupted/")

    # BR-CO-10: subtotal drifts away from the sum of its lines.
    # QBO invoice: lines sum 2215.00, subtotal claimed 2315.00.
    save("br_co_10_line_totals_sum_to_subtotal", "F-0001_qbo_invoice",
         corrupt(load("F-0001_qbo_invoice"), subtotal=D("2315.00")))

    # BR-CO-13: the discount is stated but NOT applied — the unapplied-
    # discount anomaly in miniature. Xero: net should be 3390-339=3051,
    # claimed 3390.00 (discount ignored).
    save("br_co_13_total_excl_tax_identity", "F-0002_xero_invoice",
         corrupt(load("F-0002_xero_invoice"),
                 total_excl_tax=D("3390.00"),
                 tax=D("678.00"),           # 20% of the wrong net,
                 total=D("4068.00")))       # so ONLY BR-CO-13 breaks

    # BR-CO-15: tax and net no longer add to the total.
    # NetSuite SO: 6479.00 + 388.74 = 6867.74, total claimed 6877.74.
    save("br_co_15_total_incl_tax_identity", "F-0005_netsuite_so",
         corrupt(load("F-0005_netsuite_so"), total=D("6877.74"),
                 amount_due=None))

    # LINE_NET_AMOUNT: a quantity mis-read. SAP PO line 00010: qty read as
    # 550 instead of 500; 550 x 12.35 = 6792.50 != 6175.00 stated.
    sap = load("F-0003_sap_po")
    lines = [li.model_copy(update={"quantity": D("550")})
             if li.line_id == "LI-001" else li for li in sap.doc.line_items]
    save("line_net_amount_identity", "F-0003_sap_po",
         corrupt(sap, line_items=lines))

    # AMOUNTS_APPEAR_IN_SOURCE — the consistent hallucination. Stripe
    # invoice: every 2400.00 becomes 2450.00 IN LOCKSTEP. BR-CO-10, 13, 15
    # and the line identity all still hold; the number simply never appears
    # in the document. Arithmetic cannot catch this; only the source can.
    st = load("F-0007_stripe_invoice")
    li = st.doc.line_items[0].model_copy(update={
        "unit_price": D("2450.00"), "line_total": D("2450.00")})
    save("amounts_appear_in_source", "F-0007_stripe_invoice",
         corrupt(st, line_items=[li],
                 subtotal=D("2450.00"), total_excl_tax=D("2450.00"),
                 total=D("2450.00"), amount_due=D("2450.00")))

    # DATES_PARSE_AND_ORDER: normalised date contradicts the printed one.
    # Wave invoice printed "January 30, 2026"; extraction normalised to 2025.
    from datetime import date
    save("dates_parse_and_order", "F-0010_wave_invoice",
         corrupt(load("F-0010_wave_invoice"),
                 doc_date=date(2025, 1, 30)))

    print("done")


if __name__ == "__main__":
    main()
