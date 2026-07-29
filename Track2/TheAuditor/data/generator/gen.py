"""A1 — the clean-chain generator.

Emits internally-consistent quote-to-cash chains. NO anomalies: a generator
that produces broken documents by accident is unusable as ground truth, so
correctness comes first and injection (A3) comes second, on top of this.

Every chain satisfies the EN 16931 identities exactly:
    BR-CO-10   sum(line_total)          == subtotal
    BR-CO-13   total_excl_tax           == subtotal - allowance + charge
    BR-CO-15   total                    == total_excl_tax + tax
Rounding happens at the AGGREGATE level, once, per EN 16931 — never per line
and then summed.

TWO LAYOUTS, DIVERGENT ON PURPOSE
---------------------------------
Layout A and Layout B carry IDENTICAL VALUES and share no field names,
ordering, or structure. Layout A prints fixed 2-decimal amounts with comma
grouping; Layout B prints natural precision with trailing zeros stripped and
no grouping. That second difference is not cosmetic: it means the two layouts
state different PRECISION for the same value, which is exactly what
precision.py infers tolerance from. A reconciler that only works when both
sides print to the cent is not robust, and this makes that failure visible.

SEEDED AND REPRODUCIBLE
-----------------------
Output is a pure function of (generator, seed). data/generated/ is gitignored,
so the seed is the artifact — record it wherever the corpus is referenced or
"run the generator" is underspecified for anyone reproducing our numbers.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from schemas import (
    SCHEMA_VERSION,
    AnswerKey,
    CanonicalDoc,
    ChainKey,
    DocType,
    ExtractedRecord,
    ExtractionMeta,
    Layout,
    LineItem,
    Tier,
)

CENT = Decimal("0.01")


def q(x: Decimal) -> Decimal:
    """Quantise to cents. Every monetary value passes through here exactly
    once, at the point it is decided — never twice, never mid-identity."""
    return x.quantize(CENT)


# --- catalogue ---------------------------------------------------------------

PARTIES = [
    "Northwind Traders", "Contoso Manufacturing", "Fabrikam Industrial",
    "Tailspin Logistics", "Litware Components", "Adventure Works Supply",
]

PRODUCTS = [
    ("Hydraulic valve assembly", "EA", Decimal("1450.00"), Decimal("2200.00")),
    ("Stainless bracket, 40mm", "EA", Decimal("12.50"), Decimal("48.00")),
    ("Industrial coolant", "L", Decimal("8.20"), Decimal("19.75")),
    ("Bearing housing, cast", "EA", Decimal("310.00"), Decimal("890.00")),
    ("Installation labour", "HUR", Decimal("95.00"), Decimal("180.00")),
    ("Conveyor belt section", "M", Decimal("64.00"), Decimal("155.00")),
    ("Control panel, 8-channel", "EA", Decimal("2100.00"), Decimal("3400.00")),
    ("Calibration service", "HUR", Decimal("120.00"), Decimal("240.00")),
]

TAX_RATES = [Decimal("0.10"), Decimal("0.0825"), Decimal("0.20")]
FREIGHT = [Decimal("45.00"), Decimal("75.00"), Decimal("120.00"), Decimal("310.00")]
TERMS = ["Net 30", "Net 45", "Net 30, 2% 10", "Due on receipt", "Net 60"]


@dataclass
class ChainBundle:
    chain_id: str
    docs: list[CanonicalDoc]      # source_text intentionally empty until render
    key: ChainKey


def _lines(rng: random.Random) -> list[LineItem]:
    """Priced line items. Quantities are sometimes fractional because labour
    and bulk materials genuinely are, and an integer-only generator would
    hide a whole class of extraction error."""
    out: list[LineItem] = []
    for i, (desc, uom, lo, hi) in enumerate(
        rng.sample(PRODUCTS, rng.randint(2, 4)), start=1
    ):
        if uom in ("HUR", "L", "M"):
            qty = Decimal(str(rng.choice([1, 1.5, 2, 2.5, 3, 4, 7.5, 12])))
        else:
            qty = Decimal(rng.randint(1, 40))
        span = hi - lo
        price = q(lo + (span * Decimal(rng.randint(0, 100)) / 100))
        out.append(LineItem(
            line_id=f"LI-{i:03d}",
            description=desc,
            quantity=qty,
            unit_of_measure=uom,
            unit_price=price,
            line_total=q(qty * price),
        ))
    return out


def generate_chain(chain_index: int, seed: int,
                   force_allowance: bool = False) -> ChainBundle:
    """One deal, six documents, arithmetically airtight."""
    rng = random.Random(seed * 100_000 + chain_index)
    chain_id = f"C-{chain_index:04d}"
    party = rng.choice(PARTIES)
    terms = rng.choice(TERMS)
    base = date(2026, 1, 1) + timedelta(days=rng.randint(0, 150))

    items = _lines(rng)
    subtotal = q(sum((li.line_total for li in items), Decimal(0)))

    # STRATIFIED, not sampled. Independent 35%/40% coin flips left the
    # allowance-only combination at 1 chain in 20 on seed 1337 and made the
    # both-terms BR-CO-13 path a coin toss. Round-robin guarantees all four
    # combos every 4 chains; magnitudes stay random (that is where variance
    # belongs). force_allowance lets the injector demand an allowance for
    # unapplied_discount chains.
    has_allowance = force_allowance or chain_index % 4 in (2, 3)
    has_charge = chain_index % 4 in (1, 3)
    allowance = q(subtotal * Decimal(rng.randint(3, 12)) / 100) if has_allowance else None
    charge = rng.choice(FREIGHT) if has_charge else None

    net = subtotal - (allowance or Decimal(0)) + (charge or Decimal(0))
    total_excl_tax = q(net)
    tax = q(total_excl_tax * rng.choice(TAX_RATES))
    total = q(total_excl_tax + tax)

    n = rng.randint(1000, 9999)
    num = {
        DocType.QUOTE: f"Q-{n}",
        DocType.SALES_ORDER: f"SO-{n}",
        DocType.PURCHASE_ORDER: f"PO-{n}-A",
        DocType.GOODS_RECEIPT: f"GRN-{n}",
        DocType.INVOICE: f"INV-{n}",
        DocType.PAYMENT: f"PAY-{n}",
    }
    offset = {
        DocType.QUOTE: 0, DocType.SALES_ORDER: 2, DocType.PURCHASE_ORDER: 3,
        DocType.GOODS_RECEIPT: 10, DocType.INVOICE: 11, DocType.PAYMENT: 41,
    }

    def doc(dt: DocType, seq: int, **kw) -> CanonicalDoc:
        return CanonicalDoc(
            doc_id=f"D-{chain_index:04d}-{seq}",
            doc_number=num[dt],
            doc_type=dt,
            party_name=party,
            doc_date=base + timedelta(days=offset[dt]),
            currency="USD",
            source_text="",              # filled by render()
            **kw,
        )

    priced = dict(
        line_items=items, subtotal=subtotal, allowance_total=allowance,
        charge_total=charge, total_excl_tax=total_excl_tax, tax=tax,
        total=total, payment_terms=terms,
    )

    # A goods receipt records WHAT ARRIVED, not what it cost. Quantities only,
    # prices absent — the reason LineItem prices became Optional in v1.2.
    received = [
        LineItem(line_id=li.line_id, description=li.description,
                 quantity=li.quantity, unit_of_measure=li.unit_of_measure)
        for li in items
    ]

    docs = [
        doc(DocType.QUOTE, 1, references=[], **priced),
        doc(DocType.SALES_ORDER, 2, references=[num[DocType.QUOTE]], **priced),
        doc(DocType.PURCHASE_ORDER, 3, references=[num[DocType.QUOTE]], **priced),
        doc(DocType.GOODS_RECEIPT, 4, line_items=received,
            references=[num[DocType.PURCHASE_ORDER]]),
        doc(DocType.INVOICE, 5,
            references=[num[DocType.PURCHASE_ORDER], num[DocType.GOODS_RECEIPT]],
            amount_due=total, **priced),
        # A remittance settles an amount; it has no lines and no tax breakdown.
        doc(DocType.PAYMENT, 6, total=total,
            references=[num[DocType.INVOICE]]),
    ]

    key = ChainKey(
        chain_id=chain_id,
        doc_ids=[d.doc_id for d in docs],
        anomalies=[],                    # CLEAN. Injection is A3.
        generator_seed=seed,
        layouts_emitted=[Layout.A, Layout.B],
    )
    return ChainBundle(chain_id=chain_id, docs=docs, key=key)


# --- rendering ---------------------------------------------------------------

def _a(d: Decimal | None) -> str:
    return "" if d is None else f"{d:,.2f}"


def _b(d: Decimal | None) -> str:
    """Natural precision: trailing zeros stripped, no grouping. A whole-dollar
    amount prints as '4500', which states unit precision and legitimately
    widens the inferred tolerance."""
    if d is None:
        return ""
    s = format(d, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def render_a(doc: CanonicalDoc) -> str:
    """Layout A — header block, pipe table, totals last. Fixed 2dp, grouped."""
    L = [
        f"{doc.doc_type.replace('_', ' ').upper()}",
        "=" * 46,
        f"Document No. : {doc.doc_number}",
        f"Date         : {doc.doc_date.strftime('%d %b %Y')}",
        f"Customer     : {doc.party_name}",
        f"Currency     : {doc.currency}",
    ]
    if doc.references:
        L.append(f"References   : {', '.join(doc.references)}")
    if doc.payment_terms:
        L.append(f"Terms        : {doc.payment_terms}")
    if doc.line_items:
        L += ["", "Description                     | Qty | UOM | Unit Price | Amount",
              "-" * 66]
        for li in doc.line_items:
            L.append(
                f"{li.description:<31}| {_b(li.quantity):>3} | {li.unit_of_measure or '':<3} "
                f"| {_a(li.unit_price):>10} | {_a(li.line_total):>10}")
    L.append("")
    for label, val in [("Subtotal", doc.subtotal), ("Discount", doc.allowance_total),
                       ("Freight", doc.charge_total), ("Net Amount", doc.total_excl_tax),
                       ("Tax", doc.tax), ("Grand Total", doc.total),
                       ("Amount Payable", doc.amount_due)]:
        if val is not None:
            L.append(f"{label:<16}: {_a(val):>12}")
    return "\n".join(L) + "\n"


def render_b(doc: CanonicalDoc) -> str:
    """Layout B — totals FIRST, inline line sentences, different labels,
    natural precision. A regex cannot turn this into Layout A."""
    L = [f"<<{doc.doc_type.replace('_', '-')}>>",
         f"ref={doc.doc_number}",
         f"issued={doc.doc_date.isoformat()}",
         f"counterparty={doc.party_name}",
         f"ccy={doc.currency}"]
    if doc.references:
        L.append("links=" + "|".join(doc.references))
    if doc.total is not None:
        L.append(f"TOTAL PAYABLE {_b(doc.total)}")
    if doc.tax is not None:
        L.append(f"  tax component {_b(doc.tax)}")
    if doc.total_excl_tax is not None:
        L.append(f"  pre-tax {_b(doc.total_excl_tax)}")
    if doc.charge_total is not None:
        L.append(f"  surcharge {_b(doc.charge_total)}")
    if doc.allowance_total is not None:
        L.append(f"  allowance applied {_b(doc.allowance_total)}")
    if doc.subtotal is not None:
        L.append(f"  goods value {_b(doc.subtotal)}")
    if doc.amount_due is not None:
        L.append(f"settlement due {_b(doc.amount_due)}")
    if doc.payment_terms:
        L.append(f"payment_conditions::{doc.payment_terms}")
    if doc.line_items:
        L.append("items{")
        for li in doc.line_items:
            tail = ("" if li.unit_price is None
                    else f" @ {_b(li.unit_price)} => {_b(li.line_total)}")
            L.append(f"  [{li.line_id}] {_b(li.quantity)} {li.unit_of_measure or ''} "
                     f"{li.description}{tail}")
        L.append("}")
    return "\n".join(L) + "\n"


RENDERERS = {Layout.A: render_a, Layout.B: render_b}

#: How each layout PRINTS the date. Ground truth must agree with the page:
#: the document shows a date, so doc_date_raw is not None. Without this the
#: date check is never exercised by the corpus, and extraction is penalised
#: for correctly reading a date that the answer key claims is absent.
RAW_DATE = {
    Layout.A: lambda d: d.strftime("%d %b %Y"),   # 02 May 2026
    Layout.B: lambda d: d.isoformat(),            # 2026-05-02
}


def materialise(bundle: ChainBundle, layout: Layout) -> list[CanonicalDoc]:
    """Same semantic documents, rendered into one layout, source_text filled."""
    render = RENDERERS[layout]
    raw = RAW_DATE[layout]
    return [d.model_copy(update={"source_text": render(d),
                                 "doc_date_raw": raw(d.doc_date)})
            for d in bundle.docs]


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate clean quote-to-cash chains.")
    p.add_argument("--n", type=int, default=20, help="number of chains")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=Path, default=Path("data/generated"))
    p.add_argument("--layouts", default="layout_a,layout_b")
    p.add_argument("--clean", action="store_true",
                   help="skip anomaly injection (A1 behaviour)")
    a = p.parse_args()

    layouts = [Layout(x.strip()) for x in a.layouts.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    records, chains = [], []

    from inject import inject_for_slot, needs_allowance, slot_for

    for i in range(a.n):
        bundle = generate_chain(i, a.seed,
                                force_allowance=needs_allowance(i))
        if not a.clean:
            rng = random.Random(a.seed * 7_000_003 + i)
            docs, anomalies = inject_for_slot(slot_for(i), bundle.docs, rng)
            bundle = ChainBundle(
                chain_id=bundle.chain_id, docs=docs,
                key=bundle.key.model_copy(update={
                    "anomalies": anomalies,
                    "doc_ids": [d.doc_id for d in docs]}))
        chains.append(bundle.key)
        cdir = a.out / bundle.chain_id
        cdir.mkdir(exist_ok=True)
        for layout in layouts:
            for doc in materialise(bundle, layout):
                (cdir / f"{doc.doc_id}.{layout.value}.txt").write_text(
                    doc.source_text, encoding="utf-8")
                records.append(ExtractedRecord(
                    doc=doc,
                    meta=ExtractionMeta(tier=Tier.NONE, layout=layout,
                                        model_id="generator",
                                        schema_version=SCHEMA_VERSION)))

    (a.out / "records.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in records) + "\n", encoding="utf-8")
    (a.out / "answer_key.json").write_text(
        AnswerKey(chains=chains).model_dump_json(indent=2), encoding="utf-8")

    print(f"{a.n} chains, {len(records)} documents -> {a.out}  (seed {a.seed})")


if __name__ == "__main__":
    main()
