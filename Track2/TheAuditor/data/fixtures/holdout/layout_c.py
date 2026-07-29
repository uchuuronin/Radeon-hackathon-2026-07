"""LAYOUT C — THE SEALED HOLDOUT. Do not import from src/, do not open
during development, do not add to RENDERERS, do not run the verifier on its
output before demo day.

Written once on Day 2 and committed unread by the rest of the codebase. The
point is evidential: at the demo, extraction meets a layout that no prompt,
no test and no tolerance decision has ever seen. Every day of development
knowledge makes writing this less honest, which is why it happens today.

Structural choices, deliberately unlike A and B: narrative sentence order,
amounts written as "USD 4,500.00 exactly" mid-sentence, date as "2026, May
14", labels drawn from legal boilerplate (consideration, levy, settlement),
line items as numbered clauses.
"""

from __future__ import annotations

MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _amt(d):
    return "" if d is None else f"USD {d:,.2f} exactly"


def render_c(doc) -> str:
    dt = doc.doc_date
    L = [f"MEMORANDUM OF {doc.doc_type.replace('_', ' ').upper()}",
         f"entered on {dt.year}, {MONTHS[dt.month]} {dt.day}, "
         f"between the undersigned and {doc.party_name}.",
         f"This memorandum bears the designation {doc.doc_number}."]
    if doc.references:
        L.append("It is made pursuant to " + " and ".join(doc.references) + ".")
    if doc.line_items:
        L.append("The parties record the following particulars:")
        for n, li in enumerate(doc.line_items, 1):
            head = (f"  Clause {n}. {li.quantity} {li.unit_of_measure or 'unit'}"
                    f" of {li.description}")
            if li.unit_price is not None:
                head += (f", at a consideration of {_amt(li.unit_price)} per "
                         f"unit, amounting to {_amt(li.line_total)}")
            L.append(head + ".")
    for label, v in [("The aggregate consideration is", doc.subtotal),
                     ("An abatement is allowed of", doc.allowance_total),
                     ("Carriage is charged at", doc.charge_total),
                     ("The sum before levy stands at", doc.total_excl_tax),
                     ("A levy applies of", doc.tax),
                     ("The whole sum payable is", doc.total),
                     ("Settlement is due in the amount of", doc.amount_due)]:
        if v is not None:
            L.append(f"{label} {_amt(v)}.")
    if doc.payment_terms:
        L.append(f"Settlement terms: {doc.payment_terms}.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import random
    import sys
    from pathlib import Path
    sys.path[:0] = ["src", "data/generator"]
    from gen import generate_chain
    from inject import inject_for_slot, needs_allowance, slot_for

    out = Path(__file__).parent / "docs"
    out.mkdir(exist_ok=True)
    seed = 1337
    for i in range(5):
        b = generate_chain(i, seed, force_allowance=needs_allowance(i))
        rng = random.Random(seed * 7_000_003 + i)
        docs, _ = inject_for_slot(slot_for(i), b.docs, rng)
        for d in docs:
            (out / f"{d.doc_id}.layout_c.txt").write_text(
                render_c(d), encoding="utf-8")
    print(f"sealed: {sum(1 for _ in out.iterdir())} documents")
