"""Precision inference — deriving tolerance from what a document actually wrote.

WHY THIS EXISTS
---------------
A fixed tolerance is wrong in both directions. A document that prints
`4,500.00` is asserting cent precision; a band of +/-0.50 there would swallow
a real 40-cent error. A document that prints `4,500` is asserting unit
precision; a band of +/-0.01 there will flag legitimate rounding on every
single line.

Beancount hit this exact wall. Its original implementation used a global
constant tolerance and its author describes that approach as weak and a
kludge; the replacement infers tolerance from the number of digits the user
actually wrote, per currency, per transaction, in isolation. We have strictly
more information available than Beancount does, because `source_text` is
byte-preserved: we can see the rendered form of every amount.

WHERE WE DIVERGE, AND WHY
-------------------------
For a SUM of N independently-rounded amounts, the worst-case accumulated
error is the sum of the half-ULPs of the operands, not the max. Summing ten
values each rounded to the cent can legitimately drift 5 cents from an exact
computation. So `inferred_tolerance` sums half-ULPs across all operands
rather than taking a single one. This is arithmetic, not a citation.

SCOPE CUT (v1, stated deliberately)
-----------------------------------
We assume '.' is the decimal separator and ',' is a thousands separator.
European-format `4.500,00` is NOT supported and raises rather than silently
misparsing — a silent misparse here would corrupt every downstream number.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Iterable

# Default buffer on inferred tolerances. Beancount exposes the same idea as
# `inferred_tolerance_multiplier`; 1.1 is its documented recommendation.
DEFAULT_INFERRED_MULTIPLIER = Decimal("1.1")

# A number token inside free document text: optional sign, digits with
# optional thousands separators, optional fractional part.
_AMOUNT_TOKEN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")

# Characters we strip before parsing: currency symbols, spaces, codes.
_STRIP = re.compile(r"[^\d.,\-]")


class AmbiguousAmountError(ValueError):
    """Raised when a rendered amount cannot be parsed unambiguously under the
    stated separator convention. Loud failure beats a silent misparse."""


def normalise(rendered: str) -> str:
    """Strip currency symbols and thousands separators, leaving a plain
    decimal string. Raises rather than guessing on ambiguous input."""
    s = _STRIP.sub("", rendered).strip()
    if not s:
        raise AmbiguousAmountError(f"no numeric content in {rendered!r}")
    if s.count(".") > 1:
        raise AmbiguousAmountError(
            f"{rendered!r} has multiple '.' — European formatting is out of "
            f"scope in v1; see the scope cut in precision.py")
    # A comma AFTER the decimal point means the convention is inverted.
    if "." in s and "," in s and s.index(",") > s.index("."):
        raise AmbiguousAmountError(
            f"{rendered!r} appears to use ',' as a decimal separator")
    return s.replace(",", "")


def parse_amount(rendered: str) -> Decimal:
    """Rendered string -> Decimal, preserving the written precision.
    `Decimal("4500.00")` and `Decimal("4500")` are equal in value but carry
    different exponents, and we rely on that distinction."""
    return Decimal(normalise(rendered))


def decimal_places(rendered: str) -> int:
    """How many fractional digits the document actually printed."""
    s = normalise(rendered)
    return len(s.split(".")[1]) if "." in s else 0


def ulp(rendered: str) -> Decimal:
    """Unit in the last place — the magnitude of the least significant digit
    the document committed to. '4500' -> 1, '4,500.00' -> 0.01."""
    return Decimal(1).scaleb(-decimal_places(rendered))


def half_ulp(rendered: str) -> Decimal:
    """Maximum rounding error implied by a single written amount."""
    return ulp(rendered) / 2


def inferred_tolerance(
    rendered: Iterable[str],
    multiplier: Decimal = DEFAULT_INFERRED_MULTIPLIER,
) -> Decimal:
    """Tolerance for an identity over the given rendered operands.

    Sum of half-ULPs, scaled by the buffer multiplier. Returns 0 for an empty
    operand list — callers fall back to their configured absolute floor.
    """
    total = sum((half_ulp(r) for r in rendered), Decimal(0))
    return total * multiplier


def find_amounts(text: str) -> set[Decimal]:
    """Every numeric value appearing in a block of document text, as Decimals.

    This is what makes the `amounts_appear_in_source` check layout-robust.
    A raw substring search would fail the moment one layout prints `4,500.00`
    and another prints `4500` for the same value — the extraction is correct
    in both cases and the check must not punish it. Comparing VALUES rather
    than STRINGS keeps the check meaningful without making it layout-specific.
    """
    out: set[Decimal] = set()
    for token in _AMOUNT_TOKEN.findall(text):
        try:
            out.add(parse_amount(token))
        except AmbiguousAmountError:
            continue
    return out


def count_occurrences(value: Decimal, text: str) -> int:
    """How many numeric tokens in `text` equal `value`."""
    return sum(1 for v in find_amounts(text) if v == value)


def amount_in_source(value: Decimal, source_text: str) -> bool:
    """Value-equality membership test, tolerant of rendering differences.

    NOTE: membership alone is too weak to ground a monetary claim — a
    hallucinated total of 500.00 "appears" in a document that prints
    `Qty 500`. Use unexplained_claims() for the real check; this remains for
    callers that only need presence.
    """
    return count_occurrences(value, source_text) > 0


def unexplained_claims(
    claimed: Iterable[tuple[str, Decimal]],
    non_monetary: Iterable[Decimal],
    text: str,
) -> list[tuple[str, Decimal]]:
    """Monetary claims with no independent evidence in the source.

    THE COUNTING ARGUMENT
    ---------------------
    A hallucinated amount often coincides with a number that is genuinely on
    the page for another reason — a quantity, a year, a line number. Asking
    "does this value appear?" cannot tell the two apart. Asking "does it
    appear MORE OFTEN than the non-monetary fields already account for?" can.

        hallucinated total 500.00, document prints only `Qty 500`
            occurrences 1, explained by quantity 1  ->  0 left  ->  FAIL
        genuine total, document prints `Qty 500 ... Total 500.00`
            occurrences 2, explained by quantity 1  ->  1 left  ->  PASS

    `non_monetary` is the values we already know are on the page for
    non-monetary reasons: line quantities, and the components of the document
    date. Every one of them is taken from the extracted record itself, so no
    assumption is made about layout, currency symbols or label wording — the
    reason the "look for a nearby $" approach was rejected.

    Claims are de-duplicated BY VALUE, not by field: a Coupa-style purchase
    order where subtotal, net and total are all 11100 prints that figure once
    and is perfectly correct.
    """
    present = find_amounts_multiset(text)
    explained: dict[Decimal, int] = {}
    for v in non_monetary:
        explained[v] = explained.get(v, 0) + 1

    distinct: dict[Decimal, str] = {}
    for path, value in claimed:
        distinct.setdefault(value, path)

    missing: list[tuple[str, Decimal]] = []
    for value, path in distinct.items():
        if present.get(value, 0) <= explained.get(value, 0):
            missing.append((path, value))
    return missing


def find_amounts_multiset(text: str) -> dict[Decimal, int]:
    """Every numeric value in `text` with its occurrence COUNT.

    find_amounts() collapses duplicates into a set, which loses exactly the
    information the counting argument needs.
    """
    out: dict[Decimal, int] = {}
    for token in _AMOUNT_TOKEN.findall(text):
        try:
            v = parse_amount(token)
        except AmbiguousAmountError:
            continue
        out[v] = out.get(v, 0) + 1
    return out
