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
For a SUM of N independently-rounded amounts, Beancount's single-operand
inference is not enough: the accumulated error depends on all N operands.
There are two defensible bounds and we implement BOTH, because they answer
different questions and quoting only one is a misrepresentation:

  LINEAR  sum of half-ULPs.  The WORST CASE, every rounding error aligned in
          sign. The standard interval-arithmetic forward bound. Grows O(N).
  RSS     sqrt(sum of squared half-ULPs). The PROBABILISTIC bound when the
          rounding errors are independent, zero-mean and roughly uniform,
          which is the realistic model for independently-rounded document
          amounts. Grows O(sqrt(N)) and is therefore TIGHTER.

Default is RSS. Linear is a ceiling that, at N=4 operands, is twice as wide
and would silently absorb a real one-cent-per-line error across four lines.
Set mode="linear" where a conservative bound is wanted and say so in the
spec; do not report a number without naming the mode that produced it.

SCOPE CUT (v1, stated deliberately)
-----------------------------------
We assume '.' is the decimal separator and ',' is a thousands separator.
European-format `4.500,00` is NOT supported and raises rather than silently
misparsing — a silent misparse here would corrupt every downstream number.
"""

from __future__ import annotations

import re
from decimal import Decimal
from functools import lru_cache
from typing import Iterable, Literal, Optional

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


#: Accumulation model for multi-operand tolerance. See the header.
ToleranceMode = Literal["rss", "linear"]
DEFAULT_TOLERANCE_MODE: ToleranceMode = "rss"


def _isqrt_decimal(x: Decimal) -> Decimal:
    """Square root of a Decimal without importing math (which would force a
    float round-trip and reintroduce exactly the binary-float imprecision the
    whole project bans). decimal.Decimal.sqrt is correctly rounded under the
    active context, which is what we want."""
    return x.sqrt()


def inferred_tolerance(
    rendered: Iterable[str],
    multiplier: Decimal = DEFAULT_INFERRED_MULTIPLIER,
    mode: ToleranceMode = DEFAULT_TOLERANCE_MODE,
) -> Decimal:
    """Tolerance for an identity over the given rendered operands.

    mode="rss"    sqrt(sum of squared half-ULPs) — the independent-error
                  bound. Default. Tighter, and the honest model for amounts
                  rounded independently by different systems.
    mode="linear" sum of half-ULPs — the worst-case aligned-error ceiling.

    Returns 0 for an empty operand list; callers fall back to their configured
    absolute floor. One operand gives the same answer under both modes, so
    single-amount call sites are unaffected by the default change.
    """
    halves = [half_ulp(r) for r in rendered]
    if not halves:
        return Decimal(0)
    if mode == "linear":
        acc = sum(halves, Decimal(0))
    else:
        acc = _isqrt_decimal(sum((h * h for h in halves), Decimal(0)))
    return acc * multiplier


def find_amounts(text: str) -> set[Decimal]:
    """Every numeric value appearing in a block of document text, as Decimals.

    This is what makes the `amounts_appear_in_source` check layout-robust.
    A raw substring search would fail the moment one layout prints `4,500.00`
    and another prints `4500` for the same value — the extraction is correct
    in both cases and the check must not punish it. Comparing VALUES rather
    than STRINGS keeps the check meaningful without making it layout-specific.
    """
    return {v for v, _ in _multiset_cached(text)}


def count_occurrences(value: Decimal, text: str) -> int:
    """How many numeric tokens in `text` equal `value`.

    Was a full rescan per value, so checking V amounts on one document cost
    O(V x T). The shared cached multiset makes it one scan plus V dict hits.
    """
    return dict(_multiset_cached(text)).get(value, 0)


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
    exempt: Optional[Iterable[Decimal]] = None,
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

    `exempt` is the DERIVABLE set: values the document never printed but which
    follow arithmetically from values it DID print. This closes the check's
    worst false-reject. A layout that prints line items and a grand total but
    no subtotal line is completely ordinary, and the extractor is RIGHT to
    emit the subtotal; failing it for "hallucinating" a number it correctly
    computed would punish exactly the behaviour we want. Membership counting
    can only ever ground VERBATIM values, so derived values need a different
    warrant, and arithmetic derivability from grounded operands is that
    warrant. The caller computes the set because only it knows the identities.
    """
    present = find_amounts_multiset(text)
    explained: dict[Decimal, int] = {}
    for v in non_monetary:
        explained[v] = explained.get(v, 0) + 1

    distinct: dict[Decimal, str] = {}
    for path, value in claimed:
        distinct.setdefault(value, path)

    exempt_set = set(exempt or ())

    missing: list[tuple[str, Decimal]] = []
    for value, path in distinct.items():
        if value in exempt_set:
            continue
        if present.get(value, 0) <= explained.get(value, 0):
            missing.append((path, value))
    return missing


@lru_cache(maxsize=512)
def _multiset_cached(text: str) -> tuple[tuple[Decimal, int], ...]:
    """Tokenise once per distinct source_text.

    WHY THIS IS CACHED: Mechanism B samples the SAME document N times for
    self-consistency, and every sample re-runs the verifier against a
    byte-identical source_text. Without a cache the tokenising regex runs
    N times over the same kilobyte for zero new information; the run is
    O(N x T) where it should be O(T). Documents are ~1 KB, so 512 entries is
    well under a megabyte. Returns a tuple because lru_cache requires a
    hashable return the caller cannot mutate.
    """
    out: dict[Decimal, int] = {}
    for token in _AMOUNT_TOKEN.findall(text):
        try:
            v = parse_amount(token)
        except AmbiguousAmountError:
            continue
        out[v] = out.get(v, 0) + 1
    return tuple(out.items())


def find_amounts_multiset(text: str) -> dict[Decimal, int]:
    """Every numeric value in `text` with its occurrence COUNT.

    find_amounts() collapses duplicates into a set, which loses exactly the
    information the counting argument needs.
    """
    return dict(_multiset_cached(text))
