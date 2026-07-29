"""Deterministic normalisation for the two non-numeric checks.

Zero dependencies. Both functions are the ESCALATION LADDER applied at
micro-scale: try the cheap exact thing, fall back to the slightly less cheap
thing, and report honest uncertainty rather than manufacturing a verdict.

WHY NOT dateparser AS THE PRIMARY PARSER
----------------------------------------
dateparser is the best general-purpose date library available, but its
published numbers disqualify it from rung 0: roughly 400 ms per call with
language autodetection (about 0.25 ms when pinned to one language), 0.5-1 s
of import time, and a documented degradation in long-running processes where
parse times climb into the seconds. Rung 0's whole claim is that it is free.
Spending 400 ms per document to check a date would destroy the claim the
architecture is built on.

So: a fixed format table handles the layouts we can enumerate, in
microseconds, with no dependency. dateparser is available as an OPTIONAL
fallback for the residue, off by default, and callers who enable it should
pin languages=["en"] for the 1600x speedup. That is the same
cheapest-rung-first discipline as the rest of the system.

WHY SUFFIX DIFFERENCES ARE NOT A PASS
-------------------------------------
Company-name normalisation guidance is consistent about stripping legal
suffixes for CRM matching and deduplication — and equally consistent that the
exception is financial, contractual and KYC/KYB contexts, where the rule is
"normalise for matching, preserve for compliance". We are that exception.
"Acme Inc" and "Acme LLC" may be different legal entities, and lookalike
vendor names are a live invoice-fraud vector, so a suffix difference must
SURFACE (WITHIN_TOLERANCE, both raw names retained) and must never silently
pass. Stripping is used to decide how loudly to complain, never to decide
that two parties are the same.
"""

from __future__ import annotations

import re
from datetime import date
from typing import NamedTuple, Optional, Sequence

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

#: Formats observed across the ten market-software fixtures, plus the obvious
#: neighbours. Unambiguous formats first; the ambiguous numeric ones are
#: handled separately because order cannot be inferred from the string alone.
_UNAMBIGUOUS = (
    "%Y-%m-%d",        # 2026-05-02   ISO / Layout B
    "%d %b %Y",        # 2 Apr 2026   Xero
    "%d %B %Y",        # 2 April 2026
    "%b %d, %Y",       # Jul 1, 2026  Stripe
    "%B %d, %Y",       # January 30, 2026  Wave
    "%d-%b-%Y",        # 02-Apr-2026
    "%Y/%m/%d",        # 2026/05/02
)

#: Purely numeric formats. The SAME STRING can be a valid date under more than
#: one of these, which is a property of the world, not a defect in the parser.
_NUMERIC = (
    ("%m/%d/%Y", "MDY"), ("%d/%m/%Y", "DMY"),
    ("%m-%d-%Y", "MDY"), ("%d-%m-%Y", "DMY"),
    ("%d.%m.%Y", "DMY"), ("%m.%d.%Y", "MDY"),   # SAP prints 27.03.2026
    ("%m/%d/%y", "MDY"), ("%d/%m/%y", "DMY"),
)


class DateReading(NamedTuple):
    """What a raw date string can legitimately mean.

    `candidates` holds every date the string could denote. One candidate means
    the string is unambiguous. Two means the document is genuinely ambiguous
    and only knowledge of the vendor can settle it — which is why DATE_ORDER
    is a per-party setting, exactly like tolerance.
    """
    candidates: tuple[date, ...]
    ambiguous: bool
    orders: tuple[str, ...] = ()          # parallel to candidates: MDY / DMY

    @property
    def unique(self) -> Optional[date]:
        return self.candidates[0] if len(self.candidates) == 1 else None


def read_date(raw: str, date_order: Optional[str] = None) -> DateReading:
    """Parse a printed date string into every reading it supports.

    `date_order` ("MDY" or "DMY") is the per-party override. Supplying it
    collapses ambiguity to a single reading; omitting it reports the ambiguity
    honestly rather than guessing. dateparser's own default is MDY for
    English, which would silently misread a UK/EU document — we would rather
    say "ambiguous" than be confidently wrong.
    """
    import datetime as _dt

    s = " ".join(raw.strip().split())
    for fmt in _UNAMBIGUOUS:
        try:
            return DateReading((_dt.datetime.strptime(s, fmt).date(),), False)
        except ValueError:
            continue

    seen: dict[date, str] = {}
    for fmt, order in _NUMERIC:
        try:
            d = _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
        if date_order and order != date_order:
            continue
        seen.setdefault(d, order)

    if not seen:
        return DateReading((), False)
    cands = tuple(seen.keys())
    return DateReading(cands, len(cands) > 1, tuple(seen.values()))


# ---------------------------------------------------------------------------
# Party names
# ---------------------------------------------------------------------------

#: Legal-form designators. Deliberately NOT used to decide that two parties
#: are the same — only to decide whether a mismatch is a suffix difference
#: (surfaced, reviewable) or a different company entirely (a failure).
_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "l.l.c", "llp", "lp", "ltd", "limited", "plc",
    "gmbh", "ag", "kg", "mbh", "sa", "sas", "sarl", "srl", "spa",
    "bv", "nv", "ab", "as", "oy", "pty", "pte", "kk", "kft", "zrt",
}

#: Names where the legal-form word IS the brand. Over-normalisation damages
#: data as badly as none: "The Limited" is not "The". Short by design; grows
#: only when a real case appears.
_EXCEPTIONS = {"the limited", "the corporation"}

_PUNCT = re.compile(r"[^\w\s&]")
_WS = re.compile(r"\s+")


def normalise_party(name: str) -> str:
    """Casefold, collapse whitespace, drop punctuation. NON-DESTRUCTIVE: the
    legal form survives, so this alone never merges two entities."""
    s = _PUNCT.sub(" ", name).casefold()
    return _WS.sub(" ", s).strip()


def strip_legal_form(name: str) -> str:
    """Remove trailing legal designators. Used ONLY to classify the KIND of
    mismatch, never to assert identity."""
    base = normalise_party(name)
    if base in _EXCEPTIONS:
        return base
    parts = base.split()
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts) if parts else base


class PartyMatch(NamedTuple):
    exact: bool           # identical after cosmetic normalisation
    same_base: bool       # identical once legal form is set aside
    left: str             # raw, preserved for the audit trail
    right: str


def compare_parties(a: str, b: str) -> PartyMatch:
    """Three-way, matching the verifier's outcome model.

        exact=True                  -> PASS
        exact=False, same_base=True -> WITHIN_TOLERANCE, both names shown.
                                       "Averill Fastener" vs "Averill
                                       Fastener GmbH" is almost always the
                                       same vendor written two ways, and
                                       occasionally is not. A human decides;
                                       the system never decides silently.
        neither                     -> FAIL
    """
    na, nb = normalise_party(a), normalise_party(b)
    return PartyMatch(exact=na == nb,
                      same_base=strip_legal_form(a) == strip_legal_form(b),
                      left=a, right=b)


# ---------------------------------------------------------------------------
# Optional dateparser escape hatch
# ---------------------------------------------------------------------------

def read_date_with_fallback(raw: str, date_order: Optional[str] = None,
                            allow_dateparser: bool = False) -> DateReading:
    """Format table first; dateparser only if explicitly permitted.

    Off by default so rung 0 stays free and the verifier keeps running on
    pydantic alone. Enable per-run when a corpus turns out to contain
    layouts the table does not cover, and pin languages=["en"].
    """
    reading = read_date(raw, date_order)
    if reading.candidates or not allow_dateparser:
        return reading
    try:
        import dateparser                                   # noqa: PLC0415
    except ImportError:
        return reading
    settings = {"STRICT_PARSING": True}
    if date_order:
        settings["DATE_ORDER"] = date_order
    parsed = dateparser.parse(raw, languages=["en"], settings=settings)
    return DateReading((parsed.date(),), False) if parsed else reading


def matches_reading(reading: DateReading, claimed: date) -> Sequence[bool]:
    """(matches_any, was_ambiguous) — the verifier's decision inputs."""
    return (claimed in reading.candidates, reading.ambiguous)
