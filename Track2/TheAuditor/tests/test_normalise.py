"""Unit tests for date and party normalisation."""

from datetime import date
import pytest

from normalise import compare_parties, read_date, strip_legal_form


def test_unambiguous_formats_across_the_market_layouts():
    cases = [("2026-05-02", date(2026, 5, 2)), ("2 Apr 2026", date(2026, 4, 2)),
             ("January 30, 2026", date(2026, 1, 30)),
             ("July 1, 2026", date(2026, 7, 1)),
             ("27.03.2026", date(2026, 3, 27)),      # SAP, 27 cannot be a month
             ("18/02/2026", date(2026, 2, 18)),      # Zoho, 18 cannot be a month
             ("05/14/2026", date(2026, 5, 14))]      # QBO, 14 cannot be a month
    for raw, expected in cases:
        r = read_date(raw)
        assert r.unique == expected, raw
        assert not r.ambiguous


def test_genuinely_ambiguous_dates_report_both_readings():
    r = read_date("06/09/2026")
    assert r.ambiguous
    assert set(r.candidates) == {date(2026, 6, 9), date(2026, 9, 6)}
    assert r.unique is None


def test_party_date_order_collapses_ambiguity():
    assert read_date("06/09/2026", "MDY").unique == date(2026, 6, 9)
    assert read_date("06/09/2026", "DMY").unique == date(2026, 9, 6)


def test_unparseable_returns_no_candidates():
    assert read_date("sometime last autumn").candidates == ()


def test_no_dateparser_import_on_the_fast_path():
    """dateparser costs ~400ms/call with autodetection and 0.5-1s to import.
    Rung 0 must not pay that."""
    import sys
    sys.modules.pop("dateparser", None)
    read_date("2026-05-02")
    assert "dateparser" not in sys.modules


def test_strip_legal_form_handles_common_designators():
    assert strip_legal_form("Averill Fastener GmbH") == "averill fastener"
    assert strip_legal_form("Acme, Inc.") == "acme"
    assert strip_legal_form("Foo Holdings Pty Ltd") == "foo holdings"


def test_exception_list_protects_brands_named_after_a_legal_form():
    """Over-normalisation damages data as badly as none: The Limited is not
    The."""
    assert strip_legal_form("The Limited") == "the limited"


def test_stripping_never_empties_a_name():
    assert strip_legal_form("Limited") == "limited"


def test_compare_parties_three_ways():
    assert compare_parties("Acme Inc", "acme,  inc.").exact
    m = compare_parties("Acme Inc", "Acme LLC")
    assert not m.exact and m.same_base       # DIFFERENT legal entities possible
    assert not compare_parties("Acme", "Acuity").same_base


def test_raw_names_are_preserved_for_the_audit_trail():
    m = compare_parties("Acme Inc", "Acme LLC")
    assert m.left == "Acme Inc" and m.right == "Acme LLC"


# ---------------------------------------------------------------------------
# Tolerance accumulation model
# ---------------------------------------------------------------------------

class TestToleranceAccumulation:
    """Two defensible bounds, and the difference between them is a real
    decision rather than a detail. Both are implemented so a reported number
    can name its model."""

    def test_single_operand_is_identical_under_both(self):
        from precision import inferred_tolerance
        assert (inferred_tolerance(["4500.00"])
                == inferred_tolerance(["4500.00"], mode="linear"))

    def test_rss_is_tighter_than_linear_and_the_gap_grows(self):
        from decimal import Decimal
        from precision import inferred_tolerance
        prev = Decimal(0)
        for n in (2, 4, 9, 16):
            ops = ["1.00"] * n
            rss = inferred_tolerance(ops)
            lin = inferred_tolerance(ops, mode="linear")
            assert rss < lin
            ratio = lin / rss
            assert ratio > prev            # linear pulls away as sqrt(n)
            prev = ratio

    def test_linear_is_the_worst_case_and_rss_is_not_exceeded_in_practice(self):
        """Four cent-rounded values: linear allows 2.2 cents, rss allows 1.1.
        The linear band would absorb a genuine one-cent-per-line error, which
        is why rss is the default."""
        from decimal import Decimal
        from precision import inferred_tolerance
        ops = ["1.00"] * 4
        assert inferred_tolerance(ops, mode="linear") == Decimal("0.0220")
        assert inferred_tolerance(ops) == Decimal("0.0110")

    def test_stated_precision_still_widens_the_band(self):
        """A document printing whole units is asserting less precision and
        must get a wider band under either model."""
        from precision import inferred_tolerance
        assert inferred_tolerance(["4500"]) > inferred_tolerance(["4500.00"])


class TestDateShapeDispatch:
    """Dispatch must not change a single verdict, only the work done."""

    @pytest.mark.parametrize("raw,expect_n,ambiguous", [
        ("2026-05-02", 1, False),
        ("2 Apr 2026", 1, False),
        ("Jul 1, 2026", 1, False),
        ("January 30, 2026", 1, False),
        ("27.03.2026", 1, False),
        ("06/09/2026", 2, True),
        ("02-04-2026", 2, True),
        ("13/09/2026", 1, False),
        ("not a date", 0, False),
        ("", 0, False),
    ])
    def test_readings_are_unchanged(self, raw, expect_n, ambiguous):
        from normalise import read_date
        r = read_date(raw)
        assert len(r.candidates) == expect_n
        assert r.ambiguous is ambiguous

    def test_alpha_month_never_falls_through_to_numeric(self):
        """A written month cannot be an MDY/DMY reading, so the numeric
        family is skipped entirely rather than tried and discarded."""
        from normalise import read_date
        assert read_date("Nonsuch 4, 2026").candidates == ()

    def test_per_party_order_still_collapses_ambiguity(self):
        from normalise import read_date
        assert read_date("06/09/2026", "DMY").unique is not None
        assert read_date("06/09/2026", "MDY").unique is not None
        assert (read_date("06/09/2026", "DMY").unique
                != read_date("06/09/2026", "MDY").unique)
