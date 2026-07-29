"""Unit tests for date and party normalisation."""

from datetime import date

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
