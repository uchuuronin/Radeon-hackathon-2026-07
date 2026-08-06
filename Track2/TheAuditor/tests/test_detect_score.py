"""Tests for the detection oracle, written before the reconciler.

The oracle's job is to be hard to satisfy. These tests exist to prove it is,
because a scorer nobody has fed a wrong answer to is an untested claim about
measurement, and every headline detection number will rest on it.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bench.detect_score import (DetectionScore, TIERS, _tier_of,  # noqa: E402
                                score, wilson)
from schemas import (AnomalyType, AnswerKey, ChainKey, ChainVerdict,  # noqa: E402
                     Discrepancy, PlantedAnomaly)


def _planted(**kw) -> PlantedAnomaly:
    base = dict(anomaly_type=AnomalyType.PRICE_DRIFT,
                doc_ids_involved=["D-1", "D-2"],
                field_path="line_items[LI-001].unit_price",
                expected_delta=Decimal("42.00"), is_within_tolerance=False,
                note="")
    base.update(kw)
    return PlantedAnomaly(**base)


def _reported(**kw) -> Discrepancy:
    base = dict(anomaly_type=AnomalyType.PRICE_DRIFT,
                doc_ids_involved=["D-1", "D-2"],
                field_path="line_items[LI-001].unit_price",
                delta=Decimal("42.00"))
    base.update(kw)
    return Discrepancy(**base)


def _key(anoms) -> AnswerKey:
    return AnswerKey(chains=[ChainKey(
        chain_id="C-1", doc_ids=["D-1", "D-2"], anomalies=list(anoms),
        generator_seed=1, layouts_emitted=[])])


def _verdict(discs) -> ChainVerdict:
    return ChainVerdict(chain_id="CH-D-1", doc_ids=["D-1", "D-2"],
                        discrepancies=list(discs))


class TestTheOracleIsHardToSatisfy:

    def test_a_hit_needs_the_right_pair_and_the_right_field(self):
        s = score([_verdict([_reported()])], _key([_planted()]), {})
        assert (s.tp, s.fp, s.fn) == (1, 0, 0)

    def test_right_field_wrong_documents_is_not_a_hit(self):
        """Attribution is the product. Naming the wrong two documents sends an
        analyst to the wrong place, so it is a miss AND a false positive, not a
        partial credit."""
        s = score([_verdict([_reported(doc_ids_involved=["D-1", "D-9"])])],
                  _key([_planted()]), {})
        assert (s.tp, s.fn) == (0, 1) and s.fp == 1

    def test_right_documents_wrong_field_is_not_a_hit(self):
        s = score([_verdict([_reported(field_path="total")])],
                  _key([_planted()]), {})
        assert (s.tp, s.fn) == (0, 1) and s.fp == 1

    def test_flagging_a_decoy_is_a_false_positive(self):
        """130 of the 210 planted price drifts are decoys whose line value
        genuinely differs. Flagging one is the failure the corpus exists to
        provoke."""
        s = score([_verdict([_reported()])],
                  _key([_planted(is_within_tolerance=True)]), {})
        assert s.fp == 1 and s.decoys_flagged == 1 and s.tp == 0

    def test_leaving_a_decoy_alone_costs_nothing(self):
        s = score([_verdict([])], _key([_planted(is_within_tolerance=True)]), {})
        assert (s.tp, s.fp, s.fn) == (0, 0, 0)
        assert s.decoys_total == 1 and s.decoys_flagged == 0

    def test_flagging_a_clean_chain_is_counted_separately(self):
        """150 chains are clean by construction. A reconciler that emits
        nothing for them is indistinguishable from one that crashed, so clean
        chains are counted rather than ignored."""
        s = score([_verdict([_reported()])], _key([]), {})
        assert s.clean_chains_total == 1 and s.clean_chains_flagged == 1
        assert s.fp == 1


class TestPartialCreditIsReportedNotAwarded:

    def test_misclassifying_a_located_drift_is_reported_not_a_miss(self):
        """Found the right drift, called it the wrong thing. An analyst is
        looking at the right two documents and the right field, so this is a
        lesser failure than missing it, and collapsing the two would hide which
        is happening."""
        s = score([_verdict([_reported(anomaly_type=AnomalyType.TERM_CHANGE)])],
                  _key([_planted()]), {})
        assert s.tp == 1 and s.mistyped == 1

    def test_a_wrong_magnitude_on_a_located_drift_is_reported(self):
        s = score([_verdict([_reported(delta=Decimal("1.00"))])],
                  _key([_planted()]), {})
        assert s.tp == 1 and s.wrong_delta == 1

    def test_a_non_numeric_anomaly_never_counts_as_a_wrong_delta(self):
        """term_change carries expected_delta None. Comparing against it would
        manufacture a failure on every single one."""
        s = score([_verdict([_reported(anomaly_type=AnomalyType.TERM_CHANGE,
                                       field_path="payment_terms", delta=None)])],
                  _key([_planted(anomaly_type=AnomalyType.TERM_CHANGE,
                                 field_path="payment_terms",
                                 expected_delta=None)]), {})
        assert s.tp == 1 and s.wrong_delta == 0


class TestReportingShape:

    def test_rates_carry_wilson_intervals(self):
        """Never a bare percentage: 3 correct out of 3 is not 100%."""
        p, lo, hi = wilson(3, 3)
        assert p == 1.0 and lo < 1.0
        assert wilson(0, 0) == (0.0, 0.0, 1.0)

    def test_tiers_partition_the_ratio_line(self):
        assert _tier_of(0.0) == TIERS[0][0]
        assert "boundary" in _tier_of(1.0)
        assert _tier_of(99.0) == TIERS[-1][0]
        assert _tier_of(None) is None

    def test_render_refuses_to_average_the_types(self):
        s = DetectionScore()
        s.by_type["price_drift"] = [80, 80]
        s.by_type["near_duplicate"] = [0, 30]
        out = s.render()
        assert "price_drift" in out and "near_duplicate" in out
        assert "never averaged" in out


class TestTheNaiveBaselineFailsAsPredicted:
    """The baseline is not a straw man. It is what a reasonable person writes
    first, and the corpus is built to expose exactly it."""

    def test_a_field_by_field_comparator_flags_every_decoy(self):
        import json
        from bench.detect_score import load_corpus, naive_field_comparator
        recs = ROOT / "data/generated/records.jsonl"
        keyp = ROOT / "data/generated/answer_key.json"
        if not recs.exists():                     # corpus is gitignored
            import pytest
            pytest.skip("run data/generator/gen.py first")
        docs, key = load_corpus(recs, keyp)
        verdicts = [naive_field_comparator([docs[i] for i in c.doc_ids
                                            if i in docs])
                    for c in key.chains if any(i in docs for i in c.doc_ids)]
        s = score(verdicts, key, docs)
        # Perfect recall on the line-level numeric classes...
        assert s.by_type["price_drift"][0] == s.by_type["price_drift"][1]
        # ...bought with every single decoy.
        assert s.decoys_flagged == s.decoys_total == 130
        assert s.precision[0] < 0.25
        # And blind to everything without a line-level numeric signature.
        for t in ("near_duplicate", "term_change", "unapplied_discount"):
            assert s.by_type[t][0] == 0
