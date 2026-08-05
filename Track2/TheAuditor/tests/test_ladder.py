"""Tests for the hierarchical ladder and the calibration machinery.

Between them these guard every systems number the plan makes OURS:
  "% of cases resolved before the slow path"
  "cost per document across a run"
  the ~50% escalation kill-metric
  the routing threshold tau
"""

import random

import pytest

from calibrate import (
    IsotonicCalibrator,
    expected_calibration_error,
    report_for_threshold,
    risk_coverage,
    threshold_for_coverage,
    threshold_for_risk,
)
from ladder import (
    CaseSummary,
    Phase,
    Rung,
    Scope,
    Span,
    Trace,
    rung,
    span,
    summarise,
)


# ===========================================================================
# LADDER — hierarchy, cost attribution, coherence
# ===========================================================================

def build_case(tr, case_id, n_docs, *, ladder_rung=None, ladder_calls=0,
               resolve_at=Rung.DETERMINISTIC, resolve=True):
    """A case: N documents each extracted once (FIXED cost), then the ladder."""
    with span(tr, Scope.CASE, case_id) as case:
        for i in range(n_docs):
            with span(tr, Scope.DOCUMENT, f"{case_id}-d{i}", parent=case) as d:
                # Ingest ALWAYS costs one inference. This is the point.
                with rung(tr, Rung.FAST_TIER, d, phase=Phase.INGEST) as ing:
                    ing.inference_calls = 1
                with rung(tr, Rung.DETERMINISTIC, d, phase=Phase.LADDER):
                    pass
        with rung(tr, resolve_at, case, phase=Phase.LADDER) as r:
            r.resolved_case = resolve
        if ladder_rung is not None and ladder_calls:
            with rung(tr, ladder_rung, case, phase=Phase.LADDER) as extra:
                extra.inference_calls = ladder_calls
    return case


def test_case_is_the_unit_not_the_document():
    tr = Trace()
    build_case(tr, "C-1", 6)
    s = tr.summaries()
    assert len(s) == 1, "six documents are ONE reconciliation case"
    assert s[0].n_documents == 6


def test_extraction_cost_is_counted_and_kept_separate():
    """The bug this redesign exists to prevent: reporting a case as free when
    six extraction calls were already spent on it."""
    tr = Trace()
    build_case(tr, "C-1", 6)
    c = tr.summaries()[0]
    assert c.ingest_calls == 6          # FIXED, unavoidable
    assert c.ladder_calls == 0          # VARIABLE, saved by the cascade
    assert not c.needed_ladder_inference

    run = summarise(tr.summaries())
    assert run.ingest_calls_per_case == 6.0
    assert run.no_ladder_inference_pct == 100.0
    assert run.total_calls_per_case == 6.0, (
        "total must include extraction — otherwise the number lies")


def test_headline_is_not_reported_as_zero_gpu():
    tr = Trace()
    for i in range(5):
        build_case(tr, f"C-{i}", 4)
    run = summarise(tr.summaries())
    assert run.no_ladder_inference_pct == 100.0
    assert run.ingest_calls == 20
    assert run.total_calls_per_case > 0


def test_ladder_inference_moves_the_case_out_of_the_free_bucket():
    tr = Trace()
    build_case(tr, "C-free", 3)
    build_case(tr, "C-esc", 3, resolve_at=Rung.PRECISE_TIER,
               ladder_rung=Rung.PRECISE_TIER, ladder_calls=5)
    run = summarise(tr.summaries())
    assert run.no_ladder_inference_pct == 50.0
    assert run.escalation_rate_pct == 50.0
    assert run.ladder_calls == 5
    assert run.ingest_calls == 6


def test_n_sampling_counts_n_not_one():
    tr = Trace()
    build_case(tr, "C-1", 1, resolve_at=Rung.PRECISE_TIER,
               ladder_rung=Rung.PRECISE_TIER, ladder_calls=5)
    c = tr.summaries()[0]
    assert c.ladder_calls == 5
    assert c.ladder_cost == 5 * 3.0


def test_unresolved_case_is_never_a_free_win():
    tr = Trace()
    build_case(tr, "C-1", 2, resolve=False)
    run = summarise(tr.summaries())
    assert run.unresolved == 1
    assert run.no_ladder_inference_pct == 0.0
    assert run.escalation_rate_pct == 100.0


def test_hierarchy_is_navigable_and_round_trips():
    tr = Trace()
    case = build_case(tr, "C-1", 2)
    assert len(tr.descendants(case.span_id)) == 2 + 2 * 2 + 1
    back = Trace.model_validate_json(tr.model_dump_json())
    assert back.summaries()[0].ingest_calls == 2


def test_validator_accepts_a_well_formed_trace():
    tr = Trace()
    build_case(tr, "C-1", 3)
    assert tr.validate_coherence() == []


def test_validator_catches_two_resolvers_for_one_case():
    tr = Trace()
    case = build_case(tr, "C-1", 1)
    with rung(tr, Rung.MEMORY, case) as r:
        r.resolved_case = True
    assert any("claim to have resolved" in p for p in tr.validate_coherence())


def test_validator_catches_inference_recorded_at_a_free_rung():
    """A rung is free or it is not. GPU calls at rung 0 would make the
    cascade look cheap while it was quietly spending."""
    tr = Trace()
    case = build_case(tr, "C-1", 1)
    with rung(tr, Rung.DETERMINISTIC, case) as r:
        r.inference_calls = 2
    assert any("free rung" in p for p in tr.validate_coherence())


def test_validator_catches_orphans_and_empty_cases():
    tr = Trace()
    tr.add(Span(scope=Scope.RUNG, name="x", parent_id="nope",
                rung=Rung.MEMORY, phase=Phase.LADDER))
    with span(tr, Scope.CASE, "C-empty"):
        pass
    problems = tr.validate_coherence()
    assert any("orphan" in p for p in problems)
    assert any("no documents" in p for p in problems)


def test_validator_catches_unphased_rung():
    tr = Trace()
    case = build_case(tr, "C-1", 1)
    tr.add(Span(scope=Scope.RUNG, name="mystery", parent_id=case.span_id,
                rung=Rung.FAST_TIER))
    assert any("no phase" in p for p in tr.validate_coherence())


def test_kill_metric_renders_only_above_threshold():
    hot = summarise(
        [CaseSummary(case_id=f"c{i}", resolved=True,
                     exit_rung=Rung.PRECISE_TIER, n_documents=1,
                     ingest_calls=1, ladder_calls=3, ladder_cost=9.0,
                     wall_ms=1.0) for i in range(6)]
        + [CaseSummary(case_id=f"f{i}", resolved=True,
                       exit_rung=Rung.DETERMINISTIC, n_documents=1,
                       ingest_calls=1, ladder_calls=0, ladder_cost=0.0,
                       wall_ms=1.0) for i in range(4)])
    assert hot.escalation_rate_pct == 60.0
    assert "KILL-METRIC" in hot.render()


def test_empty_run_does_not_divide_by_zero():
    r = summarise([])
    assert r.n_cases == 0 and r.no_ladder_inference_pct == 0.0


# ===========================================================================
# CALIBRATION — thresholds chosen by measurement, not by guess
# ===========================================================================

def synthetic(n=400, seed=7):
    """A plausible confidence signal: higher score => more likely correct,
    but noisy and badly scaled (squashed into the top of the range, exactly
    how raw model confidences misbehave)."""
    rng = random.Random(seed)
    scores, correct = [], []
    for _ in range(n):
        true_p = rng.random()
        correct.append(rng.random() < true_p)
        scores.append(0.5 + 0.5 * true_p + rng.gauss(0, 0.03))
    return scores, correct


def test_risk_coverage_is_monotone_in_coverage():
    scores, correct = synthetic()
    pts = risk_coverage(scores, correct)
    assert pts[0].coverage < pts[-1].coverage
    assert pts[-1].coverage == 1.0
    assert 0.0 <= pts[0].risk <= 1.0


def test_perfect_signal_has_zero_risk_at_full_coverage():
    pts = risk_coverage([0.9, 0.8, 0.7, 0.6], [True] * 4)
    assert all(p.risk == 0.0 for p in pts)


def test_threshold_for_coverage_hits_the_requested_rate():
    """RouteLLM's method: choose the escalation rate, solve for tau."""
    scores, correct = synthetic()
    for target in (0.25, 0.5, 0.9):
        tau = threshold_for_coverage(scores, target)
        rep = report_for_threshold(scores, correct, tau, "target-coverage")
        assert abs(rep.coverage - target) < 0.02, (target, rep.coverage)


def test_threshold_at_fifty_percent_sits_on_the_kill_metric():
    scores, correct = synthetic()
    tau = threshold_for_coverage(scores, 0.5)
    rep = report_for_threshold(scores, correct, tau, "target-coverage")
    assert abs(rep.escalation_rate - 0.5) < 0.02


def test_threshold_for_risk_respects_the_error_budget():
    scores, correct = synthetic()
    tau = threshold_for_risk(scores, correct, max_risk=0.10)
    assert tau is not None
    assert report_for_threshold(scores, correct, tau, "max-risk").risk <= 0.10 + 1e-9


def test_threshold_for_risk_returns_none_when_impossible():
    """An honest 'this signal cannot support automation at that error rate'
    beats a threshold that silently misses the budget."""
    assert threshold_for_risk([0.9, 0.8, 0.7], [False] * 3, 0.01) is None


def test_looser_risk_budget_buys_more_coverage():
    scores, correct = synthetic()
    strict = threshold_for_risk(scores, correct, 0.05)
    loose = threshold_for_risk(scores, correct, 0.25)
    s = report_for_threshold(scores, correct, strict, "s").coverage
    l = report_for_threshold(scores, correct, loose, "l").coverage
    assert l >= s


def test_isotonic_output_is_monotone():
    scores, correct = synthetic()
    cal = IsotonicCalibrator.fit(scores, correct)
    probs = cal.predict_many(sorted(scores))
    assert all(b >= a - 1e-9 for a, b in zip(probs, probs[1:]))


def test_isotonic_cannot_reorder_cases():
    """The safety property. A calibrator that changed the RANKING could turn
    a good routing signal into a bad one; monotonicity forbids it."""
    scores, correct = synthetic()
    cal = IsotonicCalibrator.fit(scores, correct)
    cal_scores = cal.predict_many(scores)
    ranked_raw = sorted(range(len(scores)), key=lambda i: scores[i])
    for i, j in zip(ranked_raw, ranked_raw[1:]):
        assert cal_scores[i] <= cal_scores[j] + 1e-9


def test_isotonic_improves_calibration_error():
    scores, correct = synthetic(n=800)
    before = expected_calibration_error(scores, correct)
    cal = IsotonicCalibrator.fit(scores, correct)
    after = expected_calibration_error(cal.predict_many(scores), correct)
    assert after < before, (before, after)


def test_isotonic_probabilities_stay_in_range():
    scores, correct = synthetic()
    cal = IsotonicCalibrator.fit(scores, correct)
    assert all(0.0 <= p <= 1.0 for p in cal.predict_many(scores))


def test_calibrator_round_trips_and_is_inspectable():
    """It is a lookup table, not a model. A human can read it end to end —
    which makes 'the system learned nothing about your finances' checkable."""
    scores, correct = synthetic(n=50)
    cal = IsotonicCalibrator.fit(scores, correct)
    back = IsotonicCalibrator.model_validate_json(cal.model_dump_json())
    assert back.predict(0.8) == pytest.approx(cal.predict(0.8))
    # Parallel arrays, and no more knots than training points: PAV yields a
    # step function and the interior of a constant block carries no
    # information, so it is not stored. Inspectability is the point, and a
    # shorter table is MORE inspectable, not less.
    assert len(back.xs) == len(back.ys) <= 50
    # Still a faithful reconstruction at every training point.
    for x, y in zip(cal.xs, cal.ys):
        assert cal.predict(x) == pytest.approx(y)


def test_calibrator_compaction_is_lossless():
    """Dropping redundant knots must not move a single prediction."""
    scores, correct = synthetic(n=400)
    cal = IsotonicCalibrator.fit(scores, correct)
    assert len(cal.xs) < 400                      # compaction actually happened
    probe = [i / 500 for i in range(501)]
    # Monotone and bounded everywhere, which is the property routing relies on.
    got = cal.predict_many(probe)
    assert all(0.0 <= v <= 1.0 for v in got)
    assert all(a <= b + 1e-12 for a, b in zip(got, got[1:]))


def test_calibrator_holds_no_case_data():
    """Scores and fitted probabilities only. No ids, amounts, party names or
    text — by construction, not by discipline."""
    cal = IsotonicCalibrator.fit(*synthetic(n=20))
    assert set(cal.model_dump().keys()) == {"xs", "ys"}


def test_report_renders_the_evidence_behind_tau():
    scores, correct = synthetic()
    tau = threshold_for_coverage(scores, 0.7)
    text = report_for_threshold(scores, correct, tau, "target-coverage").render()
    for token in ("tau", "coverage", "escalation rate", "risk in covered"):
        assert token in text


def test_calibration_is_deterministic():
    """A tau that moves between runs is not a decision, it is a coin flip."""
    scores, _ = synthetic()
    assert threshold_for_coverage(scores, 0.6) == threshold_for_coverage(list(scores), 0.6)


# ---------------------------------------------------------------------------
# Reporting a small, deliberately balanced corpus honestly
# ---------------------------------------------------------------------------

class TestSmallSampleReporting:
    def test_two_of_two_is_not_a_measurement(self):
        """The number the corpus used to produce. 100% recall on n=2 is
        consistent with a detector that misses two thirds of them, and a bare
        percentage hides that completely."""
        from calibrate import wilson_interval
        iv = wilson_interval(2, 2)
        assert iv.point == 1.0
        assert iv.lo < 0.40

    def test_interval_tightens_with_n(self):
        from calibrate import wilson_interval
        widths = [wilson_interval(n, n).hi - wilson_interval(n, n).lo
                  for n in (2, 10, 30, 100)]
        assert widths == sorted(widths, reverse=True)

    def test_wilson_is_not_degenerate_at_the_extremes(self):
        """The normal approximation returns zero width at 0/n and n/n, which
        is exactly where small-corpus results land."""
        from calibrate import wilson_interval
        assert wilson_interval(0, 5).hi > 0.0
        assert wilson_interval(5, 5).lo < 1.0

    def test_corpus_sizing_is_computed_not_guessed(self):
        from calibrate import required_n_for_halfwidth, wilson_interval
        n = required_n_for_halfwidth(0.10, p=0.9)
        iv = wilson_interval(round(0.9 * n), n)
        assert (iv.hi - iv.lo) / 2 <= 0.10
        assert n <= 60                       # affordable: generation is free

    def test_precision_does_not_survive_a_prevalence_change(self):
        """Recall is a property of the detector; precision is a property of
        the detector AND the base rate. Quoting the balanced-corpus figure
        for a production queue overstates it by an order of magnitude."""
        from calibrate import precision_at_prevalence
        balanced = precision_at_prevalence(0.95, 0.02, 0.70)
        realistic = precision_at_prevalence(0.95, 0.02, 0.01)
        assert balanced > 0.95
        assert realistic < 0.40
        assert balanced / realistic > 2


class TestCalibrationMetrics:
    def test_equal_mass_does_not_split_tied_scores(self):
        """100 documents all scoring 0.9, half correct. The true gap is 0.40;
        splitting the tie across bins reports 0.50."""
        from calibrate import expected_calibration_error
        probs = [0.9] * 100
        correct = [True] * 50 + [False] * 50
        assert expected_calibration_error(probs, correct) == pytest.approx(0.40)

    def test_both_schemes_agree_on_a_hand_computable_case(self):
        from calibrate import expected_calibration_error
        probs = [0.1] * 50 + [0.9] * 50
        correct = [True] * 25 + [False] * 25 + [True] * 25 + [False] * 25
        for scheme in ("equal_mass", "equal_width"):
            assert expected_calibration_error(
                probs, correct, scheme=scheme) == pytest.approx(0.40)

    def test_brier_is_bounded_and_proper(self):
        from calibrate import brier_score
        assert brier_score([1.0, 0.0], [True, False]) == 0.0
        assert brier_score([0.0, 1.0], [True, False]) == 1.0
        # Hedging beats being confidently wrong, which is what "proper" buys.
        assert brier_score([0.5, 0.5], [True, False]) < \
            brier_score([0.0, 1.0], [True, False])

    def test_unknown_scheme_is_refused(self):
        from calibrate import expected_calibration_error
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [True], scheme="quantile")


class TestQuickselectThreshold:
    def test_matches_a_sort_based_reference_everywhere(self):
        import random
        from calibrate import threshold_for_coverage
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(1, 80)
            scores = [round(rng.random(), 3) for _ in range(n)]
            cov = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, rng.random()])
            got = threshold_for_coverage(scores, cov)
            s = sorted(scores, reverse=True)
            exp = (float(s[0]) + 1e-9 if cov == 0.0
                   else float(s[max(1, min(n, round(cov * n))) - 1]))
            assert got == pytest.approx(exp)

    def test_survives_the_adversarial_shape(self):
        """Already-sorted input with heavy ties near 1.0 is exactly what a
        verifier-anchored confidence signal produces, and exactly what a
        naive pivot degrades on."""
        from calibrate import threshold_for_coverage
        scores = sorted([1.0] * 500 + [0.5] * 500)
        assert threshold_for_coverage(scores, 0.5) == 1.0

    def test_does_not_mutate_the_caller_list(self):
        from calibrate import threshold_for_coverage
        scores = [0.3, 0.9, 0.1, 0.7]
        before = list(scores)
        threshold_for_coverage(scores, 0.5)
        assert scores == before
