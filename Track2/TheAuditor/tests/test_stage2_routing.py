"""Stage 2 routing. No GPU: this is the decision layer, not the inference.

Routing lives in ladder.py rather than in its own module, because the decision
and the cost of the decision are the same subject. A router that cannot name
the rung it escalates to cannot be costed, and a cost model that does not know
what the router does cannot be checked against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladder import (Rung, RouteAction, break_even_escalation_rate,  # noqa: E402
                    naive_cost_per_case, route, tiered_cost_per_case)


class TestTheSignalComposition:

    def test_verified_and_confident_auto_resolves(self):
        d = route(verify_pass=True, agreement=0.98)
        assert d.action is RouteAction.AUTO_RESOLVE
        assert d.next_rung is None

    def test_failed_verification_never_auto_resolves(self):
        """Deterministic arithmetic is the component that needs no calibration.
        High sample agreement cannot buy past a failed identity: the samples
        agreeing on a wrong number is exactly what agreement cannot detect."""
        d = route(verify_pass=False, agreement=1.0)
        assert d.action is RouteAction.ESCALATE

    def test_a_failed_identity_goes_to_the_precise_tier_not_a_human(self):
        """A failed identity usually means a misread number, not a broken
        deal, and re-reading is what the precise tier is for. Sending it
        straight to a person spends the most expensive rung on a problem the
        cheaper one solves."""
        d = route(verify_pass=False, agreement=0.99, from_rung=Rung.FAST_TIER)
        assert d.next_rung is Rung.PRECISE_TIER

    def test_once_the_precise_tier_has_looked_a_human_is_next(self):
        """The same evidence routes differently depending on what has already
        been spent, which is why from_rung is an input."""
        d = route(verify_pass=False, agreement=0.99,
                  from_rung=Rung.PRECISE_TIER)
        assert d.next_rung is Rung.HUMAN


class TestTauLoActuallyDoesSomething:
    """The original router returned "escalate" for both the low band and the
    middle band, so tau_lo could not change any output. A threshold that
    cannot change an output is not a threshold."""

    def test_the_low_band_and_the_middle_band_give_different_reasons(self):
        low = route(verify_pass=True, agreement=0.50, tau_lo=0.70, tau_hi=0.95)
        mid = route(verify_pass=True, agreement=0.80, tau_lo=0.70, tau_hi=0.95)
        assert low.action is mid.action is RouteAction.ESCALATE
        assert "genuinely disagree" in low.reason
        assert "not confident enough" in mid.reason
        assert low.reason != mid.reason

    def test_moving_tau_lo_moves_the_boundary(self):
        assert "genuinely disagree" not in route(
            verify_pass=True, agreement=0.80, tau_lo=0.70).reason
        assert "genuinely disagree" in route(
            verify_pass=True, agreement=0.80, tau_lo=0.85, tau_hi=0.95).reason

    def test_tau_hi_is_inclusive_at_the_boundary(self):
        assert route(verify_pass=True, agreement=0.95,
                     tau_hi=0.95).action is RouteAction.AUTO_RESOLVE


class TestThresholdsAreValidated:
    """Thresholds arrive from a calibration step. A silently-swapped pair would
    invert the routing and still produce plausible-looking output."""

    @pytest.mark.parametrize("kw", [
        {"tau_lo": 0.95, "tau_hi": 0.70},          # swapped
        {"tau_hi": 1.5},
        {"tau_lo": -0.1},
    ])
    def test_incoherent_thresholds_raise(self, kw):
        with pytest.raises(ValueError):
            route(verify_pass=True, agreement=0.9, **kw)

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_agreement_outside_the_unit_interval_raises(self, bad):
        with pytest.raises(ValueError):
            route(verify_pass=True, agreement=bad)
