"""The cascade's cost model, and the kill-metric derived from it.

These bind to ladder.py rather than to helpers defined in this file. A cost
function that exists only inside its own test asserts properties of itself, and
nothing in the product is held to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladder import (Rung, TIER_WEIGHT, break_even_escalation_rate,  # noqa: E402
                    naive_cost_per_case, tiered_cost_per_case)


class TestRoutingSavesOnlyWhenEscalationIsLow:

    def test_low_escalation_beats_always_using_the_precise_model(self):
        assert tiered_cost_per_case(0.10) < naive_cost_per_case()

    def test_total_escalation_costs_MORE_than_the_baseline(self):
        """The headline kill-metric result, and it is not a tie.

        At 100% escalation every case pays the fast tier AND the precise tier,
        so the cascade costs the baseline plus the entire fast tier on top: 4.0
        against 3.0 at the current weights, 33% worse for nothing. This is the
        measured phenomenon behind cascades scoring BELOW
        always-using-the-larger-model while costing more, and asserting
        equality here would state the finding backwards.
        """
        assert tiered_cost_per_case(1.0) > naive_cost_per_case()
        assert tiered_cost_per_case(1.0) == pytest.approx(4.0)

    def test_the_break_even_is_derived_not_the_plan_s_round_number(self):
        """~50% is a target we set, not an industry constant. The real
        break-even falls out of the tier ratio."""
        assert break_even_escalation_rate(1) == pytest.approx(2 / 3, abs=1e-9)
        assert tiered_cost_per_case(break_even_escalation_rate(1)) == \
            pytest.approx(naive_cost_per_case())

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_at_the_break_even_the_cascade_is_exactly_the_baseline(self, n):
        r = break_even_escalation_rate(n)
        assert tiered_cost_per_case(r, n) == pytest.approx(naive_cost_per_case())

    def test_sampling_eats_the_margin_far_faster_than_the_rate_suggests(self):
        """66.7% at N=1 but 13.3% at N=5. Self-consistency is the dominant cost
        term in the whole cascade, which is why N is a knob set against
        measured escalation volume rather than a constant."""
        assert break_even_escalation_rate(1) > break_even_escalation_rate(3) \
            > break_even_escalation_rate(5)
        assert break_even_escalation_rate(5) < 0.15


class TestTheModelIsHonestAboutItsInputs:

    def test_every_case_pays_the_fast_tier(self):
        """Extraction and the fast pass are FIXED. Only the escalated share is
        variable, and a model that let the floor fall to zero would promise
        savings the architecture cannot deliver."""
        assert tiered_cost_per_case(0.0) == TIER_WEIGHT[Rung.FAST_TIER]

    def test_n_samples_multiplies_only_the_escalated_share(self):
        assert tiered_cost_per_case(0.2, 5) - tiered_cost_per_case(0.2, 1) == \
            pytest.approx(0.2 * 4 * TIER_WEIGHT[Rung.PRECISE_TIER])

    def test_a_cascade_with_no_tier_gap_cannot_save_anything(self):
        """If the measured TIER_WEIGHT comes back near 1.0 the honest move is
        to collapse to a single model and report it, so the model must return
        a break-even of zero rather than a tempting-looking number."""
        flat = {Rung.FAST_TIER: 1.0, Rung.PRECISE_TIER: 1.0}
        assert break_even_escalation_rate(1, flat) == 0.0

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_an_impossible_escalation_rate_raises(self, bad):
        with pytest.raises(ValueError):
            tiered_cost_per_case(bad)

    def test_zero_samples_raises(self):
        with pytest.raises(ValueError):
            tiered_cost_per_case(0.2, 0)
