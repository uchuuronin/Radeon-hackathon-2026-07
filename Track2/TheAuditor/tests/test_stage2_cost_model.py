"""
Stage 2 logical constraints.

These tests do not claim GPU throughput.
They validate the mathematical behavior of the two-tier routing system:
- routing should save cost when escalation is limited
- higher confidence thresholds should reduce auto coverage
- escalation cost should be measurable
"""


def tiered_cost(
    n_docs: int,
    fast_cost: float,
    slow_cost: float,
    escalation_rate: float,
) -> float:
    return (
        n_docs * fast_cost
        + n_docs * escalation_rate * slow_cost
    )


def coverage(
    agreements: list[float],
    threshold: float,
) -> float:
    auto = sum(a >= threshold for a in agreements)
    return auto / len(agreements)


def test_tiered_routing_beats_naive_when_escalation_is_low():
    docs = 1000

    naive = docs * 10

    tiered = tiered_cost(
        n_docs=docs,
        fast_cost=2,
        slow_cost=10,
        escalation_rate=0.10,
    )

    assert tiered < naive


def test_high_escalation_can_destroy_savings():
    docs = 1000

    naive = docs * 10

    tiered = tiered_cost(
        n_docs=docs,
        fast_cost=2,
        slow_cost=10,
        escalation_rate=1.0,
    )

    assert tiered == naive


def test_higher_threshold_reduces_coverage():
    agreements = [
        0.99,
        0.97,
        0.90,
        0.80,
        0.60,
    ]

    loose = coverage(
        agreements,
        threshold=0.80,
    )

    strict = coverage(
        agreements,
        threshold=0.95,
    )

    assert strict < loose


def test_perfect_confidence_allows_resolution():
    agreements = [1.0, 1.0, 1.0]

    assert coverage(
        agreements,
        threshold=0.95,
    ) == 1.0


def test_zero_confidence_escalates_everything():
    agreements = [
        0.10,
        0.20,
        0.30,
    ]

    assert coverage(
        agreements,
        threshold=0.95,
    ) == 0.0
