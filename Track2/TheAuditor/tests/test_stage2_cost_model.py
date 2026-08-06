"""
Logical tests for Stage 2 two-tier routing cost model.
"""


def tiered_cost(
    n_docs: int,
    fast_cost: float,
    slow_cost: float,
    escalation_rate: float,
) -> float:
    """
    Escalation routes to slow tier instead of paying both tiers.
    """
    fast_docs = n_docs * (1 - escalation_rate)
    slow_docs = n_docs * escalation_rate

    return (
        fast_docs * fast_cost
        + slow_docs * slow_cost
    )


def test_no_escalation_uses_fast_tier_only():
    docs = 1000

    assert tiered_cost(
        docs,
        fast_cost=2,
        slow_cost=10,
        escalation_rate=0.0,
    ) == 2000


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


def test_partial_escalation_cost():
    docs = 1000

    tiered = tiered_cost(
        n_docs=docs,
        fast_cost=2,
        slow_cost=10,
        escalation_rate=0.5,
    )

    assert tiered == 6000


def test_more_escalation_costs_more():
    docs = 1000

    low = tiered_cost(
        docs,
        fast_cost=2,
        slow_cost=10,
        escalation_rate=0.1,
    )

    high = tiered_cost(
        docs,
        fast_cost=2,
        slow_cost=10,
        escalation_rate=0.9,
    )

    assert high > low
