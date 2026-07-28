"""Instrumentation tests.

These guard the two systems metrics the plan makes OURS:
    "% of cases resolved before the slow path"
    "cost per document across a run"
and the ~50% escalation kill-metric that decides whether the cascade lives.
"""

from pathlib import Path

from schemas import ExtractedRecord
from ladder import TIER_WEIGHT, CaseTrace, Rung, record, summarise
from verify.engine import verify_doc

FIXDIR = Path(__file__).parent.parent / "data" / "fixtures"


def _trace(case_id: str, *rungs_resolved) -> CaseTrace:
    t = CaseTrace(case_id=case_id)
    for rung, resolved, calls in rungs_resolved:
        with record(t, rung, inference_calls=calls) as ev:
            ev.resolved = resolved
    return t


# --- ladder ordering --------------------------------------------------------

def test_rung_ordering_and_gpu_flags():
    assert [r.index for r in Rung] == [0, 1, 2, 3, 4]
    assert not Rung.DETERMINISTIC.uses_gpu
    assert not Rung.MEMORY.uses_gpu
    assert Rung.FAST_TIER.uses_gpu and Rung.PRECISE_TIER.uses_gpu
    assert not Rung.HUMAN.uses_gpu   # top rung is a human, not a cloud model


def test_exit_rung_is_the_first_resolving_rung():
    t = _trace("c1",
               (Rung.DETERMINISTIC, False, 0),
               (Rung.MEMORY, False, 0),
               (Rung.FAST_TIER, True, 1))
    assert t.exit_rung == Rung.FAST_TIER
    assert t.inference_calls == 1
    assert t.touched_gpu


def test_free_exit_records_no_inference_and_no_gpu():
    t = _trace("c2", (Rung.DETERMINISTIC, True, 0))
    assert t.exit_rung == Rung.DETERMINISTIC
    assert t.inference_calls == 0 and t.weighted_cost == 0.0
    assert not t.touched_gpu


def test_wall_time_is_recorded():
    t = _trace("c3", (Rung.DETERMINISTIC, True, 0))
    assert t.wall_ms >= 0.0 and len(t.events) == 1


# --- cost accounting --------------------------------------------------------

def test_precise_tier_costs_more_than_fast_tier():
    """A precise-tier call is not a fast-tier call; weighting is why
    cost-per-document means anything."""
    fast = _trace("f", (Rung.FAST_TIER, True, 1))
    precise = _trace("p", (Rung.PRECISE_TIER, True, 1))
    assert precise.weighted_cost > fast.weighted_cost
    assert precise.weighted_cost == TIER_WEIGHT[Rung.PRECISE_TIER]


def test_n_sampling_counts_n_calls_not_one():
    """Self-consistency batches into ONE request, so wall-clock is about one
    generation — but the GPU genuinely does N. Counting it as 1 would
    understate the expensive rung and flatter the cascade."""
    t = _trace("n", (Rung.PRECISE_TIER, True, 5))
    assert t.inference_calls == 5
    assert t.weighted_cost == 5 * TIER_WEIGHT[Rung.PRECISE_TIER]


# --- the headline metrics ---------------------------------------------------

def test_resolved_before_gpu_and_escalation_are_complements():
    traces = [
        _trace("a", (Rung.DETERMINISTIC, True, 0)),
        _trace("b", (Rung.DETERMINISTIC, False, 0), (Rung.MEMORY, True, 0)),
        _trace("c", (Rung.DETERMINISTIC, False, 0), (Rung.FAST_TIER, True, 1)),
        _trace("d", (Rung.DETERMINISTIC, False, 0), (Rung.PRECISE_TIER, True, 3)),
    ]
    s = summarise(traces)
    assert s.n_cases == 4
    assert s.resolved_before_gpu_pct == 50.0
    assert s.escalation_rate_pct == 50.0
    assert s.total_inference_calls == 4
    assert s.inference_calls_per_case == 1.0


def test_unresolved_case_counts_as_escalated():
    """A case nothing resolved is not a free win — it is the most expensive
    outcome there is. It must never flatter the metric."""
    traces = [_trace("x", (Rung.DETERMINISTIC, False, 0))]
    s = summarise(traces)
    assert s.unresolved == 1
    assert s.escalation_rate_pct == 100.0
    assert s.resolved_before_gpu_pct == 0.0


def test_kill_metric_shows_in_the_report_above_fifty_percent():
    traces = ([_trace(f"g{i}", (Rung.FAST_TIER, True, 1)) for i in range(6)]
              + [_trace(f"f{i}", (Rung.DETERMINISTIC, True, 0))
                 for i in range(4)])
    s = summarise(traces)
    assert s.escalation_rate_pct == 60.0
    assert "KILL-METRIC" in s.render()


def test_kill_metric_silent_below_the_threshold():
    traces = [_trace(f"f{i}", (Rung.DETERMINISTIC, True, 0)) for i in range(10)]
    assert "KILL-METRIC" not in summarise(traces).render()


def test_empty_run_does_not_divide_by_zero():
    s = summarise([])
    assert s.n_cases == 0 and s.resolved_before_gpu_pct == 0.0


# --- end to end against real fixtures ---------------------------------------

def test_hand_written_fixtures_all_exit_at_rung_zero():
    """Every clean fixture is a STRICT pass, so the whole set resolves for
    free. This is the cascade's claim, measured rather than asserted."""
    traces = []
    for p in sorted((FIXDIR / "records").glob("*.json")):
        rec = ExtractedRecord.model_validate_json(p.read_text(encoding="utf-8"))
        t = CaseTrace(case_id=rec.doc.doc_id)
        with record(t, Rung.DETERMINISTIC) as ev:
            ev.resolved = verify_doc(rec.doc).strict_pass
        traces.append(t)
    s = summarise(traces)
    assert s.n_cases == 10
    assert s.resolved_before_gpu_pct == 100.0
    assert s.total_inference_calls == 0


def test_corrupted_fixtures_all_escalate():
    """The complement: nothing broken resolves for free. If a corrupted
    document exited at rung 0, the verifier would be waving errors through."""
    traces = []
    for p in sorted((FIXDIR / "corrupted").glob("*.json")):
        rec = ExtractedRecord.model_validate_json(p.read_text(encoding="utf-8"))
        t = CaseTrace(case_id=rec.doc.doc_id)
        with record(t, Rung.DETERMINISTIC) as ev:
            ev.resolved = verify_doc(rec.doc).strict_pass
        traces.append(t)
    s = summarise(traces)
    assert s.resolved_before_gpu_pct == 0.0
    assert s.escalation_rate_pct == 100.0


def test_trace_round_trips_through_json():
    """Traces are run artifacts written as JSONL; they must survive the trip
    or the Stage 2 charts are built on nothing."""
    t = _trace("rt", (Rung.DETERMINISTIC, False, 0), (Rung.FAST_TIER, True, 2))
    back = CaseTrace.model_validate_json(t.model_dump_json())
    assert back.exit_rung == Rung.FAST_TIER
    assert back.inference_calls == 2
