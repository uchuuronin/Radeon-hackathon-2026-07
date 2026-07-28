"""Escalation-ladder instrumentation (src/ladder.py, NOT trace.py — `trace` is
a Python stdlib module name and a flat src/ would shadow it) — where a case exited and what it cost.

WHY THIS EXISTS ON DAY 2 AND NOT ON DAY 6
-----------------------------------------
Two of the ten headline metrics in the plan are ours, and both are computed
from exactly this data:

    "% of cases resolved BEFORE the slow path"  — the cascade's entire
                                                  justification
    "cost per document across a run"            — efficiency compounds as
                                                  memory fills

Neither can be reconstructed after the fact. If the pipeline does not record
which rung resolved a case as it happens, the number does not exist. This is
the same argument the plan already makes for the audit trail: wire it in at
Stage 1 so it is not retrofitted under time pressure at Stage 4.

It also feeds the ~50% escalation kill-metric. That threshold is measured
from escalation_rate() below, not from anything on Person B's side.

WHY NOT IN schemas.py
---------------------
schemas.py is the A<->B WIRE CONTRACT and it is frozen. Traces are run
artifacts produced by our pipeline driver and consumed by our reporting;
B emits ExtractedRecord and reports his own token counts through the
benchmark harness. Keeping traces out of schemas.py means this file can
evolve freely without touching B's prompt or invalidating a benchmark.
If B ever needs to emit RungEvents directly, we promote it then — with a
version bump, per the rules.

UNITS
-----
We deliberately do NOT invent dollar costs. Cost is reported in the two
units we can actually measure: INFERENCE CALLS (weighted by tier, because a
precise-tier call is not a fast-tier call) and GPU-SECONDS when the serving
layer reports them. Wall-clock is recorded separately — with batching,
wall-clock and GPU work diverge, and conflating them would overstate the
cascade's win.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from enum import Enum
from statistics import median
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field


class Rung(str, Enum):
    """The escalation ladder, cheapest first. Ordering is the whole design:
    never spend a GPU token you don't have to."""
    DETERMINISTIC = "0_deterministic"   # free — no GPU
    MEMORY = "1_memory"                 # near-free — signature table hit
    FAST_TIER = "2_fast_tier"           # 1 inference, quantised
    PRECISE_TIER = "3_precise_tier"     # N inferences, self-consistency
    HUMAN = "4_human"                   # analyst console

    @property
    def index(self) -> int:
        return int(self.value[0])

    @property
    def uses_gpu(self) -> bool:
        return self in (Rung.FAST_TIER, Rung.PRECISE_TIER)


#: Relative weight of one inference call at each tier, for cost-per-document.
#: Provisional until Person B's Stage 0 sweep measures the real ratio; the
#: point is that the ratio is a MEASURED input, not a constant we invented.
TIER_WEIGHT: dict[Rung, float] = {
    Rung.FAST_TIER: 1.0,
    Rung.PRECISE_TIER: 3.0,
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class RungEvent(_Base):
    """One rung, attempted. `resolved` marks the rung the case exited at."""
    rung: Rung
    resolved: bool = False
    wall_ms: float = 0.0
    inference_calls: int = Field(
        default=0,
        description="GPU calls made AT THIS RUNG. N-sample self-consistency "
                    "is N calls even though they batch into one request — "
                    "wall-clock is ~1 generation but GPU work is genuinely N.")
    gpu_seconds: Optional[float] = Field(
        default=None,
        description="Reported by the serving layer when available. None on "
                    "the free rungs, which is the point.")
    note: str = ""


class CaseTrace(_Base):
    """The full ladder walk for one case (a document, or later a deal chain)."""
    case_id: str
    events: list[RungEvent] = Field(default_factory=list)

    @property
    def exit_rung(self) -> Optional[Rung]:
        for e in self.events:
            if e.resolved:
                return Rung(e.rung)
        return None

    @property
    def wall_ms(self) -> float:
        return sum(e.wall_ms for e in self.events)

    @property
    def inference_calls(self) -> int:
        return sum(e.inference_calls for e in self.events)

    @property
    def weighted_cost(self) -> float:
        return sum(TIER_WEIGHT.get(Rung(e.rung), 0.0) * e.inference_calls
                   for e in self.events)

    @property
    def touched_gpu(self) -> bool:
        return any(Rung(e.rung).uses_gpu and e.inference_calls
                   for e in self.events)


@contextmanager
def record(trace: CaseTrace, rung: Rung, **kw):
    """Time a rung and append the event.

        with record(trace, Rung.DETERMINISTIC) as ev:
            report = verify_doc(doc)
            ev.resolved = report.strict_pass

    Three lines at each stage. The event is mutable inside the block so a
    stage can set `resolved`, `inference_calls` or `note` from its result.
    """
    ev = RungEvent(rung=rung, **kw)
    t0 = time.perf_counter()
    try:
        yield ev
    finally:
        ev.wall_ms = (time.perf_counter() - t0) * 1000
        trace.events.append(ev)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(int(round(q * (len(s) - 1))), len(s) - 1)
    return s[k]


class RunSummary(_Base):
    """The reportable shape. Every field here is a claim we can defend with
    a number, which is the standard the plan sets for headline claims."""
    n_cases: int
    exit_counts: dict[str, int]
    resolved_before_gpu_pct: float
    escalation_rate_pct: float
    unresolved: int
    median_wall_ms: float
    p95_wall_ms: float
    total_inference_calls: int
    inference_calls_per_case: float
    weighted_cost_per_case: float
    gpu_seconds: Optional[float] = None

    def render(self) -> str:
        lines = [
            f"{self.n_cases} cases",
            f"  resolved before GPU : {self.resolved_before_gpu_pct:.1f}%",
            f"  escalation rate     : {self.escalation_rate_pct:.1f}%"
            + ("   <-- KILL-METRIC: cascade is not saving compute"
               if self.escalation_rate_pct > 50 else ""),
            f"  latency ms          : median {self.median_wall_ms:.2f} "
            f"/ p95 {self.p95_wall_ms:.2f}",
            f"  inference calls     : {self.total_inference_calls} "
            f"({self.inference_calls_per_case:.2f}/case)",
            f"  weighted cost/case  : {self.weighted_cost_per_case:.2f}",
        ]
        for rung in Rung:
            n = self.exit_counts.get(rung.value, 0)
            if n:
                pct = 100 * n / self.n_cases if self.n_cases else 0
                lines.append(f"    exit {rung.value:<16} {n:>5}  ({pct:.1f}%)")
        if self.unresolved:
            lines.append(f"    UNRESOLVED         {self.unresolved:>5}")
        return "\n".join(lines)


def summarise(traces: Iterable[CaseTrace]) -> RunSummary:
    traces = list(traces)
    n = len(traces)
    counts: dict[str, int] = {}
    unresolved = 0
    for t in traces:
        r = t.exit_rung
        if r is None:
            unresolved += 1
        else:
            counts[r.value] = counts.get(r.value, 0) + 1

    before_gpu = sum(1 for t in traces
                     if t.exit_rung is not None and not t.exit_rung.uses_gpu)
    escalated = sum(1 for t in traces
                    if t.exit_rung is None or t.exit_rung.uses_gpu)
    walls = [t.wall_ms for t in traces]
    gpu = [t for tr in traces for t in
           (e.gpu_seconds for e in tr.events) if t is not None]

    return RunSummary(
        n_cases=n,
        exit_counts=counts,
        resolved_before_gpu_pct=100 * before_gpu / n if n else 0.0,
        escalation_rate_pct=100 * escalated / n if n else 0.0,
        unresolved=unresolved,
        median_wall_ms=median(walls) if walls else 0.0,
        p95_wall_ms=_pct(walls, 0.95),
        total_inference_calls=sum(t.inference_calls for t in traces),
        inference_calls_per_case=(
            sum(t.inference_calls for t in traces) / n if n else 0.0),
        weighted_cost_per_case=(
            sum(t.weighted_cost for t in traces) / n if n else 0.0),
        gpu_seconds=sum(gpu) if gpu else None,
    )
