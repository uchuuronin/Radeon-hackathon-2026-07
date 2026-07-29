"""Escalation-ladder instrumentation — hierarchical, and honest about cost.

(src/ladder.py, NOT trace.py — `trace` is a Python stdlib module name and a
flat src/ would shadow it.)

WHY HIERARCHICAL
----------------
v1 recorded one flat trace per DOCUMENT and called a document "resolved" when
its internal arithmetic passed. That produced "100% resolved before GPU",
which is vacuous: a document whose totals add up is not a settled
reconciliation. The cascade's unit is a CASE — a deal chain — and a case
contains documents, and documents contain rung attempts.

We borrow OpenTelemetry's data model (parent span id, nested children,
attributes) without the SDK. A collector, an exporter and an OTLP pipeline
are the wrong dependencies for a single-box 14-day project; the model is
about twenty lines and is the part that carries the value.

    CASE      chain C-0007                      <- the cascade's unit
      DOCUMENT  D-0007-1 .. D-0007-6
        RUNG      extract          (phase=INGEST, costs inference ALWAYS)
        RUNG      deterministic    (phase=LADDER, free)
      RUNG      link / reconcile / score        (phase=LADDER, case-level)

FIXED COST vs VARIABLE COST — the thing that keeps the metric honest
--------------------------------------------------------------------
Ingest is stage 1 of the pipeline: every document is extracted by a model
before any rung of the ladder runs. That call is UNAVOIDABLE. If we report
"resolved before the GPU" while N extraction calls have already been spent,
the number flatters us and a judge will see through it in ten seconds.

So every rung span carries a Phase:
    INGEST  — extraction. Fixed cost. Proportional to document count.
    LADDER  — the cascade. VARIABLE cost. This is where routing saves money,
              and the only place a cost claim is meaningful.

The headline metric is therefore "% of cases that needed NO ladder inference
beyond extraction", reported ALONGSIDE the extraction cost, never instead
of it.

PRIVACY — traces are the shareable artifact
-------------------------------------------
Traces are aggregate telemetry: ids, counts, timings, rung names. They must
NOT carry monetary values, party names or document text. Financial values
belong in the VerificationReport, which lives beside the document and stays
on the box. Keeping traces value-free is what lets us put a run log in a
demo video, a benchmark artifact or a bug report without exposing anyone's
finances. `attributes` is str->str for cheap labels only.

Nothing here trains anything. No model weights are updated by any of this,
ever.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from enum import Enum
from itertools import count
from statistics import median
from typing import Iterable, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field


class Scope(str, Enum):
    CASE = "case"          # a reconciliation case — a deal chain. THE unit.
    DOCUMENT = "document"  # one document within a case
    RUNG = "rung"          # one attempt at one rung


class Phase(str, Enum):
    INGEST = "ingest"      # extraction — fixed, unavoidable, per document
    LADDER = "ladder"      # the cascade — variable, where routing saves


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


#: Relative cost of one inference call at each tier. PROVISIONAL until Person
#: B's Stage 0 sweep measures the real ratio on the W7900 — the point is that
#: this is a measured input, not a constant we invented.
TIER_WEIGHT: dict[Rung, float] = {
    Rung.FAST_TIER: 1.0,
    Rung.PRECISE_TIER: 3.0,
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


_ids = count(1)


class Span(_Base):
    span_id: str = Field(default_factory=lambda: f"s{next(_ids)}")
    parent_id: Optional[str] = None
    scope: Scope
    name: str = Field(description="chain id, doc id, or rung label")
    phase: Optional[Phase] = None
    rung: Optional[Rung] = None
    resolved_case: bool = Field(
        default=False,
        description="This rung settled its CASE. Exactly one span per case "
                    "may set it; validate_coherence() enforces that.")
    wall_ms: float = 0.0
    inference_calls: int = Field(
        default=0,
        description="GPU calls made in this span. N-sample self-consistency "
                    "is N even though it batches into one request — "
                    "wall-clock is ~1 generation but GPU work is genuinely N.")
    gpu_seconds: Optional[float] = None
    note: str = ""
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Cheap non-financial labels only (model id, tier, layout). "
                    "NEVER amounts, party names or document text — traces are "
                    "the artifact we show people.")


class CaseSummary(_Base):
    """One reconciliation case, costed. The unit every headline number is
    computed over."""
    case_id: str
    resolved: bool
    exit_rung: Optional[Rung]
    n_documents: int
    ingest_calls: int          # FIXED    — extraction, unavoidable
    ladder_calls: int          # VARIABLE — the cascade
    ladder_cost: float         # tier-weighted
    wall_ms: float
    gpu_seconds: Optional[float] = None

    @property
    def needed_ladder_inference(self) -> bool:
        return self.ladder_calls > 0


class Trace(_Base):
    """A flat span list with parent links — the OTel shape. Flat storage keeps
    JSONL round-tripping trivial; the hierarchy is reconstructed on read."""
    spans: list[Span] = Field(default_factory=list)

    def add(self, span_: Span) -> Span:
        self.spans.append(span_)
        return span_

    def by_id(self) -> dict[str, Span]:
        return {s.span_id: s for s in self.spans}

    def children(self, span_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_id == span_id]

    def descendants(self, span_id: str) -> list[Span]:
        out, frontier = [], [span_id]
        while frontier:
            for c in self.children(frontier.pop()):
                out.append(c)
                frontier.append(c.span_id)
        return out

    def cases(self) -> list[Span]:
        return [s for s in self.spans if s.scope == Scope.CASE]

    # --- aggregation --------------------------------------------------------

    def case_summary(self, case_span: Span) -> CaseSummary:
        kin = self.descendants(case_span.span_id)
        docs = [s for s in kin if s.scope == Scope.DOCUMENT]
        rungs = [s for s in kin if s.scope == Scope.RUNG]

        resolver = next((s for s in rungs if s.resolved_case), None)
        ingest = [s for s in rungs if s.phase == Phase.INGEST]
        ladder = [s for s in rungs if s.phase == Phase.LADDER]
        gpu = [s.gpu_seconds for s in rungs if s.gpu_seconds is not None]

        return CaseSummary(
            case_id=case_span.name,
            resolved=resolver is not None,
            exit_rung=Rung(resolver.rung) if resolver and resolver.rung else None,
            n_documents=len(docs),
            ingest_calls=sum(s.inference_calls for s in ingest),
            ladder_calls=sum(s.inference_calls for s in ladder),
            ladder_cost=sum(TIER_WEIGHT.get(Rung(s.rung), 0.0) * s.inference_calls
                            for s in ladder if s.rung),
            wall_ms=case_span.wall_ms or sum(s.wall_ms for s in rungs),
            gpu_seconds=sum(gpu) if gpu else None,
        )

    def summaries(self) -> list[CaseSummary]:
        return [self.case_summary(c) for c in self.cases()]

    # --- self-defence -------------------------------------------------------

    def validate_coherence(self) -> list[str]:
        """Structural problems that would silently corrupt a metric.

        A trace failing these produces numbers that LOOK fine and ARE wrong —
        the worst failure mode an instrument can have. So we fail loudly.
        """
        problems: list[str] = []
        ids = self.by_id()

        for s in self.spans:
            if s.parent_id and s.parent_id not in ids:
                problems.append(f"{s.span_id} ({s.name}): orphan, parent "
                                f"{s.parent_id} not in trace")
            if s.scope == Scope.RUNG and s.rung is None:
                problems.append(f"{s.span_id} ({s.name}): rung span with no rung")
            if s.scope == Scope.RUNG and s.phase is None:
                problems.append(f"{s.span_id} ({s.name}): rung span with no "
                                f"phase — cost cannot be attributed")
            if s.inference_calls and s.rung and not Rung(s.rung).uses_gpu:
                problems.append(f"{s.span_id} ({s.name}): {s.inference_calls} "
                                f"inference call(s) at free rung {s.rung} — "
                                f"a rung is free or it is not")
            if s.resolved_case and s.scope != Scope.RUNG:
                problems.append(f"{s.span_id} ({s.name}): only a rung may "
                                f"resolve a case")

        for case in self.cases():
            kin = self.descendants(case.span_id)
            resolvers = [s for s in kin if s.resolved_case]
            if len(resolvers) > 1:
                problems.append(
                    f"case {case.name}: {len(resolvers)} spans claim to have "
                    f"resolved it; exactly one may")
            if not any(s.scope == Scope.DOCUMENT for s in kin):
                problems.append(f"case {case.name}: no documents")

        return problems


# --- recording ---------------------------------------------------------------

@contextmanager
def span(trace: Trace, scope: Scope, name: str,
         parent: Optional[Span] = None, **kw) -> Iterator[Span]:
    """Time a span and attach it to its parent."""
    s = Span(scope=scope, name=name,
             parent_id=parent.span_id if parent else None, **kw)
    t0 = time.perf_counter()
    try:
        yield s
    finally:
        s.wall_ms = (time.perf_counter() - t0) * 1000
        trace.add(s)


@contextmanager
def rung(trace: Trace, r: Rung, parent: Span,
         phase: Phase = Phase.LADDER, **kw) -> Iterator[Span]:
    """Shorthand: a rung attempt under a case or a document."""
    with span(trace, Scope.RUNG, r.value, parent=parent,
              rung=r, phase=phase, **kw) as s:
        yield s


# --- reporting ---------------------------------------------------------------

def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(round(q * (len(s) - 1))), len(s) - 1)]


class RunSummary(_Base):
    n_cases: int
    n_documents: int
    unresolved: int
    exit_counts: dict[str, int]

    #: The honest headline. NOT "resolved before the GPU" — extraction always
    #: costs. This is the share of cases the cascade settled without a single
    #: inference call BEYOND the unavoidable extraction.
    no_ladder_inference_pct: float
    escalation_rate_pct: float

    ingest_calls: int
    ingest_calls_per_case: float
    ladder_calls: int
    ladder_calls_per_case: float
    ladder_cost_per_case: float
    #: Extraction is flat; only this falls as memory fills.
    total_calls_per_case: float

    median_wall_ms: float
    p95_wall_ms: float
    gpu_seconds: Optional[float] = None

    def render(self) -> str:
        L = [
            f"{self.n_cases} cases · {self.n_documents} documents",
            "",
            "  FIXED    extraction (unavoidable, scales with documents)",
            f"    calls              : {self.ingest_calls} "
            f"({self.ingest_calls_per_case:.2f}/case)",
            "",
            "  VARIABLE ladder (where routing saves)",
            f"    no ladder inference: {self.no_ladder_inference_pct:.1f}% of cases",
            f"    escalation rate    : {self.escalation_rate_pct:.1f}%"
            + ("   <-- KILL-METRIC: cascade is not saving compute"
               if self.escalation_rate_pct > 50 else ""),
            f"    calls              : {self.ladder_calls} "
            f"({self.ladder_calls_per_case:.2f}/case)",
            f"    weighted cost/case : {self.ladder_cost_per_case:.2f}",
            "",
            f"  total calls/case     : {self.total_calls_per_case:.2f}",
            f"  latency ms           : median {self.median_wall_ms:.2f} "
            f"/ p95 {self.p95_wall_ms:.2f}",
        ]
        for r in Rung:
            n = self.exit_counts.get(r.value, 0)
            if n:
                L.append(f"    exit {r.value:<16} {n:>5}  "
                         f"({100 * n / self.n_cases:.1f}%)")
        if self.unresolved:
            L.append(f"    UNRESOLVED         {self.unresolved:>5}")
        return "\n".join(L)


def summarise(summaries: Iterable[CaseSummary]) -> RunSummary:
    s = list(summaries)
    n = len(s)

    counts: dict[str, int] = {}
    for c in s:
        if c.exit_rung:
            counts[c.exit_rung] = counts.get(c.exit_rung, 0) + 1

    # A case with no ladder inference was settled by deterministic checks and
    # memory alone. UNRESOLVED never counts as a win — it is the most
    # expensive outcome there is.
    free = sum(1 for c in s if c.resolved and not c.needed_ladder_inference)
    escalated = sum(1 for c in s if not c.resolved or c.needed_ladder_inference)
    walls = [c.wall_ms for c in s]
    gpu = [c.gpu_seconds for c in s if c.gpu_seconds is not None]

    ingest = sum(c.ingest_calls for c in s)
    ladder = sum(c.ladder_calls for c in s)

    return RunSummary(
        n_cases=n,
        n_documents=sum(c.n_documents for c in s),
        unresolved=sum(1 for c in s if not c.resolved),
        exit_counts=counts,
        no_ladder_inference_pct=100 * free / n if n else 0.0,
        escalation_rate_pct=100 * escalated / n if n else 0.0,
        ingest_calls=ingest,
        ingest_calls_per_case=ingest / n if n else 0.0,
        ladder_calls=ladder,
        ladder_calls_per_case=ladder / n if n else 0.0,
        ladder_cost_per_case=(sum(c.ladder_cost for c in s) / n) if n else 0.0,
        total_calls_per_case=(ingest + ladder) / n if n else 0.0,
        median_wall_ms=median(walls) if walls else 0.0,
        p95_wall_ms=_pct(walls, 0.95),
        gpu_seconds=sum(gpu) if gpu else None,
    )
