"""Confidence calibration and threshold selection.

Built BEFORE the learned confidence signal exists, deliberately. "Decide the
threshold later" reliably becomes "never decided", and by Stage 2 there is no
time to build machinery and interpret results at once. This module is pure
Python and testable against synthetic scores today; at Stage 2 it takes real
ones and the decision becomes a measurement.

WHAT IS HERE, AND WHY EACH EARNS ITS PLACE
------------------------------------------
1. risk_coverage(...)          — the standard selective-prediction picture:
                                 as you auto-resolve more, how much error do
                                 you admit? The plan already commits to
                                 choosing tau from this curve.

2. threshold_for_coverage(...) — RouteLLM's method. Do not guess a confidence
                                 number; choose the ESCALATION RATE you want
                                 and solve for the threshold on a sample. This
                                 ties tau directly to the ~50% kill-metric
                                 already in the plan. It is a quantile: no
                                 dependency, no training.

3. threshold_for_risk(...)     — the dual: the loosest threshold whose
                                 auto-resolved population stays under a
                                 maximum error rate. Answers "how much can we
                                 automate at 1% error?", which is the question
                                 a finance buyer actually asks.

4. IsotonicCalibrator          — Pool Adjacent Violators, ~30 lines, no
                                 sklearn. Raw confidence signals are poorly
                                 calibrated; a measured cascade cut cost 31%
                                 only AFTER an isotonic step, and a raw
                                 token-margin signal was weak used directly.
                                 Monotone, so it cannot reorder cases — it
                                 only maps scores onto honest probabilities.

5. expected_calibration_error  — the number that says whether 4 helped.

NOT HERE, ON PURPOSE
--------------------
Conformal prediction (MAPIE, crepes). Real and rigorous, but it assumes
exchangeable calibration data, our signal is a composite (a boolean AND an
agreement score) rather than a probability, and its guarantee concerns
coverage of prediction sets rather than routing correctness. The plan already
marks calibration [AMBITIOUS] after routing works, and that is the right
shelf for it. This module is the part that must exist before Stage 2.

PRIVACY AND "NO LEARNING FROM OUR FINANCES"
-------------------------------------------
Nothing in this file trains, fine-tunes or updates a model. Calibration fits
a monotone step function with a few hundred floats — a lookup table over
CONFIDENCE SCORES, not over customer data. It holds no amounts, no party
names, no document text. It is computed on the box, stored on the box, and is
inspectable by a human in its entirety. If an auditor asks "what did the
system learn about our finances?", the answer is: nothing. It learned where
its own uncertainty sits.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CurvePoint(_Base):
    threshold: float
    coverage: float = Field(description="fraction auto-resolved at this tau")
    risk: float = Field(description="error rate AMONG the auto-resolved")
    n_covered: int
    n_errors: int


def risk_coverage(scores: Sequence[float],
                  correct: Sequence[bool]) -> list[CurvePoint]:
    """The risk-coverage curve.

    Each point answers: if we auto-resolve everything scoring >= tau, what
    fraction do we cover and what error rate do we accept inside it? Risk is
    computed over the COVERED population only — that is the number that
    matters, because escalated cases get a second look and covered ones do
    not.
    """
    if len(scores) != len(correct):
        raise ValueError("scores and correct must be the same length")
    if not scores:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n = len(scores)
    points: list[CurvePoint] = []
    errors = 0
    for k, idx in enumerate(order, start=1):
        if not correct[idx]:
            errors += 1
        # Only emit a point where the threshold actually changes, so the curve
        # has one row per distinct decision boundary.
        if k < n and scores[order[k]] == scores[idx]:
            continue
        points.append(CurvePoint(
            threshold=float(scores[idx]),
            coverage=k / n,
            risk=errors / k,
            n_covered=k,
            n_errors=errors,
        ))
    return points


def threshold_for_coverage(scores: Sequence[float], target_coverage: float) -> float:
    """RouteLLM's method: pick the RATE, solve for the threshold.

    `target_coverage` is the fraction you want auto-resolved, so the implied
    escalation rate is 1 - target_coverage. Choosing 0.5 here puts you exactly
    on the plan's kill-metric boundary, which makes the relationship between
    threshold and kill-metric explicit rather than accidental.

    Needs no labels — only a representative sample of scores.
    """
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be in [0, 1]")
    if not scores:
        raise ValueError("no scores")
    s = sorted(scores, reverse=True)
    if target_coverage == 0.0:
        return float(s[0]) + 1e-9          # cover nothing
    k = max(1, min(len(s), round(target_coverage * len(s))))
    return float(s[k - 1])


def threshold_for_risk(scores: Sequence[float], correct: Sequence[bool],
                       max_risk: float) -> Optional[float]:
    """The dual question: the LOOSEST threshold whose auto-resolved population
    stays at or below `max_risk` error.

    Returns None when no threshold satisfies the constraint — which is a real
    answer, not a failure: it means the signal cannot support automation at
    that error rate and the honest move is to say so.
    """
    best: Optional[CurvePoint] = None
    for p in risk_coverage(scores, correct):
        if p.risk <= max_risk and (best is None or p.coverage > best.coverage):
            best = p
    return best.threshold if best else None


# ---------------------------------------------------------------------------
# Isotonic calibration — Pool Adjacent Violators
# ---------------------------------------------------------------------------

class IsotonicCalibrator(_Base):
    """Maps raw confidence scores onto calibrated probabilities.

    Monotone by construction, so it CANNOT change the ranking of cases — it
    only fixes what the numbers mean. That property is why it is safe: a
    calibrator that reordered cases could turn a good routing signal into a
    bad one, and this one provably cannot.

    Stored as two parallel arrays. Human-inspectable in full; no weights, no
    gradients, no customer data.
    """
    xs: list[float] = Field(default_factory=list)
    ys: list[float] = Field(default_factory=list)

    @classmethod
    def fit(cls, scores: Sequence[float],
            correct: Sequence[bool]) -> "IsotonicCalibrator":
        if len(scores) != len(correct):
            raise ValueError("scores and correct must be the same length")
        if not scores:
            raise ValueError("no data to fit")
        pairs = sorted(zip(scores, (1.0 if c else 0.0 for c in correct)),
                       key=lambda p: p[0])
        # PAV: walk left to right, merging any block that violates monotonicity
        # with its predecessor until order is restored.
        stack: list[tuple[float, float]] = []          # (mean, weight)
        for _, y in pairs:
            mean, w = y, 1.0
            while stack and stack[-1][0] > mean:
                pm, pw = stack.pop()
                mean = (pm * pw + mean * w) / (pw + w)
                w += pw
            stack.append((mean, w))
        fitted: list[float] = []
        for mean, w in stack:
            fitted.extend([mean] * int(round(w)))
        return cls(xs=[p[0] for p in pairs], ys=fitted)

    def predict(self, score: float) -> float:
        """Step lookup with linear interpolation between knots."""
        if not self.xs:
            raise ValueError("calibrator is not fitted")
        if score <= self.xs[0]:
            return self.ys[0]
        if score >= self.xs[-1]:
            return self.ys[-1]
        lo, hi = 0, len(self.xs) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.xs[mid] <= score:
                lo = mid
            else:
                hi = mid
        x0, x1 = self.xs[lo], self.xs[hi]
        y0, y1 = self.ys[lo], self.ys[hi]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (score - x0) / (x1 - x0)

    def predict_many(self, scores: Iterable[float]) -> list[float]:
        return [self.predict(s) for s in scores]


def expected_calibration_error(probs: Sequence[float],
                               correct: Sequence[bool],
                               bins: int = 10) -> float:
    """ECE: mean |confidence - accuracy| across equal-width bins.

    The number that says whether calibration helped. Report it before and
    after; if it does not move, say so rather than shipping the extra step.
    """
    if len(probs) != len(correct):
        raise ValueError("probs and correct must be the same length")
    if not probs:
        return 0.0
    total = 0.0
    n = len(probs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs)
               if (p > lo or (b == 0 and p >= lo)) and p <= hi]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        total += (len(idx) / n) * abs(conf - acc)
    return total


class ThresholdReport(_Base):
    """What we would put in the spec: the threshold, and every number that
    justifies it. A tau with no risk-coverage evidence behind it is a guess
    wearing a decimal point."""
    tau: float
    strategy: str
    coverage: float
    risk: float
    n_covered: int
    n_errors: int
    escalation_rate: float
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None

    def render(self) -> str:
        L = [f"tau = {self.tau:.4f}   ({self.strategy})",
             f"  coverage        : {self.coverage:.1%} auto-resolved",
             f"  escalation rate : {self.escalation_rate:.1%}"
             + ("   <-- at/over the kill-metric"
                if self.escalation_rate > 0.5 else ""),
             f"  risk in covered : {self.risk:.2%} "
             f"({self.n_errors}/{self.n_covered} wrong)"]
        if self.ece_before is not None:
            L.append(f"  ECE             : {self.ece_before:.4f} -> "
                     f"{self.ece_after:.4f}")
        return "\n".join(L)


def report_for_threshold(scores: Sequence[float], correct: Sequence[bool],
                         tau: float, strategy: str) -> ThresholdReport:
    covered = [i for i, s in enumerate(scores) if s >= tau]
    errors = sum(1 for i in covered if not correct[i])
    n = len(scores)
    cov = len(covered) / n if n else 0.0
    return ThresholdReport(
        tau=float(tau), strategy=strategy, coverage=cov,
        risk=errors / len(covered) if covered else 0.0,
        n_covered=len(covered), n_errors=errors,
        escalation_rate=1.0 - cov,
    )
