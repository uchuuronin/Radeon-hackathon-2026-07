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

5. expected_calibration_error  — whether 4 helped. Equal-MASS bins by
                                 default; equal-width is available for
                                 comparability with papers and is not our
                                 headline. Reported with brier_score, a
                                 proper scoring rule that no choice of bins
                                 can move.

6. wilson_interval             — the interval every small-sample rate must
                                 carry, and required_n_for_halfwidth to size
                                 the corpus instead of guessing at it.

7. precision_at_prevalence     — precision projected off the deliberately
                                 balanced corpus onto a realistic base rate.
                                 Recall survives a prevalence change;
                                 precision does not.

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

from typing import Iterable, NamedTuple, Optional, Sequence

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


def _select_kth_largest(values: list[float], k: int) -> float:
    """The k-th largest value (k is 1-based), in O(n) expected time.

    Quickselect with a median-of-three pivot and Hoare partitioning: the
    standard answer to "kth largest element in an array". Sorting to read ONE
    order statistic is O(n log n) work for an O(n) question, and this is not
    a one-off call - the Stage 2 sweep solves for a threshold at every
    candidate coverage, on every configuration, over the whole score vector.
    Mutates a copy, never the caller's list.

    Median-of-three matters here rather than being a flourish: confidence
    scores arrive already sorted by case id and heavily tied near 1.0, and a
    first-element pivot degrades to O(n^2) on exactly that shape.
    """
    a = list(values)
    lo, hi = 0, len(a) - 1
    target = k - 1                       # index into descending order
    while lo < hi:
        mid = (lo + hi) // 2
        x, y, z = a[lo], a[mid], a[hi]
        pivot = max(min(x, y), min(max(x, y), z))     # median of three
        i, j = lo, hi
        while i <= j:                    # partition DESCENDING
            while a[i] > pivot:
                i += 1
            while a[j] < pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if target <= j:
            hi = j
        elif target >= i:
            lo = i
        else:
            return float(a[target])
    return float(a[target])


def threshold_for_coverage(scores: Sequence[float], target_coverage: float) -> float:
    """RouteLLM's method: pick the RATE, solve for the threshold.

    `target_coverage` is the fraction you want auto-resolved, so the implied
    escalation rate is 1 - target_coverage. Choosing 0.5 here puts you exactly
    on the plan's kill-metric boundary, which makes the relationship between
    threshold and kill-metric explicit rather than accidental.

    Needs no labels — only a representative sample of scores.

    WHAT THIS DOES AND DOES NOT CONTROL. It controls COVERAGE, not error. A
    quantile over unlabelled scores fixes how much work is auto-resolved and
    says nothing about how much of it is wrong; only threshold_for_risk,
    which needs labels, controls quality. They are duals and the pair must be
    reported together, because quoting a coverage threshold as though it
    bounded error is the standard way this method gets misused.
    """
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be in [0, 1]")
    if not scores:
        raise ValueError("no scores")
    if target_coverage == 0.0:
        return float(max(scores)) + 1e-9          # cover nothing
    k = max(1, min(len(scores), round(target_coverage * len(scores))))
    return _select_kth_largest(list(scores), k)


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
        # Expand blocks back over the sorted inputs, then KEEP ONLY THE
        # KNOTS. PAV produces a step function; storing one (x, y) pair per
        # training point stores the same y hundreds of times over, and every
        # redundant point also widens the binary search in predict(). sklearn
        # does the same compaction (it keeps X_thresholds_/y_thresholds_ and
        # discards interior points of a constant block), and it is lossless:
        # a step function is fully described by where it steps. On a
        # realistic calibration set this is the difference between storing n
        # points and storing the number of distinct fitted values, which for
        # a monotone signal is typically a small fraction of n.
        fitted: list[float] = []
        for mean, w in stack:
            fitted.extend([mean] * int(round(w)))

        xs_all = [q[0] for q in pairs]
        xs: list[float] = []
        ys: list[float] = []
        for i, (x, y) in enumerate(zip(xs_all, fitted)):
            first = i == 0
            last = i == len(fitted) - 1
            # A point is a knot if it starts a block, ends a block, or is an
            # endpoint. Interior points of a run are pure redundancy.
            if first or last or y != fitted[i - 1] or y != fitted[i + 1]:
                if xs and x == xs[-1] and y == ys[-1]:
                    continue
                xs.append(x)
                ys.append(y)
        return cls(xs=xs, ys=ys)

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


def brier_score(probs: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the probabilities. Lower is better.

    Reported alongside ECE because it is a PROPER SCORING RULE: it is
    minimised only by the truthful probability, it needs no bins, and it
    therefore cannot be moved by re-binning. ECE can. Where the two disagree,
    Brier is the one to trust.
    """
    if len(probs) != len(correct):
        raise ValueError("probs and correct must be the same length")
    if not probs:
        return 0.0
    return sum((p - (1.0 if c else 0.0)) ** 2
               for p, c in zip(probs, correct)) / len(probs)


def expected_calibration_error(probs: Sequence[float],
                               correct: Sequence[bool],
                               bins: int = 10,
                               scheme: str = "equal_mass") -> float:
    """ECE: mean |confidence - accuracy| across bins.

    scheme="equal_mass" (DEFAULT) — every bin holds the same NUMBER of
        points. This is the adaptive-binning form, and it is the default for
        a measured reason: our confidence signal is a composite anchored on a
        deterministic verifier that passes most documents, so scores pile up
        against 1.0. Under equal-width bins nine bins are then empty or
        near-empty and the statistic is decided by whatever noise lands in
        the tenth. Equal-mass binning is documented as the lower-bias choice
        and it is the one that survives a skewed score distribution.
    scheme="equal_width" — fixed [b/B, (b+1)/B) bins. Kept because it is what
        most papers report and comparability matters, but it is not our
        headline.

    ECE is biased and binning-sensitive under BOTH schemes, which is why
    brier_score() is reported next to it rather than instead of it.

    Single pass: bin membership is computed by index arithmetic instead of
    re-scanning every probability once per bin, so this is O(n log n) for the
    sort under equal-mass and O(n) under equal-width, not O(bins x n).
    """
    if len(probs) != len(correct):
        raise ValueError("probs and correct must be the same length")
    if not probs:
        return 0.0
    n = len(probs)
    hits = [1.0 if c else 0.0 for c in correct]

    if scheme == "equal_mass":
        order = sorted(range(n), key=lambda i: probs[i])
        # Snap every edge forward past any run of TIED scores. Splitting a
        # tie across two bins is the known pathology of equal-mass binning:
        # 100 documents all scoring 0.9, half of them correct, get carved
        # into all-right and all-wrong bins and report an error of 0.50 for a
        # sample whose true calibration gap is 0.40. Identical scores are one
        # population and must be binned as one. This matters here because a
        # verifier-anchored signal produces heavy ties by construction.
        raw = [round(b * n / bins) for b in range(bins + 1)]
        edges = [0]
        for e in raw[1:-1]:
            e = max(e, edges[-1])
            while 0 < e < n and probs[order[e]] == probs[order[e - 1]]:
                e += 1
            edges.append(e)
        edges.append(n)
        total = 0.0
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            if hi <= lo:
                continue
            idx = order[lo:hi]
            m = len(idx)
            conf = sum(probs[i] for i in idx) / m
            acc = sum(hits[i] for i in idx) / m
            total += (m / n) * abs(conf - acc)
        return total

    if scheme != "equal_width":
        raise ValueError("scheme must be 'equal_mass' or 'equal_width'")

    conf_sum = [0.0] * bins
    acc_sum = [0.0] * bins
    count = [0] * bins
    for p, h in zip(probs, hits):
        b = min(int(p * bins), bins - 1) if p > 0 else 0
        conf_sum[b] += p
        acc_sum[b] += h
        count[b] += 1
    return sum((count[b] / n) * abs(conf_sum[b] / count[b] - acc_sum[b] / count[b])
               for b in range(bins) if count[b])


# ---------------------------------------------------------------------------
# Reporting honestly on a small, deliberately balanced corpus
# ---------------------------------------------------------------------------

class Interval(NamedTuple):
    point: float
    lo: float
    hi: float

    def render(self, pct: bool = True) -> str:
        f = 100.0 if pct else 1.0
        u = "%" if pct else ""
        return f"{self.point * f:.1f}{u} [{self.lo * f:.1f}, {self.hi * f:.1f}]"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a proportion.

    EVERY per-type recall figure must be quoted with one of these. The
    stratified corpus gives a fixed number of instances per anomaly type, and
    at small counts a bare percentage is not a measurement: 2 of 2 detected
    is "100% recall" with a lower bound near 0.34, which is to say it is
    consistent with a system that misses two thirds of them. Wilson rather
    than the normal approximation because the normal interval is degenerate
    exactly where our counts live - it returns zero width at 0/n and n/n, and
    can run outside [0, 1].

    Use it to SIZE the corpus, not just to decorate the result: pick the
    instances-per-type that brings the half-width under the precision the
    claim needs, then generate that many. Generation is free.
    """
    if n <= 0:
        return Interval(0.0, 0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def required_n_for_halfwidth(target_halfwidth: float, p: float = 0.9,
                             z: float = 1.96) -> int:
    """Instances per class needed for a Wilson half-width at most the target.

    Answers "how big must the corpus be?" with a number instead of a habit.
    """
    n = 1
    while n < 100_000:
        iv = wilson_interval(round(p * n), n, z)
        if (iv.hi - iv.lo) / 2 <= target_halfwidth:
            return n
        n += 1
    return n


def precision_at_prevalence(recall: float, false_positive_rate: float,
                            prevalence: float) -> float:
    """Precision the system would show at a DIFFERENT base rate.

    Recall and FPR are properties of the detector and survive the move.
    Precision is not: it is a function of how rare the thing is. Our corpus
    plants anomalies in most chains because that is the only way to measure
    per-type recall at all, so its precision figure describes a world where
    most invoices are wrong. Quoting that number for a production AP queue,
    where the rate is a fraction of a percent, would overstate precision by
    an order of magnitude, and it is the single easiest number in this
    project to be accidentally dishonest about.

    So: report measured precision WITH the corpus prevalence beside it, and
    this projection at the prevalence a buyer would actually see.
    """
    tp = recall * prevalence
    fp = false_positive_rate * (1.0 - prevalence)
    return tp / (tp + fp) if (tp + fp) else 0.0


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
    risk_ci_lo: Optional[float] = None
    risk_ci_hi: Optional[float] = None
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None
    brier_before: Optional[float] = None
    brier_after: Optional[float] = None
    corpus_prevalence: Optional[float] = None
    projected_prevalence: Optional[float] = None
    projected_precision: Optional[float] = None

    def render(self) -> str:
        L = [f"tau = {self.tau:.4f}   ({self.strategy})",
             f"  coverage        : {self.coverage:.1%} auto-resolved",
             f"  escalation rate : {self.escalation_rate:.1%}"
             + ("   <-- at/over the kill-metric"
                if self.escalation_rate > 0.5 else ""),
             f"  risk in covered : {self.risk:.2%} "
             f"({self.n_errors}/{self.n_covered} wrong)"]
        if self.risk_ci_lo is not None:
            L.append(f"  risk 95% CI     : [{self.risk_ci_lo:.2%}, "
                     f"{self.risk_ci_hi:.2%}]")
        if self.corpus_prevalence is not None:
            L.append(f"  corpus prevalence: {self.corpus_prevalence:.1%} "
                     f"(BALANCED BY DESIGN — precision above is measured at "
                     f"this rate, not a deployment rate)")
        if self.projected_precision is not None:
            L.append(f"  precision @ {self.projected_prevalence:.2%} "
                     f"prevalence: {self.projected_precision:.1%}")
        if self.ece_before is not None:
            L.append(f"  ECE (equal-mass): {self.ece_before:.4f} -> "
                     f"{self.ece_after:.4f}")
        if self.brier_before is not None:
            L.append(f"  Brier           : {self.brier_before:.4f} -> "
                     f"{self.brier_after:.4f}   (proper score; trust this "
                     f"where it disagrees with ECE)")
        return "\n".join(L)


def report_for_threshold(scores: Sequence[float], correct: Sequence[bool],
                         tau: float, strategy: str) -> ThresholdReport:
    covered = [i for i, s in enumerate(scores) if s >= tau]
    errors = sum(1 for i in covered if not correct[i])
    n = len(scores)
    cov = len(covered) / n if n else 0.0
    risk_ci = wilson_interval(errors, len(covered)) if covered else None
    return ThresholdReport(
        tau=float(tau), strategy=strategy, coverage=cov,
        risk=errors / len(covered) if covered else 0.0,
        n_covered=len(covered), n_errors=errors,
        escalation_rate=1.0 - cov,
        risk_ci_lo=risk_ci.lo if risk_ci else None,
        risk_ci_hi=risk_ci.hi if risk_ci else None,
    )
