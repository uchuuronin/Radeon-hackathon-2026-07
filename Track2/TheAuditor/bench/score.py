"""Score extraction against ground truth. No GPU, no network.

THE ONE NUMBER THAT DECIDES THE TIER PAIR
-----------------------------------------
Line-item numeric accuracy, scored on its own. Not overall field accuracy, not
a benchmark score off a model card.

This is not a preference. The published measurement that matters most to us is
a small model scoring ~89-99% on document TOTALS and ~48% on item-level amounts
in the same run, and a 4B collapsing to roughly 19% on line-item numeric fields
while still reading party names fluently. Totals are few, large and printed in
a predictable place; line items are many, small and structurally repetitive.
A configuration can look excellent on a headline average and be useless for
reconciliation, because reconciliation walks line items.

So this scorer reports four field classes separately and never averages them
into one number:

    line_numeric  quantity, unit_price, line_total     <- decides the tier
    doc_numeric   subtotal, tax, total, and friends
    identifier    doc_number, references               <- hardest class measured
    text          party_name, descriptions, dates

`identifier` gets its own class because alphanumeric identifiers are the single
hardest field for every model in the published comparisons, with 0/O and 1/l
confusions dominating, and because our linker runs on exactly those fields: a
dropped character there does not corrupt a number, it silently detaches a
document from its chain.

EXACT VERSUS RELAXED
--------------------
Both are computed, always, and every reported number states which it is.
    exact    the string matches after whitespace trimming
    relaxed  numerics compare by VALUE (4,500.00 == 4500.00 == 4500), text
             compares casefolded with punctuation collapsed
Relaxed is the honest measure of whether the model READ the document. Exact is
the honest measure of whether it obeyed the output contract. They answer
different questions and a single number would hide one of them.

Note what relaxed does NOT forgive: precision. "4500" and "4500.00" are equal
in value but state different precision, and the verifier derives its tolerance
band from stated precision, so treating them as identical would score away a
100x difference in downstream strictness. `precision_loss` counts them.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibrate import wilson_interval                          # noqa: E402
from schemas import CanonicalDoc, ExtractedRecord              # noqa: E402

FieldClass = str
CLASSES: tuple[FieldClass, ...] = ("line_numeric", "doc_numeric",
                                   "identifier", "text")

_DOC_NUMERIC = ("subtotal", "allowance_total", "charge_total", "total_excl_tax",
                "tax", "total", "rounding_amount", "amount_due")
_LINE_NUMERIC = ("quantity", "unit_price", "line_allowance", "line_total")
_TEXT = ("party_name", "doc_date", "doc_date_raw", "currency", "payment_terms")

_PUNCT = re.compile(r"[^\w]+")


def _norm_text(v: object) -> str:
    return _PUNCT.sub("", str(v or "").casefold())


def _as_decimal(v: object) -> Optional[Decimal]:
    if v is None or v == "":
        return None
    try:
        return Decimal(str(v).replace(",", "").replace(" ", ""))
    except (InvalidOperation, ValueError):
        return None


@dataclass
class Miss:
    """One wrong field. The reason is what makes a sweep row actionable."""
    doc_id: str
    field_path: str
    field_class: FieldClass
    expected: str
    got: str
    reason: str


@dataclass
class Score:
    n: dict[FieldClass, int] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    #: Truth documents the run never sent. NOT failures. Kept apart from
    #: parse_failures because a sweep runs --limit 60 against a 6180-document
    #: corpus, and folding the other 6120 in would print "6120 of 6180
    #: documents did not yield a valid record" on every row: a headline number
    #: that is pure artefact of the limit and says nothing about the model.
    not_attempted: int = 0
    #: Calls that never reached the model, read from the run manifest. The
    #: scorer alone cannot tell these from a model that answered unusably,
    #: because both leave no record, and they are opposite findings.
    api_errors: int = 0
    exact: dict[FieldClass, int] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    relaxed: dict[FieldClass, int] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    precision_loss: int = 0
    parse_failures: int = 0
    docs: int = 0
    misses: list[Miss] = field(default_factory=list)

    def _hit(self, cls: FieldClass, exact: bool, relaxed: bool) -> None:
        self.n[cls] += 1
        self.exact[cls] += int(exact)
        self.relaxed[cls] += int(relaxed)

    def rate(self, cls: FieldClass, mode: str = "relaxed") -> tuple[float, float, float]:
        hits = (self.relaxed if mode == "relaxed" else self.exact)[cls]
        iv = wilson_interval(hits, self.n[cls])
        return iv.point, iv.lo, iv.hi

    def render(self, label: str = "") -> str:
        L = [f"## {label}" if label else "## extraction score", ""]
        if self.not_attempted:
            L.append(f"scored {self.docs} of {self.docs + self.not_attempted} "
                     f"corpus documents ({self.not_attempted} not sent this run)")
        if self.parse_failures:
            unusable = self.parse_failures - self.api_errors
            L.append(f"NO RECORD: {self.parse_failures} of {self.docs} "
                     f"attempted documents produced nothing scoreable")
            if self.api_errors:
                L.append(f"  of which {self.api_errors} never reached the "
                         f"model at all (server errors, not extraction "
                         f"failures). This row is about the RUN, not the "
                         f"model, and must not be used to eliminate a "
                         f"configuration.")
            if unusable > 0 and self.api_errors:
                L.append(f"  and {unusable} were answered but unparseable.")
        L.append(f"{'class':<14}{'n':>6}  {'exact':>22}  {'relaxed':>22}")
        for cls in CLASSES:
            if not self.n[cls]:
                continue
            e, el, eh = self.rate(cls, "exact")
            r, rl, rh = self.rate(cls, "relaxed")
            L.append(f"{cls:<14}{self.n[cls]:>6}  "
                     f"{e:>7.1%} [{el:.1%},{eh:.1%}]  "
                     f"{r:>7.1%} [{rl:.1%},{rh:.1%}]")
        L.append("")
        L.append(f"precision loss (right value, wrong stated precision): "
                 f"{self.precision_loss}")
        if self.n["line_numeric"]:
            p, lo, hi = self.rate("line_numeric", "relaxed")
            L.append("")
            L.append(f"TIER DECISION METRIC — line-item numeric, relaxed: "
                     f"{p:.1%} [{lo:.1%}, {hi:.1%}] on n={self.n['line_numeric']}")
            if hi - lo > 0.20:
                L.append("  ^ interval wider than 20 points: score more "
                         "documents before choosing on this.")
        return "\n".join(L)

    #: The agreed column order. A row missing any of these is not
    #: comparable to another row, and comparing them anyway is how a tier gets
    #: chosen on noise.
    COLUMNS = ("config", "guided", "mode", "n", "line_numeric", "identifier",
               "doc_numeric", "precision_loss", "cache", "docs/s @ util",
               "peak VRAM", "run health")

    def row(self, config: str, manifest: Optional[dict] = None,
            mode: str = "relaxed", vram_gb: Optional[float] = None,
            utilisation: Optional[int] = None) -> str:
        """One markdown table row, for bench/tier_selection.md.

        Every accuracy carries its Wilson interval inline rather than in a
        footnote, because the decision the table exists for is "are these two
        configurations distinguishable", and that is an interval question. Two
        rows whose intervals overlap have not been separated by this
        measurement however different their point estimates look.
        """
        m = manifest or {}

        def cell(cls: str) -> str:
            if not self.n[cls]:
                return "n/a"
            p, lo, hi = self.rate(cls, mode)
            return f"{p:.1%} [{lo:.0%},{hi:.0%}]"

        thr = m.get("docs_per_s")
        thr_s = "not measured" if thr is None else (
            f"{thr:.2f}" + (f" @ {utilisation}%" if utilisation is not None
                            else " @ UTILISATION NOT RECORDED"))
        health = (f"{m.get('ok', self.docs)}/{m.get('documents', self.docs)} ok")
        for k, lab in (("parse_failures", "parse"), ("api_errors", "api"),
                       ("truncated", "trunc")):
            if m.get(k):
                health += f", {m[k]} {lab}"
        cache = m.get("cache_hit_rate")
        return "| " + " | ".join([
            config,
            str(m.get("guided", "?")),
            mode,
            str(self.n["line_numeric"]),
            cell("line_numeric"),
            cell("identifier"),
            cell("doc_numeric"),
            str(self.precision_loss),
            "?" if cache is None else f"{cache:.0%}",
            thr_s,
            "not measured" if vram_gb is None else f"{vram_gb:.1f} GB",
            health,
        ]) + " |"

    @classmethod
    def header(cls) -> str:
        return ("| " + " | ".join(cls.COLUMNS) + " |\n|"
                + "|".join(["---"] * len(cls.COLUMNS)) + "|")

    def top_misses(self, k: int = 10) -> str:
        if not self.misses:
            return "no misses"
        out = ["", f"first {min(k, len(self.misses))} misses:"]
        for m in self.misses[:k]:
            out.append(f"  {m.doc_id:<12} {m.field_path:<34} "
                       f"want {m.expected!r:<14} got {m.got!r:<14} {m.reason}")
        return "\n".join(out)


def _compare(score: Score, doc_id: str, path: str, cls: FieldClass,
             want: object, got: object) -> None:
    numeric = cls in ("line_numeric", "doc_numeric")

    if want is None and got is None:
        score._hit(cls, True, True)
        return
    if want is None or got is None:
        score._hit(cls, False, False)
        # Called out separately because the two directions fail differently:
        # a hallucinated value corrupts a number, a dropped one costs coverage.
        score.misses.append(Miss(doc_id, path, cls, str(want), str(got),
                                 "hallucinated (absent in truth)" if want is None
                                 else "dropped (present in truth)"))
        return

    exact = str(want).strip() == str(got).strip()
    if numeric:
        dw, dg = _as_decimal(want), _as_decimal(got)
        relaxed = dw is not None and dg is not None and dw == dg
        if relaxed and not exact:
            score.precision_loss += 1
    else:
        relaxed = _norm_text(want) == _norm_text(got)

    score._hit(cls, exact, relaxed)
    if not relaxed:
        reason = "wrong value"
        if numeric and _as_decimal(got) is None:
            reason = "not parseable as a number"
        elif cls == "identifier":
            reason = "identifier mismatch (0/O, 1/l, or a stripped prefix?)"
        score.misses.append(Miss(doc_id, path, cls, str(want), str(got), reason))
    elif not exact and numeric:
        score.misses.append(Miss(doc_id, path, cls, str(want), str(got),
                                 "PRECISION: right value, wrong decimals — "
                                 "widens the verifier tolerance band"))


def score_doc(truth: CanonicalDoc, got: CanonicalDoc, score: Score) -> None:
    d = truth.doc_id
    score.docs += 1

    _compare(score, d, "doc_number", "identifier", truth.doc_number, got.doc_number)
    _compare(score, d, "references", "identifier",
             "|".join(truth.references), "|".join(got.references))
    for f in _TEXT:
        _compare(score, d, f, "text", getattr(truth, f), getattr(got, f))
    for f in _DOC_NUMERIC:
        _compare(score, d, f, "doc_numeric", getattr(truth, f), getattr(got, f))

    # Line items match on line_id where the model supplied one, otherwise by
    # position. Position is the fallback rather than the default because a
    # model that drops one line would otherwise shift every subsequent line and
    # score zero on all of them, which reports a formatting slip as a total
    # numeric collapse.
    by_id = {li.line_id: li for li in got.line_items if li.line_id}
    for i, want_li in enumerate(truth.line_items):
        got_li = by_id.get(want_li.line_id)
        if got_li is None:
            got_li = got.line_items[i] if i < len(got.line_items) else None
        for f in _LINE_NUMERIC:
            _compare(score, d, f"line_items[{want_li.line_id}].{f}",
                     "line_numeric", getattr(want_li, f),
                     getattr(got_li, f, None) if got_li else None)
        _compare(score, d, f"line_items[{want_li.line_id}].description", "text",
                 want_li.description,
                 getattr(got_li, "description", None) if got_li else None)

    for extra in got.line_items[len(truth.line_items):]:
        score.misses.append(Miss(d, f"line_items[{extra.line_id}]", "line_numeric",
                                 "<no such line>", str(extra.line_total),
                                 "hallucinated line item"))
        score._hit("line_numeric", False, False)


def load_jsonl(path: Path) -> dict[str, CanonicalDoc]:
    out: dict[str, CanonicalDoc] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = ExtractedRecord.model_validate_json(line)
            out[rec.doc.doc_id] = rec.doc
    return out


def attempted_ids(got_path: Path) -> Optional[set[str]]:
    """Which documents the run actually sent, from the manifest beside the
    output. Written by run.py.

    Without this the scorer cannot tell "the model failed on this document"
    from "this document was never sent", and those are opposite findings.
    """
    manifest = got_path.with_suffix(".manifest.json")
    if not manifest.exists():
        return None
    ids = json.loads(manifest.read_text(encoding="utf-8")).get("doc_ids")
    return set(ids) if ids else None


def score_files(truth_path: Path, got_path: Path,
                layout: Optional[str] = None,
                attempted: Optional[set[str]] = None) -> Score:
    truth = load_jsonl(truth_path)
    if layout:
        keep = {r["doc"]["doc_id"] for r in
                (json.loads(x) for x in
                 truth_path.read_text(encoding="utf-8").splitlines() if x.strip())
                if r["meta"].get("layout") == layout}
        truth = {k: v for k, v in truth.items() if k in keep}
    got = load_jsonl(got_path)
    if attempted is None:
        attempted = attempted_ids(got_path)

    s = Score()
    for doc_id, want in truth.items():
        if attempted is not None and doc_id not in attempted:
            s.not_attempted += 1
            continue
        have = got.get(doc_id)
        if have is None:
            if attempted is None:
                # No manifest: we cannot distinguish a failure from a document
                # that was never sent, so we do not guess. Say so rather than
                # inventing a number in either direction.
                s.not_attempted += 1
                continue
            s.parse_failures += 1
            s.docs += 1
            continue
        score_doc(want, have, s)
    manifest = got_path.with_suffix(".manifest.json")
    if manifest.exists():
        s.api_errors = json.loads(
            manifest.read_text(encoding="utf-8")).get("api_errors", 0) or 0
    if attempted is None and s.not_attempted:
        print(f"  [score] no manifest beside {got_path.name}: "
              f"{s.not_attempted} truth documents had no extraction and are "
              f"EXCLUDED rather than counted as failures. Re-run through "
              f"run.py so the manifest records which documents were sent.")
    return s


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Score extraction against truth.")
    p.add_argument("--truth", type=Path, required=True,
                   help="data/generated/records.jsonl")
    p.add_argument("--got", type=Path, required=True,
                   help="the model's extracted.jsonl")
    p.add_argument("--layout", default=None, help="score one layout only")
    p.add_argument("--label", default="", help="model x quantisation, for the row")
    p.add_argument("--misses", type=int, default=10)
    p.add_argument("--mode", default="relaxed", choices=["relaxed", "exact"],
                   help="which number the ROW reports; both are always printed")
    p.add_argument("--append-row", type=Path, default=None,
                   help="append a comparable row to bench/tier_selection.md")
    p.add_argument("--vram-gb", type=float, default=None,
                   help="peak VRAM from rocm-smi, NOT from the model card")
    p.add_argument("--utilisation", type=int, default=None,
                   help="mean GPU utilisation during the run, from rocm-smi. "
                        "Throughput without it is not a systems result.")
    a = p.parse_args()

    s = score_files(a.truth, a.got, a.layout)
    print(s.render(a.label))
    print(s.top_misses(a.misses))

    if a.append_row:
        manifest = a.got.with_suffix(".manifest.json")
        m = json.loads(manifest.read_text(encoding="utf-8")) \
            if manifest.exists() else {}
        a.append_row.parent.mkdir(parents=True, exist_ok=True)
        if not a.append_row.exists() or "| config |" not in \
                a.append_row.read_text(encoding="utf-8"):
            with a.append_row.open("a", encoding="utf-8") as fh:
                fh.write("\n" + Score.header() + "\n")
        with a.append_row.open("a", encoding="utf-8") as fh:
            fh.write(s.row(a.label or a.got.parent.name, m, a.mode,
                           a.vram_gb, a.utilisation) + "\n")
        print(f"\n  appended row -> {a.append_row}")


if __name__ == "__main__":
    main()
