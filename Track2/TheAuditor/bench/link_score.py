"""Score the linker against the generator's own ground truth.

Closes the gap the working brief flagged in §2.3: this measurement lived at
/tmp/lscore.py, uncommitted -- "a number nobody can re-derive is a number a
judge is right to discount." No GPU, no network, deterministic (everything
below the CLI entrypoint is seeded).

METRIC
------
Pairwise precision/recall over "these two documents belong to the same
deal", plus exact-chain-set recovery (does the predicted chain's doc_id set
equal the true chain's doc_id set, exactly).

Pairwise rather than per-document accuracy, because the failure mode that
matters is asymmetric (see src/linker/link.py's module docstring): a WRONG
link merges two deals and poisons every downstream comparison silently; a
MISSED link isolates a document, which is visible and safe. Pairwise
precision/recall separates those two failure directions; a per-document
accuracy number would blend them into one figure that hides which kind of
mistake a configuration is making.

    precision = true-positive pairs / all pairs the linker put together
    recall    = true-positive pairs / all pairs that truly belong together

CORRUPTION MODEL
----------------
Two independent corruptions, applied to `references` only (never to
doc_number, doc_date, party_name, or amounts -- those are the attributes the
ATTRIBUTE_MATCH fallback needs to still agree, which is the point of testing
recall under corruption rather than under general data loss):

  glyph      each character in a stated reference has `rate` probability of
             being replaced by one of its LOOKALIKES confusions (the same
             table link.py's REPAIRED_REFERENCE rung repairs against). This
             is what "10% corrupted" / "30% corrupted" means below.
  drop       the reference is removed from the document entirely (nothing
             to repair, nothing to disambiguate -- forces ATTRIBUTE_MATCH
             or isolation).

USAGE
-----
    PYTHONPATH=src python bench/link_score.py                  # full table
    PYTHONPATH=src python bench/link_score.py --n 510 --seed 1337
    PYTHONPATH=src python bench/link_score.py --md bench/link_score.md
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linker.link import LinkPolicy, LOOKALIKES, link  # noqa: E402
from schemas import AnswerKey, CanonicalDoc, Chain  # noqa: E402


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

def corrupt_glyph(docs: list[CanonicalDoc], rate: float,
                  rng: random.Random) -> list[CanonicalDoc]:
    """Return docs with each reference character independently confused at
    probability `rate`. Confusions drawn from the SAME LOOKALIKES table the
    linker's repair rung uses -- this measures whether repair earns its
    keep against its own intended failure mode, not against arbitrary noise.
    """
    out = []
    for d in docs:
        new_refs = []
        for ref in d.references:
            chars = list(ref)
            for i, ch in enumerate(chars):
                if ch in LOOKALIKES and rng.random() < rate:
                    chars[i] = rng.choice(LOOKALIKES[ch])
            new_refs.append("".join(chars))
        out.append(d.model_copy(update={"references": new_refs}))
    return out


def corrupt_drop(docs: list[CanonicalDoc], rate: float,
                 rng: random.Random) -> list[CanonicalDoc]:
    """Return docs with each reference independently removed at `rate`."""
    out = []
    for d in docs:
        kept = [r for r in d.references if rng.random() >= rate]
        out.append(d.model_copy(update={"references": kept}))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkScore:
    precision: float
    recall: float
    exact_chains: int
    total_chains: int
    tp: int
    fp: int
    fn: int

    @property
    def label(self) -> str:
        return f"{self.precision:.4f} / {self.recall:.4f} / {self.exact_chains}/{self.total_chains}"


def _pairs_within(groups: list[list[str]]) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for g in groups:
        for a, b in combinations(sorted(g), 2):
            out.add(frozenset((a, b)))
    return out


def score(predicted: list[Chain], truth: AnswerKey) -> LinkScore:
    true_groups = [ck.doc_ids for ck in truth.chains]
    pred_groups = [c.doc_ids for c in predicted]

    true_pairs = _pairs_within(true_groups)
    pred_pairs = _pairs_within(pred_groups)

    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    true_sets = {frozenset(g) for g in true_groups}
    pred_sets = {frozenset(g) for g in pred_groups}
    exact_chains = len(true_sets & pred_sets)

    return LinkScore(precision=precision, recall=recall,
                     exact_chains=exact_chains, total_chains=len(true_groups),
                     tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# The five conditions the brief's table reports
# ---------------------------------------------------------------------------

def run_condition(name: str, docs: list[CanonicalDoc], truth: AnswerKey,
                  rng: random.Random, glyph_rate: float = 0.0,
                  drop_rate: float = 0.0,
                  policy: LinkPolicy = LinkPolicy()) -> tuple[str, LinkScore]:
    working = docs
    if glyph_rate:
        working = corrupt_glyph(working, glyph_rate, rng)
    if drop_rate:
        working = corrupt_drop(working, drop_rate, rng)
    predicted = link(working, policy=policy)
    return name, score(predicted, truth)


def full_table(docs: list[CanonicalDoc], truth: AnswerKey,
               seed: int) -> list[tuple[str, LinkScore]]:
    rows = []
    rows.append(run_condition("clean", docs, truth, random.Random(seed)))
    rows.append(run_condition("10% glyph-corrupted", docs, truth,
                              random.Random(seed + 1), glyph_rate=0.10))
    rows.append(run_condition("30% glyph-corrupted", docs, truth,
                              random.Random(seed + 2), glyph_rate=0.30))
    rows.append(run_condition("30% corrupted, repair off", docs, truth,
                              random.Random(seed + 2), glyph_rate=0.30,
                              policy=LinkPolicy(repair_references=False)))
    rows.append(run_condition("30% references dropped", docs, truth,
                              random.Random(seed + 3), drop_rate=0.30))
    return rows


def render_markdown(rows: list[tuple[str, LinkScore]]) -> str:
    lines = ["| condition | precision | recall | exact chains |",
            "|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s.precision:.4f} | {s.recall:.4f} | "
                    f"{s.exact_chains}/{s.total_chains} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_corpus(out_dir: Path, layout: str) -> tuple[list[CanonicalDoc], AnswerKey]:
    from schemas import ExtractedRecord
    docs = []
    for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = ExtractedRecord.model_validate_json(line)
        if rec.meta.layout is not None and str(rec.meta.layout) != layout:
            continue
        docs.append(rec.doc)
    truth = AnswerKey.model_validate_json(
        (out_dir / "answer_key.json").read_text(encoding="utf-8"))
    return docs, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=510)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--layout", default="layout_a")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="existing data/generated dir; generated fresh if omitted")
    ap.add_argument("--md", type=Path, default=None,
                    help="write the table to this path in addition to stdout")
    args = ap.parse_args()

    if args.corpus is None:
        import os
        import subprocess
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="link_score_corpus_"))
        root = Path(__file__).resolve().parents[1]
        # gen.py does `from schemas import ...`, which only resolves if
        # PYTHONPATH includes src/. Pytest's `pythonpath` ini setting (and
        # any PYTHONPATH the caller's shell happens to have exported) does
        # NOT propagate to a subprocess -- only real environment variables
        # do. Set it explicitly so this works regardless of how link_score.py
        # itself was invoked, instead of silently depending on the caller's
        # shell state.
        env = {**os.environ,
              "PYTHONPATH": str(root / "src") + os.pathsep
                            + os.environ.get("PYTHONPATH", "")}
        subprocess.run(
            [sys.executable, str(root / "data" / "generator" / "gen.py"),
             "--n", str(args.n), "--seed", str(args.seed),
             "--out", str(tmp), "--layouts", args.layout],
            check=True, env=env)
        corpus_dir = tmp
    else:
        corpus_dir = args.corpus

    docs, truth = _load_corpus(corpus_dir, args.layout)
    rows = full_table(docs, truth, args.seed)
    table = render_markdown(rows)
    print(f"linker score -- {len(docs)} documents, {len(truth.chains)} true "
         f"chains, seed {args.seed}\n")
    print(table)
    if args.md:
        args.md.write_text(table, encoding="utf-8")
        print(f"written to {args.md}")


if __name__ == "__main__":
    main()
