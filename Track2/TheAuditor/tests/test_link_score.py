"""Tests for bench/link_score.py. No GPU, no network.

Fast, hand-built corpora exercise the scoring math directly; one small
(--n 12) real generator run smoke-tests the CLI end to end without paying
the cost of the full 510-chain corpus on every push.
"""
from __future__ import annotations

import random
import subprocess
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from schemas import AnswerKey, CanonicalDoc, Chain, ChainKey, DocType  # noqa: E402
from linker.link import link  # noqa: E402
import link_score  # noqa: E402


def _doc(doc_id, number, dtype, refs=(), party="Averill Fastener GmbH", day=1,
        total="1000.00") -> CanonicalDoc:
    return CanonicalDoc(
        doc_id=doc_id, doc_number=number, doc_type=dtype, party_name=party,
        doc_date=date(2026, 1, 1) + timedelta(days=day), currency="EUR",
        references=list(refs), total=Decimal(total),
        source_text=f"{number} {total}")


def _two_chain_corpus():
    """Two independent 3-doc chains: 6 docs, 6 true within-chain pairs."""
    docs = [
        _doc("A-1", "Q-100", DocType.QUOTE, ()),
        _doc("A-2", "SO-100", DocType.SALES_ORDER, ("Q-100",), day=2),
        _doc("A-3", "INV-100", DocType.INVOICE, ("SO-100",), day=3),
        _doc("B-1", "Q-200", DocType.QUOTE, (), party="Zelnick Tooling"),
        _doc("B-2", "SO-200", DocType.SALES_ORDER, ("Q-200",),
             party="Zelnick Tooling", day=2),
        _doc("B-3", "INV-200", DocType.INVOICE, ("SO-200",),
             party="Zelnick Tooling", day=3),
    ]
    truth = AnswerKey(chains=[
        ChainKey(chain_id="CH-A", doc_ids=["A-1", "A-2", "A-3"], generator_seed=1),
        ChainKey(chain_id="CH-B", doc_ids=["B-1", "B-2", "B-3"], generator_seed=1),
    ])
    return docs, truth


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

def test_perfect_prediction_scores_1_1():
    docs, truth = _two_chain_corpus()
    predicted = link(docs)
    s = link_score.score(predicted, truth)
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.exact_chains == 2
    assert s.total_chains == 2


def test_merged_chains_hurt_precision_not_recall():
    """Predicting one big chain when truth has two: every true pair is still
    found (recall 1.0) but half the predicted pairs are wrong (precision <
    1.0). This is the WRONG-link failure mode the linker's docstring calls
    more dangerous, so the metric must be able to see it."""
    _, truth = _two_chain_corpus()
    merged = [Chain(chain_id="CH-MERGED",
                    doc_ids=["A-1", "A-2", "A-3", "B-1", "B-2", "B-3"])]
    s = link_score.score(merged, truth)
    assert s.recall == 1.0
    assert s.precision < 1.0
    assert s.exact_chains == 0


def test_all_isolated_hurts_recall_not_precision():
    """Every document its own chain: no wrong pairs asserted (precision is
    vacuously 1.0), but every true pair is missed (recall 0.0). The MISSED
    -link failure mode -- visible, safe, and it must score as the opposite
    shape from the merged case above."""
    docs, truth = _two_chain_corpus()
    isolated = [Chain(chain_id=f"CH-{d.doc_id}", doc_ids=[d.doc_id]) for d in docs]
    s = link_score.score(isolated, truth)
    assert s.precision == 1.0
    assert s.recall == 0.0
    assert s.exact_chains == 0


def test_exact_chains_requires_exact_set_not_just_correct_pairs():
    docs, truth = _two_chain_corpus()
    # correct pairs for chain A, but chain A is missing a member -> not exact
    partial = [Chain(chain_id="CH-A", doc_ids=["A-1", "A-2"]),
              Chain(chain_id="CH-B", doc_ids=["B-1", "B-2", "B-3"])]
    s = link_score.score(partial, truth)
    assert s.exact_chains == 1  # only B is exact
    assert s.recall < 1.0


# ---------------------------------------------------------------------------
# corruption
# ---------------------------------------------------------------------------

def test_glyph_corruption_is_deterministic_under_a_seeded_rng():
    docs, _ = _two_chain_corpus()
    out1 = link_score.corrupt_glyph(docs, 0.9, random.Random(42))
    out2 = link_score.corrupt_glyph(docs, 0.9, random.Random(42))
    assert [d.references for d in out1] == [d.references for d in out2]


def test_glyph_corruption_never_touches_non_reference_fields():
    docs, _ = _two_chain_corpus()
    out = link_score.corrupt_glyph(docs, 1.0, random.Random(1))
    for orig, corrupted in zip(docs, out):
        assert corrupted.doc_number == orig.doc_number
        assert corrupted.party_name == orig.party_name
        assert corrupted.total == orig.total


def test_drop_at_rate_1_removes_every_reference():
    docs, _ = _two_chain_corpus()
    out = link_score.corrupt_drop(docs, 1.0, random.Random(1))
    assert all(d.references == [] for d in out)


def test_heavy_corruption_measurably_lowers_recall_vs_clean():
    docs, truth = _two_chain_corpus()
    clean_score = link_score.score(link(docs), truth)
    corrupted = link_score.corrupt_drop(docs, 1.0, random.Random(3))
    corrupted_score = link_score.score(link(corrupted), truth)
    assert corrupted_score.recall <= clean_score.recall


# ---------------------------------------------------------------------------
# full_table() / rendering
# ---------------------------------------------------------------------------

def test_full_table_has_five_named_conditions():
    docs, truth = _two_chain_corpus()
    rows = link_score.full_table(docs, truth, seed=7)
    names = [name for name, _ in rows]
    assert names == ["clean", "10% glyph-corrupted", "30% glyph-corrupted",
                     "30% corrupted, repair off", "30% references dropped"]


def test_markdown_table_has_a_row_per_condition():
    docs, truth = _two_chain_corpus()
    rows = link_score.full_table(docs, truth, seed=7)
    md = link_score.render_markdown(rows)
    assert md.count("\n") == len(rows) + 2  # header + separator + N rows


# ---------------------------------------------------------------------------
# CLI smoke test against a small real generator run (not the full 510)
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end_on_a_small_generated_corpus(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "link_score.py"),
         "--n", "12", "--seed", "99", "--md", str(tmp_path / "out.md")],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
    assert (tmp_path / "out.md").exists()
    content = (tmp_path / "out.md").read_text()
    assert "30% references dropped" in content
