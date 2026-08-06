"""Deterministic format robustness test.

The generator produces the same semantic chain, then materialises it into
Layout A and Layout B. The reconciler should produce the same verdict because
it operates on CanonicalDoc values, not rendered formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "generator"))

from harmonize.reconcile import reconcile
from schemas import Chain, Layout
from gen import generate_chain, materialise


SEED = 1337


def _run(docs):
    docs_by_id = {d.doc_id: d for d in docs}
    chain = Chain(
        chain_id="CH-TEST",
        doc_ids=[d.doc_id for d in docs],
    )
    verdict = reconcile(chain, docs_by_id)

    return {
        "clean": verdict.is_clean,
        "discrepancies": [
            {
                "type": d.anomaly_type,
                "docs": d.doc_ids_involved,
                "field": d.field_path,
                "delta": str(d.delta),
            }
            for d in verdict.discrepancies
        ],
    }


def test_reconcile_is_format_robust():
    mismatches = []

    for i in range(510):
        bundle = generate_chain(i, SEED)

        layout_a = materialise(bundle, Layout.A)
        layout_b = materialise(bundle, Layout.B)

        result_a = _run(layout_a)
        result_b = _run(layout_b)

        if result_a != result_b:
            mismatches.append(
                {
                    "chain": i,
                    "layout_a": result_a,
                    "layout_b": result_b,
                }
            )

    assert mismatches == []
