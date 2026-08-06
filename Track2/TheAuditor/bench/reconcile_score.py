#!/usr/bin/env python3
"""
Benchmark the reconciler against generator ground truth.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "generator"))

from harmonize.reconcile import reconcile
from schemas import Chain

from gen import generate_chain
from inject import (
    inject_near_duplicate,
    inject_partial_shipment,
    inject_price_drift,
    inject_quantity_mismatch,
    inject_term_change,
    inject_unapplied_discount,
)


INJECTORS = {
    "PRICE_DRIFT": inject_price_drift,
    "UNAPPLIED_DISCOUNT": inject_unapplied_discount,
    "QUANTITY_MISMATCH": inject_quantity_mismatch,
    "PARTIAL_SHIPMENT": inject_partial_shipment,
    "NEAR_DUPLICATE": inject_near_duplicate,
    "TERM_CHANGE": inject_term_change,
}


def run_one(anomaly_name: str, index: int, seed: int):
    rng = random.Random(seed + index)

    if anomaly_name == "UNAPPLIED_DISCOUNT":
        docs = generate_chain(
            index,
            seed,
            force_allowance=True,
        ).docs
    else:
        docs = generate_chain(
            index,
            seed,
        ).docs

    injector = INJECTORS[anomaly_name]

    docs, planted = injector(
        docs,
        rng,
    )

    docs_by_id = {
        d.doc_id: d
        for d in docs
    }

    chain = Chain(
        chain_id="BENCH",
        doc_ids=[
            d.doc_id
            for d in docs
        ],
    )

    verdict = reconcile(
        chain,
        docs_by_id,
    )

    found = [
        d
        for d in verdict.discrepancies
        if d.anomaly_type == planted.anomaly_type
    ]

    return {
        "type": anomaly_name,
        "found": bool(found),
        "within_tolerance": planted.is_within_tolerance,
        "false_positive": (
            planted.is_within_tolerance
            and bool(found)
        ),
    }


def run(n: int, seed: int):
    results = []

    for anomaly_name in INJECTORS:
        for i in range(n):
            results.append(
                run_one(
                    anomaly_name,
                    i,
                    seed,
                )
            )

    return results


def render(results):
    grouped = defaultdict(list)

    for r in results:
        grouped[r["type"]].append(r)

    lines = [
        "| anomaly | samples | expected violations | detected | recall | false positives |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, rows in grouped.items():

        expected = [
            r
            for r in rows
            if not r["within_tolerance"]
        ]

        detected = sum(
            r["found"]
            for r in expected
        )

        recall = (
            detected / len(expected)
            if expected
            else 1.0
        )

        false_positive = sum(
            r["false_positive"]
            for r in rows
        )

        lines.append(
            f"| {name} | {len(rows)} | "
            f"{len(expected)} | "
            f"{detected} | "
            f"{recall:.4f} | "
            f"{false_positive} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=510,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    parser.add_argument(
        "--md",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    results = run(
        args.n,
        args.seed,
    )

    report = render(results)

    print(report)

    if args.md:
        args.md.write_text(
            report,
            encoding="utf-8",
        )

        print(
            f"written to {args.md}"
        )


if __name__ == "__main__":
    main()
