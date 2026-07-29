# The Auditor

A private, local-first agentic reconciliation engine for the quote-to-cash
document chain (quote → sales order → purchase order → goods receipt →
invoice → payment). It links documents to the deal they belong to, walks each
chain to find where money and terms drift, explains why, and lets an analyst
work the exception queue conversationally.

Every inference runs locally on a single AMD Radeon PRO W7900 via ROCm and
vLLM. **No remote API is used for any core function and no financial data
leaves the machine.**

Track 2 · AMD AI DevMaster Hackathon 2026.

## Architecture in one paragraph

An escalation ladder ordered by cost: free deterministic verification
(EN 16931 arithmetic identities BR-CO-10/13/15, source-evidence checks,
precision-inferred tolerance) → near-free signature memory (human-approved
precedents only) → a fast quantised model tier → a precise tier with
self-consistency sampling → a human analyst. Never spend a GPU token you
don't have to; batch the ones you must. Confidence is anchored in the
deterministic verifier — code checking the model, never the model checking
itself — so routing needs no fragile learned calibration to function.

## Repository layout

    src/            schemas.py (the frozen A↔B wire contract), precision,
                    normalise, ladder (hierarchical cost instrumentation),
                    calibrate (threshold selection), config, verify/,
                    linker/, harmonize/, extraction/, console/
    data/generator/ seeded synthetic chain generator + anomaly injector
    data/fixtures/  records/ (10 hand-written docs in real market layouts)
                    corrupted/ (one broken invariant per check, named for it)
                    holdout/ (SEALED — see its README; do not open)
    tests/          560+ tests; no GPU, no network, sub-2s
    config/         policy.json — auto-resolve gate, per-vendor date order
    bench/, infra/  Person B: model serving, quantisation sweeps, versions

## Reproduce (any machine, no GPU required for this half)

    pip install -r requirements.txt
    $env:PYTHONPATH = "src"          # PowerShell — once per shell
    export PYTHONPATH=src            # bash

    python -m pytest                                        # full suite
    python data/generator/gen.py --n 20 --seed 1337 --out data/generated
    python -m verify "data/generated/records.jsonl" -q      # rung 0 + ladder report
    python -m verify "data/fixtures/corrupted/*.json"       # every check firing

The dataset is a pure function of (generator, seed). The canonical corpus is
seed **1337**, 20 chains: 6 clean, one anomaly of each of six types twice
over, and two below-tolerance decoys the system is scored on NOT flagging.
`data/generated/` is gitignored; the seed is the artifact.

## Dependency audit (the locality claim, checkable)

Runtime dependency of everything in `src/`, `data/` and `tests/`: **pydantic**.
Dev: pytest, ruff. Verify it yourself:

    grep -rE "^\s*(import|from)\s+(requests|httpx|urllib|socket|openai|anthropic|torch|vllm|boto3|google)" src/ data/ tests/

returns nothing. The GPU half (Person B) uses the OpenAI-compatible *client
library* pointed exclusively at a local vLLM endpoint on this machine; the
library is local and no remote endpoint is configured anywhere. Model
serving dependencies and pinned ROCm/vLLM/PyTorch versions: `infra/versions.md`.

Nothing in this system trains on customer data. The only artefacts that
persist across cases are (a) human-approved resolution precedents with
provenance, and (b) a monotone calibration table over the system's own
confidence scores — a few hundred floats, no amounts, no party names,
inspectable in full.

## Status

Stage 0–1 (data, contract, deterministic layer, instrumentation) complete.
Extraction, reconciliation, memory and the analyst console land in Stages 1–3
per the specification in `docs/`.
