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
(EN 16931 arithmetic identities BR-CO-10/13/15 and the BR-DEC two-decimal
family, source-evidence checks, precision-inferred tolerance) → near-free signature memory (human-approved
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
    tests/          640+ tests; no GPU, no network, sub-2s
    bench/          conformance harness + extraction scorer
    vendor/         CEN artefacts, fetched not committed
    config/         policy.json — auto-resolve gate, per-vendor date order
    bench/, infra/  model serving, quantisation sweeps, pinned versions

## Reproduce (any machine, no GPU required for this half)

    pip install -r requirements.txt
    $env:PYTHONPATH = "src"          # PowerShell — once per shell
    export PYTHONPATH=src            # bash

    python -m pytest                                        # full suite
    python data/generator/gen.py --n 510 --seed 1337 --out data/generated
    python -m verify "data/generated/records.jsonl" -q      # rung 0 + ladder report
    python -m verify "data/fixtures/corrupted/*.json"       # every check firing

The dataset is a pure function of (generator, seed). The canonical corpus is
seed **1337**, **510 chains** (6180 documents in two layouts). `data/generated/`
is gitignored; the seed is the artifact.

510 is 30 turns of a 17-slot stratified table, so every category appears
exactly 30 times whatever the seed. 30 is not a round number picked by feel:
it is what `calibrate.required_n_for_halfwidth` returns for a Wilson
half-width under ten points, and every per-type rate we report carries that
interval. The earlier 20-chain corpus gave two instances per type, where
"100% recall" has a lower bound near 0.34 and is not a measurement.

Composition per 17 chains: 5 clean, 6 anomaly types once each, and 6
price-drift cases **placed relative to the cross-document tolerance band** at
0.50x, 0.80x and 0.95x (negatives, must not flag) and 1.05x and 1.20x
(positives, must flag), plus one gross drift. The 0.95/1.05 pair differs by a
tenth of the band and demands opposite verdicts; without cases there the
whole region around the decision boundary is empty and every threshold in it
scores identically. Labels are assigned from the **achieved** delta after
cent quantisation, never from the intended one, so the key cannot disagree
with the documents.

**Prevalence is balanced by design and precision must be read accordingly.**
About 70% of chains carry an anomaly, because that is the only way to measure
per-type recall at this corpus size. Recall transfers to a production base
rate; precision does not. Every precision figure is reported with the corpus
prevalence beside it and with `calibrate.precision_at_prevalence` projecting
it onto a realistic rate.

## Dependency audit (the locality claim, checkable)

Runtime dependency of everything in `src/`, `data/` and `tests/`: **pydantic**.
Dev: pytest, ruff. Verify it yourself:

    grep -rE "^\s*(import|from)\s+(requests|httpx|urllib|socket|openai|anthropic|torch|vllm|boto3|google)" src/ data/ tests/

returns nothing. The GPU half uses the OpenAI-compatible *client
library* pointed exclusively at a local vLLM endpoint on this machine; the
library is local and no remote endpoint is configured anywhere. Model
serving dependencies and pinned ROCm/vLLM/PyTorch versions: `infra/versions.md`.

Nothing in this system trains on customer data. The only artefacts that
persist across cases are (a) human-approved resolution precedents with
provenance, and (b) a monotone calibration table over the system's own
confidence scores — a few hundred floats, no amounts, no party names,
inspectable in full.

## Conformance: checked against the CEN reference, not just asserted

Our verifier claims to implement EN 16931 calculation rules. That claim used to
rest on us having read the standard and written Python we believed matched it,
which is unfalsifiable by anyone who does not re-read the standard themselves.

CEN publishes those rules as machine-readable Schematron and the European
Commission publishes a compiled XSLT of it (EUPL v1.2). So we run our own
generated invoices through the reference implementation and compare verdicts:

    python bench/conformance.py                # PYTHONPATH=src

The artefacts (EUPL v1.2, pinned to a tag) download on first run: no shell
script, no git, no execution-policy prompt, identical on Windows and Linux.
`python infra/fetch_cen_artefacts.py` fetches them separately if you would
rather do it up front, and `--no-fetch` makes the harness fail instead of
downloading.

| corpus | result |
| --- | --- |
| Clean generated invoices | no rule under test fires on either side |
| Our mutation sweep: one break per rule, per invoice | 2000 comparisons, 100% agreement |
| **CEN's own rule fixtures, strict precision** | **48 comparisons, 100% agreement, Wilson [92.6%, 100%]** |
| CEN's own rule fixtures, inferred precision | 89.6%, and the gap is a documented design difference, below |

The third row is the one that matters, and it is worth more than the second
despite being forty times smaller. Our mutation sweep uses documents we wrote,
breaks we chose and expectations we set, so it inherits whatever we assumed an
invoice looks like. The artefacts ship a directory of rule unit-tests
(`test/Invoice-unit-UBL/BR-*.xml`) in which every fragment carries an expected
outcome written by the standard's own maintainers: `<success>BR-CO-15</success>`
or `<error>BR-CO-15</error>`. Running those through our verifier removes us from
the oracle entirely.

They are also far harder. Our mutants break an identity by 25 or 100 units;
CEN's BR-CO-15 error case states a gross total of 1250.01 where 1250.00 is
required. One cent.

**It found two real defects in our verifier immediately.**

The tolerance floor was 0.01, and the comparison is inclusive, so a one-cent
discrepancy passed on every within-document identity: `delta 0.01 <= band 0.01`.
One cent is the smallest meaningful financial error and the floor sat exactly
where it blinded us. Worse, `tests/test_schemas_contract.py` contained a test
named `test_floor_prevents_flagging_a_cent` that asserted this behaviour, so the
suite was defending the bug. The floor is now 0.005, a half-ULP at cent
precision, which agrees with the inference model instead of being a round
number. The test now asserts the opposite and explains why.

The UBL reader took the first `TaxTotal` element. An invoice in DKK may restate
its VAT in EUR for a tax authority, so the first one can be the wrong currency;
and two differing amounts in the same currency is itself an error, because there
is then no single VAT total for the identity to use. Both cases now resolve
correctly, and the second yields no tax total rather than silently picking one.

**The remaining disagreement is a finding, not a bug.** Under inferred precision
we disagree with CEN on 5 of 48 fixtures, all the same shape: CEN states a tax
amount as `250` and a total one cent out, and calls it an error. Our model reads
`250` as a value someone rounded, standing for anything in [249.5, 250.5), so a
cent is well inside the band. Both readings are correct for their own context:
inference is right for documents of unknown provenance, which is what an
extraction pipeline handles, and strict is right for a conformant e-invoice
where the standard already fixes precision at two decimals. The context is
therefore selectable (`schemas.STRICT_PRECISION`) rather than assumed, and the
harness reports both numbers.

Scope, stated plainly: 48 of roughly 909 fixtures are comparable, because the
rest test rules we do not implement. The harness prints that count rather than
quietly excluding them.

**Why this is worth more than the unit tests.** Our unit tests were written by
the same person who wrote the verifier, from the same reading of the standard.
If that reading is wrong, the tests are wrong in the same direction and every
one still passes. A second implementation, written by other people from the
normative text, does not share our misreadings.

It found a real defect within minutes. Our generator emitted `EA` as the unit
of measure for "each"; `EA` is not in the UN/ECE Recommendation 20 list that
EN 16931 requires (BR-CL-23), and the correct code is `H87`. Our verifier does
not check codelists, so our generator and our fixtures were wrong together and
no internal test could have told us. `tests/test_ubl.py` now guards it.

The harness also reports the rules CEN checks and we do not, rather than hiding
them: we implement BR-CO-10/13/15/16 and the BR-DEC decimal family, and CEN
implements the whole standard. BR-CO-16 (amount due = total − paid + rounding)
was added because the differential run surfaced it as a CEN-only finding, and it
is the rule a payment chain actually turns on: a part-payment recorded against
the wrong invoice satisfies every other identity and breaks only this one.
Rules such as BR-S-08 and BR-Z-10 remain CEN-only, which is the honest place for
the gap to appear.

`saxonche` is a dev dependency used only by this harness. It runs XSLT locally,
makes no network call, and nothing in `src/` imports it, so the no-egress claim
is unaffected.

## What the format-robustness claim does and does not cover

Layouts A and B carry identical values through disjoint label vocabularies,
inverted section ordering and different stated numeric precision, and layout
C is sealed in `data/fixtures/holdout/` until demo day. That establishes
robustness to **surface serialisation**, which is the claim we make.

It is not yet a template-generalisation result in the sense VRDU's unseen
template setting means: one held-out layout is a single sample, and it
renders the same underlying chains as the training layouts, so the content is
not held out with it. Stating the narrower claim is deliberate. Widening it
needs several structurally distinct layouts held out over held-out chains,
and that is a scoped decision rather than an oversight.

## Status

Stage 0 (contract, generator, deterministic layer, instrumentation,
calibration machinery, conformance harness) complete: 659 tests, no GPU, no
network, under two seconds. Extraction, reconciliation, memory and the analyst console land in
Stages 1–3.
