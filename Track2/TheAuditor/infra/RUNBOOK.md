# Runbook: closing Checkpoint 0

Everything on the GPU side is written, unit-tested and now **exercised end to
end** against a loopback replay server. What remains genuinely needs the card.

Two conditions of Checkpoint 0 are still red: the card is not characterised and
no model has served. Nothing on the deterministic side moves either of them.
This is the shortest path from an empty instance to a printed verdict.

---

## 0 · Before the instance (no GPU, five minutes)

Do this on a laptop. It is free and it is where harness bugs are cheap.

    pip install -r requirements.txt
    export PYTHONPATH=src

    python -m pytest                                    # 666, under 2s
    python data/generator/gen.py --n 510 --seed 1337 --out data/generated
    python bench/selftest.py                     # the extraction chain, no GPU

`selftest.py` starts `bench/mock_server.py` on loopback, drives the real
client, parser, driver and scorer over a real socket, and asserts that each
injected failure moves its own counter and no other. It must print
**SELFTEST PASSED** before the instance is worth paying for. `perfect` scoring
below 100% is a harness bug, not a model result, and finding one here costs
nothing.

---

## 1 · Instance up, card characterised

    bash infra/characterise.sh | tee infra/versions.md

This file becomes the README environment section and reproducibility is an
explicit submission requirement, so it is written on day one, not recalled
later. The two lines that decide everything downstream are the `gfx` target and
the VRAM figure: card generation fixes which quantisation methods exist at all.

Instance setup, from the track notices: GH-proxy-stable image, **Persistent
(PVC)** storage, SSH key added in Profile with the toggle enabled at template
creation. Destroy the instance when done. There is a scheduled platform stress
test that kills all instances, announced in-channel; PVC is what makes that
survivable.

---

## 2 · Serve any model. **This is THE gate.**

    bash infra/serve.sh Qwen/Qwen3-8B          # BF16, no quantisation
    curl http://localhost:8000/v1/models

Any model answering beats the right model not answering. A served small model
is 40 points of core-inference-on-Radeon; a perfect quantisation plan that
never served is zero.

> **If this is not green, the fallback fires now, not after a sweep.**
> One model, BF16, no quantisation matrix. Take the loss on the 20-point bonus
> rather than the 40 points sitting beside it. A late fallback costs both.
> Fallback order: BF16 vLLM → llama.cpp ROCm (the Lemonade build reference) →
> say so the same day, because the deterministic half is unaffected and stays
> demoable, but the plan changes shape and it changes better early.

Then confirm the endpoint really is answering our prompt, not just alive:

    python bench/gate.py --model Qwen/Qwen3-8B

---

## 3 · Which quantisations initialise at all

    bash infra/quant_probe.sh Qwen/Qwen3-8B | tee bench/quant_probe.md

One hour, and it eliminates half the sizing-sweep matrix before that sweep is
scheduled. It does
not measure quality; it asks whether a method loads and emits twenty non-garbage
tokens.

Expect, and record either way:

- **FP8 does not exist on consumer RDNA3 (gfx1100).** Instinct-class only. The
  plan is AWQ INT4 + BF16, not INT4 + FP8.
- **AWQ has historically been weak on ROCm**, no Marlin kernels, unsupported
  outright for a period.
- **BNB-nf4 is the worst measured option** and a common vLLM/HuggingFace
  default. Avoid it for numeric work.
- A method that **loads but emits empty or replacement characters** is a
  failure, and a more dangerous one than a method that refuses to load: a sweep
  would score it as a very bad model rather than as a broken kernel.

**Negative results are the deliverable.** "AWQ INT4 would not initialise on
gfx1100 at vLLM x.y.z, here is the error" is worth more in the spec than one
more successful row.

---

## 4 · The extraction prompt, on the free shared endpoint

The prompt is built and cache-safe. Develop against the ten hand-written market
fixtures (`data/fixtures/records/`) on the free Qwen/DeepSeek endpoint before
spending a credit.

Before changing anything in it, know what it costs: blocks 1–4 are frozen and
byte-identical, ~3.7k tokens against a ~130-token document, so a cache hit skips
the overwhelming majority of prefill on every document after the first. **One
changed byte anywhere above the document caches nothing, silently.** vLLM logs
the hit rate; `run.py` prints it and warns below 30%.

Three things to check on the first real run, all of which are open questions
the deterministic side is waiting on:

1. **Does anything in the fixtures print three decimals in a money field?**
   If yes, `BR_DEC_MAX_2` is wrong about the world and the check changes
   before it manufactures false failures in Stage 1.
2. **Does the extractor emit `null` or `0` for absent money on a goods
   receipt?** A zero silently destroys partial-shipment detection instead of
   failing loudly. `bench/mock_server.py`'s `zero_for_absent` profile shows
   exactly how it presents in the scorer: as hallucinated values, not dropped
   ones.
3. **Anchor grounding, yes or no.** Wire-breaking after the prompt
   freezes; cheap before. The standing recommendation is yes. If yes, it should
   land in
   the same wire break as anything else outstanding — two breaks is the outcome
   to avoid.

---

## 5 · The sizing sweep

Only once serving and the quantisation probe are green. Once per surviving
configuration:

    bash bench/throughput.sh <model> <label>       # find the concurrency knee
    VRAM_GB=<peak> UTIL=<mean> bash bench/sweep.sh <model> <label>

`throughput.sh` sweeps `--concurrency` while sampling `rocm-smi` at 1 Hz, so
every docs/sec arrives with the utilisation it was measured at. Take the knee:
past it you are buying latency with no throughput, and p95 shows it first.

`sweep.sh` scores 60 documents and appends one comparable row to
`bench/tier_selection.md`. The column contract is in that file's header and it
is not optional: a row missing `mode` or `n` cannot be compared to another row.

**What the table is for.** This is the shape you are looking for, and it is only
visible because the field classes are never averaged together:

| config | line_numeric | identifier | doc_numeric |
|---|---|---|---|
| a good configuration | 100.0% [99%,100%] | 100.0% [97%,100%] | 100.0% [99%,100%] |
| one that looks fine on average | 83.8% [81%,86%] | 100.0% [97%,100%] | 100.0% [99%,100%] |

*(illustrative, from `bench/mock_server.py`. Never put replay numbers in the
spec.)*

The second row reads totals and identifiers perfectly and drops line-item
money. Averaged, it looks respectable. Reconciliation walks line items, so it is
useless. That gap is the entire reason the tier decision metric is
`line_numeric` alone.

Then measure the ratio the cost model is blocked on:

    python bench/tier_weight.py runs/<fast>/…manifest.json runs/<precise>/…manifest.json

`TIER_WEIGHT` is a 3.0 placeholder in `src/ladder.py`. Until it is measured,
cost per document cannot be reported honestly and the ~50% escalation
kill-metric has nothing to sit against — that figure is our target, not an
industry constant, and the real break-even falls out of this ratio.

---

## 6 · Run the gate

    python bench/gate.py --model <served-model-name>

Seven conditions, one printed verdict, no checklist anyone ticks by hand.

---

## Things that quietly cost points if nobody does them

- **Keep `src/extraction/` the only file importing a model client.** The
  dependency-audit grep is the README's locality evidence and it runs on
  camera. `bench/mock_server.py` is stdlib-only and lives outside `src/` for
  this reason; it must never be imported by the pipeline.
- **Never point anything at `data/fixtures/holdout/`.** Not the prompt, not the
  sweep, not a quick sanity check. It is the only unseen-layout evidence the
  project will ever have and it is worth exactly as much as the discipline
  around it.
- **No API keys anywhere in configuration.** One committed key in a public PR
  undoes the entire locality claim in front of a judge. The client prints a
  warning if any provider key is set in the environment; unset them before the
  demo.
- **The PR touches nothing outside `Track2/TheAuditor/`,** including
  `.github/workflows/`. Run tests locally.
- **Record the failures, not just the winner**, in `bench/quant_probe.md`.
- **Say so the moment the serving gate slips.** The deterministic half is
  unaffected and stays demoable, but the plan changes shape and it changes
  better early.
