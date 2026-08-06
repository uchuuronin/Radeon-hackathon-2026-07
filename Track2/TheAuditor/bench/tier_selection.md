# Tier selection

One row per served configuration, appended by `bench/sweep.sh`. Nothing is
written here by hand: a hand-written row cannot be traced to a run manifest,
and a number nobody can re-derive is a number a judge is right to discount.

    VRAM_GB=21.4 UTIL=82 bash bench/sweep.sh <served-model-name> <label>

VRAM and utilisation come from `bench/throughput.sh`, which samples `rocm-smi`
alongside a run. They are not guessable from inside the process, so a row
without them says so rather than leaving the cell blank.

## What each column has to be there for

| column | why a row is not comparable without it |
|---|---|
| **config** | model **and** quantisation. Quantisation damage is not uniform across models: what INT4 does to one 8B does not transfer to another, so the pairing is the unit, not either half. |
| **guided** | `guided_json`, `response_format` or `off`. If the server rejected the flag and the client fell back, the accuracy number means something different and the row must say which. |
| **mode** | `exact` or `relaxed`. Relaxed measures whether the model READ the document; exact measures whether it obeyed the output contract. Different questions, and a single number hides one. |
| **n** | field instances the accuracy rests on. Without it no interval can be computed and the percentage is decorative. |
| **line_numeric** | **the tier decision metric.** Quantity, unit price, line total, on their own. A configuration can score 89-99% on document totals and ~48% on item-level amounts in the same run; a 4B collapsed to roughly 19% on line-item numerics while still reading party names fluently. Reconciliation walks line items, so an average is the wrong number. |
| **identifier** | `doc_number` and `references`, scored apart. The hardest field class in every published comparison (0/O, 1/l), and the one the linker runs on: a dropped character there does not corrupt a number, it silently detaches a document from its chain. |
| **doc_numeric** | totals. Kept beside line_numeric precisely so the gap between them is visible rather than averaged away. |
| **precision_loss** | right value, wrong stated precision. "4500" and "4500.00" are equal in value and the verifier derives its tolerance band from stated decimals, so this counts a 100x change in downstream strictness that relaxed scoring forgives. |
| **cache** | prefix-cache hit rate. The frozen prefix is ~3.7k tokens against a ~130-token document, so this should sit well above 90%. A low number means the prefix is not byte-stable, which fails silently and costs roughly three times the prefill for the whole run. |
| **docs/s @ util** | throughput **with** the GPU utilisation it was measured at. N docs/sec at 40% utilisation is an unfinished measurement, not a systems result. |
| **peak VRAM** | from `rocm-smi`, never from the model card. It is what decides whether both tiers stay resident, which is what makes routing cost zero model swap. |
| **run health** | ok/sent, plus parse failures, API errors and truncations. A row where the server fell over is a finding about the run, not the model, and must not eliminate a configuration. |

## Choosing on this table

Two rows whose Wilson intervals **overlap** have not been separated by this
measurement, however different their point estimates look. Choosing between
them on the point estimate is a coin flip that would then have to be defended
on camera. If the intervals overlap and the decision matters, score more
documents rather than picking.

Record the configurations that **failed to initialise** too, in
`bench/quant_probe.md`. "AWQ INT4 would not load on gfx1100 at vLLM x.y.z" is
worth more in the spec than one more successful row: it is evidence the choice
was measured rather than assumed, which is what the quantisation bonus is
asking for.

<!-- rows are appended below this line by bench/sweep.sh -->
