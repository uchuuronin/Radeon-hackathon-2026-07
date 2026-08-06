"""Drive extraction over a document set and record what it cost.

CONCURRENCY IS THE WHOLE OPTIMISATION HERE
------------------------------------------
vLLM's continuous batching only has something to batch if requests are in
flight together. A loop that sends one document, waits, then sends the next
leaves the card almost entirely idle between decodes and will report a
throughput figure that says nothing about the hardware. Submitting N requests
concurrently is what turns "we ran on a Radeon" into "we saturated a Radeon",
and the second is what the 20-point optimisation criterion is asking for.

So documents go through a thread pool. The requests are IO-bound from our side
(we are waiting on a socket), so threads are the right primitive and the GIL is
irrelevant: all the real work happens in the server process.

`--concurrency` is a knob to SWEEP, not a constant to pick. Raise it until
throughput stops improving or latency percentiles blow out, and report the
throughput at the utilisation rocm-smi shows at that point. Throughput without
its utilisation is not a systems result.

WHAT THIS RECORDS AND WHY
-------------------------
Every run writes a manifest next to the output: model, concurrency, wall clock,
tokens, prefix-cache hit rate, and per-document latency percentiles. The cache
hit rate is the one to read first, because a prefix that is not being reused
fails silently and costs roughly three times the prefill for the entire run.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extraction.client import (LocalVLLM, Usage, guided_mode,       # noqa: E402
                               thinking_disabled)
from extraction.prompt import (build_messages, guided_json_schema,  # noqa: E402
                               prefix_token_estimate, sanity_check)
from schemas import (CanonicalDoc, ExtractedRecord, ExtractionMeta,  # noqa: E402
                     Tier)


@dataclass
class RunStats:
    model: str = ""
    tier: str = ""
    concurrency: int = 1
    documents: int = 0
    ok: int = 0
    #: The model answered and the answer was not a valid CanonicalDoc.
    parse_failures: int = 0
    #: The server did not answer at all. Kept SEPARATE from parse_failures on
    #: purpose: "60 of 60 unparseable" reads in a sweep table as a model that
    #: cannot extract, when the actual event was a server that fell over. One
    #: is a finding about the model, the other is a finding about the run, and
    #: conflating them is how a configuration gets wrongly eliminated.
    api_errors: int = 0
    #: max_tokens cut the JSON off mid-object. Also presents as a parse
    #: failure, and is also not the model's fault.
    truncated: int = 0
    guided: str = ""
    thinking_disabled: bool = True
    wall_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Every document SENT, successful or not. The scorer needs this to tell a
    #: model failure from a document that was outside --limit.
    doc_ids: list[str] = field(default_factory=list)

    def as_manifest(self) -> dict:
        """Everything a sweep row needs, INCLUDING the derived numbers.

        `asdict` alone drops docs_per_s and cache_hit_rate, because they are
        properties rather than fields, so the manifest would be missing the
        two figures the run exists to produce.
        """
        d = {k: v for k, v in asdict(self).items() if k != "latencies_ms"}
        lat = sorted(self.latencies_ms)
        d.update(docs_per_s=round(self.docs_per_s, 3),
                 cache_hit_rate=round(self.cache_hit_rate, 4),
                 latency_p50_ms=round(statistics.median(lat), 1) if lat else 0.0,
                 latency_p95_ms=round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else 0.0,
                 errors=self.errors[:20])
        return d

    @property
    def docs_per_s(self) -> float:
        return self.ok / self.wall_s if self.wall_s else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def render(self) -> str:
        lat = sorted(self.latencies_ms)
        p50 = statistics.median(lat) if lat else 0.0
        p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
        expected = prefix_token_estimate()
        L = [
            "",
            f"  model            : {self.model}  (tier {self.tier})",
            f"  documents        : {self.ok} ok / {self.documents} "
            f"({self.parse_failures} unparseable, {self.api_errors} api errors,"
            f" {self.truncated} truncated)",
            f"  decoding         : guided={self.guided}  "
            f"thinking_disabled={self.thinking_disabled}",
            f"  wall clock       : {self.wall_s:.1f} s at concurrency "
            f"{self.concurrency}",
            f"  throughput       : {self.docs_per_s:.2f} docs/sec",
            f"     ^ report this WITH the rocm-smi utilisation it was measured at",
            f"  latency ms       : median {p50:.0f} / p95 {p95:.0f}",
            f"  tokens           : {self.prompt_tokens} prompt / "
            f"{self.completion_tokens} completion",
            f"  prefix cache     : {self.cache_hit_rate:.1%} of prompt tokens "
            f"served from cache",
        ]
        if self.truncated:
            L.append(f"     ^ {self.truncated} responses hit max_tokens. Raise "
                     f"--max-tokens; these are NOT extraction failures and "
                     f"must not be scored as if they were.")
        if self.api_errors:
            L.append(f"     ^ {self.api_errors} calls never reached the model. "
                     f"First distinct errors:")
            for e in dict.fromkeys(self.errors[:3]):
                L.append(f"       {e}")
        if self.prompt_tokens and self.cache_hit_rate < 0.30:
            L.append(f"     ^ LOW. The frozen prefix is ~{expected} tokens; if it "
                     f"were being reused this should be well above 50%.")
            L.append(f"       Check --enable-prefix-caching is on, and that "
                     f"nothing varies before the document.")
        return "\n".join(L)


def _parse(raw: str, doc_id: str, source_text: str) -> Optional[CanonicalDoc]:
    """Model output to CanonicalDoc.

    Guided decoding should make the fence-stripping unnecessary. It is here
    anyway because the fallback path (guided decoding unavailable, or disabled
    to measure its effect) produces markdown fences often enough that losing a
    whole sweep row to it would be silly.
    """
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        i, j = txt.find("{"), txt.rfind("}")
        if i == -1 or j <= i:
            return None
        try:
            obj = json.loads(txt[i:j + 1])
        except json.JSONDecodeError:
            return None
    # doc_id and source_text are OURS, not the model's: doc_id is the ground
    # truth handle the scorer joins on, and source_text must be the bytes we
    # sent, or the grounding check would be verifying the model against its own
    # paraphrase of the document.
    obj["doc_id"] = doc_id
    obj["source_text"] = source_text
    try:
        return CanonicalDoc.model_validate(obj)
    except Exception:                                           # noqa: BLE001
        return None


def extract_many(docs: list[tuple[str, str, Optional[str]]], llm: LocalVLLM,
                 tier: Tier = Tier.FAST, concurrency: int = 8,
                 guided: bool = True, prompt_id: str = "",
                 max_tokens: Optional[int] = None,
                 n_samples: int = 1) -> tuple[
                     list[ExtractedRecord], RunStats]:
    """docs: list of (doc_id, source_text, layout).

    `n_samples > 1` requests N sampled extractions per document IN ONE REQUEST.
    This is the only way the self-consistency signal S2 routes on can exist,
    and it has to happen HERE, on the card, because the samples cannot be
    reconstructed afterwards from a single recorded answer.

    Sent as one request with n=N so vLLM shares the prefill and batches the
    decodes: wall-clock is roughly one generation rather than N sequential
    ones. The GPU work is genuinely N, which is why the cost model counts N and
    the wall-clock does not.

    Every sample is emitted as its own ExtractedRecord carrying
    `meta.n_sample_index`, so the whole set survives to disk and every
    downstream question about agreement, calibration and routing can be
    answered on a laptop, forever, with no further GPU time.
    """
    problems = sanity_check(s for _, s, _ in docs[:20])
    if problems:
        raise RuntimeError("prompt is not cache-safe: " + "; ".join(problems))

    schema = guided_json_schema() if guided else None
    stats = RunStats(model=llm.model, tier=tier.value, concurrency=concurrency,
                     documents=len(docs), doc_ids=[d[0] for d in docs])
    out: list[ExtractedRecord] = []

    def one(item):
        doc_id, source_text, layout = item
        try:
            texts, usage = llm.complete(build_messages(source_text),
                                        guided_json=schema,
                                        max_tokens=max_tokens, n=n_samples)
        except Exception as exc:                                # noqa: BLE001
            return doc_id, [], Usage(), layout, f"{type(exc).__name__}: {exc}"
        parsed = [_parse(t, doc_id, source_text) for t in texts]
        return doc_id, parsed, usage, layout, None

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for doc_id, parsed, usage, layout, err in pool.map(one, docs):
            stats.prompt_tokens += usage.prompt_tokens
            stats.completion_tokens += usage.completion_tokens
            stats.cached_tokens += usage.cached_tokens
            if usage.truncated:
                stats.truncated += 1
            if usage.latency_ms:
                stats.latencies_ms.append(usage.latency_ms)
            if err is not None:
                stats.api_errors += 1
                stats.errors.append(err)
                continue
            if not any(p is not None for p in parsed):
                stats.parse_failures += 1
                continue
            stats.ok += 1
            for i, doc in enumerate(parsed):
                if doc is None:
                    continue        # one bad sample does not lose the others
                out.append(ExtractedRecord(
                    doc=doc,
                    meta=ExtractionMeta(
                        model_id=llm.model, tier=tier, prompt_id=prompt_id,
                        n_sample_index=i if n_samples > 1 else None,
                        layout=layout if isinstance(layout, str) else None)))
    stats.wall_s = time.perf_counter() - t0
    stats.guided = guided_mode() if guided else "off"
    stats.thinking_disabled = thinking_disabled()
    return out, stats


def load_sources(path: Path, limit: Optional[int] = None,
                 layout: Optional[str] = None) -> list[tuple[str, str, Optional[str]]]:
    """Read (doc_id, source_text, layout) from a generated records.jsonl."""
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        lay = rec["meta"].get("layout")
        if layout and lay != layout:
            continue
        docs.append((rec["doc"]["doc_id"], rec["doc"]["source_text"], lay))
        if limit and len(docs) >= limit:
            break
    return docs


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run extraction over a document set.")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--tier", default="fast", choices=["fast", "precise"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--layout", default=None)
    p.add_argument("--no-guided", action="store_true",
                   help="disable guided JSON, to measure what it is worth")
    p.add_argument("--prompt-id", default="baseline")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--samples", type=int, default=1,
                   help="sampled extractions per document, in ONE request. "
                        ">1 is what makes the self-consistency signal exist; "
                        "it cannot be reconstructed later.")
    a = p.parse_args()

    docs = load_sources(a.records, a.limit, a.layout)
    llm = LocalVLLM(model=a.model, base_url=a.base_url)
    recs, stats = extract_many(docs, llm, Tier(a.tier), a.concurrency,
                               not a.no_guided, a.prompt_id, a.max_tokens,
                               a.samples)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(r.model_dump_json() for r in recs) + "\n",
                     encoding="utf-8")
    a.out.with_suffix(".manifest.json").write_text(
        json.dumps(stats.as_manifest(), indent=2), encoding="utf-8")
    print(stats.render())
    print(f"\n  wrote {len(recs)} records -> {a.out}")


if __name__ == "__main__":
    main()
