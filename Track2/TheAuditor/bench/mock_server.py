"""A LOCAL, OFFLINE, OpenAI-compatible stand-in for vLLM. DEV TOOL ONLY.

WHAT THIS IS NOT
----------------
It is not a model, it does not do inference, and no number it produces belongs
in a sweep row, a spec document or a demo. It is stdlib-only, binds 127.0.0.1
and nothing else, and lives in bench/ rather than src/ precisely so the
dependency-audit grep keeps returning exactly one file.

WHY IT EXISTS
-------------
Every artefact on the GPU side of the seam was written and unit-tested but never
RUN: no serving endpoint had ever answered, so run.py, score.py, sweep.sh and
gate.py had between them never executed against a live socket. "Written and
tested" and "run once" are different states, and the difference is normally
discovered by burning instance time on a metered card, at the point in the
schedule where there is least of it.

So this replays ground truth over HTTP and lets the whole chain execute today:

    python bench/mock_server.py --records data/generated/records.jsonl &
    bash bench/sweep.sh mock/perfect perfect
    python bench/gate.py --model mock/perfect

It answers a second question the unit tests could not. The scorer's entire
justification is that it detects failure classes an averaged accuracy number
would hide. That is a claim ABOUT THE SCORER, and the only way to test it is to
inject each failure deliberately and confirm the corresponding counter moves.
Hence the profiles: each one is a documented model failure mode, reproduced.

    perfect               ground truth. The pipeline's own zero point: anything
                          below 100% here is a harness bug, not a model result.
    line_item_cliff       a digit corrupted in line-item money. The published
                          failure: strong on totals, collapsed on line items,
                          respectable on average. Must move line_numeric while
                          leaving doc_numeric intact.
    precision_loss        "4500.00" -> "4500". Same value, 100x looser
                          tolerance band downstream. Must move precision_loss
                          while relaxed accuracy stays at 100%.
    identifier_confusion  0/O in doc_number and references. Hardest field class
                          measured, and the one the linker runs on.
    zero_for_absent       "0.00" where the document states nothing. Open
                          question 7, and the quiet mistake that destroys
                          partial-shipment detection instead of failing loudly.
    truncate              stops mid-JSON with finish_reason "length". Must be
                          counted as truncation, NOT as an extraction failure.
    flaky                 500s at a fixed rate. Must be counted as api_errors,
                          NOT as parse failures.
    chatty                valid JSON wrapped in prose and a markdown fence.
                          The non-guided fallback path.

The model NAME carries the profile: `mock/line_item_cliff@0.3` is that profile
at rate 0.3. Nothing about the harness needs to know this file exists.

Prefix caching is simulated on the real rule (a byte-identical prefix already
seen is charged as cached), so the cache-hit-rate plumbing in run.py is
exercised rather than assumed. The number is fictional; the plumbing is not.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extraction.prompt import (DOC_CLOSE, DOC_OPEN,  # noqa: E402
                               HARNESS_SUPPLIED)

PROFILES = ("perfect", "line_item_cliff", "precision_loss",
            "identifier_confusion", "zero_for_absent", "truncate", "flaky",
            "chatty")

_LINE_MONEY = ("unit_price", "line_total")
_DOC_MONEY = ("subtotal", "allowance_total", "charge_total", "total_excl_tax",
              "tax", "total", "paid_amount", "rounding_amount", "amount_due")

# Ground truth keyed on the exact document text, because that is the only thing
# the request carries. Whitespace is preserved on both sides, which also means
# a lookup miss is itself informative: it says the prompt mutated the document
# on the way to the model, which would silently break the source-grounding
# check in production.
_TRUTH: dict[str, dict] = {}
_TRUTH_LOOSE: dict[str, dict] = {}
_SEEN_PREFIXES: set[str] = set()
_LOCK = threading.Lock()


def load_truth(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            src = rec["doc"]["source_text"]
            _TRUTH[src] = rec["doc"]
            _TRUTH_LOOSE[src.strip()] = rec["doc"]
    return len(_TRUTH)


def _parse_profile(model: str) -> tuple[str, float]:
    name = model.split("/")[-1]
    rate = 0.25
    if "@" in name:
        name, _, r = name.partition("@")
        rate = float(r)
    if name not in PROFILES:
        name = "perfect"
    return name, rate


def _extract_document(messages: list[dict]) -> str:
    """The document exactly as build_prompt fenced it: one newline in, one out.

    Deliberately NOT `.strip()`. The generator emits documents that end in
    blank lines, and stripping them here would make every lookup miss while
    looking like a model failure. The same trap is live in production: the
    source-grounding check searches source_text byte for byte, so any
    normalisation between the corpus and the prompt weakens it silently.
    """
    for m in reversed(messages):
        c = m.get("content") or ""
        if DOC_OPEN in c:
            body = c.split(DOC_OPEN, 1)[1].rsplit(DOC_CLOSE, 1)[0]
            if body.startswith("\n"):
                body = body[1:]
            if body.endswith("\n"):
                body = body[:-1]
            return body
    return ""


#: The documented identifier failure is character CONFUSION, not a random
#: digit: 0 read as O, 1 as l, 5 as S. A plain "0" -> "O" substitution is a
#: no-op on identifiers like "Q-3987" that contain no zero, which would make
#: this profile silently inject nothing at all.
_LOOKALIKE = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
              "8": "B", "B": "8", "2": "Z", "6": "G"}


def _confuse(s: str, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(s) if ch in _LOOKALIKE]
    if not idx:
        return s
    i = rng.choice(idx)
    return s[:i] + _LOOKALIKE[s[i]] + s[i + 1:]


def _corrupt_digit(s: str, rng: random.Random) -> str:
    digits = [i for i, ch in enumerate(s) if ch.isdigit()]
    if not digits:
        return s
    i = rng.choice(digits)
    return s[:i] + rng.choice([d for d in "0123456789" if d != s[i]]) + s[i + 1:]


def _degrade(doc: dict, profile: str, rate: float, rng: random.Random) -> dict:
    out = {k: v for k, v in doc.items() if k not in HARNESS_SUPPLIED}
    lines = out.get("line_items") or []

    if profile == "line_item_cliff":
        for li in lines:
            for f in _LINE_MONEY:
                if li.get(f) and rng.random() < rate:
                    li[f] = _corrupt_digit(str(li[f]), rng)
    elif profile == "precision_loss":
        for f in _DOC_MONEY:
            if isinstance(out.get(f), str) and out[f].endswith(".00"):
                out[f] = out[f][:-3]
        for li in lines:
            for f in _LINE_MONEY:
                if isinstance(li.get(f), str) and li[f].endswith(".00"):
                    li[f] = li[f][:-3]
    elif profile == "identifier_confusion":
        if out.get("doc_number") and rng.random() < rate:
            out["doc_number"] = _confuse(str(out["doc_number"]), rng)
        out["references"] = [
            _confuse(r, rng) if rng.random() < rate else r
            for r in (out.get("references") or [])]
    elif profile == "zero_for_absent":
        # The mistake, exactly as it would be made: a helpful-looking zero
        # instead of the absence the document actually states.
        for f in _DOC_MONEY:
            if out.get(f) is None:
                out[f] = "0.00"
        for li in lines:
            for f in _LINE_MONEY:
                if li.get(f) is None:
                    li[f] = "0.00"
    return out


def _render(doc: dict, profile: str, rng: random.Random) -> str:
    body = json.dumps(doc, separators=(",", ":"), default=str)
    if profile == "truncate":
        return body[:max(20, int(len(body) * 0.6))]
    if profile == "chatty":
        return ("Here is the extracted record:\n```json\n" + body +
                "\n```\nLet me know if you need anything else.")
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                                  # noqa: D102
        pass

    def _send(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):                                           # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": f"mock/{p}", "object": "model"} for p in PROFILES]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):                                          # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        body = json.loads(self.rfile.read(
            int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        model = body.get("model", "mock/perfect")
        profile, rate = _parse_profile(model)
        messages = body.get("messages", [])
        rng = random.Random(hash(json.dumps(messages, sort_keys=True)) & 0xFFFF)

        if profile == "flaky" and rng.random() < rate:
            self._send(503, {"error": {
                "message": "simulated: engine is busy or has crashed",
                "type": "server_error"}})
            return

        src = _extract_document(messages)
        doc = _TRUTH.get(src)
        if doc is None and src:
            doc = _TRUTH_LOOSE.get(src.strip())
            if doc is not None:
                print("  [mock] document matched only after whitespace "
                      "normalisation. In production that means the prompt "
                      "altered the document, which weakens the byte-exact "
                      "source-grounding check.", file=sys.stderr)
        if doc is None:
            # A health check, or a document the prompt altered in transit.
            content = "ready" if not src else json.dumps(
                {"error": "document not in replay corpus"})
            finish = "stop"
            completion_tokens = 4
        else:
            rendered = _render(_degrade(json.loads(json.dumps(doc)), profile,
                                        rate, rng), profile, rng)
            content = rendered
            finish = "length" if profile == "truncate" else "stop"
            completion_tokens = max(1, len(rendered) // 4)

        # Prefix caching, on the real rule: the frozen blocks are whatever the
        # system message holds, and a repeat of a prefix already seen is
        # charged as cached. If the prompt were not byte-stable this would
        # report ~0%, which is exactly the signal run.py warns on.
        prefix = (messages[0].get("content", "") if messages else "")
        with _LOCK:
            cached_hit = prefix in _SEEN_PREFIXES
            _SEEN_PREFIXES.add(prefix)
        prefix_tokens = max(1, len(prefix) // 4)
        doc_tokens = max(1, len(src) // 4) + 8

        self._send(200, {
            "id": "chatcmpl-mock", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": prefix_tokens + doc_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prefix_tokens + doc_tokens + completion_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": prefix_tokens if cached_hit else 0},
            },
        })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--records", type=Path,
                   default=ROOT / "data/generated/records.jsonl")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    n = load_truth(a.records)
    print(f"replay corpus: {n} documents from {a.records}")
    print(f"profiles: {', '.join(PROFILES)}   (use e.g. mock/line_item_cliff@0.3)")
    # 127.0.0.1, never 0.0.0.0. The locality guard in the client refuses
    # anything else, and a dev tool that made the guard look satisfiable by a
    # remote host would be worse than no dev tool.
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"listening on http://127.0.0.1:{a.port}/v1  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
