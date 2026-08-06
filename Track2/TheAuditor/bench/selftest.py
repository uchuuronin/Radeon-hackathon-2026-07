"""Exercise the entire extraction chain with no GPU, no network and no model.

    python bench/selftest.py

It starts bench/mock_server.py on loopback, drives run.py -> score.py against
each fault profile, and asserts that the scorer's counter for that fault moved
and that the OTHER counters did not.

WHY THIS IS A TEST AND NOT A DEMO
---------------------------------
The scorer's whole justification is that it separates failure classes an
average would hide. That justification is a claim about the scorer, and until
each failure is injected deliberately and the matching counter checked, it is
an untested claim sitting in a docstring. The unit tests construct Score
objects directly; this drives the real path, over a real socket, through the
real client, parser and manifest.

It also gives every unrun artefact its first execution somewhere other than a
metered instance. `perfect` is the zero point: anything below 100% there is a
harness bug, and finding one here costs nothing.

Exit code is 0 only if every expectation holds.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RECORDS = ROOT / "data/generated/records.jsonl"
PORT = 8127                       # not 8000: never collide with a real vLLM
BASE = f"http://127.0.0.1:{PORT}/v1"
N = 24


def _wait_for_server(timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE}/models", timeout=1)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    raise RuntimeError("mock server did not come up")


def run_profile(profile: str) -> tuple[object, object]:
    from bench.score import score_files
    from extraction.client import LocalVLLM
    from extraction.run import extract_many, load_sources
    from schemas import Tier

    docs = load_sources(RECORDS, N, layout="layout_a")
    llm = LocalVLLM(model=f"mock/{profile}", base_url=BASE)
    recs, stats = extract_many(docs, llm, Tier.FAST, concurrency=6,
                               prompt_id=f"selftest-{profile}")
    if stats.api_errors:
        # Without this the failure presents as 24 identical zeros with no
        # cause, which is the same shape as a broken scorer and sends you
        # looking in the wrong file.
        for e in list(dict.fromkeys(stats.errors))[:2]:
            print(f"    [{profile}] {stats.api_errors} calls never reached "
                  f"the server: {e}")
    out = ROOT / f"runs/selftest/{profile}/extracted.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(r.model_dump_json() for r in recs) + "\n",
                   encoding="utf-8")
    return (score_files(RECORDS, out, layout="layout_a",
                        attempted=set(stats.doc_ids)), stats)


def main() -> int:
    # Fail on the cause, not on the symptom. Every downstream number depends
    # on the client being able to open a socket, and an import error or a
    # proxy hijack both present as "24 documents did not round-trip", which
    # reads like a harness bug and is not one.
    try:
        import openai                                            # noqa: F401
    except ImportError:
        print("bench/selftest.py drives the REAL client, which needs the\n"
              "openai package (a local HTTP client pointed at a local\n"
              "endpoint; no remote endpoint is ever configured).\n\n"
              "    pip install openai\n")
        return 2

    import os
    hijack = [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                          "http_proxy", "https_proxy", "all_proxy",
                          "OPENAI_BASE_URL")
              if os.environ.get(k) and "NO_PROXY" not in k]
    if hijack:
        print(f"  [selftest] {', '.join(hijack)} set in the environment. The "
              f"openai client honours these and will route loopback traffic "
              f"through a proxy, which fails as a connection error. Unset "
              f"them, or set NO_PROXY=127.0.0.1,localhost.")

    if not RECORDS.exists():
        print(f"missing {RECORDS}\n"
              f"run: python data/generator/gen.py --n 510 --seed 1337 "
              f"--out data/generated")
        return 2

    srv = subprocess.Popen(
        [sys.executable, str(ROOT / "bench/mock_server.py"),
         "--records", str(RECORDS), "--port", str(PORT)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    failures: list[str] = []
    try:
        _wait_for_server()

        # --- the zero point ------------------------------------------------
        s, st = run_profile("perfect")
        acc, lo, hi = s.rate("line_numeric", "exact")
        print(f"perfect               line_numeric exact {acc:.1%} "
              f"n={s.n['line_numeric']}  ok={st.ok}/{st.documents}  "
              f"cache={st.cache_hit_rate:.0%}  guided={st.guided}")
        if st.ok != st.documents:
            failures.append(f"perfect: {st.documents - st.ok} documents did not "
                            f"round-trip; the harness is losing records")
        if acc < 1.0:
            failures.append(f"perfect: exact accuracy {acc:.1%} < 100%; ground "
                            f"truth does not score as correct, so every other "
                            f"number is measured against a broken zero")
        if st.cache_hit_rate < 0.5:
            failures.append(f"perfect: prefix cache {st.cache_hit_rate:.0%}; "
                            f"the frozen prefix is not byte-stable")
        base_ident = s.rate("identifier", "relaxed")[0]

        # --- each fault must move ITS counter and only its counter ---------
        cases = [
            ("line_item_cliff@0.6", "line_numeric",
             lambda sc: sc.rate("line_numeric", "relaxed")[0] < 0.95
             and sc.rate("doc_numeric", "relaxed")[0] > 0.99,
             "line-item money degraded without touching totals: the exact "
             "shape of the published failure, and the reason the tier metric "
             "is line-item numeric and not an average"),
            ("precision_loss", "precision_loss",
             lambda sc: sc.precision_loss > 0
             and sc.rate("line_numeric", "relaxed")[0] > 0.99
             and sc.rate("line_numeric", "exact")[0] < 0.95,
             "same VALUE, different stated precision: relaxed stays perfect, "
             "exact falls, precision_loss counts it. This is the 100x "
             "tolerance-band trap and an averaged number would hide it"),
            ("identifier_confusion@0.8", "identifier",
             lambda sc: sc.rate("identifier", "relaxed")[0] < base_ident,
             "0/O confusion in doc_number and references: the hardest field "
             "class measured, and the one the linker runs on"),
            ("zero_for_absent", "doc_numeric",
             lambda sc: any("hallucinated" in m.reason for m in sc.misses),
             "0.00 emitted where the document states nothing: open question 7, "
             "reported as a hallucinated value rather than a dropped one"),
            ("chatty", "parse",
             lambda sc: sc.parse_failures == 0,
             "prose and a markdown fence around valid JSON still parses, so "
             "the non-guided fallback path does not lose whole rows"),
        ]
        for model, label, ok, why in cases:
            sc, stt = run_profile(model)
            good = ok(sc)
            print(f"{model:<22}{label:<16}{'PASS' if good else 'FAIL'}   {why}")
            if not good:
                failures.append(f"{model}: scorer did not register the "
                                f"injected fault")

        # --- run-level faults must NOT be scored as extraction failures ----
        _, stt = run_profile("truncate")
        print(f"truncate              truncated={stt.truncated} "
              f"parse_failures={stt.parse_failures} api_errors={stt.api_errors}")
        if stt.truncated == 0:
            failures.append("truncate: finish_reason 'length' was not counted; "
                            "a max_tokens ceiling would be misread as the "
                            "model being unable to extract")

        _, stt = run_profile("flaky@0.5")
        print(f"flaky@0.5             api_errors={stt.api_errors} "
              f"parse_failures={stt.parse_failures} ok={stt.ok}")
        if stt.api_errors == 0 and stt.ok == N:
            failures.append("flaky: retries hid every server error, so a "
                            "failing server would not be visible at all")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()

    print()
    if failures:
        print("SELFTEST FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELFTEST PASSED — the extraction chain runs end to end and the "
          "scorer "
          "detects every failure class it claims to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
