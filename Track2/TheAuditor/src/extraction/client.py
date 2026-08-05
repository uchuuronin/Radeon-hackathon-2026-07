"""The ONLY file in the repo permitted to import a model client.

The seam rule from the handoff, restated as code: the dependency-audit grep in
the README should return exactly this file. Everything else runs on pydantic
alone, which is what keeps A's half demoable if the GPU side is unavailable.

LOCALITY IS ENFORCED HERE, NOT PROMISED IN THE README
----------------------------------------------------
"Core inference runs locally, no remote APIs" is a competition rule AND our
product claim, and a claim you can only support by grepping configuration is a
weak one. So `_assert_local` runs on every client construction and raises on
any host that is not loopback. The failure mode it prevents is not malice, it
is an OPENAI_BASE_URL left in a shell profile, or a default base_url quietly
reasserting itself when a config key is misspelled. Either would send customer
financial data off the box and neither would look like an error.

This is defence in depth, not the whole defence: the container running with
--network=none is what makes egress impossible. This is what makes it LOUD
while you are still developing, when the network is still up.
"""

from __future__ import annotations

import ipaddress
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://localhost:8000/v1"

#: Hostnames that are the local machine by definition.
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", ""}


def _assert_local(base_url: str) -> None:
    """Raise unless `base_url` points at this machine."""
    host = urlparse(base_url).hostname or ""
    if host in _LOCAL_NAMES:
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(
        f"refusing to send documents to non-local host {host!r}. Core "
        f"inference must run on this machine: it is a competition rule, it is "
        f"the product claim, and financial documents do not leave the box. If "
        f"the endpoint really is local, address it as localhost or 127.0.0.1."
    )


def _assert_no_stray_credentials() -> list[str]:
    """Report remote-provider credentials in the environment.

    Not fatal: a shared image may carry them for unrelated reasons. But an API
    key present while we claim no egress is exactly the thing a judge would
    find, so it is surfaced rather than ignored.
    """
    return [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN",
                        "AZURE_OPENAI_API_KEY", "GOOGLE_API_KEY")
            if os.environ.get(k)]


@dataclass
class Usage:
    """What a call cost. Fed straight into the ladder's cost accounting."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class LocalVLLM:
    """Thin wrapper over the OpenAI-compatible endpoint vLLM serves.

    Thin on purpose. vLLM already does paged attention, continuous batching,
    prefix caching and guided decoding; the optimisation lives in the SERVER
    flags and in the prompt, not in a client abstraction. Anything clever here
    would sit between us and the thing doing the work.
    """
    model: str
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _assert_local(self.base_url)
        stray = _assert_no_stray_credentials()
        if stray:
            print(f"  [locality] WARNING: {', '.join(stray)} set in the "
                  f"environment. Not used, but unset them before the demo.")

    @property
    def client(self):
        """Imported lazily so the module is importable, and testable, without
        the openai package present. A's tests must never need it."""
        if self._client is None:
            from openai import OpenAI              # noqa: PLC0415
            self._client = OpenAI(base_url=self.base_url, api_key="local",
                                  timeout=self.timeout)
        return self._client

    def complete(self, messages: list[dict],
                 guided_json: Optional[dict] = None,
                 n: int = 1) -> tuple[list[str], Usage]:
        """One call. `n>1` samples in a SINGLE request on purpose.

        Self-consistency needs N samples of the same prompt. Sending them as
        one request with n=N lets vLLM share the prefill and batch the decodes,
        so wall-clock is roughly one generation rather than N sequential ones.
        The GPU work is still genuinely N times, which is why the cost model
        counts N and the wall-clock does not.
        """
        extra: dict[str, Any] = {}
        if guided_json is not None:
            extra["guided_json"] = guided_json
        # Qwen3 and friends: reasoning tokens inflate latency and interfere
        # with guided decoding. Extraction is transcription, not reasoning.
        extra["chat_template_kwargs"] = {"enable_thinking": False}

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, n=n,
            temperature=self.temperature if n == 1 else max(self.temperature, 0.7),
            max_tokens=self.max_tokens, extra_body=extra)
        dt = (time.perf_counter() - t0) * 1000

        u = getattr(resp, "usage", None)
        cached = 0
        details = getattr(u, "prompt_tokens_details", None) if u else None
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return (
            [c.message.content or "" for c in resp.choices],
            Usage(prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
                  completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
                  cached_tokens=cached, latency_ms=dt),
        )

    def health(self) -> tuple[bool, str]:
        """B2's gate, as one call. Returns (ok, message)."""
        try:
            out, usage = self.complete(
                [{"role": "user", "content": "Reply with the single word: ready"}])
            return True, f"{self.model} responded in {usage.latency_ms:.0f} ms: {out[0][:40]!r}"
        except Exception as exc:                                # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
