# HOLDOUT — DO NOT OPEN UNTIL DEMO DAY

Layout C lives here: `layout_c.py` and the pre-rendered chains in `docs/`.
It is reserved from development so the question "what about a layout you
didn't design for?" can be answered live, on camera, with evidence instead
of assertion.

Rules, in force from the commit that adds this file:
- `render_c` is never imported by `src/`, `tests/`, or the generator's
  RENDERERS table.
- The verifier is never run against `docs/` before the demo.
- Extraction prompts are never developed or evaluated against these files.
- The seal breaks exactly once, on camera, at S4.

Sealed on Day 2, when the least was known about verifier behaviour — the
latest date at which "unseen" is honest. There is no way to un-see it.
