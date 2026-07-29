"""Runtime policy — the numbers and switches that are DECISIONS, not code.

JSON validated by a pydantic model. Deliberately not TOML: the project pins
Python >=3.10 and tomllib landed in 3.11, so TOML would cost a dependency for
no gain. The file is optional; absence means documented defaults.

Everything here is auditable configuration a reviewer can read in one screen:
which gate auto-resolves, which vendors use which date order. No amounts, no
party financial data — vendor names and format preferences only.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONFIG_PATH = Path("config") / "policy.json"


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_resolve_gate: Literal["verify_pass", "strict_pass"] = Field(
        default="verify_pass",
        description="Which report property lets rung 0 settle a case. "
                    "verify_pass: nothing failed (within-tolerance allowed). "
                    "strict_pass: every applicable check exact — measured on "
                    "the ten market fixtures, that gate escalates 2 clean "
                    "documents in 10 over date-format ambiguity alone, which "
                    "is a manufactured 20% escalation rate. strict_pass "
                    "remains the bar for MEMORY WRITES, where a false "
                    "precedent poisons future lookups.")
    party_date_order: dict[str, str] = Field(
        default_factory=dict,
        description="Vendor -> MDY|DMY. A vendor's printed date format is "
                    "stable even though the format space is not; registering "
                    "it collapses genuine ambiguity (06/09/2026) to one "
                    "reading. Same per-party principle as tolerance.")

    def date_order_for(self, party: str) -> Optional[str]:
        return self.party_date_order.get(party)


@lru_cache(maxsize=1)
def load(path: Optional[Path] = None) -> Policy:
    p = path or CONFIG_PATH
    if p.exists():
        return Policy.model_validate(json.loads(p.read_text(encoding="utf-8")))
    return Policy()


def reset_cache() -> None:
    load.cache_clear()
