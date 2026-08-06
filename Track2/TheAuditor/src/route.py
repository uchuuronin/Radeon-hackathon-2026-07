"""Stage 2 routing logic.

GPU-independent decision layer.
Uses deterministic verification plus confidence agreement
to decide whether a case can be auto-resolved or escalated.
"""

from __future__ import annotations


def route_decision(
    *,
    verify_pass: bool,
    agreement: float,
    tau_hi: float = 0.95,
    tau_lo: float = 0.70,
) -> str:
    """Return routing action.

    Rules:
    - verified + high agreement -> auto
    - failed verification -> escalate
    - low agreement -> escalate
    - uncertain middle zone -> escalate conservatively

    The thresholds are intentionally parameters because calibration
    happens later on real model data.
    """

    if not verify_pass:
        return "escalate"

    if agreement >= tau_hi:
        return "auto"

    if agreement < tau_lo:
        return "escalate"

    # undefined middle region: choose safety
    return "escalate"
