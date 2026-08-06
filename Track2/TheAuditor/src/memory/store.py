from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class MemoryEntry:
    signature: str
    resolution: str
    confidence: float
    created_at: str


class MemoryStore:
    """
    Offline gated memory store.

    Reuse only when:
    - exact signature match
    - confidence threshold satisfied
    """

    def __init__(self, min_confidence: float = 0.9):
        self.min_confidence = min_confidence
        self._store: dict[str, MemoryEntry] = {}

    def put(
        self,
        signature: str,
        resolution: str,
        confidence: float,
    ) -> None:
        if confidence < self.min_confidence:
            return

        self._store[signature] = MemoryEntry(
            signature=signature,
            resolution=resolution,
            confidence=confidence,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def lookup(self, signature: str) -> MemoryEntry | None:
        entry = self._store.get(signature)

        if entry is None:
            return None

        if entry.confidence < self.min_confidence:
            return None

        return entry

    def __len__(self) -> int:
        return len(self._store)
