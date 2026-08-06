from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsoleSession:
    """
    Analyst workflow:

    1. explain
    2. query memory
    3. recheck
    4. approve
    """

    verdict: object
    memory: object | None = None
    events: list[str] = field(default_factory=list)

    def explain(self) -> str:
        self.events.append("explain")
        return f"Reasoning: {self.verdict}"

    def query_memory(self, signature: str):
        self.events.append("memory_query")

        if self.memory is None:
            return None

        return self.memory.lookup(signature)

    def recheck(self, callback):
        self.events.append("recheck")
        return callback(self.verdict)

    def approve(self):
        self.events.append("approve")
        return {
            "approved": True,
            "events": self.events,
        }
