from src.console.session import ConsoleSession
from src.memory.store import MemoryStore


def test_console_explain():
    session = ConsoleSession(verdict="price mismatch")

    result = session.explain()

    assert "price mismatch" in result
    assert "explain" in session.events


def test_console_memory_lookup():
    memory = MemoryStore()

    memory.put(
        signature="case1",
        resolution="approved",
        confidence=0.95,
    )

    session = ConsoleSession(
        verdict="case1",
        memory=memory,
    )

    result = session.query_memory("case1")

    assert result.resolution == "approved"


def test_console_recheck_callback():
    session = ConsoleSession(verdict="needs review")

    result = session.recheck(
        lambda verdict: f"checked {verdict}"
    )

    assert result == "checked needs review"
    assert "recheck" in session.events


def test_console_approve():
    session = ConsoleSession(verdict="ok")

    result = session.approve()

    assert result["approved"] is True
    assert "approve" in result["events"]
