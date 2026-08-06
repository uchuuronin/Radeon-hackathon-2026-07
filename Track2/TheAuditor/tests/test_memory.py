from src.memory.store import MemoryStore


def test_memory_stores_high_confidence_resolution():
    memory = MemoryStore()

    memory.put(
        signature="invoice|po|123",
        resolution="accepted previous resolution",
        confidence=0.95,
    )

    result = memory.lookup("invoice|po|123")

    assert result is not None
    assert result.resolution == "accepted previous resolution"


def test_memory_rejects_low_confidence_entries():
    memory = MemoryStore()

    memory.put(
        signature="invoice|po|bad",
        resolution="wrong",
        confidence=0.5,
    )

    assert memory.lookup("invoice|po|bad") is None


def test_memory_misses_unknown_signature():
    memory = MemoryStore()

    assert memory.lookup("missing") is None
