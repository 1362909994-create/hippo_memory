from __future__ import annotations

from hippocampus_memory.conflict_detector import ConflictDetector
from hippocampus_memory.memory_writer import MemoryWriter


def test_conflict_detector_finds_simple_interface_conflict(db):
    writer = MemoryWriter(db)
    writer.write(
        project="glasses",
        memory_type="technical_fact",
        content="TFT interface is SPI.",
        entities=["TFT"],
    )
    writer.write(
        project="glasses",
        memory_type="technical_fact",
        content="TFT interface is RGB.",
        entities=["TFT"],
    )

    conflicts = ConflictDetector(db).detect_for_project("glasses")

    assert conflicts
    assert conflicts[0]["entity"] == "TFT"

    ConflictDetector(db).detect_for_project("glasses")
    with db.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
    assert count == 1
