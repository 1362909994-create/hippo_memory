from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter


def test_write_memory_success(db):
    result = MemoryWriter(db).write(
        project="glasses",
        memory_type="constraint",
        content="User rejects long optical paths.",
        importance=0.9,
        confidence=0.95,
    )
    memory = db.get_memory(result.memory_id)
    assert result.created is True
    assert memory is not None
    assert memory.content == "User rejects long optical paths."


def test_duplicate_memory_returns_existing_id(db):
    writer = MemoryWriter(db)
    first = writer.write(project="glasses", memory_type="task_state", content="Screen is dark.")
    second = writer.write(project="glasses", memory_type="task_state", content="Screen is dark.")
    assert first.memory_id == second.memory_id
    assert second.created is False
    assert second.duplicate is True
