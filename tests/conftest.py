from __future__ import annotations

import pytest

from hippocampus_memory.db import Database


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "hippo.db")
    database.initialize()
    return database
