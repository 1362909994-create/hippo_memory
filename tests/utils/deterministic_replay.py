from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hippocampus_memory.db import Database
from tests.utils.system_simulator import SimulationEvent, SystemSimulationHarness


@dataclass(frozen=True, slots=True)
class DeterministicReplayResult:
    matched: bool
    first_snapshot: dict[str, Any]
    second_snapshot: dict[str, Any]
    first_steps: list[dict[str, Any]]
    second_steps: list[dict[str, Any]]

    @property
    def differences(self) -> dict[str, Any]:
        return {
            "snapshot_equal": self.first_snapshot == self.second_snapshot,
            "steps_equal": self.first_steps == self.second_steps,
        }


class DeterministicReplay:
    def __init__(self, tmp_path: Path, *, project: str = "replay") -> None:
        self.tmp_path = tmp_path
        self.project = project

    def replay(self, events: list[SimulationEvent]) -> DeterministicReplayResult:
        first = self._run("first", events)
        second = self._run("second", events)
        matched = first["snapshot"] == second["snapshot"] and first["steps"] == second["steps"]
        return DeterministicReplayResult(
            matched=matched,
            first_snapshot=first["snapshot"],
            second_snapshot=second["snapshot"],
            first_steps=first["steps"],
            second_steps=second["steps"],
        )

    def _run(self, name: str, events: list[SimulationEvent]) -> dict[str, Any]:
        db = Database(self.tmp_path / f"{name}.db")
        db.initialize()
        harness = SystemSimulationHarness(
            db,
            project=self.project,
            state_dir=self.tmp_path / name,
        )
        steps = harness.run_events(events)
        return {
            "snapshot": _normalize_snapshot(harness.snapshot(include_archived=True)),
            "steps": [step.normalized() for step in steps],
        }


def _normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(snapshot)
    normalized["contents"] = sorted(tuple(item) for item in snapshot.get("contents", []))
    return normalized
