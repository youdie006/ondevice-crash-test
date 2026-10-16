"""Application state schema and store for On-Device Crash Test."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ArgonUsage(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thought_tokens: int = 0
    total_cached_tokens: int = 0
    total_tokens: int = 0


class ArgonCallRecord(BaseModel):
    iteration: int
    interaction_id: str
    thinking_level: str = "high"
    latency_s: float
    usage: ArgonUsage
    diagnosis: list[dict[str, Any]] = Field(default_factory=list)
    action: dict[str, Any] = Field(default_factory=dict)
    gate_verdict: str = "not yet run"
    rationale: str = ""


class RunState(BaseModel):
    run_id: str
    scenario_id: str = "robot_vacuum"
    phase: str = "not yet run"
    mode: str = "showcase"
    n_images: int | str = "not yet run"
    baseline_control: dict[str, Any] | str = "not yet run"
    final_control: dict[str, Any] | str = "not yet run"
    budget_reference: dict[str, Any] | str = "not yet run"
    conditions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    iterations: list[dict[str, Any]] = Field(default_factory=list)
    argon_logs: list[ArgonCallRecord] = Field(default_factory=list)
    needs_human: bool = False
    human_decision: str | None = None
    error: str | None = None


class StateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._active_run_id: str | None = None

    def get_active_run_id(self) -> str | None:
        with self._lock:
            return self._active_run_id

    def set_run(self, run_id: str, state: dict[str, Any], active: bool = False) -> None:
        with self._lock:
            self._runs[run_id] = state
            if active:
                self._active_run_id = run_id

    def update_run(self, run_id: str, patch: dict[str, Any], clear_active: bool = False) -> dict[str, Any]:
        with self._lock:
            cur = self._runs.get(run_id, {"run_id": run_id})
            cur.update(patch)
            self._runs[run_id] = cur
            if clear_active and self._active_run_id == run_id:
                self._active_run_id = None
            return dict(cur)

    def get_run(self, run_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            target = run_id or self._active_run_id
            if target and target in self._runs:
                return dict(self._runs[target])
        return load_default_state()


def load_default_state(scenario_id: str = "robot_vacuum") -> dict[str, Any]:
    showcase_state_path = Path("showcase") / scenario_id / "state.json"
    if showcase_state_path.exists():
        try:
            return json.loads(showcase_state_path.read_text())
        except Exception:
            pass

    baseline_path = Path("showcase") / scenario_id / "baseline.json"
    baseline_data: dict[str, Any] | None = None
    if baseline_path.exists():
        try:
            baseline_data = json.loads(baseline_path.read_text())
        except Exception:
            baseline_data = None

    return {
        "run_id": "showcase_default",
        "scenario_id": scenario_id,
        "phase": "baseline_ready" if baseline_data else "not yet run",
        "mode": "showcase",
        "n_images": baseline_data["n_images"] if baseline_data else "not yet run",
        "baseline_control": baseline_data["control_metrics"] if baseline_data else "not yet run",
        "final_control": "not yet run",
        "baseline_int8": baseline_data["baseline_int8"] if baseline_data else "not yet run",
        "fp32": baseline_data["fp32"] if baseline_data else "not yet run",
        "budget_reference": baseline_data["budget_reference"] if baseline_data else "not yet run",
        "conditions": {},
        "iterations": [],
        "argon_logs": [],
        "needs_human": False,
        "human_decision": None,
    }


STORE = StateStore()
