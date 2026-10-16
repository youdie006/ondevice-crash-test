"""Application state schema and store for On-Device Crash Test."""

from __future__ import annotations

import copy
import json
import math
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ACTIVE_PHASES: frozenset[str] = frozenset(
    {"starting", "evaluating", "diagnosing", "repairing", "re-evaluating", "gate"}
)
TERMINAL_PHASES: frozenset[str] = frozenset({"done", "needs_human", "error"})
DEFAULT_CONTROL_SANITY_THRESHOLD: float = 0.80


def _is_finite_number(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def compute_pass_threshold(
    baseline_control: Any,
    explicit_threshold: Any = None,
) -> float | None:
    """Compute condition pass threshold = baseline clean control F1 - 0.05.

    Never hardcodes 0.90. Returns None if baseline control F1 is not yet measured.
    """
    if _is_finite_number(explicit_threshold):
        return round(float(explicit_threshold), 4)
    if isinstance(baseline_control, dict):
        raw_f1 = baseline_control.get("raw_f1")
        if _is_finite_number(raw_f1):
            return round(float(raw_f1) - 0.05, 4)
        f1 = baseline_control.get("f1")
        if _is_finite_number(f1):
            return round(float(f1) - 0.05, 4)
    return None


class ArgonUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_thought_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_tokens: int | None = None


class ArgonCallRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    iteration: int
    interaction_id: str
    thinking_level: str = "high"
    latency_s: float | None = None
    usage: ArgonUsage = Field(default_factory=ArgonUsage)
    diagnosis: list[dict[str, Any]] = Field(default_factory=list)
    action: dict[str, Any] = Field(default_factory=dict)
    gate_verdict: str = "not yet run"
    rationale: str = ""


class ProgressStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    done: int | None = None
    total: int | None = None
    label: str = ""


class ProgressRecentItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_id: str
    condition: str
    thumb_url: str | None = None
    raw_thumb_url: str | None = None
    fp32_boxes: int | None = None
    int8_boxes: int | None = None
    matched: int | None = None
    missed: int | None = None
    geometry: dict[str, Any] | None = None
    excluded: bool = False
    collapsed: bool = False
    exclude_reason: str | None = None


class ProgressArgon(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str | None = None
    images_sent: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_s: float | None = None


class ProgressAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    params_chips: list[str] = Field(default_factory=list)


class ProgressGate(BaseModel):
    model_config = ConfigDict(extra="allow")

    verdict: str
    reasons: list[str] = Field(default_factory=list)
    control_delta: float | None = None
    budget_ratios: dict[str, Any] | None = None


class RunProgress(BaseModel):
    model_config = ConfigDict(extra="allow")

    phase: str
    phase_started_at: str | None = None
    phase_elapsed_s: float | None = None
    step: ProgressStep | None = None
    recent: list[ProgressRecentItem] = Field(default_factory=list)
    argon: ProgressArgon | None = None
    action: ProgressAction | None = None
    gate: ProgressGate | None = None


class RunState(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    scenario_id: str = "robot_vacuum"
    phase: str = "not yet run"
    mode: str = "showcase"
    n_images: int | str = "not yet run"
    baseline_control: dict[str, Any] | str = "not yet run"
    final_control: dict[str, Any] | str = "not yet run"
    baseline_int8: dict[str, Any] | str = "not yet run"
    fp32: dict[str, Any] | str = "not yet run"
    budget_reference: dict[str, Any] | str = "not yet run"
    pass_threshold: float | None = None
    control_sanity_threshold: float = DEFAULT_CONTROL_SANITY_THRESHOLD
    conditions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    iterations: list[dict[str, Any]] = Field(default_factory=list)
    argon_logs: list[ArgonCallRecord] = Field(default_factory=list)
    progress: dict[str, Any] | None = None
    needs_human: bool = False
    human_decision: str | None = None
    error: str | None = None


def _is_run_dict_active(state: dict[str, Any] | None) -> bool:
    if not state or not isinstance(state, dict):
        return False
    if state.get("error"):
        return False
    phase = str(state.get("phase") or "")
    if phase in TERMINAL_PHASES:
        return False
    progress = state.get("progress")
    prog_phase = str(progress.get("phase") or "") if isinstance(progress, dict) else ""
    if prog_phase in TERMINAL_PHASES:
        return False
    return phase in ACTIVE_PHASES or prog_phase in ACTIVE_PHASES


class StateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._active_run_id: str | None = None

    def get_active_run_id(self) -> str | None:
        with self._lock:
            if self._active_run_id:
                cur = self._runs.get(self._active_run_id)
                if _is_run_dict_active(cur):
                    return self._active_run_id
                if cur and str(cur.get("phase") or "") in TERMINAL_PHASES:
                    self._active_run_id = None
                    return None
            return self._active_run_id

    def set_run(self, run_id: str, state: dict[str, Any], active: bool = False) -> None:
        with self._lock:
            copied = copy.deepcopy(state)
            copied["run_id"] = run_id
            if "control_sanity_threshold" not in copied or copied["control_sanity_threshold"] is None:
                copied["control_sanity_threshold"] = DEFAULT_CONTROL_SANITY_THRESHOLD
            if copied.get("pass_threshold") is None:
                copied["pass_threshold"] = compute_pass_threshold(
                    copied.get("baseline_control"),
                    copied.get("condition_pass_threshold"),
                )
            self._runs[run_id] = copied
            if active:
                self._active_run_id = run_id
            elif self._active_run_id == run_id and not _is_run_dict_active(copied):
                self._active_run_id = None

    def try_reserve_run(self, run_id: str, initial_state: dict[str, Any]) -> bool:
        """Atomically check that no active run is in progress and reserve run_id."""
        with self._lock:
            if self._active_run_id is not None:
                cur = self._runs.get(self._active_run_id)
                if _is_run_dict_active(cur):
                    return False
                self._active_run_id = None

            copied = copy.deepcopy(initial_state)
            copied["run_id"] = run_id
            if "control_sanity_threshold" not in copied or copied["control_sanity_threshold"] is None:
                copied["control_sanity_threshold"] = DEFAULT_CONTROL_SANITY_THRESHOLD
            if copied.get("pass_threshold") is None:
                copied["pass_threshold"] = compute_pass_threshold(
                    copied.get("baseline_control"),
                    copied.get("condition_pass_threshold"),
                )
            self._runs[run_id] = copied
            self._active_run_id = run_id
            return True

    def update_run(
        self,
        run_id: str,
        patch: dict[str, Any],
        clear_active: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            cur = copy.deepcopy(self._runs.get(run_id, {"run_id": run_id}))
            fixed_threshold = cur.get("pass_threshold")
            fixed_sanity = cur.get("control_sanity_threshold", DEFAULT_CONTROL_SANITY_THRESHOLD)

            if isinstance(patch, dict):
                for key, val in patch.items():
                    if key == "run_id":
                        continue
                    if (
                        key == "conditions"
                        and isinstance(cur.get("conditions"), dict)
                        and isinstance(val, dict)
                    ):
                        merged_conds = copy.deepcopy(cur["conditions"])
                        for cid, cval in val.items():
                            if isinstance(merged_conds.get(cid), dict) and isinstance(cval, dict):
                                merged_conds[cid].update(copy.deepcopy(cval))
                            else:
                                merged_conds[cid] = copy.deepcopy(cval)
                        cur["conditions"] = merged_conds
                    elif (
                        key == "progress"
                        and isinstance(cur.get("progress"), dict)
                        and isinstance(val, dict)
                    ):
                        merged_prog = copy.deepcopy(cur["progress"])
                        merged_prog.update(copy.deepcopy(val))
                        if isinstance(merged_prog.get("recent"), list):
                            merged_prog["recent"] = merged_prog["recent"][:8]
                        cur["progress"] = merged_prog
                    else:
                        cur[key] = copy.deepcopy(val)

            # Never allow patch to overwrite the server-generated run_id
            cur["run_id"] = run_id

            # Sync top-level phase from progress.phase when patch only updates progress
            prog = cur.get("progress")
            if isinstance(prog, dict):
                if isinstance(prog.get("recent"), list):
                    prog["recent"] = prog["recent"][:8]
                prog_phase = prog.get("phase")
                if prog_phase and ("phase" not in patch or cur.get("phase") in ("not yet run", "starting", "baseline_ready")):
                    cur["phase"] = prog_phase

            # Preserve pass_threshold fixed at run start; compute once if baseline_control just arrived
            if _is_finite_number(fixed_threshold):
                cur["pass_threshold"] = float(fixed_threshold)
            elif not _is_finite_number(cur.get("pass_threshold")):
                cur["pass_threshold"] = compute_pass_threshold(
                    cur.get("baseline_control"),
                    cur.get("condition_pass_threshold"),
                )

            if not _is_finite_number(cur.get("control_sanity_threshold")):
                cur["control_sanity_threshold"] = (
                    float(fixed_sanity) if _is_finite_number(fixed_sanity) else DEFAULT_CONTROL_SANITY_THRESHOLD
                )

            if cur.get("needs_human") is True and cur.get("phase") not in ("error", "done"):
                if isinstance(patch, dict) and patch.get("phase") == "needs_human":
                    cur["phase"] = "needs_human"

            eff_phase = str(cur.get("phase") or "")
            prog_phase_str = str(prog.get("phase") or "") if isinstance(prog, dict) else ""
            is_terminal = (
                clear_active
                or eff_phase in TERMINAL_PHASES
                or prog_phase_str in TERMINAL_PHASES
                or bool(cur.get("error"))
            )
            if is_terminal and self._active_run_id == run_id:
                self._active_run_id = None

            self._runs[run_id] = cur
            return copy.deepcopy(cur)

    def get_run(self, run_id: str | None = None, scenario_id: str = "robot_vacuum") -> dict[str, Any]:
        with self._lock:
            target = run_id or self._active_run_id
            if target and target in self._runs:
                return copy.deepcopy(self._runs[target])
        return load_default_state(scenario_id)


def load_default_state(scenario_id: str = "robot_vacuum") -> dict[str, Any]:
    baseline_path = Path("showcase") / scenario_id / "baseline.json"
    baseline_data: dict[str, Any] | None = None
    if baseline_path.exists():
        try:
            baseline_data = json.loads(baseline_path.read_text())
        except Exception:
            baseline_data = None

    baseline_control = baseline_data.get("control_metrics", "not yet run") if baseline_data else "not yet run"
    explicit_thresh = baseline_data.get("condition_pass_threshold") if baseline_data else None
    pass_threshold = compute_pass_threshold(baseline_control, explicit_thresh)
    sanity_bar = (
        float(baseline_data["control_sanity_bar"])
        if baseline_data and _is_finite_number(baseline_data.get("control_sanity_bar"))
        else DEFAULT_CONTROL_SANITY_THRESHOLD
    )

    default_base: dict[str, Any] = {
        "run_id": "showcase_default",
        "scenario_id": scenario_id,
        "phase": "baseline_ready" if baseline_data else "not yet run",
        "mode": "showcase",
        "n_images": baseline_data.get("n_images", "not yet run") if baseline_data else "not yet run",
        "baseline_control": baseline_control,
        "final_control": "not yet run",
        "baseline_int8": baseline_data.get("baseline_int8", "not yet run") if baseline_data else "not yet run",
        "fp32": baseline_data.get("fp32", "not yet run") if baseline_data else "not yet run",
        "budget_reference": baseline_data.get("budget_reference", "not yet run") if baseline_data else "not yet run",
        "pass_threshold": pass_threshold,
        "control_sanity_threshold": sanity_bar,
        "conditions": {},
        "iterations": [],
        "argon_logs": [],
        "progress": None,
        "needs_human": False,
        "human_decision": None,
        "error": None,
    }

    showcase_state_path = Path("showcase") / scenario_id / "state.json"
    if showcase_state_path.exists():
        try:
            loaded = json.loads(showcase_state_path.read_text())
            if isinstance(loaded, dict):
                merged = copy.deepcopy(default_base)
                merged.update(loaded)
                if merged.get("pass_threshold") is None:
                    merged["pass_threshold"] = compute_pass_threshold(
                        merged.get("baseline_control"),
                        merged.get("condition_pass_threshold") or explicit_thresh,
                    )
                if not _is_finite_number(merged.get("control_sanity_threshold")):
                    merged["control_sanity_threshold"] = sanity_bar
                return merged
        except Exception:
            pass

    return default_base


STORE = StateStore()
