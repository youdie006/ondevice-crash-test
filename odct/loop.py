"""Closed-loop diagnose -> repair -> re-evaluate -> gate orchestrator for On-Device Crash Test."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml
from PIL import Image, ImageDraw

from app.state import DEFAULT_CONTROL_SANITY_THRESHOLD, compute_pass_threshold
from odct.argon import (
    ArgonCallError,
    diagnose,
    select_image_pairs_for_diagnosis,
    verify_diagnosis,
)
from odct.budget import Measured, check, measure_model
from odct.infer import Det, Detector
from odct.metrics import CondMetrics, _greedy_pairs, evaluate_condition, match, persisted
from odct.quant import count_quantized_ops
from odct.repairs import (
    RepairContext,
    RepairRejectedError,
     apply as apply_repair,
    validate_and_resolve_action,
)


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_raw_passing(raw_f1: float | None, threshold_raw: float) -> bool:
    if raw_f1 is None or isinstance(raw_f1, bool):
        return False
    return bool(math.isfinite(float(raw_f1)) and float(raw_f1) >= float(threshold_raw))


def _load_scenario_cfg(scenario_id: str) -> dict[str, Any]:
    path = Path("scenarios") / f"{scenario_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Scenario config not found: {path}")
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _class_id_to_name(scenario_cfg: dict[str, Any]) -> dict[int, str]:
    classes = scenario_cfg.get("classes") or {}
    mapping: dict[int, str] = {}
    if isinstance(classes, dict):
        for cname, cid in classes.items():
            try:
                mapping[int(cid)] = str(cname)
            except Exception:
                pass
    return mapping


def _load_manifest_lookup(
    scenario_id: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Return ((condition_id, file_name) -> entry, alias_id -> canonical_file_name)."""
    manifest_path = Path("data") / scenario_id / "manifest.json"
    entry_map: dict[tuple[str, str], dict[str, Any]] = {}
    id_map: dict[str, str] = {}
    if not manifest_path.is_file():
        return entry_map, id_map

    try:
        raw = json.loads(manifest_path.read_text())
        conditions_block = raw.get("conditions") if isinstance(raw, dict) else None
        entries_list: list[dict[str, Any]] = []
        if isinstance(conditions_block, dict):
            for cid, cblock in conditions_block.items():
                if isinstance(cblock, dict) and isinstance(cblock.get("entries"), list):
                    for item in cblock["entries"]:
                        if isinstance(item, dict):
                            cp = dict(item)
                            cp.setdefault("condition_id", str(cid))
                            entries_list.append(cp)
                elif isinstance(cblock, list):
                    for item in cblock:
                        if isinstance(item, dict):
                            cp = dict(item)
                            cp.setdefault("condition_id", str(cid))
                            entries_list.append(cp)
        elif isinstance(raw, dict) and isinstance(raw.get("entries"), list):
            entries_list = [x for x in raw["entries"] if isinstance(x, dict)]

        for item in entries_list:
            cid = str(item.get("condition_id") or item.get("condition") or "")
            fname = str(
                item.get("file_name")
                or Path(str(item.get("condition_path") or item.get("original_path") or "")).name
                or ""
            )
            if not fname:
                continue
            if not fname.endswith(".jpg") and "." not in fname:
                fname = f"{fname}.jpg"
            stem = Path(fname).stem
            id_map[fname] = fname
            id_map[stem] = fname
            img_id = str(item.get("image_id") or "").strip()
            if img_id:
                id_map[img_id] = fname
            src_id = str(item.get("source_image_id") or "").strip()
            if src_id:
                id_map[src_id] = fname
            if cid:
                entry_map[(cid, fname)] = item
    except Exception:
        pass

    return entry_map, id_map


def _render_overlay_png(
    image_path: Path,
    ref_dets: Sequence[Det],
    int8_dets: Sequence[Det],
    out_path: Path,
    class_names: dict[int, str],
) -> tuple[int, int]:
    """Write FP32 (green) and INT8 (orange) bounding box overlay PNG and return (width, height)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        canvas = im.convert("RGB")
        width, height = canvas.size

    draw = ImageDraw.Draw(canvas)
    for det in ref_dets:
        x1, y1, x2, y2 = det.box
        draw.rectangle([x1, y1, x2, y2], outline=(22, 163, 74), width=3)
        cname = class_names.get(int(det.cls), str(det.cls))
        label = f"FP32 {cname}"
        ty = max(0.0, y1 - 14.0)
        draw.rectangle(
            [x1, ty, min(float(width), x1 + len(label) * 6 + 6), ty + 13],
            fill=(22, 163, 74),
        )
        draw.text((x1 + 3, ty + 1), label, fill=(255, 255, 255))

    for det in int8_dets:
        x1, y1, x2, y2 = det.box
        ix1 = min(x2, x1 + 2.0)
        iy1 = min(y2, y1 + 2.0)
        ix2 = max(ix1, x2 - 2.0)
        iy2 = max(iy1, y2 - 2.0)
        draw.rectangle([ix1, iy1, ix2, iy2], outline=(234, 88, 12), width=2)
        cname = class_names.get(int(det.cls), str(det.cls))
        label = f"INT8 {cname}"
        by = (
            min(float(height - 14), y2 - 14.0)
            if (y2 - y1) > 28
            else min(float(height - 14), y2)
        )
        draw.rectangle(
            [ix1, by, min(float(width), ix1 + len(label) * 6 + 6), by + 13],
            fill=(234, 88, 12),
        )
        draw.text((ix1 + 3, by + 1), label, fill=(255, 255, 255))

    canvas.save(out_path, format="PNG")
    return width, height


def _build_geometry(
    width: int,
    height: int,
    ref_dets: Sequence[Det],
    int8_dets: Sequence[Det],
    matched_r: set[int],
) -> dict[str, Any]:
    return {
        "width": int(width),
        "height": int(height),
        "fp32": [
            {
                "box": [round(float(c), 2) for c in d.box],
                "cls": int(d.cls),
                "conf": round(float(d.conf), 4),
            }
            for d in ref_dets
        ],
        "int8": [
            {
                "box": [round(float(c), 2) for c in d.box],
                "cls": int(d.cls),
                "conf": round(float(d.conf), 4),
            }
            for d in int8_dets
        ],
        "missed": [
            {
                "box": [round(float(c), 2) for c in d.box],
                "cls": int(d.cls),
            }
            for idx, d in enumerate(ref_dets)
            if idx not in matched_r
        ],
    }


def _evaluate_split_b_target(
    *,
    scenario_id: str,
    condition_id: str,
    stage_key: str,
    run_id: str,
    fp32_det: Detector,
    int8_det: Detector,
    split_b_files: Sequence[str],
    manifest_entries: dict[tuple[str, str], dict[str, Any]],
    class_names: dict[int, str],
    pass_threshold_raw: float,
    step_offset: int,
    step_total: int,
    on_image_progress: Callable[[dict[str, Any], int, str], None],
) -> tuple[CondMetrics, list[dict[str, Any]]]:
    """Evaluate a condition (or 'control') on Split B images, writing overlay PNGs and emitting per-image progress."""
    scenario_dir = Path("data") / scenario_id
    real_dir = scenario_dir / "real"
    cond_dir = real_dir if condition_id == "control" else (scenario_dir / "conditions" / condition_id)

    originals: list[Path] = []
    cond_images: list[Path] = []
    pre_excluded = 0
    pre_reasons: dict[str, int] = {}
    step_counter = step_offset

    # Track per-image pairs for Argon multimodal input
    diagnostic_pairs: list[dict[str, Any]] = []
    seen_in_metrics: set[str] = set()

    for fname in split_b_files:
        orig_p = real_dir / fname
        if condition_id == "control":
            cond_p = orig_p
        else:
            m_entry = manifest_entries.get((condition_id, fname))
            if m_entry is not None:
                if m_entry.get("valid") is False or m_entry.get("allow_in_pipeline") is False:
                    pre_excluded += 1
                    reason = str(m_entry.get("validity_status") or "manifest_excluded")
                    pre_reasons[reason] = pre_reasons.get(reason, 0) + 1
                    step_counter += 1
                    raw_rel = f"data/{scenario_id}/conditions/{condition_id}/{fname}"
                    on_image_progress(
                        {
                            "image_id": fname,
                            "condition": condition_id,
                            "thumb_url": f"/artifacts/{raw_rel}",
                            "raw_thumb_url": f"/artifacts/{raw_rel}",
                            "fp32_boxes": 0,
                            "int8_boxes": 0,
                            "matched": 0,
                            "missed": 0,
                            "excluded": True,
                            "exclude_reason": reason,
                        },
                        step_counter,
                        f"{condition_id}: {fname} (excluded: {reason})",
                    )
                    continue
                c_path_str = m_entry.get("condition_path")
                cond_p = Path(c_path_str) if c_path_str else (cond_dir / fname)
            else:
                cond_p = cond_dir / fname

        originals.append(orig_p)
        cond_images.append(cond_p)

    overlay_dir = Path("showcase") / scenario_id / "overlays" / run_id / stage_key / condition_id
    showcase_img_dir = Path("showcase") / scenario_id / "images"
    showcase_img_dir.mkdir(parents=True, exist_ok=True)

    def _handle_valid_image(info: dict[str, Any]) -> None:
        nonlocal step_counter
        orig_p: Path = info["orig_path"]
        cond_p: Path = info["cond_path"]
        ref: list[Det] = info["ref"]
        q: list[Det] = info["q"]
        tp: int = int(info["tp"])
        fn: int = int(info["fn"])
        fname = orig_p.name
        stem = orig_p.stem
        seen_in_metrics.add(fname)

        ov_path = overlay_dir / f"{stem}.png"
        width, height = _render_overlay_png(cond_p, ref, q, ov_path, class_names)

        # Also write standard showcase preview path for baseline / repaired preview routes
        model_tag = "baseline" if stage_key == "baseline" else "repaired"
        target_tag = "original" if condition_id == "control" else condition_id
        compat_jpg = showcase_img_dir / f"{target_tag}_{stem}_{model_tag}.jpg"
        try:
            with Image.open(ov_path) as pim:
                pim.convert("RGB").save(compat_jpg, format="JPEG", quality=88)
        except Exception:
            pass

        matched_r, _ = _greedy_pairs(ref, q, iou=0.5)
        geom = _build_geometry(width, height, ref, q, matched_r)

        rel_ov = ov_path.as_posix()
        rel_raw = (
            f"data/{scenario_id}/real/{fname}"
            if condition_id == "control"
            else f"data/{scenario_id}/conditions/{condition_id}/{fname}"
        )
        m_entry = manifest_entries.get((condition_id, fname))
        img_alias = str(m_entry.get("image_id") or fname) if m_entry else fname

        step_counter += 1
        recent_item = {
            "image_id": fname,
            "manifest_image_id": img_alias,
            "condition": condition_id,
            "thumb_url": f"/artifacts/{rel_ov}",
            "raw_thumb_url": f"/artifacts/{rel_raw}",
            "fp32_boxes": len(ref),
            "int8_boxes": len(q),
            "matched": tp,
            "missed": fn,
            "geometry": geom,
        }
        on_image_progress(
            recent_item,
            step_counter,
            f"{condition_id}: {fname} ({ step_counter }/{ step_total })",
        )

        diagnostic_pairs.append(
            {
                "image_id": fname,
                "manifest_image_id": img_alias,
                "condition": condition_id,
                "stage": stage_key,
                "orig_path": str(orig_p),
                "cond_path": str(cond_p),
                "overlay_path": str(compat_jpg if compat_jpg.is_file() else ov_path),
                "overlay_png_path": str(ov_path),
                "tp": tp,
                "fn": fn,
                "fp": int(info["fp"]),
                "ref_count": len(ref),
                "int8_count": len(q),
            }
        )

    metrics = evaluate_condition(
        fp32=fp32_det,
        int8=int8_det,
        originals=originals,
        condition_images=cond_images,
        iou=0.5,
        pass_threshold=pass_threshold_raw,
        on_image=_handle_valid_image,
        pre_excluded_images=pre_excluded,
        pre_exclude_reasons=pre_reasons,
        requested_n_images=len(split_b_files),
    )

    # Emit progress entries for any images that were collapsed (P=0) or excluded inside evaluate_condition
    for orig_p, cond_p in zip(originals, cond_images):
        fname = orig_p.name
        if fname in seen_in_metrics:
            continue
        step_counter += 1
        rel_raw = (
            f"data/{scenario_id}/real/{fname}"
            if condition_id == "control"
            else f"data/{scenario_id}/conditions/{condition_id}/{fname}"
        )
        reason = "fp32_condition_collapse"
        if not orig_p.is_file() or not cond_p.is_file():
            reason = "missing_file"
        else:
            try:
                d0_check = fp32_det.detect(orig_p)
                if len(d0_check) == 0:
                    reason = "empty_fp32_original"
            except Exception:
                reason = "corrupt_image"

        on_image_progress(
            {
                "image_id": fname,
                "condition": condition_id,
                "thumb_url": f"/artifacts/{rel_raw}",
                "raw_thumb_url": f"/artifacts/{rel_raw}",
                "fp32_boxes": 0,
                "int8_boxes": 0,
                "matched": 0,
                "missed": 0,
                "collapsed": reason == "fp32_condition_collapse",
                "excluded": reason != "fp32_condition_collapse",
                "exclude_reason": reason,
            },
            step_counter,
            f"{condition_id}: {fname} ({reason})",
        )

    # Order diagnostic pairs so images with the most missed detections (FN) come first for Argon
    diagnostic_pairs.sort(key=lambda item: (int(item.get("fn", 0)), int(item.get("fp", 0))), reverse=True)
    return metrics, diagnostic_pairs


def _cond_status(
    base_metrics: dict[str, Any],
    final_metrics: dict[str, Any] | None,
    pass_threshold_raw: float,
) -> str:
    """Compute condition status using unrounded raw_f1 against fixed pass_threshold_raw.
    ACCEPTED gate is NEVER 'repaired' unless final raw_f1 actually crosses pass_threshold_raw.
    """
    base_raw = base_metrics.get("raw_f1")
    base_pass = _is_raw_passing(base_raw, pass_threshold_raw)
    if final_metrics is None:
        return "passed" if base_pass else "failed"
    fin_raw = final_metrics.get("raw_f1")
    fin_pass = _is_raw_passing(fin_raw, pass_threshold_raw)
    if base_pass and fin_pass:
        return "passed"
    if (not base_pass) and fin_pass:
        return "repaired"
    return "failed"


def run(
    scenario_id: str = "robot_vacuum",
    condition_ids: Sequence[str] | None = None,
    n_images: int = 20,
    mode: str = "live",
    on_update: Callable[[dict[str, Any]], None] | None = None,
    *,
    max_iterations: int | None = None,
    timeout_s: float = 150.0,
    run_id: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Execute the On-Device Crash Test closed loop on Split B and return the final state dict."""
    t_run_start = time.perf_counter()
    deadline = t_run_start + float(timeout_s)
    eff_run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    eff_mode = str(mode or "live").strip().lower()
    if max_iterations is None:
        max_iterations = 2 if eff_mode == "live" else 3

    scenario_cfg = _load_scenario_cfg(scenario_id)
    allowed_classes = [int(v) for v in (scenario_cfg.get("classes") or {}).values()]
    class_names = _class_id_to_name(scenario_cfg)
    all_cfg_conds = [
        str(c.get("id")) for c in (scenario_cfg.get("conditions") or []) if isinstance(c, dict) and c.get("id")
    ]
    if not condition_ids:
        target_conditions = list(all_cfg_conds) if eff_mode != "live" else ["fog"]
    else:
        target_conditions = [str(c).strip() for c in condition_ids if str(c).strip()]

    splits_path = Path("data") / scenario_id / "splits.json"
    splits_data = json.loads(splits_path.read_text())
    all_split_b: list[str] = [str(x) for x in splits_data.get("split_b", [])]
    eff_n_images = min(max(1, int(n_images)), len(all_split_b))
    split_b_files = all_split_b[:eff_n_images]

    manifest_entries, canonical_id_map = _load_manifest_lookup(scenario_id)
    for fname in split_b_files:
        canonical_id_map.setdefault(fname, fname)
        canonical_id_map.setdefault(Path(fname).stem, fname)

    # Load fixed threshold from baseline.json
    baseline_json_path = Path("showcase") / scenario_id / "baseline.json"
    baseline_json: dict[str, Any] = {}
    if baseline_json_path.is_file():
        try:
            baseline_json = json.loads(baseline_json_path.read_text())
        except Exception:
            baseline_json = {}

    control_sanity_threshold = float(
        baseline_json.get("control_sanity_bar") or DEFAULT_CONTROL_SANITY_THRESHOLD
    )
    fixed_threshold_raw: float | None = None
    if isinstance(baseline_json.get("condition_pass_threshold_raw"), (int, float)):
        fixed_threshold_raw = float(baseline_json["condition_pass_threshold_raw"])
    elif isinstance(baseline_json.get("control_metrics"), dict) and isinstance(
        baseline_json["control_metrics"].get("raw_f1"), (int, float)
    ):
        fixed_threshold_raw = float(baseline_json["control_metrics"]["raw_f1"]) - 0.05

    fp32_path = Path("models/fp32.onnx")
    baseline_int8_path = Path("models/baseline_int8.onnx")

    # Prepare events.jsonl writers (fresh for this run)
    events_dir = Path("showcase") / scenario_id
    events_dir.mkdir(parents=True, exist_ok=True)
    events_file = events_dir / "events.jsonl"
    run_events_dir = events_dir / "runs" / eff_run_id
    run_events_dir.mkdir(parents=True, exist_ok=True)
    run_events_file = run_events_dir / "events.jsonl"
    try:
        events_file.write_text("")
        run_events_file.write_text("")
    except Exception:
        pass

    recent_ring: list[dict[str, Any]] = []
    phase_name = "evaluating"
    phase_started_iso = _iso_now()
    phase_t0 = time.perf_counter()

    current_step: dict[str, Any] = {"done": 0, "total": eff_n_images * (1 + len(target_conditions)), "label": "Starting evaluation"}
    current_argon: dict[str, Any] | None = None
    current_action: dict[str, Any] | None = None
    current_gate: dict[str, Any] | None = None

    state: dict[str, Any] = {
        "run_id": eff_run_id,
        "scenario_id": scenario_id,
        "phase": "evaluating",
        "mode": eff_mode,
        "n_images": eff_n_images,
        "baseline_control": "not yet run",
        "final_control": "not yet run",
        "budget_reference": "not yet run",
        "baseline_int8": "not yet run",
        "fp32": "not yet run",
        "pass_threshold": round(fixed_threshold_raw, 4) if fixed_threshold_raw is not None else None,
        "condition_pass_threshold": round(fixed_threshold_raw, 4) if fixed_threshold_raw is not None else None,
        "condition_pass_threshold_raw": fixed_threshold_raw,
        "control_sanity_threshold": control_sanity_threshold,
        "conditions": {},
        "iterations": [],
        "argon_logs": [],
        "needs_human": False,
        "human_decision": None,
        "error": None,
        "progress": None,
    }

    def _emit_update(extra_state_fields: dict[str, Any] | None = None) -> None:
        prog = {
            "phase": phase_name,
            "phase_started_at": phase_started_iso,
            "phase_elapsed_s": round(time.perf_counter() - phase_t0, 3),
            "step": copy.deepcopy(current_step),
            "recent": copy.deepcopy(recent_ring[:8]),
            "argon": copy.deepcopy(current_argon),
            "action": copy.deepcopy(current_action),
            "gate": copy.deepcopy(current_gate),
        }
        state["phase"] = phase_name
        state["progress"] = prog
        if extra_state_fields:
            for k, v in extra_state_fields.items():
                state[k] = v

        event_record = {
            "ts": _iso_now(),
            "run_id": eff_run_id,
            "phase": phase_name,
            "progress": prog,
        }
        line = json.dumps(event_record) + chr(10)
        for ef in (events_file, run_events_file):
            try:
                with open(ef, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass

        if on_update is not None:
            on_update(copy.deepcopy(state))

    def _set_phase(new_phase: str, step_done: int = 0, step_total: int = 1, label: str = "") -> None:
        nonlocal phase_name, phase_started_iso, phase_t0, current_step
        phase_name = new_phase
        phase_started_iso = _iso_now()
        phase_t0 = time.perf_counter()
        current_step = {"done": step_done, "total": step_total, "label": label}

    def _push_recent(item: dict[str, Any], done_cnt: int, label: str) -> None:
        nonlocal recent_ring, current_step
        recent_ring.insert(0, item)
        if len(recent_ring) > 8:
            recent_ring = recent_ring[:8]
        current_step["done"] = done_cnt
        current_step["label"] = label
        _emit_update()

    # 1. Measure FP32 and Baseline INT8 in the SAME session (warm-20 median latency)
    _set_phase("evaluating", 0, eff_n_images * (1 + len(target_conditions)), "Measuring baseline models")
    _emit_update()

    fp32_det = Detector(fp32_path, preprocess="stretch", allowed_classes=allowed_classes)
    base_int8_det = Detector(baseline_int8_path, preprocess="stretch", allowed_classes=allowed_classes)

    fp32_ms = fp32_det.latency_ms(n_runs=20)
    base_int8_ms = base_int8_det.latency_ms(n_runs=20)
    fp32_size = fp32_path.stat().st_size
    base_int8_size = baseline_int8_path.stat().st_size
    base_q_ops, base_total_ops, base_q_ratio = count_quantized_ops(baseline_int8_path)

    fp32_measured = Measured(size_bytes=fp32_size, median_ms=fp32_ms, quantized_ops=1)
    baseline_measured = Measured(
        size_bytes=base_int8_size,
        median_ms=base_int8_ms,
        quantized_ops=base_q_ops,
    )
    _, base_budget_details = check(baseline_measured, baseline_measured, fp32=fp32_measured)

    state["fp32"] = {
        "path": str(fp32_path),
        "sha256": _sha256_file(fp32_path),
        "size_bytes": fp32_size,
        "median_ms": round(fp32_ms, 3),
    }
    state["baseline_int8"] = {
        "path": str(baseline_int8_path),
        "sha256": _sha256_file(baseline_int8_path),
        "size_bytes": base_int8_size,
        "median_ms": round(base_int8_ms, 3),
        "quantized_ops": base_q_ops,
        "total_ops": base_total_ops,
        "quantized_op_ratio": round(base_q_ratio, 4),
        "op_types_to_quantize": ["Conv"],
    }
    state["budget_reference"] = base_budget_details

    # 2. Evaluate Baseline Clean Control on Split B
    eval_total_steps = eff_n_images * (1 + len(target_conditions))
    current_step["total"] = eval_total_steps

    base_ctrl_metrics, control_pairs = _evaluate_split_b_target(
        scenario_id=scenario_id,
        condition_id="control",
        stage_key="baseline",
        run_id=eff_run_id,
        fp32_det=fp32_det,
        int8_det=base_int8_det,
        split_b_files=split_b_files,
        manifest_entries=manifest_entries,
        class_names=class_names,
        pass_threshold_raw=control_sanity_threshold,
        step_offset=0,
        step_total=eval_total_steps,
        on_image_progress=_push_recent,
    )
    base_ctrl_dict = asdict(base_ctrl_metrics)
    state["baseline_control"] = base_ctrl_dict
    state["final_control"] = copy.deepcopy(base_ctrl_dict)

    if fixed_threshold_raw is None:
        if base_ctrl_metrics.raw_f1 is not None:
            fixed_threshold_raw = float(base_ctrl_metrics.raw_f1) - 0.05
        else:
            fixed_threshold_raw = 0.80
    fixed_threshold_display = round(float(fixed_threshold_raw), 4)
    state["pass_threshold"] = fixed_threshold_display
    state["condition_pass_threshold"] = fixed_threshold_display
    state["condition_pass_threshold_raw"] = fixed_threshold_raw

    # Verify clean control sanity >= 0.80 (using unrounded raw_f1)
    if not _is_raw_passing(base_ctrl_metrics.raw_f1, control_sanity_threshold):
        state["needs_human"] = True
        state["error"] = (
            f"Baseline clean control raw_f1={base_ctrl_metrics.raw_f1} failed sanity threshold >= {control_sanity_threshold}"
        )
        _set_phase("needs_human", eval_total_steps, eval_total_steps, "Baseline control sanity check failed")
        _emit_update()
        return state

    # 3. Evaluate Requested Conditions on Split B
    baseline_cond_metrics: dict[str, dict[str, Any]] = {}
    current_cond_metrics: dict[str, dict[str, Any]] = {}
    current_pairs_by_cond: dict[str, list[dict[str, Any]]] = {}

    for idx, cid in enumerate(target_conditions):
        c_metrics, c_pairs = _evaluate_split_b_target(
            scenario_id=scenario_id,
            condition_id=cid,
            stage_key="baseline",
            run_id=eff_run_id,
            fp32_det=fp32_det,
            int8_det=base_int8_det,
            split_b_files=split_b_files,
            manifest_entries=manifest_entries,
            class_names=class_names,
            pass_threshold_raw=fixed_threshold_raw,
            step_offset=eff_n_images * (idx + 1),
            step_total=eval_total_steps,
            on_image_progress=_push_recent,
        )
        m_dict = asdict(c_metrics)
        # Ensure passed uses unrounded raw_f1 >= fixed_threshold_raw (model_limitation is INFORMATIONAL only)
        m_dict["passed"] = _is_raw_passing(c_metrics.raw_f1, fixed_threshold_raw)
        baseline_cond_metrics[cid] = m_dict
        current_cond_metrics[cid] = copy.deepcopy(m_dict)
        current_pairs_by_cond[cid] = c_pairs

        c_status = _cond_status(m_dict, None, fixed_threshold_raw)
        state["conditions"][cid] = {
            "baseline_f1": m_dict["f1"],
            "final_f1": m_dict["f1"],
            "ref_objects": m_dict["ref_objects"],
            "status": c_status,
            "limitation": bool(m_dict["limitation"]),
            "baseline_metrics": copy.deepcopy(m_dict),
            "final_metrics": copy.deepcopy(m_dict),
            "pass_threshold": fixed_threshold_display,
            "pass_threshold_raw": fixed_threshold_raw,
            "evidence_images": [p["image_id"] for p in c_pairs[:3]],
        }
        _emit_update()

    # Check if all target conditions already pass at baseline
    failing_cids = [
        cid
        for cid in target_conditions
        if not _is_raw_passing(current_cond_metrics[cid].get("raw_f1"), fixed_threshold_raw)
    ]
    if not failing_cids:
        _set_phase("done", eval_total_steps, eval_total_steps, "All conditions passed at baseline")
        _emit_update()
        return state

    # 4. Closed-Loop Repair Iterations (Argon -> R1/R2/R3 -> Re-evaluate -> Gate)
    repair_ctx = RepairContext(
        scenario_id=scenario_id,
        fp32_path=fp32_path,
        baseline_int8_path=baseline_int8_path,
        out_dir=Path("/tmp/odct_cache/candidates"),
        data_root=Path("data"),
    )
    current_ctrl_dict = copy.deepcopy(base_ctrl_dict)
    previous_iteration_pairs: list[dict[str, Any]] = []
    last_interaction_id: str | None = None

    for iteration in range(1, max_iterations + 1):
        repair_ctx.iteration = iteration
        remaining_s = deadline - time.perf_counter()
        if remaining_s <= 8.0:
            state["needs_human"] = True
            _set_phase("needs_human", iteration - 1, max_iterations, "Run time budget exhausted before repair completion")
            _emit_update()
            return state

        failing_cids = [
            cid
            for cid in target_conditions
            if not _is_raw_passing(current_cond_metrics[cid].get("raw_f1"), fixed_threshold_raw)
        ]
        if not failing_cids:
            break

        # Build diagnostic image pairs
        failing_pairs: list[dict[str, Any]] = []
        passing_pairs_by_cond: dict[str, Sequence[dict[str, Any]]] = {}
        for cid in target_conditions:
            if cid in failing_cids:
                failing_pairs.extend(current_pairs_by_cond.get(cid, []))
            else:
                passing_pairs_by_cond[cid] = current_pairs_by_cond.get(cid, [])

        selected_pairs = select_image_pairs_for_diagnosis(
            failing_pairs=failing_pairs[:6] if eff_mode == "live" else failing_pairs,
            previous_pairs=previous_iteration_pairs[:2] if eff_mode == "live" else previous_iteration_pairs,
            passing_pairs_by_cond=passing_pairs_by_cond,
            control_pairs=control_pairs[:2] if eff_mode == "live" else control_pairs,
            mode=eff_mode,
        )

        # Phase: diagnosing
        current_action = None
        current_gate = None
        current_argon = {
            "status": "sending",
            "images_sent": min(len(selected_pairs) * 2, 400),
            "input_tokens": None,
            "output_tokens": None,
            "elapsed_s": 0.0,
        }
        _set_phase(
            "diagnosing",
            iteration,
            max_iterations,
            f"Iteration {iteration}/{max_iterations}: Gemini 4 Argon diagnosing {', '.join(failing_cids)}",
        )
        _emit_update()

        def _on_argon_status(
            st: str,
            imgs_sent: int,
            usage_info: dict[str, Any] | None,
            el_s: float,
        ) -> None:
            nonlocal current_argon
            current_argon = {
                "status": st,
                "images_sent": int(imgs_sent),
                "input_tokens": (
                    int(usage_info["total_input_tokens"])
                    if isinstance(usage_info, dict) and usage_info.get("total_input_tokens") is not None
                    else None
                ),
                "output_tokens": (
                    int(usage_info["total_output_tokens"])
                    if isinstance(usage_info, dict) and usage_info.get("total_output_tokens") is not None
                    else None
                ),
                "elapsed_s": round(float(el_s), 3),
            }
            _emit_update()

        metrics_for_argon: dict[str, Any] = {
            "control": current_ctrl_dict,
            "note": (
                "IMPORTANT: model_limitation (FP32 retention < 0.70) is INFORMATIONAL only. "
                "Any condition with raw_f1 < condition_pass_threshold_raw (including fog) is a COMPRESSION FAILURE "
                "and MUST be repaired with an active R1, R2, or R3 action (e.g. R1+R2+R3 with method='percentile', "
                "per_channel=True, duplication_weight=2, targets=['fog']). Do NOT return action.type='none'."
            ),
        }
        for cid in target_conditions:
            cm = copy.deepcopy(current_cond_metrics[cid])
            cm["compression_failed"] = not _is_raw_passing(cm.get("raw_f1"), fixed_threshold_raw)
            cm["model_limitation_informational_only"] = bool(cm.get("limitation"))
            metrics_for_argon[cid] = cm

        history_for_argon = {
            "iterations": state["iterations"],
            "rejected_history": repair_ctx.rejected_history,
            "accepted_history": repair_ctx.accepted_history,
        }

        argon_timeout = min(120.0, max(5.0, deadline - time.perf_counter()))
        t_argon0 = time.perf_counter()
        try:
            action_json, usage_dict, interaction_id = diagnose(
                metrics=metrics_for_argon,
                history=state["iterations"] + repair_ctx.rejected_history,
                image_pairs=selected_pairs,
                previous_id=last_interaction_id,
                budget_info=state["budget_reference"],
                thresholds={
                    "condition_pass_threshold": fixed_threshold_display,
                    "condition_pass_threshold_raw": fixed_threshold_raw,
                    "control_sanity_threshold": control_sanity_threshold,
                },
                current_config=repair_ctx.current_config.to_dict(),
                valid_image_ids=split_b_files,
                timeout_s=argon_timeout,
                on_status=_on_argon_status,
                client=client,
            )
            argon_latency_s = round(time.perf_counter() - t_argon0, 3)
            last_interaction_id = interaction_id

            verification = verify_diagnosis(
                action_json=action_json,
                metrics_by_condition={"control": current_ctrl_dict, **current_cond_metrics},
                valid_image_ids=set(split_b_files),
                canonical_id_map=canonical_id_map,
            )
        except Exception as exc:
            state["needs_human"] = True
            state["error"] = f"Argon diagnosis failed: {exc}"
            _set_phase("needs_human", iteration, max_iterations, f"Diagnosis error: {exc}")
            _emit_update()
            return state

        diag_list = action_json.get("diagnosis") or []
        proposed_action = action_json.get("action") or {}
        rationale = str(action_json.get("rationale") or "")
        expected_effect = str(action_json.get("expected_effect") or "")

        # Update per-condition evidence_images from verified diagnosis
        for diag_entry in diag_list:
            if isinstance(diag_entry, dict):
                dcid = str(diag_entry.get("condition") or "")
                ev_imgs = diag_entry.get("evidence_images")
                if dcid in state["conditions"] and isinstance(ev_imgs, list) and ev_imgs:
                    state["conditions"][dcid]["evidence_images"] = [str(x) for x in ev_imgs]

        log_entry: dict[str, Any] = {
            "iteration": iteration,
            "interaction_id": interaction_id,
            "thinking_level": "high",
            "latency_s": argon_latency_s,
            "usage": usage_dict,
            "diagnosis": diag_list,
            "verification": verification,
            "action": proposed_action,
            "expected_effect": expected_effect,
            "gate_verdict": "pending",
            "rationale": rationale,
        }
        state["argon_logs"].append(log_entry)
        _emit_update()

        raw_action_type = str(proposed_action.get("type") or "").strip()
        if not raw_action_type or raw_action_type.lower() == "none":
            reason_msg = "Argon proposed action.type='none' while failing conditions remain below threshold"
            repair_ctx.rollback_candidate([reason_msg], action=proposed_action)
            current_gate = {
                "verdict": "REJECTED",
                "reasons": [reason_msg],
                "control_delta": 0.0,
                "budget_ratios": None,
            }
            log_entry["gate_verdict"] = "REJECTED"
            state["iterations"].append(
                {
                    "iteration": iteration,
                    "interaction_id": interaction_id,
                    "action": proposed_action,
                    "params_chips": [],
                    "gate_verdict": "REJECTED",
                    "gate": copy.deepcopy(current_gate),
                    "rationale": rationale,
                }
            )
            _set_phase("gate", iteration, max_iterations, reason_msg)
            _emit_update()
            continue

        # Phase: repairing
        try:
            _, _, preview_chips, _ = validate_and_resolve_action(proposed_action, repair_ctx)
        except RepairRejectedError as exc:
            reason_msg = str(exc)
            repair_ctx.rollback_candidate([reason_msg], action=proposed_action)
            current_action = {"type": raw_action_type, "params_chips": []}
            current_gate = {
                "verdict": "REJECTED",
                "reasons": [reason_msg],
                "control_delta": None,
                "budget_ratios": None,
            }
            log_entry["gate_verdict"] = "REJECTED"
            state["iterations"].append(
                {
                    "iteration": iteration,
                    "interaction_id": interaction_id,
                    "action": proposed_action,
                    "params_chips": [],
                    "gate_verdict": "REJECTED",
                    "gate": copy.deepcopy(current_gate),
                    "rationale": rationale,
                }
            )
            _set_phase("gate", iteration, max_iterations, reason_msg)
            _emit_update()
            continue

        current_action = {
            "type": raw_action_type,
            "params_chips": list(preview_chips),
        }
        _set_phase(
            "repairing",
            0,
            1,
            f"Iteration {iteration}: Applying {raw_action_type} ({', '.join(preview_chips)})",
        )
        _emit_update()

        try:
            quant_res = apply_repair(proposed_action, repair_ctx)
        except Exception as exc:
            reason_msg = f"Quantization repair error: {exc}"
            repair_ctx.rollback_candidate([reason_msg], action=proposed_action)
            current_gate = {
                "verdict": "REJECTED",
                "reasons": [reason_msg],
                "control_delta": None,
                "budget_ratios": None,
            }
            log_entry["gate_verdict"] = "REJECTED"
            state["iterations"].append(
                {
                    "iteration": iteration,
                    "interaction_id": interaction_id,
                    "action": proposed_action,
                    "params_chips": list(preview_chips),
                    "gate_verdict": "REJECTED",
                    "gate": copy.deepcopy(current_gate),
                    "rationale": rationale,
                }
            )
            _set_phase("gate", iteration, max_iterations, reason_msg)
            _emit_update()
            continue

        current_step["done"] = 1
        current_step["label"] = f"Quantized candidate {Path(quant_res.path).name}"
        _emit_update()

        # Phase: re-evaluating
        reeval_total_steps = eff_n_images * (1 + len(target_conditions))
        _set_phase(
            "re-evaluating",
            0,
            reeval_total_steps,
            f"Iteration {iteration}: Re-evaluating candidate INT8 on Split B",
        )
        _emit_update()

        cand_path = Path(quant_res.path)
        cand_det = Detector(cand_path, preprocess="stretch", allowed_classes=allowed_classes)
        cand_ms = cand_det.latency_ms(n_runs=20)
        cand_size = cand_path.stat().st_size
        cand_q_ops, cand_total_ops, cand_q_ratio = count_quantized_ops(cand_path)
        cand_measured = Measured(
            size_bytes=cand_size,
            median_ms=cand_ms,
            quantized_ops=cand_q_ops,
        )

        cand_ctrl_metrics, cand_ctrl_pairs = _evaluate_split_b_target(
            scenario_id=scenario_id,
            condition_id="control",
            stage_key=f"iter{iteration}",
            run_id=eff_run_id,
            fp32_det=fp32_det,
            int8_det=cand_det,
            split_b_files=split_b_files,
            manifest_entries=manifest_entries,
            class_names=class_names,
            pass_threshold_raw=control_sanity_threshold,
            step_offset=0,
            step_total=reeval_total_steps,
            on_image_progress=_push_recent,
        )
        cand_ctrl_dict = asdict(cand_ctrl_metrics)

        cand_cond_metrics: dict[str, dict[str, Any]] = {}
        cand_pairs_by_cond: dict[str, list[dict[str, Any]]] = {}
        for idx, cid in enumerate(target_conditions):
            cm, cpairs = _evaluate_split_b_target(
                scenario_id=scenario_id,
                condition_id=cid,
                stage_key=f"iter{iteration}",
                run_id=eff_run_id,
                fp32_det=fp32_det,
                int8_det=cand_det,
                split_b_files=split_b_files,
                manifest_entries=manifest_entries,
                class_names=class_names,
                pass_threshold_raw=fixed_threshold_raw,
                step_offset=eff_n_images * (idx + 1),
                step_total=reeval_total_steps,
                on_image_progress=_push_recent,
            )
            cm_dict = asdict(cm)
            cm_dict["passed"] = _is_raw_passing(cm.raw_f1, fixed_threshold_raw)
            cand_cond_metrics[cid] = cm_dict
            cand_pairs_by_cond[cid] = cpairs

        # Phase: gate
        budget_ok, budget_details = check(cand_measured, baseline_measured, fp32=fp32_measured)
        gate_reasons: list[str] = list(budget_details.get("reasons") or [])

        base_ctrl_raw = base_ctrl_dict.get("raw_f1")
        cand_ctrl_raw = cand_ctrl_dict.get("raw_f1")
        control_delta_raw: float | None = None
        if isinstance(base_ctrl_raw, (int, float)) and isinstance(cand_ctrl_raw, (int, float)):
            control_delta_raw = float(cand_ctrl_raw) - float(base_ctrl_raw)

        if not _is_raw_passing(cand_ctrl_raw, control_sanity_threshold):
            gate_reasons.append(
                f"control raw_f1 {cand_ctrl_raw} dropped below sanity threshold {control_sanity_threshold}"
            )
        if control_delta_raw is not None and (float(base_ctrl_raw) - float(cand_ctrl_raw)) > 0.02:
            gate_reasons.append(
                f"control raw_f1 dropped by {float(base_ctrl_raw) - float(cand_ctrl_raw):.4f} (> 0.02 max drop)"
            )

        # Zero-regression check: no previously passing condition may drop below fixed_threshold_raw
        for cid in target_conditions:
            prev_raw = current_cond_metrics[cid].get("raw_f1")
            new_raw = cand_cond_metrics[cid].get("raw_f1")
            if _is_raw_passing(prev_raw, fixed_threshold_raw) and not _is_raw_passing(new_raw, fixed_threshold_raw):
                gate_reasons.append(
                    f"previously passing condition '{cid}' regressed below threshold ({new_raw:.4f} < {fixed_threshold_raw:.4f})"
                )

        # Ensure candidate improves at least one failing condition (or crosses threshold)
        improved_any = False
        for cid in failing_cids:
            prev_raw = current_cond_metrics[cid].get("raw_f1")
            new_raw = cand_cond_metrics[cid].get("raw_f1")
            if isinstance(new_raw, (int, float)) and (
                prev_raw is None or float(new_raw) > float(prev_raw) or _is_raw_passing(new_raw, fixed_threshold_raw)
            ):
                improved_any = True
                break
        if not improved_any:
            gate_reasons.append("candidate did not improve F1 on any failing condition")

        verdict = "ACCEPTED" if (budget_ok and len(gate_reasons) == 0) else "REJECTED"
        control_delta_display = round(control_delta_raw, 4) if control_delta_raw is not None else None

        current_gate = {
            "verdict": verdict,
            "reasons": gate_reasons,
            "control_delta": control_delta_display,
            "budget_ratios": budget_details,
        }
        log_entry["gate_verdict"] = verdict

        before_after_conds: dict[str, Any] = {}
        for cid in target_conditions:
            before_after_conds[cid] = {
                "before_f1": current_cond_metrics[cid].get("f1"),
                "before_raw_f1": current_cond_metrics[cid].get("raw_f1"),
                "after_f1": cand_cond_metrics[cid].get("f1"),
                "after_raw_f1": cand_cond_metrics[cid].get("raw_f1"),
                "passed": _is_raw_passing(cand_cond_metrics[cid].get("raw_f1"), fixed_threshold_raw),
            }

        if verdict == "ACCEPTED":
            repair_ctx.commit_candidate()
            current_ctrl_dict = cand_ctrl_dict
            current_cond_metrics = cand_cond_metrics
            control_pairs = cand_ctrl_pairs
            previous_iteration_pairs = list(failing_pairs)
            current_pairs_by_cond = cand_pairs_by_cond

            state["final_control"] = copy.deepcopy(current_ctrl_dict)
            state["repaired_model_path"] = str(cand_path)
            state["final_int8"] = {
                "path": str(cand_path),
                "sha256": _sha256_file(cand_path),
                "size_bytes": cand_size,
                "median_ms": round(cand_ms, 3),
                "quantized_ops": cand_q_ops,
                "total_ops": cand_total_ops,
                "quantized_op_ratio": round(cand_q_ratio, 4),
                "config": repair_ctx.current_config.to_dict(),
            }
            for cid in target_conditions:
                fin_m = current_cond_metrics[cid]
                base_m = baseline_cond_metrics[cid]
                state["conditions"][cid].update(
                    {
                        "final_f1": fin_m["f1"],
                        "final_metrics": copy.deepcopy(fin_m),
                        "status": _cond_status(base_m, fin_m, fixed_threshold_raw),
                    }
                )
        else:
            repair_ctx.rollback_candidate(gate_reasons, action=proposed_action)

        state["iterations"].append(
            {
                "iteration": iteration,
                "interaction_id": interaction_id,
                "action": proposed_action,
                "params_chips": list(preview_chips),
                "model_path": str(cand_path),
                "gate_verdict": verdict,
                "gate": copy.deepcopy(current_gate),
                "control_before_f1": base_ctrl_dict.get("f1"),
                "control_after_f1": cand_ctrl_dict.get("f1"),
                "control_delta": control_delta_display,
                "budget": budget_details,
                "conditions": before_after_conds,
                "rationale": rationale,
            }
        )
        state["rejected_history"] = copy.deepcopy(repair_ctx.rejected_history)

        _set_phase("gate", iteration, max_iterations, f"Iteration {iteration} gate: {verdict}")
        _emit_update()

        # Check if all target conditions now cross fixed_threshold_raw
        still_failing = [
            cid
            for cid in target_conditions
            if not _is_raw_passing(current_cond_metrics[cid].get("raw_f1"), fixed_threshold_raw)
        ]
        if not still_failing:
            break

    # Finalize run state
    unresolved = [
        cid
        for cid in target_conditions
        if not _is_raw_passing(current_cond_metrics[cid].get("raw_f1"), fixed_threshold_raw)
    ]
    for cid in target_conditions:
        base_m = baseline_cond_metrics[cid]
        fin_m = current_cond_metrics[cid]
        state["conditions"][cid]["final_f1"] = fin_m["f1"]
        state["conditions"][cid]["final_metrics"] = copy.deepcopy(fin_m)
        state["conditions"][cid]["status"] = _cond_status(base_m, fin_m, fixed_threshold_raw)

    total_elapsed_s = round(time.perf_counter() - t_run_start, 3)
    state["elapsed_s"] = total_elapsed_s

    if unresolved:
        state["needs_human"] = True
        _set_phase(
            "needs_human",
            len(state["iterations"]),
            max_iterations,
            f"Needs human review: {', '.join(unresolved)} below threshold {fixed_threshold_display:.4f}",
        )
    else:
        state["needs_human"] = False
        _set_phase(
            "done",
            len(state["iterations"]),
            max_iterations,
            f"Completed in {total_elapsed_s:.1f}s — all target conditions passing",
        )

    _emit_update()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the On-Device Crash Test closed loop.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", help="Run in live mode (default 20 images, max 2 iterations)")
    mode_group.add_argument("--showcase", action="store_true", help="Run in showcase mode (40 images, max 3 iterations)")
    parser.add_argument("--scenario", default="robot_vacuum", help="Scenario ID")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["fog"],
        help="Condition ID(s) to evaluate and repair (space or comma separated)",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of Split B images to evaluate")
    parser.add_argument("--max-iterations", type=int, default=None, help="Maximum repair iterations")
    parser.add_argument("--timeout", type=float, default=150.0, help="Overall timeout in seconds")
    args = parser.parse_args()

    conds: list[str] = []
    for raw_c in args.conditions:
        for part in str(raw_c).split(","):
            p = part.strip()
            if p:
                conds.append(p)

    mode = "showcase" if args.showcase else "live"
    n_imgs = 40 if (args.showcase and args.n == 20) else args.n

    last_printed_phase: str | None = None

    def _cli_progress(st: dict[str, Any]) -> None:
        nonlocal last_printed_phase
        prog = st.get("progress") or {}
        ph = prog.get("phase") or st.get("phase")
        step = prog.get("step") or {}
        ar = prog.get("argon")
        gate = prog.get("gate")
        if ph != last_printed_phase:
            print(f"[phase={ph}] {step.get('label', '')}", flush=True)
            last_printed_phase = ph
        elif ph == "diagnosing" and isinstance(ar, dict):
            print(
                f"  [argon status={ar.get('status')}] images_sent={ar.get('images_sent')} "
                f"in_tokens={ar.get('input_tokens')} out_tokens={ar.get('output_tokens')} "
                f"elapsed_s={ar.get('elapsed_s')}",
                flush=True,
            )
        elif ph == "gate" and isinstance(gate, dict):
            print(
                f"  [gate verdict={gate.get('verdict')}] control_delta={gate.get('control_delta')} "
                f"reasons={gate.get('reasons')}",
                flush=True,
            )

    final_state = run(
        scenario_id=args.scenario,
        condition_ids=conds,
        n_images=n_imgs,
        mode=mode,
        on_update=_cli_progress,
        max_iterations=args.max_iterations,
        timeout_s=args.timeout,
    )

    summary = {
        "run_id": final_state.get("run_id"),
        "scenario_id": final_state.get("scenario_id"),
        "phase": final_state.get("phase"),
        "mode": final_state.get("mode"),
        "n_images": final_state.get("n_images"),
        "elapsed_s": final_state.get("elapsed_s"),
        "pass_threshold": final_state.get("pass_threshold"),
        "condition_pass_threshold_raw": final_state.get("condition_pass_threshold_raw"),
        "baseline_control_f1": (final_state.get("baseline_control") or {}).get("f1")
        if isinstance(final_state.get("baseline_control"), dict)
        else None,
        "final_control_f1": (final_state.get("final_control") or {}).get("f1")
        if isinstance(final_state.get("final_control"), dict)
        else None,
        "conditions": {
            cid: {
                "baseline_f1": cinfo.get("baseline_f1"),
                " baseline_raw_f1".strip(): (cinfo.get("baseline_metrics") or {}).get("raw_f1"),
                "final_f1": cinfo.get("final_f1"),
                "final_raw_f1": (cinfo.get("final_metrics") or {}).get("raw_f1"),
                "retention": (cinfo.get("final_metrics") or {}).get("retention"),
                "limitation": cinfo.get("limitation"),
                "status": cinfo.get("status"),
                "evidence_images": cinfo.get("evidence_images"),
            }
            for cid, cinfo in (final_state.get("conditions") or {}).items()
        },
        "argon_logs": [
            {
                "iteration": lg.get("iteration"),
                "interaction_id": lg.get("interaction_id"),
                "thinking_level": lg.get("thinking_level"),
                "latency_s": lg.get("latency_s"),
                "usage": lg.get("usage"),
                "action": lg.get("action"),
                "gate_verdict": lg.get("gate_verdict"),
                "rationale": lg.get("rationale"),
            }
            for lg in (final_state.get("argon_logs") or [])
        ],
        "iterations": final_state.get("iterations"),
        "needs_human": final_state.get("needs_human"),
        "error": final_state.get("error"),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
