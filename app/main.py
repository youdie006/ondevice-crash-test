"""FastAPI web service for On-Device Crash Test."""

from __future__ import annotations

import copy
import io
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from app.state import STORE, compute_pass_threshold, load_default_state

app = FastAPI(title="On-Device Crash Test")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_DETECTOR_LOCK = threading.Lock()
_DETECTOR_CACHE: dict[tuple[str, str, float, tuple[int, ...]], Any] = {}
_PREVIEW_IMAGE_CACHE: dict[tuple[str, str, str, str, float, float], bytes] = {}
_PREVIEW_META_CACHE: dict[tuple[str, str, str, str], dict[str, int]] = {}


class RunRequest(BaseModel):
    scenario_id: str = "robot_vacuum"
    conditions: list[str] = Field(default_factory=lambda: ["night"])
    n_images: int = 20
    mode: str = "live"


class HumanDecisionRequest(BaseModel):
    run_id: str | None = None
    decision: str


def _allowed_roots() -> list[Path]:
    cwd = Path.cwd().resolve()
    return [
        (cwd / "data").resolve(),
        (cwd / "showcase").resolve(),
        (cwd / "models").resolve(),
    ]


def _is_within_allowed_roots(candidate: Path) -> bool:
    resolved = candidate.resolve()
    return any(resolved.is_relative_to(root) for root in _allowed_roots())


def _safe_resolve_under(base: Path, *parts: str) -> Path | None:
    try:
        base_resolved = base.resolve()
        candidate = base_resolved.joinpath(*parts).resolve()
        if candidate.is_relative_to(base_resolved) and _is_within_allowed_roots(candidate):
            return candidate
    except Exception:
        return None
    return None


def _load_scenario_profile(scenario_id: str) -> dict[str, Any]:
    if not scenario_id or "/" in scenario_id or chr(92) in scenario_id or ".." in scenario_id:
        return {"id": str(scenario_id), "name": str(scenario_id), "status": "not yet run", "conditions": [], "classes": {}}
    path = Path("scenarios") / f"{scenario_id}.yaml"
    if not path.exists():
        return {"id": scenario_id, "name": scenario_id, "status": "not yet run", "conditions": [], "classes": {}}
    try:
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"id": scenario_id, "name": scenario_id, "status": "not yet run", "conditions": [], "classes": {}}


def _class_id_to_name_map(profile: dict[str, Any]) -> dict[str, str]:
    classes = profile.get("classes") or {}
    mapping: dict[str, str] = {}
    if isinstance(classes, dict):
        for name, cid in classes.items():
            mapping[str(cid)] = str(name)
    return mapping


def _load_splits_data(scenario_id: str) -> dict[str, Any]:
    splits_path = _safe_resolve_under(Path("data"), scenario_id, "splits.json")
    if splits_path and splits_path.is_file():
        try:
            data = json.loads(splits_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


HERO_PREFERRED_FILES: tuple[str, ...] = (
    "000000000139.jpg",
    "000000004495.jpg",
    "000000012576.jpg",
)


def _norm_img_stem(image_id: str) -> str:
    return Path(str(image_id)).stem


def _norm_img_filename(image_id: str) -> str:
    s = str(image_id).strip()
    if not s:
        return ""
    name = Path(s).name
    if "." not in name:
        return f"{name}.jpg"
    return name


def _load_raw_scenario_manifest(scenario_id: str) -> dict[str, Any]:
    manifest_path = _safe_resolve_under(Path("data"), scenario_id, "manifest.json")
    if manifest_path and manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _load_condition_manifest_entries(
    scenario_id: str,
    condition_id: str,
    raw_scenario_manifest: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (entries, condition_block_meta) from scenario manifest or per-condition manifest fallback."""
    scen_man = raw_scenario_manifest if isinstance(raw_scenario_manifest, dict) else _load_raw_scenario_manifest(scenario_id)
    conds_block = scen_man.get("conditions")
    if isinstance(conds_block, dict) and isinstance(conds_block.get(condition_id), dict):
        cblock = conds_block[condition_id]
        entries = cblock.get("entries")
        if isinstance(entries, list):
            return [e for e in entries if isinstance(e, dict)], cblock

    cond_man_path = _safe_resolve_under(Path("data"), scenario_id, "conditions", condition_id, "manifest.json")
    if cond_man_path and cond_man_path.is_file():
        try:
            cdata = json.loads(cond_man_path.read_text())
            if isinstance(cdata, dict) and isinstance(cdata.get("entries"), list):
                return [e for e in cdata["entries"] if isinstance(e, dict)], cdata
            if isinstance(cdata, list):
                return [e for e in cdata if isinstance(e, dict)], {}
        except Exception:
            pass
    return [], {}


def _resolve_manifest_entry_for_split_b(
    scenario_id: str,
    condition_id: str | None,
    image_id: str,
    raw_scenario_manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    stem = _norm_img_stem(image_id)
    if not stem:
        return None
    scen_man = raw_scenario_manifest if isinstance(raw_scenario_manifest, dict) else _load_raw_scenario_manifest(scenario_id)
    cond_ids = [condition_id] if condition_id else list((scen_man.get("conditions") or {}).keys())
    for cid in cond_ids:
        if not cid:
            continue
        entries, _ = _load_condition_manifest_entries(scenario_id, str(cid), scen_man)
        for entry in entries:
            if str(entry.get("split") or "").upper() != "B":
                continue
            e_id = str(entry.get("image_id") or "")
            e_src = str(entry.get("source_image_id") or "")
            e_fname = str(entry.get("file_name") or "")
            if stem in (e_id, e_src, _norm_img_stem(e_fname)):
                return entry
    return None


def _find_split_b_original(scenario_id: str, image_id: str) -> Path | None:
    entry = _resolve_manifest_entry_for_split_b(scenario_id, None, image_id)
    if entry and isinstance(entry.get("original_path"), str):
        cand = _safe_resolve_under(Path.cwd(), entry["original_path"])
        if cand and cand.is_file():
            return cand

    fname = _norm_img_filename(image_id)
    if not fname:
        return None
    splits = _load_splits_data(scenario_id)
    split_b = splits.get("split_b")
    if isinstance(split_b, list) and split_b:
        stems = {_norm_img_stem(x): str(x) for x in split_b}
        stem = _norm_img_stem(fname)
        if stem not in stems:
            return None
        fname = stems[stem]
    p = _safe_resolve_under(Path("data"), scenario_id, "real", fname)
    if p and p.is_file():
        return p
    return None


def _find_condition_image(scenario_id: str, condition_id: str, image_id: str) -> Path | None:
    if not condition_id or "/" in condition_id or ".." in condition_id:
        return None
    entry = _resolve_manifest_entry_for_split_b(scenario_id, condition_id, image_id)
    if entry and isinstance(entry.get("condition_path"), str):
        cand = _safe_resolve_under(Path.cwd(), entry["condition_path"])
        if cand and cand.is_file():
            return cand

    stem = _norm_img_stem(image_id)
    if not stem:
        return None
    for ext in (".jpg", ".png", ".jpeg"):
        cand = _safe_resolve_under(Path("data"), scenario_id, "conditions", condition_id, f"{stem}{ext}")
        if cand and cand.is_file():
            return cand
    return None


def _find_showcase_overlay(scenario_id: str, target: str, image_id: str, model_key: str = "baseline") -> Path | None:
    stem = _norm_img_stem(image_id)
    if not stem:
        return None
    # For fp32 or repaired, ONLY allow files that explicitly specify that model_key so baseline is never mislabeled as repaired
    explicit_candidates = [
        (scenario_id, "images", f"{target}_{stem}_{model_key}.jpg"),
        (scenario_id, "images", target, f"{stem}_{model_key}.jpg"),
        (scenario_id, "overlays", target, f"{stem}_{model_key}.jpg"),
    ]
    for parts in explicit_candidates:
        cand = _safe_resolve_under(Path("showcase"), *parts)
        if cand and cand.is_file():
            return cand

    if model_key == "baseline":
        fallback_candidates = [
            (scenario_id, "images", f"{target}_{stem}.jpg"),
            (scenario_id, "images", target, f"{stem}.jpg"),
            (scenario_id, "overlays", target, f"{stem}.jpg"),
        ]
        for parts in fallback_candidates:
            cand = _safe_resolve_under(Path("showcase"), *parts)
            if cand and cand.is_file():
                return cand
    return None


def _load_manifest_summary(scenario_id: str, profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    scen_man = _load_raw_scenario_manifest(scenario_id)
    conditions_cfg = profile.get("conditions") or []

    for c in conditions_cfg:
        cid = c.get("id")
        if not cid:
            continue
        entries, cblock = _load_condition_manifest_entries(scenario_id, cid, scen_man)
        split_b_entries = [e for e in entries if str(e.get("split") or "").upper() == "B"]
        b_summary = cblock.get("split_b_summary") if isinstance(cblock.get("split_b_summary"), dict) else {}

        generators_used: list[str] = []
        generator_labels: list[str] = []
        valid_b_count = 0
        invalid_b_count = 0
        fallback_b_count = 0
        audited_b_count = 0
        audit_failed_b_count = 0
        fallback_reasons: list[str] = []
        entries_by_id: dict[str, dict[str, Any]] = {}

        for item in split_b_entries:
            cond_path = item.get("condition_path")
            file_ok = False
            if isinstance(cond_path, str) and cond_path:
                resolved_p = _safe_resolve_under(Path.cwd(), cond_path)
                file_ok = resolved_p is not None and resolved_p.is_file()
            else:
                fname = str(item.get("file_name") or item.get("image_id") or "")
                file_ok = _find_condition_image(scenario_id, cid, fname) is not None

            is_valid = bool(
                file_ok
                and item.get("valid", True) is not False
                and item.get("allow_in_pipeline", True) is not False
                and str(item.get("validity_status") or "valid") != "invalid"
            )
            if is_valid:
                valid_b_count += 1
            else:
                invalid_b_count += 1

            g_code = item.get("generator")
            if isinstance(g_code, str) and g_code and g_code not in generators_used:
                generators_used.append(g_code)
            g_label = item.get("generator_label") or g_code
            if isinstance(g_label, str) and g_label and g_label not in generator_labels:
                generator_labels.append(g_label)

            if item.get("used_fallback") is True:
                fallback_b_count += 1
                f_reason = item.get("fallback_reason")
                if isinstance(f_reason, str) and f_reason and f_reason not in fallback_reasons:
                    fallback_reasons.append(f_reason)

            a_passed = item.get("audit_passed")
            if isinstance(a_passed, bool):
                audited_b_count += 1
                if not a_passed:
                    audit_failed_b_count += 1

            compact_entry = {
                "image_id": item.get("image_id"),
                "source_image_id": item.get("source_image_id"),
                "file_name": item.get("file_name"),
                "valid": is_valid,
                "validity_status": item.get("validity_status"),
                "allow_in_pipeline": item.get("allow_in_pipeline"),
                "retention": item.get("retention"),
                "generator": item.get("generator"),
                "generator_label": item.get("generator_label"),
                "audit_status": item.get("audit_status"),
                "audit_passed": item.get("audit_passed"),
                "used_fallback": item.get("used_fallback"),
                "fallback_reason": item.get("fallback_reason"),
                "thumb_url": item.get("thumb_url"),
                "original_url": item.get("original_url"),
            }
            for k in (
                str(item.get("image_id") or ""),
                str(item.get("source_image_id") or ""),
                str(item.get("file_name") or ""),
            ):
                if k:
                    entries_by_id[k] = compact_entry

        if not split_b_entries:
            splits = _load_splits_data(scenario_id)
            split_b_files = splits.get("split_b") if isinstance(splits.get("split_b"), list) else []
            for fname in split_b_files:
                if _find_condition_image(scenario_id, cid, str(fname)) is not None:
                    valid_b_count += 1

        audit_failure_rate: float | None = None
        if audited_b_count > 0:
            audit_failure_rate = round(float(audit_failed_b_count) / float(audited_b_count), 4)

        summary[cid] = {
            "condition_id": cid,
            "requested_generator": c.get("generator"),
            "fallback_configured": c.get("fallback"),
            "generators_used": generators_used or (b_summary.get("generators_used") or []),
            "generator_label": ", ".join(generator_labels) if generator_labels else "evaluation pending",
            "split_b_count": len(split_b_entries) if split_b_entries else int(b_summary.get("count") or 0),
            "available_images": valid_b_count,
            "valid_b_count": valid_b_count,
            "invalid_b_count": invalid_b_count,
            "fallback_b_count": fallback_b_count,
            "fallback_reason": ", ".join(fallback_reasons) if fallback_reasons else None,
            "audited_count": audited_b_count,
            "audit_failed_count": audit_failed_b_count,
            "audit_failure_rate": audit_failure_rate,
            "retention_b": b_summary.get("retention"),
            "entries_by_id": entries_by_id,
        }

    return summary


def _get_cached_detector(onnx_path: Path, preprocess: str, allowed_classes: list[int]) -> Any:
    from odct.infer import Detector

    resolved = onnx_path.resolve()
    mtime = resolved.stat().st_mtime
    key = (str(resolved), preprocess, mtime, tuple(sorted(allowed_classes)))
    with _DETECTOR_LOCK:
        det = _DETECTOR_CACHE.get(key)
        if det is None:
            det = Detector(resolved, preprocess=preprocess, allowed_classes=allowed_classes)
            _DETECTOR_CACHE[key] = det
        return det


def _resolve_model_path_for_preview(state: dict[str, Any], model_key: str) -> Path | None:
    if model_key == "fp32":
        fp32_info = state.get("fp32")
        p_str = fp32_info.get("path") if isinstance(fp32_info, dict) else "models/fp32.onnx"
        cand = _safe_resolve_under(Path.cwd(), str(p_str or "models/fp32.onnx"))
        return cand if (cand and cand.is_file()) else None

    if model_key == "repaired":
        candidates: list[str] = []
        for field_name in ("repaired_model_path", "final_model_path"):
            val = state.get(field_name)
            if isinstance(val, str) and val:
                candidates.append(val)
        final_int8 = state.get("final_int8")
        if isinstance(final_int8, dict) and isinstance(final_int8.get("path"), str):
            candidates.append(final_int8["path"])
        iterations = state.get("iterations")
        if isinstance(iterations, list):
            for it in reversed(iterations):
                if isinstance(it, dict):
                    verdict = str(it.get("gate_verdict") or (it.get("gate") or {}).get("verdict") or "")
                    mpath = it.get("model_path") or it.get("int8_path")
                    if verdict.upper().startswith("ACCEPTED") and isinstance(mpath, str):
                        candidates.append(mpath)
                        break
        for c_str in candidates:
            cand = _safe_resolve_under(Path.cwd(), c_str)
            if cand and cand.is_file():
                return cand
        return None

    base_info = state.get("baseline_int8")
    p_str = base_info.get("path") if isinstance(base_info, dict) else "models/baseline_int8.onnx"
    cand = _safe_resolve_under(Path.cwd(), str(p_str or "models/baseline_int8.onnx"))
    return cand if (cand and cand.is_file()) else None


def _render_overlay_bytes(
    scenario_id: str,
    target: str,
    image_id: str,
    model_key: str,
    state: dict[str, Any],
) -> tuple[bytes, dict[str, int]] | None:
    orig_path = _find_split_b_original(scenario_id, image_id)
    if orig_path is None:
        return None

    if target == "original":
        target_path = orig_path
    else:
        target_path = _find_condition_image(scenario_id, target, image_id)
        if target_path is None:
            return None

    fp32_path = _resolve_model_path_for_preview(state, "fp32")
    if fp32_path is None:
        return None

    int8_path: Path | None = None
    if model_key != "fp32":
        int8_path = _resolve_model_path_for_preview(state, model_key)
        if int8_path is None:
            return None

    target_mtime = target_path.stat().st_mtime
    model_mtime = int8_path.stat().st_mtime if int8_path else fp32_path.stat().st_mtime
    cache_key = (scenario_id, target, _norm_img_stem(image_id), model_key, target_mtime, model_mtime)
    meta_key = (scenario_id, target, _norm_img_stem(image_id), model_key)

    if cache_key in _PREVIEW_IMAGE_CACHE and meta_key in _PREVIEW_META_CACHE:
        return _PREVIEW_IMAGE_CACHE[cache_key], _PREVIEW_META_CACHE[meta_key]

    profile = _load_scenario_profile(scenario_id)
    classes_cfg = profile.get("classes") or {}
    allowed_classes = [int(v) for v in classes_cfg.values()] if isinstance(classes_cfg, dict) else []
    cls_names = _class_id_to_name_map(profile)

    try:
        from odct.metrics import match, persisted

        fp32_det = _get_cached_detector(fp32_path, "stretch", allowed_classes)
        d0 = fp32_det.detect(orig_path)
        if target == "original":
            dc = d0
            ref = d0
        else:
            dc = fp32_det.detect(target_path)
            ref = persisted(d0, dc, iou=0.5)

        if model_key == "fp32" or int8_path is None:
            q = []
            tp, fp, fn = len(ref), 0, 0
        else:
            int8_det = _get_cached_detector(int8_path, "stretch", allowed_classes)
            q = int8_det.detect(target_path)
            tp, fp, fn = match(ref, q, iou=0.5, dc_all=dc)

        with Image.open(target_path) as im:
            canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)

        # Draw FP32 reference boxes in green (#16a34a)
        for det in ref:
            x1, y1, x2, y2 = det.box
            draw.rectangle([x1, y1, x2, y2], outline=(22, 163, 74), width=3)
            cname = cls_names.get(str(det.cls), str(det.cls))
            label = f"FP32 {cname}"
            ty = max(0.0, y1 - 14.0)
            draw.rectangle([x1, ty, min(float(canvas.width), x1 + len(label) * 6 + 6), ty + 13], fill=(22, 163, 74))
            draw.text((x1 + 3, ty + 1), label, fill=(255, 255, 255))

        # Draw INT8 boxes in orange (#ea580c), slightly inset so overlapping boxes stay visible
        for det in q:
            x1, y1, x2, y2 = det.box
            ix1 = min(x2, x1 + 2.0)
            iy1 = min(y2, y1 + 2.0)
            ix2 = max(ix1, x2 - 2.0)
            iy2 = max(iy1, y2 - 2.0)
            draw.rectangle([ix1, iy1, ix2, iy2], outline=(234, 88, 12), width=2)
            cname = cls_names.get(str(det.cls), str(det.cls))
            label = f"INT8 {cname}"
            by = min(float(canvas.height - 14), y2 - 14.0) if (y2 - y1) > 28 else min(float(canvas.height - 14), y2)
            draw.rectangle([ix1, by, min(float(canvas.width), ix1 + len(label) * 6 + 6), by + 13], fill=(234, 88, 12))
            draw.text((ix1 + 3, by + 1), label, fill=(255, 255, 255))

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=88)
        img_bytes = buf.getvalue()
        meta = {
            "fp32_boxes": len(ref),
            "int8_boxes": len(q),
            "matched": tp,
            "missed": fn,
            "fp": fp,
        }
        if len(_PREVIEW_IMAGE_CACHE) > 256:
            _PREVIEW_IMAGE_CACHE.clear()
        _PREVIEW_IMAGE_CACHE[cache_key] = img_bytes
        _PREVIEW_META_CACHE[meta_key] = meta
        return img_bytes, meta
    except Exception:
        return None


def _select_crash_wall_image_id(scenario_id: str, profile: dict[str, Any], state: dict[str, Any]) -> str | None:
    explicit = state.get("crash_wall_image_id")
    if isinstance(explicit, str) and explicit and _find_split_b_original(scenario_id, explicit):
        orig_p = _find_split_b_original(scenario_id, explicit)
        return orig_p.name if orig_p else _norm_img_filename(explicit)

    cw = state.get("crash_wall")
    if isinstance(cw, dict) and isinstance(cw.get("image_id"), str):
        orig_p = _find_split_b_original(scenario_id, cw["image_id"])
        if orig_p:
            return orig_p.name

    scen_man = _load_raw_scenario_manifest(scenario_id)
    hero_cfg = scen_man.get("hero_crash_wall")
    if isinstance(hero_cfg, dict) and str(hero_cfg.get("split") or "B").upper() == "B":
        hero_fname = hero_cfg.get("file_name") or hero_cfg.get("image_id")
        if isinstance(hero_fname, str) and hero_fname:
            orig_p = _find_split_b_original(scenario_id, hero_fname)
            if orig_p is not None:
                return orig_p.name

    for pref in HERO_PREFERRED_FILES:
        orig_p = _find_split_b_original(scenario_id, pref)
        if orig_p is not None:
            return orig_p.name

    splits = _load_splits_data(scenario_id)
    split_b = splits.get("split_b") if isinstance(splits.get("split_b"), list) else []
    cond_ids = [str(c.get("id")) for c in (profile.get("conditions") or []) if c.get("id")]

    if split_b:
        best_img = str(split_b[0])
        best_count = -1
        for img_name in split_b[:20]:
            fname = str(img_name)
            avail = sum(1 for cid in cond_ids if _find_condition_image(scenario_id, cid, fname) is not None)
            if avail > best_count:
                best_count = avail
                best_img = fname
                if avail == len(cond_ids) and avail > 0:
                    break
        return best_img

    return None


def _extract_raw_pass_threshold(state: dict[str, Any], scenario_id: str) -> float | None:
    raw_t = state.get("pass_threshold_raw") or state.get("condition_pass_threshold_raw")
    if isinstance(raw_t, (int, float)) and not isinstance(raw_t, bool):
        return float(raw_t)
    b_ctrl = state.get("baseline_control")
    if isinstance(b_ctrl, dict):
        for k in ("raw_f1", "f1"):
            v = b_ctrl.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v) - 0.05
    baseline_path = _safe_resolve_under(Path("showcase"), scenario_id, "baseline.json")
    if baseline_path and baseline_path.is_file():
        try:
            bdata = json.loads(baseline_path.read_text())
            if isinstance(bdata, dict):
                v = bdata.get("condition_pass_threshold_raw")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
        except Exception:
            pass
    t = state.get("pass_threshold")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    return None


def _normalize_condition_entry(
    cid: str,
    cstate: dict[str, Any],
    raw_threshold: float | None,
    disp_threshold: float | None,
) -> dict[str, Any]:
    norm = copy.deepcopy(cstate) if isinstance(cstate, dict) else {}
    bm = norm.get("baseline_metrics") if isinstance(norm.get("baseline_metrics"), dict) else None
    fm = norm.get("final_metrics") if isinstance(norm.get("final_metrics"), dict) else None

    b_f1 = norm.get("baseline_f1")
    if not isinstance(b_f1, (int, float)) or isinstance(b_f1, bool):
        b_f1 = bm.get("f1") if (bm and isinstance(bm.get("f1"), (int, float))) else norm.get("f1")
    if not isinstance(b_f1, (int, float)) or isinstance(b_f1, bool):
        b_f1 = None

    f_f1 = norm.get("final_f1")
    if not isinstance(f_f1, (int, float)) or isinstance(f_f1, bool):
        f_f1 = fm.get("f1") if (fm and isinstance(fm.get("f1"), (int, float))) else None
    if not isinstance(f_f1, (int, float)) or isinstance(f_f1, bool):
        f_f1 = None

    b_raw = norm.get("baseline_raw_f1")
    if not isinstance(b_raw, (int, float)) or isinstance(b_raw, bool):
        b_raw = bm.get("raw_f1") if (bm and isinstance(bm.get("raw_f1"), (int, float))) else norm.get("raw_f1")
    if not isinstance(b_raw, (int, float)) or isinstance(b_raw, bool):
        b_raw = b_f1

    f_raw = norm.get("final_raw_f1")
    if not isinstance(f_raw, (int, float)) or isinstance(f_raw, bool):
        f_raw = fm.get("raw_f1") if (fm and isinstance(fm.get("raw_f1"), (int, float))) else f_f1
    if not isinstance(f_raw, (int, float)) or isinstance(f_raw, bool):
        f_raw = f_f1

    active_f1 = f_f1 if f_f1 is not None else b_f1
    active_raw = f_raw if f_raw is not None else b_raw

    c_raw_thresh = norm.get("pass_threshold_raw")
    if not isinstance(c_raw_thresh, (int, float)) or isinstance(c_raw_thresh, bool):
        c_raw_thresh = raw_threshold

    c_disp_thresh = norm.get("pass_threshold")
    if not isinstance(c_disp_thresh, (int, float)) or isinstance(c_disp_thresh, bool):
        c_disp_thresh = disp_threshold

    has_measured = (b_f1 is not None) or (f_f1 is not None)
    passed_bool: bool | None = None
    if has_measured:
        if fm and isinstance(fm.get("passed"), bool) and f_raw is not None:
            passed_bool = fm["passed"]
        elif active_raw is not None and c_raw_thresh is not None:
            passed_bool = float(active_raw) >= float(c_raw_thresh)
        elif isinstance(norm.get("passed"), bool):
            passed_bool = norm["passed"]
        elif active_f1 is not None and c_disp_thresh is not None:
            passed_bool = float(active_f1) >= float(c_disp_thresh)

    limitation = bool(
        norm.get("limitation") is True
        or (bm and bm.get("limitation") is True)
        or (fm and fm.get("limitation") is True)
    )

    active_m = fm or bm or norm
    ref_objs = norm.get("ref_objects")
    if not isinstance(ref_objs, (int, float)) or isinstance(ref_objs, bool):
        ref_objs = active_m.get("ref_objects") if isinstance(active_m.get("ref_objects"), (int, float)) else None

    norm["condition_id"] = cid
    norm["baseline_f1"] = b_f1
    norm["final_f1"] = f_f1
    norm["baseline_raw_f1"] = b_raw
    norm["final_raw_f1"] = f_raw
    norm["ref_objects"] = ref_objs
    norm["passed"] = passed_bool
    norm["limitation"] = limitation
    norm["pass_threshold"] = c_disp_thresh
    norm["pass_threshold_raw"] = c_raw_thresh
    # Block C 14:08 rule: compression pass/fail applies to every condition; limitation is informational
    if has_measured:
        norm["status"] = "PASS" if passed_bool else "FAIL"
    else:
        norm["status"] = "evaluation pending"
    return norm


def _build_crash_wall(scenario_id: str, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    image_id = _select_crash_wall_image_id(scenario_id, profile, state)
    splits = _load_splits_data(scenario_id)
    fp32_counts = splits.get("fp32_object_counts") if isinstance(splits.get("fp32_object_counts"), dict) else {}
    scen_man = _load_raw_scenario_manifest(scenario_id)
    hero_cfg = scen_man.get("hero_crash_wall") if isinstance(scen_man.get("hero_crash_wall"), dict) else {}
    hero_paths = hero_cfg.get("paths") if isinstance(hero_cfg.get("paths"), dict) else {}

    recent_list = []
    prog = state.get("progress")
    if isinstance(prog, dict) and isinstance(prog.get("recent"), list):
        recent_list = [r for r in prog["recent"] if isinstance(r, dict)]

    tiles: list[dict[str, Any]] = []
    if not image_id:
        return {"image_id": None, "file_name": None, "manifest_image_id": None, "tiles": tiles}

    stem = _norm_img_stem(image_id)
    orig_file = _find_split_b_original(scenario_id, image_id)
    if orig_file is None and isinstance(hero_paths.get("original"), str):
        cand_orig = _safe_resolve_under(Path.cwd(), hero_paths["original"])
        if cand_orig and cand_orig.is_file():
            orig_file = cand_orig
    orig_available = orig_file is not None
    file_name = orig_file.name if orig_file else _norm_img_filename(image_id)

    sample_entry = _resolve_manifest_entry_for_split_b(scenario_id, None, file_name, scen_man)
    manifest_image_id = (
        str(sample_entry.get("image_id"))
        if (sample_entry and sample_entry.get("image_id"))
        else (str(hero_cfg.get("image_id")) if hero_cfg.get("image_id") else file_name)
    )

    # Warm original Split-B preview metadata once so FP32 and INT8 box counts are real measured values
    orig_meta = _PREVIEW_META_CACHE.get((scenario_id, "original", stem, "baseline"))
    if orig_meta is None and orig_available:
        rendered = _render_overlay_bytes(scenario_id, "original", file_name, "baseline", state)
        if rendered is not None:
            orig_meta = rendered[1]

    orig_fp32_boxes = (
        orig_meta["fp32_boxes"]
        if orig_meta is not None
        else (fp32_counts.get(file_name) if isinstance(fp32_counts.get(file_name), int) else None)
    )
    orig_int8_boxes = orig_meta["int8_boxes"] if orig_meta is not None else None
    orig_missed = orig_meta["missed"] if orig_meta is not None else None

    b_ctrl = state.get("baseline_control")
    ctrl_f1 = b_ctrl.get("f1") if (isinstance(b_ctrl, dict) and isinstance(b_ctrl.get("f1"), (int, float))) else None
    ctrl_raw_f1 = (
        b_ctrl.get("raw_f1")
        if (isinstance(b_ctrl, dict) and isinstance(b_ctrl.get("raw_f1"), (int, float)))
        else ctrl_f1
    )
    sanity_bar = float(state.get("control_sanity_threshold") or 0.80)
    ctrl_passed = (
        (float(ctrl_raw_f1) >= sanity_bar and b_ctrl.get("passed", True) is not False)
        if (ctrl_raw_f1 is not None and isinstance(b_ctrl, dict))
        else None
    )

    tiles.append(
        {
            "id": "original",
            "name": "Original (Split B Control)",
            "is_original": True,
            "image_id": file_name,
            "manifest_image_id": manifest_image_id,
            "available": orig_available,
            "valid": True,
            "thumb_url": f"/preview/{scenario_id}/original/{file_name}?model=baseline" if orig_available else None,
            "fp32_boxes": orig_fp32_boxes,
            "int8_boxes": orig_int8_boxes,
            "missed": orig_missed,
            "f1": ctrl_f1,
            "raw_f1": ctrl_raw_f1,
            "passed": ctrl_passed,
            "generator_label": "Split B clean source",
            "status": ("PASS" if ctrl_passed else "BELOW SANITY") if ctrl_f1 is not None else "evaluation pending",
        }
    )

    conditions_state = state.get("conditions") if isinstance(state.get("conditions"), dict) else {}
    raw_thresh = _extract_raw_pass_threshold(state, scenario_id)

    for cond_cfg in profile.get("conditions") or []:
        cid = str(cond_cfg.get("id") or "")
        if not cid:
            continue
        cname = str(cond_cfg.get("name") or cid)
        cstate = conditions_state.get(cid) if isinstance(conditions_state.get(cid), dict) else {}
        entry = _resolve_manifest_entry_for_split_b(scenario_id, cid, file_name, scen_man)

        matched_recent = next(
            (
                r
                for r in recent_list
                if str(r.get("condition") or "") == cid
                and _norm_img_stem(str(r.get("image_id") or "")) in (stem, _norm_img_stem(manifest_image_id))
            ),
            None,
        )

        cond_file = _find_condition_image(scenario_id, cid, file_name)
        if cond_file is None and isinstance(hero_paths.get(cid), str):
            cand_cond = _safe_resolve_under(Path.cwd(), hero_paths[cid])
            if cand_cond and cand_cond.is_file():
                cond_file = cand_cond

        entry_valid = True
        if isinstance(entry, dict):
            if (
                entry.get("valid") is False
                or entry.get("allow_in_pipeline") is False
                or str(entry.get("validity_status") or "valid") == "invalid"
            ):
                entry_valid = False

        showcase_ov = _find_showcase_overlay(scenario_id, cid, file_name, "baseline")

        meta = _PREVIEW_META_CACHE.get((scenario_id, cid, stem, "baseline"))
        if meta is None and cond_file is not None and entry_valid:
            rendered_cond = _render_overlay_bytes(scenario_id, cid, file_name, "baseline", state)
            if rendered_cond is not None:
                meta = rendered_cond[1]

        thumb_url: str | None = None
        if not entry_valid:
            thumb_url = None
        elif matched_recent and isinstance(matched_recent.get("thumb_url"), str) and matched_recent["thumb_url"]:
            thumb_url = matched_recent["thumb_url"]
        elif showcase_ov is not None:
            rel = showcase_ov.resolve().relative_to(Path.cwd().resolve())
            thumb_url = f"/artifacts/{rel.as_posix()}"
        elif cond_file is not None:
            thumb_url = f"/preview/{scenario_id}/{cid}/{file_name}?model=baseline"

        fp32_b = (
            matched_recent.get("fp32_boxes")
            if matched_recent and matched_recent.get("fp32_boxes") is not None
            else (
                meta["fp32_boxes"]
                if meta is not None
                else (entry.get("persisted_count") if isinstance(entry, dict) else None)
            )
        )
        int8_b = (
            matched_recent.get("int8_boxes")
            if matched_recent and matched_recent.get("int8_boxes") is not None
            else (meta["int8_boxes"] if meta is not None else None)
        )
        missed_b = (
            matched_recent.get("missed")
            if matched_recent and matched_recent.get("missed") is not None
            else (meta["missed"] if meta is not None else None)
        )

        base_f1 = cstate.get("baseline_f1")
        fin_f1 = cstate.get("final_f1")
        passed_flag = cstate.get("passed")
        has_metrics = (base_f1 is not None) or (fin_f1 is not None)
        tile_status = ("PASS" if passed_flag else "FAIL") if has_metrics else "evaluation pending"

        tiles.append(
            {
                "id": cid,
                "name": cname,
                "is_original": False,
                "image_id": file_name,
                "manifest_image_id": str(entry.get("image_id")) if (entry and entry.get("image_id")) else manifest_image_id,
                "available": (thumb_url is not None) and entry_valid,
                "valid": entry_valid,
                "validity_status": entry.get("validity_status") if isinstance(entry, dict) else "valid",
                "generator_label": (
                    entry.get("generator_label") or entry.get("generator")
                    if isinstance(entry, dict)
                    else cstate.get("generator_summary")
                ),
                "audit_status": entry.get("audit_status") if isinstance(entry, dict) else None,
                "fallback_reason": entry.get("fallback_reason") if isinstance(entry, dict) else None,
                "retention": entry.get("retention") if isinstance(entry, dict) else None,
                "thumb_url": thumb_url,
                "fp32_boxes": fp32_b,
                "int8_boxes": int8_b,
                "missed": missed_b,
                "baseline_f1": base_f1,
                "final_f1": fin_f1,
                "passed": passed_flag,
                "pass_threshold_raw": raw_thresh,
                "status": tile_status,
            }
        )

    return {
        "image_id": file_name,
        "file_name": file_name,
        "manifest_image_id": manifest_image_id,
        "tiles": tiles,
    }


def _enrich_state_for_response(raw_state: dict[str, Any], scenario_override: str | None = None) -> dict[str, Any]:
    state = copy.deepcopy(raw_state) if isinstance(raw_state, dict) else {}
    scenario_id = str(state.get("scenario_id") or scenario_override or "robot_vacuum")
    state["scenario_id"] = scenario_id

    raw_threshold = _extract_raw_pass_threshold(state, scenario_id)
    state["pass_threshold_raw"] = raw_threshold
    if state.get("pass_threshold") is None and raw_threshold is not None:
        state["pass_threshold"] = round(float(raw_threshold), 4)
    if state.get("control_sanity_threshold") is None:
        state["control_sanity_threshold"] = 0.80

    # Sanitize baseline_control and final_control so missing keys in partial dicts never crash Jinja
    for ctrl_key in ("baseline_control", "final_control"):
        c_obj = state.get(ctrl_key)
        if isinstance(c_obj, dict):
            f1_v = c_obj.get("f1")
            raw_v = c_obj.get("raw_f1")
            if not isinstance(f1_v, (int, float)) or isinstance(f1_v, bool):
                f1_v = round(float(raw_v), 4) if (isinstance(raw_v, (int, float)) and not isinstance(raw_v, bool)) else None
            if not isinstance(raw_v, (int, float)) or isinstance(raw_v, bool):
                raw_v = f1_v
            c_obj["f1"] = f1_v
            c_obj["raw_f1"] = raw_v
            c_obj.setdefault("ref_objects", None)
            c_obj.setdefault("n_images", state.get("n_images", 40))
            if f1_v is not None:
                sanity = float(state.get("control_sanity_threshold") or 0.80)
                c_obj["passed"] = bool(float(raw_v) >= sanity and c_obj.get("passed", True) is not False)
            else:
                c_obj["passed"] = None

    # Load measured baseline condition matrix on default/showcase landing when state["conditions"] is empty
    conds_in_state = state.get("conditions")
    if (not isinstance(conds_in_state, dict) or len(conds_in_state) == 0) and state.get("run_id") in (
        None,
        "showcase_default",
    ):
        cb_path = _safe_resolve_under(Path("showcase"), scenario_id, "conditions_baseline.json")
        if cb_path and cb_path.is_file():
            try:
                cb_data = json.loads(cb_path.read_text())
                if isinstance(cb_data, dict) and isinstance(cb_data.get("conditions"), dict):
                    conds_in_state = copy.deepcopy(cb_data["conditions"])
            except Exception:
                conds_in_state = {}

    normalized_conditions: dict[str, dict[str, Any]] = {}
    if isinstance(conds_in_state, dict):
        for cid, cval in conds_in_state.items():
            if isinstance(cval, dict):
                normalized_conditions[str(cid)] = _normalize_condition_entry(
                    str(cid),
                    cval,
                    raw_threshold,
                    state.get("pass_threshold"),
                )
    state["conditions"] = normalized_conditions

    profile = _load_scenario_profile(scenario_id)
    state["scenario_meta"] = {
        "id": profile.get("id", scenario_id),
        "name": profile.get("name", scenario_id),
        "description": profile.get("description", ""),
        "calibration_rule": profile.get("calibration_rule", ""),
        "status": profile.get("status", "not yet run"),
        "class_names": _class_id_to_name_map(profile),
        "conditions": profile.get("conditions", []),
    }
    state["manifest_summary"] = _load_manifest_summary(scenario_id, profile)
    if "crash_wall" not in state or not isinstance(state.get("crash_wall"), dict):
        state["crash_wall"] = _build_crash_wall(scenario_id, profile, state)

    repaired_path = _resolve_model_path_for_preview(state, "repaired")
    state["has_repaired_model"] = repaired_path is not None
    return state


def _safe_json_for_script_tag(data: Any) -> str:
    """Serialize data to JSON with <, >, & escaped as \\uXXXX so </script> can never break HTML parsing."""
    raw = json.dumps(data)
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _run_loop_worker(run_id: str, req: RunRequest) -> None:
    def on_update(patch_or_state: dict[str, Any]) -> None:
        if isinstance(patch_or_state, dict):
            STORE.update_run(run_id, patch_or_state)

    try:
        from odct.loop import run as loop_run

        try:
            result = loop_run(
                scenario_id=req.scenario_id,
                condition_ids=req.conditions,
                n_images=req.n_images,
                mode=req.mode,
                on_update=on_update,
                run_id=run_id,
            )
        except TypeError:
            result = loop_run(
                scenario_id=req.scenario_id,
                condition_ids=req.conditions,
                n_images=req.n_images,
                mode=req.mode,
                on_update=on_update,
            )
        if isinstance(result, dict):
            final_patch = copy.deepcopy(result)
            cur_phase = str(final_patch.get("phase") or "")
            if cur_phase not in ("done", "needs_human", "error"):
                final_patch["phase"] = "needs_human" if final_patch.get("needs_human") else "done"
            STORE.update_run(run_id, final_patch, clear_active=True)
        else:
            STORE.update_run(run_id, {"phase": "done"}, clear_active=True)
    except Exception as exc:
        STORE.update_run(
            run_id,
            {
                "phase": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
            clear_active=True,
        )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, scenario: str = "robot_vacuum") -> Any:
    profile = _load_scenario_profile(scenario)
    fridge_profile = _load_scenario_profile("fridge_interior")
    fridge_showcase_exists = (Path("showcase") / "fridge_interior" / "state.json").exists()
    active_id = STORE.get_active_run_id()
    raw_state = STORE.get_run(active_id, scenario_id=scenario) if active_id else load_default_state(scenario)
    state = _enrich_state_for_response(raw_state, scenario_override=scenario)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "scenario": profile,
            "fridge_available": fridge_showcase_exists,
            "fridge_status": fridge_profile.get("status", "not yet run"),
            "state": state,
            "state_json": _safe_json_for_script_tag(state),
        },
    )


@app.get("/state")
def get_state(run_id: str | None = None, scenario: str = "robot_vacuum") -> JSONResponse:
    if run_id:
        raw = STORE.get_run(run_id, scenario_id=scenario)
        return JSONResponse(_enrich_state_for_response(raw, scenario_override=scenario))
    active_id = STORE.get_active_run_id()
    if active_id:
        raw = STORE.get_run(active_id, scenario_id=scenario)
        return JSONResponse(_enrich_state_for_response(raw, scenario_override=scenario))
    raw = load_default_state(scenario)
    return JSONResponse(_enrich_state_for_response(raw, scenario_override=scenario))


@app.post("/run")
def start_run(req: RunRequest) -> JSONResponse:
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    default_state = load_default_state(req.scenario_id)
    pass_threshold = compute_pass_threshold(
        default_state.get("baseline_control"),
        default_state.get("pass_threshold"),
    )
    initial_state: dict[str, Any] = {
        "run_id": run_id,
        "scenario_id": req.scenario_id,
        "phase": "evaluating",
        "mode": req.mode,
        "n_images": req.n_images,
        "baseline_control": default_state.get("baseline_control", "not yet run"),
        "final_control": "not yet run",
        "baseline_int8": default_state.get("baseline_int8", "not yet run"),
        "fp32": default_state.get("fp32", "not yet run"),
        "budget_reference": default_state.get("budget_reference", "not yet run"),
        "pass_threshold": pass_threshold,
        "condition_pass_threshold": pass_threshold,
        "condition_pass_threshold_raw": default_state.get("condition_pass_threshold_raw"),
        "control_sanity_threshold": float(default_state.get("control_sanity_threshold") or 0.80),
        "conditions": {},
        "iterations": [],
        "argon_logs": [],
        "progress": None,
        "needs_human": False,
        "human_decision": None,
        "error": None,
    }

    if not STORE.try_reserve_run(run_id, initial_state):
        raise HTTPException(status_code=409, detail="A live run is already in progress.")

    try:
        worker = threading.Thread(
            target=_run_loop_worker,
            args=(run_id, req),
            daemon=True,
            name=f"odct_{run_id}",
        )
        worker.start()
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        STORE.update_run(
            run_id,
            {"phase": "error", "error": err_msg},
            clear_active=True,
        )
        raise HTTPException(status_code=500, detail=f"Failed to start worker: {err_msg}") from exc

    return JSONResponse({"run_id": run_id, "phase": initial_state["phase"]})


@app.post("/human_decision")
def record_human_decision(req: HumanDecisionRequest) -> JSONResponse:
    if req.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="Decision must be 'approved' or 'rejected'.")
    target_id = req.run_id or STORE.get_active_run_id()
    if not target_id:
        raise HTTPException(status_code=404, detail="No run found to record decision.")
    updated = STORE.update_run(target_id, {"human_decision": req.decision})
    return JSONResponse({"run_id": updated["run_id"], "human_decision": updated.get("human_decision")})


@app.get("/preview/{scenario_id}/{target}/{image_id}")
def get_preview_overlay(
    scenario_id: str,
    target: str,
    image_id: str,
    model: str = "baseline",
    run_id: str | None = None,
) -> Response:
    if model not in ("baseline", "fp32", "repaired"):
        raise HTTPException(status_code=400, detail="Unsupported model preview mode")

    state = STORE.get_run(run_id, scenario_id=scenario_id) if (run_id or STORE.get_active_run_id()) else load_default_state(scenario_id)

    showcase_ov = _find_showcase_overlay(scenario_id, target, image_id, model)
    if showcase_ov is not None:
        return FileResponse(str(showcase_ov))

    rendered = _render_overlay_bytes(scenario_id, target, image_id, model, state)
    if rendered is not None:
        return Response(content=rendered[0], media_type="image/jpeg")

    # Safe fallback to raw image only if the image itself exists and model is baseline/fp32
    if model in ("baseline", "fp32"):
        raw_path = (
            _find_split_b_original(scenario_id, image_id)
            if target == "original"
            else _find_condition_image(scenario_id, target, image_id)
        )
        if raw_path is not None and raw_path.is_file():
            return FileResponse(str(raw_path))

    raise HTTPException(status_code=404, detail="Preview image not available")


@app.get("/replay_events")
def get_replay_events(scenario: str = "robot_vacuum") -> JSONResponse:
    events_path = _safe_resolve_under(Path("showcase"), scenario, "events.jsonl")
    if not events_path or not events_path.is_file():
        return JSONResponse({"scenario": scenario, "available": False, "events": []})
    events: list[dict[str, Any]] = []
    try:
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
    except Exception:
        pass
    return JSONResponse({"scenario": scenario, "available": len(events) > 0, "events": events})


@app.get("/artifacts/{file_path:path}")
def get_artifact(file_path: str) -> FileResponse:
    if not file_path or file_path.strip() == "":
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        candidate = (Path.cwd().resolve() / file_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc

    if not _is_within_allowed_roots(candidate):
        raise HTTPException(status_code=403, detail="Access denied")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(str(candidate))
