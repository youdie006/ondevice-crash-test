"""Gemini 4 Argon multimodal diagnosis via the Interactions API with evidence and claim verification."""

from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "pattern": {"type": "string"},
                    "cause": {
                        "type": "string",
                        "enum": ["compression", "model_limitation", "edit_artifact"],
                    },
                    "evidence_images": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition": {"type": "string"},
                                "metric": {"type": "string"},
                                "comparator": {
                                    "type": "string",
                                    "enum": ["<", "<=", ">", ">=", "==", "!="],
                                },
                                "value": {"type": "number"},
                            },
                            "required": ["condition", "metric", "comparator", "value"],
                        },
                    },
                },
                "required": ["condition", "pattern", "cause", "evidence_images", "claims"],
            },
        },
        "action": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": (
                        "Repair knob selection. Active knobs today: R1 (recalibrate with Split-A lab_calib_25 + Split-A condition images), "
                        "R2 (calibration method: percentile 99.999 or entropy), R3 (per-channel weight quantization), "
                        "or combinations like R1+R2, R2+R3, R1+R2+R3, or 'none'. "
                        "R4 (letterbox) and R5 (node exclusion) are kept in schema for completeness but will be REJECTED by the gate today."
                    ),
                },
                "params": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "Calibration method for R2: 'percentile' (CalibPercentile=99.999) or 'entropy'.",
                        },
                        "per_channel": {
                            "type": "boolean",
                            "description": "Enable per-channel weight quantization for R3.",
                        },
                        "duplication_weight": {
                            "type": "integer",
                            "description": "Duplication weight 1..3 for R1 condition images. Allowed ONLY with percentile or entropy; forbidden under MinMax.",
                        },
                    },
                },
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target condition IDs for R1 recalibration (uses Split-A images only).",
                },
            },
            "required": ["type", "params", "targets"],
        },
        "expected_effect": {"type": "string"},
        "stop": {"type": "boolean"},
        "needs_human": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "diagnosis",
        "action",
        "expected_effect",
        "stop",
        "needs_human",
        "rationale",
    ],
}

MAX_IMAGES_PER_CALL = 400


class ArgonCallError(RuntimeError):
    """Raised when the Gemini 4 Argon Interactions API call fails, times out, or returns invalid output."""


def _encode_image_part(path: str | Path) -> dict[str, str] | None:
    p = Path(path)
    if not p.is_file():
        return None
    raw = p.read_bytes()
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return {
        "type": "image",
        "data": base64.b64encode(raw).decode("ascii"),
        "mime_type": mime,
    }


def build_diagnostic_image_inputs(
    image_pairs: Sequence[dict[str, Any]],
    max_images: int = MAX_IMAGES_PER_CALL,
) -> tuple[list[dict[str, Any]], int, set[str]]:
    """Convert image pair descriptors into Interactions API input items, capped at max_images (<= 400)."""
    cap = min(int(max_images), MAX_IMAGES_PER_CALL)
    items: list[dict[str, Any]] = []
    images_sent = 0
    sent_ids: set[str] = set()

    for pair in image_pairs:
        if images_sent >= cap:
            break
        img_id = str(pair.get("image_id") or "")
        cond = str(pair.get("condition") or "unknown")
        stage = str(pair.get("stage") or "current")

        # Each pair can have orig_path and/or cond_path (overlay_path)
        orig_p = pair.get("orig_path")
        cond_p = pair.get("overlay_path") or pair.get("cond_path") or pair.get("image_path")

        if orig_p and images_sent + 2 <= cap:
            part_orig = _encode_image_part(orig_p)
            part_cond = _encode_image_part(cond_p) if cond_p else None
            if part_orig and part_cond:
                items.append(
                    {
                        "type": "text",
                        "text": f"[image_id={img_id} condition={cond} stage={stage} view=original_fp32]",
                    }
                )
                items.append(part_orig)
                items.append(
                    {
                        "type": "text",
                        "text": f"[image_id={img_id} condition={cond} stage={stage} view=condition_fp32_green_int8_orange]",
                    }
                )
                items.append(part_cond)
                images_sent += 2
                if img_id:
                    sent_ids.add(img_id)
                continue

        if cond_p and images_sent + 1 <= cap:
            part_cond = _encode_image_part(cond_p)
            if part_cond:
                items.append(
                    {
                        "type": "text",
                        "text": f"[image_id={img_id} condition={cond} stage={stage} view=condition_fp32_green_int8_orange]",
                    }
                )
                items.append(part_cond)
                images_sent += 1
                if img_id:
                    sent_ids.add(img_id)

    return items, images_sent, sent_ids


def select_image_pairs_for_diagnosis(
    failing_pairs: Sequence[dict[str, Any]],
    previous_pairs: Sequence[dict[str, Any]],
    passing_pairs_by_cond: dict[str, Sequence[dict[str, Any]]],
    control_pairs: Sequence[dict[str, Any]],
    mode: str = "live",
    max_images: int = MAX_IMAGES_PER_CALL,
) -> list[dict[str, Any]]:
    """Select diagnostic image pairs per Section 10b:
    - Live mode (20 images): all evaluated pairs (failing + previous + passing + control) up to 400 images.
    - Showcase mode: all failing pairs + previous iteration pairs + up to 5 pairs per passing condition + up to 10 control pairs, capped at 400 images.
    """
    selected: list[dict[str, Any]] = []
    selected.extend(failing_pairs)
    selected.extend(previous_pairs)

    if mode == "live":
        for cond_id in sorted(passing_pairs_by_cond.keys()):
            selected.extend(passing_pairs_by_cond[cond_id])
        selected.extend(control_pairs)
    else:
        for cond_id in sorted(passing_pairs_by_cond.keys()):
            selected.extend(list(passing_pairs_by_cond[cond_id])[:5])
        selected.extend(list(control_pairs)[:10])

    # Cap by max_images (each pair is up to 2 images)
    max_pairs = max(1, max_images // 2)
    return selected[:max_pairs]


def _resolve_metric_value(cond_metrics: dict[str, Any], metric_name: str) -> float | None:
    m = metric_name.strip().lower()
    if m in ("f1", "compression_f1"):
        val = cond_metrics.get("f1")
    elif m == "raw_f1":
        val = cond_metrics.get("raw_f1", cond_metrics.get("f1"))
    elif m in ("recall", " precision".strip(), "retention", "ref_objects", "orig_objects", "tp", "fp", "fn", "valid_images", "collapse_images"):
        val = cond_metrics.get(m)
    elif m in ("recall_small", "per_size.small", "small", "size_small"):
        val = (cond_metrics.get("per_size") or {}).get("small")
    elif m in ("recall_medium", "per_size.medium", "medium", "size_medium"):
        val = (cond_metrics.get("per_size") or {}).get("medium")
    elif m in ("recall_large", "per_size.large", "large", "size_large"):
        val = (cond_metrics.get("per_size") or {}).get("large")
    elif m.startswith("per_class.") or m.startswith("class_"):
        key = m.split(".", 1)[-1].replace("class_", "").replace("_f1", "")
        val = (cond_metrics.get("per_class") or {}).get(key)
    else:
        val = cond_metrics.get(metric_name)

    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    fval = float(val)
    return fval if math.isfinite(fval) else None


def _compare_values(measured: float, comparator: str, target: float) -> bool:
    op = comparator.strip()
    if op == "<":
        return measured < target
    if op == "<=":
        return measured <= target
    if op == ">":
        return measured > target
    if op == ">=":
        return measured >= target
    if op in ("==", "="):
        return abs(measured - target) <= 1e-3
    if op == "!=":
        return abs(measured - target) > 1e-3
    return False


def verify_diagnosis(
    action_json: dict[str, Any],
    metrics_by_condition: dict[str, dict[str, Any]],
    valid_image_ids: set[str],
    canonical_id_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify evidence_images and numerical claims in Argon's diagnosis output.
    Mutates and returns verification summary with verified/unverified annotations.
    """
    diagnoses = action_json.get("diagnosis")
    if not isinstance(diagnoses, list) or len(diagnoses) == 0:
        raise ArgonCallError("Argon response missing required non-empty 'diagnosis' array")

    id_lookup: dict[str, str] = {img_id: img_id for img_id in valid_image_ids}
    for img_id in valid_image_ids:
        stem = Path(img_id).stem
        id_lookup.setdefault(stem, img_id)
        if stem.startswith("b_"):
            id_lookup.setdefault(stem[2:], img_id)
        else:
            id_lookup.setdefault(f"b_{stem}", img_id)
    if canonical_id_map:
        for k, v in canonical_id_map.items():
            id_lookup[k] = v

    all_evidence_ok = True
    total_claims = 0
    verified_claims = 0
    unverified_claims = 0
    invalid_evidence: list[str] = []

    for diag in diagnoses:
        if not isinstance(diag, dict):
            raise ArgonCallError("Each diagnosis entry must be a JSON object")
        if "evidence_images" not in diag or not isinstance(diag["evidence_images"], list) or len(diag["evidence_images"]) == 0:
            raise ArgonCallError("Diagnosis entry missing required non-empty 'evidence_images' list")
        if "claims" not in diag or not isinstance(diag["claims"], list) or len(diag["claims"]) == 0:
            raise ArgonCallError("Diagnosis entry missing required non-empty 'claims' list")

        canonical_ev: list[str] = []
        missing_ev: list[str] = []
        for raw_ev in diag["evidence_images"]:
            ev_str = str(raw_ev).strip()
            if ev_str in id_lookup:
                canonical_ev.append(id_lookup[ev_str])
            else:
                all_evidence_ok = False
                missing_ev.append(ev_str)
                invalid_evidence.append(ev_str)

        diag["evidence_images"] = canonical_ev if canonical_ev else [str(x) for x in diag["evidence_images"]]
        diag["evidence_verified"] = len(missing_ev) == 0
        if missing_ev:
            diag["invalid_evidence_images"] = missing_ev

        diag_verified_count = 0
        diag_claims = diag["claims"]
        for claim in diag_claims:
            total_claims += 1
            if not isinstance(claim, dict):
                claim_obj = {"raw": str(claim), "verified": False, "status": "unverified"}
                unverified_claims += 1
                continue
            c_cond = str(claim.get("condition") or diag.get("condition") or "")
            c_metric = str(claim.get("metric") or "")
            c_comp = str(claim.get("comparator") or "")
            c_val = claim.get("value")

            cond_m = metrics_by_condition.get(c_cond)
            if cond_m is None or isinstance(c_val, bool) or not isinstance(c_val, (int, float)):
                claim["verified"] = False
                claim["status"] = "unverified"
                claim["measured"] = None
                claim["note"] = "unverified (condition or target value unavailable)"
                unverified_claims += 1
                continue

            measured_val = _resolve_metric_value(cond_m, c_metric)
            if measured_val is None:
                claim["verified"] = False
                claim["status"] = "unverified"
                claim["measured"] = None
                claim["note"] = f"unverified (metric '{c_metric}' not numerically checkable)"
                unverified_claims += 1
                continue

            ok = _compare_values(measured_val, c_comp, float(c_val))
            claim["measured"] = round(float(measured_val), 4)
            claim["verified"] = bool(ok)
            claim["status"] = "verified" if ok else "unverified"
            if ok:
                verified_claims += 1
                diag_verified_count += 1
            else:
                claim["note"] = (
                    f"unverified (measured {measured_val:.4f} does not satisfy {c_comp} {float(c_val):.4f})"
                )
                unverified_claims += 1

        n_c = max(1, len(diag_claims))
        ev_factor = 1.0 if diag["evidence_verified"] else 0.0
        diag["confidence"] = round(ev_factor * (float(diag_verified_count) / float(n_c)), 4)

    return {
        "all_evidence_verified": all_evidence_ok,
        "invalid_evidence_images": invalid_evidence,
        "total_claims": total_claims,
        "verified_claims": verified_claims,
        "unverified_claims": unverified_claims,
    }


def _build_diagnosis_prompt(
    metrics: dict[str, Any],
    history: Sequence[dict[str, Any]],
    valid_image_ids: Sequence[str],
    budget_info: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
    current_config: dict[str, Any] | None = None,
) -> str:
    payload = {
        "task": (
            "Diagnose why the INT8 QDQ quantized detector degrades relative to its FP32 parent on Split B "
            "and propose a single cumulative re-quantization repair action within the compression budget."
        ),
        "strict_rules": {
            "evaluation_split": "B (40 hold-out images; NEVER used for calibration)",
            "calibration_split": "A (lab_calib_25 + Split-A condition images ONLY for R1)",
            "op_types_to_quantize": ["Conv"],
            "allowed_repairs": {
                "R1": "Recalibrate using Split-A 25 lab-like images + Split-A condition images of targeted condition(s). Optional duplication_weight 1..3 is allowed ONLY when method is percentile or entropy (forbidden under MinMax).",
                "R2": "Change calibration method to 'percentile' (CalibPercentile=99.999) or 'entropy'.",
                "R3": "Enable per_channel=True weight quantization.",
                "combinations": "You may combine knobs (e.g. type='R1+R2', 'R2+R3', 'R1+R2+R3') by specifying params={'method': 'percentile', 'per_channel': True, 'duplication_weight': 1} and targets=['night'].",
                "forbidden_in_active_scope": "R4 (letterbox) and R5 (node exclusion) are disabled today and will be REJECTED by the gate.",
            },
            "compression_budget_vs_baseline_int8": {
                "size_ratio_max": 1.10,
                "median_latency_ratio_max": 1.10,
                "quantized_ops_ratio_min": 0.90,
                "current_budget_reference": budget_info or {},
            },
            "zero_regression_gate": {
                "max_control_f1_drop_vs_baseline": 0.02,
                "condition_pass_threshold": (thresholds or {}).get("condition_pass_threshold"),
                "condition_pass_threshold_raw": (thresholds or {}).get("condition_pass_threshold_raw"),
                "control_sanity_threshold": 0.80,
                "rule": "No previously passing non-limitation condition may drop below the fixed condition_pass_threshold.",
            },
            "citation_requirements": (
                "Every diagnosis entry MUST cite real image IDs from available_evidence_image_ids in 'evidence_images' "
                "and include machine-checkable numerical 'claims' (e.g. metric='f1', 'recall', 'precision', 'retention', 'recall_small', 'recall_medium', 'recall_large') "
                "matching the provided Split-B metrics."
            ),
        },
        "current_quant_config": current_config or {},
        "measured_split_b_metrics": metrics,
        "iteration_history": list(history),
        "available_evidence_image_ids": list(valid_image_ids),
    }
    return chr(10).join(
        [
            "You are diagnosing an INT8 on-device object detector crash test.",
            "Return ONLY valid JSON matching the required schema.",
            json.dumps(payload, indent=2),
        ]
    )


def _extract_interaction_text(r: Any) -> str | None:
    raw_text = getattr(r, "output_text", None)
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text
    outputs = getattr(r, "outputs", None)
    if isinstance(outputs, (list, tuple)):
        text_parts: list[str] = []
        for item in outputs:
            t = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
            if isinstance(t, str) and t.strip():
                text_parts.append(t)
        if text_parts:
            return chr(10).join(text_parts)
        return raw_text if isinstance(raw_text, str) else None

def diagnose(
    metrics: dict[str, Any],
    history: Sequence[dict[str, Any]],
    image_pairs: Sequence[dict[str, Any]],
    previous_id: str | None = None,
    *,
    budget_info: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
    current_config: dict[str, Any] | None = None,
    valid_image_ids: Sequence[str] | None = None,
    timeout_s: float = 120.0,
    on_status: Callable[[str, int, dict[str, Any] | None, float], None] | None = None,
    client: Any = None,
) -> tuple[dict[str, Any], dict[str, int], str]:
    """Call Gemini 4 Argon via the Interactions API (API_NOTES.md Section 3).
    Returns (action_json, usage_dict, interaction_id).
    """
    image_items, images_sent, sent_ids = build_diagnostic_image_inputs(
        image_pairs, max_images=MAX_IMAGES_PER_CALL
    )
    all_valid_ids = list(valid_image_ids) if valid_image_ids is not None else sorted(sent_ids)
    for sid in sent_ids:
        if sid not in all_valid_ids:
            all_valid_ids.append(sid)

    prompt_text = _build_diagnosis_prompt(
        metrics=metrics,
        history=history,
        valid_image_ids=all_valid_ids,
        budget_info=budget_info,
        thresholds=thresholds,
        current_config=current_config,
    )

    input_payload: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    input_payload.extend(image_items)

    t0 = time.perf_counter()
    if on_status is not None:
        on_status("sending", images_sent, None, 0.0)

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ArgonCallError("GEMINI_API_KEY is not set in the environment")
        from google import genai

        client = genai.Client(api_key=api_key)

    if on_status is not None:
        on_status("thinking", images_sent, None, round(time.perf_counter() - t0, 3))

    try:
        r = client.interactions.create(
            model="gemini-4-argon",
            input=input_payload,
            generation_config={"thinking_level": "high", "max_output_tokens": 4096},
            response_format=ACTION_SCHEMA,
            previous_interaction_id=previous_id,
            timeout=timeout_s,
        )
    except TypeError:
        # In case a test double or SDK signature does not accept timeout kwarg directly
        r = client.interactions.create(
            model="gemini-4-argon",
            input=input_payload,
            generation_config={"thinking_level": "high", "max_output_tokens": 4096},
            response_format=ACTION_SCHEMA,
            previous_interaction_id=previous_id,
        )
    except Exception as exc:
        raise ArgonCallError(f"Argon Interactions API error: {exc}") from exc

    elapsed = round(time.perf_counter() - t0, 3)
    raw_text = _extract_interaction_text(r)
    interaction_id = getattr(r, "id", None)
    usage_obj = getattr(r, "usage", None)

    if not interaction_id or not isinstance(interaction_id, str):
        raise ArgonCallError("Argon response did not contain a valid interaction id (r.id)")
    if not raw_text or not isinstance(raw_text, str):
        raise ArgonCallError("Argon response did not contain output_text")

    usage_dict = {
        "total_input_tokens": int(getattr(usage_obj, "total_input_tokens", 0) or 0),
        "total_output_tokens": int(getattr(usage_obj, "total_output_tokens", 0) or 0),
        "total_thought_tokens": int(getattr(usage_obj, "total_thought_tokens", 0) or 0),
        "total_cached_tokens": int(getattr(usage_obj, "total_cached_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }

    if on_status is not None:
        on_status("received", images_sent, usage_dict, elapsed)

    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = chr(10).join(lines).strip()

    try:
        action_json = json.loads(cleaned_text)
    except Exception as exc:
        raise ArgonCallError(f"Argon output_text is not valid JSON: {exc}") from exc

    if not isinstance(action_json, dict):
        raise ArgonCallError("Argon JSON output must be an object")

    return action_json, usage_dict, interaction_id

