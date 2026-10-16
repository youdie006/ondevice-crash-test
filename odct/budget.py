"""Compression budget checks relative to baseline INT8."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Measured:
    size_bytes: int
    median_ms: float
    quantized_ops: int


def _is_valid_positive_number(val: int | float) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val)) and float(val) > 0.0


def _validate_measured(m: Measured, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(m, Measured):
        return [f"{label} is not a Measured instance"]
    if not _is_valid_positive_number(m.size_bytes):
        errors.append(f"invalid {label}.size_bytes={m.size_bytes} (must be finite and > 0)")
    if not _is_valid_positive_number(m.median_ms):
        errors.append(f"invalid {label}.median_ms={m.median_ms} (must be finite and > 0)")
    if not _is_valid_positive_number(m.quantized_ops):
        errors.append(f"invalid {label}.quantized_ops={m.quantized_ops} (must be finite and > 0)")
    return errors


def check(
    candidate: Measured,
    baseline: Measured,
    fp32: Measured | None = None,
) -> tuple[bool, dict]:
    validation_errors = _validate_measured(baseline, "baseline") + _validate_measured(candidate, "candidate")
    if validation_errors:
        return False, {
            "ok": False,
            "size_ratio_vs_baseline": "N/A",
            "ms_ratio_vs_baseline": "N/A",
            "ops_ratio_vs_baseline": "N/A",
            "size_ok": False,
            "ms_ok": False,
            "ops_ok": False,
            "reasons": validation_errors,
        }

    size_ratio = float(candidate.size_bytes) / float(baseline.size_bytes)
    ms_ratio = float(candidate.median_ms) / float(baseline.median_ms)
    ops_ratio = float(candidate.quantized_ops) / float(baseline.quantized_ops)

    size_ok = bool(math.isfinite(size_ratio) and size_ratio <= 1.10)
    ms_ok = bool(math.isfinite(ms_ratio) and ms_ratio <= 1.10)
    ops_ok = bool(math.isfinite(ops_ratio) and ops_ratio >= 0.90)
    ok = bool(size_ok and ms_ok and ops_ok)

    reasons: list[str] = []
    if not size_ok:
        reasons.append(f"size {size_ratio:.2f}x baseline (> 1.10x)")
    if not ms_ok:
        reasons.append(f"latency {ms_ratio:.2f}x baseline (> 1.10x)")
    if not ops_ok:
        reasons.append(f"quantized_ops {ops_ratio:.2f}x baseline (< 0.90x)")

    details: dict = {
        "ok": ok,
        "size_ratio_vs_baseline": round(size_ratio, 4),
        "ms_ratio_vs_baseline": round(ms_ratio, 4),
        "ops_ratio_vs_baseline": round(ops_ratio, 4),
        "size_ok": size_ok,
        "ms_ok": ms_ok,
        "ops_ok": ops_ok,
        "reasons": reasons,
    }
    if fp32 is not None:
        fp32_size_ok = _is_valid_positive_number(fp32.size_bytes)
        fp32_ms_ok = _is_valid_positive_number(fp32.median_ms)
        details["size_ratio_vs_fp32"] = (
            round(float(candidate.size_bytes) / float(fp32.size_bytes), 4) if fp32_size_ok else "N/A"
        )
        details["ms_ratio_vs_fp32"] = (
            round(float(candidate.median_ms) / float(fp32.median_ms), 4) if fp32_ms_ok else "N/A"
        )
    return ok, details
