"""Compression budget checks relative to baseline INT8."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Measured:
    size_bytes: int
    median_ms: float
    quantized_ops: int


def check(
    candidate: Measured,
    baseline: Measured,
    fp32: Measured | None = None,
) -> tuple[bool, dict]:
    size_ratio = float(candidate.size_bytes) / float(baseline.size_bytes) if baseline.size_bytes > 0 else 1.0
    ms_ratio = float(candidate.median_ms) / float(baseline.median_ms) if baseline.median_ms > 0 else 1.0
    ops_ratio = float(candidate.quantized_ops) / float(baseline.quantized_ops) if baseline.quantized_ops > 0 else 1.0

    size_ok = size_ratio <= 1.10
    ms_ok = ms_ratio <= 1.10
    ops_ok = ops_ratio >= 0.90
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
        details["size_ratio_vs_fp32"] = (
            round(float(candidate.size_bytes) / float(fp32.size_bytes), 4) if fp32.size_bytes > 0 else None
        )
        details["ms_ratio_vs_fp32"] = (
            round(float(candidate.median_ms) / float(fp32.median_ms), 4) if fp32.median_ms > 0 else None
        )
    return ok, details
