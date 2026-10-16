"""Compression budget checks relative to baseline INT8."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Measured:
    size_bytes: int
    median_ms: float
    quantized_ops: int


def machine_runtime_fingerprint(
    preprocess: str = "stretch",
    imgsz: int = 320,
    intra_threads: int = 4,
    inter_threads: int = 1,
) -> str:
    import onnxruntime as ort

    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "node": platform.node(),
        "release": platform.release(),
        "cpu_count": os.cpu_count(),
        "k_service": os.environ.get("K_SERVICE", ""),
        "k_revision": os.environ.get("K_REVISION", ""),
        "ort_version": getattr(ort, "__version__", "unknown"),
        "preprocess": preprocess,
        "imgsz": int(imgsz),
        "intra_threads": int(intra_threads),
        "inter_threads": int(inter_threads),
        "providers": ["CPUExecutionProvider"],
    }
    raw = json.dumps(info, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def measure_model(
    onnx_path: str | Path,
    preprocess: str = "stretch",
    n_runs: int = 20,
    cache_dir: str | Path | None = None,
    force_remeasure: bool = False,
    is_fp32: bool = False,
) -> Measured:
    from odct.infer import Detector
    from odct.quant import count_quantized_ops, sha256_file

    path = Path(onnx_path)
    size_bytes = path.stat().st_size
    model_sha = sha256_file(path)
    machine_fp = machine_runtime_fingerprint(preprocess=preprocess, imgsz=320)
    cache_key = hashlib.sha256(
        f"{model_sha}:{preprocess}:{n_runs}:{machine_fp}:{int(is_fp32)}".encode("utf-8")
    ).hexdigest()

    c_dir = Path(cache_dir) if cache_dir is not None else Path("/tmp/odct_cache/latency")
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"{cache_key}.json"

    if not force_remeasure and cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text())
            if (
                data.get("machine_fp") == machine_fp
                and data.get("model_sha256") == model_sha
                and data.get("n_runs") == int(n_runs)
            ):
                return Measured(
                    size_bytes=int(size_bytes),
                    median_ms=float(data["median_ms"]),
                    quantized_ops=int(data["quantized_ops"]),
                )
        except Exception:
            pass

    det = Detector(path, preprocess=preprocess, imgsz=320)
    median_ms = det.latency_ms(n_runs=n_runs)
    if is_fp32:
        q_ops = 1
    else:
        q_ops, _, _ = count_quantized_ops(path)

    payload = {
        "machine_fp": machine_fp,
        "model_sha256": model_sha,
        "preprocess": preprocess,
        "n_runs": int(n_runs),
        "size_bytes": int(size_bytes),
        "median_ms": float(median_ms),
        "quantized_ops": int(q_ops),
    }
    tmp_f = c_dir / f".{cache_key}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        tmp_f.write_text(json.dumps(payload, indent=2))
        os.replace(tmp_f, cache_file)
    finally:
        if tmp_f.exists():
            tmp_f.unlink(missing_ok=True)

    return Measured(size_bytes=int(size_bytes), median_ms=float(median_ms), quantized_ops=int(q_ops))


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
