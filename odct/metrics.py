"""Label-free oracle metrics: persisted objects, greedy matching, and condition evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from PIL import Image

from odct.budget import Measured, check
from odct.infer import Det, Detector
from odct.quant import count_quantized_ops


@dataclass(frozen=True)
class CondMetrics:
    f1: float
    recall: float
    precision: float
    retention: float
    per_class: dict[str, float]
    per_size: dict[str, float]
    n_images: int
    limitation: bool
    ref_objects: int = 0
    orig_objects: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ci90_low: float | None = None
    ci90_high: float | None = None


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def persisted(d0: list[Det], dc: list[Det], iou: float = 0.5) -> list[Det]:
    result: list[Det] = []
    for d in d0:
        if any(d.cls == d_prime.cls and box_iou(d.box, d_prime.box) >= iou for d_prime in dc):
            result.append(d)
    return result


def _greedy_pairs(ref: list[Det], pred: list[Det], iou: float = 0.5) -> tuple[set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for r_idx, r in enumerate(ref):
        for p_idx, p in enumerate(pred):
            if r.cls == p.cls:
                score = box_iou(r.box, p.box)
                if score >= iou:
                    candidates.append((score, r_idx, p_idx))
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_r: set[int] = set()
    matched_p: set[int] = set()
    for _, r_idx, p_idx in candidates:
        if r_idx not in matched_r and p_idx not in matched_p:
            matched_r.add(r_idx)
            matched_p.add(p_idx)
    return matched_r, matched_p


def match(
    ref: list[Det],
    pred: list[Det],
    iou: float = 0.5,
    dc_all: list[Det] | None = None,
) -> tuple[int, int, int]:
    matched_r, matched_p = _greedy_pairs(ref, pred, iou=iou)
    tp = len(matched_r)
    fn = len(ref) - tp
    if dc_all is None:
        fp = len(pred) - len(matched_p)
    else:
        fp = 0
        for p_idx, q in enumerate(pred):
            if p_idx in matched_p:
                continue
            if not any(q.cls == d_c.cls and box_iou(q.box, d_c.box) >= iou for d_c in dc_all):
                fp += 1
    return tp, fp, fn


def _size_bucket(box: tuple[float, float, float, float], orig_w: int, orig_h: int, imgsz: int = 320) -> str:
    scale_x = float(imgsz) / float(max(1, orig_w))
    scale_y = float(imgsz) / float(max(1, orig_h))
    w_320 = max(0.0, (box[2] - box[0]) * scale_x)
    h_320 = max(0.0, (box[3] - box[1]) * scale_y)
    area = w_320 * h_320
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def evaluate_condition(
    fp32: Detector,
    int8: Detector,
    originals: Sequence[str | Path],
    condition_images: Sequence[str | Path],
    iou: float = 0.5,
) -> CondMetrics:
    if len(originals) != len(condition_images):
        raise ValueError("originals and condition_images must have the same length")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_d0 = 0
    total_p = 0

    class_tp: dict[int, int] = {}
    class_fp: dict[int, int] = {}
    class_fn: dict[int, int] = {}

    size_tp = {"small": 0, "medium": 0, "large": 0}
    size_total = {"small": 0, "medium": 0, "large": 0}

    per_img_counts: list[tuple[int, int, int]] = []

    for orig_p, cond_p in zip(originals, condition_images):
        orig_path = Path(orig_p)
        cond_path = Path(cond_p)
        with Image.open(orig_path) as im:
            orig_w, orig_h = im.size

        d0 = fp32.detect(orig_path)
        if orig_path.resolve() == cond_path.resolve():
            dc = d0
            ref = d0
        else:
            dc = fp32.detect(cond_path)
            ref = persisted(d0, dc, iou=iou)

        q = int8.detect(cond_path)

        matched_r, matched_p = _greedy_pairs(ref, q, iou=iou)
        tp, fp, fn = match(ref, q, iou=iou, dc_all=dc)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_d0 += len(d0)
        total_p += len(ref)
        per_img_counts.append((tp, fp, fn))

        for r_idx, r_det in enumerate(ref):
            c = r_det.cls
            b_size = _size_bucket(r_det.box, orig_w, orig_h, imgsz=fp32.imgsz)
            size_total[b_size] += 1
            if r_idx in matched_r:
                class_tp[c] = class_tp.get(c, 0) + 1
                size_tp[b_size] += 1
            else:
                class_fn[c] = class_fn.get(c, 0) + 1

        for p_idx, q_det in enumerate(q):
            if p_idx in matched_p:
                continue
            if not any(q_det.cls == d_c.cls and box_iou(q_det.box, d_c.box) >= iou for d_c in dc):
                c = q_det.cls
                class_fp[c] = class_fp.get(c, 0) + 1

    denom = 2 * total_tp + total_fp + total_fn
    f1 = float(2 * total_tp) / float(denom) if denom > 0 else 1.0
    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    retention = float(total_p) / float(total_d0) if total_d0 > 0 else 1.0

    all_classes = sorted(set(class_tp.keys()) | set(class_fp.keys()) | set(class_fn.keys()))
    per_class: dict[str, float] = {}
    for c in all_classes:
        ctp = class_tp.get(c, 0)
        cfp = class_fp.get(c, 0)
        cfn = class_fn.get(c, 0)
        cdenom = 2 * ctp + cfp + cfn
        per_class[str(c)] = round(float(2 * ctp) / float(cdenom), 4) if cdenom > 0 else 1.0

    per_size: dict[str, float] = {}
    for b_key in ("small", "medium", "large"):
        if size_total[b_key] > 0:
            per_size[b_key] = round(float(size_tp[b_key]) / float(size_total[b_key]), 4)
        else:
            per_size[b_key] = 1.0

    ci_low: float | None = None
    ci_high: float | None = None
    if len(per_img_counts) >= 2:
        rng = np.random.default_rng(42)
        arr = np.asarray(per_img_counts, dtype=np.float64)
        n_img = len(arr)
        boot_f1s: list[float] = []
        for _ in range(200):
            sample_idx = rng.integers(0, n_img, size=n_img)
            sums = arr[sample_idx].sum(axis=0)
            b_denom = 2.0 * sums[0] + sums[1] + sums[2]
            boot_f1s.append(float(2.0 * sums[0] / b_denom) if b_denom > 0 else 1.0)
        ci_low = round(float(np.percentile(boot_f1s, 5)), 4)
        ci_high = round(float(np.percentile(boot_f1s, 95)), 4)

    return CondMetrics(
        f1=round(f1, 4),
        recall=round(recall, 4),
        precision=round(precision, 4),
        retention=round(retention, 4),
        per_class=per_class,
        per_size=per_size,
        n_images=len(originals),
        limitation=bool(retention < 0.7),
        ref_objects=total_p,
        orig_objects=total_d0,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        ci90_low=ci_low,
        ci90_high=ci_high,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline INT8 control metrics on split B.")
    parser.add_argument("--control", action="store_true", help="Run split B control evaluation and write baseline.json")
    parser.add_argument("--scenario", default="robot_vacuum", help="Scenario ID")
    parser.add_argument("--fp32", default="models/fp32.onnx", help="FP32 model path")
    parser.add_argument("--int8", default="models/baseline_int8.onnx", help="Baseline INT8 model path")
    args = parser.parse_args()

    if args.control:
        scenario_file = Path("scenarios") / f"{args.scenario}.yaml"
        scenario_cfg = yaml.safe_load(scenario_file.read_text())
        allowed_classes = list(scenario_cfg["classes"].values())

        split_file = Path("data") / args.scenario / "splits.json"
        splits = json.loads(split_file.read_text())
        split_b_paths = [Path("data") / args.scenario / "real" / fname for fname in splits["split_b"]]

        fp32_det = Detector(args.fp32, preprocess="stretch", allowed_classes=allowed_classes)
        int8_det = Detector(args.int8, preprocess="stretch", allowed_classes=allowed_classes)

        metrics = evaluate_condition(fp32_det, int8_det, split_b_paths, split_b_paths)

        fp32_size = Path(args.fp32).stat().st_size
        int8_size = Path(args.int8).stat().st_size
        fp32_ms = fp32_det.latency_ms(n_runs=20)
        int8_ms = int8_det.latency_ms(n_runs=20)
        q_ops, total_ops, q_ratio = count_quantized_ops(args.int8)

        baseline_meas = Measured(size_bytes=int8_size, median_ms=int8_ms, quantized_ops=q_ops)
        fp32_meas = Measured(size_bytes=fp32_size, median_ms=fp32_ms, quantized_ops=0)
        budget_ok, budget_details = check(baseline_meas, baseline_meas, fp32=fp32_meas)

        out_dir = Path("showcase") / args.scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "baseline.json"

        payload = {
            "scenario": args.scenario,
            "split": "B",
            "n_images": len(split_b_paths),
            "control_metrics": asdict(metrics),
            "baseline_int8": {
                "path": args.int8,
                "size_bytes": int8_size,
                "median_ms": round(int8_ms, 3),
                "quantized_ops": q_ops,
                "total_ops": total_ops,
                "quantized_op_ratio": round(q_ratio, 4),
            },
            "fp32": {
                "path": args.fp32,
                "size_bytes": fp32_size,
                "median_ms": round(fp32_ms, 3),
            },
            "budget_reference": budget_details,
        }
        out_file.write_text(json.dumps(payload, indent=2) + chr(10))
        print(json.dumps(payload, indent=2))
        if metrics.f1 < 0.90:
            raise SystemExit(f"Control F1 {metrics.f1:.4f} is below required 0.90 threshold.")


if __name__ == "__main__":
    main()
