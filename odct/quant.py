"""Static INT8 QDQ quantization via onnxruntime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from odct.infer import preprocess_image


@dataclass(frozen=True)
class QuantResult:
    path: str
    size_bytes: int
    quantized_op_ratio: float
    quantized_ops: int = 0
    total_ops: int = 0
    config_hash: str = ""
    cached: bool = False


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_quant_config_hash(
    fp32_path: str | Path,
    calib_images: Sequence[str | Path],
    method: str = "minmax",
    per_channel: bool = False,
    preprocess: str = "stretch",
    nodes_to_exclude: Sequence[str] | None = None,
    op_types_to_quantize: Sequence[str] = ("Conv",),
    extra_options: dict[str, Any] | None = None,
    imgsz: int = 320,
) -> str:
    fp32_sha = sha256_file(fp32_path)
    calib_entries = []
    for p in calib_images:
        pth = Path(p)
        calib_entries.append({"name": pth.name, "sha256": sha256_file(pth)})

    payload = {
        "fp32_sha256": fp32_sha,
        "calib_images": calib_entries,
        "method": method.lower(),
        "per_channel": bool(per_channel),
        "preprocess": preprocess,
        "nodes_to_exclude": sorted(set(nodes_to_exclude or [])),
        "op_types_to_quantize": list(op_types_to_quantize),
        "extra_options": extra_options or {},
        "imgsz": int(imgsz),
        "ort_version": getattr(ort, "__version__", "unknown"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ImageListCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        image_paths: Sequence[str | Path],
        input_name: str = "images",
        preprocess: str = "stretch",
        imgsz: int = 320,
    ) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.input_name = input_name
        self.preprocess = preprocess
        self.imgsz = imgsz
        self._iter = iter(self.image_paths)

    def get_next(self) -> dict | None:
        try:
            img_path = next(self._iter)
        except StopIteration:
            return None
        tensor, _ = preprocess_image(img_path, preprocess=self.preprocess, imgsz=self.imgsz)
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self._iter = iter(self.image_paths)


def count_quantized_ops(onnx_path: str | Path) -> tuple[int, int, float]:
    """Count target compute ops (Conv, Gemm, MatMul) where BOTH activation (input[0]) and weight (input[1]) come from DequantizeLinear."""
    model = onnx.load(str(onnx_path))
    target_ops = {"Conv", "Gemm", "MatMul"}
    nodes = list(model.graph.node)
    total_target = 0
    quantized_target = 0
    dq_outputs = {
        out
        for node in nodes
        if node.op_type == "DequantizeLinear"
        for out in node.output
    }
    for node in nodes:
        if node.op_type in target_ops:
            total_target += 1
            if len(node.input) >= 2 and node.input[0] in dq_outputs and node.input[1] in dq_outputs:
                quantized_target += 1
    ratio = float(quantized_target) / float(total_target) if total_target > 0 else 0.0
    return quantized_target, total_target, ratio


def get_preset_exclusion_nodes(onnx_path: str | Path, preset: str) -> list[str]:
    import re

    model = onnx.load(str(onnx_path))
    conv_nodes = [n.name for n in model.graph.node if n.op_type == "Conv" and n.name]
    if preset == "first_conv":
        return conv_nodes[:2]
    if preset == "detect_head":
        pattern = re.compile(r"(?:^|/)model\.(\d+)(?:/|$)")
        indices: list[int] = []
        for name in conv_nodes:
            m = pattern.search(name)
            if m:
                indices.append(int(m.group(1)))
        if indices:
            max_idx = max(indices)
            head_pattern = re.compile(rf"(?:^|/)model\.{max_idx}(?:/|$)")
            return [name for name in conv_nodes if head_pattern.search(name)]
        return conv_nodes[-6:]
    raise ValueError(f"Unknown exclusion preset: {preset}")


def quantize(
    fp32_path: str | Path,
    calib_images: Sequence[str | Path],
    out_path: str | Path,
    method: str = "minmax",
    per_channel: bool = False,
    preprocess: str = "stretch",
    nodes_to_exclude: list[str] | None = None,
    use_cache: bool = True,
) -> QuantResult:
    fp32_p = Path(fp32_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }
    method_key = method.lower()
    if method_key not in method_map:
        raise ValueError(f"Unsupported calibration method: {method}")

    extra_options: dict[str, Any] = {}
    if method_key == "percentile":
        extra_options["CalibPercentile"] = 99.999

    config_hash = compute_quant_config_hash(
        fp32_path=fp32_p,
        calib_images=calib_images,
        method=method_key,
        per_channel=per_channel,
        preprocess=preprocess,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=["Conv"],
        extra_options=extra_options,
        imgsz=320,
    )

    meta_p = out_p.with_name(f"{out_p.name}.meta.json")
    if use_cache and out_p.is_file() and meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text())
            if (
                meta.get("config_hash") == config_hash
                and meta.get("model_sha256") == sha256_file(out_p)
            ):
                return QuantResult(
                    path=str(out_p),
                    size_bytes=int(meta["size_bytes"]),
                    quantized_op_ratio=float(meta["quantized_op_ratio"]),
                    quantized_ops=int(meta["quantized_ops"]),
                    total_ops=int(meta["total_ops"]),
                    config_hash=config_hash,
                    cached=True,
                )
        except Exception:
            pass

    pre_p = fp32_p.with_name(f"{fp32_p.stem}_pre.onnx")
    if not pre_p.is_file():
        tmp_pre = pre_p.with_name(f".{pre_p.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            quant_pre_process(str(fp32_p), str(tmp_pre))
            os.replace(tmp_pre, pre_p)
        finally:
            if tmp_pre.exists():
                tmp_pre.unlink(missing_ok=True)

    model_meta = onnx.load(str(pre_p))
    input_name = model_meta.graph.input[0].name if model_meta.graph.input else "images"

    reader = ImageListCalibrationReader(
        calib_images,
        input_name=input_name,
        preprocess=preprocess,
        imgsz=320,
    )

    tmp_out = out_p.with_name(f".{out_p.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    tmp_meta = meta_p.with_name(f".{meta_p.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        quantize_static(
            str(pre_p),
            str(tmp_out),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv"],
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=bool(per_channel),
            calibrate_method=method_map[method_key],
            nodes_to_exclude=nodes_to_exclude or [],
            extra_options=extra_options,
        )
        size_bytes = tmp_out.stat().st_size
        q_ops, total_ops, ratio = count_quantized_ops(tmp_out)
        model_sha = sha256_file(tmp_out)
        os.replace(tmp_out, out_p)

        meta_payload = {
            "config_hash": config_hash,
            "model_sha256": model_sha,
            "size_bytes": size_bytes,
            "quantized_ops": q_ops,
            "total_ops": total_ops,
            "quantized_op_ratio": ratio,
        }
        tmp_meta.write_text(json.dumps(meta_payload, indent=2))
        os.replace(tmp_meta, meta_p)
    finally:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        if tmp_meta.exists():
            tmp_meta.unlink(missing_ok=True)

    return QuantResult(
        path=str(out_p),
        size_bytes=size_bytes,
        quantized_op_ratio=ratio,
        quantized_ops=q_ops,
        total_ops=total_ops,
        config_hash=config_hash,
        cached=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize FP32 ONNX detector to INT8 QDQ.")
    parser.add_argument("--baseline", action="store_true", help="Build baseline INT8 model using lab_calib_25.json")
    parser.add_argument("--scenario", default="robot_vacuum", help="Scenario ID")
    parser.add_argument("--fp32", default="models/fp32.onnx", help="Input FP32 ONNX path")
    parser.add_argument("--out", default="models/baseline_int8.onnx", help="Output INT8 ONNX path")
    args = parser.parse_args()

    if args.baseline:
        calib_file = Path("data") / args.scenario / "lab_calib_25.json"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}. Run select_lab_calib.py first.")
        calib_data = json.loads(calib_file.read_text())
        calib_images = calib_data["selected_paths"]
        res = quantize(
            fp32_path=args.fp32,
            calib_images=calib_images,
            out_path=args.out,
            method="minmax",
            per_channel=False,
            preprocess="stretch",
            nodes_to_exclude=None,
        )
        print(
            json.dumps(
                {
                    "path": res.path,
                    "size_bytes": res.size_bytes,
                    "quantized_ops": res.quantized_ops,
                    "total_ops": res.total_ops,
                    "quantized_op_ratio": round(res.quantized_op_ratio, 4),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
