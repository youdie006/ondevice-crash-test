"""Static INT8 QDQ quantization via onnxruntime."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import onnx
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
    model = onnx.load(str(onnx_path))
    conv_nodes = [n.name for n in model.graph.node if n.op_type == "Conv" and n.name]
    if preset == "first_conv":
        return conv_nodes[:2]
    if preset == "detect_head":
        head_nodes = [
            n.name
            for n in model.graph.node
            if n.op_type == "Conv" and n.name and "/model.23/" in n.name and n.name.endswith(".2/Conv")
        ]
        return head_nodes if head_nodes else conv_nodes[-6:]
    raise ValueError(f"Unknown exclusion preset: {preset}")


def quantize(
    fp32_path: str | Path,
    calib_images: Sequence[str | Path],
    out_path: str | Path,
    method: str = "minmax",
    per_channel: bool = False,
    preprocess: str = "stretch",
    nodes_to_exclude: list[str] | None = None,
) -> QuantResult:
    fp32_p = Path(fp32_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    pre_p = fp32_p.with_name(f"{fp32_p.stem}_pre.onnx")
    if not pre_p.exists():
        quant_pre_process(str(fp32_p), str(pre_p))

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }
    method_key = method.lower()
    if method_key not in method_map:
        raise ValueError(f"Unsupported calibration method: {method}")

    model_meta = onnx.load(str(pre_p))
    input_name = model_meta.graph.input[0].name if model_meta.graph.input else "images"

    reader = ImageListCalibrationReader(
        calib_images,
        input_name=input_name,
        preprocess=preprocess,
        imgsz=320,
    )

    extra_options: dict = {}
    if method_key == "percentile":
        extra_options["CalibPercentile"] = 99.999

    quantize_static(
        str(pre_p),
        str(out_p),
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

    size_bytes = out_p.stat().st_size
    q_ops, total_ops, ratio = count_quantized_ops(out_p)
    return QuantResult(
        path=str(out_p),
        size_bytes=size_bytes,
        quantized_op_ratio=ratio,
        quantized_ops=q_ops,
        total_ops=total_ops,
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
