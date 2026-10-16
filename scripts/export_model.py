"""Export YOLO26n (or fallback YOLO11n) to ONNX with raw (1,84,2100) output layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


def export_detector(out_path: str | Path = "models/fp32.onnx", imgsz: int = 320) -> dict:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    candidates = ["yolo26n.pt", "yolo11n.pt"]
    used_model = None
    exported_file = None

    for weight_name in candidates:
        try:
            model = YOLO(weight_name)
            res_path = model.export(format="onnx", imgsz=imgsz, opset=17)
            exported_file = Path(res_path)
            sess = ort.InferenceSession(str(exported_file), providers=["CPUExecutionProvider"])
            out_shape = tuple(sess.get_outputs()[0].shape)
            if out_shape != (1, 84, 2100):
                raise RuntimeError(f"Unexpected output shape {out_shape} from {weight_name}")
            used_model = weight_name
            break
        except Exception as exc:
            print(f"Export attempt with {weight_name} failed: {exc}")
            continue

    if exported_file is None or used_model is None:
        raise RuntimeError("Failed to export both yolo26n.pt and yolo11n.pt")

    shutil.move(str(exported_file), str(out_p))
    sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]

    info = {
        "source_weights": used_model,
        "onnx_path": str(out_p),
        "size_bytes": out_p.stat().st_size,
        "input_name": in_meta.name,
        "input_shape": list(in_meta.shape),
        "output_name": out_meta.name,
        "output_shape": list(out_meta.shape),
    }
    print(json.dumps(info, indent=2))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO detector to raw ONNX (1,84,2100).")
    parser.add_argument("--out", default="models/fp32.onnx", help="Output ONNX file path")
    parser.add_argument("--imgsz", type=int, default=320, help="Input resolution")
    args = parser.parse_args()
    export_detector(out_path=args.out, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
