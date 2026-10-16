"""ONNX detector inference with stretch/letterbox preprocessing and raw (1,84,2100) decoding."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass(frozen=True)
class Det:
    cls: int
    conf: float
    box: tuple[float, float, float, float]


def preprocess_image(
    image: Image.Image | Path | str,
    preprocess: str = "stretch",
    imgsz: int = 320,
) -> tuple[np.ndarray, dict[str, float]]:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    orig_w, orig_h = img.size

    if preprocess == "stretch":
        resized = img.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))[None, ...]
        meta = {
            "mode": 0.0,
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
            "scale_x": float(orig_w) / float(imgsz),
            "scale_y": float(orig_h) / float(imgsz),
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        return np.ascontiguousarray(chw, dtype=np.float32), meta

    if preprocess == "letterbox":
        r = min(float(imgsz) / float(orig_w), float(imgsz) / float(orig_h))
        new_w = int(round(orig_w * r))
        new_h = int(round(orig_h * r))
        pad_x = (imgsz - new_w) / 2.0
        pad_y = (imgsz - new_h) / 2.0
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas.paste(resized, (left, top))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))[None, ...]
        meta = {
            "mode": 1.0,
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
            "scale_x": 1.0 / r,
            "scale_y": 1.0 / r,
            "pad_x": float(left),
            "pad_y": float(top),
        }
        return np.ascontiguousarray(chw, dtype=np.float32), meta

    raise ValueError(f"Unsupported preprocess mode: {preprocess}")


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thresh: float = 0.5,
    max_det: int = 100,
) -> list[int]:
    if len(boxes) == 0:
        return []
    keep_indices: list[int] = []
    unique_classes = np.unique(classes)
    for c in unique_classes:
        cls_mask = np.where(classes == c)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        order = cls_scores.argsort()[::-1]
        while order.size > 0:
            idx = int(order[0])
            keep_indices.append(int(cls_mask[idx]))
            if order.size == 1:
                break
            rest = order[1:]
            ious = _box_iou_one_to_many(cls_boxes[idx], cls_boxes[rest])
            order = rest[ious < iou_thresh]
    keep_indices.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep_indices[:max_det]


def decode_raw_yolo_output(
    out: np.ndarray,
    meta: dict[str, float],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    allowed_classes: set[int] | None = None,
) -> list[Det]:
    if out.ndim != 3 or out.shape[0] != 1 or out.shape[1] < 5:
        raise ValueError(f"Expected raw YOLO output shape (1, 84, N), got {out.shape}")

    p = out[0].T  # (N, 84)
    scores = p[:, 4:]
    conf = scores.max(axis=1)
    cls = scores.argmax(axis=1)

    keep = conf >= conf_thresh
    if allowed_classes is not None:
        cls_mask = np.isin(cls, list(allowed_classes))
        keep = keep & cls_mask

    if not np.any(keep):
        return []

    xywh = p[keep, :4]
    conf = conf[keep]
    cls = cls[keep]

    xyxy = np.c_[
        xywh[:, 0] - xywh[:, 2] / 2.0,
        xywh[:, 1] - xywh[:, 3] / 2.0,
        xywh[:, 0] + xywh[:, 2] / 2.0,
        xywh[:, 1] + xywh[:, 3] / 2.0,
    ]

    kept_idx = class_aware_nms(xyxy, conf, cls, iou_thresh=iou_thresh, max_det=100)
    if not kept_idx:
        return []

    xyxy = xyxy[kept_idx]
    conf = conf[kept_idx]
    cls = cls[kept_idx]

    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    scale_x = meta["scale_x"]
    scale_y = meta["scale_y"]

    x1 = np.clip((xyxy[:, 0] - pad_x) * scale_x, 0.0, orig_w)
    y1 = np.clip((xyxy[:, 1] - pad_y) * scale_y, 0.0, orig_h)
    x2 = np.clip((xyxy[:, 2] - pad_x) * scale_x, 0.0, orig_w)
    y2 = np.clip((xyxy[:, 3] - pad_y) * scale_y, 0.0, orig_h)

    dets: list[Det] = []
    for i in range(len(kept_idx)):
        if x2[i] > x1[i] and y2[i] > y1[i]:
            dets.append(
                Det(
                    cls=int(cls[i]),
                    conf=float(conf[i]),
                    box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                )
            )
    return dets


class Detector:
    def __init__(
        self,
        onnx_path: str | Path,
        preprocess: str = "stretch",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 320,
        allowed_classes: Sequence[int] | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if preprocess not in ("stretch", "letterbox"):
            raise ValueError(f"Invalid preprocess mode: {preprocess}")
        self.preprocess = preprocess
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.allowed_classes = set(allowed_classes) if allowed_classes is not None else None

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, image_path: str | Path | Image.Image) -> list[Det]:
        tensor, meta = preprocess_image(image_path, preprocess=self.preprocess, imgsz=self.imgsz)
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return decode_raw_yolo_output(
            out,
            meta=meta,
            conf_thresh=self.conf,
            iou_thresh=self.iou,
            allowed_classes=self.allowed_classes,
        )

    def latency_ms(self, n_runs: int = 20) -> float:
        rng = np.random.default_rng(42)
        dummy = rng.random((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(5):
            self.session.run([self.output_name], {self.input_name: dummy})
        times_ms: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: dummy})
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        return float(np.median(times_ms))
