"""Unit tests for odct.infer, odct.metrics, and odct.budget."""

from __future__ import annotations

import numpy as np
from PIL import Image

from odct.budget import Measured, check
from odct.infer import Det, decode_raw_yolo_output, preprocess_image
from odct.metrics import box_iou, match, persisted


def test_preprocess_and_decode_raw_yolo() -> None:
    img = Image.new("RGB", (640, 480), (120, 130, 140))
    tensor_s, meta_s = preprocess_image(img, preprocess="stretch", imgsz=320)
    assert tensor_s.shape == (1, 3, 320, 320)
    assert tensor_s.dtype == np.float32

    tensor_l, meta_l = preprocess_image(img, preprocess="letterbox", imgsz=320)
    assert tensor_l.shape == (1, 3, 320, 320)

    raw = np.zeros((1, 84, 2100), dtype=np.float32)
    # Place two overlapping boxes of class 0 (person) and one box of class 15 (cat)
    raw[0, 0:4, 10] = [160.0, 160.0, 80.0, 80.0]
    raw[0, 4 + 0, 10] = 0.90

    raw[0, 0:4, 11] = [162.0, 162.0, 80.0, 80.0]
    raw[0, 4 + 0, 11] = 0.60

    raw[0, 0:4, 50] = [50.0, 60.0, 40.0, 40.0]
    raw[0, 4 + 15, 50] = 0.85

    dets = decode_raw_yolo_output(raw, meta=meta_s, conf_thresh=0.25, iou_thresh=0.5)
    assert len(dets) == 2
    assert dets[0].cls == 0
    assert abs(dets[0].conf - 0.90) < 1e-5
    # Verify stretch unscaling back to 640x480
    assert abs(dets[0].box[0] - 240.0) < 1e-3
    assert abs(dets[0].box[1] - 180.0) < 1e-3
    assert abs(dets[0].box[2] - 400.0) < 1e-3
    assert abs(dets[0].box[3] - 300.0) < 1e-3


def test_metrics_persisted_and_match() -> None:
    d0 = [
        Det(cls=0, conf=0.9, box=(10.0, 10.0, 110.0, 110.0)),
        Det(cls=15, conf=0.8, box=(200.0, 200.0, 300.0, 300.0)),
    ]
    dc = [
        Det(cls=0, conf=0.85, box=(12.0, 12.0, 108.0, 108.0)),
        # Extra non-persisted FP32 detection on condition image
        Det(cls=56, conf=0.4, box=(400.0, 400.0, 500.0, 500.0)),
    ]
    p = persisted(d0, dc, iou=0.5)
    assert len(p) == 1
    assert p[0].cls == 0

    # INT8 predicts the persisted object AND agrees with the non-persisted dc detection, plus 1 true hallucination
    q = [
        Det(cls=0, conf=0.8, box=(11.0, 11.0, 109.0, 109.0)),
        Det(cls=56, conf=0.5, box=(402.0, 402.0, 498.0, 498.0)),
        Det(cls=16, conf=0.7, box=(50.0, 300.0, 120.0, 380.0)),
    ]
    tp, fp, fn = match(p, q, iou=0.5, dc_all=dc)
    assert tp == 1
    assert fn == 0
    # Agreeing with dc[1] (cls 56) is not counted as FP; only cls 16 is FP
    assert fp == 1
    assert box_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0


def test_budget_check() -> None:
    baseline = Measured(size_bytes=3_000_000, median_ms=10.0, quantized_ops=60)
    cand_ok = Measured(size_bytes=3_200_000, median_ms=10.8, quantized_ops=56)
    ok, details = check(cand_ok, baseline)
    assert ok is True
    assert details["size_ok"] is True
    assert details["ms_ok"] is True
    assert details["ops_ok"] is True

    cand_bad = Measured(size_bytes=3_500_000, median_ms=11.5, quantized_ops=50)
    ok2, details2 = check(cand_bad, baseline)
    assert ok2 is False
    assert len(details2["reasons"]) == 3
