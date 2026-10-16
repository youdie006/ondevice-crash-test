"""Unit tests for odct.infer, odct.metrics, odct.budget, and odct.quant."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

from odct.budget import Measured, check
from odct.infer import Det, decode_raw_yolo_output, preprocess_image
from odct.metrics import CondMetrics, box_iou, evaluate_condition, is_passing_f1, match, persisted
from odct.quant import count_quantized_ops


def test_preprocess_and_decode_raw_yolo() -> None:
    img = Image.new("RGB", (640, 480), (120, 130, 140))
    tensor_s, meta_s = preprocess_image(img, preprocess="stretch", imgsz=320)
    assert tensor_s.shape == (1, 3, 320, 320)
    assert tensor_s.dtype == np.float32

    tensor_l, meta_l = preprocess_image(img, preprocess="letterbox", imgsz=320)
    assert tensor_l.shape == (1, 3, 320, 320)

    raw = np.zeros((1, 84, 2100), dtype=np.float32)
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
    assert abs(dets[0].box[0] - 240.0) < 1e-3
    assert abs(dets[0].box[1] - 180.0) < 1e-3
    assert abs(dets[0].box[2] - 400.0) < 1e-3
    assert abs(dets[0].box[3] - 300.0) < 1e-3


def test_metrics_persisted_and_match_including_iou_non_transitivity() -> None:
    d0 = [
        Det(cls=0, conf=0.9, box=(10.0, 10.0, 110.0, 110.0)),
        Det(cls=15, conf=0.8, box=(200.0, 200.0, 300.0, 300.0)),
    ]
    dc = [
        Det(cls=0, conf=0.85, box=(12.0, 12.0, 108.0, 108.0)),
        Det(cls=56, conf=0.4, box=(400.0, 400.0, 500.0, 500.0)),
    ]
    p = persisted(d0, dc, iou=0.5)
    assert len(p) == 1
    assert p[0].cls == 0

    q = [
        Det(cls=0, conf=0.8, box=(11.0, 11.0, 109.0, 109.0)),
        Det(cls=56, conf=0.5, box=(402.0, 402.0, 498.0, 498.0)),
        Det(cls=16, conf=0.7, box=(50.0, 300.0, 120.0, 380.0)),
    ]
    tp, fp, fn = match(p, q, iou=0.5, dc_all=dc)
    assert tp == 1
    assert fn == 0
    assert fp == 1
    assert box_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0

    # IoU non-transitivity boundary test:
    r_non_trans = [Det(cls=0, conf=0.9, box=(0.0, 0.0, 100.0, 100.0))]
    dc_non_trans = [Det(cls=0, conf=0.85, box=(30.0, 0.0, 130.0, 100.0))]
    q_non_trans = [Det(cls=0, conf=0.8, box=(-30.0, 0.0, 70.0, 100.0))]
    tp_nt, fp_nt, fn_nt = match(r_non_trans, q_non_trans, iou=0.5, dc_all=dc_non_trans)
    assert tp_nt == 1
    assert fn_nt == 0
    assert fp_nt == 1


def test_empty_set_and_zero_denominator_metrics(tmp_path: Path) -> None:
    class DummyDetector:
        imgsz = 320

        def __init__(self, outputs: dict[str, list[Det]]) -> None:
            self.outputs = outputs

        def detect(self, path: str | Path) -> list[Det]:
            return self.outputs.get(Path(path).name, [])

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (320, 320), (100, 100, 100)).save(img1)
    Image.new("RGB", (320, 320), (100, 100, 100)).save(img2)

    # Case 1: Empty detections everywhere -> None (rendered as N/A in UI) and passed=False
    fp32_empty = DummyDetector({})
    int8_empty = DummyDetector({})
    m_empty = evaluate_condition(fp32_empty, int8_empty, [img1, img2], [img1, img2])
    assert m_empty.f1 is None
    assert m_empty.recall is None
    assert m_empty.precision is None
    assert m_empty.retention is None
    assert m_empty.per_size["small"] is None
    assert m_empty.per_size["medium"] is None
    assert m_empty.per_size["large"] is None
    assert m_empty.passed is False
    assert is_passing_f1(m_empty) is False
    assert m_empty.excluded_images == 2
    assert m_empty.valid_images == 0
    assert m_empty.collapse_images == 0

    # Case 2: Unrounded F1 boundary (e.g., 0.79996 rounds to 0.8000 but raw_f1 < 0.80 -> must fail)
    m_borderline = CondMetrics(
        f1=0.80,
        recall=0.80,
        precision=0.80,
        retention=1.0,
        per_class={},
        per_size={"small": None, "medium": None, "large": 0.80},
        n_images=10,
        limitation=False,
        raw_f1=0.79996,
        passed=False,
    )
    assert is_passing_f1(m_borderline, threshold=0.80) is False


def test_budget_check_and_boundaries() -> None:
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

    # Reproduction of independent acceptance failure: zero baseline must FAIL
    ok_zero_base, details_zero_base = check(Measured(10, 1.0, 1), Measured(0, 0.0, 0))
    assert ok_zero_base is False
    assert len(details_zero_base["reasons"]) >= 3

    # Negative, zero, NaN, Inf candidate or baseline values must FAIL
    assert check(Measured(0, 1.0, 1), baseline)[0] is False
    assert check(Measured(-10, 1.0, 1), baseline)[0] is False
    assert check(Measured(100, -1.0, 1), baseline)[0] is False
    assert check(Measured(100, math.nan, 1), baseline)[0] is False
    assert check(Measured(100, math.inf, 1), baseline)[0] is False
    assert check(cand_ok, Measured(3_000_000, math.nan, 60))[0] is False


def test_count_quantized_ops_requires_both_activation_and_weight_qdq(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])

    dq_act1 = helper.make_node("DequantizeLinear", ["x_q", "s1", "z1"], ["act1_dq"], name="dq_act1")
    dq_w1 = helper.make_node("DequantizeLinear", ["w1_q", "sw1", "zw1"], ["w1_dq"], name="dq_w1")
    conv1 = helper.make_node("Conv", ["act1_dq", "w1_dq"], ["conv1_out"], name="Conv_quantized")

    dq_act2 = helper.make_node("DequantizeLinear", ["c1_q", "s2", "z2"], ["act2_dq"], name="dq_act2")
    conv2 = helper.make_node("Conv", ["act2_dq", "w2_fp32_raw"], ["y"], name="Conv_fp32_weight")

    graph = helper.make_graph([dq_act1, dq_w1, conv1, dq_act2, conv2], "test_qdq", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_file = tmp_path / "test_qdq.onnx"
    onnx.save(model, str(onnx_file))

    q_ops, total_ops, ratio = count_quantized_ops(onnx_file)
    assert total_ops == 2
    assert q_ops == 1
    assert abs(ratio - 0.5) < 1e-6
