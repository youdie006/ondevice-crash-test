#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="."
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

echo "=== [1/5] Running unit tests (infer, metrics, budget) ==="
"$PYTHON_BIN" -m pytest -q

echo "=== [2/5] Exporting detector to ONNX (raw 1x84x2100) ==="
if [ ! -f models/fp32.onnx ]; then
  "$PYTHON_BIN" scripts/export_model.py --out models/fp32.onnx
else
  echo "models/fp32.onnx already exists, skipping export."
fi

echo "=== [3/5] Fetching 80 COCO val2017 images and writing seeded A40/B40 split ==="
if [ ! -f data/robot_vacuum/splits.json ]; then
  "$PYTHON_BIN" scripts/fetch_data.py --scenario robot_vacuum --count 80 --fp32 models/fp32.onnx
else
  echo "data/robot_vacuum/splits.json already exists, skipping download."
fi

echo "=== [4/5] Selecting 25 brightest and sharpest Split-A images and quantizing baseline INT8 ==="
"$PYTHON_BIN" scripts/select_lab_calib.py --scenario robot_vacuum --top-k 25
"$PYTHON_BIN" -m odct.quant --baseline --scenario robot_vacuum --fp32 models/fp32.onnx --out models/baseline_int8.onnx

echo "=== [5/5] Evaluating Split-B control F1 and recording baseline budget reference ==="
"$PYTHON_BIN" -m odct.metrics --control --scenario robot_vacuum --fp32 models/fp32.onnx --int8 models/baseline_int8.onnx
