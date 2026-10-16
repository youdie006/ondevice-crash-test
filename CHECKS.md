# Acceptance Checks (Blocks A & B)

## Block A: Cloud Run Deployment Skeleton & HTTP 200 Verification

### 1. Cloud Run Deploy Command
```bash
gcloud run deploy ondevice-crash-test --source . --region asia-northeast3 --cpu 4 --memory 8Gi --concurrency 1 --min-instances 1 --max-instances 1 --no-cpu-throttling --timeout 3600 --allow-unauthenticated --quiet
```

**Actual Output:**
```text
Building using Dockerfile and deploying container to Cloud Run service [ondevice-crash-test] in project [h2gdm-hack26icn-2684] region [asia-northeast3]
✓ Building and deploying new service... Done.
  ✓ Creating Container Repository...
  ✓ Validating configuration...
  ✓ Uploading sources...
  ✓ Building Container... Logs are available at [ https://console.cloud.google.com/cloud-build/builds;region=asia-northeast3/933bf352-10eb-4606-9deb-1a776d5eede1?project=1056265768652 ].
  ✓ Creating Revision...
  ✓ Routing traffic...
  ✓ Setting IAM Policy...
Done.
Service [ondevice-crash-test] revision [ondevice-crash-test-00001-w76] has been deployed and is serving 100 percent of traffic.
Service URL: https://ondevice-crash-test-1056265768652.asia-northeast3.run.app
```
- **Exit Code:** `0`

### 2. HTTP 200 Acceptance Check on Deployed URLs
```bash
curl -s -o /dev/null -w "%{http_code}
" "https://ondevice-crash-test-1056265768652.asia-northeast3.run.app" && echo "EXIT_CODE=$?" && curl -s -o /dev/null -w "%{http_code}
" "https://ondevice-crash-test-igvnvmmxvq-du.a.run.app" && echo "EXIT_CODE=$?"
```

**Actual Output:**
```text
200
EXIT_CODE=0
200
EXIT_CODE=0
```
- **Exit Code:** `0`
- **Verdict:** **PASS**

---

## Block B: Model Export, 80-Photo Fetch (A40/B40 Split), Lab Calibration (25 Images), Baseline INT8 (`op_types_to_quantize=["Conv"]`), and Split B Control Metrics

### 1. Unit & Boundary Tests (`pytest`)
```bash
.venv/bin/python -m pytest -v
```

**Actual Output:**
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0 -- /Users/manon/hackathon/ondevice-crash-test/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/manon/hackathon/ondevice-crash-test
plugins: anyio-4.15.1
collected 5 items                                                              

tests/test_core.py::test_preprocess_and_decode_raw_yolo PASSED           [ 20%]
tests/test_core.py::test_metrics_persisted_and_match_including_iou_non_transitivity PASSED [ 40%]
tests/test_core.py::test_empty_set_and_zero_denominator_metrics PASSED   [ 60%]
tests/test_core.py::test_budget_check_and_boundaries PASSED              [ 80%]
tests/test_core.py::test_count_quantized_ops_requires_both_activation_and_weight_qdq PASSED [100%]

============================== 5 passed in 0.30s ===============================
```
- **Exit Code:** `0`

### 2. Block B Full Pipeline (`./scripts/run_block_b.sh`)
```bash
./scripts/run_block_b.sh && echo "BLOCK_B_EXIT_CODE=$?"
```

**Actual Output:**
```text
=== [1/5] Running unit tests (infer, metrics, budget) ===
.....                                                                    [100%]
5 passed in 0.30s
=== [2/5] Exporting detector to ONNX (raw 1x84x2100) ===
models/fp32.onnx already exists, skipping export.
=== [3/5] Fetching 80 COCO val2017 images and writing seeded A40/B40 split ===
data/robot_vacuum/splits.json already exists, skipping download.
=== [4/5] Selecting 25 brightest and sharpest Split-A images and quantizing baseline INT8 ===
{
  "scenario": "robot_vacuum",
  "selected_count": 25,
  "output_file": "data/robot_vacuum/lab_calib_25.json",
  "top_3": [
    {
      "file": "000000005529.jpg",
      "path": "data/robot_vacuum/real/000000005529.jpg",
      "mean_luminance": 167.6246,
      "laplacian_variance": 8754.9463,
      "lab_score": 1.7716
    },
    {
      "file": "000000000785.jpg",
      "path": "data/robot_vacuum/real/000000000785.jpg",
      "mean_luminance": 196.2083,
      "laplacian_variance": 1594.284,
      "lab_score": 1.2943
    },
    {
      "file": "000000007977.jpg",
      "path": "data/robot_vacuum/real/000000007977.jpg",
      "mean_luminance": 148.6007,
      "laplacian_variance": 5126.7668,
      "lab_score": 1.1813
    }
  ]
}
{
  "path": "models/baseline_int8.onnx",
  "size_bytes": 2788749,
  "quantized_ops": 102,
  "total_ops": 106,
  "quantized_op_ratio": 0.9623
}
=== [5/5] Evaluating Split-B control F1 and recording baseline budget reference ===
{
  "scenario": "robot_vacuum",
  "split": "B",
  "n_images": 40,
  "passed": true,
  "control_sanity_bar": 0.8,
  "condition_pass_threshold": 0.8337,
  "condition_pass_threshold_raw": 0.8337209302325581,
  "condition_pass_threshold_rule": "baseline_control_raw_f1 - 0.05",
  "note": "baseline.json is written before threshold assertion for inspection; check 'passed' and unrounded 'control_metrics.raw_f1' >= 0.80 for control acceptance.",
  "control_metrics": {
    "f1": 0.8837,
    "recall": 0.8444,
    "precision": 0.9268,
    "retention": 1.0,
    "per_class": {
      "0": 0.892,
      "15": 1.0,
      "41": 0.8,
      "56": 0.8333,
      "57": 1.0,
      "58": 0.6667,
      "62": 0.8571
    },
    "per_size": {
      "small": 0.5385,
      "medium": 0.8507,
      "large": 0.9091
    },
    "n_images": 40,
    "limitation": false,
    "raw_f1": 0.8837209302325582,
    "passed": true,
    "valid_images": 40,
    "excluded_images": 0,
    "collapse_images": 0,
    "ref_objects": 135,
    "orig_objects": 135,
    "tp": 114,
    "fp": 9,
    "fn": 21,
    "ci90_low": 0.8453,
    "ci90_high": 0.9207
  },
  "baseline_int8": {
    "path": "models/baseline_int8.onnx",
    "sha256": "0f927c68172b5d93f3e92f1abd6bbdae671b2700365d89943cb1eccda53926de",
    "size_bytes": 2788749,
    "median_ms": 7.864,
    "quantized_ops": 102,
    "total_ops": 106,
    "quantized_op_ratio": 0.9623,
    "op_types_to_quantize": [
      "Conv"
    ]
  },
  "fp32": {
    "path": "models/fp32.onnx",
    "sha256": "77c7293669cd3bc39fb03a028720edd453d2ed133074a5af1b90fd679b735cf1",
    "size_bytes": 9804172,
    "median_ms": 5.674
  },
  "budget_reference": {
    "ok": true,
    "size_ratio_vs_baseline": 1.0,
    "ms_ratio_vs_baseline": 1.0,
    "ops_ratio_vs_baseline": 1.0,
    "size_ok": true,
    "ms_ok": true,
    "ops_ok": true,
    "reasons": [],
    "size_ratio_vs_fp32": 0.2844,
    "ms_ratio_vs_fp32": 1.386
  }
}
BLOCK_B_EXIT_CODE=0
```
- **Exit Code:** `0`
- **Verdict:** **PASS** (`raw_f1 = 0.8837209302325582 >= 0.80`, `condition_pass_threshold = 0.8337`, `size_bytes = 2788749`, `median_ms = 7.864`, `quantized_ops = 102 / 106`)

---

## Remaining Blocks Status

- **Block C (Condition generation & 7-condition matrix evaluation):** `NOT RUN` (deferred per instructions until Block A+B completion report).
- **Block D (Repairs R1/R2/R3 & Argon diagnosis loop):** `NOT RUN` (deferred per instructions until Block C completes).
- **Block E (Web live process view & redeploy):** `NOT RUN` (deferred per instructions).
- **Block F (Full showcase run on Cloud Run):** `NOT RUN` (deferred per instructions).
- **Block G / S1 (Second scenario `fridge_interior`):** `NOT RUN` (deferred per instructions).


## Independent checkpoint: classical condition matrix

Observed 2026-10-16 14:23 KST. This is the deterministic classical checkpoint, before any subsequent bounded Nano Banana attempt. Fourteen files left by an interrupted generation attempt were restored with the original seed and severity; all fourteen SHA256 values were independently checked against the deterministic reference. No severity, split membership, baseline model, or pass threshold changed.

Command:
```sh
.venv/bin/python scripts/gen_conditions.py --scenario robot_vacuum --evaluate-only
```
Measured fields from the real output (exit 0; compacted for this checkpoint):
```text
Scenario: robot_vacuum | Split: B (n=40)
Control raw F1: 0.8837209302325582
Condition cutoff: 0.8337209302325581
Condition    F1      Recall  Precision  Ref objects  Original objects  Retention
night        0.9057  0.9114  0.9000     79           135               0.5852
glare        0.8962  0.8407  0.9596     113          135               0.8370
rain         0.9107  0.9027  0.9189     113          135               0.8370
fog          0.8026  0.6932  0.9531     88           135               0.6519
occlusion    0.9333  0.9333  0.9333     90           135               0.6667
blur         0.9136  0.8506  0.9867     87           135               0.6444
smudge       0.8909  0.8376  0.9515     117          135               0.8667
```

Known pending correction at this checkpoint: the CLI printed `MODEL_LIMITATION` for fog. Under the 14:08 measurement amendment, fog is a compression FAIL, with a separate informational model-limitation tag. Its numeric `passed` field is false. This checkpoint does not claim that Nano Banana generation, the live repair loop, or deployed UI acceptance has passed.

Additional independent checks:
```text
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_core.py
5 passed in 0.31s

collapse metric: raw_f1=1.0, retention=0.5, collapse_images=1
PASS: collapsed reference images do not add compression false positives.
WARN: only one of two synthetic images emitted progress; excluded-image progress still pending.

detect_head preset: highest model index=23, selected indices=[23], nodes=24
PASS: only the highest model.N is selected.

Independent restored-image SHA verification: 14 / 14 PASS
```
