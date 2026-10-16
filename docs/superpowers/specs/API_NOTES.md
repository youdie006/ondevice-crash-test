# API notes (verified 2026-10-16 against official docs and source; nothing installed)

Use these shapes. Where the design or your memory disagrees with this file, this file wins. Re-check only if a call fails.

## 1. Ultralytics YOLO26 export and output layout

- `yolo26n.pt` exists (ultralytics 8.4.178 on PyPI). Asset: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt (5.5 MB).
- YOLO26 is NMS-free by design. The export `nms` flag is tri-state:
  - unset (default): raw one-to-many head, `output0` shape `(1, 84, 2100)` at imgsz 320. Rows 0–3 = decoded xywh center box in input pixels, rows 4–83 = 80 sigmoid class scores. 2100 = 40² + 20² + 10² (strides 8/16/32).
  - `nms=False`: one-to-one head, `(1, 300, 6)` = `[x1, y1, x2, y2, conf, class_id]` in input pixels, top-300, threshold only.
  - `nms=True`: embeds an NMS op. Do not use.
- Recommended: leave `nms` unset and write the numpy decoder + NMS. Use the same layout for FP32 and INT8.

```python
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="onnx", imgsz=320, opset=17)   # -> yolo26n.onnx, output0 (1,84,2100)

# decode
p = out[0].T                       # (2100, 84)
scores = p[:, 4:]
conf = scores.max(1); cls = scores.argmax(1)
keep = conf >= 0.25
xywh = p[keep, :4]; conf = conf[keep]; cls = cls[keep]
xyxy = np.c_[xywh[:,0]-xywh[:,2]/2, xywh[:,1]-xywh[:,3]/2, xywh[:,0]+xywh[:,2]/2, xywh[:,1]+xywh[:,3]/2]
# then class-aware NMS at IoU 0.5, then map back from 320-space to original pixels (undo stretch or letterbox)
```

Sources: https://docs.ultralytics.com/models/yolo26/ , https://docs.ultralytics.com/guides/end2end-detection/ , https://docs.ultralytics.com/modes/export/ , https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py

## 2. onnxruntime quantization (onnxruntime 1.31.0, Python >= 3.11)

```python
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process("fp32.onnx", "fp32_pre.onnx")          # always first; makes graphs match for the debug tools

quantize_static(
    "fp32_pre.onnx", "int8.onnx",
    calibration_data_reader=reader,                        # yields {"images": np.float32 NCHW} like inference
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,                                     # R3 flips this
    calibrate_method=CalibrationMethod.MinMax,             # | Entropy | Percentile (R2)
    nodes_to_exclude=None,                                 # R5: list of node names
    extra_options={"CalibPercentile": 99.999},             # only used by Percentile
)
```

- `CalibrationMethod`: MinMax, Entropy, Percentile (Distribution is fp8-only).
- Other `extra_options`: `CalibMovingAverage`, `CalibTensorRangeSymmetric`, `ActivationSymmetric`, `WeightSymmetric`, `TensorQuantOverrides`.
- Per-node error (skipped today, kept for reference): `onnxruntime.quantization.qdq_loss_debug` has `modify_model_output_intermediate_tensors`, `collect_activations`, `create_activation_matching`, `compute_activation_error` (SQNR dB per tensor). Needs float activations and a fresh reader per call.

Sources: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py , https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/qdq_loss_debug.py , https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

## 3. google-genai Interactions API (google-genai >= 2.3.0; 2.31.0 current)

```python
import base64
from google import genai
client = genai.Client()   # GEMINI_API_KEY from env

r = client.interactions.create(
    model="gemini-4-argon",
    input=[
        {"type": "text", "text": prompt},
        {"type": "image", "data": base64.b64encode(jpeg_bytes).decode(), "mime_type": "image/jpeg"},
        # ... more images, each preceded by a short text label with its id
    ],
    generation_config={"thinking_level": "high", "max_output_tokens": 4096},   # minimal | low | medium | high
    response_format=ACTION_SCHEMA,                                             # verified live on 2026-10-16: pass JSON schema object directly (wrapper {"type": "json_schema", ...} returns HTTP 400)
    previous_interaction_id=prev_id,                                           # None on the first call of a chain
)
text = r.output_text
usage = r.usage            # total_input_tokens, total_output_tokens, total_thought_tokens, total_cached_tokens, total_tokens
interaction_id = r.id
```

- **Live API verification note (2026-10-16):** Passing `response_format={"type": "json_schema", "json_schema": ACTION_SCHEMA}` returns HTTP 400 (`"The value 'json_schema' is not supported for 'type' at 'response_format'. Supported values: audio,text,image,integer,array,boolean,string,object,number,video"`). Passing the JSON schema object directly (`response_format=ACTION_SCHEMA` with root `"type": "object"` and `"additionalProperties": False`) succeeded live on `gemini-4-argon` with `thinking_level="high"`.
- `ImageContent` takes `data` (base64) or `uri`; there is no `file_id`.
- Usage field names are `total_*`, not `prompt_tokens` or `input_tokens`.
- Other kwargs: `system_instruction`, `tools`, `stream`, `store`, `background`, `response_modalities`.
- Do not send `temperature`, `top_p`, `top_k`, `thinking_budget` or `cached_content` to gemini-4-argon.
- Timeouts: set the HTTP client timeout to 15–20 minutes for `thinking_level="high"` calls with many images.

Managed agent (S3 only):

```python
run = client.interactions.create(
    agent="antigravity-preview-09-2026",
    agent_config={"type": "antigravity", "model": "gemini-4-argon"},
    input=goal_text,
    environment="remote",
)
# Prefer a synchronous call with a long timeout. A September 2026 forum report describes 404s when
# polling background runs of this agent with interactions.get; sync calls worked.
```

Sources: https://ai.google.dev/gemini-api/docs/interactions , https://ai.google.dev/api/interactions , https://ai.google.dev/gemini-api/docs/thinking , https://ai.google.dev/gemini-api/docs/antigravity-agent , https://github.com/googleapis/python-genai/tree/main/google/genai/_gaos/types/interactions

## 4. Nano Banana 2.1 image editing

Model id `gemini-nano-banana-2.1` is correct (GA 2026-10-06).

```python
r = client.interactions.create(
    model="gemini-nano-banana-2.1",
    input=[
        {"type": "text", "text": "The same room at night lit by one dim lamp. Keep every object exactly where it is; change only lighting."},
        {"type": "image", "data": base64.b64encode(png_bytes).decode(), "mime_type": "image/png"},
    ],
    response_format=[{"type": "image", "mime_type": "image/png", "aspect_ratio": "4:3", "image_size": "1K"}],
)
img_bytes = base64.b64decode(r.output_image.data)
# resize back to the source width x height before any detection
```

- Limits: up to 14 input images, 7 MB each inline; output sizes `1K`, `2K`, `4K` (uppercase K); aspect ratios 1:1 … 21:9; thinking always on and billed; `temperature`, `seed`, `topP` are rejected (edits are not reproducible, so cache every output).
- Pricing: about $0.0336 per 1K image plus input tokens. Billing is required (no free tier); the lab project has billing.
- Latency: no official figure; third-party logs show roughly 13 s median for 1K. Run with concurrency 8.
- Rate limits are per project and only visible in AI Studio.

Sources: https://ai.google.dev/gemini-api/docs/image-generation , https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1 , https://ai.google.dev/gemini-api/docs/pricing , https://ai.google.dev/gemini-api/docs/rate-limits

## 5. Getting 80 COCO val2017 indoor images without FiftyOne

- FiftyOne's `load_zoo_dataset("coco-2017", ...)` always downloads and extracts `annotations_trainval2017.zip` (253 MB) first, then fetches images individually. Do not use it at the venue.
- Lightweight path:
  1. `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip` (48.6 MB): YOLO-format labels, includes `labels/val2017/*.txt` (class id per line) and the file lists.
  2. Select val2017 ids whose label file contains any class of interest (COCO 80-class indices: person 0, bottle 39, cup 41, bowl 45, banana 46, apple 47, sandwich 48, orange 49, broccoli 50, carrot 51, cake 55, chair 56, couch 57, potted plant 58, tv 62, cat 15, dog 16).
  3. Download each image: `http://images.cocodataset.org/val2017/<12-digit id>.jpg` (about 160 KB each; 120 images is under 20 MB).
  4. Run FP32 and keep the first 80 with at least one detected object of interest; write the seeded A/B split.
- Fallback: `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip` (7 MB; first 128 train2017 images, mixed scenes).

Sources: https://docs.voxel51.com/dataset_zoo/datasets/coco_2017.html , https://github.com/voxel51/fiftyone/blob/develop/fiftyone/utils/coco.py , https://docs.ultralytics.com/datasets/detect/coco/ , https://docs.ultralytics.com/datasets/detect/coco128/
