# On-Device Crash Test — Design

Date: 2026-10-16
Event: Google DeepMind AI Hackathon Seoul, Track 2 (Gemini 4 Argon in hard tech)
Build window: 11:20–15:20 (4 h). Judging 15:20–16:00 is done on submissions (description, repo, video, deployed URL) before the Top 4 live demos.

One-liner: *Crash-test your on-device AI before your customers do.*

## 1. Intent

Compressed (INT8) detectors for appliance edge chips pass lab validation on bright, static images and fail silently in the field: at night, in window glare, with rain or steam on the lens, under occlusion, in motion blur. Discovering this normally needs labeled field data that a new product does not have, and fixing it needs a quantization expert.

On-Device Crash Test takes a compressed detector and 80 real photos, synthesizes adverse conditions, measures how far the compressed model drifts from its full-precision parent per condition without any labels, lets Gemini 4 Argon diagnose and repair the failures within a compression budget, proves each repair with a zero-regression gate, and leaves a human in the approval seat.

Primary user: the embedded ML engineer who ships INT8 vision models to appliance SoCs and has no labeled field data for a new product; the one who gets the call when the vacuum stops seeing the cat at night. The tool is released as an open-source pre-deployment check. No employer, project, customer or publication is named anywhere in the deliverables (README, video, description, Q&A). Demo scenarios: a robot-vacuum front camera (primary, full repair loop) and a refrigerator interior camera (second profile, to show the tool is product-agnostic). Scenarios are data-driven profiles: adding one is a YAML file plus 80 photos.

## 2. Why this design scores

- Working demo and verification loop: red card → Argon diagnosis → repair → re-evaluation → green card, live, under 2.5 minutes.
- Argon is load-bearing: every diagnosis sees all failing image pairs, the previous iteration's pairs and the full iteration history in one long multimodal context (the measured input token count is shown per call; the page never says "1M"), chained server-side; repairs are reasoned under a hard compression budget, and every diagnosis must cite the images and metrics it relied on, which the evaluator checks; an optional managed-agent sandbox writes its own repair when the closed set runs out. Argon also designs each product's crash-test matrix from the scenario description and sample photos, choosing from a catalog of condition generators and writing the edit prompts, so the tool is product-agnostic without hand-written condition lists.
- Companion models with clear roles: Nano Banana synthesizes weather and lighting conditions; Flash independently audits that edits preserved the objects.
- Problem and impact: a vivid, specific user (embedded ML engineers shipping INT8 vision models to appliance SoCs with no labeled field data), any ONNX detector, zero labels required, MIT license, a CLI that runs in five minutes.
- Originality: label-free oracle, compression-failure vs model-limitation split, budget-constrained repair with zero-regression gate, provenance of every repair.
- Safety: nothing ships automatically; unresolved conditions go to human review.

## 3. Goals, non-goals, anti-goals

Goals (must exist by 15:20):

- G1 Deployed Cloud Run app whose landing page shows showcase results for 7 conditions.
- G2 Live run on 1 condition × 20 images: evaluate → diagnose → repair → re-evaluate, visible, ≤ 2.5 min.
- G3 At least one condition fails at baseline and passes after repair with zero regressions on the control set and on previously passing conditions.
- G4 Every Argon call logged with interaction id, input/output tokens, latency, and shown in the UI.
- G5 Public GitHub repo, English README, demo video, project description.

Stretch, in priority order:

- S1 Second scenario, refrigerator interior: its own 50 photos and Argon-designed matrix (≥ 5 conditions), evaluate-only showcase plus one repair iteration if time allows. Shown as a second tab on the landing page. Timebox 25 minutes of hands-on work; generation and evaluation run in the background while other blocks proceed.
- S2 "Your photo" mode: a judge uploads a photo; one-image night comparison in ≤ 30 s. The judge-panel review rated this above S1 for the 40-minute scorers; whichever of S1 and S2 fits the remaining time ships first.
- S3 Managed-agent open repair, showcase only.
- S4 Crash-test certificate page.
- S5 Star ratings on cards.

Non-goals: no training or fine-tuning; no real device or NPU; no auth or multi-user; no persistence beyond /tmp plus the committed showcase; no non-ONNX backends; no annotations anywhere.

Anti-goals (failure even if it runs): any fabricated or estimated number presented as measured; padding the model list; claiming "only Argon can do this"; a repair that silently un-compresses the model; reusing any company SDK code; naming the author's employer, project, customers or publications; live demo that depends on live image generation; dark-neon dashboard styling.

## 4. Constraints

- 4 hours, one person (confirmed), parallel Antigravity agents allowed with exclusive file ownership.
- No references to the author's employer, project, customers or publications in any deliverable; the user story stays generic.
- Google AI tooling must be used to build; deployment on Google Cloud; all deliverables in English.
- All demo code written during the event; open-source libraries are fine.
- The gcplab account and project are deleted after the event: push every 30 minutes.
- Venue Wi-Fi and unknown API quotas: pre-generate and cache all condition images; the live path never generates images.
- Cloud Run CPU only (4 vCPU, 8 GiB). One live iteration ≤ 2.5 min.

## 5. Scenarios and data

A scenario is a profile file `scenarios/<id>.yaml` with: a one-paragraph product description written for Argon (where the camera sits, what it must detect, what goes wrong in the field), the classes of interest, the FiftyOne query used to fetch photos, and the condition matrix (either fixed or produced by the matrix-design call in section 10 and cached). Everything downstream (conditions, metrics, repairs, UI) is scenario-agnostic.

Primary: `robot_vacuum`. A robot vacuum must detect people, pets and furniture while driving under couches (dark), past windows (glare), after mopping (fogged or smudged lens), and while moving (blur). Classes (COCO): person, cat, dog, chair, couch, potted plant, bottle, cup, tv.

Second (S1): `fridge_interior`. A camera inside a refrigerator door must recognize food items for inventory; it faces condensation when the door opens in summer, frost, glare from the interior LED on glossy packaging, a blue color cast, dim light when the bulb fails, and items stacked in front of each other. Classes (COCO): bottle, cup, bowl, apple, orange, banana, broccoli, carrot, sandwich, cake. Photos: FiftyOne query for `refrigerator` images (open fridges with visible contents); if fewer than 40 usable images, the scenario is dropped from the showcase and left in the repo as a profile with a "not yet run" tag. It is never shown with partial or estimated numbers.

Images: 80 real photos per scenario from COCO val2017 containing at least one class of interest. Fetch path: download `coco2017labels.zip` (48.6 MB, YOLO-format labels incl. val2017 file names) from the ultralytics assets release, select val2017 ids whose labels contain the classes of interest, download those images individually from `http://images.cocodataset.org/val2017/<id>.jpg`, and keep the first 80 on which FP32 detects at least one object of interest. Do not use the FiftyOne zoo at the venue: it always pulls the 253 MB annotation archive first. Fallback: ultralytics `coco128.zip` (7 MB, train2017 images, mixed scenes) for the primary scenario. Stored at `data/<scenario>/real/*.jpg`, long side 640, then split by a seeded shuffle into split A (40 images: calibration and repairs may use these and their condition images) and split B (40 images: every reported number is computed here; nothing from B is ever used for calibration). The unedited split-B images are the control set. No annotation files are downloaded or used.

## 6. Models

- Detector: YOLO26n (Ultralytics 8.4.x, `yolo26n.pt`) exported with `YOLO("yolo26n.pt").export(format="onnx", imgsz=320)` and the `nms` flag left unset. That yields the raw one-to-many head: `output0` of shape `(1, 84, 2100)` at 320 (rows 0–3 = xywh center box in input pixels, rows 4–83 = sigmoid class scores; 2100 = 40² + 20² + 10²). Do not pass `nms=False`: on YOLO26 that selects the one-to-one `(1, 300, 6)` head, which is also usable but changes the decoder; pick one layout and use it for FP32 and INT8 alike. Fallback: YOLO11n, same raw layout. Stored at `models/fp32.onnx`. See `API_NOTES.md`.
- Inference: onnxruntime CPU. Preprocessing modes `stretch` (baseline) and `letterbox`. Decode (`p = out[0].T`, conf = max class score, class = argmax, xywh → xyxy) plus class-aware NMS in numpy (IoU 0.5, conf 0.25, max 100 boxes).
- Quantization: `onnxruntime.quantization.quant_pre_process` then `quantize_static`, `quant_format=QDQ`, `activation_type=QUInt8`, `weight_type=QInt8`, `calibrate_method` MinMax | Entropy | Percentile (`extra_options={"CalibPercentile": 99.999}`), `per_channel`, `nodes_to_exclude`. Baseline config is the recipe a team actually ships after lab validation, fixed by rule before any condition is evaluated and committed before conditions are generated: MinMax calibration, per-tensor, stretch preprocessing, no exclusions, calibration set = the 25 "lab-like" images of split A (highest mean luminance and Laplacian sharpness; rule in `scripts/select_lab_calib.py`). The landing page states the rule in one sentence ("quantized the way teams do: calibrated on bright, sharp lab images"). The rule is never changed after the first evaluation.
- Gemini:
  - `gemini-4-argon` via the Interactions API: `thinking_level="high"`, JSON output, `previous_interaction_id` chaining, usage captured.
  - `gemini-nano-banana-2.1`: image editing for night, glare, rain, fog.
  - `gemini-3.8-flash`: independent audit of edited images.
  - `antigravity-preview-09-2026` managed agent (S3 only).
- SDK: `google-genai >= 2.3.0` (2.31.0 current). Interactions calls take `input=[{"type":"text",...},{"type":"image","data":<base64>,"mime_type":...}]`, `generation_config={"thinking_level": ...}`, `previous_interaction_id`, `response_format` for structured output; usage is `r.usage.total_input_tokens / total_output_tokens / total_thought_tokens`. Install `gemini-skills` (`gemini-api-dev`) in the coding agents before writing any Gemini call. Exact shapes and sources are in `API_NOTES.md`.

## 7. Conditions

Conditions are built from a catalog of generator primitives: Nano Banana edit templates (lighting, weather, lens state, with a mandatory "keep every object exactly where it is" clause) and deterministic classical operators (gamma, noise, motion kernel, rectangle occlusion, local blur blobs, haze blend, radial highlight, white-balance shift, droplet and frost overlays). A scenario's matrix picks from this catalog. Each condition has an id, generator, fixed severity parameters, a seed, and a validity gate.

Primary scenario `robot_vacuum`, seven conditions:

| id | name | generator | fallback / notes |
|---|---|---|---|
| night | Night / low light | Nano Banana: "the same room at night lit by one dim lamp; keep every object exactly where it is" | gamma 0.35 + Gaussian noise σ 12 |
| glare | Window backlight | Nano Banana: "strong sunlight from a window behind the objects with lens flare; keep every object in place" | radial highlight + contrast reduction |
| rain | Rain on lens | Nano Banana: "water droplets on the camera lens; keep every object in place" | droplet overlay |
| fog | Fog / steam on lens | Nano Banana: "lens fogged by steam; keep every object in place" | white alpha blend + blur |
| occlusion | Partial occlusion | classical: two gray rectangles covering 15–25% of the image, seeded | deterministic |
| blur | Motion blur | classical: horizontal motion kernel, 15 px | deterministic |
| smudge | Lens smudge | classical: local Gaussian blobs + slight haze | deterministic |

Second scenario `fridge_interior` (S1), six conditions, default matrix if the matrix-design call is skipped:

| id | name | generator | fallback / notes |
|---|---|---|---|
| condensation | Condensation on lens | Nano Banana: "lens covered in fine condensation droplets after the door opened on a humid day; keep every item in place" | droplet overlay + mild blur |
| frost | Frost | Nano Banana: "thin frost on packaging and shelf edges; keep every item in place" | white speckle overlay |
| led_glare | LED glare on packaging | Nano Banana: "harsh interior LED reflecting off glossy packaging; keep every item in place" | radial highlights on bright regions |
| color_cast | Blue LED color cast | classical: white-balance shift toward blue | deterministic |
| dim | Bulb failure | classical: gamma 0.4 | deterministic |
| stacking | Items stacked in front | classical: rectangles 15–25% at lower half | deterministic |

Validity gate per image, all generators: run FP32 on the original (D0) and on the condition image (Dc). Persisted objects P = { d in D0 : exists d' in Dc with same class and IoU ≥ 0.5 }. An image is valid if |D0| ≥ 1, |P| ≥ 1, and, for Nano Banana outputs, the Flash audit passes. Retention |P| / |D0| is not a validity criterion; it is reserved for the model-limitation tag in section 8 so that the tag can actually fire. Invalid Nano Banana outputs are regenerated up to two times, then the fallback generator is used.

Independent audit: Flash receives every Nano Banana output for split B (original + edited) and answers "same objects in the same positions: yes/no, list differences". A "no" is blocking (regenerate, then fallback). Split-A edits are audited on a random 20% sample, non-blocking, logged. Audit results go into the manifest and each card shows its audit failure rate. If Flash quota or time makes the full split-B audit impossible, the fallback gate is retention ≥ 0.5 and the page says so.

Generation budget: Nano Banana edits take roughly 13 s each at 1K output and cannot be seeded, so they are made only for split B (4 conditions × 40 images = 160 edits, concurrency 8, about 6 minutes) through `client.interactions.create(model="gemini-nano-banana-2.1", input=[text, image], response_format=[{"type":"image","mime_type":"image/png","image_size":"1K","aspect_ratio":<closest to source>}])`; the output is resized back to the source width and height before any detection so boxes stay comparable. Split-A condition images, which only feed calibration, use the classical generators. This also means a repair is never calibrated on the same generator it is evaluated against, and the page says so.

All condition images are pre-generated and cached at `data/<scenario>/conditions/<id>/<image>.jpg` with `data/<scenario>/manifest.json` recording generator, seed, validity, and audit result per image.

Severity: parameters are fixed in the profile before the first evaluation and displayed on each card. They are not raised afterwards to produce a red card. If the baseline passes every condition, that is the reported result: the demo then shows the audit and a live run on the hardest condition, and the README says the recipe was robust for this model.

## 8. Oracle and metrics (label-free)

Per image, per condition:

- Reference R = P, the FP32 objects that persist on the condition image. Using persisted objects rather than raw FP32 detections keeps FP32 hallucinations under fog out of the reference.
- INT8 detections Q on the condition image.
- Greedy matching at IoU ≥ 0.5 with the same class. TP = |Q matched to R|, FN = |R| − TP, FP = |{ q in Q : q matches nothing in Dc }|, so INT8 is not penalized for agreeing with an FP32 detection that did not persist.

Per condition, on split B only: micro F1 = 2TP / (2TP + FP + FN) over all images, plus recall, precision, per-class F1, recall by box size bucket (small < 32², medium, large at 320 scale), the object count |R| behind every number, and a bootstrap 90% interval over images (200 resamples) when time allows. With about 100 objects, 0.89 vs 0.91 is noise; the page shows the count so nobody has to guess.

- Pass threshold: F1 ≥ 0.90.
- Control: same metric on unedited split-B images with R = D0. Baseline control F1 must be ≥ 0.90; below that the pipeline (preprocessing, decode, export) is assumed broken and is debugged. The recipe is not changed to pass this bar.
- Model-limitation tag: retention = Σ|P| / Σ|D0| per condition. If retention < 0.7, the condition is tagged "model limitation: FP32 also degrades", shown separately, and does not count against the compression verdict.
- Compression budget, hard constraints on any accepted repair, defined relative to the baseline INT8 model measured on the serving machine (CPU latency does not predict the appliance NPU, and an FP32-relative latency target can be unreachable for a nano model on x86): file size ≤ 1.10 × baseline INT8 size; median latency ≤ 1.10 × baseline INT8 latency (20 warm runs); quantized ops ≥ 0.90 × the baseline's quantized-op count. FP32-relative size and latency ratios are displayed as measured facts, never used as gates. The budget exists so that a repair cannot win by handing back the compression.
- Zero-regression gate: a repair is accepted only if control F1 drops by ≤ 0.02 from baseline, no previously passing condition falls below 0.90, and the budget holds. Gate outcomes are first-class log entries ("ACCEPTED", "REJECTED: control F1 −0.03", "REJECTED: latency 1.14× baseline") shown in the UI. Rejected repairs stay in the log as failed attempts.
- Determinism: fixed seeds, fixed ORT session options; identical inputs give identical numbers.

## 9. Repair actions (closed set for the safe path)

Each returns a new INT8 model and a provenance record (config, calibration image list, parent iteration).

- R1 recalibrate: calibration set = lab-like split-A images + split-A condition images of the targeted condition(s), with optional duplication weight 1–3×. Duplication is admitted only when the method is Percentile or Entropy; under MinMax it is a mathematical no-op and the schema rejects it.
- R2 calibration method: MinMax → Percentile(99.99) → Entropy.
- R3 per-channel weight quantization.
- R4 preprocessing: stretch → letterbox, applied to both calibration and inference. The FP32 reference and the control baseline are recomputed under the candidate's preprocessing (cached per mode) and the UI labels the mode, so R4 is compared against the right reference.
- R5 node exclusion: exclude nodes from quantization using preset groups `first_conv`, `detect_head`, or `top_k_error(k)` from per-node error statistics, limited by the budget.

Combinations are allowed. Parameters are validated against a schema; invalid actions are rejected and logged. The schema documents what each knob does (how MinMax, Percentile and Entropy set activation ranges; what per-channel changes; what exclusion costs in budget) so Argon reasons from semantics rather than from labels.

Per-node error statistics: computed once per INT8 model with onnxruntime's QDQ loss-debug utilities on 10 images (FP32 vs INT8 activations), top 20 nodes by error, included in Argon's input. If the utility is unavailable in the pinned version, R5 keeps only the preset groups.

## 10. Argon protocols

### 10a. Matrix design (once per scenario, offline, cached)

Input: the scenario description from the profile, the class list, 10 sample photos, and the generator catalog with its parameter ranges. Output JSON: an ordered list of 5–7 conditions, each with catalog primitive, Nano Banana prompt (if applicable), severity parameters within range, and a one-sentence reason tied to the product ("camera is 8 cm above the floor, so under-furniture darkness comes first"). The result is written into the scenario profile and shown on the landing page under "Why these conditions". The fixed tables in section 7 are the fallback and the acceptance baseline; the builder may ship the fixed tables and run matrix design only for the second scenario if time is short. Validity gates and fallbacks apply to Argon-designed conditions exactly as to fixed ones.

### 10b. Diagnosis and repair loop

One Interactions chain per run. Each iteration sends:

- Text: goal, budget, thresholds, metrics JSON (per condition, class, size bucket), control metrics, per-node error top 20, full history of actions with outcomes and rejection reasons, allowed-action schema.
- Images: for every currently failing condition, every split-B pair (original with FP32 boxes; condition image with FP32 reference boxes in green and INT8 boxes in orange), plus the same pairs from the previous iteration so Argon compares before and after visually, capped at 400 images per call; 5 pairs per passing condition; 10 control pairs. In live mode (20 images) everything is sent. Every image carries an id the diagnosis can cite.
- Required JSON output:

```json
{
  "diagnosis": [{"condition": "night", "pattern": "...", "cause": "compression | model_limitation | edit_artifact",
                 "evidence_images": ["b_017", "b_032"],
                 "claims": [{"condition": "night", "metric": "recall_small", "comparator": "<", "value": 0.6}]}],
  "action": {"type": "R1 | R2 | R3 | R4 | R5 | none", "params": {}, "targets": ["night"]},
  "expected_effect": "...",
  "stop": false,
  "needs_human": false,
  "rationale": "..."
}
```

- Loop: at most 4 iterations. Stops when all non-limitation conditions pass, when the action is `none`, or when the budget cannot be met. Unresolved → `needs_human = true`; the UI shows "Needs human review" with Argon's summary and Approve/Reject controls that only record a decision. Nothing is shipped.
- Logging per call: interaction id, input and output tokens from usage, latency, thinking level; shown verbatim in the UI iteration log.
- Guardrails: actions validated against the schema; Argon executes no code in the safe path.
- Diagnosis verification: every claim is checked against the measured metrics and marked verified or unverified in the UI; cited evidence images are rendered under the diagnosis. Unverified claims lower a confidence field and count toward `needs_human`. This is the difference between "the model explained" and "the explanation was checked".
- Caching: INT8 models and their measurements are cached by configuration hash. A live iteration that reaches an already measured configuration reuses it and the log line says "cached from an identical configuration". The live Argon chain is always fresh; it is never seeded with showcase history.
- Live run: if iteration 1 is rejected, iteration 2 runs if time allows; a live run that ends in `needs_human` is shown as such. The presenter never restarts a run to get a green card. If Argon's first move is the obvious recalibration and the gate rejects it, that is an honest rejected iteration and it stays on screen.

## 11. Open repair (S3, showcase only)

If the safe path ends in `needs_human`, package `harness/` (fp32.onnx, control and condition images, `eval.py`, `quant_config.py`, budget) and start `antigravity-preview-09-2026` with `agent_config={"type":"antigravity","model":"gemini-4-argon"}`, `environment="remote"`, as a synchronous call with a long timeout (a September 2026 report describes 404s when polling background runs of this agent), goal: "Edit only `quant_config.py` (calibration composition, method, per-channel, exclusion list) until `eval.py` reports every condition ≥ 0.90 with zero regressions and the budget satisfied. Do not modify `eval.py`." The returned diff is applied locally and re-verified by our own evaluator; the sandbox's self-report is never trusted. Logged as a distinct iteration type.

## 12. Web app

Single FastAPI service.

- `GET /` landing, legible without a click: one sentence of the problem in the user's words ("the detector passed lab validation and fails under couches at night") and one sentence that says what a repair is ("a re-quantization recipe: calibration set, method, granularity, exclusions. No training."); scenario tabs (Robot vacuum | Fridge interior, the second only if its showcase exists); a before → after strip ("Baseline INT8 fails k/7 conditions. After n Argon iterations: m/7 pass. Control F1 x → y. Size and latency vs baseline INT8 and vs FP32, measured."); condition cards showing baseline F1 → final F1 with the object count, the action that fixed it, the iteration number, the fixed severity and the audit failure rate, with surviving red cards labeled "model limitation" or "needs human review"; a "Why these conditions" line per scenario; one failing card expanded as four panels (original + FP32; condition with FP32 green vs baseline INT8 orange; the same with repaired INT8; per-class and per-size deltas) with the verbatim one-line diagnosis, its cited evidence images, the chosen action and the gate verdict; the Argon call log (interaction id, input and output tokens, latency, thinking level) on the page, not in a tab; the fixed-policy comparison table when it exists (section 12b); "Run live (about 2 min)" button; "Your photo" (S2). Default content is the showcase from `showcase/<scenario>/state.json`.
- `POST /run` `{conditions, n_images, mode}` → starts a background thread, returns `run_id`. One run at a time.
- `GET /state?run_id=` → phase (`evaluating | diagnosing | repairing | re-evaluating | done | needs_human`), per-condition metrics per iteration, iteration log (Argon JSON + usage), artifact URLs.
- `GET /artifacts/<path>` static files.
- `POST /photo` (S2): upload → FP32 detect → deterministic night filter immediately, Nano Banana edit asynchronously replacing it when ready → INT8 detect → pair image + agreement. Target 30 s.
- `GET /certificate?run_id=` (S4).
- UI: vanilla HTML/JS, 2 s polling, light theme, system font, no gradients, large images, only measured numbers, Argon diagnosis text shown verbatim.

## 12b. Fixed-policy comparison and tool packaging

Comparison: two scripted policies run through the same evaluator and gates, "always R1 on failing conditions" and "kitchen sink: R1 + Percentile + per-channel", next to the Argon run. The landing page shows the three-row table (conditions passing, control delta, budget, iterations, wall time) whatever it says. If Argon only ties the fixed recipes, the README says so and positions Argon's value as diagnosis, fewer iterations and the stop/needs-human decision. This table is the honest answer to "why not just loop the knobs".

Packaging: `python -m odct crash-test model.onnx photos/ --scenario robot_vacuum --profile profile.json` produces `recipe.json` (calibration image list, method, granularity, exclusions), the repaired ONNX and an HTML report. `profile.json` declares the target's quantization constraints (allowed granularity, activation type, op types that may stay float, budget ratios) and masks repairs accordingly; the README calls these "constraints from the profile, not measurements on the device". Human approval remains the step that writes the recipe.

## 13. Deployment

- Cloud Run, `asia-northeast3`, 4 vCPU, 8 GiB, concurrency 1, `--no-cpu-throttling`, `--min-instances 1`, `--timeout 3600`, unauthenticated.
- `GEMINI_API_KEY` from Secret Manager via `--set-secrets`; plain env var only as a last resort.
- If the lab project's org policy refuses the `allUsers` invoker binding (`--allow-unauthenticated` fails), ask the Google mentors for the exception at once and keep building locally meanwhile. Last resort: deploy the same container to a personal GCP project with a personal Gemini key, which still satisfies "deployed on Google Cloud"; say so in the README. If Cloud Build or Artifact Registry is blocked, build locally with Docker and push to Artifact Registry in whichever project allows it.
- Container `python:3.12-slim`, pinned requirements; `models/`, `data/` and `showcase/` baked into the image; image under 1 GB.
- The same app runs locally with `uvicorn` as the stage fallback; the deployed URL stays up for judges regardless.

## 14. Repository layout

```
ondevice-crash-test/
  README.md  LICENSE (MIT)  AGENTS.md  Dockerfile  requirements.txt
  app/            main.py  state.py  templates/  static/
  odct/           infer.py  quant.py  metrics.py  conditions.py  repairs.py
                  argon.py  loop.py  budget.py  nodeerr.py
  scripts/        export_model.py  fetch_data.py  design_matrix.py  gen_conditions.py  run_showcase.py
  scenarios/      robot_vacuum.yaml  fridge_interior.yaml
  models/         fp32.onnx
  data/<scenario>/   real/  conditions/<id>/  manifest.json
  showcase/<scenario>/   state.json  images/
  docs/superpowers/specs/2026-10-16-ondevice-crash-test-design.md
  tests/          smoke tests
```

## 15. Build order and acceptance checks

Checks are run by the supervisor; an agent's own report is not acceptance.

- A Deploy skeleton: `curl -s -o /dev/null -w "%{http_code}" $URL` → `200`.
- B Model, data, baseline: `python scripts/fetch_data.py --scenario robot_vacuum` (80 images, split file written) `&& python scripts/select_lab_calib.py && python -m odct.quant --baseline && python -m odct.metrics --control` → control F1 on split B ≥ 0.90; baseline size, latency and quantized-op count recorded as the budget reference.
- C Conditions and matrix: `python scripts/gen_conditions.py && python -m odct.loop --evaluate-only` → 7-row table, at least one row F1 < 0.90, manifest lists generator per image.
- D Loop: `python -m odct.loop --live --conditions night --n 20` → night ≥ 0.90 or an honest `needs_human`; for any accepted repair control drop ≤ 0.02 and budget OK; gate verdicts logged; Argon log with ids and tokens; claims marked verified or unverified; wall time ≤ 150 s locally.
- E Web: `POST /run` on the deployed URL → `/state` reaches `done`; cards update.
- F Showcase: `scripts/run_showcase.py --scenario robot_vacuum` on Cloud Run → `showcase/robot_vacuum/state.json` committed; landing renders it.
- H (12b) Fixed-policy comparison: `scripts/run_policies.py` → three-row table committed with the showcase; only if F is done by 14:40.
- G (S1) Second scenario: `scripts/fetch_data.py --scenario fridge_interior` yields ≥ 40 images; `scripts/design_matrix.py --scenario fridge_interior` writes a matrix with reasons; `run_showcase.py --scenario fridge_interior --evaluate-only` produces a table; the tab appears only when this check passes.

## 16. Risks and cut lines

- Baseline fails nothing → report it as the result; do not retune severity or the baseline. The live demo then runs the hardest condition and the narrative is the audit. Check this at the end of block C so the video script can adapt.
- Baseline INT8 not faster than FP32 on Cloud Run CPU → display the fact; gates are baseline-relative, so repairs stay acceptable.
- Submission mechanics (deadline, video length, description length) were not published at design time → assume 15:20 sharp, video ≤ 3 min on YouTube unlisted, description ≤ 300 words; code freeze 14:50 because a deploy cycle takes 5–8 min.
- Nano Banana moves objects → validity gate and fallback; worst case all seven conditions classical, Nano Banana used only in S2 ("Your photo").
- Nano Banana latency (about 13 s per edit) or quota at the venue → split-B-only generation with concurrency 8; if the 160 edits are not done by the end of block C, finish the matrix with classical generators for the missing conditions and label them.
- Argon latency or quota with hundreds of images → per-call image cap; live mode is small by design.
- YOLO26 export trouble → YOLO11n.
- QDQ debug utility missing → preset exclusion groups only.
- Cloud Run too slow → live `n_images` 12; showcase precomputed.
- UI unfinished at T+2:50 → JSON table plus a matplotlib grid page.
- Fridge photos unusable (fewer than 40 with visible contents) or its generation not finished by T+3:00 → ship the profile file only, tagged "not yet run"; never show partial numbers.

## 17. Honesty statement (README)

Adverse conditions are synthetic. Unedited controls are always shown. Conditions where the full-precision model also degrades are reported as model limitations, not compression failures. All numbers are measured on Cloud Run CPU; there are no device or NPU measurements. Repairs are constrained to keep the compression benefit.

## 18. Left to the builder

Exact prompts; trimming the class list; Flash audit sampling; UI layout details; NMS implementation; overlay rendering; polling vs SSE; container base image details.

## 19. Decisions resolved with the author

- Scenarios: robot-vacuum front camera as primary (every condition has a physical cause in that product, all classes are COCO classes, no medical or safety-advice connotation) and refrigerator interior as the second profile (author's request; proves the tool is product-agnostic and lets Argon's matrix design show different conditions for a different product). Fall-detection cameras were rejected: the class is not in COCO and the health framing invites the human-in-the-loop penalty.
- Public statements: none. No employer, project, customer or publication is named or implied in any deliverable. The user story is generic (section 1).
- Name: On-Device Crash Test. Repo slug `ondevice-crash-test`, package `odct`.
- Team: one person. Parallel Antigravity agents own `odct/quant+metrics+repairs`, `odct/conditions`, and `app/` respectively; the author owns `odct/argon.py`, `odct/loop.py`, integration and acceptance checks.
- Review-driven changes (builder check and judge-panel review, 12:35): split A/B hold-out so nothing reported was ever calibrated on; validity gate decoupled from the model-limitation tag, with a blocking Flash audit on split B; false positives counted against all FP32 detections on the condition image; budget re-based on the baseline INT8 measured on the serving machine; baseline calibration fixed by a documented lab-like rule, severity fixed, no post-hoc retuning; diagnoses must cite evidence images and machine-checkable claims, which the evaluator verifies; configuration-hash caching allowed, live chain never seeded; landing page leads with before → after and a four-panel card; fixed-policy comparison and CLI packaging added as 12b; org-policy fallback for deployment; R4 compares against a same-preprocessing reference.

## 20. Verified facts and what is still open

Verified against official docs and source on 2026-10-16 (details and sources in `API_NOTES.md`): YOLO26n export and raw output layout; `quantize_static` signature, calibration methods and `qdq_loss_debug`; Interactions API parameter names and usage fields; Nano Banana model id, call shape, limits and pricing; the FiftyOne annotation-archive behaviour.

Still open at the venue, in this order:

- The lab project accepts the `allUsers` invoker binding: deploy the hello world before anything else.
- Nano Banana and Argon quota and real latency on the lab key: one call each in block A.
- Download speed for `coco2017labels.zip` and 80 images on venue Wi-Fi.
