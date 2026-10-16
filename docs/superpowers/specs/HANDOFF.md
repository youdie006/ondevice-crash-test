# Handoff: On-Device Crash Test

Read `2026-10-16-ondevice-crash-test-design.md` in this folder first, then `API_NOTES.md` (verified call shapes for YOLO26 export, onnxruntime quantization, the Interactions API, Nano Banana and the COCO fetch path; use them as written). This file tells you what to build in what order under the remaining time, what is cut, and what must not happen. Where they disagree: this file wins on scope and order, the design wins on definitions, API_NOTES wins on call shapes.

## Submission (announced 13:28)

Portal: https://cerebralvalley.ai/e/google-deepmind-ai-hackathon (log in with the hackathon registration account). Hard deadline 15:20 KST. Deployed on Google Cloud; everything in English. Open the form early to learn its required fields (description length, video host, cover image, repo and live URLs) and whether a submission can be edited after it is sent; if it can, submit a first version by 14:50 and update it.

## Time

Hacking ends 15:20 KST today. Assume about 2 h 50 min from 12:30. Judging happens on the submission (deployed URL, public repo, README, video, description), so a smaller working thing beats a larger broken thing. Freeze features at 14:50 and spend the rest on the showcase run, README, video and submission.

## Scope for today

Build only goals G1–G5 from the design. Everything below is deferred unless G1–G5 are done and verified before 14:35:

- S1 second scenario (fridge interior): you may fetch its 50 photos during block B and start its condition generation in the background during block D. Add the tab only if check G passes; otherwise ship the profile file tagged "not yet run".
- S2 "Your photo", S3 open repair, S4 certificate, S5 stars: skip.
- Section 10a matrix design: skip for the primary scenario; use the fixed table in section 7.
- Flash audit: required and blocking for Nano Banana outputs on split B (about 160 short calls, run with concurrency); 20% non-blocking sample on split A. If quota or time makes it impossible, fall back to a retention >= 0.5 gate and say so on the page.
- Per-node error statistics: skip; R5 uses the preset groups `first_conv` and `detect_head` only.
- R4 (letterbox) is cut today. Implement R1, R2, R3 and R5 presets. Keep the schema field so the design stays valid.
- Section 12b: the fixed-policy comparison (`scripts/run_policies.py`) and the CLI packaging (`recipe.json`, `profile.json`) are done only after check F passes and before 14:40; the comparison table is worth more than the packaging.
- Diagnosis `claims` verification: implement the check (a short function comparing each claim to the measured metrics) if block D finishes inside its slot; otherwise render claims as "unverified" rather than dropping the field.

## Build order and acceptance

Run each check yourself and paste the real output in the commit message or a `CHECKS.md`. A check that could not run is a failure, not a pass.

- A (15 min) Repo, Dockerfile, FastAPI hello world, deploy to Cloud Run with the flags in section 13. Check: `curl -s -o /dev/null -w "%{http_code}" $URL` prints 200. If `--allow-unauthenticated` is refused by org policy, tell the human immediately (they will ask the Google mentors) and continue with the next block; do not spend more than 5 minutes on it yourself.
- B (25 min) Export detector to ONNX (`nms` flag unset; raw `(1,84,2100)` output, see API_NOTES 1), fetch 80 photos via the labels-zip path in API_NOTES 5 (not FiftyOne) and write the seeded A/B split, `odct/infer.py`, lab-like calibration selection (`scripts/select_lab_calib.py`: 25 brightest and sharpest split-A images), baseline INT8 via `odct/quant.py`, control metrics on split B. Check: control F1 >= 0.90; baseline size, latency and quantized-op count written to `showcase/robot_vacuum/baseline.json` as the budget reference. Commit the calibration rule and severity values before block C starts. If YOLO26 export fails inside 10 minutes, switch to YOLO11n without discussion.
- C (30 min) Classical conditions first (occlusion, blur, smudge, and classical versions of night, glare, rain, fog for split A), then Nano Banana conditions (night, glare, rain, fog) for split B only, 160 edits at concurrency 8 (about 6 minutes; start them as soon as split B exists and keep working), each output resized back to the source size, with the validity gate, the Flash audit on split B, and classical fallback. Evaluate the 7-condition matrix on split B. Check: table printed with object counts, manifest lists generator, seed and audit result per image. Do NOT raise severity or change the baseline if nothing fails: report it and tell the human so the demo script can adapt.
- D (30 min) Repairs R1, R2, R3, R5 presets; budget check relative to baseline INT8 (size <= 1.10x, latency <= 1.10x, quantized ops >= 0.90x); zero-regression gate with first-class ACCEPTED / REJECTED log lines; configuration-hash cache for INT8 models and measurements; Argon diagnosis call on the Interactions API with `previous_interaction_id`, evidence image ids and claims in the JSON; the loop. Check: `python -m odct.loop --live --conditions night --n 20` ends either with night >= 0.90 (control drop <= 0.02, budget OK) or with an honest `needs_human`, and logs interaction ids, token counts and gate verdicts. Wall time <= 150 s. The live chain must be fresh; never seed it with showcase history.
- E (25 min) Web app from section 12: problem sentence and "re-quantization recipe, no training" sentence at the top, before -> after strip, cards with baseline -> final F1 and object counts, one failing card expanded as the four-panel comparison with the verbatim diagnosis, cited evidence images, action and gate verdict, Argon call log on the page, Run live button. Redeploy. Check: `POST /run` on the deployed URL reaches `done` or `needs_human` and cards update.
- F (15 min) Showcase run on Cloud Run for all 7 conditions, save to `showcase/robot_vacuum/`, commit. Landing renders it by default. Rehearse the live run twice.
- Then README (English, honest limitations from section 17, architecture diagram, how to run), 2–3 minute video, description, submit.

Cut lines are in section 16 of the design. Apply them without waiting.

## Module contracts (so parallel agents agree)

```
odct/infer.py
  class Detector(onnx_path, preprocess="stretch"|"letterbox", conf=0.25, iou=0.5, imgsz=320)
    detect(image_path) -> list[Det]          # Det(cls:int, conf:float, box:(x1,y1,x2,y2) original pixels)
    latency_ms(n_runs=20) -> float           # median, warm
odct/quant.py
  quantize(fp32_path, calib_images, out_path, method="minmax"|"percentile"|"entropy",
           per_channel=False, preprocess="stretch", nodes_to_exclude=None) -> QuantResult(path, size_bytes, quantized_op_ratio)
odct/metrics.py
  persisted(d0: list[Det], dc: list[Det], iou=0.5) -> list[Det]
  match(ref, pred, iou=0.5) -> (tp, fp, fn)
  evaluate_condition(fp32: Detector, int8: Detector, originals, condition_images)
      -> CondMetrics(f1, recall, precision, retention, per_class, per_size, n_images, limitation: bool)
odct/conditions.py
  CATALOG: dict[str, Generator]
  generate(scenario_id, condition_id, image_paths, out_dir, seed) -> list[ManifestEntry]   # applies validity gate + fallback
odct/budget.py
  check(candidate: Measured, baseline: Measured) -> (ok: bool, details: dict)
      # Measured(size_bytes, median_ms, quantized_ops); gates: size <= 1.10x, ms <= 1.10x, ops >= 0.90x baseline
      # FP32-relative ratios are reported in details for display only
odct/repairs.py
  apply(action: dict, ctx) -> QuantResult        # maps R1..R5 to quant.quantize params; validates against schema
odct/argon.py
  diagnose(metrics, history, image_pairs, previous_id) -> (action_json, usage, interaction_id)
odct/loop.py
  run(scenario_id, condition_ids, n_images, mode, on_update) -> state dict   # schema lives in app/state.py
app/
  main.py (routes from section 12), state.py (state schema), templates/index.html, static/
```

Ownership if you parallelize: agent 1 `odct/infer.py quant.py metrics.py budget.py repairs.py`; agent 2 `odct/conditions.py scripts/fetch_data.py scripts/gen_conditions.py`; agent 3 `app/`. The human owns `odct/argon.py`, `odct/loop.py`, integration and all checks.

## Non-negotiables

- Every line of code is written today. Do not import or copy any pre-existing company code.
- No employer, project, customer or publication is named or implied anywhere: code comments, README, video, description, UI.
- Only measured numbers appear anywhere. Never display partial, estimated or placeholder values. An empty state says "not yet run".
- Every reported number comes from split B. Split A and its condition images are the only data calibration may touch.
- The baseline calibration rule and the condition severities are fixed and committed before the first evaluation and never retuned to produce a red card.
- The page never says "1M". It shows the measured input token count of each Argon call.
- English for all deliverables. No emojis in code, docs or UI.
- Light theme, system font, no gradients, no neon. Big images, few numbers.
- `GEMINI_API_KEY` via Secret Manager; never commit a key; never print one.
- Commit and push to the public GitHub repo at least every 30 minutes with real commit messages.
- The live demo path never generates images; all condition images are pre-generated and cached.
- Nothing ships automatically. Unresolved conditions end in `needs_human`.
- Do not claim "only Argon can do this". The README says what Argon is used for and why (full-set multimodal diagnosis, chained state, budget-constrained repair).

## Verify in the first 15 minutes

Install the official Gemini skills before writing any Gemini call: `npx skills add google-gemini/gemini-skills --skill gemini-api-dev` (or the Antigravity plugin form). The API shapes are already verified in `API_NOTES.md`; do not re-research them. What is still open and must be checked live: the lab project accepts `--allow-unauthenticated`; one Argon call and one Nano Banana call succeed on the lab key (note the latency); `coco2017labels.zip` downloads at a usable speed. Record what you find in `AGENTS.md`.
