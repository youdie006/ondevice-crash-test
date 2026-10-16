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

## Block B finding (13:50) and the two rules it changed

The first baseline quantized every op and scored control F1 0.11: the final Concat mixes box coordinates (0–320) and class scores (0–1) in one tensor, and one uint8 scale crushes the scores. Verified fix: `op_types_to_quantize=["Conv"]` in `odct/quant.py` for every model, baseline and repairs. With it the lab-like baseline measures 0.884 on clean split B; Percentile 0.913, per-channel 0.935, all-40 calibration 0.898. Rules changed before any condition was evaluated: the condition pass threshold is baseline control F1 − 0.05 (fixed at run start, shown on each card), and the control sanity bar is 0.80. The regression gate (control drop ≤ 0.02, no passing condition falls below the threshold, budget holds) is unchanged. Also measured: INT8 is 0.28× the FP32 size but not faster on this x86 CPU (about 7 ms vs 5 ms); the page states that as a fact, the budget is baseline-relative.

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

## Block E visual brief (added 13:40 after seeing the deployed skeleton)

The deployed landing page is a wall of "not yet run" text. Judges will scroll past it. Block E must make the page visual with real data only:

1. Hero strip, above the cards: one split-B photo shown under all seven conditions as thumbnails in a row (original first), each with FP32 reference boxes in green and INT8 boxes in orange. This is "the crash wall" and it is the first thing a judge sees. It can render as soon as condition images exist, before any repair, labelled "evaluation pending" until metrics arrive.
2. Expanded failing card: the four-panel comparison from design section 12 at large size (at least 320 px per panel on desktop), with the verbatim diagnosis under it.
3. Cards: baseline F1 → final F1 as the biggest text on the card, colored by pass/fail, object count next to it, severity rendered as small chips (gamma 0.35, noise 12), never a raw Python dict. The "Why this condition" line stays.
4. Empty state: never a column of "not yet run". If metrics are missing, show the condition thumbnail and the words "evaluation pending" once per card.
5. Fix the escaping bug: "Before &rarr; After" renders literally. Use the arrow character directly.
6. Do not add charts of fake data, mascots, gradients or animations. Photos with boxes and large measured numbers are the design.

### Live process view (blocks D and E together)

The judge must see the loop happen, not only its result. Block D's loop emits progress; block E renders it. All of it is real: real images, real timestamps, real token counts.

State additions (written by `odct/loop.py` through `on_update`, served by `/state`, polled every 2 s):

```
progress: {
  phase: "evaluating" | "diagnosing" | "repairing" | "re-evaluating" | "gate" | "done" | "needs_human",
  phase_started_at: <iso>, phase_elapsed_s: <float>,
  step: {done: <int>, total: <int>, label: "night 13/20"},
  recent: [ {image_id, condition, thumb_url, fp32_boxes: n, int8_boxes: n, matched: n, missed: n}, ... up to 8, newest first ],
  argon: {status: "sending" | "thinking" | "received", images_sent: n, input_tokens: n|null, output_tokens: n|null, elapsed_s: f},
  action: {type, params_chips: ["Percentile 99.999", "+20 night images"]} | null,
  gate: {verdict: "ACCEPTED" | "REJECTED", reasons: [...], control_delta, budget_ratios} | null
}
```

Loop obligations: write an overlay image (FP32 green, INT8 orange) for every evaluated image to the artifact dir and push a `recent` entry as soon as it is evaluated, not at the end of the condition; update `argon.status` before the call, when the response arrives and after parsing; update `action` when chosen; update `gate` when decided. Keep each `on_update` cheap (write the state JSON, no image work inside the callback beyond the path).

Page obligations (block E):

- A stepper across the top of the run panel: Evaluate, Diagnose, Repair, Re-evaluate, Gate. The current phase is highlighted and shows its real elapsed seconds; finished phases show their duration.
- A filmstrip under the stepper that fills from the left as `recent` grows: thumbnail with boxes, a small label "FP32 3 | INT8 2 | missed 1". During re-evaluation the same strip refills with the repaired model's boxes, so the judge sees boxes reappear on the same photos.
- A diagnosis panel that shows "sending 40 images" → "Argon thinking 00:37" (live counter) → the returned diagnosis text verbatim, with the cited evidence images pulled out of the filmstrip and enlarged.
- An action line with the chips, then a gate banner: green "ACCEPTED" with the control delta and budget ratios, or red "REJECTED: <reason>". Rejected attempts stay visible below as a history.
- Cards update in place when a condition's F1 changes; a card that flips to pass changes color at that moment, no animation beyond the color change.
- Replay: `?replay=showcase` plays the recorded showcase `progress` events at 4x speed from `showcase/<scenario>/events.jsonl` so the landing page is never static. Record every `on_update` payload to that file during the showcase run.

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
