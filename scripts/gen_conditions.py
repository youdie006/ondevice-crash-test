"""Generate adverse condition images (classical 7-condition pass + Split B Nano Banana pass) and evaluate Split B matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from odct.conditions import (
    CATALOG,
    DEFAULT_CONDITION_SEEDS,
    NANO_BANANA_MODEL,
    generate,
)
from odct.infer import Detector
from odct.metrics import CondMetrics, evaluate_condition, is_passing_f1


PRIMARY_SEVEN_CONDITIONS = ["night", "glare", "rain", "fog", "occlusion", "blur", "smudge"]
DEFAULT_NANO_CONDITIONS = ["night", "glare"]


def _load_scenario_cfg(scenario_id: str) -> dict[str, Any]:
    path = Path("scenarios") / f"{scenario_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Scenario profile not found: {path}")
    return yaml.safe_load(path.read_text())


def _load_splits(scenario_id: str) -> dict[str, Any]:
    path = Path("data") / scenario_id / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found: {path}")
    return json.loads(path.read_text())


def _resolve_pass_threshold(scenario_id: str, explicit_threshold: float | None = None) -> tuple[float, float | None, str]:
    """Resolve the condition pass threshold from baseline.json (raw_f1 - 0.05) or explicit override."""
    baseline_path = Path("showcase") / scenario_id / "baseline.json"
    control_raw_f1: float | None = None
    if baseline_path.exists():
        try:
            bdata = json.loads(baseline_path.read_text())
            c_metrics = bdata.get("control_metrics", {})
            if isinstance(c_metrics, dict) and c_metrics.get("raw_f1") is not None:
                control_raw_f1 = float(c_metrics["raw_f1"])
            if explicit_threshold is None and bdata.get("condition_pass_threshold_raw") is not None:
                return (
                    float(bdata["condition_pass_threshold_raw"]),
                    control_raw_f1,
                    "baseline.json:condition_pass_threshold_raw (baseline_control_raw_f1 - 0.05)",
                )
            if explicit_threshold is None and control_raw_f1 is not None:
                return (
                    float(control_raw_f1 - 0.05),
                    control_raw_f1,
                    "computed:baseline_control_raw_f1 - 0.05",
                )
        except Exception:
            pass

    if explicit_threshold is not None:
        return float(explicit_threshold), control_raw_f1, "cli_explicit_override"

    # Default fallback if baseline.json is missing
    return 0.8337209302325581, control_raw_f1, "fallback_default_0.83372093"


def update_scenario_manifest(
    scenario_id: str,
    condition_ids: list[str],
    evaluation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge per-condition manifests into `data/<scenario>/manifest.json` without overwriting splits.json."""
    scen_dir = Path("data") / scenario_id
    cond_root = scen_dir / "conditions"
    splits_data = _load_splits(scenario_id)

    conditions_block: dict[str, Any] = {}
    all_entries: list[dict[str, Any]] = []

    total_nano_calls = 0
    total_flash_calls = 0
    total_regenerations = 0
    total_fallbacks = 0

    for cid in condition_ids:
        c_manifest_path = cond_root / cid / "manifest.json"
        if not c_manifest_path.exists():
            continue
        c_data = json.loads(c_manifest_path.read_text())
        conditions_block[cid] = c_data
        entries = c_data.get("entries", [])
        all_entries.extend(entries)
        s_all = c_data.get("overall_summary", {})
        total_nano_calls += int(s_all.get("nano_api_calls", 0))
        total_flash_calls += int(s_all.get("flash_api_calls", 0))
        total_regenerations += int(s_all.get("regeneration_count", 0))
        total_fallbacks += int(s_all.get("fallback_count", 0))

    # Build hero strip / crash wall lookup for Split B first image (or highest object count image)
    split_b_files: list[str] = splits_data.get("split_b", [])
    hero_file = split_b_files[0] if split_b_files else ""
    hero_strip: dict[str, str] = {}
    if hero_file:
        hero_strip["original"] = f"data/{scenario_id}/real/{hero_file}"
        for cid in condition_ids:
            cand = f"data/{scenario_id}/conditions/{cid}/{hero_file}"
            if (Path.cwd() / cand).exists():
                hero_strip[cid] = cand

    manifest_path = scen_dir / "manifest.json"
    existing_eval = None
    if manifest_path.exists():
        try:
            old_m = json.loads(manifest_path.read_text())
            existing_eval = old_m.get("split_b_evaluation")
        except Exception:
            existing_eval = None

    payload: dict[str, Any] = {
        "scenario_id": scenario_id,
        "path_root": "project_root",
        "splits_reference": f"data/{scenario_id}/splits.json",
        "seed": splits_data.get("seed", 20261016),
        "split_a_count": splits_data.get("split_a_count", len(splits_data.get("split_a", []))),
        "split_b_count": splits_data.get("split_b_count", len(splits_data.get("split_b", []))),
        "schedule_policy": {
            "split_a": "classical_for_all_7_conditions",
            "split_b_nano_banana": ["night", "glare"],
            "split_b_classical_labelled": ["rain", "fog", "occlusion", "blur", "smudge"],
        },
        "api_call_totals": {
            "nano_banana_calls": total_nano_calls,
            "flash_audit_calls": total_flash_calls,
            "total_gemini_calls": total_nano_calls + total_flash_calls,
            "total_regenerations": total_regenerations,
            "total_fallbacks": total_fallbacks,
        },
        "hero_crash_wall": {
            "split": "B",
            "image_id": "b_001" if hero_file else None,
            "file_name": hero_file,
            "paths": hero_strip,
        },
        "conditions": conditions_block,
        "split_b_evaluation": evaluation_summary if evaluation_summary is not None else existing_eval,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + chr(10))
    return payload


def evaluate_split_b_matrix(
    scenario_id: str,
    condition_ids: list[str],
    fp32_path: str = "models/fp32.onnx",
    int8_path: str = "models/baseline_int8.onnx",
    pass_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate baseline INT8 vs FP32 reference across all 7 conditions on Split B and print the 7-row table."""
    cfg = _load_scenario_cfg(scenario_id)
    allowed_classes = [int(v) for v in cfg.get("classes", {}).values()]
    splits = _load_splits(scenario_id)
    split_b_files: list[str] = splits["split_b"]

    real_dir = Path("data") / scenario_id / "real"
    orig_paths = [real_dir / fname for fname in split_b_files]

    fp32_det = Detector(fp32_path, preprocess="stretch", allowed_classes=allowed_classes)
    int8_det = Detector(int8_path, preprocess="stretch", allowed_classes=allowed_classes)

    cutoff, control_raw_f1, cutoff_source = _resolve_pass_threshold(scenario_id, explicit_threshold=pass_threshold)
    cutoff_display = round(cutoff, 4)

    rows: list[dict[str, Any]] = []
    cond_root = Path("data") / scenario_id / "conditions"

    for cid in condition_ids:
        cond_dir = cond_root / cid
        cond_paths = [cond_dir / fname for fname in split_b_files]
        missing = [str(p) for p in cond_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} condition images for '{cid}' (e.g. {missing[0]})")

        m: CondMetrics = evaluate_condition(fp32_det, int8_det, orig_paths, cond_paths)
        passed = is_passing_f1(m, threshold=cutoff)

        # Read generator provenance from condition manifest
        c_manifest_file = cond_dir / "manifest.json"
        gen_label = f"classical_{cid}"
        nano_count = 0
        fallback_count = 0
        audit_fail_rate: float | str = "N/A"
        if c_manifest_file.exists():
            c_man = json.loads(c_manifest_file.read_text())
            sb = c_man.get("split_b_summary", {})
            nano_count = int(sb.get("nano_banana_count", 0))
            fallback_count = int(sb.get("fallback_count", 0))
            a_pass = int(sb.get("audit_passed_count", 0))
            a_fail = int(sb.get("audit_failed_count", 0))
            if (a_pass + a_fail) > 0:
                audit_fail_rate = round(float(a_fail) / float(a_pass + a_fail), 4)
            if cid in ("night", "glare") and nano_count > 0:
                if fallback_count > 0:
                    gen_label = f"{NANO_BANANA_MODEL} ({nano_count}/40, fb={fallback_count})"
                else:
                    gen_label = f"{NANO_BANANA_MODEL} ({nano_count}/40)"
            else:
                gen_label = "classical (deterministic) in this run"

        status_str = "PASS" if passed else "FAIL"
        limitation_status = "MODEL_LIMITATION" if m.limitation else None
        m_dict = asdict(m)
        m_dict["passed"] = passed
        m_dict["pass_threshold"] = cutoff_display
        m_dict["pass_threshold_raw"] = cutoff
        m_dict["status"] = status_str
        m_dict["limitation_status"] = limitation_status
        m_dict["generator_summary"] = gen_label
        m_dict["nano_banana_b_count"] = nano_count
        m_dict["fallback_b_count"] = fallback_count
        m_dict["audit_failure_rate"] = audit_fail_rate
        rows.append({"condition_id": cid, **m_dict})

    # Print formatted 7-row Split B table
    print("=== Split B 7-Condition Crash-Test Matrix ===")
    print(
        f"Scenario: {scenario_id} | Split: B (n={len(split_b_files)}) | "
        f"Control raw F1: {control_raw_f1 if control_raw_f1 is not None else 'N/A'} | "
        f"Condition Pass Cutoff: >= {cutoff_display:.4f} (raw {cutoff:.6f}, {cutoff_source})"
    )
    header = (
        f"{'Condition':<11} | {'Generator (Split B)':<34} | {'F1':>6} | {'Recall':>6} | "
        f"{'Prec':>6} | {'|R|':>4} | {'|D0|':>4} | {'Retent':>6} | {'Limitation':<16} | {'Status':<6}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        f1_s = f"{r['f1']:.4f}" if isinstance(r["f1"], (int, float)) else str(r["f1"])
        rec_s = f"{r['recall']:.4f}" if isinstance(r["recall"], (int, float)) else str(r["recall"])
        prec_s = f"{r['precision']:.4f}" if isinstance(r["precision"], (int, float)) else str(r["precision"])
        ret_s = f"{r['retention']:.4f}" if isinstance(r["retention"], (int, float)) else str(r["retention"])
        lim_s = "MODEL_LIMITATION" if r["limitation"] else "none"
        print(
            f"{r['condition_id']:<11} | {r['generator_summary']:<34} | {f1_s:>6} | {rec_s:>6} | "
            f"{prec_s:>6} | {r['ref_objects']:>4} | {r['orig_objects']:>4} | {ret_s:>6} | {lim_s:<16} | {r['status']:<6}"
        )
    print("-" * len(header))

    passing_count = sum(1 for r in rows if r["passed"])
    failing_count = len(rows) - passing_count
    limitation_count = sum(1 for r in rows if r["limitation"])

    eval_summary = {
        "scenario_id": scenario_id,
        "split": "B",
        "n_images": len(split_b_files),
        "control_raw_f1": control_raw_f1,
        "pass_threshold": cutoff_display,
        "pass_threshold_raw": cutoff,
        "pass_threshold_source": cutoff_source,
        "passing_conditions": passing_count,
        "failing_conditions": failing_count,
        "limitation_conditions": limitation_count,
        "conditions": {r["condition_id"]: r for r in rows},
    }

    # Save standalone evaluation summary in showcase/<scenario>/conditions_baseline.json for Block D/E
    showcase_dir = Path("showcase") / scenario_id
    showcase_dir.mkdir(parents=True, exist_ok=True)
    (showcase_dir / "conditions_baseline.json").write_text(json.dumps(eval_summary, indent=2) + chr(10))

    return eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate adverse condition datasets (classical + Split B Nano Banana) and evaluate Split B matrix."
    )
    parser.add_argument("--scenario", default="robot_vacuum", help="Scenario ID (default: robot_vacuum)")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Subset of condition IDs to process (default: all 7 scenario conditions)",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Run only deterministic classical generators for the selected conditions and splits.",
    )
    parser.add_argument(
        "--nano-conditions",
        nargs="*",
        default=None,
        help="Run Nano Banana 2.1 edits on Split B for these conditions (default in full run: night glare).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["A", "B"],
        default=["A", "B"],
        help="Which dataset splits to generate in classical pass (default: A B). Nano Banana always runs on B only.",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrency for Nano Banana / Flash API calls (default: 8)")
    parser.add_argument("--max-regenerations", type=int, default=2, help="Max Nano Banana regenerations on invalid/audit fail (default: 2)")
    parser.add_argument("--no-flash-audit", action="store_true", help="Disable Flash audit on Nano Banana Split B edits")
    parser.add_argument("--force-regenerate-nano", action="store_true", help="Do not reuse existing valid Nano Banana outputs")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the Split B 7-condition matrix after generation")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing condition images on Split B without generating")
    parser.add_argument("--pass-threshold", type=float, default=None, help="Explicit condition pass threshold override (default: baseline raw_f1 - 0.05)")
    parser.add_argument("--fp32", default="models/fp32.onnx", help="FP32 ONNX model path")
    parser.add_argument("--int8", default="models/baseline_int8.onnx", help="Baseline INT8 ONNX model path")
    args = parser.parse_args()

    cfg = _load_scenario_cfg(args.scenario)
    scenario_cond_ids = [str(c["id"]) for c in cfg.get("conditions", [])] or list(PRIMARY_SEVEN_CONDITIONS)
    target_conditions = list(args.conditions) if args.conditions else scenario_cond_ids

    if args.evaluate_only:
        eval_summary = evaluate_split_b_matrix(
            scenario_id=args.scenario,
            condition_ids=target_conditions,
            fp32_path=args.fp32,
            int8_path=args.int8,
            pass_threshold=args.pass_threshold,
        )
        update_scenario_manifest(args.scenario, scenario_cond_ids, evaluation_summary=eval_summary)
        return

    splits_data = _load_splits(args.scenario)
    real_dir = Path("data") / args.scenario / "real"
    cond_base_dir = Path("data") / args.scenario / "conditions"

    allowed_classes = [int(v) for v in cfg.get("classes", {}).values()]
    fp32_det = Detector(args.fp32, preprocess="stretch", allowed_classes=allowed_classes)

    seed_map = {str(c["id"]): int(c.get("seed", DEFAULT_CONDITION_SEEDS.get(str(c["id"]), 101))) for c in cfg.get("conditions", [])}

    # Determine execution phases:
    # 1. If --classical-only is set, run classical on requested splits.
    # 2. If --nano-conditions is explicitly passed without --classical-only, run Nano on Split B for those conditions.
    # 3. If neither flag is passed (default full Block C run), run classical on A+B for all target conditions first,
    #    then Nano Banana on Split B for night and glare.
    run_classical_phase = args.classical_only or (args.nano_conditions is None)
    if args.classical_only:
        nano_conds: list[str] = []
    elif args.nano_conditions is not None:
        nano_conds = [c for c in args.nano_conditions if c in CATALOG]
    else:
        nano_conds = [c for c in DEFAULT_NANO_CONDITIONS if c in target_conditions]

    if run_classical_phase:
        selected_paths: list[Path] = []
        if "A" in args.splits:
            selected_paths.extend(real_dir / fname for fname in splits_data["split_a"])
        if "B" in args.splits:
            selected_paths.extend(real_dir / fname for fname in splits_data["split_b"])

        for cid in target_conditions:
            t0 = time.perf_counter()
            c_seed = seed_map.get(cid, DEFAULT_CONDITION_SEEDS.get(cid, 101))
            out_dir = cond_base_dir / cid
            entries = generate(
                args.scenario,
                cid,
                selected_paths,
                out_dir,
                c_seed,
                mode="classical",
                detector=fp32_det,
            )
            dt = time.perf_counter() - t0
            valid_c = sum(1 for e in entries if e.valid)
            print(
                json.dumps(
                    {
                        "phase": "classical",
                        "condition": cid,
                        "images_processed": len(entries),
                        "valid_count": valid_c,
                        "seed": c_seed,
                        "elapsed_s": round(dt, 2),
                    }
                ),
                flush=True,
            )

    if nano_conds:
        split_b_paths = [real_dir / fname for fname in splits_data["split_b"]]
        for cid in nano_conds:
            t0 = time.perf_counter()
            c_seed = seed_map.get(cid, DEFAULT_CONDITION_SEEDS.get(cid, 101))
            out_dir = cond_base_dir / cid
            entries = generate(
                args.scenario,
                cid,
                split_b_paths,
                out_dir,
                c_seed,
                mode="nano_banana",
                detector=fp32_det,
                concurrency=args.concurrency,
                max_regenerations=args.max_regenerations,
                enable_flash_audit=not args.no_flash_audit,
                reuse_existing_nano=not args.force_regenerate_nano,
            )
            dt = time.perf_counter() - t0
            nano_ok = sum(1 for e in entries if e.generator == NANO_BANANA_MODEL)
            fb_c = sum(1 for e in entries if e.used_fallback)
            n_calls = sum(e.nano_api_calls for e in entries)
            f_calls = sum(e.flash_api_calls for e in entries)
            regens = sum(e.regeneration_count for e in entries)
            print(
                json.dumps(
                    {
                        "phase": "nano_banana_split_b",
                        "condition": cid,
                        "split": "B",
                        "images_processed": len(entries),
                        "nano_banana_kept": nano_ok,
                        "classical_fallbacks": fb_c,
                        "regenerations": regens,
                        "nano_api_calls": n_calls,
                        "flash_api_calls": f_calls,
                        "elapsed_s": round(dt, 2),
                    }
                ),
                flush=True,
            )

    eval_summary = None
    if args.evaluate or (not args.classical_only and args.nano_conditions is None):
        eval_summary = evaluate_split_b_matrix(
            scenario_id=args.scenario,
            condition_ids=scenario_cond_ids,
            fp32_path=args.fp32,
            int8_path=args.int8,
            pass_threshold=args.pass_threshold,
        )

    manifest = update_scenario_manifest(args.scenario, scenario_cond_ids, evaluation_summary=eval_summary)
    print(
        json.dumps(
            {
                "manifest_updated": f"data/{args.scenario}/manifest.json",
                "api_call_totals": manifest["api_call_totals"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

