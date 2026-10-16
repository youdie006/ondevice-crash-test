"""Repair actions (R1, R2, R3) with schema validation, Split-A-only guardrails, and rollback state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from odct.quant import QuantResult, quantize

ALLOWED_ACTIVE_TYPES = {"R1", "R2", "R3"}
FORBIDDEN_TYPES = {"R4", "R5"}
ALLOWED_METHODS = {"minmax", "percentile", "entropy"}


class RepairRejectedError(ValueError):
    """Raised when a proposed repair action violates schema, scope, or calibration rules."""


@dataclass(frozen=True)
class RepairConfig:
    method: str = "minmax"
    per_channel: bool = False
    preprocess: str = "stretch"
    r1_targets: tuple[str, ...] = ()
    r1_weight: int = 1
    r1_max_per_condition: int | None = None
    nodes_to_exclude: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "per_channel": self.per_channel,
            "preprocess": self.preprocess,
            "r1_targets": list(self.r1_targets),
            "r1_weight": self.r1_weight,
            "r1_max_per_condition": self.r1_max_per_condition,
            "nodes_to_exclude": list(self.nodes_to_exclude),
            "op_types_to_quantize": ["Conv"],
        }


@dataclass
class RepairContext:
    scenario_id: str = "robot_vacuum"
    fp32_path: Path = Path("models/fp32.onnx")
    baseline_int8_path: Path = Path("models/baseline_int8.onnx")
    out_dir: Path = Path("/tmp/odct_cache/candidates")
    data_root: Path = Path("data")
    current_config: RepairConfig = field(default_factory=RepairConfig)
    current_model_path: str = "models/baseline_int8.onnx"
    iteration: int = 0
    accepted_history: list[dict[str, Any]] = field(default_factory=list)
    rejected_history: list[dict[str, Any]] = field(default_factory=list)
    last_candidate_config: RepairConfig | None = None
    last_candidate_result: QuantResult | None = None
    last_params_chips: list[str] = field(default_factory=list)
    last_calib_images: list[str] = field(default_factory=list)
    last_provenance: dict[str, Any] = field(default_factory=dict)

    def commit_candidate(self) -> None:
        if self.last_candidate_config is not None and self.last_candidate_result is not None:
            self.current_config = self.last_candidate_config
            self.current_model_path = self.last_candidate_result.path
            self.accepted_history.append(
                {
                    "iteration": self.iteration,
                    "config": self.current_config.to_dict(),
                    "model_path": self.current_model_path,
                    "provenance": dict(self.last_provenance),
                }
            )
            self.last_candidate_config = None
            self.last_candidate_result = None

    def rollback_candidate(self, reasons: Sequence[str], action: dict[str, Any] | None = None) -> None:
        self.rejected_history.append(
            {
                "iteration": self.iteration,
                "action": action or {},
                "attempted_config": (
                    self.last_candidate_config.to_dict() if self.last_candidate_config else None
                ),
                "candidate_path": (
                    self.last_candidate_result.path if self.last_candidate_result else None
                ),
                "reasons": list(reasons),
                "rolled_back_to_config": self.current_config.to_dict(),
                "rolled_back_to_model": self.current_model_path,
            }
        )
        self.last_candidate_config = None
        self.last_candidate_result = None


def _parse_action_types(raw_type: Any) -> list[str]:
    if isinstance(raw_type, (list, tuple)):
        tokens: list[str] = []
        for item in raw_type:
            tokens.extend(_parse_action_types(item))
        return tokens
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise RepairRejectedError("REJECTED: missing or invalid action.type")
    cleaned = raw_type.strip()
    if cleaned.lower() == "none":
        return ["none"]
    for sep in ("+", ",", "/", " "):
        cleaned = cleaned.replace(sep, "|")
    parts = [p.strip().upper() for p in cleaned.split("|") if p.strip()]
    if not parts:
        raise RepairRejectedError(f"REJECTED: invalid action.type '{raw_type}'")
    return parts


def _normalize_method(raw_method: Any) -> str:
    if not isinstance(raw_method, str) or not raw_method.strip():
        raise RepairRejectedError(f"REJECTED: invalid calibration method '{raw_method}'")
    m = raw_method.strip().lower()
    if m.startswith("percentile"):
        return "percentile"
    if m == "entropy":
        return "entropy"
    if m == "minmax":
        return "minmax"
    raise RepairRejectedError(
        f"REJECTED: unsupported calibration method '{raw_method}' (allowed: Percentile 99.999, Entropy, MinMax)"
    )


def _load_splits_and_base_calib(ctx: RepairContext) -> tuple[set[str], set[str], list[Path]]:
    scenario_dir = Path(ctx.data_root) / ctx.scenario_id
    splits_path = scenario_dir / "splits.json"
    calib_path = scenario_dir / "lab_calib_25.json"

    if not splits_path.is_file():
        raise RepairRejectedError(f"REJECTED: splits.json not found at {splits_path}")
    if not calib_path.is_file():
        raise RepairRejectedError(f"REJECTED: lab_calib_25.json not found at {calib_path}")

    splits = json.loads(splits_path.read_text())
    split_a = set(splits.get("split_a", []))
    split_b = set(splits.get("split_b", []))

    calib_data = json.loads(calib_path.read_text())
    selected_paths = calib_data.get("selected_paths", [])
    base_paths: list[Path] = []
    for p_str in selected_paths:
        p = Path(p_str)
        if not p.is_absolute() and not p.exists():
            alt = scenario_dir / "real" / p.name
            if alt.exists():
                p = alt
        if p.name in split_b:
            raise RepairRejectedError(
                f"REJECTED: Split B image '{p.name}' detected in base calibration list"
            )
        if p.name not in split_a:
            raise RepairRejectedError(
                f"REJECTED: Calibration image '{p.name}' is not in Split A"
            )
        if not p.is_file():
            raise RepairRejectedError(f"REJECTED: Base calibration file missing: {p}")
        base_paths.append(p)

    if len(base_paths) != 25:
        raise RepairRejectedError(
            f"REJECTED: Expected 25 lab-like Split-A calibration images, got {len(base_paths)}"
        )
    return split_a, split_b, base_paths


def _is_manifest_entry_valid_for_split_a(entry: dict[str, Any]) -> bool:
    if "valid" in entry and entry["valid"] is False:
        return False
    if "validity" in entry and entry["validity"] is False:
        return False
    return True


def _resolve_manifest_entry_fname(item: dict[str, Any], split_a: set[str], split_b: set[str]) -> str:
    raw = (
        item.get("file_name")
        or item.get("condition_path")
        or item.get("original_path")
        or item.get("file")
        or item.get("image")
        or item.get("image_id")
        or ""
    )
    fname = Path(str(raw)).name
    if fname and not fname.endswith(".jpg") and (f"{fname}.jpg" in split_a or f"{fname}.jpg" in split_b):
        fname = f"{fname}.jpg"
    return fname


def _collect_split_a_condition_images(
    ctx: RepairContext,
    targets: Sequence[str],
    split_a: set[str],
    split_b: set[str],
    max_per_condition: int | None = None,
) -> tuple[list[Path], dict[str, int]]:
    scenario_dir = Path(ctx.data_root) / ctx.scenario_id
    manifest_path = scenario_dir / "manifest.json"
    manifest_map: dict[tuple[str, str], dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            m_data = json.loads(manifest_path.read_text())
            entries = m_data.get("entries") or m_data.get("images") or []
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, dict):
                        cid = str(item.get("condition") or item.get("condition_id") or "")
                        fname = _resolve_manifest_entry_fname(item, split_a, split_b)
                        if cid and fname:
                            manifest_map[(cid, fname)] = item
            elif isinstance(m_data.get("conditions"), dict):
                for cid, cond_block in m_data["conditions"].items():
                    items = cond_block.get("entries") or cond_block.get("images") or cond_block
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                fname = _resolve_manifest_entry_fname(item, split_a, split_b)
                                if fname:
                                    manifest_map[(str(cid), fname)] = item
                    elif isinstance(items, dict):
                        for k_name, item in items.items():
                            if isinstance(item, dict):
                                fname = _resolve_manifest_entry_fname(item, split_a, split_b) or Path(str(k_name)).name
                                if fname and not fname.endswith(".jpg") and (f"{fname}.jpg" in split_a or f"{fname}.jpg" in split_b):
                                    fname = f"{fname}.jpg"
                                manifest_map[(str(cid), fname)] = item
        except Exception:
            pass

    sorted_split_a = sorted(split_a)
    collected: list[Path] = []
    counts_by_cond: dict[str, int] = {}

    for cond_id in targets:
        cond_dir = scenario_dir / "conditions" / cond_id
        if not cond_dir.is_dir():
            raise RepairRejectedError(
                f"REJECTED: Split-A condition directory does not exist for '{cond_id}': {cond_dir}"
            )
        cond_paths: list[Path] = []
        for fname in sorted_split_a:
            if fname in split_b:
                raise RepairRejectedError(f"REJECTED: Split B image '{fname}' forbidden in R1")
            cand_p = cond_dir / fname
            if not cand_p.is_file():
                continue
            m_entry = manifest_map.get((cond_id, fname))
            if m_entry is not None:
                split_tag = str(m_entry.get("split", "A")).upper()
                if split_tag == "B":
                    raise RepairRejectedError(
                        f"REJECTED: Manifest marks '{fname}' as Split B; forbidden in calibration"
                    )
                if not _is_manifest_entry_valid_for_split_a(m_entry):
                    continue
            cond_paths.append(cand_p)
            if max_per_condition is not None and len(cond_paths) >= max_per_condition:
                break

        if not cond_paths:
            raise RepairRejectedError(
                f"REJECTED: No valid Split-A condition images found for target '{cond_id}'"
            )
        counts_by_cond[cond_id] = len(cond_paths)
        collected.extend(cond_paths)

    return collected, counts_by_cond


def validate_and_resolve_action(
    action: dict[str, Any],
    ctx: RepairContext,
) -> tuple[RepairConfig, list[Path], list[str], dict[str, Any]]:
    """Validate an action dict against Block D rules and return
    (new_config, calib_image_paths, params_chips, provenance).
    """
    if not isinstance(action, dict):
        raise RepairRejectedError("REJECTED: action must be a JSON object")

    action_types = _parse_action_types(action.get("type"))
    if "none" in action_types:
        raise RepairRejectedError("REJECTED: action type is 'none'")

    for t in action_types:
        if t in FORBIDDEN_TYPES:
            raise RepairRejectedError(
                f"REJECTED: Action {t} is excluded from active repair scope (only R1, R2, R3 allowed)"
            )
        if t not in ALLOWED_ACTIVE_TYPES:
            raise RepairRejectedError(
                f"REJECTED: Unknown action type '{t}' (only R1, R2, R3 allowed)"
            )

    params = action.get("params") or {}
    if not isinstance(params, dict):
        raise RepairRejectedError("REJECTED: action.params must be a dict")

    # Guard against hidden R4 / R5 parameters in params
    if "preprocess" in params and str(params["preprocess"]).lower() != "stretch":
        raise RepairRejectedError(
            f"REJECTED: R4 preprocessing change ('{params['preprocess']}') is excluded from active scope"
        )
    if params.get("nodes_to_exclude") or params.get("exclude_preset") or params.get("preset"):
        raise RepairRejectedError(
            "REJECTED: R5 node exclusion is excluded from active repair scope"
        )
    if "op_types_to_quantize" in params and list(params["op_types_to_quantize"]) != ["Conv"]:
        raise RepairRejectedError(
            "REJECTED: op_types_to_quantize must remain ['Conv']"
        )

    # Guard against explicit calibration image paths in params referencing split B
    split_a, split_b, base_calib_paths = _load_splits_and_base_calib(ctx)
    if "calib_images" in params or "images" in params:
        raw_imgs = params.get("calib_images") or params.get("images") or []
        if isinstance(raw_imgs, list):
            for item in raw_imgs:
                fname = Path(str(item)).name
                if not fname.endswith(".jpg") and f"{fname}.jpg" in split_b:
                    fname = f"{fname}.jpg"
                if fname in split_b or "b_" in str(item).lower():
                    raise RepairRejectedError(
                        f"REJECTED: Split B image '{item}' is strictly forbidden in calibration"
                    )
                if fname not in split_a:
                    raise RepairRejectedError(
                        f"REJECTED: Image '{item}' is not in Split A"
                    )

    cur = ctx.current_config
    new_method = cur.method
    new_per_channel = cur.per_channel
    new_targets = list(cur.r1_targets)
    new_weight = cur.r1_weight
    new_max_per_cond = cur.r1_max_per_condition

    # Apply R2 (or method override in params if valid)
    if "R2" in action_types or ("method" in params and params["method"] is not None) or ("calibrate_method" in params and params["calibrate_method"] is not None):
        if "R2" not in action_types and "method" in params:
            # Allow combined specification if method is valid
            new_method = _normalize_method(params["method"])
        elif "R2" not in action_types and "calibrate_method" in params:
            new_method = _normalize_method(params["calibrate_method"])
        else:
            raw_m = params.get("method") or params.get("calibrate_method")
            if raw_m is not None:
                new_method = _normalize_method(raw_m)
                if new_method == "minmax":
                    raise RepairRejectedError(
                        "REJECTED: R2 requires Percentile(99.999) or Entropy, not MinMax"
                    )
            else:
                new_method = "percentile" if cur.method == "minmax" else "entropy"

    # Apply R3 (or per_channel override in params)
    if "R3" in action_types or params.get("per_channel") is True:
        if "per_channel" in params and params["per_channel"] is False:
            raise RepairRejectedError("REJECTED: R3 must enable per_channel=True")
        new_per_channel = True

    # Apply R1
    if "R1" in action_types:
        raw_targets = action.get("targets") or params.get("targets") or params.get("conditions") or []
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise RepairRejectedError("REJECTED: R1 requires non-empty target condition(s)")
        for t_cond in raw_targets:
            t_str = str(t_cond).strip()
            if not t_str or t_str == "control":
                raise RepairRejectedError(f"REJECTED: Invalid R1 target condition '{t_cond}'")
            if t_str not in new_targets:
                new_targets.append(t_str)

        raw_w = params.get("duplication_weight", params.get("weight", new_weight))
        if isinstance(raw_w, bool) or not isinstance(raw_w, int) or raw_w < 1 or raw_w > 3:
            raise RepairRejectedError(
                f"REJECTED: R1 duplication weight must be an integer in 1..3, got {raw_w}"
            )
        new_weight = int(raw_w)

        raw_max = params.get("max_images_per_condition", params.get("n_images", new_max_per_cond))
        if raw_max is not None:
            if isinstance(raw_max, bool) or not isinstance(raw_max, int) or raw_max < 1:
                raise RepairRejectedError(f"REJECTED: invalid R1 image count {raw_max}")
            new_max_per_cond = int(raw_max)
    else:
        # Check if weight was passed without R1 or on MinMax
        if "duplication_weight" in params or "weight" in params:
            raw_w = params.get("duplication_weight", params.get("weight", 1))
            if isinstance(raw_w, bool) or not isinstance(raw_w, int) or raw_w < 1 or raw_w > 3:
                raise RepairRejectedError(
                    f"REJECTED: duplication weight must be an integer in 1..3, got {raw_w}"
                )
            if int(raw_w) > 1 and not new_targets:
                raise RepairRejectedError(
                    "REJECTED: duplication weight > 1 requires R1 target condition images"
                )
            new_weight = int(raw_w)

    # Rule: R1 duplication weight 1-3 is ONLY allowed when method is Percentile or Entropy; forbidden under MinMax
    if new_weight > 1 and new_method == "minmax":
        raise RepairRejectedError(
            "REJECTED: R1 duplication weight > 1 is forbidden under MinMax (mathematical no-op); use Percentile or Entropy"
        )

    cand_cfg = RepairConfig(
        method=new_method,
        per_channel=new_per_channel,
        preprocess="stretch",
        r1_targets=tuple(new_targets),
        r1_weight=new_weight,
        r1_max_per_condition=new_max_per_cond,
        nodes_to_exclude=(),
    )

    calib_paths: list[Path] = list(base_calib_paths)
    cond_counts: dict[str, int] = {}
    if cand_cfg.r1_targets:
        cond_imgs, cond_counts = _collect_split_a_condition_images(
            ctx,
            cand_cfg.r1_targets,
            split_a=split_a,
            split_b=split_b,
            max_per_condition=cand_cfg.r1_max_per_condition,
        )
        for _ in range(cand_cfg.r1_weight):
            calib_paths.extend(cond_imgs)

    # Final safety verification: every single calibration path must be in Split A and never Split B
    for p in calib_paths:
        if p.name in split_b or p.name not in split_a:
            raise RepairRejectedError(
                f"REJECTED: Calibration image '{p}' is not a valid Split-A image"
            )

    chips: list[str] = []
    if cand_cfg.method == "percentile":
        chips.append("Percentile 99.999")
    elif cand_cfg.method == "entropy":
        chips.append("Entropy")
    else:
        chips.append("MinMax")

    if cand_cfg.per_channel:
        chips.append("per-channel")

    for t_cond in cand_cfg.r1_targets:
        cnt = cond_counts.get(t_cond, 0)
        if cand_cfg.r1_weight > 1:
            chips.append(f"+{cnt} {t_cond} Split-A (x{cand_cfg.r1_weight})")
        else:
            chips.append(f"+{cnt} {t_cond} images")

    provenance = {
        "parent_iteration": ctx.iteration - 1 if ctx.iteration > 0 else 0,
        "parent_model": ctx.current_model_path,
        "parent_config": cur.to_dict(),
        "candidate_config": cand_cfg.to_dict(),
        "base_calib_count": len(base_calib_paths),
        "condition_calib_counts": cond_counts,
        "duplication_weight": cand_cfg.r1_weight,
        "total_calib_images": len(calib_paths),
        "calib_images": [str(p) for p in calib_paths],
    }

    return cand_cfg, calib_paths, chips, provenance


def apply(action: dict[str, Any], ctx: RepairContext) -> QuantResult:
    """Validate repair action, build candidate INT8 model in candidate directory without overwriting baseline, and return QuantResult."""
    cand_cfg, calib_paths, chips, provenance = validate_and_resolve_action(action, ctx)

    out_dir = Path(ctx.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # First compute hash or quantize to a deterministic candidate path in out_dir
    from odct.quant import compute_quant_config_hash

    extra_options = {"CalibPercentile": 99.999} if cand_cfg.method == "percentile" else {}
    cfg_hash = compute_quant_config_hash(
        fp32_path=ctx.fp32_path,
        calib_images=calib_paths,
        method=cand_cfg.method,
        per_channel=cand_cfg.per_channel,
        preprocess=cand_cfg.preprocess,
        nodes_to_exclude=list(cand_cfg.nodes_to_exclude),
        op_types_to_quantize=["Conv"],
        extra_options=extra_options,
        imgsz=320,
    )

    cand_path = out_dir / f"int8_{cfg_hash[:16]}.onnx"
    if cand_path.resolve() == Path(ctx.baseline_int8_path).resolve():
        cand_path = out_dir / f"candidate_{cfg_hash[:16]}.onnx"

    res = quantize(
        fp32_path=ctx.fp32_path,
        calib_images=calib_paths,
        out_path=cand_path,
        method=cand_cfg.method,
        per_channel=cand_cfg.per_channel,
        preprocess=cand_cfg.preprocess,
        nodes_to_exclude=list(cand_cfg.nodes_to_exclude),
        use_cache=True,
    )

    ctx.last_candidate_config = cand_cfg
    ctx.last_candidate_result = res
    ctx.last_params_chips = chips
    ctx.last_calib_images = [str(p) for p in calib_paths]
    provenance["config_hash"] = res.config_hash
    provenance["cached"] = res.cached
    provenance["candidate_path"] = res.path
    ctx.last_provenance = provenance
    return res
