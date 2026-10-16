"""Select the 25 brightest and sharpest images from Split A for baseline lab-like calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def select_lab_calibration_images(scenario_id: str = "robot_vacuum", top_k: int = 25) -> dict:
    split_path = Path("data") / scenario_id / "splits.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Splits file not found: {split_path}")

    splits = json.loads(split_path.read_text())
    split_a: list[str] = splits["split_a"]
    real_dir = Path("data") / scenario_id / "real"

    records: list[dict] = []
    for fname in split_a:
        fpath = real_dir / fname
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            raise ValueError(f"Failed to read image: {fpath}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        records.append(
            {
                "file": fname,
                "path": str(fpath),
                "mean_luminance": round(mean_lum, 4),
                "laplacian_variance": round(lap_var, 4),
                "log_laplacian": float(np.log1p(max(0.0, lap_var))),
            }
        )

    lums = np.array([r["mean_luminance"] for r in records], dtype=np.float64)
    laps = np.array([r["log_laplacian"] for r in records], dtype=np.float64)

    lum_z = (lums - lums.mean()) / (lums.std() + 1e-8)
    lap_z = (laps - laps.mean()) / (laps.std() + 1e-8)
    scores = 0.5 * lum_z + 0.5 * lap_z

    for idx, r in enumerate(records):
        r["lab_score"] = round(float(scores[idx]), 4)
        r.pop("log_laplacian", None)

    ranked = sorted(records, key=lambda x: x["lab_score"], reverse=True)
    selected = ranked[:top_k]

    payload = {
        "scenario": scenario_id,
        "rule": "Top 25 images from Split A by 0.5 * z(mean_luminance) + 0.5 * z(log1p(laplacian_variance))",
        "split_source": "A",
        "top_k": top_k,
        "selected_files": [r["file"] for r in selected],
        "selected_paths": [r["path"] for r in selected],
        "selected_details": selected,
    }
    out_path = Path("data") / scenario_id / "lab_calib_25.json"
    out_path.write_text(json.dumps(payload, indent=2) + chr(10))
    print(
        json.dumps(
            {
                "scenario": scenario_id,
                "selected_count": len(selected),
                "output_file": str(out_path),
                "top_3": selected[:3],
            },
            indent=2,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 25 brightest and sharpest Split A images.")
    parser.add_argument("--scenario", default="robot_vacuum", help="Scenario ID")
    parser.add_argument("--top-k", type=int, default=25, help="Number of calibration images to select")
    args = parser.parse_args()
    select_lab_calibration_images(scenario_id=args.scenario, top_k=args.top_k)


if __name__ == "__main__":
    main()
