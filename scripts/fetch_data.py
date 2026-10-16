"""Fetch 80 COCO val2017 images via coco2017labels.zip (without FiftyOne) and create seeded A40/B40 split."""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from PIL import Image

from odct.infer import Detector

LABELS_ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
COCO128_ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_VAL_IMG_URL = "http://images.cocodataset.org/val2017/{stem}.jpg"


def _resize_long_side(img: Image.Image, max_side: int = 640) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = float(max_side) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def _download_bytes(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "odct-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _select_candidate_stems(cache_zip: Path, target_classes: set[int]) -> list[str]:
    if not cache_zip.exists():
        cache_zip.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {LABELS_ZIP_URL} ...")
        data = _download_bytes(LABELS_ZIP_URL, timeout=60.0)
        cache_zip.write_bytes(data)

    candidates: list[str] = []
    with zipfile.ZipFile(cache_zip, "r") as zf:
        val_txt_names = sorted(
            name
            for name in zf.namelist()
            if "labels/val2017/" in name and name.endswith(".txt") and not name.endswith("/")
        )
        for name in val_txt_names:
            raw = zf.read(name).decode("utf-8", errors="ignore")
            present_classes = set()
            for line in raw.splitlines():
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    present_classes.add(int(parts[0]))
            if present_classes & target_classes:
                stem = Path(name).stem
                candidates.append(stem)
    return candidates


def _download_one_val_image(stem: str) -> tuple[str, bytes | None]:
    url = COCO_VAL_IMG_URL.format(stem=stem)
    try:
        return stem, _download_bytes(url, timeout=15.0)
    except Exception:
        return stem, None


def fetch_scenario_data(
    scenario_id: str = "robot_vacuum",
    target_count: int = 80,
    fp32_path: str = "models/fp32.onnx",
    seed: int = 20261016,
) -> dict:
    scenario_file = Path("scenarios") / f"{scenario_id}.yaml"
    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario profile not found: {scenario_file}")
    cfg = yaml.safe_load(scenario_file.read_text())
    target_classes = {int(v) for v in cfg["classes"].values()}

    detector = Detector(fp32_path, preprocess="stretch", conf=0.25, iou=0.5, allowed_classes=list(target_classes))

    real_dir = Path("data") / scenario_id / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    kept_files: list[str] = []
    object_counts: dict[str, int] = {}

    existing = sorted(p for p in real_dir.glob("*.jpg"))
    for p in existing:
        dets = detector.detect(p)
        if len(dets) >= 1:
            kept_files.append(p.name)
            object_counts[p.name] = len(dets)
            if len(kept_files) >= target_count:
                break

    if len(kept_files) < target_count:
        cache_zip = Path("/tmp/coco2017labels.zip")
        try:
            stems = _select_candidate_stems(cache_zip, target_classes)
        except Exception as exc:
            print(f"coco2017labels.zip fetch failed ({exc}), falling back to coco128.zip")
            stems = []

        already = {Path(f).stem for f in kept_files}
        stems = [s for s in stems if s not in already]

        batch_size = 32
        cursor = 0
        while len(kept_files) < target_count and cursor < len(stems):
            batch = stems[cursor : cursor + batch_size]
            cursor += batch_size
            downloaded: dict[str, bytes] = {}
            with ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(_download_one_val_image, s) for s in batch]
                for fut in as_completed(futs):
                    stem, raw_bytes = fut.result()
                    if raw_bytes is not None:
                        downloaded[stem] = raw_bytes

            for stem in batch:
                if len(kept_files) >= target_count:
                    break
                raw_bytes = downloaded.get(stem)
                if not raw_bytes:
                    continue
                try:
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    img = _resize_long_side(img, max_side=640)
                    out_file = real_dir / f"{stem}.jpg"
                    img.save(out_file, format="JPEG", quality=92)
                    dets = detector.detect(out_file)
                    if len(dets) >= 1:
                        kept_files.append(out_file.name)
                        object_counts[out_file.name] = len(dets)
                    else:
                        out_file.unlink(missing_ok=True)
                except Exception:
                    continue

    if len(kept_files) < target_count:
        c128_zip = Path("/tmp/coco128.zip")
        if not c128_zip.exists():
            c128_zip.write_bytes(_download_bytes(COCO128_ZIP_URL, timeout=60.0))
        with zipfile.ZipFile(c128_zip, "r") as zf:
            img_names = sorted(n for n in zf.namelist() if n.endswith(".jpg") and "/images/" in n)
            for name in img_names:
                if len(kept_files) >= target_count:
                    break
                fname = Path(name).name
                if fname in kept_files:
                    continue
                raw_bytes = zf.read(name)
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                img = _resize_long_side(img, max_side=640)
                out_file = real_dir / fname
                img.save(out_file, format="JPEG", quality=92)
                dets = detector.detect(out_file)
                if len(dets) >= 1:
                    kept_files.append(out_file.name)
                    object_counts[out_file.name] = len(dets)
                else:
                    out_file.unlink(missing_ok=True)

    kept_files = sorted(kept_files)[:target_count]
    for extra in real_dir.glob("*.jpg"):
        if extra.name not in set(kept_files):
            extra.unlink(missing_ok=True)

    if len(kept_files) < target_count:
        raise RuntimeError(f"Only collected {len(kept_files)} valid images (required {target_count}).")

    rng = random.Random(seed)
    shuffled = list(kept_files)
    rng.shuffle(shuffled)
    half = target_count // 2
    split_a = sorted(shuffled[:half])
    split_b = sorted(shuffled[half : half * 2])

    split_payload = {
        "scenario": scenario_id,
        "seed": seed,
        "total_images": len(kept_files),
        "split_a_count": len(split_a),
        "split_b_count": len(split_b),
        "split_a": split_a,
        "split_b": split_b,
        "fp32_object_counts": {k: object_counts[k] for k in kept_files},
    }
    split_path = Path("data") / scenario_id / "splits.json"
    split_path.write_text(json.dumps(split_payload, indent=2) + chr(10))
    print(
        json.dumps(
            {
                "scenario": scenario_id,
                "total_images": len(kept_files),
                "split_a_count": len(split_a),
                "split_b_count": len(split_b),
                "splits_file": str(split_path),
            },
            indent=2,
        )
    )
    return split_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch COCO val2017 images and write seeded A/B split.")
    parser.add_argument("--scenario", default="robot_vacuum", help="Scenario ID")
    parser.add_argument("--count", type=int, default=80, help="Total images to keep")
    parser.add_argument("--fp32", default="models/fp32.onnx", help="FP32 ONNX model path")
    parser.add_argument("--seed", type=int, default=20261016, help="Split RNG seed")
    args = parser.parse_args()
    fetch_scenario_data(
        scenario_id=args.scenario,
        target_count=args.count,
        fp32_path=args.fp32,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
