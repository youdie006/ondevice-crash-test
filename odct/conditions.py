"""Condition generators, validity gate, Flash audit, and manifest provenance for On-Device Crash Test."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, Field

from odct.infer import Det, Detector
from odct.metrics import persisted

# Model identifiers per verified official Gemini API specs and API_NOTES.md
NANO_BANANA_MODEL = "gemini-nano-banana-2.1"
FLASH_AUDIT_MODEL = "gemini-3.8-flash"

MANDATORY_PRESERVE_CLAUSE = "Keep every object exactly where it is."

# Fixed classical fallback constants (defined once, never retuned post-evaluation)
FIXED_CLASSICAL_SEVERITIES: dict[str, dict[str, float | int]] = {
    "night": {"gamma": 0.35, "noise_sigma": 12.0},
    "glare": {"highlight_strength": 0.65, "contrast_scale": 0.60, "flare_radius_frac": 0.55},
    "rain": {"droplet_count": 28, "blur_sigma": 3.5, "droplet_radius_min": 6, "droplet_radius_max": 18, "droplet_alpha": 0.38},
    "fog": {"alpha_white": 0.45, "blur_sigma": 2.5},
    "occlusion": {"num_rectangles": 2, "area_fraction_min": 0.15, "area_fraction_max": 0.25, "fill_gray": 114},
    "blur": {"kernel_size": 15},
    "smudge": {"num_blobs": 3, "blob_sigma": 25.0, "haze_alpha": 0.20, "blob_radius_frac": 0.16},
    "color_cast": {"blue_gain": 1.35, "red_gain": 0.75, "green_gain": 1.00},
    "condensation": {"droplet_count": 32, "blur_sigma": 2.5, "droplet_radius_min": 4, "droplet_radius_max": 12, "droplet_alpha": 0.35},
    "frost": {"alpha_white": 0.35, "blur_sigma": 1.5},
    "led_glare": {"highlight_strength": 0.60, "contrast_scale": 0.65, "flare_radius_frac": 0.50},
    "dim": {"gamma": 0.40, "noise_sigma": 8.0},
    "stacking": {"num_rectangles": 2, "area_fraction_min": 0.15, "area_fraction_max": 0.25, "fill_gray": 114},
}

DEFAULT_CONDITION_SEEDS: dict[str, int] = {
    "night": 101,
    "glare": 102,
    "rain": 103,
    "fog": 104,
    "occlusion": 105,
    "blur": 106,
    "smudge": 107,
    "condensation": 201,
    "frost": 202,
    "led_glare": 203,
    "color_cast": 204,
    "dim": 205,
    "stacking": 206,
}

DEFAULT_NANO_PROMPTS: dict[str, str] = {
    "night": "The same room at night lit by one dim lamp. Keep every object exactly where it is; change only lighting.",
    "glare": "Strong sunlight from a window behind the objects with lens flare. Keep every object exactly where it is; change only lighting and lens flare.",
    "rain": "Water droplets on the camera lens. Keep every object exactly where it is.",
    "fog": "Lens fogged by steam. Keep every object exactly where it is.",
    "condensation": "Lens covered in fine condensation droplets after the door opened on a humid day. Keep every object exactly where it is.",
    "frost": "Thin frost on packaging and shelf edges. Keep every object exactly where it is.",
    "led_glare": "Harsh interior LED reflecting off glossy packaging. Keep every object exactly where it is.",
}

SUPPORTED_ASPECT_RATIOS: list[tuple[str, float]] = [
    ("1:1", 1.0),
    ("4:3", 4.0 / 3.0),
    ("3:4", 3.0 / 4.0),
    ("3:2", 3.0 / 2.0),
    ("2:3", 2.0 / 3.0),
    ("16:9", 16.0 / 9.0),
    ("9:16", 9.0 / 16.0),
    ("5:4", 5.0 / 4.0),
    ("4:5", 4.0 / 5.0),
    ("21:9", 21.0 / 9.0),
]

_CLIENT_LOCK = threading.Lock()
_LAZY_CLIENT: Any = None


def get_gemini_client() -> Any:
    """Lazy-initialize the google.genai Client from GEMINI_API_KEY env var only."""
    global _LAZY_CLIENT
    with _CLIENT_LOCK:
        if _LAZY_CLIENT is not None:
            return _LAZY_CLIENT
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        from google import genai

        _LAZY_CLIENT = genai.Client(api_key=api_key)
        return _LAZY_CLIENT


class FlashAuditResponse(BaseModel):
    same_objects_and_positions: bool = Field(
        description="True if and only if the edited image contains the exact same objects in the same positions as the original image without adding, removing, or moving objects."
    )
    differences: list[str] = Field(
        default_factory=list,
        description="List of any object additions, removals, or position shifts observed between the original and edited images.",
    )
    reason: str = Field(
        default="",
        description="Concise explanation of whether objects and positions were preserved.",
    )


@dataclass
class ManifestEntry:
    image_id: str
    source_image_id: str
    file_name: str
    split: str
    scenario_id: str
    condition_id: str
    original_path: str
    condition_path: str
    thumb_url: str
    original_url: str
    requested_generator: str
    generator: str
    generator_label: str
    seed: int
    image_seed: int
    severity: dict[str, Any]
    source_size: list[int]
    output_size: list[int]
    valid: bool
    validity_status: str
    allow_in_pipeline: bool
    d0_count: int
    dc_count: int
    persisted_count: int
    retention: float | None
    audit_status: str
    audit_passed: bool | None
    audit_reason: str
    audit_differences: list[str] = field(default_factory=list)
    regeneration_count: int = 0
    nano_attempts: int = 0
    nano_api_calls: int = 0
    flash_api_calls: int = 0
    used_fallback: bool = False
    fallback_reason: str | None = None
    prompt: str | None = None
    interaction_ids: list[str] = field(default_factory=list)
    path_root: str = "project_root"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_image_seed(base_seed: int, condition_id: str, image_name: str) -> int:
    """Deterministically derive a 32-bit seed from base_seed, condition_id, and image filename."""
    stem = Path(image_name).stem
    payload = f"{base_seed}:{condition_id}:{stem}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _from_rgb_array(arr: np.ndarray) -> Image.Image:
    clipped = np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def classical_night(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Darken image via luminance scaling (I / 255)^(1 / gamma) * 255 with gamma=0.35 (<1 darkens) + Gaussian noise sigma=12."""
    defaults = FIXED_CLASSICAL_SEVERITIES["night"]
    gamma = float(severity.get("gamma", defaults["gamma"]))
    noise_sigma = float(severity.get("noise_sigma", defaults["noise_sigma"]))

    arr = _to_rgb_array(image) / 255.0
    # Explicit darkening convention: gamma < 1.0 darkens by raising normalized luminance to exponent (1.0 / gamma)
    exponent = (1.0 / max(gamma, 1e-4)) if gamma < 1.0 else gamma
    darkened = np.power(np.clip(arr, 0.0, 1.0), exponent) * 255.0

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_sigma, size=darkened.shape).astype(np.float32)
    return _from_rgb_array(darkened + noise)


def classical_glare(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Apply radial window backlight highlight and global contrast reduction."""
    defaults = FIXED_CLASSICAL_SEVERITIES["glare"]
    highlight_strength = float(severity.get("highlight_strength", defaults["highlight_strength"]))
    contrast_scale = float(severity.get("contrast_scale", defaults["contrast_scale"]))
    flare_radius_frac = float(severity.get("flare_radius_frac", defaults["flare_radius_frac"]))

    arr = _to_rgb_array(image)
    h, w, _ = arr.shape
    rng = np.random.default_rng(seed)

    # Deterministic upper-region window glare source
    cx = float(rng.uniform(0.25 * w, 0.75 * w))
    cy = float(rng.uniform(0.10 * h, 0.35 * h))
    max_r = max(1.0, flare_radius_frac * float(max(w, h)))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = np.exp(-0.5 * (dist / (0.45 * max_r)) ** 2)[..., None]

    mean_val = np.mean(arr, axis=(0, 1), keepdims=True)
    low_contrast = (arr - mean_val) * contrast_scale + mean_val + 25.0 * (1.0 - contrast_scale)
    glare_layer = low_contrast + radial * (highlight_strength * 255.0)
    return _from_rgb_array(glare_layer)


def classical_rain(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Apply deterministic lens water droplets with localized refraction blur and specular rims."""
    defaults = FIXED_CLASSICAL_SEVERITIES["rain"]
    droplet_count = int(severity.get("droplet_count", defaults["droplet_count"]))
    blur_sigma = float(severity.get("blur_sigma", defaults["blur_sigma"]))
    r_min = int(severity.get("droplet_radius_min", defaults["droplet_radius_min"]))
    r_max = int(severity.get("droplet_radius_max", defaults["droplet_radius_max"]))
    droplet_alpha = float(severity.get("droplet_alpha", defaults["droplet_alpha"]))

    arr = _to_rgb_array(image)
    h, w, _ = arr.shape
    rng = np.random.default_rng(seed)

    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=max(0.5, blur_sigma), sigmaY=max(0.5, blur_sigma))
    mask = np.zeros((h, w), dtype=np.float32)
    rim = np.zeros((h, w), dtype=np.float32)

    for _ in range(droplet_count):
        cx = int(rng.integers(0, max(1, w)))
        cy = int(rng.integers(0, max(1, h)))
        rx = int(rng.integers(r_min, max(r_min + 1, r_max + 1)))
        ry = int(max(3, round(rx * float(rng.uniform(0.85, 1.25)))))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 1.0, -1)
        cv2.ellipse(rim, (cx, max(0, cy - max(1, ry // 3))), (max(1, rx // 2), max(1, ry // 3)), 0.0, 0.0, 360.0, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0, sigmaY=2.0)[..., None]
    rim = cv2.GaussianBlur(rim, (0, 0), sigmaX=1.5, sigmaY=1.5)[..., None]

    # Blend blurred interior and bright wet droplet sheen
    wet_blur = blurred * (1.0 - droplet_alpha) + 235.0 * droplet_alpha
    out = arr * (1.0 - mask) + wet_blur * mask + rim * 38.0
    return _from_rgb_array(out)


def classical_fog(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Apply atmospheric/steam white alpha blend and Gaussian blur."""
    _ = seed
    defaults = FIXED_CLASSICAL_SEVERITIES["fog"]
    alpha_white = float(severity.get("alpha_white", defaults["alpha_white"]))
    blur_sigma = float(severity.get("blur_sigma", defaults["blur_sigma"]))

    arr = _to_rgb_array(image)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=max(0.5, blur_sigma), sigmaY=max(0.5, blur_sigma))
    out = blurred * (1.0 - alpha_white) + 255.0 * alpha_white
    return _from_rgb_array(out)


def classical_occlusion(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Place two gray rectangles covering a combined 15-25% of the image area."""
    defaults = FIXED_CLASSICAL_SEVERITIES["occlusion"]
    num_rectangles = int(severity.get("num_rectangles", defaults["num_rectangles"]))
    area_min = float(severity.get("area_fraction_min", defaults["area_fraction_min"]))
    area_max = float(severity.get("area_fraction_max", defaults["area_fraction_max"]))
    fill_gray = int(severity.get("fill_gray", defaults["fill_gray"]))

    arr = _to_rgb_array(image).copy()
    h, w, _ = arr.shape
    total_area = float(h * w)
    rng = np.random.default_rng(seed)

    target_total_frac = float(rng.uniform(area_min, area_max))
    if num_rectangles <= 1:
        splits = [target_total_frac]
    else:
        raw_weights = rng.uniform(0.35, 0.65, size=num_rectangles)
        splits = [float(w_i / raw_weights.sum() * target_total_frac) for w_i in raw_weights]

    # Place non-overlapping or minimally overlapping rectangles so total occluded area stays in [area_min, area_max]
    sectors = np.linspace(0, w, num_rectangles + 1, dtype=int)
    for idx, frac in enumerate(splits):
        rect_area = total_area * frac
        aspect = float(rng.uniform(0.65, 1.55))
        rw = int(np.clip(round(np.sqrt(rect_area * aspect)), 8, max(8, w // max(1, num_rectangles))))
        rh = int(np.clip(round(rect_area / max(1.0, float(rw))), 8, max(8, int(0.75 * h))))
        # Re-adjust rw so rw * rh closely matches rect_area
        rw = int(np.clip(round(rect_area / max(1.0, float(rh))), 8, w))

        x_min_sector = int(sectors[idx])
        x_max_sector = max(x_min_sector, int(sectors[idx + 1]) - rw)
        if x_max_sector > x_min_sector:
            x0 = int(rng.integers(x_min_sector, x_max_sector + 1))
        else:
            x0 = int(np.clip(x_min_sector, 0, max(0, w - rw)))
        y0 = int(rng.integers(0, max(1, h - rh + 1)))

        arr[y0 : y0 + rh, x0 : x0 + rw, :] = float(fill_gray)

    return _from_rgb_array(arr)


def classical_blur(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Apply horizontal motion blur with a 15 px linear kernel."""
    _ = seed
    defaults = FIXED_CLASSICAL_SEVERITIES["blur"]
    kernel_size = max(3, int(severity.get("kernel_size", defaults["kernel_size"])))

    arr = _to_rgb_array(image)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / float(kernel_size)
    blurred = cv2.filter2D(arr, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    return _from_rgb_array(blurred)


def classical_smudge(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Apply localized Gaussian blur blobs and slight haze across the lens."""
    defaults = FIXED_CLASSICAL_SEVERITIES["smudge"]
    num_blobs = int(severity.get("num_blobs", defaults["num_blobs"]))
    blob_sigma = float(severity.get("blob_sigma", defaults["blob_sigma"]))
    haze_alpha = float(severity.get("haze_alpha", defaults["haze_alpha"]))
    blob_radius_frac = float(severity.get("blob_radius_frac", defaults["blob_radius_frac"]))

    arr = _to_rgb_array(image)
    h, w, _ = arr.shape
    rng = np.random.default_rng(seed)

    heavy_blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=max(1.0, blob_sigma), sigmaY=max(1.0, blob_sigma))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob_mask = np.zeros((h, w), dtype=np.float32)
    radius = max(12.0, blob_radius_frac * float(min(w, h)))

    for _ in range(num_blobs):
        cx = float(rng.uniform(0.18 * w, 0.82 * w))
        cy = float(rng.uniform(0.18 * h, 0.82 * h))
        rx = radius * float(rng.uniform(0.85, 1.35))
        ry = radius * float(rng.uniform(0.85, 1.35))
        g = np.exp(-0.5 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        blob_mask = np.maximum(blob_mask, g)

    blob_mask = np.clip(blob_mask, 0.0, 1.0)[..., None]
    smudged = arr * (1.0 - blob_mask) + heavy_blur * blob_mask
    hazy = smudged * (1.0 - haze_alpha * (0.4 + 0.6 * blob_mask)) + 215.0 * (haze_alpha * (0.4 + 0.6 * blob_mask))
    return _from_rgb_array(hazy)


def classical_color_cast(image: Image.Image, severity: dict[str, Any], seed: int) -> Image.Image:
    """Apply deterministic white-balance shift toward blue."""
    _ = seed
    defaults = FIXED_CLASSICAL_SEVERITIES["color_cast"]
    red_gain = float(severity.get("red_gain", defaults["red_gain"]))
    green_gain = float(severity.get("green_gain", defaults["green_gain"]))
    blue_gain = float(severity.get("blue_gain", defaults["blue_gain"]))

    arr = _to_rgb_array(image).copy()
    arr[:, :, 0] *= red_gain
    arr[:, :, 1] *= green_gain
    arr[:, :, 2] *= blue_gain
    return _from_rgb_array(arr)


def select_closest_aspect_ratio(width: int, height: int) -> str:
    """Return the Nano Banana supported aspect ratio string closest to width / height."""
    target = float(max(1, width)) / float(max(1, height))
    best_label = "4:3"
    best_diff = float("inf")
    for label, ratio in SUPPORTED_ASPECT_RATIOS:
        diff = abs(target - ratio)
        if diff < best_diff:
            best_diff = diff
            best_label = label
    return best_label


def ensure_preserve_clause(prompt: str) -> str:
    """Ensure the prompt explicitly carries the mandatory object preservation instruction."""
    lower = prompt.lower()
    if "keep every object exactly where it is" in lower:
        return prompt
    stripped = prompt.strip()
    if not stripped.endswith("."):
        stripped += "."
    return f"{stripped} {MANDATORY_PRESERVE_CLAUSE}"


def _encode_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _encode_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _extract_interaction_image_bytes(interaction: Any) -> bytes:
    out_img = getattr(interaction, "output_image", None)
    if out_img is not None and getattr(out_img, "data", None):
        data = out_img.data
        return base64.b64decode(data) if isinstance(data, str) else bytes(data)

    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", None) == "model_output":
            for block in getattr(step, "content", []) or []:
                if getattr(block, "type", None) == "image" and getattr(block, "data", None):
                    data = block.data
                    return base64.b64decode(data) if isinstance(data, str) else bytes(data)
    raise RuntimeError("No image block found in Nano Banana interaction response.")


def call_nano_banana_edit(
    image: Image.Image,
    prompt: str,
    client: Any | None = None,
) -> tuple[Image.Image, str]:
    """Edit an image with gemini-nano-banana-2.1 via the Interactions API and resize back to original width x height."""
    orig_w, orig_h = image.size
    enforced_prompt = ensure_preserve_clause(prompt)
    png_bytes = _encode_png_bytes(image)
    active_client = client if client is not None else get_gemini_client()

    input_payload = [
        {"type": "text", "text": enforced_prompt},
        {"type": "image", "data": base64.b64encode(png_bytes).decode("utf-8"), "mime_type": "image/png"},
    ]

    interaction = active_client.interactions.create(
        model=NANO_BANANA_MODEL,
        input=input_payload,
        response_format=[{"type": "image"}],
    )

    raw_bytes = _extract_interaction_image_bytes(interaction)
    edited = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if edited.size != (orig_w, orig_h):
        edited = edited.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    interaction_id = str(getattr(interaction, "id", "") or "nano_banana_call")
    return edited, interaction_id


def call_flash_audit(
    original_image: Image.Image,
    edited_image: Image.Image,
    condition_id: str,
    client: Any | None = None,
) -> tuple[bool, str, list[str], str]:
    """Run independent gemini-3.8-flash audit checking same objects in the same positions."""
    active_client = client if client is not None else get_gemini_client()
    orig_b64 = base64.b64encode(_encode_jpeg_bytes(original_image)).decode("utf-8")
    edit_b64 = base64.b64encode(_encode_jpeg_bytes(edited_image)).decode("utf-8")

    audit_prompt = (
        f"Audit these two images (1: original photo, 2: edited photo for condition '{condition_id}'). "
        "Answer whether the edited photo keeps the same objects in the same positions as the original photo. "
        "Lighting, glare, darkness, haze, or lens effects are expected and allowed, "
        "but adding, removing, or moving physical objects is a failure."
    )

    interaction = active_client.interactions.create(
        model=FLASH_AUDIT_MODEL,
        input=[
            {"type": "text", "text": audit_prompt},
            {"type": "image", "data": orig_b64, "mime_type": "image/jpeg"},
            {"type": "image", "data": edit_b64, "mime_type": "image/jpeg"},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": FlashAuditResponse.model_json_schema(),
        },
    )

    raw_text = getattr(interaction, "output_text", "") or ""
    parsed = FlashAuditResponse.model_validate_json(raw_text)
    interaction_id = str(getattr(interaction, "id", "") or "flash_audit_call")
    return (
        bool(parsed.same_objects_and_positions),
        parsed.reason or ("passed" if parsed.same_objects_and_positions else "objects shifted or changed"),
        list(parsed.differences),
        interaction_id,
    )


@dataclass(frozen=True)
class Generator:
    id: str
    kind: str  # "classical" | "nano_banana"
    classical_fn: Callable[[Image.Image, dict[str, Any], int], Image.Image]
    default_severity: dict[str, Any]
    default_seed: int = 101
    default_prompt: str | None = None
    fallback_id: str | None = None

    def apply_classical(
        self,
        image: Image.Image,
        severity: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        merged = dict(self.default_severity)
        if severity:
            merged.update(severity)
        actual_seed = self.default_seed if seed is None else int(seed)
        return self.classical_fn(image, merged, actual_seed)

    def __call__(
        self,
        image: Image.Image,
        severity: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        return self.apply_classical(image, severity=severity, seed=seed)


CATALOG: dict[str, Generator] = {
    # Primary 7 conditions in catalog
    "night": Generator(
        id="night",
        kind="classical",
        classical_fn=classical_night,
        default_severity=FIXED_CLASSICAL_SEVERITIES["night"],
        default_seed=DEFAULT_CONDITION_SEEDS["night"],
        default_prompt=DEFAULT_NANO_PROMPTS["night"],
        fallback_id="classical_night",
    ),
    "glare": Generator(
        id="glare",
        kind="classical",
        classical_fn=classical_glare,
        default_severity=FIXED_CLASSICAL_SEVERITIES["glare"],
        default_seed=DEFAULT_CONDITION_SEEDS["glare"],
        default_prompt=DEFAULT_NANO_PROMPTS["glare"],
        fallback_id="classical_glare",
    ),
    "rain": Generator(
        id="rain",
        kind="classical",
        classical_fn=classical_rain,
        default_severity=FIXED_CLASSICAL_SEVERITIES["rain"],
        default_seed=DEFAULT_CONDITION_SEEDS["rain"],
        default_prompt=DEFAULT_NANO_PROMPTS["rain"],
        fallback_id="classical_rain",
    ),
    "fog": Generator(
        id="fog",
        kind="classical",
        classical_fn=classical_fog,
        default_severity=FIXED_CLASSICAL_SEVERITIES["fog"],
        default_seed=DEFAULT_CONDITION_SEEDS["fog"],
        default_prompt=DEFAULT_NANO_PROMPTS["fog"],
        fallback_id="classical_fog",
    ),
    "occlusion": Generator(
        id="occlusion",
        kind="classical",
        classical_fn=classical_occlusion,
        default_severity=FIXED_CLASSICAL_SEVERITIES["occlusion"],
        default_seed=DEFAULT_CONDITION_SEEDS["occlusion"],
        default_prompt=None,
        fallback_id="classical_occlusion",
    ),
    "blur": Generator(
        id="blur",
        kind="classical",
        classical_fn=classical_blur,
        default_severity=FIXED_CLASSICAL_SEVERITIES["blur"],
        default_seed=DEFAULT_CONDITION_SEEDS["blur"],
        default_prompt=None,
        fallback_id="classical_blur",
    ),
    "smudge": Generator(
        id="smudge",
        kind="classical",
        classical_fn=classical_smudge,
        default_severity=FIXED_CLASSICAL_SEVERITIES["smudge"],
        default_seed=DEFAULT_CONDITION_SEEDS["smudge"],
        default_prompt=None,
        fallback_id="classical_smudge",
    ),
    # Explicit classical_* and nano_banana aliases and second scenario conditions
    "classical_night": Generator(
        id="classical_night",
        kind="classical",
        classical_fn=classical_night,
        default_severity=FIXED_CLASSICAL_SEVERITIES["night"],
        default_seed=DEFAULT_CONDITION_SEEDS["night"],
    ),
    "classical_glare": Generator(
        id="classical_glare",
        kind="classical",
        classical_fn=classical_glare,
        default_severity=FIXED_CLASSICAL_SEVERITIES["glare"],
        default_seed=DEFAULT_CONDITION_SEEDS["glare"],
    ),
    "classical_rain": Generator(
        id="classical_rain",
        kind="classical",
        classical_fn=classical_rain,
        default_severity=FIXED_CLASSICAL_SEVERITIES["rain"],
        default_seed=DEFAULT_CONDITION_SEEDS["rain"],
    ),
    "classical_fog": Generator(
        id="classical_fog",
        kind="classical",
        classical_fn=classical_fog,
        default_severity=FIXED_CLASSICAL_SEVERITIES["fog"],
        default_seed=DEFAULT_CONDITION_SEEDS["fog"],
    ),
    "classical_occlusion": Generator(
        id="classical_occlusion",
        kind="classical",
        classical_fn=classical_occlusion,
        default_severity=FIXED_CLASSICAL_SEVERITIES["occlusion"],
        default_seed=DEFAULT_CONDITION_SEEDS["occlusion"],
    ),
    "classical_blur": Generator(
        id="classical_blur",
        kind="classical",
        classical_fn=classical_blur,
        default_severity=FIXED_CLASSICAL_SEVERITIES["blur"],
        default_seed=DEFAULT_CONDITION_SEEDS["blur"],
    ),
    "classical_smudge": Generator(
        id="classical_smudge",
        kind="classical",
        classical_fn=classical_smudge,
        default_severity=FIXED_CLASSICAL_SEVERITIES["smudge"],
        default_seed=DEFAULT_CONDITION_SEEDS["smudge"],
    ),
    "classical_color_cast": Generator(
        id="classical_color_cast",
        kind="classical",
        classical_fn=classical_color_cast,
        default_severity=FIXED_CLASSICAL_SEVERITIES["color_cast"],
        default_seed=DEFAULT_CONDITION_SEEDS["color_cast"],
    ),
    "condensation": Generator(
        id="condensation",
        kind="classical",
        classical_fn=classical_rain,
        default_severity=FIXED_CLASSICAL_SEVERITIES["condensation"],
        default_seed=DEFAULT_CONDITION_SEEDS["condensation"],
        default_prompt=DEFAULT_NANO_PROMPTS["condensation"],
        fallback_id="classical_rain",
    ),
    "frost": Generator(
        id="frost",
        kind="classical",
        classical_fn=classical_fog,
        default_severity=FIXED_CLASSICAL_SEVERITIES["frost"],
        default_seed=DEFAULT_CONDITION_SEEDS["frost"],
        default_prompt=DEFAULT_NANO_PROMPTS["frost"],
        fallback_id="classical_fog",
    ),
    "led_glare": Generator(
        id="led_glare",
        kind="classical",
        classical_fn=classical_glare,
        default_severity=FIXED_CLASSICAL_SEVERITIES["led_glare"],
        default_seed=DEFAULT_CONDITION_SEEDS["led_glare"],
        default_prompt=DEFAULT_NANO_PROMPTS["led_glare"],
        fallback_id="classical_glare",
    ),
    "color_cast": Generator(
        id="color_cast",
        kind="classical",
        classical_fn=classical_color_cast,
        default_severity=FIXED_CLASSICAL_SEVERITIES["color_cast"],
        default_seed=DEFAULT_CONDITION_SEEDS["color_cast"],
        fallback_id="classical_color_cast",
    ),
    "dim": Generator(
        id="dim",
        kind="classical",
        classical_fn=classical_night,
        default_severity=FIXED_CLASSICAL_SEVERITIES["dim"],
        default_seed=DEFAULT_CONDITION_SEEDS["dim"],
        fallback_id="classical_night",
    ),
    "stacking": Generator(
        id="stacking",
        kind="classical",
        classical_fn=classical_occlusion,
        default_severity=FIXED_CLASSICAL_SEVERITIES["stacking"],
        default_seed=DEFAULT_CONDITION_SEEDS["stacking"],
        fallback_id="classical_occlusion",
    ),
}


def _load_scenario_condition_spec(scenario_id: str, condition_id: str) -> dict[str, Any]:
    scenario_path = Path("scenarios") / f"{scenario_id}.yaml"
    if scenario_path.exists():
        cfg = yaml.safe_load(scenario_path.read_text()) or {}
        for cond in cfg.get("conditions", []):
            if cond.get("id") == condition_id:
                return dict(cond)
    gen = CATALOG.get(condition_id)
    if gen is None:
        raise KeyError(f"Unknown condition_id: {condition_id}")
    return {
        "id": condition_id,
        "name": condition_id,
        "generator": f"classical_{condition_id}" if f"classical_{condition_id}" in CATALOG else condition_id,
        "fallback": gen.fallback_id or f"classical_{condition_id}",
        "severity": dict(gen.default_severity),
        "seed": gen.default_seed,
        "prompt": gen.default_prompt,
    }


def _load_split_lookup(scenario_id: str) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    """Return mapping of filename -> (split_label 'A'|'B', image_id 'a_001'|'b_001') and fp32_object_counts."""
    split_file = Path("data") / scenario_id / "splits.json"
    if not split_file.exists():
        return {}, {}
    data = json.loads(split_file.read_text())
    lookup: dict[str, tuple[str, str]] = {}
    for idx, fname in enumerate(data.get("split_a", []), start=1):
        lookup[fname] = ("A", f"a_{idx:03d}")
    for idx, fname in enumerate(data.get("split_b", []), start=1):
        lookup[fname] = ("B", f"b_{idx:03d}")
    counts = {k: int(v) for k, v in data.get("fp32_object_counts", {}).items()}
    return lookup, counts


def _rel_path_str(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _check_fp32_validity(
    detector: Detector | None,
    orig_image: Image.Image,
    cond_image: Image.Image,
    cached_d0: list[Det] | None = None,
    iou: float = 0.5,
) -> tuple[str, bool, int, int, int, float | None, list[Det]]:
    """Evaluate FP32 validity: valid iff |D0| >= 1 and |P| >= 1 (same class, IoU >= 0.5).
    When detector is None, never fabricate validated=True; mark as 'unverified_no_detector'.
    """
    if detector is None:
        return "unverified_no_detector", False, 0, 0, 0, None, []

    d0 = cached_d0 if cached_d0 is not None else detector.detect(orig_image)
    dc = detector.detect(cond_image)
    p = persisted(d0, dc, iou=iou)
    d0_count = len(d0)
    dc_count = len(dc)
    p_count = len(p)
    retention = round(float(p_count) / float(d0_count), 4) if d0_count > 0 else None

    if d0_count < 1:
        return "invalid_empty_d0", False, d0_count, dc_count, p_count, retention, d0
    if p_count < 1:
        return "invalid_no_persisted_objects", False, d0_count, dc_count, p_count, retention, d0
    return "valid", True, d0_count, dc_count, p_count, retention, d0


def generate(
    scenario_id: str,
    condition_id: str,
    image_paths: Sequence[str | Path],
    out_dir: str | Path,
    seed: int,
    *,
    mode: str = "classical",
    detector: Detector | None = None,
    split_override: str | None = None,
    concurrency: int = 8,
    max_regenerations: int = 2,
    enable_flash_audit: bool = True,
    reuse_existing_nano: bool = True,
    client: Any | None = None,
    nano_edit_fn: Callable[[Image.Image, str], tuple[Image.Image, str]] | None = None,
    flash_audit_fn: Callable[[Image.Image, Image.Image, str], tuple[bool, str, list[str], str]] | None = None,
    project_root: Path | None = None,
) -> list[ManifestEntry]:
    """Generate condition images for `image_paths` into `out_dir` and return a list of ManifestEntry records.

    Schedule & rule compliance:
    - Classical generators are defined for all 7 primary conditions (night, glare, rain, fog, occlusion, blur, smudge).
    - Split A always uses classical generators.
    - Under `mode="nano_banana"`, only split B (or explicit non-A split) images for Nano-enabled conditions
      run Nano Banana 2.1 edits (up to `max_regenerations` retries = 3 total attempts), resized to source size,
      checked for FP32 validity (|D0| >= 1, |P| >= 1) and Flash audit ("same objects in the same positions").
      On failure or quota/error, falls back to the deterministic classical generator and records the reason.
    - When `detector` is None, validity is recorded as `unverified_no_detector` (`valid=False`) rather than faking validation.
    """
    if condition_id not in CATALOG:
        raise ValueError(f"Condition '{condition_id}' not found in CATALOG.")

    root = (project_root or Path.cwd()).resolve()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    spec = _load_scenario_condition_spec(scenario_id, condition_id)
    base_gen = CATALOG[condition_id]
    fallback_gen_id = str(spec.get("fallback") or base_gen.fallback_id or f"classical_{condition_id}")
    if fallback_gen_id not in CATALOG:
        fallback_gen_id = condition_id

    # Fixed severity from scenario profile merged with fixed catalog constants
    severity: dict[str, Any] = dict(base_gen.default_severity)
    if spec.get("severity"):
        severity.update(spec["severity"])

    prompt_raw = spec.get("prompt") or base_gen.default_prompt
    prompt_str = ensure_preserve_clause(str(prompt_raw)) if prompt_raw else None

    split_lookup, _ = _load_split_lookup(scenario_id)

    # Pre-check existing manifest entries if reusing cached Nano Banana outputs
    existing_entries_by_file: dict[str, dict[str, Any]] = {}
    cond_manifest_file = out_path / "manifest.json"
    if reuse_existing_nano and cond_manifest_file.exists():
        try:
            prev_data = json.loads(cond_manifest_file.read_text())
            for item in prev_data.get("entries", []):
                if isinstance(item, dict) and item.get("file_name"):
                    existing_entries_by_file[str(item["file_name"])] = item
        except Exception:
            existing_entries_by_file = {}

    def _process_single(idx: int, src_p: str | Path) -> tuple[int, ManifestEntry]:
        src_path = Path(src_p)
        fname = src_path.name
        stem = src_path.stem

        if split_override is not None:
            split_label = split_override
            prefix = "a" if split_label.upper() == "A" else ("b" if split_label.upper() == "B" else "u")
            img_id = f"{prefix}_{idx + 1:03d}"
        elif fname in split_lookup:
            split_label, img_id = split_lookup[fname]
        else:
            split_label = "unassigned"
            img_id = f"img_{idx + 1:03d}"

        img_seed = derive_image_seed(seed, condition_id, fname)
        dst_path = out_path / fname

        with Image.open(src_path) as raw_im:
            orig_im = raw_im.convert("RGB")
        orig_w, orig_h = orig_im.size

        # Determine whether this image should attempt Nano Banana or run classical directly
        wants_nano = (
            mode == "nano_banana"
            and split_label != "A"
            and condition_id in ("night", "glare")
            and prompt_str is not None
        )
        requested_gen = NANO_BANANA_MODEL if wants_nano else fallback_gen_id

        # If a valid Nano Banana output already exists on disk and reuse_existing_nano is enabled, reuse it
        if wants_nano and reuse_existing_nano and dst_path.exists() and fname in existing_entries_by_file:
            prev = existing_entries_by_file[fname]
            if prev.get("generator") == NANO_BANANA_MODEL and prev.get("valid") is True and prev.get("audit_passed") is True:
                return idx, ManifestEntry(**{k: v for k, v in prev.items() if k in ManifestEntry.__annotations__})

        interaction_ids: list[str] = []
        nano_attempts = 0
        nano_api_calls = 0
        flash_api_calls = 0
        regeneration_count = 0
        used_fallback = False
        fallback_reason: str | None = None
        audit_status = "not_applicable"
        audit_passed: bool | None = None
        audit_reason = "Classical deterministic generator (Flash audit not required)."
        audit_diffs: list[str] = []
        cached_d0: list[Det] | None = None

        final_im: Image.Image | None = None
        actual_gen = fallback_gen_id
        actual_gen_label = f"classical ({fallback_gen_id})"

        if wants_nano and prompt_str is not None:
            max_total_attempts = 1 + max(0, int(max_regenerations))
            last_fail_reason = "unknown_nano_failure"

            for attempt_idx in range(max_total_attempts):
                nano_attempts += 1
                if attempt_idx > 0:
                    regeneration_count += 1
                try:
                    nano_api_calls += 1
                    if nano_edit_fn is not None:
                        cand_im, call_id = nano_edit_fn(orig_im, prompt_str)
                    else:
                        cand_im, call_id = call_nano_banana_edit(orig_im, prompt_str, client=client)
                    if call_id:
                        interaction_ids.append(call_id)
                except Exception as exc:
                    err_name = type(exc).__name__
                    last_fail_reason = f"nano_api_error:{err_name}"
                    break

                if cand_im.size != (orig_w, orig_h):
                    cand_im = cand_im.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

                v_status, is_valid, d0_c, dc_c, p_c, ret_val, d0_list = _check_fp32_validity(
                    detector, orig_im, cand_im, cached_d0=cached_d0
                )
                if detector is not None:
                    cached_d0 = d0_list

                if detector is not None and not is_valid:
                    last_fail_reason = f"validity_gate_failed:{v_status}(d0={d0_c},persisted={p_c})"
                    continue

                # Run Flash audit on Split B Nano Banana outputs
                if enable_flash_audit:
                    try:
                        flash_api_calls += 1
                        if flash_audit_fn is not None:
                            f_ok, f_reason, f_diffs, f_id = flash_audit_fn(orig_im, cand_im, condition_id)
                        else:
                            f_ok, f_reason, f_diffs, f_id = call_flash_audit(
                                orig_im, cand_im, condition_id, client=client
                            )
                        if f_id:
                            interaction_ids.append(f_id)
                        audit_diffs = f_diffs
                        if f_ok:
                            audit_status = "passed"
                            audit_passed = True
                            audit_reason = f_reason
                            final_im = cand_im
                            actual_gen = NANO_BANANA_MODEL
                            actual_gen_label = f"Nano Banana 2.1 ({NANO_BANANA_MODEL})"
                            break
                        else:
                            audit_status = "failed"
                            audit_passed = False
                            audit_reason = f_reason
                            last_fail_reason = f"flash_audit_rejected:{f_reason}"
                            continue
                    except Exception as audit_exc:
                        # Explicit design fallback gate if Flash quota/API is unavailable: retention >= 0.5
                        err_name = type(audit_exc).__name__
                        if ret_val is not None and ret_val >= 0.5:
                            audit_status = "fallback_retention_gate_passed"
                            audit_passed = None
                            audit_reason = (
                                f"Flash audit unavailable ({err_name}); passed explicit retention>=0.5 fallback gate "
                                f"(retention={ret_val:.4f}). Not a Flash audit pass."
                            )
                            final_im = cand_im
                            actual_gen = NANO_BANANA_MODEL
                            actual_gen_label = f"Nano Banana 2.1 ({NANO_BANANA_MODEL}, retention-fallback-gate)"
                            break
                        else:
                            audit_status = "fallback_retention_gate_failed"
                            audit_passed = False
                            audit_reason = (
                                f"Flash audit unavailable ({err_name}) and retention fallback gate failed "
                                f"(retention={ret_val})."
                            )
                            last_fail_reason = f"flash_unavailable_and_low_retention:{err_name}"
                            continue
                else:
                    audit_status = "skipped_by_config"
                    audit_passed = None
                    audit_reason = "Flash audit disabled by caller configuration."
                    final_im = cand_im
                    actual_gen = NANO_BANANA_MODEL
                    actual_gen_label = f"Nano Banana 2.1 ({NANO_BANANA_MODEL})"
                    break

            if final_im is None:
                used_fallback = True
                fallback_reason = last_fail_reason
                if audit_status == "not_applicable":
                    audit_status = "nano_failed_used_classical_fallback"
                    audit_reason = f"Fell back to {fallback_gen_id}: {last_fail_reason}"

        if final_im is None:
            final_im = base_gen.apply_classical(orig_im, severity=severity, seed=img_seed)
            if final_im.size != (orig_w, orig_h):
                final_im = final_im.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            actual_gen = fallback_gen_id
            actual_gen_label = "classical (deterministic) in this run"

        final_im.save(dst_path, format="JPEG", quality=92)
        out_w, out_h = final_im.size

        v_status, is_valid, d0_c, dc_c, p_c, ret_val, _ = _check_fp32_validity(
            detector, orig_im, final_im, cached_d0=cached_d0
        )

        # Determine pipeline admissibility for Block D/E consumers
        if detector is None:
            allow_in_pipeline = False
        else:
            allow_in_pipeline = bool(is_valid)

        rel_orig = _rel_path_str(src_path, root)
        rel_cond = _rel_path_str(dst_path, root)

        entry = ManifestEntry(
            image_id=img_id,
            source_image_id=stem,
            file_name=fname,
            split=split_label,
            scenario_id=scenario_id,
            condition_id=condition_id,
            original_path=rel_orig,
            condition_path=rel_cond,
            thumb_url=f"/artifacts/{rel_cond}",
            original_url=f"/artifacts/{rel_orig}",
            requested_generator=requested_gen,
            generator=actual_gen,
            generator_label=actual_gen_label,
            seed=int(seed),
            image_seed=int(img_seed),
            severity=severity,
            source_size=[orig_w, orig_h],
            output_size=[out_w, out_h],
            valid=is_valid,
            validity_status=v_status,
            allow_in_pipeline=allow_in_pipeline,
            d0_count=d0_c,
            dc_count=dc_c,
            persisted_count=p_c,
            retention=ret_val,
            audit_status=audit_status,
            audit_passed=audit_passed,
            audit_reason=audit_reason,
            audit_differences=audit_diffs,
            regeneration_count=regeneration_count,
            nano_attempts=nano_attempts,
            nano_api_calls=nano_api_calls,
            flash_api_calls=flash_api_calls,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            prompt=prompt_str if wants_nano else None,
            interaction_ids=interaction_ids,
            path_root="project_root",
        )
        return idx, entry

    results: list[tuple[int, ManifestEntry]] = []
    max_workers = max(1, int(concurrency)) if mode == "nano_banana" else 1
    if max_workers > 1 and len(image_paths) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_single, i, p) for i, p in enumerate(image_paths)]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for i, p in enumerate(image_paths):
            results.append(_process_single(i, p))

    results.sort(key=lambda x: x[0])
    ordered_entries = [entry for _, entry in results]

    # Merge with any existing entries in out_dir/manifest.json (so Split B Nano updates preserve Split A classical entries)
    merged_by_file: dict[str, dict[str, Any]] = {}
    if cond_manifest_file.exists():
        try:
            existing_payload = json.loads(cond_manifest_file.read_text())
            for item in existing_payload.get("entries", []):
                if isinstance(item, dict) and item.get("file_name"):
                    merged_by_file[str(item["file_name"])] = item
        except Exception:
            merged_by_file = {}

    for entry in ordered_entries:
        merged_by_file[entry.file_name] = entry.to_dict()

    all_cond_entries = [merged_by_file[k] for k in sorted(merged_by_file.keys())]
    cond_summary = summarize_entries(all_cond_entries, scenario_id=scenario_id, condition_id=condition_id)
    cond_manifest_file.write_text(json.dumps(cond_summary, indent=2) + chr(10))

    return ordered_entries


def summarize_entries(
    entries: Sequence[dict[str, Any] | ManifestEntry],
    scenario_id: str,
    condition_id: str,
) -> dict[str, Any]:
    dict_entries = [e.to_dict() if isinstance(e, ManifestEntry) else dict(e) for e in entries]
    split_a = [e for e in dict_entries if e.get("split") == "A"]
    split_b = [e for e in dict_entries if e.get("split") == "B"]

    def _agg(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        valid_n = sum(1 for e in sub if e.get("valid") is True)
        nano_n = sum(1 for e in sub if e.get("generator") == NANO_BANANA_MODEL)
        classical_n = n - nano_n
        fallback_n = sum(1 for e in sub if e.get("used_fallback") is True)
        regen_total = sum(int(e.get("regeneration_count", 0)) for e in sub)
        nano_calls = sum(int(e.get("nano_api_calls", 0)) for e in sub)
        flash_calls = sum(int(e.get("flash_api_calls", 0)) for e in sub)
        audit_passed_n = sum(1 for e in sub if e.get("audit_status") == "passed")
        audit_failed_n = sum(1 for e in sub if e.get("audit_status") == "failed")
        d0_sum = sum(int(e.get("d0_count", 0)) for e in sub)
        p_sum = sum(int(e.get("persisted_count", 0)) for e in sub)
        ret = round(float(p_sum) / float(d0_sum), 4) if d0_sum > 0 else None
        gens = sorted({str(e.get("generator")) for e in sub if e.get("generator")})
        return {
            "count": n,
            "valid_count": valid_n,
            "invalid_count": sum(1 for e in sub if str(e.get("validity_status", "")).startswith("invalid_")),
            "unverified_count": sum(1 for e in sub if e.get("validity_status") == "unverified_no_detector"),
            "nano_banana_count": nano_n,
            "classical_count": classical_n,
            "fallback_count": fallback_n,
            "regeneration_count": regen_total,
            "nano_api_calls": nano_calls,
            "flash_api_calls": flash_calls,
            "audit_passed_count": audit_passed_n,
            "audit_failed_count": audit_failed_n,
            "orig_objects": d0_sum,
            "persisted_objects": p_sum,
            "retention": ret,
            "generators_used": gens,
        }

    return {
        "scenario_id": scenario_id,
        "condition_id": condition_id,
        "path_root": "project_root",
        "total_entries": len(dict_entries),
        "split_a_summary": _agg(split_a),
        "split_b_summary": _agg(split_b),
        "overall_summary": _agg(dict_entries),
        "entries": dict_entries,
    }
