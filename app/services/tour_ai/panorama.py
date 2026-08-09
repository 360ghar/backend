"""
Metadata-driven equirectangular panorama reconstruction (numpy).

Port of the mobile stitching modules (panorama_geometry / panorama_blending /
panorama_quality) so the cloud refinement path produces the same geometry
with the same honest quality metrics. Pure functions only — no DB, no I/O.

Conventions match the mobile orientation engine: world x=east, y=north,
z=up; positive roll = phone top tilted right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# BGR luma weights (cv2 frames are BGR).
_LUMA_BGR = np.array([0.114, 0.587, 0.299])

OUT_WIDTH = 2048
OUT_HEIGHT = 1024

# Quality-gate thresholds for cloud refinement (see stitch.py).
MIN_COVERAGE_PERCENT = 60.0
MIN_OVERALL_SCORE = 40.0


@dataclass(frozen=True)
class FrameSpec:
    """One captured frame ready for blending (BGR working copy)."""
    image: np.ndarray
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    low_quality: bool = False


@dataclass(frozen=True)
class BlendStats:
    """Deterministic per-pixel statistics of a blend."""
    total_pixels: int
    covered_pixels: int
    top_total: int
    top_covered: int
    middle_total: int
    middle_covered: int
    bottom_total: int
    bottom_covered: int
    seam_disagreement: float
    seam_weight: float


def camera_basis(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """3x3 matrix whose rows are the camera's world-space right/up/forward
    basis vectors (level pose matches the mobile naive-stitcher convention;
    roll rotates the image plane about the optical axis)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = np.array([math.sin(yaw) * cp, math.cos(yaw) * cp, sp])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    up = np.array([-math.sin(yaw) * sp, -math.cos(yaw) * sp, cp])
    cr, sr = math.cos(roll), math.sin(roll)
    right_p = cr * right - sr * up
    up_p = sr * right + cr * up
    return np.stack([right_p, up_p, fwd])


def compute_gains(mean_lumas: list[float], min_gain: float = 0.85, max_gain: float = 1.25) -> list[float]:
    """Per-frame luminance gains pulling each frame's mean toward the median
    of all means, clamped so correction stays natural and cannot clip."""
    if not mean_lumas:
        return []
    target = float(np.median(mean_lumas))
    if target <= 1e-6:
        return [1.0] * len(mean_lumas)
    return [
        (1.0 if m <= 1e-6 else float(np.clip(target / m, min_gain, max_gain)))
        for m in mean_lumas
    ]


def frame_weight(sx: np.ndarray, sy: np.ndarray, low_quality: bool) -> np.ndarray:
    """Contribution weight at normalized coords in [-1, 1]: center (gaussian
    falloff) × edge (taper over the outer 15%) × quality (0.5 for
    low-quality frames — reduced, never removed)."""
    r2 = sx * sx + sy * sy
    center = np.exp(-r2 * 1.6)
    edge_x = np.clip(1.0 - (np.abs(sx) - 0.85) / 0.15, 0.0, 1.0)
    edge_y = np.clip(1.0 - (np.abs(sy) - 0.85) / 0.15, 0.0, 1.0)
    quality = 0.5 if low_quality else 1.0
    return center * edge_x * edge_y * quality


def blend_equirect(
    frames: list[FrameSpec],
    h_fov: float = 55.0,
    v_fov: float = 69.0,
    width: int = OUT_WIDTH,
    height: int = OUT_HEIGHT,
    gains: list[float] | None = None,
) -> tuple[np.ndarray, BlendStats, list[float]]:
    """Blend [frames] into an equirect panorama.

    Returns (panorama BGR uint8, stats, gains). Uncovered pixels stay black —
    honestly uncovered, never fabricated.
    """
    if not frames:
        raise ValueError("no frames to blend")
    if gains is None:
        means = [float(np.mean(cv2_gray(f.image))) for f in frames]
        gains = compute_gains(means)
    if len(gains) != len(frames):
        gains = [1.0] * len(frames)

    xs = np.arange(width)
    ys = np.arange(height)
    lon = (xs / width) * 2.0 * math.pi - math.pi
    lat = math.pi / 2.0 - (ys / height) * math.pi
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    # World direction per pixel (H, W, 3).
    d_x = np.outer(cos_lat, sin_lon)
    d_y = np.outer(cos_lat, cos_lon)
    d_z = np.repeat(sin_lat[:, None], width, axis=1)
    directions = np.stack([d_x, d_y, d_z], axis=-1)

    tan_h = math.tan(math.radians(h_fov) / 2.0)
    tan_v = math.tan(math.radians(v_fov) / 2.0)

    sum_bgr = np.zeros((height, width, 3), dtype=np.float32)
    sum_w = np.zeros((height, width), dtype=np.float32)
    best_w = np.zeros((height, width), dtype=np.float32)
    best_lum = np.zeros((height, width), dtype=np.float32)
    sec_w = np.zeros((height, width), dtype=np.float32)
    sec_lum = np.zeros((height, width), dtype=np.float32)

    for frame, gain in zip(frames, gains, strict=False):
        basis = camera_basis(frame.yaw_deg, frame.pitch_deg, frame.roll_deg)
        right, up, fwd = basis[0], basis[1], basis[2]
        fwd_dot = directions @ fwd
        behind = fwd_dot <= 0.05
        safe = np.where(behind, 1.0, fwd_dot)
        u = (directions @ right) / safe
        v = (directions @ up) / safe
        sx = u / tan_h
        sy = v / tan_v
        inside = ~behind & (np.abs(sx) <= 1.0) & (np.abs(sy) <= 1.0)

        # cos(off-axis) Jacobian: grazing regions project onto larger
        # equirect areas, so their per-pixel weight is scaled down.
        weight = frame_weight(sx, sy, frame.low_quality) * fwd_dot
        weight = np.where(inside, weight, 0.0)

        # Vectorized bilinear sample with per-frame exposure gain.
        img_h, img_w = frame.image.shape[:2]
        fx = (sx + 1.0) / 2.0 * (img_w - 1.0)
        fy = (1.0 - sy) / 2.0 * (img_h - 1.0)
        x0 = np.clip(np.floor(fx).astype(np.int64), 0, img_w - 1)
        y0 = np.clip(np.floor(fy).astype(np.int64), 0, img_h - 1)
        x1 = np.clip(x0 + 1, 0, img_w - 1)
        y1 = np.clip(y0 + 1, 0, img_h - 1)
        tx = fx - x0
        ty = fy - y0
        img_f = frame.image.astype(np.float32) * gain
        c00 = img_f[y0, x0]
        c10 = img_f[y0, x1]
        c01 = img_f[y1, x0]
        c11 = img_f[y1, x1]
        sampled = (
            (c00 * (1.0 - tx[..., None]) + c10 * tx[..., None]) * (1.0 - ty[..., None])
            + (c01 * (1.0 - tx[..., None]) + c11 * tx[..., None]) * ty[..., None]
        )
        sampled = np.clip(sampled, 0.0, 255.0)

        active = weight > 0.001
        active_w = np.where(active, weight, 0.0)
        sum_bgr += sampled * active_w[..., None]
        sum_w += active_w

        lum = sampled @ _LUMA_BGR
        better = weight > best_w
        sec_w = np.where(better, best_w, sec_w)
        sec_lum = np.where(better, best_lum, sec_lum)
        best_w = np.maximum(best_w, weight)
        best_lum = np.where(better, lum, best_lum)
        second = (~better) & (weight > sec_w)
        sec_w = np.where(second, weight, sec_w)
        sec_lum = np.where(second, lum, sec_lum)

    covered = sum_w > 0.0
    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    out = np.where(covered[..., None], sum_bgr / np.where(covered, sum_w, 1.0)[..., None], 0.0)
    panorama[:] = np.clip(out, 0.0, 255.0)

    stats = _stats_from_accumulators(covered, best_w, best_lum, sec_w, sec_lum, height, width)
    return panorama, stats, gains


def _stats_from_accumulators(
    covered: np.ndarray,
    best_w: np.ndarray,
    best_lum: np.ndarray,
    sec_w: np.ndarray,
    sec_lum: np.ndarray,
    height: int,
    width: int,
) -> BlendStats:
    top_band = height // 5
    bot_band = height - height // 5
    rows = np.arange(height)
    top_rows = rows < top_band
    bot_rows = rows >= bot_band

    total = height * width
    def band_covered(row_mask: np.ndarray) -> int:
        return int(np.count_nonzero(covered[row_mask, :]))

    overlap = np.minimum(best_w, sec_w)
    disagreement = float(np.sum(overlap * np.abs(best_lum - sec_lum)))
    seam_weight = float(np.sum(overlap))

    return BlendStats(
        total_pixels=total,
        covered_pixels=int(np.count_nonzero(covered)),
        top_total=int(top_band * width),
        top_covered=band_covered(top_rows),
        middle_total=int((bot_band - top_band) * width),
        middle_covered=band_covered(~top_rows & ~bot_rows),
        bottom_total=int((height - bot_band) * width),
        bottom_covered=band_covered(bot_rows),
        seam_disagreement=disagreement,
        seam_weight=seam_weight,
    )


def seam_score(stats: BlendStats) -> float:
    """0..100: 100 when overlapping contributors agree (<12% luma diff)."""
    if stats.seam_weight <= 0:
        return 100.0
    mean_disagreement = stats.seam_disagreement / stats.seam_weight / 255.0
    return round(100.0 - min(100.0, 100.0 * mean_disagreement / 0.12), 1)


def exposure_score(gains: list[float]) -> float:
    """0..100: how much gain correction the frames needed."""
    if not gains:
        return 100.0
    cost = float(np.mean([abs(g - 1.0) for g in gains]))
    return round(100.0 - min(100.0, 100.0 * cost / 0.30), 1)


def sharpness_score(image: np.ndarray, sample_size: int = 256) -> float:
    """0..100: monotone compression of the variance of Laplacian."""
    import cv2

    small = image
    if small.shape[1] > sample_size:
        scale = sample_size / small.shape[1]
        small = cv2.resize(
            small,
            (sample_size, max(1, int(small.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    return round(100.0 * variance / (variance + 400.0), 1)


def metrics_from_stats(
    stats: BlendStats,
    gains: list[float],
    sharpness: float,
    width: int,
    height: int,
    low_quality_count: int,
) -> dict:
    """Structured quality report for a blend (mirrors the mobile
    PanoramaMetrics)."""
    coverage = 100.0 * stats.covered_pixels / stats.total_pixels
    horizontal = 100.0 * stats.middle_covered / stats.middle_total
    vertical = min(
        100.0 * stats.top_covered / stats.top_total,
        100.0 * stats.bottom_covered / stats.bottom_total,
    )
    seam = seam_score(stats)
    exposure = exposure_score(gains)
    overall = round(min(100.0, 0.45 * coverage + 0.25 * seam + 0.15 * exposure + 0.15 * sharpness), 1)

    warnings = []
    if coverage < 95:
        warnings.append(f"Coverage incomplete ({round(coverage)}%)")
    if vertical < 90:
        warnings.append(f"Upper/lower coverage low ({round(vertical)}%) — capture more up/down shots")
    if seam < 70:
        warnings.append("Visible seam discontinuities detected")
    if low_quality_count:
        warnings.append(f"{low_quality_count} low-quality frame(s) contributed")

    return {
        "width": width,
        "height": height,
        "coverage_percent": round(coverage, 1),
        "horizontal_coverage_percent": round(horizontal, 1),
        "vertical_coverage_percent": round(vertical, 1),
        "seam_score": seam,
        "exposure_score": exposure,
        "sharpness_score": sharpness,
        "overall_score": overall,
        "warnings": warnings,
    }


def validate_equirect(image: np.ndarray, coverage_percent: float) -> list[str]:
    """Hard checks a cloud stitch must pass before replacing a scene image.
    Returns a list of problems (empty = valid)."""
    import cv2

    problems = []
    if image is None or image.size == 0:
        problems.append("stitched image is empty")
        return problems
    height, width = image.shape[:2]
    if width != 2 * height:
        problems.append(f"output is not 2:1 equirect ({width}x{height})")
    if coverage_percent < MIN_COVERAGE_PERCENT:
        problems.append(f"coverage {coverage_percent:.1f}% below minimum {MIN_COVERAGE_PERCENT}%")
    ok, _ = cv2.imencode(".jpg", image)
    if not ok:
        problems.append("output cannot be encoded as JPEG")
    return problems


def cv2_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale of a BGR frame (lazy cv2 import keeps the module light for
    pure-math tests)."""
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
