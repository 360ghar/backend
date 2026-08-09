"""
Tests for the metadata-driven panorama module (pure numpy math).

Covers geometry (basis, roll), blending (coverage, roll alignment), gains,
and the quality metrics + validation gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.tour_ai import panorama


def _textured_frame(width: int = 320, height: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    # Bright vertical marker at the center column so frame positions are
    # checkable in the output.
    image[:, width // 2, :] = 255
    return image


def _euler_basis(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """INDEPENDENT derivation of the camera basis using standard Euler
    composition R = Rz(-yaw) @ Rx(pitch) @ Ry(roll) applied to the level
    camera axes. Used only by tests to pin the production convention
    (panorama.camera_basis) and to render synthetic frames without sharing
    the production code path."""

    def _rot(axis: str, rad: float) -> np.ndarray:
        c, s = math.cos(rad), math.sin(rad)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    yaw, pitch, roll = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    r = _rot("z", -yaw) @ _rot("x", pitch) @ _rot("y", roll)
    right = r @ np.array([1.0, 0.0, 0.0])
    up = r @ np.array([0.0, 0.0, 1.0])
    fwd = r @ np.array([0.0, 1.0, 0.0])
    return np.stack([right, up, fwd])


def _render_from_world_field(
    field, width: int, height: int, h_fov: float, v_fov: float,
    yaw_deg: float, pitch_deg: float, roll_deg: float,
) -> np.ndarray:
    """Renders a world-direction field (callable: Vec3 -> luma 0..255)
    through an independent pose. The field value at world direction d is
    THE SAME in every frame — this is what makes overlap agreement a real
    geometry test (real frames photograph the same world)."""
    basis = _euler_basis(yaw_deg, pitch_deg, roll_deg)
    right, up, fwd = basis
    tan_h = math.tan(math.radians(h_fov) / 2.0)
    tan_v = math.tan(math.radians(v_fov) / 2.0)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for py in range(height):
        sy = 1.0 - 2.0 * py / (height - 1)
        for px in range(width):
            sx = 2.0 * px / (width - 1) - 1.0
            d = right * (sx * tan_h) + up * (sy * tan_v) + fwd
            d /= np.linalg.norm(d)
            v = int(np.clip(field(d), 0, 255))
            image[py, px] = (v, v, v)
    return image


def _pitch_field(d: np.ndarray) -> float:
    """World field depending only on latitude — the roll test's content."""
    pitch = math.degrees(math.asin(float(d[2])))
    return (pitch + 90.0) / 180.0 * 255.0


def _frame(yaw: float, pitch: float = 0.0, roll: float = 0.0, seed: int = 0, **kw) -> panorama.FrameSpec:
    return panorama.FrameSpec(
        image=_textured_frame(seed=seed),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        **kw,
    )


class TestCameraBasis:
    def test_level_pose_is_orthonormal(self):
        basis = panorama.camera_basis(37.0, 12.0, 0.0)
        right, up, fwd = basis
        for a, b in [(right, up), (up, fwd), (fwd, right)]:
            assert abs(float(np.dot(a, b))) < 1e-9
        assert np.allclose(np.cross(right, up), -fwd, atol=1e-9)

    def test_roll_rotates_image_plane_only(self):
        basis = panorama.camera_basis(0.0, 0.0, 10.0)
        right, up, fwd = basis
        # Optical axis unchanged by roll.
        assert np.allclose(fwd, [0.0, 1.0, 0.0], atol=1e-12)
        # Positive roll (top tilted right) tips the image "up" toward +x.
        assert up[0] > 0.1
        # Handedness preserved: right' x up' == right x up.
        level = panorama.camera_basis(0.0, 0.0, 0.0)
        assert np.allclose(
            np.cross(right, up), np.cross(level[0], level[1]), atol=1e-9
        )

    def test_roll_matches_mobile_sign_convention(self):
        # Mobile invariant: top-center pixel of a level rolled frame lands at
        # yaw ~= atan2(tan(vFov/2) * sin(roll), 1) (positive for positive roll).
        roll_deg = 10.0
        v_fov = 69.0
        basis = panorama.camera_basis(0.0, 0.0, roll_deg)
        tan_v = math.tan(math.radians(v_fov) / 2.0)
        direction = basis[1] * tan_v + basis[2]
        yaw = math.degrees(math.atan2(direction[0], direction[1]))
        expected = math.degrees(math.atan2(math.sin(math.radians(roll_deg)) * tan_v, 1.0))
        assert yaw == pytest.approx(expected, abs=0.5)

    @pytest.mark.parametrize("pose", [(0, 0, 0), (37, 12, 0), (0, 0, 10), (45, 30, -15), (-120, -20, 5)])
    def test_production_basis_matches_independent_euler(self, pose):
        # Pins panorama.camera_basis against a standard Euler composition.
        assert np.allclose(
            panorama.camera_basis(*pose), _euler_basis(*pose), atol=1e-9
        )


class TestGains:
    def test_pulls_toward_median_and_clamps(self):
        gains = panorama.compute_gains([200.0, 100.0, 96.0], min_gain=0.85, max_gain=1.25)
        # Target = 100 (median): 200 -> 0.5 clamped to 0.85; 96 -> ~1.04.
        assert gains[0] == pytest.approx(0.85)
        assert gains[1] == pytest.approx(1.0)
        assert gains[2] == pytest.approx(100.0 / 96.0, rel=1e-3)

    def test_all_black_frames_get_identity_gains(self):
        assert panorama.compute_gains([0.0, 0.0]) == [1.0, 1.0]

    def test_exposure_score(self):
        assert panorama.exposure_score([1.0, 1.0, 1.0]) == 100.0
        assert panorama.exposure_score([0.85, 1.25]) < 60.0


class TestFrameWeight:
    def test_center_beats_edge_and_low_quality_halves(self):
        center = panorama.frame_weight(np.array([0.0]), np.array([0.0]), low_quality=False)
        edge = panorama.frame_weight(np.array([0.95]), np.array([0.0]), low_quality=False)
        assert center[0] > edge[0]
        half = panorama.frame_weight(np.array([0.0]), np.array([0.0]), low_quality=True)
        assert half[0] == pytest.approx(center[0] * 0.5)

    def test_weight_is_monotonic_falloff(self):
        values = [
            panorama.frame_weight(np.array([u]), np.array([0.0]), low_quality=False)[0]
            for u in (0.0, 0.3, 0.6, 0.9)
        ]
        assert values == sorted(values, reverse=True)


class TestBlend:
    def test_horizon_ring_covers_full_circle(self):
        frames = [_frame(yaw, seed=i) for i, yaw in enumerate([0, 90, 180, 270])]
        pano, stats, _ = panorama.blend_equirect(frames, h_fov=100.0, v_fov=69.0, width=512, height=256)
        assert pano.shape == (256, 512, 3)
        # 4 x 100°-wide frames tile the azimuth but only cover the
        # ±34.5° latitude band: area coverage ≈ 69/180 ≈ 38%.
        assert stats.covered_pixels / stats.total_pixels > 0.35
        # Marker of each frame lands at its yaw on the horizon row.
        mid_y = 128
        for yaw in (0, 90, 180, 270):
            x = int(((yaw + 180) / 360) * 512) % 512
            assert pano[mid_y, x].sum() > 200

    def test_roll_geometry_is_consistent(self):
        # Two rolled frames of the SAME world field (latitude gradient) at
        # nearly the same yaw: with correct roll handling they agree on the
        # overlap (high seam score). Telling the stitcher the WRONG roll
        # sign must measurably break that agreement.
        h_fov, v_fov = 60.0, 69.0
        frame_a = _render_from_world_field(_pitch_field, 320, 400, h_fov, v_fov, 0.0, 0.0, 10.0)
        frame_b = _render_from_world_field(_pitch_field, 320, 400, h_fov, v_fov, 1.0, 0.0, -10.0)
        correct = [
            panorama.FrameSpec(image=frame_a, yaw_deg=0.0, roll_deg=10.0),
            panorama.FrameSpec(image=frame_b, yaw_deg=1.0, roll_deg=-10.0),
        ]
        _, stats_ok, _ = panorama.blend_equirect(correct, h_fov=h_fov, width=512, height=256)
        assert panorama.seam_score(stats_ok) >= 90

        flipped = [
            panorama.FrameSpec(image=frame_a, yaw_deg=0.0, roll_deg=-10.0),
            panorama.FrameSpec(image=frame_b, yaw_deg=1.0, roll_deg=10.0),
        ]
        _, stats_bad, _ = panorama.blend_equirect(flipped, h_fov=h_fov, width=512, height=256)
        assert panorama.seam_score(stats_bad) < 70

    def test_pole_rings_cover_upper_and_lower_hemispheres(self):
        frames = []
        for yaw_i in range(10):
            frames.append(_frame(yaw=yaw_i * 36.0, seed=yaw_i))
        for yaw_i in range(7):
            frames.append(_frame(yaw=yaw_i * 360.0 / 7, pitch=55.5, seed=100 + yaw_i))
            frames.append(_frame(yaw=yaw_i * 360.0 / 7, pitch=-55.5, seed=200 + yaw_i))
        pano, stats, _ = panorama.blend_equirect(frames, h_fov=55.0, v_fov=69.0, width=512, height=256)
        metrics = panorama.metrics_from_stats(stats, [], 80.0, 512, 256, 0)
        assert metrics["coverage_percent"] >= 99
        assert metrics["vertical_coverage_percent"] >= 95

    def test_missing_frames_leave_honest_gaps(self):
        # 3 frames at 120° spacing with a 55° FOV cover the azimuth only
        # inside the ±34.5° band: ~3 x 55/360 x 69/180 ≈ 17% of the canvas.
        frames = [_frame(yaw, seed=i) for i, yaw in enumerate([0, 120, 240])]
        pano, stats, _ = panorama.blend_equirect(frames, h_fov=55.0, v_fov=69.0, width=512, height=256)
        coverage = 100.0 * stats.covered_pixels / stats.total_pixels
        assert 10 < coverage < 25
        metrics = panorama.metrics_from_stats(stats, [1.0, 1.0, 1.0], 80.0, 512, 256, 0)
        assert any("Coverage incomplete" in w for w in metrics["warnings"])

    def test_skipped_and_low_quality_frames_still_work(self):
        # One skipped frame (gap), one low-quality frame: the blend still
        # succeeds and the low-quality frame contributes at reduced weight.
        frames = [
            _frame(0.0, seed=0),
            _frame(45.0, seed=1),
            _frame(90.0, seed=2, low_quality=True),
            _frame(135.0, seed=3),
        ]
        pano, stats, _ = panorama.blend_equirect(frames, h_fov=60.0, width=512, height=256)
        assert stats.covered_pixels > 0
        assert pano.dtype == np.uint8


class TestQuality:
    def test_sharpness_separates_texture_from_flat(self):
        textured = _textured_frame()
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert panorama.sharpness_score(textured) > 70
        assert panorama.sharpness_score(flat) < 20

    def test_seam_score_agreement_and_clash(self):
        # Frames rendered from the SAME world field (photos of the same
        # world at different poses) must agree in their overlap: high seam
        # score. A frame of the INVERTED world (different scene) must clash.
        h_fov, v_fov = 60.0, 69.0
        agree_frames = [
            panorama.FrameSpec(
                image=_render_from_world_field(_pitch_field, 320, 400, h_fov, v_fov, 0.0, 0.0, 0.0),
                yaw_deg=0.0,
            ),
            panorama.FrameSpec(
                image=_render_from_world_field(_pitch_field, 320, 400, h_fov, v_fov, 30.0, 0.0, 0.0),
                yaw_deg=30.0,
            ),
        ]
        def inverted(d: np.ndarray) -> float:
            return 255.0 - _pitch_field(d)

        clash_frames = [
            panorama.FrameSpec(
                image=_render_from_world_field(_pitch_field, 320, 400, h_fov, v_fov, 0.0, 0.0, 0.0),
                yaw_deg=0.0,
            ),
            panorama.FrameSpec(
                image=_render_from_world_field(inverted, 320, 400, h_fov, v_fov, 30.0, 0.0, 0.0),
                yaw_deg=30.0,
            ),
        ]
        _, stats_ok, _ = panorama.blend_equirect(agree_frames, h_fov=h_fov, width=512, height=256)
        _, stats_bad, _ = panorama.blend_equirect(clash_frames, h_fov=h_fov, width=512, height=256)
        assert panorama.seam_score(stats_ok) >= 95
        assert panorama.seam_score(stats_bad) <= 30

    def test_validate_equirect(self):
        good = np.full((512, 1024, 3), 128, dtype=np.uint8)
        assert panorama.validate_equirect(good, 99.0) == []
        bad_shape = np.full((400, 500, 3), 128, dtype=np.uint8)
        problems = panorama.validate_equirect(bad_shape, 99.0)
        assert any("2:1" in p for p in problems)
        assert any("coverage" in p for p in panorama.validate_equirect(good, 30.0))
