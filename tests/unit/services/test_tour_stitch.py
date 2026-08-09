"""
Tests for cloud panorama stitching.

Covers frame decoding/downscaling, the OpenCV stitch wrapper (mocked
Stitcher for deterministic success/failure), and the background runner.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tours import Scene
from app.services.tour_ai import stitch


def _jpeg(width: int, height: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _bg_db(scalar) -> tuple[MagicMock, MagicMock]:
    """Mock bg session factory returning a session whose execute() yields scalar."""
    db = MagicMock(spec=AsyncSession)
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    db.execute = AsyncMock(return_value=result)
    factory_cm = MagicMock()
    factory_cm.__aenter__ = AsyncMock(return_value=db)
    factory_cm.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=factory_cm)
    return db, factory


class TestDecodeAndDownscale:
    def test_downscales_frames_above_long_side_cap(self):
        image = stitch._decode_and_downscale(_jpeg(3200, 400))
        assert max(image.shape[:2]) <= stitch.MAX_FRAME_LONG_SIDE

    def test_keeps_small_frames_unscaled(self):
        image = stitch._decode_and_downscale(_jpeg(640, 480))
        assert image.shape[:2] == (480, 640)

    def test_rejects_undecodable_frame(self):
        with pytest.raises(ValueError, match="decoded"):
            stitch._decode_and_downscale(b"not an image")


class TestStitchFrames:
    def test_stitcher_failure_status_raises_value_error(self):
        fake_stitcher = MagicMock()
        fake_stitcher.stitch.return_value = (cv2.Stitcher_ERR_NEED_MORE_IMGS, None)

        with patch("cv2.Stitcher.create", return_value=fake_stitcher):
            with pytest.raises(ValueError, match="stitching failed"):
                stitch._stitch_frames([_jpeg(64, 48, 1), _jpeg(64, 48, 2)])

    def test_stitcher_cv2_error_raises_value_error(self):
        fake_stitcher = MagicMock()
        fake_stitcher.stitch.side_effect = cv2.error("knn assertion failed")

        with patch("cv2.Stitcher.create", return_value=fake_stitcher):
            with pytest.raises(ValueError, match="stitching failed"):
                stitch._stitch_frames([_jpeg(64, 48, 1), _jpeg(64, 48, 2)])

    def test_success_pads_panorama_to_two_to_one_canvas(self):
        panorama = np.full((100, 300, 3), 128, dtype=np.uint8)
        fake_stitcher = MagicMock()
        fake_stitcher.stitch.return_value = (cv2.Stitcher_OK, panorama)

        with patch("cv2.Stitcher.create", return_value=fake_stitcher):
            jpeg_bytes, width, height = stitch._stitch_frames(
                [_jpeg(64, 48, 1), _jpeg(64, 48, 2)]
            )

        assert width == 2 * height
        assert width >= 300
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (height, width)


class TestRunSceneStitch:
    @pytest.mark.asyncio
    async def test_success_updates_scene_and_completes_job(self):
        scene = Scene(id="scene-1", tour_id="tour-1", image_url="https://old.example.com/p.jpg")
        db, factory = _bg_db(scene)
        fake_cloudinary = MagicMock()
        fake_cloudinary.upload_file.return_value = {
            "secure_url": "https://cdn.example.com/stitched.jpg"
        }

        with (
            patch("app.services.tour_ai.stitch.get_bg_session_factory", return_value=factory),
            patch(
                "app.services.tour_ai.stitch._download_image_bytes",
                new_callable=AsyncMock,
                return_value=b"frame-bytes",
            ),
            patch(
                "app.services.tour_ai.stitch._stitch_frames",
                return_value=(b"jpeg-bytes", 800, 400),
            ),
            patch(
                "app.services.tour_ai.stitch.update_job_status", new_callable=AsyncMock
            ) as mock_update,
            patch(
                "app.services.cloudinary.get_cloudinary_service",
                return_value=fake_cloudinary,
            ),
            patch("app.services.tour.schedule_scene_processing") as mock_schedule,
        ):
            await stitch._run_scene_stitch(
                "job-1",
                "tour-1",
                "scene-1",
                1,
                ["https://cdn.example.com/f1.jpg", "https://cdn.example.com/f2.jpg"],
            )

        assert scene.image_url == "https://cdn.example.com/stitched.jpg"
        mock_schedule.assert_called_once()
        final_call = mock_update.await_args_list[-1]
        assert final_call.args[2] == "completed"
        assert final_call.args[3] == 100
        assert final_call.kwargs["result"] == {
            "image_url": "https://cdn.example.com/stitched.jpg",
            "width": 800,
            "height": 400,
        }

    @pytest.mark.asyncio
    async def test_stitch_failure_marks_job_failed(self):
        db, factory = _bg_db(None)

        with (
            patch("app.services.tour_ai.stitch.get_bg_session_factory", return_value=factory),
            patch(
                "app.services.tour_ai.stitch._download_image_bytes",
                new_callable=AsyncMock,
                return_value=b"frame-bytes",
            ),
            patch(
                "app.services.tour_ai.stitch._stitch_frames",
                side_effect=ValueError("OpenCV stitching failed (status 1)"),
            ),
            patch(
                "app.services.tour_ai.stitch.update_job_status", new_callable=AsyncMock
            ) as mock_update,
        ):
            await stitch._run_scene_stitch(
                "job-1",
                "tour-1",
                "scene-1",
                1,
                ["https://cdn.example.com/f1.jpg", "https://cdn.example.com/f2.jpg"],
            )

        final_call = mock_update.await_args_list[-1]
        assert final_call.args[2] == "failed"
        assert "stitching failed" in final_call.kwargs["error_message"]

    @pytest.mark.asyncio
    async def test_failure_rolls_back_before_marking_job_failed(self):
        """Regression: without rollback-first, update_job_status cannot persist.

        Fake session starts clean. After a failure, update_job_status refuses
        to run unless rollback() was called first — discriminating the ordering
        bug that left jobs stuck in 'processing'.
        """
        state = {"rolled_back": False, "failed_status_updates": 0}

        db = MagicMock(spec=AsyncSession)

        async def fake_rollback() -> None:
            state["rolled_back"] = True

        async def fake_update_job_status(
            _db: AsyncSession,
            _job_id: str,
            status: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            if status == "failed" and not state["rolled_back"]:
                raise RuntimeError(
                    "PendingRollbackError: session requires rollback before execute"
                )
            if status == "failed":
                state["failed_status_updates"] += 1

        db.commit = AsyncMock()
        db.rollback = AsyncMock(side_effect=fake_rollback)
        db.execute = AsyncMock()

        factory_cm = MagicMock()
        factory_cm.__aenter__ = AsyncMock(return_value=db)
        factory_cm.__aexit__ = AsyncMock(return_value=False)
        factory = MagicMock(return_value=factory_cm)

        with (
            patch("app.services.tour_ai.stitch.get_bg_session_factory", return_value=factory),
            patch(
                "app.services.tour_ai.stitch._download_image_bytes",
                new_callable=AsyncMock,
                return_value=b"frame-bytes",
            ),
            patch(
                "app.services.tour_ai.stitch._stitch_frames",
                side_effect=ValueError("OpenCV stitching failed (status 1)"),
            ),
            patch(
                "app.services.tour_ai.stitch.update_job_status",
                new_callable=AsyncMock,
                side_effect=fake_update_job_status,
            ),
        ):
            await stitch._run_scene_stitch(
                "job-1",
                "tour-1",
                "scene-1",
                1,
                ["https://cdn.example.com/f1.jpg", "https://cdn.example.com/f2.jpg"],
            )

        assert state["rolled_back"] is True
        assert state["failed_status_updates"] == 1
        db.rollback.assert_awaited()


class TestMetadataBlendPath:
    """The metadata-driven equirect path with its quality gate."""

    FRAMES = [
        {"url": "https://cdn.example.com/f1.jpg", "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        {"url": "https://cdn.example.com/f2.jpg", "yaw": 36.0, "pitch": 0.0, "roll": 1.5},
    ]
    PROFILE = {"horizontal_fov": 55.0, "vertical_fov": 69.0}

    def _patch_blend(self, quality_ok: bool):
        """Patches the pure panorama functions so the runner exercises the
        orchestration without real blending."""
        pano = np.zeros((512, 1024, 3), dtype=np.uint8)
        stats = stitch.panorama.BlendStats(
            total_pixels=512 * 1024,
            covered_pixels=512 * 1024,
            top_total=100,
            top_covered=100,
            middle_total=100,
            middle_covered=100,
            bottom_total=100,
            bottom_covered=100,
            seam_disagreement=0.0,
            seam_weight=0.0,
        )
        quality = stitch.panorama.metrics_from_stats(
            stats, [1.0, 1.0], 80.0, 1024, 512, 0
        )
        if not quality_ok:
            quality["coverage_percent"] = 20.0
        return (
            patch("app.services.tour_ai.stitch.panorama.blend_equirect", return_value=(pano, stats, [1.0, 1.0])),
            patch("app.services.tour_ai.stitch.panorama.sharpness_score", return_value=80.0),
            patch("app.services.tour_ai.stitch.panorama.validate_equirect", side_effect=(
                lambda image, coverage: [] if quality_ok else ["coverage 20.0% below minimum 60.0%"]
            )),
            patch("app.services.tour_ai.stitch.panorama.metrics_from_stats", return_value=quality),
        )

    def _run(self, db, factory):
        return stitch._run_scene_stitch(
            "job-1", "tour-1", "scene-1", 1, [f["url"] for f in self.FRAMES],
            frames=self.FRAMES, camera_profile=self.PROFILE,
        )

    @pytest.mark.asyncio
    async def test_success_replaces_scene_and_reports_quality(self):
        scene = Scene(id="scene-1", tour_id="tour-1", image_url="https://old.example.com/p.jpg")
        db, factory = _bg_db(scene)
        fake_cloudinary = MagicMock()
        fake_cloudinary.upload_file.return_value = {
            "secure_url": "https://cdn.example.com/refined.jpg"
        }
        blend_patch, sharp_patch, validate_patch, metrics_patch = self._patch_blend(True)

        with (
            patch("app.services.tour_ai.stitch.get_bg_session_factory", return_value=factory),
            patch("app.services.tour_ai.stitch._download_image_bytes", new_callable=AsyncMock, return_value=b"bytes"),
            patch("app.services.tour_ai.stitch._decode_and_downscale", return_value=np.zeros((400, 400, 3), dtype=np.uint8)),
            blend_patch, sharp_patch, validate_patch, metrics_patch,
            patch("app.services.cloudinary.get_cloudinary_service", return_value=fake_cloudinary),
            patch("app.services.tour.schedule_scene_processing") as mock_schedule,
            patch("app.services.tour_ai.stitch.update_job_status", new_callable=AsyncMock) as mock_update,
        ):
            await self._run(db, factory)

        assert scene.image_url == "https://cdn.example.com/refined.jpg"
        mock_schedule.assert_called_once()
        final_call = mock_update.await_args_list[-1]
        assert final_call.args[2] == "completed"
        result = final_call.kwargs["result"]
        assert result["image_url"] == "https://cdn.example.com/refined.jpg"
        assert result["quality"]["coverage_percent"] == 100.0

    @pytest.mark.asyncio
    async def test_quality_failure_does_not_replace_scene(self):
        scene = Scene(id="scene-1", tour_id="tour-1", image_url="https://old.example.com/p.jpg")
        db, factory = _bg_db(scene)
        blend_patch, sharp_patch, validate_patch, metrics_patch = self._patch_blend(False)

        with (
            patch("app.services.tour_ai.stitch.get_bg_session_factory", return_value=factory),
            patch("app.services.tour_ai.stitch._download_image_bytes", new_callable=AsyncMock, return_value=b"bytes"),
            patch("app.services.tour_ai.stitch._decode_and_downscale", return_value=np.zeros((400, 400, 3), dtype=np.uint8)),
            blend_patch, sharp_patch, validate_patch, metrics_patch,
            patch("app.services.cloudinary.get_cloudinary_service") as fake_cloudinary,
            patch("app.services.tour_ai.stitch.update_job_status", new_callable=AsyncMock) as mock_update,
        ):
            await self._run(db, factory)

        # The naive panorama stays live — the scene was NOT replaced.
        assert scene.image_url == "https://old.example.com/p.jpg"
        fake_cloudinary.assert_not_called()
        final_call = mock_update.await_args_list[-1]
        assert final_call.args[2] == "failed"
        assert "quality validation" in final_call.kwargs["error_message"]
        assert final_call.kwargs["result"]["quality"]["coverage_percent"] == 20.0
