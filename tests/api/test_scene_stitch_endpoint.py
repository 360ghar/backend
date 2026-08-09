"""
API tests for POST /api/v1/scenes/{scene_id}/stitch.

The background stitch runner is patched out Ã¢â‚¬â€ these tests cover job
creation, ownership gating (404 on foreign/missing scenes), and payload
validation.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.models.enums import TourStatus, TourVisibility
from app.models.tours import Scene, Tour

FRAME_URLS = ["https://res.cloudinary.com/f1.jpg", "https://res.cloudinary.com/f2.jpg"]


async def _make_scene(db_session, user) -> Scene:
    tour = Tour(
        id=str(uuid.uuid4()),
        user_id=user.id,
        title="Stitch Tour",
        status=TourStatus.draft,
        is_public=False,
        visibility=TourVisibility.private,
    )
    db_session.add(tour)
    scene = Scene(
        id=str(uuid.uuid4()),
        tour_id=tour.id,
        image_url="https://res.cloudinary.com/pano.jpg",
        order_index=0,
    )
    db_session.add(scene)
    await db_session.flush()
    return scene


def _patched_stitch_background():
    """Patch out the background stitch runner (no bg DB session in tests)."""
    return (
        patch(
            "app.services.tour_ai.stitch._track_background_task",
            side_effect=lambda coro: coro.close(),
        ),
        patch("app.services.tour_ai.stitch._run_scene_stitch", MagicMock(return_value=None)),
    )


class TestSceneStitchEndpoint:
    @pytest.mark.asyncio
    async def test_creates_pending_stitch_job(self, user_client, db_session, test_user):
        scene = await _make_scene(db_session, test_user)
        patch_track, patch_run = _patched_stitch_background()

        with patch_track as mock_track, patch_run:
            response = await user_client.post(
                f"/api/v1/scenes/{scene.id}/stitch", json={"frame_urls": FRAME_URLS}
            )

        assert response.status_code == 200
        job = response.json()["job"]
        assert job["job_type"] == "panorama_stitch"
        assert job["status"] == "pending"
        assert job["scene_id"] == scene.id
        assert job["tour_id"] == scene.tour_id
        mock_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_foreign_scene_returns_404(
        self, user_client, db_session, test_user, test_user_2
    ):
        foreign_scene = await _make_scene(db_session, test_user_2)

        response = await user_client.post(
            f"/api/v1/scenes/{foreign_scene.id}/stitch", json={"frame_urls": FRAME_URLS}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_scene_returns_404(self, user_client, db_session, test_user):
        response = await user_client.post(
            f"/api/v1/scenes/{uuid.uuid4()}/stitch", json={"frame_urls": FRAME_URLS}
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_rejects_fewer_than_two_frames(self, user_client, db_session, test_user):
        scene = await _make_scene(db_session, test_user)

        response = await user_client.post(
            f"/api/v1/scenes/{scene.id}/stitch",
            json={"frame_urls": ["https://res.cloudinary.com/f1.jpg"]},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_more_than_32_frames(self, user_client, db_session, test_user):
        scene = await _make_scene(db_session, test_user)
        urls = [f"https://res.cloudinary.com/f{i}.jpg" for i in range(33)]

        response = await user_client.post(
            f"/api/v1/scenes/{scene.id}/stitch", json={"frame_urls": urls}
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_non_http_urls(self, user_client, db_session, test_user):
        scene = await _make_scene(db_session, test_user)

        response = await user_client.post(
            f"/api/v1/scenes/{scene.id}/stitch",
            json={"frame_urls": ["ftp://example.com/f1.jpg", "https://res.cloudinary.com/f2.jpg"]},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_accepts_frame_metadata_and_camera_profile(self, user_client, db_session, test_user):
        scene = await _make_scene(db_session, test_user)
        patch_track, patch_run = _patched_stitch_background()

        payload = {
            "frame_urls": FRAME_URLS,
            "frames": [
                {"url": FRAME_URLS[0], "yaw": 10.5, "pitch": 0.0, "roll": 1.5, "target_index": 4, "low_quality": False},
                {"url": FRAME_URLS[1], "yaw": 46.5, "pitch": 0.0, "roll": -2.0, "target_index": 5, "low_quality": True},
            ],
            "camera_profile": {"horizontal_fov": 55.0, "vertical_fov": 69.0},
        }
        with patch_track, patch_run as mock_run:
            response = await user_client.post(
                f"/api/v1/scenes/{scene.id}/stitch", json=payload
            )

        assert response.status_code == 200
        assert response.json()["job"]["status"] == "pending"
        # Metadata must be forwarded to the worker.
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["frames"] == payload["frames"]
        assert call_kwargs["camera_profile"] == payload["camera_profile"]

    @pytest.mark.asyncio
    async def test_rejects_frame_metadata_from_unallowed_host(self, user_client, db_session, test_user):
        scene = await _make_scene(db_session, test_user)
        patch_track, patch_run = _patched_stitch_background()

        payload = {
            "frame_urls": FRAME_URLS,
            "frames": [
                {"url": "https://evil.example.com/f1.jpg", "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                {"url": FRAME_URLS[1], "yaw": 36.0, "pitch": 0.0, "roll": 0.0},
            ],
        }
        with patch_track, patch_run:
            response = await user_client.post(
                f"/api/v1/scenes/{scene.id}/stitch", json=payload
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_single_frame_metadata(self, user_client, db_session, test_user):
        scene = await _make_scene(db_session, test_user)
        patch_track, patch_run = _patched_stitch_background()

        payload = {
            "frame_urls": FRAME_URLS,
            "frames": [{"url": FRAME_URLS[0], "yaw": 0.0, "pitch": 0.0, "roll": 0.0}],
        }
        with patch_track, patch_run:
            response = await user_client.post(
                f"/api/v1/scenes/{scene.id}/stitch", json=payload
            )

        assert response.status_code == 422
