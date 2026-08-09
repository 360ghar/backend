"""
Cloud panorama stitching for tour scenes.

Two paths, both tracked through an AIJob:

- **Metadata path** (frames carry yaw/pitch/roll + camera_profile): a
  metadata-driven equirect blend (see panorama.py) produces a true 2:1
  equirect with honest coverage/quality metrics. The scene image is replaced
  ONLY when the result passes quality validation — a bad stitch never
  silently replaces a good naive panorama.
- **Legacy path** (frame_urls only): OpenCV stitch padded onto a 2:1 canvas
  (kept for backward compatibility; coverage reported informationally).
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_bg_session_factory
from app.core.logging import get_logger
from app.models.enums import AIJobType
from app.models.tours import AIJob, Scene

from . import panorama
from .helpers import _download_image_bytes, _run_with_semaphore, _track_background_task
from .jobs import create_ai_job, update_job_status

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

# ponytail: one stitch at a time — cv2.Stitcher is CPU- and RAM-hungry;
# widen to a bounded pool if stitch throughput ever matters.
_STITCH_SEMAPHORE = asyncio.Semaphore(1)

MAX_STITCH_FRAMES = 32
MAX_FRAME_LONG_SIDE = 1600  # downscale frames to bound stitcher memory
STITCH_TIMEOUT_SECONDS = 300
JPEG_QUALITY = 92
# ponytail: cap decoded pixel count to bound memory from decompression-bomb
# frames; 40M px is ~5.5x a 12MP photo, comfortably above any real capture.
MAX_DECODED_PIXELS = 40_000_000


async def request_scene_stitch(
    db: AsyncSession,
    user_id: int,
    tour_id: str,
    scene_id: str,
    frame_urls: list[str],
    frames: list[dict[str, Any]] | None = None,
    camera_profile: dict[str, Any] | None = None,
) -> AIJob:
    """Create a panorama-stitch job for a scene and schedule the stitch.

    [frames]/[camera_profile] are optional capture metadata; when present the
    metadata-driven blend path is used, otherwise the legacy OpenCV path.
    """
    job = await create_ai_job(
        db,
        user_id,
        AIJobType.panorama_stitch.value,
        tour_id=tour_id,
        scene_id=scene_id,
    )
    _track_background_task(
        _run_with_semaphore(
            _run_scene_stitch(
                job.id,
                tour_id,
                scene_id,
                user_id,
                list(frame_urls),
                frames=frames,
                camera_profile=camera_profile,
            )
        )
    )
    return job


def _decode_and_downscale(frame_bytes: bytes) -> np.ndarray:
    """Decode a frame and downscale so its long side is <= MAX_FRAME_LONG_SIDE."""
    import cv2
    import numpy as np

    array = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Frame could not be decoded as an image")

    height, width = image.shape[:2]
    if width * height > MAX_DECODED_PIXELS:
        raise ValueError(
            f"Frame too large ({width}x{height} = {width * height} px, "
            f"max {MAX_DECODED_PIXELS})"
        )
    long_side = max(height, width)
    if long_side > MAX_FRAME_LONG_SIDE:
        scale = MAX_FRAME_LONG_SIDE / long_side
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _stitch_frames(frames: list[bytes]) -> tuple[bytes, int, int]:
    """Stitch frames into a panorama padded onto a 2:1 black canvas.

    Returns (jpeg_bytes, width, height). Raises ValueError when OpenCV
    cannot stitch the frames (insufficient overlap, undecodable frames).
    """
    import cv2
    import numpy as np

    images = [_decode_and_downscale(frame) for frame in frames]

    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    try:
        status, panorama = stitcher.stitch(images)
    except cv2.error as e:
        # cv2 raises (rather than returning a status) when feature
        # detection/matching finds too little to work with.
        raise ValueError(
            "OpenCV stitching failed; frames may lack enough overlapping detail to stitch"
        ) from e
    if status != cv2.Stitcher_OK or panorama is None:
        raise ValueError(
            f"OpenCV stitching failed (status {status}); "
            "frames may lack enough overlap to stitch"
        )

    # Pad onto a 2:1 black canvas so viewers can treat it as equirect aspect.
    height, width = panorama.shape[:2]
    canvas_width = max(width, 2 * height)
    canvas_width += canvas_width % 2
    canvas_height = canvas_width // 2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    y_offset = (canvas_height - height) // 2
    x_offset = (canvas_width - width) // 2
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = panorama

    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Failed to encode the stitched panorama as JPEG")
    return encoded.tobytes(), canvas_width, canvas_height


async def _stitch_and_store(
    db: AsyncSession,
    job_id: str,
    tour_id: str,
    scene_id: str,
    user_id: int,
    frame_urls: list[str],
    frames: list[dict[str, Any]] | None = None,
    camera_profile: dict[str, Any] | None = None,
) -> None:
    """Download, stitch, upload, and persist the panorama for a scene."""
    if frames:
        await _blend_and_store(
            db, job_id, tour_id, scene_id, user_id, frames, camera_profile
        )
        return

    # Legacy path: OpenCV stitch (kept for backward compatibility; its 2:1
    # letterbox is reported with an informational coverage metric).
    await update_job_status(db, job_id, "processing", 10)
    legacy_frames = [await _download_image_bytes(url) for url in frame_urls]

    await update_job_status(db, job_id, "processing", 50)
    panorama_bytes, width, height = await asyncio.to_thread(
        _stitch_frames, legacy_frames
    )
    del legacy_frames

    await update_job_status(db, job_id, "processing", 80)
    from app.services.cloudinary import get_cloudinary_service

    upload_result = await asyncio.to_thread(
        get_cloudinary_service().upload_file,
        file_bytes=panorama_bytes,
        public_id=f"{uuid4().hex[:8]}_stitched",
        folder=f"tours/{tour_id}/scenes/{scene_id}/original",
        content_type="image/jpeg",
        is_image=True,
    )
    image_url: str = upload_result["secure_url"]

    scene = (
        await db.execute(select(Scene).where(Scene.id == scene_id))
    ).scalar_one_or_none()
    if scene is None:
        raise ValueError("Scene was deleted while stitching")
    scene.image_url = image_url
    scene.is_processed = False  # thumbnail regeneration below flips it back
    await db.commit()

    # Regenerate the thumbnail/metadata through the standard scene pipeline.
    from app.services.tour import schedule_scene_processing

    schedule_scene_processing(
        scene_id=scene_id, tour_id=tour_id, image_url=image_url, user_id=user_id
    )

    await update_job_status(
        db,
        job_id,
        "completed",
        100,
        result={"image_url": image_url, "width": width, "height": height},
    )
    await db.commit()
    logger.info("Panorama stitched for scene %s (%dx%d)", scene_id, width, height)


async def _blend_and_store(
    db: AsyncSession,
    job_id: str,
    tour_id: str,
    scene_id: str,
    user_id: int,
    frames: list[dict[str, Any]],
    camera_profile: dict[str, Any] | None,
) -> None:
    """Metadata-driven equirect blend with a hard quality gate.

    The scene image is replaced ONLY when the stitched panorama passes
    structural + coverage validation (panorama.validate_equirect). A
    technically successful but poor result marks the job failed with the
    structured quality report — never silently replaces the naive pano.
    """
    import cv2

    await update_job_status(db, job_id, "processing", 10)
    raw = [await _download_image_bytes(f["url"]) for f in frames]

    def _decode_all(raw_bytes: list[bytes]) -> list[panorama.FrameSpec]:
        specs = []
        for data, meta in zip(raw_bytes, frames, strict=False):
            image = _decode_and_downscale(data)
            specs.append(
                panorama.FrameSpec(
                    image=image,
                    yaw_deg=float(meta.get("yaw", 0.0)),
                    pitch_deg=float(meta.get("pitch", 0.0)),
                    roll_deg=float(meta.get("roll", 0.0)),
                    low_quality=bool(meta.get("low_quality", False)),
                )
            )
        return specs

    specs = await asyncio.to_thread(_decode_all, raw)
    del raw

    await update_job_status(db, job_id, "processing", 50)
    h_fov = float((camera_profile or {}).get("horizontal_fov", 55.0))
    v_fov = float((camera_profile or {}).get("vertical_fov", 69.0))
    panorama_img, stats, gains = await asyncio.to_thread(
        panorama.blend_equirect, specs, h_fov, v_fov
    )
    del specs

    sharpness = await asyncio.to_thread(panorama.sharpness_score, panorama_img)
    low_quality_count = sum(1 for f in frames if f.get("low_quality"))
    quality = panorama.metrics_from_stats(
        stats,
        gains,
        sharpness,
        panorama_img.shape[1],
        panorama_img.shape[0],
        low_quality_count,
    )

    # Both the validation (JPEG-encodes a 2048x1024 canvas) and the final
    # encode are CPU-bound — keep them off the event loop like the blend.
    problems = await asyncio.to_thread(
        panorama.validate_equirect, panorama_img, quality["coverage_percent"]
    )
    if problems:
        logger.warning(
            "Stitch quality validation failed for scene %s: %s", scene_id, problems
        )
        await update_job_status(
            db,
            job_id,
            "failed",
            error_message="Stitched panorama failed quality validation: "
            + "; ".join(problems),
            result={"quality": quality},
        )
        await db.commit()
        return  # scene NOT replaced — the published naive panorama stays

    await update_job_status(db, job_id, "processing", 80)
    ok, encoded = await asyncio.to_thread(
        cv2.imencode, ".jpg", panorama_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    del panorama_img
    if not ok:
        raise ValueError("Failed to encode the stitched panorama as JPEG")
    panorama_bytes = encoded.tobytes()

    from app.services.cloudinary import get_cloudinary_service

    upload_result = await asyncio.to_thread(
        get_cloudinary_service().upload_file,
        file_bytes=panorama_bytes,
        public_id=f"{uuid4().hex[:8]}_stitched",
        folder=f"tours/{tour_id}/scenes/{scene_id}/original",
        content_type="image/jpeg",
        is_image=True,
    )
    image_url: str = upload_result["secure_url"]

    scene = (
        await db.execute(select(Scene).where(Scene.id == scene_id))
    ).scalar_one_or_none()
    if scene is None:
        raise ValueError("Scene was deleted while stitching")
    scene.image_url = image_url
    scene.is_processed = False  # thumbnail regeneration below flips it back
    await db.commit()

    # Regenerate the thumbnail/metadata through the standard scene pipeline.
    from app.services.tour import schedule_scene_processing

    schedule_scene_processing(
        scene_id=scene_id, tour_id=tour_id, image_url=image_url, user_id=user_id
    )

    await update_job_status(
        db,
        job_id,
        "completed",
        100,
        result={
            "image_url": image_url,
            "width": quality["width"],
            "height": quality["height"],
            "quality": quality,
        },
    )
    await db.commit()
    logger.info(
        "Metadata blend for scene %s (%sx%s, coverage %s%%)",
        scene_id,
        quality["width"],
        quality["height"],
        quality["coverage_percent"],
    )


async def _run_scene_stitch(
    job_id: str,
    tour_id: str,
    scene_id: str,
    user_id: int,
    frame_urls: list[str],
    frames: list[dict[str, Any]] | None = None,
    camera_profile: dict[str, Any] | None = None,
) -> None:
    """Background runner for a panorama-stitch job (owns its DB session)."""
    session_factory = get_bg_session_factory()
    async with session_factory() as db:
        try:
            async with _STITCH_SEMAPHORE:
                await asyncio.wait_for(
                    _stitch_and_store(
                        db,
                        job_id,
                        tour_id,
                        scene_id,
                        user_id,
                        frame_urls,
                        frames=frames,
                        camera_profile=camera_profile,
                    ),
                    timeout=STITCH_TIMEOUT_SECONDS,
                )
        except TimeoutError:
            logger.error("Panorama stitch timed out for scene %s", scene_id)
            await db.rollback()
            await update_job_status(
                db,
                job_id,
                "failed",
                error_message=f"Stitching timed out after {STITCH_TIMEOUT_SECONDS}s",
            )
            await db.commit()
        except Exception as e:
            logger.error("Error stitching scene %s: %s", scene_id, e, exc_info=True)
            await db.rollback()
            await update_job_status(db, job_id, "failed", error_message=str(e))
            await db.commit()
