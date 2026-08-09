"""
Scene API Endpoints.

This module provides REST API endpoints for managing scenes within virtual tours,
including CRUD operations and hotspot management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.api_v1.dependencies.auth import get_current_active_user
from app.core.database import get_db
from app.core.exceptions import ForbiddenException, SceneNotFoundException
from app.core.logging import get_logger
from app.schemas.tour import (
    AIJobResponse,
    Hotspot,
    HotspotCreate,
    Scene,
    SceneStitchRequest,
    SceneUpdate,
)
from app.schemas.user import User as UserSchema
from app.services import tour as tour_service
from app.services import tour_ai

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{scene_id}", response_model=Scene, summary="Get scene")
async def get_scene(
    scene_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user),
):
    """
    Get a scene by ID with all its hotspots.
    """
    return await tour_service.get_scene(db=db, scene_id=scene_id, user_id=current_user.id)


@router.put("/{scene_id}", response_model=Scene, summary="Update scene")
@router.patch("/{scene_id}", response_model=Scene, summary="Update scene")
async def update_scene(
    scene_id: str,
    scene_data: SceneUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user),
):
    """
    Update a scene's details.

    Can update title, description, image URLs, and metadata.
    """
    scene = await tour_service.update_scene(
        db=db,
        scene_id=scene_id,
        user_id=current_user.id,
        data=scene_data,
    )
    if not scene:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scene not found or not authorized"
        )
    return scene


@router.delete("/{scene_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete scene")
async def delete_scene(
    scene_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user),
):
    """
    Delete a scene from a tour.

    This will also delete all hotspots associated with the scene.
    """
    success = await tour_service.delete_scene(
        db=db,
        scene_id=scene_id,
        user_id=current_user.id,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scene not found or not authorized"
        )
    return None


@router.post("/{scene_id}/stitch", response_model=AIJobResponse, summary="Stitch scene frames")
async def stitch_scene(
    scene_id: str,
    stitch_data: SceneStitchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user),
):
    """
    Stitch captured frames into a panorama for a scene in the cloud.

    With per-frame metadata (`frames` + `camera_profile`) the metadata-driven
    equirect blend runs and the scene image is replaced only when quality
    validation passes. Legacy requests (frame_urls only) use the OpenCV
    path. Returns a job ID for tracking progress (also broadcast over the
    jobs WebSocket).
    """
    try:
        scene = await tour_service.get_scene(db=db, scene_id=scene_id, user_id=current_user.id)
    except ForbiddenException:
        # Present foreign scenes as missing to avoid leaking their existence.
        raise SceneNotFoundException() from None

    frames = (
        [f.model_dump() for f in stitch_data.frames]
        if stitch_data.frames
        else None
    )
    job = await tour_ai.request_scene_stitch(
        db=db,
        user_id=current_user.id,
        tour_id=scene.tour_id,
        scene_id=scene_id,
        frame_urls=stitch_data.frame_urls,
        frames=frames,
        camera_profile=(
            stitch_data.camera_profile.model_dump() if stitch_data.camera_profile else None
        ),
    )
    return {"job": job}


# Hotspot endpoints nested under scenes
@router.get("/{scene_id}/hotspots", response_model=list[Hotspot], summary="List scene hotspots")
async def list_hotspots(
    scene_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user),
    limit: int = Query(100, le=100, description="Max hotspots to return (hard cap 100)"),
):
    """
    List all hotspots for a scene.

    Returns hotspots ordered by their order_index.
    """
    # Ensures the current user owns the scene's tour.
    await tour_service.get_scene(db=db, scene_id=scene_id, user_id=current_user.id)

    hotspots = await tour_service.get_hotspots(db=db, scene_id=scene_id)
    return hotspots[:limit]


@router.post("/{scene_id}/hotspots", response_model=Hotspot, status_code=status.HTTP_201_CREATED, summary="Create scene hotspot")
async def create_hotspot(
    scene_id: str,
    hotspot_data: HotspotCreate,
    db: AsyncSession = Depends(get_db),
    current_user: UserSchema = Depends(get_current_active_user),
):
    """
    Add a new hotspot to a scene.

    Hotspots can be navigation links, info popups, or media elements.
    """
    hotspot = await tour_service.create_hotspot(
        db=db,
        scene_id=scene_id,
        user_id=current_user.id,
        data=hotspot_data,
    )
    if not hotspot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scene not found or not authorized"
        )
    return hotspot
