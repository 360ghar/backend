"""
Pydantic schemas for 360 Virtual Tour API.

These schemas define the request/response models for the tour management endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.models.enums import HotspotType, TourStatus, TourVisibility

# Allowed hosts for user-supplied frame/scene image URLs (SSRF guard).
# Extend this set if the Cloudinary delivery host ever differs from the
# default (e.g. a custom CNAME).
ALLOWED_FRAME_HOSTS = {"res.cloudinary.com"}

# ====================
# Tour Settings Schema
# ====================

class TourBrandingSettings(BaseModel):
    """Branding settings for a tour."""
    logo_url: str | None = None
    primary_color: str | None = None
    secondary_color: str | None = None
    accent_color: str | None = None
    text_color: str | None = None
    background_color: str | None = None
    font_family: str | None = None
    button_style: str | None = None  # rounded | square | pill
    show_watermark: bool | None = True
    watermark_position: str | None = None  # bottom-left, bottom-right, top-left, top-right
    custom_css: str | None = None


class FloorPlanMarker(BaseModel):
    """Marker data for floor plan hotspots."""
    scene_id: str
    x: float = Field(..., ge=0, le=100)
    y: float = Field(..., ge=0, le=100)
    label: str | None = None


class FloorPlan(BaseModel):
    """Floor plan configuration stored in tour settings."""
    id: str
    name: str
    image_url: str
    floor_number: int = 1
    markers: list[FloorPlanMarker] = Field(default_factory=list)


# ====================
# Floor Plan CRUD Schemas (for dedicated table)
# ====================

class FloorPlanCreate(BaseModel):
    """Schema for creating a floor plan."""
    name: str = Field(..., min_length=1, max_length=255)
    image_url: str = Field(..., min_length=1)
    floor_number: int = Field(default=1, ge=1)
    markers: list[FloorPlanMarker] = Field(default_factory=list)


class FloorPlanUpdate(BaseModel):
    """Schema for updating a floor plan."""
    name: str | None = Field(default=None, min_length=1, max_length=255)
    image_url: str | None = None
    floor_number: int | None = Field(default=None, ge=1)
    markers: list[FloorPlanMarker] | None = None


class FloorPlanResponse(BaseModel):
    """Response schema for floor plan."""
    id: str
    tour_id: str
    name: str
    image_url: str
    floor_number: int
    markers: list[FloorPlanMarker] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TourSettings(BaseModel):
    """Tour configuration settings."""
    auto_rotate: bool | None = False
    auto_rotate_speed: float | None = Field(default=1.0, ge=0.1, le=10.0)
    initial_scene_id: str | None = None
    initial_view: dict[str, float] | None = None  # {yaw, pitch}
    show_navbar: bool | None = True
    enable_fullscreen: bool | None = True
    enable_vr: bool | None = True
    enable_gyroscope: bool | None = True
    gyroscope_auto_start: bool | None = False
    branding: TourBrandingSettings | None = None
    floor_plans: list[FloorPlan] | None = None
    world_3d: dict[str, Any] | None = None


# ====================
# Hotspot Schemas
# ====================

class HotspotPosition(BaseModel):
    """Position of a hotspot in 3D space."""
    yaw: float = Field(..., ge=-180, le=180, description="Horizontal angle in degrees")
    pitch: float = Field(..., ge=-90, le=90, description="Vertical angle in degrees")
    radius: float | None = Field(default=None, gt=0, description="Optional radius for interaction area")


class HotspotBase(BaseModel):
    """Base hotspot schema with common fields."""
    type: HotspotType = HotspotType.info
    position: HotspotPosition
    target_scene_id: str | None = None
    title: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    icon: str | None = Field(default=None, max_length=50)
    icon_name: str | None = Field(default=None, max_length=100)
    icon_color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    icon_size: int | None = Field(default=None, ge=16, le=100)
    content: dict[str, Any] | None = None
    custom_data: dict[str, Any] | None = None


class HotspotCreate(HotspotBase):
    """Schema for creating a hotspot."""
    model_config = ConfigDict(json_schema_extra={"title": "HotspotCreate"})


class HotspotUpdate(BaseModel):
    """Schema for updating a hotspot."""
    type: HotspotType | None = None
    position: HotspotPosition | None = None
    target_scene_id: str | None = None
    title: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    icon: str | None = Field(default=None, max_length=50)
    icon_name: str | None = Field(default=None, max_length=100)
    icon_color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    icon_size: int | None = Field(default=None, ge=16, le=100)
    content: dict[str, Any] | None = None
    custom_data: dict[str, Any] | None = None
    is_active: bool | None = None


class HotspotPositionUpdate(BaseModel):
    """Schema for updating only hotspot position."""
    yaw: float = Field(..., ge=-180, le=180)
    pitch: float = Field(..., ge=-90, le=90)


class Hotspot(HotspotBase):
    """Hotspot response schema."""
    id: str
    scene_id: str
    order_index: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ====================
# Scene Metadata Schema
# ====================

class SceneInitialView(BaseModel):
    """Initial camera view for a scene."""
    yaw: float = 0
    pitch: float = 0
    zoom: float | None = 50


class SceneCameraSettings(BaseModel):
    """Camera settings for a scene."""
    fov: float | None = 70
    min_fov: float | None = 30
    max_fov: float | None = 90


class SceneMetadata(BaseModel):
    """Metadata for a scene."""
    initial_view: SceneInitialView | None = None
    camera: SceneCameraSettings | None = None
    gps: dict[str, float] | None = None  # {latitude, longitude}
    exif: dict[str, Any] | None = None


# ====================
# Scene Schemas
# ====================

class SceneBase(BaseModel):
    """Base scene schema with common fields."""
    title: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    order_index: int | None = Field(default=None, ge=0)
    metadata: SceneMetadata | None = Field(
        default=None,
        alias="scene_metadata",
        validation_alias=AliasChoices("metadata", "scene_metadata"),
        serialization_alias="metadata",
    )

    model_config = ConfigDict(populate_by_name=True)


class SceneCreate(SceneBase):
    """Schema for creating a scene."""
    image_url: str = Field(..., max_length=500)
    thumbnail_url: str | None = Field(default=None, max_length=500)


class SceneUpdate(SceneBase):
    """Schema for updating a scene."""
    image_url: str | None = Field(default=None, max_length=500)
    thumbnail_url: str | None = Field(default=None, max_length=500)


class SceneReorder(BaseModel):
    """Schema for reordering scenes."""
    scene_ids: list[str] = Field(..., min_length=1)


class Scene(SceneBase):
    """Scene response schema."""
    id: str
    tour_id: str
    image_url: str
    thumbnail_url: str | None = None
    vr_url: str | None = None
    is_processed: bool
    processing_error: str | None = None
    created_at: datetime
    updated_at: datetime
    hotspots: list[Hotspot] | None = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ====================
# Tour Schemas
# ====================

class TourBase(BaseModel):
    """Base tour schema with common fields."""
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    status: TourStatus | None = TourStatus.draft
    is_public: bool | None = False  # Deprecated: Use visibility instead
    visibility: TourVisibility | None = TourVisibility.private
    settings: TourSettings | None = None


class TourCreate(TourBase):
    """Schema for creating a tour."""
    model_config = ConfigDict(json_schema_extra={"title": "TourCreate"})


class TourUpdate(BaseModel):
    """Schema for updating a tour."""
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    status: TourStatus | None = None
    is_public: bool | None = None  # Deprecated: Use visibility instead
    visibility: TourVisibility | None = None
    is_featured: bool | None = None
    settings: TourSettings | None = None
    thumbnail_url: str | None = Field(default=None, max_length=500)


class Tour(TourBase):
    """Tour response schema."""
    id: str
    user_id: int
    short_code: str | None = None
    is_featured: bool
    visibility: TourVisibility
    view_count: int
    like_count: int
    share_count: int
    thumbnail_url: str | None = None
    published_at: datetime | None = None
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    scenes: list[Scene] | None = None
    scene_count: int | None = None

    model_config = ConfigDict(from_attributes=True)


class TourWithScenes(Tour):
    """Tour with all scenes loaded (also the public tour response shape).

    Inherits ``short_code`` from ``Tour``.
    """
    scenes: list[Scene] = []


# ====================
# Analytics Schemas
# ====================

class DeviceBreakdown(BaseModel):
    """Device type breakdown for analytics."""
    desktop: int = 0
    mobile: int = 0
    tablet: int = 0
    vr: int = 0


class DailyView(BaseModel):
    """Daily view count for analytics."""
    date: str
    views: int


class TourEventPayload(BaseModel):
    """Payload for tracking tour analytics events."""
    event_type: str
    scene_id: str | None = None
    hotspot_id: str | None = None
    session_id: str | None = None
    event_data: dict[str, Any] | None = None


class HeatmapPoint(BaseModel):
    """Heatmap point for viewer analytics."""
    scene_id: str | None = None
    yaw: float | None = None
    pitch: float | None = None
    x: float | None = None
    y: float | None = None
    intensity: float = 1.0


class TourAnalytics(BaseModel):
    """Analytics data for a tour."""
    tour_id: str
    total_views: int = 0
    unique_views: int = 0
    total_likes: int = 0
    total_shares: int = 0
    avg_session_duration: float = 0.0
    scene_views: dict[str, int] = {}
    hotspot_clicks: dict[str, int] = {}
    heatmap_points: list[HeatmapPoint] = []
    share_breakdown: dict[str, int] = {}
    session_durations: list[float] = []
    device_breakdown: DeviceBreakdown = DeviceBreakdown()
    country_breakdown: dict[str, int] = {}
    daily_views: list[DailyView] = []


class DashboardStats(BaseModel):
    """Dashboard statistics for a user."""
    total_tours: int = 0
    published_tours: int = 0
    total_views: int = 0
    total_scenes: int = 0
    storage_used: int = 0  # bytes
    storage_limit: int = 0  # bytes


class DashboardRealtimeStats(BaseModel):
    """Realtime dashboard metrics."""
    active_sessions: int = 0
    views_last_hour: int = 0
    likes_last_hour: int = 0
    shares_last_hour: int = 0
    avg_session_duration: float = 0.0
    recent_views: list[DailyView] = []


# ====================
# API Response Wrapper
# ====================

class ApiResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    data: Any
    message: str | None = None


# ====================
# AI Processing Schemas
# ====================

class AIJobBase(BaseModel):
    """Base AI Job schema."""
    id: str
    job_type: str
    status: str
    progress: int
    tour_id: str | None = None
    scene_id: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AIJobResponse(BaseModel):
    """Response containing an AI job."""
    job: AIJobBase


class SceneFrameMetadata(BaseModel):
    """Per-frame capture metadata for the metadata-driven cloud stitcher.

    When omitted (legacy requests), the frame is assumed to carry no
    orientation and the legacy OpenCV path is used.
    """
    url: str
    yaw: float = Field(default=0.0, ge=-360, le=360, description="Degrees, east-positive")
    pitch: float = Field(default=0.0, ge=-90, le=90)
    roll: float = Field(default=0.0, ge=-180, le=180)
    target_index: int | None = Field(default=None, ge=0, le=63)
    low_quality: bool = False

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_FRAME_HOSTS:
            raise ValueError(
                f"frame urls must be https URLs from allowed hosts, got: {v}"
            )
        return v


class CameraProfileIn(BaseModel):
    """Camera geometry the capture was planned with (fallback 55x69 when
    absent — matches the mobile documented fallback profile)."""
    horizontal_fov: float = Field(default=55.0, gt=1, le=360)
    vertical_fov: float = Field(default=69.0, gt=1, le=180)


class SceneStitchRequest(BaseModel):
    """Request payload for cloud panorama stitching of captured frames.

    Backward compatible: `frame_urls` alone (legacy) still works — the
    worker falls back to the OpenCV path. When `frames` (with orientation
    metadata) is present, the metadata-driven equirect blend is used.
    """
    frame_urls: list[str] = Field(..., min_length=2, max_length=32)
    frames: list[SceneFrameMetadata] | None = Field(default=None, max_length=32)
    camera_profile: CameraProfileIn | None = None

    @field_validator("frame_urls")
    @classmethod
    def validate_frame_urls(cls, v: list[str]) -> list[str]:
        for url in v:
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in ALLOWED_FRAME_HOSTS:
                raise ValueError(
                    f"frame_urls must be https URLs from allowed hosts, got: {url}"
                )
        return v

    @field_validator("frames")
    @classmethod
    def validate_frames_count(cls, v: list[SceneFrameMetadata] | None) -> list[SceneFrameMetadata] | None:
        if v is not None and len(v) < 2:
            raise ValueError("frames must contain at least 2 entries")
        return v


class SceneAnalysisResult(BaseModel):
    """Result of AI scene analysis."""
    scene_id: str
    room_type: str
    room_confidence: float = Field(..., ge=0, le=1)
    suggested_title: str
    suggested_description: str
    quality_score: int = Field(..., ge=0, le=100)
    quality_issues: list[str] | None = None
    features_detected: list[str] = []


class HotspotSuggestion(BaseModel):
    """AI-suggested hotspot."""
    id: str
    type: str = "navigation"
    position: HotspotPosition
    target_scene_id: str | None = None
    suggested_title: str | None = None
    reasoning: str
    confidence: float = Field(..., ge=0, le=1)


class AIJobStatusResponse(BaseModel):
    """Response containing AI job status with optional results."""
    job: AIJobBase
    result: dict[str, Any] | None = None


class DescriptionOptions(BaseModel):
    """Options for AI description generation."""
    tone: str | None = Field(default="professional", pattern=r"^(professional|casual|luxury|friendly)$")
    length: str | None = Field(default="medium", pattern=r"^(short|medium|long)$")
    include_features: bool | None = True
    target_audience: str | None = None


class ApplySceneAnalysis(BaseModel):
    """Request to apply AI scene analysis suggestions."""
    suggestions: list[dict[str, Any]]


class ApplyHotspotSuggestions(BaseModel):
    """Request to apply AI hotspot suggestions."""
    suggestion_ids: list[str]


class TourGenerationSceneInput(BaseModel):
    """Scene input for AI-driven tour generation."""
    image_url: str = Field(..., max_length=500)
    thumbnail_url: str | None = Field(default=None, max_length=500)
    title: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    order_index: int | None = Field(default=None, ge=0)
    metadata: SceneMetadata | None = Field(
        default=None,
        alias="scene_metadata",
        validation_alias=AliasChoices("metadata", "scene_metadata"),
        serialization_alias="metadata",
    )

    model_config = ConfigDict(populate_by_name=True)


class TourGenerationRequest(BaseModel):
    """Request payload for AI tour generation."""
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)
    status: TourStatus | None = TourStatus.draft
    is_public: bool | None = False  # Deprecated: Use visibility instead
    visibility: TourVisibility | None = TourVisibility.private
    settings: TourSettings | None = None
    scenes: list[TourGenerationSceneInput] | None = None
    image_urls: list[str] | None = None
    generate_titles: bool | None = True
    generate_descriptions: bool | None = True
    suggest_hotspots: bool | None = False
    apply_to_scenes: bool | None = True
    language: str | None = None
    # When true, build Matterport-style spatial navigation: place each navigation
    # hotspot on the actual doorway leading to the correct adjacent room (vision +
    # graph), instead of naive consecutive-scene linking. See
    # app/services/tour_ai/spatial.py.
    spatial: bool | None = False

    model_config = ConfigDict(extra="allow")


class TourGenerationResponse(BaseModel):
    """Response for AI tour generation."""
    job: AIJobBase
    tour_id: str
    scene_ids: list[str]


class TourOptimizationRequest(BaseModel):
    """Request payload for AI tour optimization."""
    goals: list[str] | None = None
    focus_areas: list[str] | None = None
    update_titles: bool | None = False
    update_descriptions: bool | None = False
    suggest_hotspots: bool | None = False
    language: str | None = None

    model_config = ConfigDict(extra="allow")


class TourOptimizationResponse(BaseModel):
    """Response for AI tour optimization."""
    job: AIJobBase
