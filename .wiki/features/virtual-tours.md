# 360 Virtual Tours

Active contributors: Saksham, Ravi

360 Virtual Tours is the immersive tour platform: users build tours from 360° scene images, attach navigation and info hotspots, generate floor plans, and publish branded experiences under custom domains. An AI layer powered by Gemini and GLM vision providers analyses scenes, suggests hotspot placements, and generates descriptions, with all AI work tracked as background `AIJob` rows.

## Directory layout

```
app/api/api_v1/endpoints/
├── tours.py               # tour CRUD, publish, duplicate, analytics
├── scenes.py              # scene CRUD, reorder, background processing
├── hotspots.py            # hotspot CRUD, position update
├── floor_plans.py         # floor plan CRUD, marker update
├── ai.py                  # AI job endpoints: analyze, suggest, generate
├── public.py              # public tour viewer + analytics event ingest
└── custom_domains.py      # domain registration, DNS verification, SSL
app/services/
├── tour/
│   ├── tours.py           # tour CRUD, publish/unpublish, duplicate
│   ├── scenes.py          # scene CRUD + background image processing
│   ├── hotspots.py        # hotspot CRUD + position update
│   ├── floor_plans.py     # floor plan CRUD + marker update
│   ├── analytics.py       # tour + dashboard stats, heatmap, realtime
│   └── helpers.py         # ownership checks, HTML sanitization, URL extraction
├── tour_ai/
│   ├── jobs.py            # AIJob CRUD
│   ├── scene_analysis.py  # scene analysis + description generation
│   ├── hotspot_suggestions.py # AI hotspot placement
│   ├── background.py      # tour generation, optimization, apply suggestions
│   ├── stitch.py          # cloud panorama stitching for scene frames
│   ├── panorama.py        # metadata-driven equirect blend + quality metrics (numpy)
│   ├── world3d.py         # "Generate 3D World": equirect→cubemap + GLB skybox mesh
│   └── helpers.py         # retry decorator, semaphore, image download
└── custom_domain.py       # domain creation, verification token, SSL status
app/models/
└── tours.py               # Tour, Scene, Hotspot, FloorPlan, AIJob, MediaFile, TourAnalyticsEvent, CustomDomain, VideoMetadata
```

## Key abstractions

| Abstraction | File | Role |
|---|---|---|
| `create_tour` / `publish_tour` | `app/services/tour/tours.py` | Tour lifecycle with status `draft → published → archived` |
| `create_scene` | `app/services/tour/scenes.py` | Scene creation + `schedule_scene_processing` for background image work |
| `create_hotspot` | `app/services/tour/hotspots.py` | Hotspot with type (navigation, info, audio, video, link, custom) |
| `_sanitize_hotspot_html` | `app/services/tour/helpers.py` | Allowlist-based HTML sanitization for hotspot content |
| `analyze_scene` | `app/services/tour_ai/scene_analysis.py` | AI scene analysis returning room type + quality score |
| `suggest_scene_hotspots` | `app/services/tour_ai/hotspot_suggestions.py` | AI-powered hotspot placement |
| `_AI_TASK_SEMAPHORE` | `app/services/tour_ai/helpers.py` | Concurrency limiter for AI background tasks |
| `create_custom_domain` | `app/services/custom_domain.py` | Domain registration with DNS TXT verification token |
| `record_analytics_event` | `app/services/tour/analytics.py` | Public viewer event ingest |
| `generate_short_code` | `app/services/tour/tours.py` | Unique 6-char share code assigned on first publish (`/v/{code}`) |
| `request_scene_stitch` | `app/services/tour_ai/stitch.py` | Cloud panorama stitch of captured frames — metadata-driven equirect blend (roll-correct, exposure-normalized, quality-gated) when frame metadata is supplied; legacy OpenCV path for `frame_urls`-only requests |
| `generate_3d_world` | `app/services/tour_ai/world3d.py` | Textured skybox-mesh GLB built from scene panoramas |

## How it works

Tour CRUD is straightforward keyset pagination on `(created_at, id)`. Scenes belong to tours and carry `order_index`; `reorder_scenes` updates positions atomically. Hotspots carry a `HotspotType` and arbitrary content that is sanitised through `_sanitize_hotspot_html` using `_HOTSPOT_HTML_ALLOWED_TAGS`, `_HOTSPOT_HTML_ALLOWED_ATTRIBUTES`, and `_HOTSPOT_HTML_ALLOWED_PROTOCOLS`. Floor plans accept marker updates for navigation overlay.

The AI layer is the complex part. Each AI operation creates an `AIJob` row with `status` (`pending, processing, completed, failed, cancelled`) and `job_type` (`scene_analysis, hotspot_generation, floor_plan_processing, panorama_stitch, generate_3d_world`). The job runs in the background under `_AI_TASK_SEMAPHORE` using a background-pool session (`get_bg_session_factory`). Image content is downloaded as base64 and passed to the AI provider as `VisionInput`. JSON responses go through `_complete_json_with_retry`, which retries with exponential backoff and appends a corrective nudge on parse failure; if the primary vision provider (Gemini) exhausts its retries, it transparently falls back to the other configured provider (GLM, or vice-versa) once via `_resolve_fallback_provider` before giving up.

```mermaid
graph TD
    Client -->|POST /tours/.../ai/analyze| EP[app/api/.../ai.py]
    EP --> AS[analyze_scene]
    AS --> JOB[create_ai_job status=pending]
    AS --> BG[_track_background_task _run_with_semaphore]
    BG --> DL[_download_image_as_base64]
    DL --> AI[AIProvider.complete_json]
    AI -->|JSON parse fail + retry| JSON[_complete_json_with_retry + nudge]
    JSON -->|primary exhausted| FB[_resolve_fallback_provider GLM<->Gemini]
    JSON & FB --> UPD[update_job_status completed]
    UPD --> Client2[SSE/ws job status]
    Client -->|POST /tours/.../ai/suggest-hotspots| SH[suggest_scene_hotspots]
    SH --> BG2[background _run_hotspot_suggestions]
    BG2 --> APPLY[apply_hotspot_suggestions]
    APPLY --> HP[(Hotspot rows)]
    Public -->|GET /tours/public/{slug}| PUB[public.py]
    PUB --> EVT[record_analytics_event]
    EVT --> TA[(TourAnalyticsEvent)]
```

Custom domains use a DNS TXT verification flow. `create_custom_domain` generates a `360ghar-verify-{token_hex(16)}` token, stores it with `verification_status=pending` and `ssl_status=pending`, and the user adds it as a DNS TXT record. Verification status transitions through `pending → verified → failed`; SSL status through `none → pending → active → failed`. The custom domain is linked to a tour for branded URL serving.

Analytics is split between owner-facing dashboards (`get_dashboard_stats`, `get_dashboard_realtime_stats`, `get_tour_heatmap`) and public-viewer event ingest (`record_analytics_event`). Public endpoints do not require auth and accept a `UserSession` identifier for funnel tracking.

**Short share links**: publishing a tour assigns a unique 6-character `short_code` (lowercase alphabet without 0/o/1/l/i, generated with `secrets.choice`, up to 5 collision retries against a partial unique index). `GET /v/{code}` (root-level, in `app/api/share.py`) resolves the code and renders the same OG/Twitter share preview as `GET /share/tours/{tour_id}`. Codes are never cleared on unpublish so existing links survive republish cycles.

**Cloud panorama stitching** (`POST /api/v1/scenes/{scene_id}/stitch`): accepts 2-32 https frame URLs, creates a `panorama_stitch` AIJob, then in a tracked background task stitches and swaps `scene.image_url`. Two paths: (a) **metadata path** - request carries `frames` (url + yaw/pitch/roll + target_index + low_quality) and optionally `camera_profile`; `tour_ai/panorama.py` blends a true 2:1 equirect (roll-correct rays, bilinear sampling, coverage-aware weighted blending, per-frame exposure gains) and produces a quality report (coverage/seam/exposure/sharpness). The scene is replaced ONLY when the result passes structural + coverage validation (>=60% coverage, 2:1, encodable); otherwise the job fails with the structured quality report and the published panorama stays. (b) **legacy path** - `frame_urls` only: `cv2.Stitcher` (PANORAMA mode), padded onto a 2:1 canvas (kept for backward compatibility). A module-level `Semaphore(1)` serialises stitches (memory-heavy) and the task carries a 5-minute timeout. The mobile app polls `GET /ai/jobs/{id}` and only marks the tour refined on `completed`.
**Generate 3D World** (`POST /api/v1/tours/{tour_id}/generate-3d`): creates a `generate_3d_world` AIJob that converts each scene's equirect panorama into a 6-face cubemap (numpy ray sampling), builds a GLB (glTF 2.0 binary, `KHR_materials_unlit`) of inward-facing textured cubes — one cube per scene laid out in a row (x += 3 units) — uploads the `.glb` to Cloudinary, and persists `{"mesh_url", "kind": "skybox_mesh", "scene_id", "scene_ids"}` into `tour.settings["world_3d"]` as well as the job result. Requires at least one scene (400 otherwise).

## Integration points

- **AI providers**: scene analysis and hotspot suggestions use `get_ai_provider` from `app/services/ai/` with Gemini and GLM providers, falling back per `VASTU_FALLBACK_PROVIDER` pattern (see [Vastu](vastu.md)).
- **Storage**: scene and floor plan images upload to Cloudinary under `TOUR_*` / `SCENE_*` storage folders via the shared [storage](../systems/services-layer.md) service.
- **MCP servers**: tour tools are not currently exposed through [MCP servers](mcp-servers.md); the [AI agent](ai-agent.md) does not register tour tools either.
- **WebSocket**: AI job status updates can be pushed through the WebSocket manager at `ws://localhost:3600/ws/jobs/{job_id}`.
- **Background sessions**: AI tasks release the request DB session and use `get_bg_session_factory()` per the streaming/session-hygiene pattern.

## Entry points for modification

Add new AI job types by extending `AIJobType` in `app/models/enums.py`, adding a runner in `tour_ai/`, and registering the endpoint in `ai.py`. New hotspot types go in `HotspotType` and must be handled in `_normalize_hotspot_content`. Custom domain verification logic lives in `app/services/custom_domain.py` — SSL provisioning is stubbed and would need a real ACME integration to activate.

## Key source files

| File | Purpose |
|---|---|
| `app/api/api_v1/endpoints/tours.py` | Tour endpoints (377 lines) |
| `app/api/api_v1/endpoints/scenes.py` | Scene endpoints |
| `app/api/api_v1/endpoints/hotspots.py` | Hotspot endpoints |
| `app/api/api_v1/endpoints/floor_plans.py` | Floor plan endpoints |
| `app/api/api_v1/endpoints/ai.py` | AI job endpoints |
| `app/api/api_v1/endpoints/public.py` | Public viewer + analytics ingest (16.2 KB) |
| `app/api/api_v1/endpoints/custom_domains.py` | Custom domain endpoints |
| `app/services/tour/tours.py` | Tour service (314 lines) |
| `app/services/tour/scenes.py` | Scene service (331 lines) |
| `app/services/tour/hotspots.py` | Hotspot service |
| `app/services/tour/analytics.py` | Analytics + dashboards (14 KB) |
| `app/services/tour/helpers.py` | Ownership + HTML sanitization (10.7 KB) |
| `app/services/tour_ai/scene_analysis.py` | Scene analysis (393 lines) |
| `app/services/tour_ai/hotspot_suggestions.py` | Hotspot suggestions (232 lines) |
| `app/services/tour_ai/background.py` | Tour generation + optimization (17 KB) |
| `app/services/tour_ai/helpers.py` | Retry + semaphore + image download |
| `app/services/custom_domain.py` | Domain registration + verification |
| `app/models/tours.py` | Tour ORM models (largest model file) |
