# MirageAEC v2.1 — System Architecture

> **Last Updated:** September 2026  
> **Stack:** React + Vite (Frontend) · FastAPI + Python (Backend) · 5 AI Models (Google Colab T4 GPU) · SQLite (Database)

---

## 1. Big Picture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                                   │
│   React + TypeScript + Three.js + Vite (localhost:5173 / prod build)    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  REST API (HTTP/JSON)
                               │  POST /api/floorplan/analyze
                               │  GET  /api/generate/status/:id
                               │  POST /api/bim/export
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PYTHON BACKEND (FastAPI)                              │
│                    localhost:8000 / uvicorn                              │
│                                                                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │ Pillar 1     │  │ Pillar 2     │  │ Pillar 3     │                 │
│   │ CV + AI      │  │ 3D BIM       │  │ MEP Routing  │                 │
│   │ Segmentation │  │ Reconstruct  │  │ (A* 3D)      │                 │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
│          │                 │                  │                         │
│          ▼                 ▼                  ▼                         │
│         SQLite DB      trimesh GLB       SMACNA/NEC/UPC Tables          │
│         (jobs, users,  export            (engineering code)             │
│          models)                                                        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │  HTTP (requests lib, timeout=120s)
                           │  Auto-discovered via HuggingFace Hub
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              GOOGLE COLAB AI SERVER (T4 GPU — Free Tier)                │
│              FastAPI + ngrok tunnel (public URL, auto-published)        │
│                                                                         │
│   Stage 2: GroundingDINO-tiny  → element detection (walls/doors/rooms) │
│   Stage 3: SAM2-hiera-small    → pixel-accurate polygon masks           │
│   Stage 4: Florence-2-base     → room caption & classification          │
│   Stage 5: DeepSeek-R1 1.5B   → MEP engineering calculations           │
│   Stage 6: Qwen2.5-VL 3B      → equipment placement reasoning          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend Layer

**Location:** `d:/Mirage-AEC/frontend/`  
**Framework:** React 18 + TypeScript + Vite  
**3D Engine:** Three.js via `@react-three/fiber` + `@react-three/drei`

### Pages / Views

| Component | Route / Purpose |
|---|---|
| `HomeView.tsx` | Landing page |
| `LoginPortal.tsx` | Auth (JWT login / register / Google OAuth) |
| `CADWorkspace.tsx` | **Main workspace** — upload floor plan → generate 3D + MEP |
| `GenerateView.tsx` | Text-to-3D generation (prompt based) |
| `DesignBoardView.tsx` | Drag-and-drop design board |
| `GalleryView.tsx` | Model gallery browser |
| `ModelViewer.tsx` | Standalone 3D GLB viewer |
| `Scene3D.tsx` | Three.js scene — renders walls, MEP pipes, ducts |
| `FloorPlan2DViewer.tsx` | 2D floor plan overlay |
| `ArchitectureView.tsx` | About / architecture documentation |
| `Navbar.tsx` | Top navigation bar |
| `AuthModal.tsx` | Login/register modal |
| `LoadingScreen.tsx` | Staged progress loader (8 stages) |

### Generation Flow in Frontend (`CADWorkspace.tsx`)

```
User uploads floor plan image
          │
          ▼
Stage 1: Upload image → POST /api/floorplan/analyze (10%)
          │
          ▼
Stage 2: CV Segmentation response received (25%)
          │
          ▼
Stage 3: POST /api/floorplan/reconstruct-3d (40%)
          │
          ▼
Stage 4: GLB geometry received → load into Three.js (55%)
          │
          ▼
Stage 5: POST /api/floorplan/mep-routing (70%)
          │
          ▼
Stage 6: MEP routes received → render pipes/ducts in scene (85%)
          │
          ▼
Stage 7: BIM metadata displayed (95%)
          │
          ▼
Stage 8: Complete — 3D model live in viewport (100%)
```

---

## 3. Backend Layer

**Location:** `d:/Mirage-AEC/backend/`  
**Framework:** FastAPI + uvicorn  
**Database:** SQLite (async, aiosqlite) via SQLAlchemy  
**Port:** 8000

### API Routers

| File | Prefix | Purpose |
|---|---|---|
| `routers/floorplan.py` | `/api/floorplan` | Upload → analyze → reconstruct → MEP |
| `routers/bim.py` | `/api/bim` | BIM export, COCO dataset generation |
| `routers/generation.py` | `/api/generate` | Text-to-3D (prompt → LLM → Gen API) |
| `routers/models.py` | `/api/models` | Saved 3D model CRUD |
| `routers/auth.py` | `/api/auth` | Register / login / JWT / Google OAuth |
| `routers/system.py` | `/api/system` | Health, AI server status, settings |
| `routers/contact.py` | `/api/contact` | Contact form |
| `routers/sketchfab.py` | `/api/sketchfab` | Sketchfab model search + embed |

### Core Services (Pillars)

```
backend/services/
│
├── cv_segmentation.py         ← PILLAR 1: Image → 2D vectors
│     AI path:   GroundingDINO → SAM2 → Florence-2
│     Fallback:  OpenCV (CLAHE + HoughLinesP + Watershed)
│     Output:    walls[], rooms[], doors[], windows[], columns[]
│
├── geometry_reconstruction.py ← PILLAR 2: 2D vectors → 3D mesh
│     Engine:    trimesh (CSG boolean, polygon extrusion)
│     Output:    GLB binary + BIM JSON schema (IFC4 hierarchy)
│
├── mep_routing.py             ← PILLAR 3: 3D A* MEP pathfinding
│     AI path:   DeepSeek-R1 (MEP specs) + Qwen-VL (equipment placement)
│     Fallback:  SMACNA/NEC/UPC/NFPA engineering tables
│     Output:    HVAC ducts, electrical conduits, plumbing pipes, sprinklers
│
├── ai_client.py               ← HTTP bridge → Colab AI server
│     Auto-discovers URL from HuggingFace Hub
│     Auto-retries discovery on connection failure
│     Graceful fallback to None when server offline
│
├── ai_url_discovery.py        ← Auto-discovers Colab ngrok URL
│     Reads server_config.json from HF Hub (MirageAEC-config dataset)
│     Caches URL for 5 minutes, re-fetches on expiry
│
├── auth_service.py            ← JWT token generation + validation
├── bim_service.py             ← BIM export helpers, COCO dataset
├── shap_e_service.py          ← OpenAI Shap-E text-to-3D (local)
├── llm_service.py             ← Prompt enhancement for text-to-3D
└── gen_service.py             ← 3D generation job queue + pipeline
```

### Key Config (`backend/config.py`)

| Setting | Default | Purpose |
|---|---|---|
| `AI_SERVER_URL` | `""` | Overrides auto-discovery (optional) |
| `HF_TOKEN` | `hf_bjsud...` | HuggingFace token for model access |
| `AI_TIMEOUT_S` | `120` | Max wait per AI stage (seconds) |
| `DATABASE_URL` | SQLite | Async DB connection |
| `SECRET_KEY` | hardcoded | JWT signing key |
| `CORS_ORIGINS` | `["*"]` | CORS allowlist |

---

## 4. AI Pipeline (Colab Server)

**Location:** `d:/Mirage-AEC/colab/mirage_aec_ai_server.py`  
**Runtime:** Google Colab Free T4 GPU (15GB VRAM) or Kaggle T4×2 (30GB)  
**Tunnel:** ngrok (free, URL changes per session — auto-published to HF Hub)

### Model Stack (T4-optimised, ~5GB total VRAM)

| Stage | Model | VRAM | Purpose |
|---|---|---|---|
| 2 | GroundingDINO-tiny | 0.7 GB | Zero-shot object detection — walls, doors, windows, rooms |
| 3 | SAM2-hiera-small | 0.9 GB | Pixel-accurate polygon masks for each detected element |
| 4 | Florence-2-base | 0.3 GB | Visual captioning for room type classification |
| 5 | DeepSeek-R1 1.5B (4-bit) | 1.0 GB | MEP engineering calculations (ASHRAE/NEC/UPC/NFPA) |
| 6 | Qwen2.5-VL 3B (4-bit) | 2.0 GB | Equipment placement via visual spatial reasoning |
| **Total** | | **~5 GB** | Fits T4 comfortably |

### Colab Endpoints

| Method | Path | Model Used | Returns |
|---|---|---|---|
| `GET` | `/health` | — | Status, VRAM used, model list |
| `POST` | `/detect` | GroundingDINO | `detections[]` with bbox + label + conf |
| `POST` | `/segment` | SAM2 | `masks[]` with pixel polygons |
| `POST` | `/classify-rooms` | Florence-2 → Qwen fallback | `classified_rooms[]` with room_type |
| `POST` | `/calculate-mep` | DeepSeek-R1 | MEP specs per room (CFM, pipe sizes, circuits) |
| `POST` | `/place-equipment` | Qwen2.5-VL | AHU/DB/riser/fire main coordinates |
| `POST` | `/check-compliance` | DeepSeek-R1 | ASHRAE/NEC/UPC/NFPA compliance report |

### Auto-Start Sequence (Single Command)

```bash
# In Colab — just run this one line:
!python3 mirage_aec_ai_server.py

# Script automatically:
# [1/7] pip install all packages
# [2/7] HuggingFace login
# [3/7] Load 5 models into GPU
# [4/7] Build FastAPI endpoints
# [5/7] Start uvicorn server (port 8888)
# [6/7] Open ngrok tunnel → get public URL
# [7/7] Publish URL to HF Hub (MirageAEC-config/server_config.json)
#       → backend discovers it automatically, no manual copy needed
```

### URL Auto-Discovery Flow

```
Colab starts
    │ publishes URL to HuggingFace Hub
    ▼
HF Hub: MirageAEC-config/server_config.json
    {
      "ai_server_url": "https://xxxx.ngrok-free.app",
      "timestamp": "2026-09-21T10:00:00Z",
      "models": [...]
    }
    │ backend reads on every request (cached 5 min)
    ▼
ai_url_discovery.py → ai_client.py → cv_segmentation.py / mep_routing.py
```

---

## 5. Data Flow — Floor Plan to 3D MEP

```
INPUT: JPG/PNG floor plan image (2D architectural drawing)

Step 1 — Upload & Scale
  POST /api/floorplan/analyze
  → cv_segmentation.py reads image size, computes px-per-meter scale

Step 2 — AI Detection (GroundingDINO)
  → Colab /detect endpoint
  → Returns bounding boxes: wall, door, window, column, room types

Step 3 — AI Segmentation (SAM2)
  → Colab /segment endpoint
  → Returns precise pixel polygons for each detected element

Step 4 — Room Classification (Florence-2 or Qwen-VL fallback)
  → Colab /classify-rooms endpoint
  → Crops each room, captions it, maps to: kitchen/bath/bed/living/corridor

Step 5 — Fallback (if Colab offline)
  → OpenCV: CLAHE preprocessing → HoughLinesP wall detection
  → Watershed segmentation for rooms
  → Heuristic door/window detection
  → Parametric grid if OpenCV also fails

Step 6 — 3D Geometry (trimesh)
  → geometry_reconstruction.py
  → Extrudes wall polygons to height 3m
  → CSG boolean subtraction for door/window openings
  → RC column boxes at intersections
  → Floor tiles per room polygon
  → Door frames (walnut) + window frames (aluminium) + glass panels
  → Ceiling slab + baseboard trim
  → Export as .glb binary

Step 7 — MEP Calculations (DeepSeek-R1)
  → Colab /calculate-mep endpoint
  → Returns per-room: CFM, duct sizes, circuit IDs, pipe diameters, sprinkler count
  → Fallback: SMACNA 2005 tables + NEC 2023 + UPC 2021 + NFPA-13

Step 8 — Equipment Placement (Qwen-VL)
  → Colab /place-equipment endpoint
  → Returns: AHU coords, electrical DB coords, plumbing riser, fire main
  → Fallback: rule-based positions (near plant room / wet zones)

Step 9 — 3D MEP Routing (A* pathfinding)
  → mep_routing.py
  → VoxelGrid3D (150mm voxel resolution)
  → 4 discipline routes simultaneously: HVAC, Electrical, Plumbing, Fire
  → Clash detection between disciplines
  → BOM with material quantities + cost rates
  → Rectangular duct meshes (SMACNA cross-sections)
  → Sprinkler heads (NFPA-13, 12m² coverage, pendent type)

OUTPUT:
  → GLB file (3D building geometry + MEP networks)
  → BIM JSON (IFC4 hierarchy: site → building → storey → elements)
  → MEP routes JSON (paths, materials, dimensions, costs)
  → Compliance report (ASHRAE/NEC/UPC/NFPA pass/fail)
  → COCO dataset JSON (for future model training)
```

---

## 6. File Structure

```
d:/Mirage-AEC/
│
├── frontend/                          # React + Vite SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── CADWorkspace.tsx       # Main workspace (upload → generate)
│   │   │   ├── Scene3D.tsx            # Three.js 3D renderer
│   │   │   ├── FloorPlan2DViewer.tsx  # 2D plan overlay
│   │   │   ├── ModelViewer.tsx        # GLB model viewer
│   │   │   ├── LoadingScreen.tsx      # Staged progress (8 steps)
│   │   │   ├── LoginPortal.tsx        # Auth UI
│   │   │   └── ...14 more components
│   │   ├── App.tsx                    # Router + layout
│   │   ├── index.css                  # Global design system
│   │   └── config.ts                  # API base URL config
│   └── package.json
│
├── backend/                           # FastAPI Python server
│   ├── main.py                        # App entry point, router mounts
│   ├── config.py                      # Settings (AI_SERVER_URL, tokens)
│   ├── database.py                    # SQLite async setup
│   ├── models.py                      # SQLAlchemy ORM models
│   ├── schemas.py                     # Pydantic request/response schemas
│   ├── routers/
│   │   ├── floorplan.py               # /api/floorplan/* endpoints
│   │   ├── generation.py              # /api/generate/* endpoints
│   │   ├── bim.py                     # /api/bim/* endpoints
│   │   ├── auth.py                    # /api/auth/* endpoints
│   │   ├── system.py                  # /api/system/* endpoints
│   │   ├── models.py                  # /api/models/* endpoints
│   │   ├── sketchfab.py               # /api/sketchfab/* endpoints
│   │   └── contact.py                 # /api/contact/* endpoints
│   └── services/
│       ├── cv_segmentation.py         # PILLAR 1: Image → 2D vectors
│       ├── geometry_reconstruction.py # PILLAR 2: 2D → 3D mesh (trimesh)
│       ├── mep_routing.py             # PILLAR 3: A* MEP pathfinding
│       ├── ai_client.py               # HTTP client → Colab AI server
│       ├── ai_url_discovery.py        # Auto-discovers ngrok URL from HF Hub
│       ├── auth_service.py            # JWT utils
│       ├── bim_service.py             # BIM export helpers
│       ├── shap_e_service.py          # Shap-E text-to-3D
│       ├── llm_service.py             # Prompt enhancement
│       └── gen_service.py             # Generation job queue
│
├── colab/
│   └── mirage_aec_ai_server.py       # FULL AUTO-RUN Colab script
│                                      # Run: !python3 mirage_aec_ai_server.py
│                                      # Installs, loads 5 models, starts server,
│                                      # opens ngrok, publishes URL to HF Hub
│
├── mirage_aec.db                      # SQLite database (auto-created)
├── start.ps1                          # Windows: starts backend + frontend
└── start.bat                          # Windows batch launcher
```

---

## 7. Engineering Standards Used

| Discipline | Standard | Applied In |
|---|---|---|
| HVAC sizing | ASHRAE 62.1-2022, SMACNA 2005 | DeepSeek-R1 MEP calc + `smacna_duct_size()` |
| Electrical | NEC 2023 Article 220, 110.26 | DeepSeek-R1 + circuit grouping |
| Plumbing | UPC 2021 Table 703.2 (DFU) | DeepSeek-R1 + `upc_pipe_size()` |
| Fire protection | NFPA 13-2022 | Sprinkler grid (12m² coverage, K=5.6) |
| Energy | ASHRAE 90.1-2022 | Compliance check endpoint |
| BIM schema | buildingSMART IFC4 | BIM JSON output |
| 3D routing | 3D A* on 150mm voxel grid | `VoxelGrid3D` + `astar_3d()` |

---

## 8. How to Run Locally

### Start Everything

```powershell
# Option 1: Use the start script
cd d:\Mirage-AEC
.\start.ps1

# Option 2: Manual
# Terminal 1 — Backend
cd d:\Mirage-AEC\backend
.\.venv\Scripts\activate
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd d:\Mirage-AEC\frontend
npm run dev
```

### Start the AI Server (Colab)

```python
# In Google Colab (Runtime > T4 GPU):
from google.colab import files
files.upload()          # upload mirage_aec_ai_server.py from d:\Mirage-AEC\colab\

# Then run:
!python3 mirage_aec_ai_server.py
# Takes ~7 min first time, ~4 min after (models cached)
# URL auto-published to HF Hub → backend finds it automatically
```

---

## 9. Fallback Hierarchy

> The system **never crashes** — each AI stage has a fallback.

```
Stage 2–4 (Image Understanding):
  GroundingDINO + SAM2 + Florence-2
      ↓ (if Colab offline)
  OpenCV: CLAHE + HoughLinesP + Watershed
      ↓ (if OpenCV fails or finds < 4 walls)
  Parametric grid (rule-based default layout)

Stage 5–6 (MEP Intelligence):
  DeepSeek-R1 + Qwen-VL
      ↓ (if Colab offline)
  SMACNA/NEC/UPC/NFPA lookup tables
      ↓ (always runs)
  3D A* routing on voxel grid

Stage 4 Florence-2 specifically:
  Florence-2-base
      ↓ (if transformers version incompatible)
  Qwen2.5-VL-3B (also on Colab, asks "what room type is this?")
```

---

## 10. Tokens & Credentials

> Stored in `backend/config.py` — do NOT commit to public repos.

| Token | Used For |
|---|---|
| `HF_TOKEN` (hf_bjsud...) | HuggingFace model downloads + HF Hub config store |
| `NGROK_TOKEN` (3JdFq...) | ngrok tunnel authentication in Colab |
| `SECRET_KEY` | JWT signing (backend) |
