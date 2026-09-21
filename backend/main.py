import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure package and root paths are available for uvicorn
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings
from database import engine, Base
from routers import auth, generation, models, contact, system, floorplan, bim

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[STARTUP] MirageAEC Database tables verified and initialized.")
    yield
    await engine.dispose()

app = FastAPI(
    title="MirageAEC - Cloud AI BIM & Generative CAD Engine",
    description="MirageAEC Cloud Platform: Floor Plan Computer Vision, 2D-to-3D Parametric BIM Reconstruction, and Automated 3D A* MEP Pathfinding",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(floorplan.router, prefix=settings.API_V1_STR)
app.include_router(bim.router, prefix=settings.API_V1_STR)
app.include_router(generation.router, prefix=settings.API_V1_STR)
app.include_router(models.router, prefix=settings.API_V1_STR)
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(contact.router, prefix=settings.API_V1_STR)
app.include_router(system.router, prefix=settings.API_V1_STR)

# Serve generated .glb files at /static/models/{job_id}.glb
STATIC_DIR = Path(__file__).parent / "static" / "models"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/models", StaticFiles(directory=str(STATIC_DIR)), name="static_models")

# Serve pre-built frontend SPA (Zero-build-time deployment)
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend_spa")
else:
    @app.get("/")
    async def root():
        return {
            "app": "MirageAEC - API Gateway",
            "tagline": "IDEAS TODAY. REAL SPACES TOMORROW.",
            "version": "1.0.0",
            "docs_url": "/docs",
            "status": "online"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
