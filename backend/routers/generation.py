import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models import Model3D, GenerationJob
from schemas import GeneratePromptRequest, GenerationJobResponse
from services.llm_service import llm_service
from services.gen_service import gen_service

router = APIRouter(prefix="/generate", tags=["3D Generation Pipeline (LLM + Gen API)"])

@router.post("/prompt", response_model=GenerationJobResponse)
async def submit_generation(
    req: GeneratePromptRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Client Browser -> API Gateway -> LLM API -> Gen API.
    1. Sends prompt to LLM API for architectural token decomposition.
    2. Enqueues 3D generation job in Gen API.
    3. Runs synthesis in background task.
    """
    # 1. LLM API Enhancement
    llm_analysis = await llm_service.enhance_prompt(
        prompt=req.prompt,
        style=req.style or "Minimalist",
        material=req.material or "Leather",
        quality=req.quality or "Ultra 4K"
    )

    # 2. Gen API Job Creation
    job_id = gen_service.create_job(
        prompt=req.prompt,
        style=req.style or "Minimalist",
        material=req.material or "Leather",
        quality=req.quality or "Ultra 4K",
        llm_data=llm_analysis
    )

    # 3. Store Job in Database
    db_job = GenerationJob(
        id=job_id,
        prompt=req.prompt,
        style=req.style or "Minimalist",
        material=req.material or "Leather",
        quality=req.quality or "Ultra 4K",
        status="queued",
        progress=5.0,
        status_message="Submitted to Gen API synthesis cluster...",
        llm_analysis=llm_analysis
    )
    db.add(db_job)
    await db.commit()

    # 4. Fire background pipeline execution
    background_tasks.add_task(gen_service.run_pipeline_step, job_id)

    job_state = gen_service.get_job(job_id)
    return {
        "job_id": job_id,
        "status": job_state["status"],
        "progress": job_state["progress"],
        "status_message": job_state["status_message"],
        "prompt": req.prompt,
        "enhanced_prompt": llm_analysis["enhanced_prompt"],
        "style": req.style or "Minimalist",
        "material": req.material or "Leather",
        "quality": req.quality or "Ultra 4K",
        "model_id": None,
        "model_url": None,
        "created_at": job_state["created_at"],
        "updated_at": job_state["updated_at"]
    }

@router.get("/status/{job_id}", response_model=GenerationJobResponse)
async def check_generation_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Polls real-time generation progress from Gen API & PostgreSQL.
    """
    job_state = gen_service.get_job(job_id)
    if not job_state:
        raise HTTPException(status_code=404, detail=f"Generation job {job_id} not found")

    return {
        "job_id": job_id,
        "status": job_state["status"],
        "progress": job_state["progress"],
        "status_message": job_state["status_message"],
        "prompt": job_state["prompt"],
        "enhanced_prompt": job_state["llm_data"].get("enhanced_prompt"),
        "style": job_state["style"],
        "material": job_state["material"],
        "quality": job_state["quality"],
        "model_id": 1 if job_state.get("model_url") else None,
        "model_url": job_state.get("model_url"),
        "created_at": job_state["created_at"],
        "updated_at": job_state["updated_at"]
    }

@router.get("/presets")
async def get_generation_presets():
    """
    Returns curated style, material, and quality choices shown on the UI pills.
    """
    return {
        "models": ["3D Model", "Mesh Wireframe", "Point Cloud", "BIM IFC", "CAD Solid"],
        "styles": ["Minimalist", "Scandinavian", "Brutalist", "Organic Modern", "Futuristic Neo-Tokyo", "Biophilic", "Parametric"],
        "materials": ["Leather", "Solid Oak", "Brushed Titanium", "Matte Carbon", "Polished Concrete", "Smoked Glass", "Terra Cotta"],
        "qualities": ["Draft (Preview)", "Standard (Web)", "Ultra 4K (Production)"],
        "quick_suggestions": [
            {"label": "Modern Chair", "prompt": "A modern chair with minimalist design, curved oak wood frame and charcoal leather seat"},
            {"label": "Minimal House", "prompt": "Minimalist architectural villa with cantilevered glass balcony and infinity pool"},
            {"label": "Sports Car", "prompt": "Futuristic aerodynamic electric sports car in obsidian black with glowing taillights"},
            {"label": "Indoor Plant", "prompt": "Sculptural indoor fiddle leaf fig tree in a hand-crafted ceramic planter"}
        ]
    }
