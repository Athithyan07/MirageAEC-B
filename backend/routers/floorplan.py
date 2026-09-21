"""
Floor Plan CV & Feature Extraction Router for MirageAEC
Endpoints for image uploading, OpenCV/AI segmentation, COCO dataset export, and archetype loading.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from fastapi.responses import JSONResponse

from services.cv_segmentation import (
    segment_floorplan_image,
    get_preset_floorplan,
    list_preset_floorplans,
    generate_coco_dataset,
    PRESET_FLOORPLANS
)

router = APIRouter(prefix="/floorplan", tags=["Floor Plan Computer Vision & Segmentation"])

@router.get("/presets", summary="List standard architectural floor plan archetypes")
async def get_presets():
    """
    Returns available architectural plan templates (2-BHK, 3-BHK Villa, Commercial Office, etc.).
    """
    return list_preset_floorplans()

@router.get("/preset/{preset_id}", summary="Get vector details of a specific archetype")
async def get_preset_details(preset_id: str):
    """
    Returns full 2D vector polygons, room boundaries, wall coordinates, and equipment sources for a preset.
    """
    if preset_id not in PRESET_FLOORPLANS:
        raise HTTPException(status_code=404, detail=f"Floorplan archetype '{preset_id}' not found.")
    return get_preset_floorplan(preset_id)

@router.post("/upload-and-segment", summary="Upload raster floor plan image and extract 2D vectors")
async def upload_and_segment(
    file: UploadFile = File(...),
    scale_pixels_per_meter: float = Query(50.0, description="Pixels per real-world meter scale")
):
    """
    Uploads a floor plan image (PNG, JPG, WebP) and executes OpenCV edge filtering,
    Douglas-Peucker polygon vectorization, and room boundary extraction.
    """
    try:
        image_bytes = await file.read()
        segmented = segment_floorplan_image(image_bytes, scale_pixels_per_meter=scale_pixels_per_meter)
        return segmented
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Floor plan segmentation error: {str(e)}")

@router.get("/coco-dataset", summary="Export COCO Instance Segmentation Dataset")
async def export_coco_dataset():
    """
    Generates standard COCO JSON annotations across all registered floor plans for CV training.
    """
    plans = list(PRESET_FLOORPLANS.values())
    coco_json = generate_coco_dataset(plans)
    return JSONResponse(content=coco_json)

@router.post("/analyze-and-build", summary="Upload floorplan, segment, and build full 3D BIM")
async def analyze_and_build(
    file: UploadFile = File(...),
    ceiling_height_m: float = Form(3.0),
    wall_thickness_m: float = Form(0.2),
    discipline: str = Form("All Disciplines")
):
    try:
        from services.bim_service import synthesize_complete_bim_project
        image_bytes = await file.read()
        
        # Pass image_bytes to synthesis pipeline which automatically segments it
        result = synthesize_complete_bim_project(
            preset_id=None,
            image_bytes=image_bytes,
            ceiling_height_m=ceiling_height_m,
            wall_thickness_m=wall_thickness_m,
            include_mep_in_glb=True
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
