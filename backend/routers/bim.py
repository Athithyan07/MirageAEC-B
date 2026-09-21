"""
BIM & Automated MEP Routing Router for MirageAEC
Endpoints for 2D-to-3D parametric reconstruction, 3D A* MEP route calculation, and Bill of Materials quantification.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from services.bim_service import synthesize_complete_bim_project
from services.cv_segmentation import get_preset_floorplan
from services.geometry_reconstruction import build_3d_bim_model
from services.mep_routing import solve_mep_routing

router = APIRouter(prefix="/bim", tags=["BIM 3D Reconstruction & MEP Routing"])

class SynthesizeBIMRequest(BaseModel):
    preset_id: Optional[str] = Field("2bhk_modern", description="Archetype identifier")
    floorplan_data: Optional[Dict[str, Any]] = Field(None, description="Custom 2D floor plan JSON schema")
    ceiling_height_m: float = Field(3.0, ge=2.2, le=6.0, description="Ceiling clear height in meters")
    wall_thickness_m: float = Field(0.2, ge=0.1, le=0.5, description="Outer wall thickness in meters")
    include_mep_in_glb: bool = Field(True, description="Whether to bake 3D MEP conduits & ducts into the GLB export")

@router.post("/synthesize", summary="Synthesize full 3D BIM model and automated MEP routes")
async def synthesize_bim(request: SynthesizeBIMRequest):
    """
    Executes end-to-end BIM generation:
    1. Reads 2D floorplan vectors.
    2. Generates parametric 3D solid meshes and opening cutouts.
    3. Runs 3D A* pathfinding for HVAC ducts, electrical conduits, and sloped plumbing pipes.
    4. Produces unified WebGL GLB export and Hierarchical BIM Schema + BOM.
    """
    try:
        result = synthesize_complete_bim_project(
            preset_id=request.preset_id,
            ceiling_height_m=request.ceiling_height_m,
            wall_thickness_m=request.wall_thickness_m,
            include_mep_in_glb=request.include_mep_in_glb
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BIM synthesis pipeline error: {str(e)}")

@router.post("/mep-route-only", summary="Calculate only 3D A* MEP paths for an existing plan")
async def calculate_mep_only(request: SynthesizeBIMRequest):
    """
    Runs isolated 3D A* routing engine without full GLB geometry rebuild.
    """
    try:
        floorplan = request.floorplan_data or get_preset_floorplan(request.preset_id or "2bhk_modern")
        mep_result = solve_mep_routing(floorplan, ceiling_height_m=request.ceiling_height_m)
        return mep_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MEP calculation error: {str(e)}")
