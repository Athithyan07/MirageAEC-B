"""
Comprehensive BIM & MEP Synthesis Orchestrator for MirageAEC
Connects Computer Vision Segmentation, 3D Parametric Reconstruction,
and Automated 3D A* MEP Routing into a unified cloud CAD engine.
"""

import io
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
import trimesh

from services.cv_segmentation import (
    segment_floorplan_image,
    get_preset_floorplan,
    list_preset_floorplans,
    generate_coco_dataset
)
from services.geometry_reconstruction import build_3d_bim_model
from services.mep_routing import (
    solve_mep_routing,
    generate_tube_mesh
)

STATIC_MODELS_DIR = Path(__file__).parent.parent / "static" / "models"
STATIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

def synthesize_complete_bim_project(
    preset_id: Optional[str] = "2bhk_modern",
    image_bytes: Optional[bytes] = None,
    ceiling_height_m: float = 3.0,
    wall_thickness_m: float = 0.2,
    include_mep_in_glb: bool = True
) -> Dict[str, Any]:
    """
    Executes full pipeline: Floorplan Vector Extraction -> 3D Geometry Extrusion -> 3D A* MEP Routing -> GLB/BIM Schema Export.
    """
    job_id = f"bim_{uuid.uuid4().hex[:12]}"
    glb_filename = f"{job_id}.glb"
    glb_path = STATIC_MODELS_DIR / glb_filename

    # Step 1: Ingest & Segment Floorplan
    if image_bytes:
        floorplan = segment_floorplan_image(image_bytes)
    else:
        floorplan = get_preset_floorplan(preset_id or "2bhk_modern")

    # Step 2: 3D Geometry Reconstruction
    geom_result = build_3d_bim_model(
        floorplan_data=floorplan,
        ceiling_height_m=ceiling_height_m,
        wall_thickness_m=wall_thickness_m,
        export_glb_path=None # We will export unified scene below
    )

    # Step 3: Solve Automated 3D A* MEP Routing
    mep_result = solve_mep_routing(
        floorplan_data=floorplan,
        ceiling_height_m=ceiling_height_m
    )

    # Step 4: Build Unified Three.js / WebGL Scene with architectural & MEP geometries
    unified_scene = trimesh.Scene()
    
    # Add architectural meshes from geometry reconstruction
    temp_scene = trimesh.load(io.BytesIO(geom_result["glb_bytes"]), file_type='glb')
    if isinstance(temp_scene, trimesh.Scene):
        for name, geom in temp_scene.geometry.items():
            unified_scene.add_geometry(geom, node_name=f"arch_{name}")
    elif isinstance(temp_scene, trimesh.Trimesh):
        unified_scene.add_geometry(temp_scene, node_name="arch_base")

    # Generate 3D Tube/Duct Meshes for MEP disciplines to embed directly in the GLB
    if include_mep_in_glb:
        # HVAC Ducts (Blue #0088FF)
        for hvac_route in mep_result.get("hvac", {}).get("routes", []):
            mesh = generate_tube_mesh(hvac_route["path"], radius=0.12, color=[0, 136, 255, 255])
            if mesh:
                unified_scene.add_geometry(mesh, node_name=f"mep_hvac_{hvac_route['id']}")

        # Electrical Conduits (Yellow #FFCC00)
        for elec_route in mep_result.get("electrical", {}).get("routes", []):
            mesh = generate_tube_mesh(elec_route["path"], radius=0.035, color=[255, 204, 0, 255])
            if mesh:
                unified_scene.add_geometry(mesh, node_name=f"mep_elec_{elec_route['id']}")

        # Plumbing Supply (Emerald Green #00CC88)
        for plumb_sup in mep_result.get("plumbing", {}).get("supply_routes", []):
            mesh = generate_tube_mesh(plumb_sup["path"], radius=0.045, color=[0, 204, 136, 255])
            if mesh:
                unified_scene.add_geometry(mesh, node_name=f"mep_plumb_sup_{plumb_sup['id']}")

        # Plumbing Drainage (Terracotta Orange #E65100)
        for plumb_drn in mep_result.get("plumbing", {}).get("drain_routes", []):
            mesh = generate_tube_mesh(plumb_drn["path"], radius=0.065, color=[230, 81, 0, 255])
            if mesh:
                unified_scene.add_geometry(mesh, node_name=f"mep_plumb_drn_{plumb_drn['id']}")

    # Rotate the scene from Z-up to Y-up for standard WebGL (Three.js) compatibility
    import math
    rot_matrix = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
    for geom in unified_scene.geometry.values():
        geom.apply_transform(rot_matrix)

    # Export Unified GLB
    try:
        final_glb_bytes = trimesh.exchange.gltf.export_glb(unified_scene)
        with open(glb_path, "wb") as f:
            f.write(final_glb_bytes)
        model_url = f"/static/models/{glb_filename}"
    except Exception as e:
        print(f"[ERROR] Exporting unified GLB: {e}")
        model_url = ""

    # Merge BIM Schema with MEP details and Quantities
    bim_schema = geom_result["bim_schema"]
    bim_schema["mep_systems"] = {
        "hvac": mep_result["hvac"],
        "electrical": mep_result["electrical"],
        "plumbing": mep_result["plumbing"]
    }
    bim_schema["bill_of_materials"] = mep_result["bill_of_materials"]

    # Calculate scene geometry vertex and triangle sums
    total_triangles = sum(len(g.faces) for g in unified_scene.geometry.values() if hasattr(g, 'faces'))
    total_vertices = sum(len(g.vertices) for g in unified_scene.geometry.values() if hasattr(g, 'vertices'))

    # Remove raw image bytes before JSON serialization to prevent FastAPI UnicodeDecodeError
    if "_raw_image_bytes" in floorplan:
        del floorplan["_raw_image_bytes"]

    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "floorplan": floorplan,
        "model_url": model_url,
        "bim_schema": bim_schema,
        "mep_routes": mep_result,
        "metrics": {
            "triangles": total_triangles,
            "vertices": total_vertices,
            "elements": geom_result["element_count"],
            "hvac_ducts_count": len(mep_result.get("hvac", {}).get("routes", [])),
            "elec_conduits_count": len(mep_result.get("electrical", {}).get("routes", [])),
            "plumbing_pipes_count": len(mep_result.get("plumbing", {}).get("supply_routes", [])) + len(mep_result.get("plumbing", {}).get("drain_routes", []))
        }
    }
