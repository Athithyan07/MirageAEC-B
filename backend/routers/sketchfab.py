import os
import re
import httpx
from typing import Optional
from fastapi import APIRouter, Query, Header, HTTPException

router = APIRouter(prefix="/sketchfab", tags=["Sketchfab 3D API"])

SKETCHFAB_API_BASE = "https://api.sketchfab.com/v3"

def extract_sketchfab_uid(input_str: str) -> str:
    """Extract 32-character hex UID from a Sketchfab URL or return raw UID."""
    input_str = input_str.strip()
    # If input is a URL like https://sketchfab.com/3d-models/modern-house-613d9a3b8d1d4d3885d58d928d11c8d6
    match = re.search(r'([a-f0-9]{32})', input_str, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return input_str

@router.get("/search")
async def search_sketchfab_models(
    q: str = Query("modern architecture", description="Search query for Sketchfab 3D models"),
    count: int = Query(12, ge=1, le=24, description="Number of models to fetch"),
    categories: Optional[str] = Query(None, description="Category filter e.g. architecture")
):
    """
    Search public 3D models on Sketchfab with architectural and design filters.
    """
    headers = {"Accept": "application/json"}
    token = os.environ.get("SKETCHFAB_API_TOKEN")
    if token:
        headers["Authorization"] = f"Token {token}"

    params = {
        "type": "models",
        "q": q,
        "count": count,
        "sort_by": "-likeCount",
    }
    if categories and categories not in ("all", "none"):
        params["categories"] = categories

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SKETCHFAB_API_BASE}/search", params=params, headers=headers)
            if resp.status_code != 200:
                # Fallback to general search without category constraint
                fallback_params = {"type": "models", "q": q, "count": count}
                resp = await client.get(f"{SKETCHFAB_API_BASE}/search", params=fallback_params, headers=headers)

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Sketchfab API error")

            data = resp.json()
            results = []
            for item in data.get("results", []):
                uid = item.get("uid")
                thumbnails = item.get("thumbnails", {}).get("images", [])
                # Get the mid-to-high resolution thumbnail
                thumb_url = None
                if thumbnails:
                    # Pick an image with width between 256 and 1024 if available
                    for img in thumbnails:
                        if img.get("width", 0) >= 256:
                            thumb_url = img.get("url")
                            break
                    if not thumb_url:
                        thumb_url = thumbnails[-1].get("url")

                results.append({
                    "uid": uid,
                    "name": item.get("name", "Untitled Model"),
                    "description": (item.get("description") or "")[:200],
                    "author": item.get("user", {}).get("displayName", "Sketchfab Creator"),
                    "author_username": item.get("user", {}).get("username"),
                    "thumbnail_url": thumb_url,
                    "viewer_url": item.get("viewerUrl"),
                    "embed_url": f"https://sketchfab.com/models/{uid}/embed",
                    "face_count": item.get("faceCount", 0),
                    "vertex_count": item.get("vertexCount", 0),
                    "like_count": item.get("likeCount", 0),
                    "is_downloadable": item.get("isDownloadable", False),
                })

            return {
                "query": q,
                "total_count": len(results),
                "results": results
            }

    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Sketchfab: {str(exc)}")

@router.get("/model/{uid_or_url:path}")
async def get_sketchfab_model(uid_or_url: str):
    """
    Get detailed metadata for a single Sketchfab 3D model by UID or URL.
    """
    uid = extract_sketchfab_uid(uid_or_url)
    headers = {"Accept": "application/json"}
    token = os.environ.get("SKETCHFAB_API_TOKEN")
    if token:
        headers["Authorization"] = f"Token {token}"

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{SKETCHFAB_API_BASE}/models/{uid}", headers=headers)
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Sketchfab model not found")
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Failed to fetch model from Sketchfab")

            item = resp.json()
            thumbnails = item.get("thumbnails", {}).get("images", [])
            thumb_url = thumbnails[-1].get("url") if thumbnails else None

            return {
                "uid": uid,
                "name": item.get("name"),
                "description": item.get("description"),
                "author": item.get("user", {}).get("displayName"),
                "author_username": item.get("user", {}).get("username"),
                "thumbnail_url": thumb_url,
                "viewer_url": item.get("viewerUrl"),
                "embed_url": f"https://sketchfab.com/models/{uid}/embed?autostart=1&ui_controls=1&ui_infos=0&ui_watermark=0",
                "face_count": item.get("faceCount", 0),
                "vertex_count": item.get("vertexCount", 0),
                "like_count": item.get("likeCount", 0),
                "is_downloadable": item.get("isDownloadable", False),
                "license": item.get("license", {}).get("label", "Standard"),
            }
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Sketchfab: {str(exc)}")

@router.get("/download/{uid}")
async def download_sketchfab_model(
    uid: str,
    x_sketchfab_token: Optional[str] = Header(None, alias="X-Sketchfab-Token")
):
    """
    Get direct downloadable GLB/glTF link from Sketchfab (requires Sketchfab API Token).
    """
    token = x_sketchfab_token or os.environ.get("SKETCHFAB_API_TOKEN")
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Sketchfab API Token required to generate direct GLB binary download links. Alternatively, use the interactive 3D WebGL embed loader."
        )

    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {token}"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{SKETCHFAB_API_BASE}/models/{uid}/download", headers=headers)
            if resp.status_code != 200:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=data.get("detail", "Unable to retrieve download URL for this model.")
                )

            data = resp.json()
            return {
                "uid": uid,
                "gltf_url": data.get("gltf", {}).get("url"),
                "usdz_url": data.get("usdz", {}).get("url"),
                "expires_in": data.get("gltf", {}).get("expires", 300)
            }
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Sketchfab: {str(exc)}")
