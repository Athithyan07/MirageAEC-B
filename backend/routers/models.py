from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import Model3D
from schemas import Model3DResponse

router = APIRouter(prefix="/models", tags=["3D Model Catalog"])

# Curated showroom models matching the design showcase
DEFAULT_MODELS = [
    {
        "id": 1,
        "title": "Modern Cabin",
        "description": "Cantilevered luxury wooden cabin situated on a floating stone island with crystal waterfalls and pine foliage.",
        "prompt": "Floating architectural modern timber cabin on rock monolith with cascading waterfall",
        "category": "Architecture",
        "style": "Organic Modern",
        "material": "Cedar Wood & Slate Stone",
        "quality": "Ultra 4K",
        "poly_count": 284000,
        "likes_count": 4820,
        "views_count": 31200,
        "thumbnail_url": "https://images.unsplash.com/photo-1518780664697-55e3ad937233?auto=format&fit=crop&w=800&q=80",
        "parameters": {
            "model_type": "cabin_island",
            "has_waterfall": True,
            "has_particles": True,
            "lighting": "sunset_golden"
        }
    },
    {
        "id": 2,
        "title": "Modern Lounge Chair",
        "description": "Ergonomic tub armchair with bent walnut shell, brushed steel legs, and premium charcoal nappa leather upholstery.",
        "prompt": "A modern chair with minimalist design, curved oak wood frame and charcoal leather seat",
        "category": "Furniture",
        "style": "Minimalist",
        "material": "Leather & Oak",
        "quality": "Ultra 4K",
        "poly_count": 142000,
        "likes_count": 6240,
        "views_count": 48900,
        "thumbnail_url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?auto=format&fit=crop&w=800&q=80",
        "parameters": {
            "model_type": "lounge_chair",
            "cushion_color": "#23262D",
            "wood_color": "#B8860B"
        }
    },
    {
        "id": 3,
        "title": "Minimal House",
        "description": "Geometric two-story concrete and glass residence with floor-to-ceiling panoramic windows and cantilever roof.",
        "prompt": "Modern architectural villa with cantilevered glass balcony and warm interior luminescence",
        "category": "Architecture",
        "style": "Brutalist",
        "material": "Polished Concrete & Tinted Glass",
        "quality": "Ultra 4K",
        "poly_count": 310000,
        "likes_count": 3910,
        "views_count": 27400,
        "thumbnail_url": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=800&q=80",
        "parameters": {
            "model_type": "minimal_house",
            "has_terrace": True
        }
    },
    {
        "id": 4,
        "title": "Hyper Sports Car",
        "description": "Low-slung aerodynamic electric concept vehicle with carbon-fiber aero blades and illuminated LED matrix lightbar.",
        "prompt": "Futuristic aerodynamic electric sports car in obsidian black with glowing taillights",
        "category": "Vehicle",
        "style": "Neo-Cyber",
        "material": "Matte Carbon & Brushed Titanium",
        "quality": "Ultra 4K",
        "poly_count": 420000,
        "likes_count": 8190,
        "views_count": 63100,
        "thumbnail_url": "https://images.unsplash.com/photo-1617788138017-80ad40651399?auto=format&fit=crop&w=800&q=80",
        "parameters": {
            "model_type": "sports_car",
            "body_color": "#121418"
        }
    },
    {
        "id": 5,
        "title": "Sculptural Indoor Plant",
        "description": "Curated architectural fiddle leaf bonsai in a fluted textured ceramic vessel.",
        "prompt": "Sculptural indoor fiddle leaf fig tree in a hand-crafted ceramic planter",
        "category": "Nature",
        "style": "Biophilic",
        "material": "Terra Cotta & Organic Foliage",
        "quality": "Ultra 4K",
        "poly_count": 98000,
        "likes_count": 2410,
        "views_count": 15800,
        "thumbnail_url": "https://images.unsplash.com/photo-1485955900006-10f4d324d411?auto=format&fit=crop&w=800&q=80",
        "parameters": {
            "model_type": "indoor_plant",
            "planter_color": "#ECE7DF"
        }
    }
]

@router.get("", response_model=list[dict])
async def list_models(db: AsyncSession = Depends(get_db)):
    """
    Returns curated 3D models available for viewing and generation.
    """
    return DEFAULT_MODELS

@router.get("/{model_id}")
async def get_model(model_id: int):
    for m in DEFAULT_MODELS:
        if m["id"] == model_id:
            return m
    return DEFAULT_MODELS[0]
