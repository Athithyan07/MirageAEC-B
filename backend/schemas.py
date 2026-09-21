from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

# Auth Schemas
class GoogleAuthRequest(BaseModel):
    id_token: str = Field(..., description="Google OAuth credential / JWT ID token")

class DemoLoginRequest(BaseModel):
    email: Optional[EmailStr] = "architect@mirageaec.com"
    name: Optional[str] = "Julian Vance"

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    avatar_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# 3D Generation Schemas
class GeneratePromptRequest(BaseModel):
    prompt: str = Field(..., min_length=2, example="A modern chair with minimalist design...")
    model_type: Optional[str] = "3D Model"
    style: Optional[str] = "Minimalist"
    material: Optional[str] = "Leather"
    quality: Optional[str] = "Ultra 4K"
    enable_llm_enhancement: Optional[bool] = True

class GenerationJobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    status_message: str
    prompt: str
    enhanced_prompt: Optional[str] = None
    style: str
    material: str
    quality: str
    model_id: Optional[int] = None
    model_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class Model3DResponse(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    prompt: str
    category: str
    style: str
    material: str
    quality: str
    mesh_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    poly_count: int
    likes_count: int
    views_count: int
    parameters: dict[str, Any] = {}
    created_at: datetime

    class Config:
        from_attributes = True

# Contact Schema
class ContactFormRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    subject: str = Field(..., min_length=2, max_length=150)
    message: str = Field(..., min_length=5, max_length=5000)

class ContactFormResponse(BaseModel):
    success: bool
    message: str
    inquiry_id: int

# System Architecture Schemas
class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    components: dict[str, Any]
    active_jobs: int
    total_models: int

class SimulateFlowRequest(BaseModel):
    action: str = Field(..., example="full_generation_pipeline") # full_generation_pipeline, oauth_handshake, llm_prompt_refine, pg_record_sync
    payload: Optional[dict[str, Any]] = None
