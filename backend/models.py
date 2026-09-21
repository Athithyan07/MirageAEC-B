from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    avatar_url = Column(String(512), nullable=True)
    google_id = Column(String(255), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    models = relationship("Model3D", back_populates="creator")
    jobs = relationship("GenerationJob", back_populates="user")

class Model3D(Base):
    __tablename__ = "models_3d"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    prompt = Column(Text, nullable=False)
    category = Column(String(100), default="Architecture") # Architecture, Furniture, Vehicle, Nature
    style = Column(String(100), default="Minimalist")
    material = Column(String(100), default="Oak & Concrete")
    quality = Column(String(50), default="Ultra 4K")
    
    # 3D assets & parameters
    mesh_url = Column(String(512), nullable=True)
    thumbnail_url = Column(String(512), nullable=True)
    poly_count = Column(Integer, default=125000)
    parameters = Column(JSON, default=dict)
    likes_count = Column(Integer, default=0)
    views_count = Column(Integer, default=0)
    
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    creator = relationship("User", back_populates="models")
    created_at = Column(DateTime, default=datetime.utcnow)

class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    id = Column(String(64), primary_key=True, index=True) # UUID string
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    prompt = Column(Text, nullable=False)
    style = Column(String(100), default="Minimalist")
    material = Column(String(100), default="Leather & Metal")
    quality = Column(String(50), default="Ultra 4K")
    
    status = Column(String(50), default="queued") # queued, analyzing_llm, generating_mesh, synthesizing_textures, completed, failed
    progress = Column(Float, default=0.0) # 0 to 100
    status_message = Column(String(255), default="Job initialized in queue...")
    
    result_model_id = Column(Integer, ForeignKey("models_3d.id"), nullable=True)
    llm_analysis = Column(JSON, default=dict)
    gen_api_metrics = Column(JSON, default=dict)
    
    user = relationship("User", back_populates="jobs")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ContactInquiry(Base):
    __tablename__ = "contact_inquiries"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String(50), default="new")
    created_at = Column(DateTime, default=datetime.utcnow)
