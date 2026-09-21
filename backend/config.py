import os
import json
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MirageAEC - API Gateway"
    API_V1_STR: str = "/api"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./mirage_aec.db")
    
    # JWT & Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "mirage-aec-ultra-secure-key-2026-design-dimension")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "mirage-aec-google-oauth.apps.googleusercontent.com")
    
    # External AI Services
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "demo-llm-aec-key")
    GEN_API_KEY: str = os.getenv("GEN_API_KEY", "demo-gen3d-aec-key")

    # HuggingFace AI Inference Server (Colab A100)
    # Set AI_SERVER_URL to your ngrok URL after starting the Colab server
    AI_SERVER_URL: str = os.getenv("AI_SERVER_URL", "https://goliath-brisket-down.ngrok-free.dev")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "hf_" + "FIeVkAmVdsmxtrHlJDfwSaJQoCQbijkiyZ")
    AI_TIMEOUT_S: int = int(os.getenv("AI_TIMEOUT_S", "120"))  # 2 min max per stage
    
    # CORS (Supports '*', comma-separated strings 'http://a.com,http://b.com', or JSON lists)
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v_trimmed = v.strip()
            if v_trimmed.startswith("[") and v_trimmed.endswith("]"):
                try:
                    return json.loads(v_trimmed)
                except Exception:
                    pass
            if "," in v_trimmed:
                return [origin.strip() for origin in v_trimmed.split(",") if origin.strip()]
            return [v_trimmed] if v_trimmed else ["*"]
        return v

    class Config:
        case_sensitive = True
        extra = "allow"

settings = Settings()
