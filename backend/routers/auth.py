from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import User
from schemas import GoogleAuthRequest, DemoLoginRequest, TokenResponse, UserResponse
from services.auth_service import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication & Google OAuth"])

@router.post("/google", response_model=TokenResponse)
async def login_with_google(payload: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    """
    Direct endpoint corresponding to Client Browser <-> Google OAuth <-> FastAPI Gateway.
    Verifies Google ID token and returns session JWT.
    """
    google_user = auth_service.verify_google_oauth_token(payload.id_token)
    
    # Check if user exists in PostgreSQL / DB
    result = await db.execute(select(User).where(User.email == google_user["email"]))
    user = result.scalars().first()

    if not user:
        user = User(
            email=google_user["email"],
            name=google_user.get("name", "Mirage Creator"),
            avatar_url=google_user.get("picture"),
            google_id=google_user.get("sub")
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    access_token = auth_service.create_access_token(data={"sub": user.email, "user_id": user.id})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.post("/demo", response_model=TokenResponse)
async def demo_login(payload: DemoLoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Convenience login for testing the application seamlessly without setting up Google Cloud Console credentials.
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalars().first()

    if not user:
        user = User(
            email=payload.email,
            name=payload.name or "Julian Vance",
            avatar_url="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=200&q=80",
            google_id="demo-google-oauth-id-01"
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    access_token = auth_service.create_access_token(data={"sub": user.email, "user_id": user.id})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user(token: str, db: AsyncSession = Depends(get_db)):
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    result = await db.execute(select(User).where(User.email == payload.get("sub")))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
