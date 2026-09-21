from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models import ContactInquiry
from schemas import ContactFormRequest, ContactFormResponse

router = APIRouter(prefix="/contact", tags=["Contact Inquiries"])

@router.post("", response_model=ContactFormResponse, status_code=status.HTTP_201_CREATED)
async def submit_contact_form(form: ContactFormRequest, db: AsyncSession = Depends(get_db)):
    """
    Saves new contact and project collaboration inquiries into the database.
    """
    inquiry = ContactInquiry(
        name=form.name,
        email=form.email,
        subject=form.subject,
        message=form.message,
        status="received"
    )
    db.add(inquiry)
    await db.commit()
    await db.refresh(inquiry)

    return {
        "success": True,
        "message": f"Thank you, {form.name}! Your message regarding '{form.subject}' has been received by our architectural engineering team.",
        "inquiry_id": inquiry.id
    }
