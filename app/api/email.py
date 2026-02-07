
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.email import Email
from app.schemas.email import EmailRequest
from app.core.config import EMAIL_FROM, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT
import smtplib
from email.mime.text import MIMEText
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/send-email/")
async def send_email(
    email_data: EmailRequest,
    user_id: str = Query(...),
    db: Session = Depends(get_db),
):
    try:
        msg = MIMEText(email_data.content)
        msg["Subject"] = email_data.subject
        msg["From"] = EMAIL_FROM
        msg["To"] = email_data.to_email

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, email_data.to_email, msg.as_string())

        email_record = Email(
            to_email=email_data.to_email,
            subject=email_data.subject,
            content=email_data.content,
            status="sent",
            user_id=user_id,
            session_id=email_data.session_id
        )
        db.add(email_record)
        db.commit()

        return {"status": "success", "message": "Email sent", "email_id": email_record.id}
        
    except Exception as e:
        email_record = Email(
            to_email=email_data.to_email,
            subject=email_data.subject,
            content=email_data.content,
            status="failed",
            user_id=user_id,
            session_id=email_data.session_id
        )
        db.add(email_record)
        db.commit()
        
        logger.error(f"Failed to send email: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-emails/")
def get_emails(
    user_id: str,
    db: Session = Depends(get_db),
    limit: int = 100,
    offset: int = 0
):
    return db.query(Email).filter(Email.user_id == user_id).order_by(Email.sent_at.desc()).offset(offset).limit(limit).all()

@router.get("/get-email/{email_id}/")
def get_email_details(
    email_id: str,
    user_id: str,
    db: Session = Depends(get_db),
):
    email = db.query(Email).filter(Email.id == email_id, Email.user_id == user_id).first()
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    return email
