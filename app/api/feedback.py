
from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.models.analytics import UserFeedback, CallSession

router = APIRouter()

@router.post("/create-user-feedback")
def submit_user_feedback(
    session_id: str = Form(...),
    user_feedback: str = Form(...),
    db: Session = Depends(get_db)
):
    # Optional: Verify session exists
    # if not db.query(CallSession).filter(CallSession.session_id == session_id).first():
    #     raise HTTPException(status_code=404, detail="Session not found")

    feedback_entry = UserFeedback(
        session_id=session_id,
        user_feedback=user_feedback,
        created_at = datetime.utcnow(),
    )
    db.add(feedback_entry)
    db.commit()

    return {"message": "Feedback saved successfully."}

@router.get("/get-user-feedback")
def get_user_feedback(
    session_id: str,
    db: Session = Depends(get_db)
):
    feedbacks = db.query(UserFeedback).filter(UserFeedback.session_id == session_id).all()
    if not feedbacks:
        raise HTTPException(status_code=404, detail="No feedback found for this session")

    return feedbacks
