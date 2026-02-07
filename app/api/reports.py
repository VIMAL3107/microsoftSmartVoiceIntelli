
from fastapi import APIRouter, HTTPException, Depends, Query, Form
from sqlalchemy.orm import Session
from datetime import date, datetime
from typing import Optional
from pydantic import EmailStr
import json
import logging

from app.core.database import get_db
from app.models.analytics import CallAnalytics
from app.services.excel_service import export_call_analytics_to_excel, export_and_mark_unreported_analytics
from app.services.email_service import send_report_email_with_attachment
from app.services.security import LICENSE_DIR

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/admin/extend-license")
def extend_license(new_expiry: str, secret: str = Form(...)):
    if secret != "INTELLICORE-SECRET":
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        with open("license.json", "r") as f:
            data = json.load(f)
        data["expires"] = new_expiry
        with open("license.json", "w") as f:
            json.dump(data, f)
        return {"message": f"License extended to {new_expiry}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="license.json not found")



