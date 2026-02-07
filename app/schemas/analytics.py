
from pydantic import BaseModel
from typing import Optional

class CallAnalyticsRequest(BaseModel):
    user_id: str
    page: int = 1
    page_size: int = 10
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None

class UserFeedbackCreate(BaseModel):
    session_id: str
    user_feedback: str
