
from pydantic import BaseModel, EmailStr
from typing import Optional

class EmailRequest(BaseModel):
    to_email: EmailStr
    subject: str
    content: str
    session_id: Optional[str] = None
