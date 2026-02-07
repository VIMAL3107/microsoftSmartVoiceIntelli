
from sqlalchemy import Column, String, Text, DateTime
from datetime import datetime
import uuid
from app.core.database import Base

class Email(Base):
    __tablename__ = "emails"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    to_email = Column(String(100), nullable=False)
    subject = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="sent")
    user_id = Column(String(36), index=True)
    session_id = Column(String(36), index=True, nullable=True)
