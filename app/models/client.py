
from sqlalchemy import Column, String, Text, DateTime
from datetime import datetime
import uuid
from app.core.database import Base

class Client(Base):
    __tablename__ = "clients"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    allowed_ip = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PdfSummary(Base):
    __tablename__ = "pdf_summary"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String(255), nullable=False)
    original_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    created_date = Column(DateTime, default=datetime.utcnow)
