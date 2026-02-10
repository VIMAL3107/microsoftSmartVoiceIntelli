
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, Text, DateTime
from datetime import datetime
import uuid
from app.core.database import Base

class CallAnalytics(Base):
    __tablename__ = "call_analytics"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), index=True, unique=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), index=True)
    AgentName = Column(String(100))
    CustomerName = Column(String(100))
    conversation_feel = Column(String(20))
    detected_language = Column(String(20), nullable=True)
    sentiment = Column(String(20))
    recognized_text = Column(Text, nullable=True)
    segments = Column(JSON, default=[])
    reason_for_sentiment_result = Column(Text)
    translation_en = Column(Text, nullable=True)
    Agent_performance_summary = Column(Text)
    agent_improvement_areas = Column(Text)
    ScriptAdherenceScore = Column(Integer)
    PolitenessProfessionalismScore = Column(Integer)
    ResolutionEffectivenessScore = Column(Integer)
    CsatPrediction = Column(Float)
    CallDisposition = Column(String(100))
    FollowUpRequired = Column(Boolean, default=False)
    CrossSellUpsellAttempts = Column(Boolean, default=False)
    CrossSellUpsellDetails = Column(Text)
    time_taken_sec = Column(Float)  
    caller = Column(String(255))
    callee = Column(String(255))
    audio_quality = Column(String(50))
    word_count = Column(Integer, default=0)
    audio_duration = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    file_name = Column(String(255))
    feedback = Column(Text)
    is_reported = Column(Boolean, default=False)

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), index=True)
    user_feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class CallSession(Base):
    __tablename__ = "call_sessions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), index=True, unique=True)
    user_id = Column(String(36), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PdfSummary(Base):
    __tablename__ = "pdf_summary"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String(255), nullable=False)
    original_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    created_date = Column(DateTime, default=datetime.utcnow)
