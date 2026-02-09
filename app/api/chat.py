
from fastapi import APIRouter, Depends, Form, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from collections import defaultdict
import json
import logging

from app.core.database import get_db
from app.models.analytics import CallAnalytics, CallSession
from app.services.llm_service import openai_client
from app.core.config import AOAI_MODEL

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session data for chat history (ephemeral)
SESSION_DATA = defaultdict(list)

@router.post("/chat/")
async def chat(
    session_id: str = Form(...),
    message: str = Form(...),
    isCallcenterChat: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    SESSION_DATA[session_id].append({
        "type": "user",
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Normalize value for easier comparison
    is_callcenter = (
        isCallcenterChat is None or
        str(isCallcenterChat).strip().lower() in ["", "yes", "true", "1"]
    )

    transcript = ""
    # Try to find transcript from CallAnalytics first (more reliable source of truth for post-call)
    call_analytics = db.query(CallAnalytics).filter(
        CallAnalytics.session_id == session_id
    ).first()
    
    if call_analytics:
        transcript = call_analytics.translation_en or call_analytics.recognized_text or ""
    
    # Fallback to CallSession if analytics not found
    if not transcript:
        call_session = db.query(CallSession).filter(
            CallSession.session_id == session_id
        ).order_by(CallSession.created_at.desc()).first()
        # Note: CallSession model assumes transcript_raw exists or we just proceed empty if not found.
        if call_session and hasattr(call_session, 'transcript_raw'):
             transcript = call_session.transcript_raw

    if is_callcenter:
        # Use call center prompt
        if transcript:
            prompt = (
                "You are a very strict call center conversation analyzer. "
                "Given the following call transcript and the user's question, analyze the conversation, agent performance, and provide insights or answer the question. "
                "If the user's question is not relevant to the call, answer as best you can.\n\n"
                f"--- Call Transcript ---\n{transcript}\n"
                f"--- User Question ---\n{message}\n"
                "Respond in this exact JSON format:\n"
                "{\n"
                '  "Assistant": "...",\n'
                '  "RelatedQuestions": ["...","...","..."]\n'
                "}\n"
            )
        else:
             prompt = (
                "You are a helpful assistant. "
                "Have a normal conversation with the user and answer their question or respond appropriately.\n\n"
                f"User: {message}\n"
                "Respond in this exact JSON format:\n"
                "{\n"
                '  "Assistant": "...",\n'
                '  "RelatedQuestions": ["...","...","..."]\n'
                "}\n"
            )
    else:
        # Use generic assistant prompt
        if transcript:
          prompt = (
            "You are a helpful assistant. "
            "Have a normal conversation with the user and answer their question or respond appropriately.\n\n"
            f"--- Call Transcript ---\n{transcript}\n"
            f"User: {message}\n"
            "Respond in this exact JSON format:\n"
            "{\n"
            '  "Assistant": "...",\n'
            '  "RelatedQuestions": ["...","...","..."]\n'
            "}\n"
          )
        else:
            prompt = (
                "You are a helpful assistant. "
                "Have a normal conversation with the user and answer their question or respond appropriately.\n\n"
                f"User: {message}\n"
                "Respond in this exact JSON format:\n"
                "{\n"
                '  "Assistant": "...",\n'
                '  "RelatedQuestions": ["...","...","..."]\n'
                "}\n"
            )

    bot_reply = ""
    related_questions = []

    try:
        if not openai_client:
            raise Exception("Azure OpenAI client not initialized")

        # call Azure OpenAI
        req = {
            "model": AOAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            # If using O-series models, temperature might be restricted
            # "temperature": 0.7, 
            "response_format": { "type": "json_object" }
        }
        
        # Adjust temperature checks based on model name if needed
        if not AOAI_MODEL.lower().startswith("o4"):
             req["temperature"] = 0.7

        response = openai_client.chat.completions.create(**req)
        response_content = response.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            parsed = json.loads(response_content)
            bot_reply = parsed.get("Assistant", "")
            related_questions = parsed.get("RelatedQuestions", [])
        except json.JSONDecodeError:
            # Fallback if model didn't output valid JSON
            bot_reply = response_content
            related_questions = []

    except Exception as e:
        logger.error(f"Chat API failed: {e}")
        bot_reply = "I'm sorry, I encountered an error processing your request."
        related_questions = []

    SESSION_DATA[session_id].append({
        "type": "bot",
        "message": bot_reply,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {
        "session_id": session_id,
        "bot_reply": bot_reply,
        "releated_questions": related_questions
    }
