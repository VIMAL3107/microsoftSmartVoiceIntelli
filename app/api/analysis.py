
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date, extract, desc
import shutil
import os
import tempfile
import time
import json
import uuid
from typing import Optional, List
from datetime import datetime, date

from app.core.database import get_db
from app.models.analytics import CallAnalytics
from app.api.deps import check_active_session
from app.services.audio_service import convert_to_wav_any, split_wav, translate_audio_autodetect, get_audio_duration
from app.services.llm_service import llm_call_qa_fields
from app.services.excel_service import export_call_analytics_to_excel
from app.services.pdf_service import generate_call_pdf_report
from app.schemas.analytics import CallAnalyticsRequest
from app.services.connection_manager import manager

import logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/progress/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep open
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@router.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    user_id: str = Query(...),
    db: Session = Depends(get_db),
    username: str = Depends(check_active_session)
):
    all_segments: list = []
    all_src: list = []
    all_en: list = []
    detected_lang: Optional[str] = None
    
    # Save upload    
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.filename or "")[1] or ".bin", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        src_path = tmp.name
    
    wav_path = None
    try:
        wav_path = convert_to_wav_any(src_path)
        # Split WAV into chunks
        raw_chunks = split_wav(wav_path, chunk_sec=60)
        chunks = []
        for chunk_path in raw_chunks:
            chunk_16k = convert_to_wav_any(chunk_path)
            chunks.append(chunk_16k)        
        
        t0 = time.time()
        for c in chunks:
            try:
                tr = translate_audio_autodetect(c)
                all_segments.extend(tr.get("segments", []))
                all_src.append(tr.get("recognized_text", ""))
                all_en.append(tr.get("translation_en", ""))
                if not detected_lang:
                    detected_lang = tr.get("detected_language")
            except HTTPException as e:
                if e.status_code == 422:
                    logger.warning("Chunk %s had no recognizble speech", c)
                    continue
                else:
                    raise e
        wall_translate = time.time() - t0
        
        # Cleanup chunks
        for c in chunks:
            try:
                os.remove(c)
            except Exception:
                pass
        
        recognized_text = " ".join(all_src).strip()
        translation_en = " ".join(all_en).strip()
        analysis_text = translation_en or recognized_text
        
        qa = llm_call_qa_fields(analysis_text)
        
        result = {
            "user_id": user_id,
            "time_taken_sec": round(wall_translate, 3),
            "detected_language": detected_lang or "",
            "recognized_text": recognized_text,
            "translation_en": translation_en,
            "segments": all_segments,
            "caller": qa.get("caller", ""),
            "callee": qa.get("callee", ""),
            "audio_quality": qa.get("audio_quality", "Good"),
            "word_count": len(recognized_text.split()),
            "audio_duration": get_audio_duration(wav_path),
            
            "conversation_feel": qa.get("conversation_feel", "Neutral"),
            "agent_improvement_areas": qa.get("agent_improvement_areas", []),
            "Agent_performance_summary": qa.get("Agent_performance_summary", ""),
            "ScriptAdherenceScore": qa.get("ScriptAdherenceScore", 3),
            "PolitenessProfessionalismScore": qa.get("PolitenessProfessionalismScore", 3),
            "ResolutionEffectivenessScore": qa.get("ResolutionEffectivenessScore", 3),
            "CsatPrediction": qa.get("CsatPrediction", 3.0),
            "CallDisposition": qa.get("CallDisposition", ""),
            "FollowUpRequired": qa.get("FollowUpRequired", False),
            "CrossSellUpsellAttempts": qa.get("CrossSellUpsellAttempts", False),
            "CrossSellUpsellDetails": qa.get("CrossSellUpsellDetails", "")
        }
        result["username"] = username

        # Save to DB
        try:
            db_record = CallAnalytics(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                detected_language=result.get("detected_language"),
                recognized_text=result.get("recognized_text"),
                translation_en=result.get("translation_en"),
                segments=result.get("segments", []),
                caller=result.get("caller"),
                callee=result.get("callee"),
                audio_quality=result.get("audio_quality"),
                word_count=result.get("word_count"),
                audio_duration=result.get("audio_duration"),
                conversation_feel=result.get("conversation_feel"),
                agent_improvement_areas=json.dumps(result.get("agent_improvement_areas", [])),
                Agent_performance_summary=result.get("Agent_performance_summary"),
                ScriptAdherenceScore=result.get("ScriptAdherenceScore"),
                PolitenessProfessionalismScore=result.get("PolitenessProfessionalismScore"),
                ResolutionEffectivenessScore=result.get("ResolutionEffectivenessScore"),
                CsatPrediction=result.get("CsatPrediction"),
                CallDisposition=result.get("CallDisposition"),
                FollowUpRequired=result.get("FollowUpRequired"),
                CrossSellUpsellAttempts=result.get("CrossSellUpsellAttempts"),
                CrossSellUpsellDetails=result.get("CrossSellUpsellDetails"),
                time_taken_sec=result.get("time_taken_sec")
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
            result["db_id"] = db_record.id
            result["session_id"] = db_record.session_id
            
            # Auto Export
            export_call_analytics_to_excel(db)
            
        except Exception as e:
            logger.exception("Failed to save DB: %s", e)
            db.rollback()
            
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        for p in (src_path, wav_path):
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass

# Duplicate dashboard/call/download endpoints removed. 
# They are implemented in main.py to verify full logic.
