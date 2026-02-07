
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date, extract, desc
from pathlib import Path
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
import logging
import fitz
import textwrap
from fpdf import FPDF
from app.models.analytics import PdfSummary
from app.services.llm_service import openai_client
from app.core.config import AOAI_MODEL, config
from pydantic import BaseModel, EmailStr
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

# ---------------- DASHBOARD ----------------
@router.get("/dashboard/")
def get_dashboard(
    user_id: str = Query(...),  # Require user_id in query string
    db: Session = Depends(get_db),
    username: str = Depends(check_active_session)
):
    now = datetime.utcnow()
    today = date.today()
    # Basic stats
    num_calls = db.query(func.count(CallAnalytics.id)).filter(CallAnalytics.user_id == user_id).scalar() or 0
    avg_time = db.query(func.avg(CallAnalytics.time_taken_sec)).filter(CallAnalytics.user_id == user_id).scalar() or 0
    # Conversation feel distribution
    conversation_status_counts = db.query(
        CallAnalytics.conversation_feel, func.count(CallAnalytics.id)
    ).filter(
        CallAnalytics.user_id == user_id,
        CallAnalytics.conversation_feel.in_(["Positive", "Negative", "Neutral"])
    ).group_by(CallAnalytics.conversation_feel).all()
    # Follow-up required
    follow_up_required_count = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        CallAnalytics.FollowUpRequired == True
    ).scalar() or 0
    # Cross-sell attempts
    cross_sell_attempts = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        CallAnalytics.CrossSellUpsellAttempts == True
    ).scalar() or 0
    # Upload counts
    today_count = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        cast(CallAnalytics.created_at, Date) == today
    ).scalar() or 0
    month_count = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        extract("year", CallAnalytics.created_at) == now.year,
        extract("month", CallAnalytics.created_at) == now.month
    ).scalar() or 0
    year_count = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        extract("year", CallAnalytics.created_at) == now.year
    ).scalar() or 0
    # CSAT distribution
    total_csat_count = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        CallAnalytics.CsatPrediction.isnot(None)
    ).scalar() or 0
    csat_below_2_5 = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        CallAnalytics.CsatPrediction < 2.5
    ).scalar() or 0
    csat_2_5_to_4 = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        CallAnalytics.CsatPrediction >= 2.5,
        CallAnalytics.CsatPrediction < 4
    ).scalar() or 0
    csat_4_to_5 = db.query(func.count(CallAnalytics.id)).filter(
        CallAnalytics.user_id == user_id,
        CallAnalytics.CsatPrediction >= 4
    ).scalar() or 0
    # Document listing
    doc_list = db.query(
        CallAnalytics.id,
        CallAnalytics.created_at,
        CallAnalytics.time_taken_sec,
        CallAnalytics.conversation_feel,
        CallAnalytics.session_id  # Add session_id to doc_list
    ).filter(CallAnalytics.user_id == user_id).all()
    docs = [
        {
            "id": id,
            "session_id": session_id,  # Include session_id in response
            "created_at": created_at.isoformat() if created_at else None,
            "time_taken_sec": time_taken_sec,
            "sentiment": conversation_feel
        }
        for id, created_at, time_taken_sec, conversation_feel, session_id in doc_list
    ]
    # Conversation feel dict
    conversation_status_dict = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for feel, count in conversation_status_counts:
        conversation_status_dict[feel] = count
    result = {
        "num_calls": num_calls,
        "avg_time_taken": avg_time,
        "conversation_status_counts": conversation_status_dict,
        "follow_up_required_count": follow_up_required_count,
        "upload_counts": {"today": today_count, "this_month": month_count, "this_year": year_count},
        "doc_list": docs,
        "cross_sell_attempts": cross_sell_attempts,
        "csat_distribution_percent": {
            "below_2_5": round((csat_below_2_5 / total_csat_count) * 100, 2) if total_csat_count else 0,
            "from_2_5_to_4": round((csat_2_5_to_4 / total_csat_count) * 100, 2) if total_csat_count else 0,
            "from_4_to_5": round((csat_4_to_5 / total_csat_count) * 100, 2) if total_csat_count else 0,
            "total": total_csat_count
        }
    }
    result["username"] = username
    return JSONResponse(content=result)


# ---------------- CALL DETAILS ----------------
class CallAnalyticsRequest(BaseModel):
    user_id: str
    page: int = 1
    page_size: int = 10
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None

@router.post("/callDetails/")
def get_user_calls(
    call_request: CallAnalyticsRequest,
    db: Session = Depends(get_db),
    username: str = Depends(check_active_session)
):
    user_id = call_request.user_id
    page = max(call_request.page, 1)
    page_size = max(min(call_request.page_size, 100), 1)
    offset = (page - 1) * page_size
    filter_by = call_request.filter_by
    sort_by = call_request.sort_by

    query = db.query(CallAnalytics).filter(CallAnalytics.user_id == user_id)
    order_column = CallAnalytics.created_at.desc()

    # --- Filtering ---
    if filter_by:
        try:
            key, value = filter_by.split(":", 1)
            if key == "conversation_feel":
                query = query.filter(CallAnalytics.conversation_feel == value)
            elif key == "follow_up_required":
                query = query.filter(CallAnalytics.FollowUpRequired == (value.lower() == "true"))
            elif key == "call_disposition":
                query = query.filter(CallAnalytics.CallDisposition == value)
            elif key == "created_at":
                filter_date_obj = datetime.strptime(value, "%Y-%m-%d").date()
                query = query.filter(cast(CallAnalytics.created_at, Date) == filter_date_obj)
            elif key == "csat_range":
                if value == "below_2_5":
                    query = query.filter(CallAnalytics.CsatPrediction < 2.5)
                elif value == "2_5_to_4":
                    query = query.filter(CallAnalytics.CsatPrediction >= 2.5,
                                         CallAnalytics.CsatPrediction < 4)
                elif value == "4_to_5":
                    query = query.filter(CallAnalytics.CsatPrediction >= 4)
        except ValueError:
            pass

    # --- Sorting ---
    if sort_by:
        try:
            field, direction = sort_by.split(":")
            column_attr = getattr(CallAnalytics, field, None)
            if column_attr is not None:
                order_column = column_attr.asc() if direction.lower() == "asc" else column_attr.desc()
        except ValueError:
            pass

    total_calls = query.count()
    call_data = query.order_by(order_column).offset(offset).limit(page_size).all()

    calls = [
        {
            "id": call.id,
            "session_id": call.session_id,
            "recognized_text": call.recognized_text,
            "translation_en": call.translation_en,
            "conversation_feel": call.conversation_feel,
            "caller": call.caller,
            "callee": call.callee,
            "audio_quality": call.audio_quality,
            "word_count": call.word_count,
            "audio_duration": call.audio_duration,
            "agent_improvement_areas": call.agent_improvement_areas,
            "Agent_performance_summary": call.Agent_performance_summary,
            "ScriptAdherenceScore": call.ScriptAdherenceScore,
            "PolitenessProfessionalismScore": call.PolitenessProfessionalismScore,
            "ResolutionEffectivenessScore": call.ResolutionEffectivenessScore,
            "CsatPrediction": call.CsatPrediction,
            "CallDisposition": call.CallDisposition,
            "FollowUpRequired": call.FollowUpRequired,
            "CrossSellUpsellAttempts": call.CrossSellUpsellAttempts,
            "CrossSellUpsellDetails": call.CrossSellUpsellDetails,
            "time_taken_sec": call.time_taken_sec,
            "created_at": call.created_at
        }
        for call in call_data
    ]

    # --- Include username in the response ---
    response_data = {
        "total_records": total_calls,
        "page": page,
        "page_size": page_size,
        "calls": calls,
        "username": username
    }

    return response_data


def sanitize_for_pdf(text: str) -> str:
    if not text:
        return ""
    # Map common characters that aren't in Latin-1
    replacements = {
        '\u20b9': 'Rs. ', # Rupee
        '\u2013': '-',    # En dash
        '\u2014': '--',   # Em dash
        '\u2018': "'",    # Left single quote
        '\u2019': "'",    # Right single quote
        '\u201c': '"',    # Left double quote
        '\u201d': '"',    # Right double quote
        '\u2026': '...',  # Ellipsis
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
        
    # Finally, encode to latin-1 ignoring/replacing remaining errors
    return text.encode('latin-1', 'replace').decode('latin-1')

@router.get("/download/")
def download_pdf(
    session_id: str = Query(..., description="Session ID of the call to download"),
    user_id: Optional[str] = Query(None, description="User ID requesting the download"),
    db: Session = Depends(get_db),
    username: str = Depends(check_active_session)
):
    # Query the record
    query = db.query(CallAnalytics).filter(CallAnalytics.session_id == session_id)
    if user_id:
        query = query.filter(CallAnalytics.user_id == user_id)
    
    call_record = query.first()
    if not call_record:
        raise HTTPException(
            status_code=404,
            detail="Session not found or you don't have permission to access this session"
        )
    
    # Create the PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize_for_pdf(f"Call Report - Session ID: {session_id}"), ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Translate:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, sanitize_for_pdf(call_record.translation_en or call_record.recognized_text or "No translation available."))
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Agent Performance Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, sanitize_for_pdf(call_record.Agent_performance_summary or "No feedback available."))

    # Save to a temporary directory
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, f"{session_id}_report.pdf")
    pdf.output(pdf_path)

    # Ensure file exists
    if not Path(pdf_path).exists():
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

    # Prepare safe headers
    headers = {
        "Content-Disposition": f"attachment; filename={session_id}_report.pdf"
    }
    if username:
        headers["username"] = str(username).encode('latin-1', 'ignore').decode('latin-1') # Ensure safe header

    # Return the PDF file response
    return FileResponse(
        path=pdf_path,
        filename=f"{session_id}_report.pdf",
        media_type="application/pdf",
        headers=headers
    )


# --- PDF UPLOAD HELPERS ---
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error("Error extracting text: %s", str(e))
        return ""

def PDF_Summarization(text: str, model: str = None) -> str:
    """
    Summarize given PDF text content using Azure OpenAI model from config.
    """
    logger.info("[Summary] Sending content to Azure OpenAI for summarization")

    # Use model from config if not provided
    if model is None:
        model = config.get("AZURE_OPENAI_MODEL", "o4-mini")

    prompt = (
        "Please summarize the key points and important information from the following PDF content. "
        "Focus on the main ideas, significant details, and any conclusions or outcomes mentioned:\n\n"
        f"{text.strip()}"
    )

    try:
        req = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.2,
        }

        # Adjust temperature only if model is not O-series
        if not model.lower().startswith("o4"):
            try:
                req["temperature"] = float(config.get("AZURE_OPENAI_TEMPERATURE", 0.2))
            except Exception:
                pass

        t0 = time.time()
        resp = openai_client.chat.completions.create(**req)
        logger.info(
            "Azure OpenAI completion done in %d ms (model=%s)",
            int((time.time() - t0) * 1000),
            model,
        )

        summary = resp.choices[0].message.content.strip()
        logger.info("[Summary] Received summary from Azure OpenAI")
        return summary

    except Exception as e:
        logger.warning("[Summary] Failed to summarize with Azure OpenAI: %s", str(e))
        return text  # fallback if summarization fails


def split_text_into_chunks(text: str, max_length: int = 300):
    return textwrap.wrap(text, width=max_length, break_long_words=False, replace_whitespace=False)
    
# API endpoint
@router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...),
                     isProspectus : bool = Form(...),
                     db: Session = Depends(get_db)):
    try:
        temp_dir = tempfile.mkdtemp()
        file_name = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(temp_dir, file_name)

        with open(file_path, "wb") as f_out:
            f_out.write(await file.read())

        text = extract_text_from_pdf(file_path)
        if not text:
            return JSONResponse(status_code=500, content={"error": "Failed to extract text from PDF"})

        summary = PDF_Summarization(text=text)
        
        # NOTE: Vector store logic (FAISS) removed/commented as dependency missing in env
        # if isProspectus :
        #     chunks = split_text_into_chunks(summary, max_length=350)
        #     # embeddings = embed_chunks(chunks)
        #     # index = build_faiss_index(embeddings)
        #     # faiss.write_index(index, "faiss_index.index")

        # Save to SQLite
        new_entry = PdfSummary(
                id=str(uuid.uuid4()),
                file_path=file_path,
                original_text=text,
                summary=summary,
                created_date=datetime.utcnow()
            )

        db.add(new_entry)
        db.commit()

        return {
            "file_path": file_path,
            "created_date": new_entry.created_date.isoformat(),
            "text": text,
            "summary": summary
        }

    except Exception as e:
        logger.error("Failed to process PDF upload: %s", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(e)})
