# microsoftSmartVoiceTranslate_autodetect_llm.py
# -*- coding: utf-8 -*-

import os, re, json, time, wave, shutil, tempfile, datetime, subprocess, logging, sys, contextvars, traceback, uuid
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form, Depends, HTTPException, status, Query
from app.services.excel_service import export_call_analytics_to_excel
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import concurrent.futures
import threading
from fpdf import FPDF
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON, Text, DateTime, extract, func, ForeignKey, Date, cast
from datetime import datetime, timedelta, timezone, date
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import declarative_base, sessionmaker, Session as OrmSession
from collections import defaultdict
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
import fitz  # PyMuPDF
# from sentence_transformers import SentenceTransformer (Moved to lazy import below)
# import faiss (Missing in environment)
import textwrap
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import requests
from cryptography.fernet import Fernet

load_dotenv()

# FERNET_KEY for decrypting config.enc
FERNET_KEY = os.getenv("FERNET_KEY")
if isinstance(FERNET_KEY, str):
    FERNET_KEY = FERNET_KEY.encode()
fernet = Fernet(FERNET_KEY)

def load_config(enc_file_path: str = "config.enc") -> dict:
    """Load and decrypt configuration from encrypted file."""
    if not os.path.exists(enc_file_path):
        return {}
    with open(enc_file_path, "rb") as f:
        encrypted_bytes = f.read()
    decrypted_bytes = fernet.decrypt(encrypted_bytes)
    config = json.loads(decrypted_bytes.decode("utf-8"))
    return config

# Logging
# ======================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
request_id_var = contextvars.ContextVar("request_id", default="-")
class ReqIdFilter(logging.Filter):
    def filter(self, record): record.request_id = request_id_var.get("-"); return True
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
)
for h in logging.getLogger().handlers:
    h.addFilter(ReqIdFilter())
logger = logging.getLogger("speech-autodetect-llm")

# Load configuration from encrypted file
config = load_config("config.enc")

# AZURE Config
# ======================
SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY", config.get("AZURE_SPEECH_KEY", "")).strip()
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", config.get("AZURE_SPEECH_REGION", "")).strip()
TARGET_LANG   = os.getenv("TARGET_LANG", config.get("TARGET_LANG", "en")).strip()

AUTODETECT_LANGS_RAW = os.getenv("AUTODETECT_LANGS", config.get("AUTODETECT_LANGS", "en-US,ta-IN,hi-IN,te-IN,kn-IN")).strip()
AUTODETECT_LANGS = [l.strip() for l in AUTODETECT_LANGS_RAW.split(",") if l.strip()][:4]   # Azure Limit: 4 languages
if not (SPEECH_KEY and SPEECH_REGION):
    raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")
AOAI_KEY      = os.getenv("AZURE_OPENAI_KEY", config.get("AZURE_OPENAI_KEY", "")).strip()
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", config.get("AZURE_OPENAI_ENDPOINT", "")).strip().rstrip('/')
AOAI_MODEL    = os.getenv("AZURE_OPENAI_MODEL", config.get("AZURE_OPENAI_MODEL", "")).strip() # e.g., "o4-mini"
AOAI_API_VER  = config.get("AZURE_OPENAI_API_VERSION", "2024-12-01-previe")
if not all([AOAI_KEY, AOAI_ENDPOINT, AOAI_MODEL]):
    raise RuntimeError("Missing Azure OpenAI envs (AZURE_OPENAI_KEY/ENDPOINT/MODEL)")
openai_client = AzureOpenAI(api_key=AOAI_KEY, azure_endpoint=AOAI_ENDPOINT, api_version=AOAI_API_VER)
logger.info("Config: region=%s target=%s autodetect=%s model=%s",
            SPEECH_REGION, TARGET_LANG, AUTODETECT_LANGS, AOAI_MODEL)

# --- Password hashing ---
VALID_LICENSE_KEYS = config.get("VALID_LICENSE_KEYS", [])
# Ensure it's a list
if isinstance(VALID_LICENSE_KEYS, str):
    try:
        VALID_LICENSE_KEYS = json.loads(VALID_LICENSE_KEYS)
    except (json.JSONDecodeError, ValueError):
        VALID_LICENSE_KEYS = []

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def check_license():
    try:
        with open("license.json", "r") as f:
            data = json.load(f)

        # Check license key
        if data.get("license_key") not in VALID_LICENSE_KEYS:
            raise Exception("Invalid license key")

        # Check expiry
        expiry = datetime.strptime(data["expires"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expiry:
            raise Exception("License expired")

    except FileNotFoundError:
        print("[LICENSE WARNING] license.json not found - skipping license check")
        return
    except Exception as e:
        print(f"[LICENSE ERROR] {e}")
        sys.exit(1)

check_license()
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_progress(self, session_id: str, data: dict):
        ws = self.active_connections.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(session_id)

# Initialize the manager
manager = ConnectionManager()

# === EMAIL CONFIGURATION ===
EMAIL_FROM     = os.getenv("EMAIL_FROM", config.get("EMAIL_FROM"))
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", config.get("EMAIL_PASSWORD"))  # Use an App Password for Gmail!
SMTP_SERVER    = os.getenv("SMTP_SERVER", config.get("SMTP_SERVER", "smtp.gmail.com"))
SMTP_PORT      = int(os.getenv("SMTP_PORT", config.get("SMTP_PORT", 587)))
FRONTEND_URL   = os.getenv("FRONTEND_URL", config.get("FRONTEND_URL", "http://localhost:8080"))  # fallback default

# --- JWT and OAuth2 setup ---
SECRET_KEY                 = config.get("SECRET_KEY")
ALGORITHM                  = config.get("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = config.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60)

# ------------------ DATABASE SETUP ------------------
# DATABASE_URL = config.get("DATABASE_URL")
# if not DATABASE_URL:
#     raise ValueError("DATABASE_URL is not defined in config!")
# DATABASE_URL = DATABASE_URL.strip()

# --- TEMPORARY SQLITE OVERRIDE ---
DATABASE_URL = "sqlite:///./voice_translate.db"
logger.warning(f"Using SQLite Database: {DATABASE_URL}")

engine = create_engine(
    DATABASE_URL, 
    echo=True,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------ SQLALCHEMY MODELS ------------------

# User Model
class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, index=True, nullable=False)  # SQL Server-safe
    email = Column(String(100), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    is_email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255), nullable=True)
    token_expiry = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

# Login Model
class Login(Base):
    __tablename__ = 'logins'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), index=True, nullable=False)
    email = Column(String(100), index=True, nullable=False)
    source_ip = Column(String(50), nullable=True)
    login_date = Column(DateTime, default=datetime.utcnow)

# CallAnalytics Model
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

# UserFeedback Model
class UserFeedback(Base):
    __tablename__ = "user_feedback"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), index=True)
    user_feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# CallSession Model
class CallSession(Base):
    __tablename__ = "call_sessions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), index=True, unique=True)
    user_id = Column(String(36), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Client Model
class Client(Base):
    __tablename__ = "clients"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    allowed_ip = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# PdfSummary Model
class PdfSummary(Base):
    __tablename__ = "pdf_summary"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String(255), nullable=False)
    original_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    created_date = Column(DateTime, default=datetime.utcnow)

# Email Model
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

# ------------------ CREATE TABLES ------------------
Base.metadata.create_all(bind=engine)

# ------------------ Pydantic SCHEMAS ------------------

# Email request schema
class EmailRequest(BaseModel):
    to_email: EmailStr
    subject: str
    content: str
    session_id: Optional[str] = None

# User schemas
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str
    

class UserRead(BaseModel):
    id: str
    username: str
    email: EmailStr
    created_at: datetime

# --- Logging and FastAPI setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Enable CORS for all origins (allow frontend to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

SESSION_DATA = defaultdict(list)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.password):
        return None
    return user    

# ======================
# Audio helpers
# ======================
def _resolve_ffmpeg_bin() -> str:
    for c in [os.getenv("FFMPEG_BIN","").strip(), "/usr/bin/ffmpeg", "ffmpeg"]:
        if not c: continue
        try:
            subprocess.run([c, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return c
        except Exception:
            pass
    raise HTTPException(500, "FFmpeg not found. Install it or set FFMPEG_BIN.")
def _assert_wav_pcm16(wav_path: str) -> None:
    try:
        with wave.open(wav_path, "rb") as wf:
            nch = wf.getnchannels()
            sr  = wf.getframerate()
            sw  = wf.getsampwidth()
            if not (nch == 1 and sr == 16000 and sw == 2):
                raise HTTPException(400, f"Incompatible WAV (channels={nch}, sr={sr}, bytes={sw}); need mono/16kHz/16-bit PCM.")
            if wf.getnframes() <= 0:
                raise HTTPException(400, "Empty/zero-length audio after conversion.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid WAV header: {e}")
def convert_to_wav_any(input_path: str) -> str:
    ffmpeg = _resolve_ffmpeg_bin()
    out_wav = input_path + ".__16k_mono_pcm.wav"
    # Robust: don't force "-map 0:a:0"; let ffmpeg pick main audio; strip video/subs/data
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-nostdin",
        "-i", input_path,
        "-vn", "-sn", "-dn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        out_wav
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(out_wav):
        err = proc.stderr.decode(errors="ignore")[-800:]
        logger.error("FFmpeg failed rc=%s tail=%s", proc.returncode, err)
        # --- FALLBACK: If input is already wav, just copy it ---
        if input_path.lower().endswith(".wav"):
             logger.warning("FFmpeg failed, but input is WAV. Attempting to use directly.")
             shutil.copy2(input_path, out_wav)
             return out_wav
        raise HTTPException(400, f"Audio conversion failed: {err}")
    logger.info("FFmpeg convert ok (%.0f ms)", (time.time()-t0)*1000)
    # _assert_wav_pcm16(out_wav) # Skip assertion validation if we are bypassing
    return out_wav
def _wav_duration(path_wav: str) -> float:
    try:
        with wave.open(path_wav, "rb") as wf:
            rate = wf.getframerate() or 16000
            return wf.getnframes() / float(rate)
    except Exception:
        return 0.0
def get_audio_duration(path_wav: str) -> float:
    return _wav_duration(path_wav)

def split_wav(path_wav: str, chunk_sec: int = 60):
    """Split WAV file into smaller chunks."""
    chunks = []
    with wave.open(path_wav, "rb") as wf:
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames_per_chunk = int(chunk_sec * framerate)
        total_frames = wf.getnframes()
        for start in range(0, total_frames, frames_per_chunk):
            wf.setpos(start)
            frames = wf.readframes(min(frames_per_chunk, total_frames - start))
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp_wav.name, "wb") as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(framerate)
                out_wf.writeframes(frames)
            chunks.append(tmp_wav.name)
    return chunks

# ======================
# Speech Translation (Auto-detect, no diarization)
# ======================
def translate_audio_autodetect(path_wav: str) -> Dict[str, Any]:
    stc = speechsdk.translation.SpeechTranslationConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    stc.add_target_language(TARGET_LANG)
    # Relax timeouts for file input
    stc.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")
    stc.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "2000")
    try:
        stc.set_profanity(speechsdk.ProfanityOption.Raw)
    except Exception:
        pass
    langs = AUTODETECT_LANGS if AUTODETECT_LANGS else ["en-US"]
    auto_cfg = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(langs)
    audio_config = speechsdk.audio.AudioConfig(filename=path_wav)
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=stc,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_cfg
    )
    segments: List[Dict[str, str]] = []
    src_chunks: List[str] = []
    en_chunks: List[str] = []
    detected = None
    done = False
    saw_speech = False
    cancel_reason = ""
    cancel_kind = None  # will hold speechsdk.CancellationReason
    def on_recognizing(evt):
        nonlocal saw_speech
        if evt and getattr(evt, "result", None) and evt.result.text:
            saw_speech = True
    def on_recognized(evt):
        nonlocal detected, saw_speech
        res = evt.result
        if res.reason == speechsdk.ResultReason.TranslatedSpeech:
            saw_speech = True
            try: detected = res.language
            except Exception: pass
            src = (res.text or "").strip()
            en  = (res.translations.get(TARGET_LANG, "") or "").strip()
            if src: src_chunks.append(src)
            if en:  en_chunks.append(en)
            if src or en: segments.append({"text": src, "translation": en})
            logger.info("Translated chunk: src_len=%d en_len=%d", len(src), len(en))
        elif res.reason == speechsdk.ResultReason.RecognizedSpeech:
            t = (res.text or "").strip()
            if t:
                saw_speech = True
                src_chunks.append(t)
                segments.append({"text": t, "translation": ""})
        elif res.reason == speechsdk.ResultReason.NoMatch:
            logger.warning("NoMatch: speech not recognized.")
    def on_canceled(evt):
        nonlocal done, cancel_reason, cancel_kind
        cancel_kind = getattr(evt, "reason", None)
        details = getattr(evt, "error_details", "")
        cancel_reason = f"{cancel_kind} | {details}"
        done = True
        try:
            if cancel_kind == speechsdk.CancellationReason.EndOfStream:
                logger.info("Translation canceled with EndOfStream (normal for file input).")
            else:
                logger.warning("Translation canceled: %s", cancel_reason)
        except Exception:
            logger.warning("Translation canceled: %s", cancel_reason)
    def on_session_started(_): logger.info("Translation session started")
    def on_session_stopped(_):
        nonlocal done
        done = True
        logger.info("Translation session stopped")
    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.start_continuous_recognition_async().get()
    dur_s = max(0.0, _wav_duration(path_wav))
    cushion = max(5.0, min(15.0, dur_s * 0.25))
    deadline = time.time() + min(300.0, max(5.0, dur_s + cushion))
    while not done and time.time() < deadline:
        time.sleep(0.1)
    try:
        fut = recognizer.stop_continuous_recognition_async()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fut.get)  # call .get() without args
            future.result(timeout=10)  # apply timeout here
    except concurrent.futures.TimeoutError:
        logger.warning("Stop recognition timed out, forcing exit")
    except Exception as e:
        logger.warning("Failed to stop recognition cleanly: %s", e)
    
    # Only treat real errors as failures; EndOfStream is normal.
    if cancel_kind and cancel_kind != speechsdk.CancellationReason.EndOfStream:
        raise HTTPException(502, f"Speech translation canceled: {cancel_reason}")
    if not saw_speech and not (src_chunks or en_chunks):
        raise HTTPException(
            422,
            "No speech recognized: input may be silent, too short (<1s), too quiet, or missing an audio stream. "
            "Try a longer/clearer sample or re-encode to mono 16k PCM."
        )
    return {
        "recognized_text": " ".join(src_chunks).strip(),
        "translation_en":  " ".join(en_chunks).strip(),
        "segments": segments,
        "detected_language": detected or ""
    }
# ======================
# LLM QA JSON (same schema/fields)
# ======================
def llm_call_qa_fields(transcript_text: str) -> Dict[str, Any]:
    schema = {
      "type":"object",
      "properties":{
        "conversation_feel":{"type":"string","enum":["Positive","Neutral","Negative"]},
        "agent_improvement_areas":{"type":"array","items":{"type":"string"}},
        "Agent_performance_summary":{"type":"string"},
        "ScriptAdherenceScore":{"type":"integer","minimum":1,"maximum":5},
        "PolitenessProfessionalismScore":{"type":"integer","minimum":1,"maximum":5},
        "ResolutionEffectivenessScore":{"type":"integer","minimum":1,"maximum":5},
        "CsatPrediction":{"type":"number","minimum":1,"maximum":5},
        "CallDisposition":{"type":"string"},
        "FollowUpRequired":{"type":"boolean"},
        "CrossSellUpsellAttempts":{"type":"boolean"},
        "CrossSellUpsellDetails":{"type":"string"}
      },
      "required":["conversation_feel","agent_improvement_areas","Agent_performance_summary",
                  "ScriptAdherenceScore","PolitenessProfessionalismScore",
                  "ResolutionEffectivenessScore","CsatPrediction","CallDisposition",
                  "FollowUpRequired","CrossSellUpsellAttempts","CrossSellUpsellDetails"]
    }
    SYSTEM = (
        "You are a QA analyst for call centers. Score 1-5 (5 best).\n"
        "Definitions:\n"
        "- Script adherence: greeting, verification, resolution, closing.\n"
        "- Resolution effectiveness: whether issue addressed.\n"
        "- Politeness/professionalism: tone, respect, no off-topic remarks.\n"
        "Use only the provided transcript. If unknown, infer conservatively."
    )
    diarization_like = f"[00:00:00–00:00:00] Speaker 1: {transcript_text}"
    prompt = (
        f"Overall sentiment: Neutral. Reason: n/a.\n\n"
        f"Transcript:\n{diarization_like}\n\n"
        "Return strict JSON that matches the schema."
    )
    req = {
        "model": AOAI_MODEL,
        "messages": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":prompt}
        ],
        "response_format":{"type":"json_schema","json_schema":{"name":"callqa","schema":schema}},
    }
    if not AOAI_MODEL.lower().startswith("o4"):
        try:
            req["temperature"] = float(os.getenv("AOAI_TEMPERATURE","0.2"))
        except ValueError:
            pass
    t0 = time.time()
    resp = openai_client.chat.completions.create(**req)
    logger.info("LLM completion done in %d ms (model=%s)", int((time.time()-t0)*1000), AOAI_MODEL)
    msg = resp.choices[0].message
    if hasattr(msg, "parsed") and getattr(msg, "parsed"):
        return msg.parsed  # type: ignore[attr-defined]
    try:
        return json.loads(msg.content or "{}")
    except Exception as e:
        logger.error("Failed to parse LLM JSON: %s", e)
        return {}
# ======================
# FastAPI app
# ======================
app = FastAPI(title="Speech → English (AutoDetect) + LLM QA", version="1.0.0")
origins = [o.strip() for o in (os.getenv("CORS_ALLOW_ORIGINS","*") or "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    token = request_id_var.set(os.urandom(6).hex())
    start = time.time()
    try:
        return await call_next(request)
    except Exception as e:
        logger.error("Unhandled error: %s", e, exc_info=True)
        tb = traceback.format_exc(limit=8)
        return PlainTextResponse(status_code=500, content=f"Internal Server Error\n{tb}")
    finally:
        logger.info("REQ done in %d ms", int((time.time()-start)*1000))
        request_id_var.reset(token)

# --- OAuth2 token endpoint ---
@app.post("/token")
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: OrmSession = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.websocket("/ws/progress/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep the socket open
    except WebSocketDisconnect:
        manager.disconnect(session_id)


@app.post("/admin/extend-license")
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

# ----------- EMAIL SENDING FUNCTION -----------
from app.services.email_service import send_verification_email, send_report_email_with_attachment

@app.get("/debug/email")
def debug_email_connection():
    """Debug endpoint to test SMTP connection and Auth"""
    import socket
    results = {
        "SMTP_SERVER": SMTP_SERVER,
        "SMTP_PORT": SMTP_PORT,
        "EMAIL_FROM": EMAIL_FROM,
        "DNS_Resovled": False,
        "Connection_Success": False,
        "Auth_Success": False,
        "Error": None
    }
    
    try:
        # 1. DNS Resolution
        socket.gethostbyname(SMTP_SERVER)
        results["DNS_Resovled"] = True
        
        # 2. Connection
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.connect(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.ehlo()
        results["Connection_Success"] = True
        
        # 3. Auth
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        results["Auth_Success"] = True
        server.quit()
        
    except Exception as e:
        results["Error"] = str(e)
        
    return results

@app.post("/admin/daily-report")
def trigger_daily_report(
    to_email: EmailStr,
    target_date: Optional[date] = None,
    secret: str = Query(..., description="Admin Secret"),
    db: OrmSession = Depends(get_db)
):
    """
    Generates a daily report for the specified date (default: today) and emails it.
    """
    if secret != "INTELLICORE-SECRET":
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not target_date:
        target_date = datetime.utcnow().date()
        
    logger.info(f"Generating daily report for {target_date}")

    # 1. Summary Metrics
    # We cast created_at to date to compare
    total_calls = db.query(CallAnalytics).filter(func.date(CallAnalytics.created_at) == target_date).count()
    
    # Example: Average Sentiment (if you have numeric mapping, otherwise just count)
    # For now, just total calls is good enough for step 1.

    # 2. Export Excel
    filename = f"Daily_Report_{target_date}.xlsx"
    try:
        export_call_analytics_to_excel(db, output_file=filename, target_date=target_date)
    except Exception as e:
        logger.error(f"Excel export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Excel export failed: {e}")

    # 3. Send Email
    subject = f"Daily Call Analytics Report - {target_date}"
    body = f"""
    Hello,

    Here is the daily summary for {target_date}.

    -----------------------------------
    Total Calls Analyzed: {total_calls}
    -----------------------------------

    Please find the detailed Excel report attached.

    Best regards,
    Voice Analytics System
    """
    
    try:
        send_report_email_with_attachment(to_email, subject, body, filename)
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        raise HTTPException(status_code=500, detail=f"Email sending failed: {e}")

    return {
        "message": f"Daily report for {target_date} sent to {to_email}",
        "metrics": {"total_calls": total_calls},
        "file": filename
    }

# ----------- SCHEDULER -----------
def run_scheduler_loop():
    """Background loop to check schedule and send emails."""
    logger.info("SCHEDULER: Starting up (10s delay)...")
    time.sleep(10)  # Initial delay
    logger.info("SCHEDULER: Loop active.")

    while True:
        try:
            # 1. Try to load from Environment Variable (Higher Priority)
            env_schedule = os.getenv("REPORT_SCHEDULE")
            t_cfg = {}
            if env_schedule:
                try:
                    t_cfg = json.loads(env_schedule)
                except json.JSONDecodeError:
                    logger.error("SCHEDULER: Failed to parse REPORT_SCHEDULE from env vars.")

            # 2. Fallback to config.enc if not in env
            if not t_cfg:
                try:
                    current_cfg = load_config("config.enc")
                    t_cfg = current_cfg.get("REPORT_SCHEDULE", {})
                except Exception:
                    t_cfg = {}

            if t_cfg.get("enabled"):
                interval = float(t_cfg.get("interval_minutes", 0))
                to_email = t_cfg.get("email_to")

                # Interval Logic
                if interval > 0 and to_email:
                    logger.info(f"SCHEDULER: Task scheduled. Waiting {interval} minutes...")
                    time.sleep(interval * 60)

                    logger.info(f"SCHEDULER: Triggering Scheduled Report to {to_email}")

                    # Manual DB session
                    # Note: We need to use valid secret
                    db_gen = get_db()
                    db = next(db_gen) # Initialize generator
                    try:
                        # Since trigger_daily_report expects an email string and other args,
                        # we call it directly. Note: trigger_daily_report is an endpoint function
                        # but we can call the logic if we mock the dependencies or extract logic.
                        # Ideally, extract the logic. Calling endpoint function directly:
                        trigger_daily_report(
                            to_email=to_email,
                            target_date=datetime.utcnow().date(),
                            secret="INTELLICORE-SECRET",
                            db=db
                        )
                    except Exception as exc:
                        logger.error(f"SCHEDULER ERROR: {exc}")
                    finally:
                        # db.close() # handled by context manager if used, but here it's a generator
                        # The generator yield block has a finally clause which closes it.
                        # But since we manually advanced it with next(), we should close it.
                        try:
                            # db is the session object yielded
                            db.close()
                        except:
                            pass
                else:
                    time.sleep(60)
            else:
                # If disabled or invalid config, wait a bit before checking again
                time.sleep(60)
        except Exception as e:
            logger.error(f"SCHEDULER LOOP CRASHED: {e}")
            time.sleep(60)

        except Exception as e:
            logger.error(f"SCHEDULER CRASH: {e}")
            time.sleep(60)

@app.on_event("startup")
async def start_scheduler():
    t = threading.Thread(target=run_scheduler_loop, daemon=True)
    t.start()

# ----------- Registration API -----------
@app.post("/register/")
def register_user(user: UserCreate, db: OrmSession = Depends(get_db)):
    logger.info(f"Register attempt for username: {user.username}, email: {user.email}")
    existing = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if existing:
        logger.warning("Registration failed: Username or email already registered.")
        raise HTTPException(status_code=400, detail="Username or email already registered")
    hashed_pwd = hash_password(user.password)
    token = str(uuid.uuid4())
    expiry = datetime.utcnow() + timedelta(hours=24)
    new_user = User(
        username=user.username,
        email=user.email,
        password=hashed_pwd,
        is_email_verified=False,
        email_verification_token=token,
        token_expiry=expiry
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"User {user.username} registered successfully. Sending verification email.")
    try:
        send_verification_email(user.email, token)
    except Exception as e:
        logger.error(f"Error sending verification email: {e}")
        raise HTTPException(status_code=500, detail="Failed to send verification email. Please contact support.")
    return {
        "message": f"User {user.username} registered successfully. Please check mail for verification and activation."
    }
# ----------- Email Verification API -----------
@app.get("/verify-email", response_class=HTMLResponse)
def verify_email(token: str, db: OrmSession = Depends(get_db)):
    user = db.query(User).filter(User.email_verification_token == token).first()
    if not user or user.token_expiry < datetime.utcnow():
        return HTMLResponse(content="<h2>Invalid or expired token.</h2>", status_code=400)
    user.is_email_verified = True
    user.email_verification_token = None
    user.token_expiry = None
    db.commit()
    return HTMLResponse(content="<h2>Email verified successfully! You can now log in.</h2>", status_code=200)

# --- License Directory and Active Sessions ---
LICENSE_DIR = Path("licenses")
LICENSE_DIR.mkdir(exist_ok=True)
active_sessions = {}
# Load public key once (can be global)
try:
    with open("public.pem", "rb") as f:
        pub_key = serialization.load_pem_public_key(f.read())
except FileNotFoundError:
    print("Warning: public_key.pem not found. License verification will not work.")
    pub_key = None


# --- License verification function ---
def verify_license_file(license_file_path: str, current_ip: str):
    """Verify license file and check if IP matches"""
    logger.info(f"Verifying license file: {license_file_path} for IP: {current_ip}")

    if pub_key is None:
        error_msg = "Public key not available for license verification"
        logger.error(error_msg)
        raise Exception(error_msg)

    if not os.path.exists(license_file_path):
        error_msg = f"License file not found: {license_file_path}"
        logger.error(error_msg)
        raise Exception(error_msg)

    try:
        with open(license_file_path, "rb") as f:
            license_blob = f.read()

        logger.info(f"License file size: {len(license_blob)} bytes")

        decoded = base64.b64decode(license_blob)
        logger.info(f"Decoded license size: {len(decoded)} bytes")

        try:
            payload_bytes, signature = decoded.split(b"::SIG::")
            logger.info(f"Payload size: {len(payload_bytes)}, Signature size: {len(signature)}")
        except ValueError as e:
            error_msg = f"Invalid license file format: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        pub_key.verify(
            signature,
            payload_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )

        payload = json.loads(payload_bytes.decode("utf-8"))
        logger.info(f"License payload: {payload}")

        now = datetime.utcnow()
        start_date = datetime.fromisoformat(payload["start_date"].replace("Z", ""))
        end_date   = datetime.fromisoformat(payload["end_date"].replace("Z", ""))
        logger.info(f"License validity: {start_date} to {end_date}, current: {now}")

        if not (start_date <= now <= end_date):
            error_msg = f"License expired or not yet valid. Current: {now}, Valid: {start_date} to {end_date}"
            logger.error(error_msg)
            raise Exception(error_msg)

        allowed_ip = payload.get("allowed_ip")
        logger.info(f"License allowed IP: {allowed_ip}, request IP: {current_ip}")

        if allowed_ip != current_ip:
            error_msg = f"IP {current_ip} not allowed by license (expected: {allowed_ip})"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info("License verification successful")
        return payload

    except Exception as e:
        logger.error(f"License verification failed: {str(e)}")
        raise

# --- Helper for login ---
def verify_user_license(username: str, client_ip: str):
    license_file = LICENSE_DIR / f"{username}_license.bin"
    return verify_license_file(str(license_file), client_ip)
# --- Dependency: check active session ---
def check_active_session(
    request: Request,
    user_id: str = Query(...)
):
    session = active_sessions.get(user_id)
    if not session:
        raise HTTPException(status_code=403, detail="User not logged in")

    # Check session expiry
    if datetime.utcnow() > session.get("expiry", datetime.utcnow()):
        del active_sessions[user_id]
        raise HTTPException(status_code=403, detail="Session expired")

    # Check IP matches
    client_ip = request.client.host
    if session.get("ip") != client_ip:
        raise HTTPException(status_code=403, detail="IP mismatch")

    # Store username in request state (optional, for middleware/other endpoints)
    request.state.username = session.get("username")

    return session.get("username")


# --- User Login Endpoint ---
@app.post("/login/")
def login(user: UserLogin, request: Request, db: OrmSession = Depends(get_db)):

    username = user.username
    password = user.password

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username or password missing")
    
    # --- Step 1: Verify username/password ---
    db_user = authenticate_user(db, username, password)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not db_user.is_email_verified:
        raise HTTPException(status_code=403, detail="Email not verified")

    # --- Step 2: Get client IP ---
    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # --- Step 3: Verify license ---
    # --- Step 3: Verify license ---
    # verify_user_license(db_user.username, client_ip)  # BYPASS for local test

    # --- Step 4: Record active session ---
    active_sessions[str(db_user.id)] = {
        "ip": client_ip,
        "login_time": datetime.utcnow(),
        "expiry": datetime.utcnow() + timedelta(hours=2),
        "user_id": db_user.id,
        "email": db_user.email
    }

    return {"status": "success", "message": "Login successful", "ip": client_ip, "user_id": db_user.id}

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "speech-autodetect-llm",
        "target": TARGET_LANG,
        "autodetect": AUTODETECT_LANGS
    }
@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    user_id: str = Query(...),
    db: OrmSession = Depends(get_db),
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
            chunk_16k = convert_to_wav_any(chunk_path)  # function that converts to PCM16 mono 16k
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
                # If a chunk has no speech (422), just log and continue
                if e.status_code == 422:
                    logger.warning("Chunk %s had no recognizble speech: %s", c, e.detail)
                    continue
                else:
                    raise e
        wall_translate = time.time() - t0
        
        # Remove temporary chunks
        for c in chunks:
            try:
                os.remove(c)
                logger.info("Temp chunk removed: %s", c)
            except Exception as e:
                logger.warning("Failed to remove temp chunk %s: %s", c, e)      
        
        recognized_text = " ".join(all_src).strip()
        translation_en = " ".join(all_en).strip()
        analysis_text = translation_en or recognized_text
        # LLM QA JSON on English translation (fallback to source if needed)
        qa = llm_call_qa_fields(analysis_text)
        result = {
            "user_id": user_id,
            "time_taken_sec": round(wall_translate, 3),
            "detected_language": detected_lang or "",
            "recognized_text": recognized_text,
            "translation_en": translation_en,
            "segments": all_segments,
            "caller": qa.get("caller", ""),   # or extract from LLM
            "callee": qa.get("callee", ""),   # or extract from metadata
            "audio_quality": qa.get("audio_quality", "Good"),  # or detect
            "word_count": len(recognized_text.split()),
            "audio_duration": get_audio_duration(wav_path),
       
            # LLM QA fields
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
        # Save to database
        try:
            db_record = CallAnalytics(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                detected_language=result.get("detected_language"),
                recognized_text=result.get("recognized_text"),
                translation_en=result.get("translation_en"),
                segments=json.dumps(result.get("segments", [])),
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
            logger.info("Saved analysis result to database with id=%d", db_record.id)
            result["db_id"] = db_record.id
            result["session_id"] = db_record.session_id  # Add session_id to response
            
            # --- AUTO EXPORT TO EXCEL ---
            export_call_analytics_to_excel(db)
            
        except Exception as e:
            logger.exception("Failed to save to database: %s", e)
            db.rollback()
        
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        for p in (src_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    logger.info("Temp file removed: %s", p)
            except Exception as e:
                logger.warning("Failed to remove temp %s: %s", p, e)  

# ---------------- DASHBOARD ----------------
@app.get("/dashboard/")
def get_dashboard(
    user_id: str = Query(...),  # Require user_id in query string
    db: OrmSession = Depends(get_db),
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

@app.get("/cloud/dashboard/")
def get_dashboard_alias(
    user_id: str = Query(...),
    db: OrmSession = Depends(get_db),
    username: str = Depends(check_active_session)
):
    """
    Alias for /dashboard/ to support legacy or cloud-prefixed frontend requests.
    """
    return get_dashboard(user_id=user_id, db=db, username=username)

# ---------------- CALL DETAILS ----------------
class CallAnalyticsRequest(BaseModel):
    user_id: str
    page: int = 1
    page_size: int = 10
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
@app.post("/callDetails/")
def get_user_calls(
    call_request: CallAnalyticsRequest,
    db: OrmSession = Depends(get_db),
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

@app.get("/download/")
def download_pdf(
    session_id: str = Query(..., description="Session ID of the call to download"),
    user_id: Optional[str] = Query(None, description="User ID requesting the download"),
    db: OrmSession = Depends(get_db),
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
    pdf.cell(0, 10, f"Call Report - Session ID: {session_id}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Translate:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, call_record.translation_en or call_record.recognized_text or "No translation available.")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Agent Performance Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, call_record.Agent_performance_summary or "No feedback available.")

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
        headers["username"] = str(username)

    # Return the PDF file response
    return FileResponse(
        path=pdf_path,
        filename=f"{session_id}_report.pdf",
        media_type="application/pdf",
        headers=headers
    )


@app.post("/send-email/")
async def send_email(
    email_data: EmailRequest,
    user_id: str,
    db: OrmSession = Depends(get_db),

):
    """
    Send an email and store the record in the database.

    Parameters:
    - to_email: Recipient email address
    - subject: Email subject
    - content: Email body content
    - session_id: Optional session ID to associate with this email

    Returns:
    - Status of the email sending operation
    """
    try:
        # Create the email message
        msg = MIMEText(email_data.content)
        msg["Subject"] = email_data.subject
        msg["From"] = EMAIL_FROM
        msg["To"] = email_data.to_email

        # Try to send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, email_data.to_email, msg.as_string())

        # If sending succeeds, store in database with status 'sent'
        email_record = Email(
            to_email=email_data.to_email,
            subject=email_data.subject,
            content=email_data.content,
            status="sent",
            user_id=user_id,
            session_id=email_data.session_id
        )
        db.add(email_record)
        db.commit()

        return {"status": "success", "message": "Email sent successfully", "email_id": email_record.id, }
        
    except Exception as e:
        # If sending fails, still store in database with status 'failed'
        email_record = Email(
            to_email=email_data.to_email,
            subject=email_data.subject,
            content=email_data.content,
            status="failed",
            # user_id=current_user.id,
            session_id=email_data.session_id
        )
        db.add(email_record)
        db.commit()

        logger.error(f"Failed to send email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send email: {str(e)}"
        )
@app.get("/get-emails/")
def get_emails(
    user_id: str,
    db: OrmSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0
):
    """
    Retrieve sent emails for the current user
    """
    emails = db.query(Email).filter(
        Email.user_id == user_id
    ).order_by(
        Email.sent_at.desc()
    ).offset(offset).limit(limit).all()

    return [
        {
            "id": email.id,
            "to_email": email.to_email,
            "subject": email.subject,
            "sent_at": email.sent_at.isoformat(),
            "status": email.status,
            "session_id": email.session_id,
        }
        for email in emails
    ]

@app.get("/get-email/{email_id}/")
def get_email_details(
    email_id: str,
    user_id: str,
    db: OrmSession = Depends(get_db),
):
    """
    Get details of a specific email
    """
    email = db.query(Email).filter(
        Email.id == email_id,
        Email.user_id == user_id
    ).first()

    if not email:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Email not found"
        )

    return {
        "id": email.id,
        "to_email": email.to_email,
        "subject": email.subject,
        "content": email.content,
        "sent_at": email.sent_at.isoformat(),
        "status": email.status,
        "session_id": email.session_id,
    }
    
class UserFeedbackCreate(BaseModel):
    session_id: str
    user_feedback: str

@app.post("/create-user-feedback")
def submit_user_feedback(
    session_id: str = Form(...),
    user_feedback: str = Form(...),
    db: OrmSession = Depends(get_db)
):
    if not db.query(CallSession).filter(CallSession.session_id == session_id).first():
        raise HTTPException(status_code=404, detail="Session not found")

    feedback_entry = UserFeedback(
        session_id=session_id,
        user_feedback=user_feedback,
        created_at = datetime.utcnow(),
    )
    db.add(feedback_entry)
    db.commit()

    return {"message": "Feedback saved successfully."}


@app.get("/get-user-feedback")
def get_user_feedback(
    session_id: str,
    db: OrmSession = Depends(get_db)
):
    feedbacks = db.query(UserFeedback).filter(UserFeedback.session_id == session_id).all()

    if not feedbacks:
        raise HTTPException(status_code=404, detail="No feedback found for this session")

    return [
        {
            "id": feedback.id,
            "session_id": feedback.session_id,
            "user_feedback": feedback.user_feedback,
            "created_at": feedback.created_at
        }
        for feedback in feedbacks
    ]

@app.get("/")
def root():
    return {"message": "Live Call Assistant WebSocket is ready."}


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

# Summarize using Azure OpenAI
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
    """
    Splits long text into manageable chunks for summarization.
    """
    return textwrap.wrap(text, width=max_length, break_long_words=False, replace_whitespace=False)

# Try to import sentence-transformers, but handle gracefully if not available
try:
    from sentence_transformers import SentenceTransformer
    sentancemodel = SentenceTransformer('all-mpnet-base-v2')
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"sentence-transformers not available: {e}")
    sentancemodel = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def embed_chunks(chunks, embedding_model=None):
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("Sentence transformers not available, skipping embedding generation")
        # Return dummy embeddings or handle gracefully
        import numpy as np
        return np.random.random((len(chunks), 384)).astype('float32')  # 384 is typical BERT embedding size

    if embedding_model is None:
        embedding_model = sentancemodel
    return embedding_model.encode(chunks, convert_to_numpy=True).astype('float32')

    try:
        import faiss
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except ImportError:
        logger.warning("FAISS not installed, skipping index build")
        return None

# API endpoint
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...),
                     isProspectus : bool = Form(...),
                     db: OrmSession = Depends(get_db)):
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

        if isProspectus :
            chunks = split_text_into_chunks(summary, max_length=350)
            embeddings = embed_chunks(chunks)
            index = build_faiss_index(embeddings)
            faiss.write_index(index, "faiss_index.index")

        # Save to SQLite
        created_date = datetime.utcnow().isoformat()
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
            "created_date": created_date,
            "text": text,
            "summary": summary
        }

    except Exception as e:
        logger.error("Failed to process PDF upload: %s", str(e))
        return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(e)})



# Strict prompt: no headings, no translation, no transliteration, no extras.
_CLEANUP_PROMPT_TMPL = """You are rewriting a noisy automatic speech recognition (ASR) transcript.

Hard rules (must follow all):
- Do NOT translate. Keep every sentence in its original language(s).
- Do NOT transliterate. Preserve the original script for each language (Tamil → Tamil script, Hindi → Devanagari, etc.).
- Do NOT add headings, labels, or meta text (e.g., “Corrected Transcript”, “Tamil, English (Code-mixed)”).
- Do NOT add any commentary, instructions, summaries, emojis, or calls to action.
- Only fix obvious ASR errors: merge split syllables/words, remove duplicated fragments, restore natural punctuation/casing.
- Preserve code-mixed brand names and entities exactly (e.g., “Kotak Mahindra”, “Goa coupon”).
- Keep speaker turns if they already exist (e.g., “Speaker 1:”); otherwise do not invent them.
- Output ONLY the corrected transcript text. Nothing else.

Input:
{chunk}
"""

def _chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    """Split text into chunks of maximum character length."""
    chunks = []
    current_chunk = ""

    for sentence in text.split('.'):
        if len(current_chunk + sentence) <= max_chars:
            current_chunk += sentence + '.'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '.'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def _azure_generate(prompt: str, model: str = None, temperature: float = None) -> str:
    if model is None:
        model = config.get("AZURE_OPENAI_MODEL", "o4-mini")
    if temperature is None:
        temperature = float(config.get("AZURE_OPENAI_TEMPERATURE", 0.0))
    
    """Generate text using Azure OpenAI."""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Azure OpenAI generation failed: {e}")
        return ""

def _preclean_asr(text: str) -> str:
    """Gentle pre-clean; avoid damaging Latin words."""
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove ellipsis clutter ("... ...")
    text = re.sub(r"(?:\.\s*){2,}", " ", text)

    # Merge split *Indic* letters only (avoid touching Latin):
    # Tamil: \u0B80-\u0BFF; Devanagari: \u0900-\u097F; add other blocks as needed.
    indic_blocks = [
        ("\u0B80-\u0BFF"),  # Tamil
        ("\u0900-\u097F"),  # Devanagari (Hindi/Marathi)
        ("\u0C80-\u0CFF"),  # Kannada
        ("\u0C00-\u0C7F"),  # Telugu
        ("\u0D00-\u0D7F"),  # Malayalam
        ("\u0A80-\u0AFF"),  # Gujarati
        ("\u0A00-\u0A7F"),  # Gurmukhi (Punjabi)
        ("\u0980-\u09FF"),  # Bengali
        ("\u0B00-\u0B7F"),  # Odia
    ]
    block_union = "".join(indic_blocks)
    # Merge single stray spaces inside Indic words only (…க் கும் -> …க்கும்)
    pattern = rf"([{block_union}])\s+([{block_union}])"
    text = re.sub(pattern, r"\1\2", text)

    # Ensure a space after punctuation where missing
    text = re.sub(r"(?<=[:\.\?\!])([^\s])", r" \1", text)
    return text.strip()

def azure_cleanup(text: str,
                  model: str = None,
                  temperature: float = None,
                  max_chars: int = 1200) -> str:
    if model is None:
        model = config.get("AZURE_OPENAI_MODEL", "o4-mini")
    if temperature is None:
        temperature = float(config.get("AZURE_OPENAI_TEMPERATURE", 0.0))
    """
    Multilingual ASR cleanup (Indian languages + English) with strict no-translation,
    no-transliteration, and no-extras policy.
    """
    text = _preclean_asr(text)  # make sure this function exists
    outputs = []

    for chunk in _chunk_text(text, max_chars=max_chars):
        prompt = _CLEANUP_PROMPT_TMPL.format(chunk=chunk)  # must be defined
        cleaned = _azure_generate(prompt, model=model, temperature=temperature)
        outputs.append(cleaned)

    final_text = "\n".join(outputs).strip()

    # Remove trailing CTA lines
    cta_regex = r"(?im)^\s*(please\s+subscribe.*|subscribe\s+to\s+my\s+channel.*|share\s+this\s+video.*|thank\s+you\.?)\s*$"
    final_text = re.sub(cta_regex, "", final_text).strip()
    return final_text

# Optional alias
Ask_Azure_Cleanup = azure_cleanup
Ask_Azure_Cleanup = azure_cleanup