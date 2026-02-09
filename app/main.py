
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os
import time
import logging
import traceback
import contextvars

from app.core.config import LOG_LEVEL, AUTODETECT_LANGS, TARGET_LANG
from app.core.database import Base, engine
from app.api import auth, analysis, reports, feedback, email, chat
from app.services.scheduler import start_scheduler
from app.services.security import check_license

# Run license check on startup import (or inside startup event)
check_license()


# Logging Setup
from logging.handlers import TimedRotatingFileHandler

if not os.path.exists("logs"):
    os.makedirs("logs")

request_id_var = contextvars.ContextVar("request_id", default="-")
class ReqIdFilter(logging.Filter):
    def filter(self, record): record.request_id = request_id_var.get("-"); return True

log_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")

# File Handler (Rotates daily at midnight, keeps 30 days)
file_handler = TimedRotatingFileHandler("logs/app.log", when="midnight", interval=1, backupCount=30)
file_handler.setFormatter(log_formatter)
file_handler.addFilter(ReqIdFilter())

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.addFilter(ReqIdFilter())

# Configure Root Logger
# Use handlers list so we have both file and console logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("speech-autodetect-llm")

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Speech -> English (AutoDetect) + LLM QA", version="1.0.0")

# CORS
origins = [o.strip() for o in (os.getenv("CORS_ALLOW_ORIGINS","*") or "*").split(",")]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True
)

# Middleware for Request ID
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

@app.on_event("startup")
async def startup_event():
    start_scheduler()

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "speech-autodetect-llm",
        "target": TARGET_LANG,
        "autodetect": AUTODETECT_LANGS
    }

@app.get("/")
def root():
    return {"message": "Live Call Assistant WebSocket is ready."}

# Include Routers
app.include_router(auth.router)
app.include_router(analysis.router) 
app.include_router(reports.router)
app.include_router(feedback.router)
app.include_router(email.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
