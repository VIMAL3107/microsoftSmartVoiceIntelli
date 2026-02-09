
import os
import json
from dotenv import load_dotenv

load_dotenv()

# General
# Prioritize environment variables, provide sensible defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_SECRET")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# Azure Speech
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "").strip()
TARGET_LANG = os.getenv("TARGET_LANG", "en").strip()
AUTODETECT_LANGS_RAW = os.getenv("AUTODETECT_LANGS", "en-US,ta-IN,hi-IN,te-IN,kn-IN").strip()
AUTODETECT_LANGS = [l.strip() for l in AUTODETECT_LANGS_RAW.split(",") if l.strip()][:4]

if not SPEECH_KEY or not SPEECH_REGION:
    print("CRITICAL WARNING: AZURE_SPEECH_KEY or AZURE_SPEECH_REGION is missing!")

# Azure OpenAI
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY", "").strip()
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip('/')
AOAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "").strip()
AOAI_API_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Email
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://microsoftsmartvoiceintelli.onrender.com")

# License
_valid_licenses_str = os.getenv("VALID_LICENSE_KEYS", "[]")
try:
    VALID_LICENSE_KEYS = json.loads(_valid_licenses_str)
except (json.JSONDecodeError, ValueError):
    VALID_LICENSE_KEYS = []

# Report Scheduling
REPORT_SCHEDULE = {}
env_report_schedule = os.getenv("REPORT_SCHEDULE")
if env_report_schedule and isinstance(env_report_schedule, str):
    try:
        parsed_schedule = json.loads(env_report_schedule)
        if isinstance(parsed_schedule, dict):
            REPORT_SCHEDULE.update(parsed_schedule)
    except json.JSONDecodeError:
        pass

REPORT_SCHEDULE_ENABLED = str(os.getenv("REPORT_SCHEDULE_ENABLED", REPORT_SCHEDULE.get("enabled", False))).lower() == "true"
REPORT_INTERVAL_MINUTES = float(os.getenv("REPORT_INTERVAL_MINUTES", REPORT_SCHEDULE.get("interval_minutes", 10.0)))
REPORT_EMAIL_TO = os.getenv("REPORT_EMAIL_TO", REPORT_SCHEDULE.get("email_to", "iamvimal3107@gmail.com"))
REPORT_DAILY_TIME = os.getenv("REPORT_DAILY_TIME", REPORT_SCHEDULE.get("daily_time"))

# Database
DATABASE_URL = "sqlite:///./voice_translate.db"
