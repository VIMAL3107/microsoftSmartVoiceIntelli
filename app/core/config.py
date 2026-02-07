
import os
import json
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

# FERNET_KEY for decrypting config.enc
# Ideally this should be in env vars too, but keeping as per original code
FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    raise ValueError("FERNET_KEY not found in environment variables")
fernet = Fernet(FERNET_KEY)

def load_encrypted_config(enc_file_path: str = "config.enc") -> dict:
    """Load and decrypt configuration from encrypted file."""
    if not os.path.exists(enc_file_path):
        # Fallback if running from a different directory (e.g. app/)
        # Try looking in parent directory
        if os.path.exists(os.path.join("..", enc_file_path)):
            enc_file_path = os.path.join("..", enc_file_path)
            
    try:
        with open(enc_file_path, "rb") as f:
            encrypted_bytes = f.read()
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        config = json.loads(decrypted_bytes.decode("utf-8"))
        return config
    except Exception as e:
        print(f"Warning: Could not load encrypted config: {e}")
        return {}

# Load configuration
config = load_encrypted_config()

# General
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
SECRET_KEY = config.get("SECRET_KEY", "CHANGE_THIS_SECRET")
ALGORITHM = config.get("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = config.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60)

# Azure Speech
SPEECH_KEY = config.get("AZURE_SPEECH_KEY", "").strip()
SPEECH_REGION = config.get("AZURE_SPEECH_REGION", "").strip()
TARGET_LANG = config.get("TARGET_LANG", "en").strip()
AUTODETECT_LANGS_RAW = config.get("AUTODETECT_LANGS", "en-US,ta-IN,hi-IN,te-IN,kn-IN").strip()
AUTODETECT_LANGS = [l.strip() for l in AUTODETECT_LANGS_RAW.split(",") if l.strip()][:4]

# Azure OpenAI
AOAI_KEY = config.get("AZURE_OPENAI_KEY", "").strip()
AOAI_ENDPOINT = config.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip('/')
AOAI_MODEL = config.get("AZURE_OPENAI_MODEL", "").strip()
AOAI_API_VER = config.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Email
EMAIL_FROM = os.getenv("EMAIL_FROM", config.get("EMAIL_FROM"))
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", config.get("EMAIL_PASSWORD"))
SMTP_SERVER = os.getenv("SMTP_SERVER", config.get("SMTP_SERVER", "smtp.gmail.com"))
SMTP_PORT = int(os.getenv("SMTP_PORT", config.get("SMTP_PORT", 587)))
FRONTEND_URL = os.getenv("FRONTEND_URL", config.get("FRONTEND_URL", "https://microsoftsmartvoiceintelli.onrender.com"))

# License
VALID_LICENSE_KEYS = config.get("VALID_LICENSE_KEYS", [])
if isinstance(VALID_LICENSE_KEYS, str):
    try:
        VALID_LICENSE_KEYS = json.loads(VALID_LICENSE_KEYS)
    except (json.JSONDecodeError, ValueError):
        VALID_LICENSE_KEYS = []

# Report Scheduling
REPORT_SCHEDULE = config.get("REPORT_SCHEDULE", {})
REPORT_SCHEDULE_ENABLED = REPORT_SCHEDULE.get("enabled", False)
REPORT_INTERVAL_MINUTES = REPORT_SCHEDULE.get("interval_minutes", 10.0)
REPORT_EMAIL_TO = REPORT_SCHEDULE.get("email_to", "iamvimal3107@gmail.com")
# Database
# Using SQLite for now as per original code
DATABASE_URL = "sqlite:///./voice_translate.db"
