
import bcrypt
import jwt
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pathlib import Path
import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from app.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, VALID_LICENSE_KEYS
from app.models.user import User

logger = logging.getLogger(__name__)

# --- Password hashing ---

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# --- JWT Helpers ---

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password):
        return None
    return user 

# --- License Helpers ---

LICENSE_DIR = Path("licenses")
LICENSE_DIR.mkdir(exist_ok=True)

# Load public key
pub_key = None
try:
    with open("public.pem", "rb") as f:
        pub_key = serialization.load_pem_public_key(f.read())
except FileNotFoundError:
    print("Warning: public_key.pem not found. License verification will not work.")

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
        # sys.exit(1) # Don't exit in service, let caller handle or raise

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

        decoded = base64.b64decode(license_blob)

        try:
            payload_bytes, signature = decoded.split(b"::SIG::")
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

        now = datetime.utcnow()
        start_date = datetime.fromisoformat(payload["start_date"].replace("Z", ""))
        end_date   = datetime.fromisoformat(payload["end_date"].replace("Z", ""))

        if not (start_date <= now <= end_date):
            error_msg = f"License expired or not yet valid. Current: {now}, Valid: {start_date} to {end_date}"
            logger.error(error_msg)
            raise Exception(error_msg)

        allowed_ip = payload.get("allowed_ip")
       #if allowed_ip != current_ip:
        if allowed_ip != "*" and allowed_ip != current_ip:
            error_msg = f"IP {current_ip} not allowed by license (expected: {allowed_ip})"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info("License verification successful")
        return payload

    except Exception as e:
        logger.error(f"License verification failed: {str(e)}")
        raise

def verify_user_license(username: str, client_ip: str):
    license_file = LICENSE_DIR / f"{username}_license.bin"
    return verify_license_file(str(license_file), client_ip)


def verify_admin_license(client_ip: str):
    license_file = LICENSE_DIR / "admin_license.bin"
    return verify_license_file(str(license_file), client_ip)