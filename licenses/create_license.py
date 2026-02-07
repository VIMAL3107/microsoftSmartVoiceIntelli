
import json
import base64
import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# --- CONFIGURATION ---
PRIVATE_KEY_FILE = "licenses/private.pem"  # You need this file to sign!
LICENSE_FILENAME = "licenses/new_user_license.bin"

LICENSE_DATA = {
    "allowed_ip": "216.24.57.251", # Allow ANY IP address (Recommended for testing)
    "start_date": datetime.datetime.utcnow().isoformat() + "Z", # "2023-10-16T12:00:00Z"
    "end_date": "2026-12-31T23:59:59Z"   # Adjust validity here
}

def create_license():
    # 1. Load Private Key
    try:
        with open(PRIVATE_KEY_FILE, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None, # Add password if your key is encrypted
                backend=default_backend()
            )
    except FileNotFoundError:
        print(f"Error: {PRIVATE_KEY_FILE} not found. You cannot create a signed license without the private key.")
        return

    # 2. Prepare Payload
    payload_json = json.dumps(LICENSE_DATA)
    payload_bytes = payload_json.encode("utf-8")
    print(f"Payload: {payload_json}")

    # 3. Sign Payload
    signature = private_key.sign(
        payload_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    # 4. Combine & Encode
    # Format: [PAYLOAD]::SIG::[SIGNATURE]
    combined = payload_bytes + b"::SIG::" + signature
    license_content = base64.b64encode(combined)

    # 5. Save File
    with open(LICENSE_FILENAME, "wb") as f:
        f.write(license_content)
    
    print(f"Successfully created license: {LICENSE_FILENAME}")

if __name__ == "__main__":
    create_license()
