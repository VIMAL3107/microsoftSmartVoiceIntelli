
import sys
import os
import json
from cryptography.fernet import Fernet

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import FERNET_KEY

def update_schedule():
    fernet = Fernet(FERNET_KEY)
    
    # Needs to find config.enc in root or parent
    # If run from scripts/, config.enc is likely in ..
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.enc")
    
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            encrypted_bytes = f.read()
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        config = json.loads(decrypted_bytes.decode("utf-8"))
    else:
        print(f"Config file not found at {config_path}")
        return

    # Update config to 1 minute for testing
    config["REPORT_SCHEDULE"] = {
        "enabled": True,
        "interval_minutes": 1, 
        "email_to": "iamvimal3107@gmail.com"
    }

    # Encrypt back
    encrypted_bytes = fernet.encrypt(json.dumps(config).encode("utf-8"))
    with open(config_path, "wb") as f:
        f.write(encrypted_bytes)

    print("Config updated to 1 minute interval.")

if __name__ == "__main__":
    update_schedule()
