
import sys
import os
import json
from cryptography.fernet import Fernet

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import FERNET_KEY

def update_email_schedule(email_to, interval_minutes):
    fernet = Fernet(FERNET_KEY)
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.enc")
    
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            encrypted_bytes = f.read()
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        config = json.loads(decrypted_bytes.decode("utf-8"))
    else:
        print(f"Config file not found at {config_path}")
        return

    # Update schedule
    print(f"Updating schedule for {email_to} to run every {interval_minutes} minutes.")
    config["REPORT_SCHEDULE"] = {
        "enabled": True,
        "interval_minutes": float(interval_minutes), 
        "email_to": email_to
    }

    # Encrypt back
    encrypted_bytes = fernet.encrypt(json.dumps(config).encode("utf-8"))
    with open(config_path, "wb") as f:
        f.write(encrypted_bytes)

    print("Configuration updated successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python update_schedule.py <email_address> <interval_minutes>")
        print("Example: python update_schedule.py client@example.com 60")
    else:
        update_email_schedule(sys.argv[1], sys.argv[2])
