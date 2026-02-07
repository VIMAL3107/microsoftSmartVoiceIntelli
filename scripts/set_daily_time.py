
import sys
import os
import json
from cryptography.fernet import Fernet

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import FERNET_KEY

def set_daily_time(daily_time, email_to):
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
    print(f"Updating schedule to run DAILY at {daily_time} for {email_to}")
    config["REPORT_SCHEDULE"] = {
        "enabled": True,
        "daily_time": daily_time,      # "20:00"
        "interval_minutes": 0,         # Disable interval mode
        "email_to": email_to
    }

    # Encrypt back
    encrypted_bytes = fernet.encrypt(json.dumps(config).encode("utf-8"))
    with open(config_path, "wb") as f:
        f.write(encrypted_bytes)

    print("Configuration updated successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python set_daily_time.py <HH:MM> [email]")
        print("Example: python set_daily_time.py 20:00 client@example.com")
        print("Example: python set_daily_time.py 08:30")
    else:
        time_str = sys.argv[1]
        email = sys.argv[2] if len(sys.argv) > 2 else "iamvimal3107@gmail.com"
        set_daily_time(time_str, email)
