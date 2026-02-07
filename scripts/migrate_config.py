
import os
import json
from dotenv import load_dotenv, set_key
from cryptography.fernet import Fernet

def migrate_config():
    print("--- Migrating config.enc to .env ---")
    
    # 1. Load environment variables (to get FERNET_KEY)
    load_dotenv()
    
    fernet_key = os.getenv("FERNET_KEY")
    if not fernet_key:
        print("[ERROR] FERNET_KEY not found in environment variables. Cannot decrypt.")
        return

    try:
        fernet = Fernet(fernet_key)
    except Exception as e:
         print(f"[ERROR] Invalid FERNET_KEY: {e}")
         return

    # 2. Locate config.enc
    enc_path = "config.enc"
    if not os.path.exists(enc_path):
        print(f"[ERROR] {enc_path} not found in current directory.")
        return

    # 3. Decrypt config.enc
    try:
        with open(enc_path, "rb") as f:
            encrypted_data = f.read()
        
        decrypted_data = fernet.decrypt(encrypted_data)
        config_data = json.loads(decrypted_data.decode("utf-8"))
        print(f"[SUCCESS] Decrypted 'config.enc'. Found {len(config_data)} keys.")
    except Exception as e:
        print(f"[ERROR] Failed to decrypt/parse config.enc: {e}")
        return

    # 4. Save to .env
    env_file = ".env"
    
    # We will use dotenv.set_key to safely update/add keys
    # Note: set_key might be slow for many keys, but it's safe.
    # Alternatively, we can append to file. 
    # Let's append to avoid dependency on set_key handling comments poorly/reformatting.
    
    existing_keys = {}
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key = line.split("=", 1)[0]
                    existing_keys[key] = True

    new_keys_count = 0
    with open(env_file, "a") as f:
        f.write("\n# --- Imported from config.enc ---\n")
        for key, value in config_data.items():
            if key in existing_keys:
                print(f"[SKIP] {key} already exists in .env")
                continue
            
            # Format value
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
            elif isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)
            
            # Write to file
            f.write(f"{key}={value_str}\n")
            print(f"[ADDED] {key}")
            new_keys_count += 1

    print(f"--- Migration Complete. Added {new_keys_count} new keys to .env ---")

if __name__ == "__main__":
    migrate_config()
