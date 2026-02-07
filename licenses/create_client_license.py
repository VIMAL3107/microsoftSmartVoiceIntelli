
import json
import base64
import datetime
import argparse
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

PRIVATE_KEY_FILE = "private.pem"

def create_client_license(client_name, allowed_ip, days_valid=365):
    # 1. Load Private Key
    try:
        with open(PRIVATE_KEY_FILE, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
    except FileNotFoundError:
        print(f"Error: {PRIVATE_KEY_FILE} not found. Run generate_keys.py first!")
        return

    # 2. Prepare Data
    start_date = datetime.datetime.utcnow()
    end_date = start_date + datetime.timedelta(days=days_valid)
    
    license_data = {
        "client_name": client_name,
        "allowed_ip": allowed_ip,
        "start_date": start_date.isoformat() + "Z",
        "end_date": end_date.isoformat() + "Z"
    }

    payload_json = json.dumps(license_data)
    payload_bytes = payload_json.encode("utf-8")
    
    print(f"Generating license for {client_name} ({allowed_ip})...")

    # 3. Sign
    signature = private_key.sign(
        payload_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    # 4. Save
    combined = payload_bytes + b"::SIG::" + signature
    license_content = base64.b64encode(combined)
    
    filename = f"licenses/{client_name}_license.bin"
    with open(filename, "wb") as f:
        f.write(license_content)
        
    print(f"SUCCESS: Created {filename}")
    print(f"Give this file AND public.pem to the client.")

if __name__ == "__main__":
    # Example usage: python create_client_license.py "clientA" "192.168.1.50"
    import sys
    if len(sys.argv) < 3:
        print("Usage: python create_client_license.py <client_name> <allowed_ip> [days_valid]")
        print("Example: python create_client_license.py client_x 10.0.0.5")
    else:
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 365
        create_client_license(sys.argv[1], sys.argv[2], days)
