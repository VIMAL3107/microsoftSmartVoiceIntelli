import requests
import os
import time
import sys
import shutil

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USERNAME = "vimal011"
PASSWORD = "1234"
AUDIO_DIR = r"c:\Users\nexge\Music\Projects\microsoftSmartVoiceIntelli\audio test"
PROCESSED_DIR = os.path.join(AUDIO_DIR, "processed")

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def login():
    """Authenticates the user and returns user_id."""
    url = f"{BASE_URL}/login/"
    payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    # print(f"Attempting login for user: {USERNAME}")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("user_id")
        else:
            print(f"Login failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {BASE_URL}. Is it running?")
        return None
    except Exception as e:
        print(f"Login error: {e}")
        return None

def analyze_file(user_id, file_path):
    """Uploads and analyzes an audio file."""
    url = f"{BASE_URL}/analyze"
    params = {"user_id": user_id}
    filename = os.path.basename(file_path)
    
    print(f"Processing: {filename}...")
    
    if not os.path.exists(file_path):
        return False

    with open(file_path, 'rb') as f:
        files = {'audio': (filename, f, 'audio/wav')}
        try:
            response = requests.post(url, params=params, files=files)
            if response.status_code == 200:
                result = response.json()
                print(f"Analysis complete for {filename}.")
                print(f"  - Detected Lang: {result.get('detected_language')}")
                return True
            else:
                print(f"Analysis failed for {filename}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            return False

def main():
    print("Starting Auto Audio Processor...")
    print(f"Watching directory: {AUDIO_DIR}")
    print("Press Ctrl+C to stop.")

    user_id = login()
    if not user_id:
        print("Initial login failed. Retrying in loop...")

    while True:
        try:
            # Re-login if needed
            if not user_id:
                user_id = login()
                if not user_id:
                    time.sleep(10)
                    continue

            # Scan for files
            files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
            
            if files:
                print(f"Found {len(files)} new files.")
                for f in files:
                    full_path = os.path.join(AUDIO_DIR, f)
                    
                    # Process
                    if analyze_file(user_id, full_path):
                        # Move to processed
                        new_path = os.path.join(PROCESSED_DIR, f)
                        # Handle overwrite if exists in processed
                        if os.path.exists(new_path):
                            base, ext = os.path.splitext(f)
                            new_path = os.path.join(PROCESSED_DIR, f"{base}_{int(time.time())}{ext}")
                        
                        shutil.move(full_path, new_path)
                        print(f"✅ Moved {f} to 'processed/'")
                    else:
                        print(f"❌ Failed to process {f}. Leaving in place.")
            
            # Wait before next scan
            time.sleep(5)

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
