import requests
import os
import time
import sys

# Configuration
BASE_URL = "https://microsoftsmartvoiceintelli.onrender.com"
USERNAME = "vimal011       n"
PASSWORD = "1234"
AUDIO_DIR = r"c:\Users\nexge\Music\Projects\microsoftSmartVoiceIntelli\audio test"
REPORT_EMAIL = "iamvimal3107@gmail.com"  # Change this to your desired email

def login():
    """Authenticates the user and returns user_id."""
    url = f"{BASE_URL}/login/"
    payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    print(f"Attempting login for user: {USERNAME}")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            user_id = data.get("user_id")
            print(f"Login Successful. User ID: {user_id}")
            return user_id
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
        print(f"File not found: {file_path}")
        return False

    with open(file_path, 'rb') as f:
        files = {'audio': (filename, f, 'audio/wav')}
        try:
            response = requests.post(url, params=params, files=files)
            if response.status_code == 200:
                result = response.json()
                print(f"Analysis complete for {filename}.")
                # Optional: Print simple summary
                print(f"  - Detected Lang: {result.get('detected_language')}")
                print(f"  - Sentiment: {result.get('conversation_feel')}")
                return True
            else:
                print(f"Analysis failed for {filename}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            return False

def send_daily_report(email_to):
    """Triggers the daily report email."""
    url = f"{BASE_URL}/admin/daily-report"
    print(f"Triggering daily report to: {email_to}")
    
    params = {
        "secret": "INTELLICORE-SECRET", 
        "to_email": email_to
    }
    
    try:
        response = requests.post(url, params=params)
        if response.status_code == 200:
            print(f"Report sent successfully!")
            print(response.json())
        else:
            print(f"Failed to send report: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending report: {e}")

def main():
    print("Starting Audio Analysis Workflow...")
    
    # Check if directory exists
    if not os.path.exists(AUDIO_DIR):
        print(f"Directory not found: {AUDIO_DIR}")
        return

    # 1. Login
    user_id = login()
    if not user_id:
        print("Aborting workflow due to login failure.")
        return

    # 2. Process Files
    files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
    if not files:
        print(f"No .wav files found in {AUDIO_DIR}")
    else:
        print(f"Found {len(files)} audio files.")
        for i, f in enumerate(files, 1):
            full_path = os.path.join(AUDIO_DIR, f)
            print(f"[{i}/{len(files)}] ", end="")
            
            # processed folder
            processed_dir = os.path.join(AUDIO_DIR, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            if analyze_file(user_id, full_path):
                # Move to processed
                import shutil
                new_path = os.path.join(processed_dir, f)
                shutil.move(full_path, new_path)
                print(f"✅ Moved {f} to 'processed/' folder.")
            else:
                print(f"❌ Failed to process {f}. Keeping in folder.")
            
            time.sleep(1) # Small delay to be polite to the server

    # 3. Send Report
    # Check if we should send report even if no files? 
    # Usually yes, to report 0 calls or just previous calls from the day.
    send_daily_report(REPORT_EMAIL)
    
    print("Workflow completed.")

if __name__ == "__main__":
    main()
