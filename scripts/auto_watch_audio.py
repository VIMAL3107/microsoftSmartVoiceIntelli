import time
import os
import sys
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Allow importing existing logic
sys.path.append(os.path.dirname(__file__))
from test_audio_workflow import analyze_file, login, USERNAME, PASSWORD, BASE_URL

AUDIO_DIR = r"c:\Users\nexge\Music\Projects\microsoftSmartVoiceIntelli\audio test"

class AudioHandler(FileSystemEventHandler):
    def __init__(self, user_id):
        self.user_id = user_id
        # Ensure processed dir exists
        self.processed_dir = os.path.join(AUDIO_DIR, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        print(f"👀 Watching directory: {AUDIO_DIR}")
        print("📂 Drop a .wav file here, and I will process it automatically!")

    def on_created(self, event):
        if event.is_directory:
            return
        
        filename = os.path.basename(event.src_path)
        if not filename.lower().endswith(".wav"):
            return
            
        print(f"\n🆕 New file detected: {filename}")
        
        # Give file a moment to finish copying (if large)
        time.sleep(1)
        
        # Process it
        if analyze_file(self.user_id, event.src_path):
            # Move to processed
            new_path = os.path.join(self.processed_dir, filename)
            # Handle collision
            if os.path.exists(new_path):
                base, ext = os.path.splitext(filename)
                new_path = os.path.join(self.processed_dir, f"{base}_{int(time.time())}{ext}")
            
            try:
                shutil.move(event.src_path, new_path)
                print(f"✅ Processed & Moved to: processed/{os.path.basename(new_path)}")
            except Exception as e:
                print(f"⚠️ Processed but failed to move: {e}")

def main():
    print("🚀 Starting Automatic Audio Watcher...")
    
    # 1. Login first
    user_id = login()
    if not user_id:
        print("❌ Login failed. Exiting.")
        return

    # 2. Start Watching
    event_handler = AudioHandler(user_id)
    observer = Observer()
    observer.schedule(event_handler, AUDIO_DIR, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
