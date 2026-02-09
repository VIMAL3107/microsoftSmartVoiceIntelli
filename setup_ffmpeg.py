import os
import zipfile
import io
import requests
import shutil
from pathlib import Path

def setup_ffmpeg():
    print("Downloading FFmpeg for Windows (Essentials build)...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        print("Download complete. Extracting...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the ffmpeg.exe inside the zip
            ffmpeg_path = None
            for name in z.namelist():
                if name.endswith("bin/ffmpeg.exe"):
                    ffmpeg_path = name
                    break
            
            if not ffmpeg_path:
                print("Error: Could not find ffmpeg.exe in the downloaded zip.")
                return

            # Extract to ./bin/ffmpeg.exe
            target_dir = Path("bin")
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / "ffmpeg.exe"
            
            with z.open(ffmpeg_path) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

            print(f"FFmpeg successfully installed to: {target_path.resolve()}")
            print("You can now run the application.")

    except Exception as e:
        print(f"Failed to download/install FFmpeg: {e}")

if __name__ == "__main__":
    setup_ffmpeg()
