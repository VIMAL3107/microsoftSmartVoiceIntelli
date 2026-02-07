
import os
import wave
import time
import shutil
import subprocess
import logging
import concurrent.futures
import tempfile
from typing import Dict, Any, List
import azure.cognitiveservices.speech as speechsdk
from fastapi import HTTPException
from app.core.config import SPEECH_KEY, SPEECH_REGION, TARGET_LANG, AUTODETECT_LANGS

logger = logging.getLogger(__name__)

def _resolve_ffmpeg_bin() -> str:
    for c in [os.getenv("FFMPEG_BIN","").strip(), "/usr/bin/ffmpeg", "ffmpeg"]:
        if not c: continue
        try:
            subprocess.run([c, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return c
        except Exception:
            pass
    # Warning instead of raising here, let caller decide or raise later
    logger.warning("FFmpeg not found. Helper functions might fail.")
    return "ffmpeg"

def _wav_duration(path_wav: str) -> float:
    try:
        with wave.open(path_wav, "rb") as wf:
            rate = wf.getframerate() or 16000
            return wf.getnframes() / float(rate)
    except Exception:
        return 0.0

def get_audio_duration(path_wav: str) -> float:
    return _wav_duration(path_wav)

def convert_to_wav_any(input_path: str) -> str:
    ffmpeg = _resolve_ffmpeg_bin()
    out_wav = input_path + ".__16k_mono_pcm.wav"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-nostdin",
        "-i", input_path,
        "-vn", "-sn", "-dn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        out_wav
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(out_wav):
        err = proc.stderr.decode(errors="ignore")[-800:]
        logger.error("FFmpeg failed rc=%s tail=%s", proc.returncode, err)
        if input_path.lower().endswith(".wav"):
             logger.warning("FFmpeg failed, but input is WAV. Attempting to use directly.")
             shutil.copy2(input_path, out_wav)
             return out_wav
        raise HTTPException(400, f"Audio conversion failed: {err}")
    logger.info("FFmpeg convert ok (%.0f ms)", (time.time()-t0)*1000)
    return out_wav

def split_wav(path_wav: str, chunk_sec: int = 60):
    chunks = []
    with wave.open(path_wav, "rb") as wf:
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames_per_chunk = int(chunk_sec * framerate)
        total_frames = wf.getnframes()
        for start in range(0, total_frames, frames_per_chunk):
            wf.setpos(start)
            frames = wf.readframes(min(frames_per_chunk, total_frames - start))
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp_wav.name, "wb") as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(framerate)
                out_wf.writeframes(frames)
            chunks.append(tmp_wav.name)
    return chunks

def translate_audio_autodetect(path_wav: str) -> Dict[str, Any]:
    try:
        stc = speechsdk.translation.SpeechTranslationConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    except Exception as e:
        logger.error(f"Failed to create SpeechTranslationConfig. Verify SPEECH_KEY and SPEECH_REGION. Error: {e}")
        raise HTTPException(status_code=500, detail="Speech service configuration failed. Please check server logs.")
    stc.add_target_language(TARGET_LANG)
    stc.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")
    stc.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "2000")
    try:
        stc.set_profanity(speechsdk.ProfanityOption.Raw)
    except Exception:
        pass
    langs = AUTODETECT_LANGS if AUTODETECT_LANGS else ["en-US"]
    auto_cfg = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(langs)
    audio_config = speechsdk.audio.AudioConfig(filename=path_wav)
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=stc,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_cfg
    )
    segments: List[Dict[str, str]] = []
    src_chunks: List[str] = []
    en_chunks: List[str] = []
    detected = None
    done = False
    saw_speech = False
    cancel_reason = ""
    cancel_kind = None
    
    def on_recognizing(evt):
        nonlocal saw_speech
        if evt and getattr(evt, "result", None) and evt.result.text:
            saw_speech = True
    def on_recognized(evt):
        nonlocal detected, saw_speech
        res = evt.result
        if res.reason == speechsdk.ResultReason.TranslatedSpeech:
            saw_speech = True
            try: detected = res.language
            except Exception: pass
            src = (res.text or "").strip()
            en  = (res.translations.get(TARGET_LANG, "") or "").strip()
            if src: src_chunks.append(src)
            if en:  en_chunks.append(en)
            if src or en: segments.append({"text": src, "translation": en})
            logger.info("Translated chunk: src_len=%d en_len=%d", len(src), len(en))
        elif res.reason == speechsdk.ResultReason.RecognizedSpeech:
            t = (res.text or "").strip()
            if t:
                saw_speech = True
                src_chunks.append(t)
                segments.append({"text": t, "translation": ""})
        elif res.reason == speechsdk.ResultReason.NoMatch:
            logger.warning("NoMatch: speech not recognized.")
            
    def on_canceled(evt):
        nonlocal done, cancel_reason, cancel_kind
        cancel_kind = getattr(evt, "reason", None)
        details = getattr(evt, "error_details", "")
        cancel_reason = f"{cancel_kind} | {details}"
        done = True
        try:
            if cancel_kind == speechsdk.CancellationReason.EndOfStream:
                logger.info("Translation canceled with EndOfStream (normal for file input).")
            else:
                logger.warning("Translation canceled: %s", cancel_reason)
        except Exception:
            logger.warning("Translation canceled: %s", cancel_reason)
            
    # Connect callbacks
    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    
    # Start
    recognizer.start_continuous_recognition_async().get()
    
    dur_s = max(0.0, _wav_duration(path_wav))
    cushion = max(5.0, min(15.0, dur_s * 0.25))
    deadline = time.time() + min(300.0, max(5.0, dur_s + cushion))
    
    while not done and time.time() < deadline:
        time.sleep(0.1)
        
    try:
        fut = recognizer.stop_continuous_recognition_async()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fut.get)
            future.result(timeout=10)
    except concurrent.futures.TimeoutError:
        logger.warning("Stop recognition timed out, forcing exit")
    except Exception as e:
        logger.warning("Failed to stop recognition cleanly: %s", e)
    
    if cancel_kind and cancel_kind != speechsdk.CancellationReason.EndOfStream:
        raise HTTPException(502, f"Speech translation canceled: {cancel_reason}")
        
    # We won't raise 422 here if silence, just return empty so partial results can be used
    if not saw_speech and not (src_chunks or en_chunks):
         # caller can handle empty result
         pass
         
    return {
        "recognized_text": " ".join(src_chunks).strip(),
        "translation_en":  " ".join(en_chunks).strip(),
        "segments": segments,
        "detected_language": detected or ""
    }
