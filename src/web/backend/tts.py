import pyttsx3
import json
import os
import queue
import threading
import time

# Load Persona
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "persona_config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"voice_rate": 150, "gender": "female", "voice_idx": 1}

# TTS Queue and Worker
tts_queue = queue.Queue()

def tts_worker():
    # Initialize engine in the worker thread to avoid COM threading issues
    engine = pyttsx3.init()
    
    # Set Rate
    engine.setProperty("rate", config.get("voice_rate", 150))
    
    # Set Voice (Gender Helper)
    voices = engine.getProperty('voices')
    target_gender = config.get("gender", "female").lower()
    target_idx = config.get("voice_idx", 0)
    
    # Try to match index first if explicit, otherwise heuristic
    if 0 <= target_idx < len(voices):
        engine.setProperty('voice', voices[target_idx].id)
    else:
        # Heuristic fallback
        found = False
        for v in voices:
            if target_gender in v.name.lower():
                engine.setProperty('voice', v.id)
                found = True
                break
        if not found and len(voices) > 1 and target_gender == "female":
             # Often index 1 is female on Windows/SAPI5
            engine.setProperty('voice', voices[1].id)

    # Processing Loop
    while True:
        try:
            text = tts_queue.get()
            if text is None: # Sentinel to stop
                break
            
            engine.say(text)
            engine.runAndWait()
            tts_queue.task_done()
        except Exception as e:
            print(f"TTS Worker Error: {e}")

# Start Daemon Thread
worker_thread = threading.Thread(target=tts_worker, daemon=True)
worker_thread.start()

def speak(text):
    """Adds text to the TTS queue to be spoken asynchronously."""
    tts_queue.put(text)
