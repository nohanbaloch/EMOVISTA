import os
import queue
import threading
import subprocess
import sys

# Load Persona (Optional, if we want to pass config later)
# For now, relying on defaults in independent script.

# TTS Queue and Worker
tts_queue = queue.Queue()

TTS_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "independent_tts.py")

def tts_worker():
    print("[TTS] Subprocess Worker thread starting...")
    
    while True:
        try:
            text = tts_queue.get()
            if text is None: # Sentinel
                break
            
            print(f"[TTS] Spawning subprocess for: {text[:30]}...")
            
            # Run independent python script to speak
            # This isolates the crashy COM stuff from our persistent server process
            result = subprocess.run(
                [sys.executable, TTS_SCRIPT_PATH, text],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[TTS] Subprocess Error: {result.stderr}")
            else:
                print(f"[TTS] Finished: {text[:10]}...")
                
            tts_queue.task_done()
            
        except Exception as e:
            print(f"[TTS] Worker Loop Error: {e}")

# Start Daemon Thread
worker_thread = threading.Thread(target=tts_worker, daemon=True)
worker_thread.start()

def speak(text):
    """Adds text to the TTS queue to be spoken asynchronously."""
    tts_queue.put(text)
