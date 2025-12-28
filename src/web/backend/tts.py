import pyttsx3
import json
import os

# Load Persona
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "persona_config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"voice_rate": 150, "gender": "female", "voice_idx": 1}

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

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        # Engine might be busy
        pass
