import sys
import os
import json
import pyttsx3
import logging

# Basic logging setup to stderr (captured by parent)
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "persona_config.json")

def load_voice_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            return config.get("voice", {})
    except Exception as e:
        logging.warning(f"Could not load persona_config.json: {e}")
        return {}

def speak(text):
    try:
        engine = pyttsx3.init()
        
        # Load config
        voice_config = load_voice_config()
        
        # Apply settings
        engine.setProperty('rate', voice_config.get("voice_rate", 145))
        engine.setProperty('volume', voice_config.get("voice_volume", 0.9))
        
        # Voice Selection
        voices = engine.getProperty('voices')
        idx = voice_config.get("voice_idx", 1)
        
        if isinstance(idx, int) and 0 <= idx < len(voices):
            engine.setProperty('voice', voices[idx].id)
        else:
            # Fallback
            for v in voices:
                if "female" in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_to_speak = sys.argv[1]
        speak(text_to_speak)
    else:
        print("Usage: python independent_tts.py 'Text to speak'")
