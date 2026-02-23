"""
Real-Time Multimodal Emotion-Aware Assistant
Combines Face (FER), Speech, and Text inputs into fused emotion output.
"""

"""
EMOVISTA – Emotion-Aware Medical AI (GUI)
---------------------------------------
• Live FER (Webcam)
• Text emotion
• Multimodal fusion
• Severity engine
• Encrypted patient memory
• Voice TTS feedback
• Emergency escalation
• Trend tracking hook
"""

import os
import warnings

# Suppress TensorFlow logs (must be before importing tf)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import argparse
import logging
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image

# ---------------- PATH FIX ----------------
ROOT_SRC = Path(__file__).resolve().parent
SRC_DIR = ROOT_SRC / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------- CORE IMPORTS ----------------
from src.fusion.emotion_fusion import load_all, fuse, fer_labels

# ---------------- MOCK/STUB MODULES FOR MISSING FILES ----------------
class TrendAnalyzer:
    def __init__(self): pass
    def analyze(self, *args, **kwargs): return {}

class EmergencyEscalation:
    def __init__(self): pass
    def check(self, history):
        return {"escalate": False, "message": ""}

class SeverityEngine:
    def __init__(self): pass
    def evaluate(self, emotion, probabilities):
        # Simple Mock Logic
        score = int(np.max(probabilities) * 100) if probabilities is not None else 0
        return {
            "level": "Normal", 
            "score": score, 
            "response": f"I see you are feeling {emotion}."
        }

class EncryptedPatientMemory:
    def __init__(self, patient_id): pass
    def add_entry(self, emotion, severity, score): pass
    def get_recent(self, n): return []

# ---------------- EMBEDDED WEB BACKEND LOGIC ----------------
import json
import subprocess
import threading
import queue

# Try imports that might be missing in desktop env
try:
    import ollama
except ImportError:
    ollama = None

# --- PERSONA CONFIGURATION ---
PERSONA_CONFIG = {
  "id": "dr_Emovista_v1",
  "name": "Dr. Emovista",
  "voice": {
    "voice_idx": 1,
    "voice_rate": 145,
    "voice_volume": 0.9
  },
  "communication": {
    "tone": ["warm", "casual", "grounded"],  # Adjusted tone
    "style": {
      "sentence_length": "short", # Relaxed from very-short
      "language_level": "simple"
    },
    "response_guidelines": [
      "Speak conversationally.",
      "If the user says they are fine, believe them, even if they look sad.", # New rule
      "Only offer emergency resources if the user EXPLICITLY asks or mentions self-harm." # Stricter trigger
    ]
  },
  "emergency_resources": {
    "us_hotlines": {"suicide_prevention": "988", "emergency": "911"},
    "global_disclaimer": "" # clear default disclaimer to avoid spamming it
  },
  "medical_guidance_rules": [
    "Do not diagnose.",
    "Be a supportive listener first."
  ]
}

def load_system_prompt():
    c = PERSONA_CONFIG
    comm = c.get('communication', {})
    tone = ", ".join(comm.get('tone', []))
    style_cfg = comm.get('style', {})
    
    length_instruction = "Keep responses concise (max 2-3 sentences)."

    style = f"Level: {style_cfg.get('language_level', 'Simple')}. {length_instruction}"
    
    rules = []
    rules.extend(comm.get('response_guidelines', []))
    rules.extend(c.get('medical_guidance_rules', []))
    
    instructions = "\n".join([f"- {r}" for r in rules])
    hotlines = c['emergency_resources']['us_hotlines']
    
    # We hide the hotlines in a separate block and instruct the model only to use them in EXTREME cases
    emergency_text = f"Emergency Hotlines (USE ONLY FOR EXPLICIT SELF-HARM THREATS): Suicide={hotlines.get('suicide_prevention')}, Emergency={hotlines.get('emergency')}"

    return f"""
You are Dr. Emovista, a friendly and supportive medical AI companion.

Tone: {tone}.
Style: {style}

CRITICAL RULES:
{length_instruction}
- Prioritize the user's text over their facial expression. If they say they are okay, accept it.
- {emergency_text}
- Do NOT provide hotlines for general sadness, anxiety, or stress. Only for clear crises.

Instructions:
{instructions}
"""

SYSTEM_PROMPT = load_system_prompt()

class OllamaAssistant:
    def __init__(self, model="gemma3:1b"):
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def respond(self, emotion, user_text):
        if not ollama:
            yield "Ollama library not installed."
            return

        # Simplified prompt to reduce model confusion
        prompt = f"""
[Input Data]
Face Expression: {emotion}
User Text: "{user_text}"

[Task]
Reply naturally to the user text. Use the facial expression only for context (e.g. to be more gentle), but do not explicitly comment on it unless asked.
"""
        self.messages.append({"role": "user", "content": prompt})

        try:
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=True
            )

            full_reply = ""
            for chunk in stream:
                token = chunk["message"]["content"]
                full_reply += token
                yield token

            self.messages.append({"role": "assistant", "content": full_reply})
        except Exception as e:
            yield f"[AI Error: {e}]"

# --- TTS LOGIC (Subprocess Isolation) ---
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None: break
        
        # Embedded script to run in subprocess
        code = f"""
import sys
import pyttsx3
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', {PERSONA_CONFIG['voice']['voice_rate']})
    engine.setProperty('volume', {PERSONA_CONFIG['voice']['voice_volume']})
    
    voices = engine.getProperty('voices')
    # Simple heuristic to find a good voice
    found = False
    for v in voices:
        if "zira" in v.name.lower() or "female" in v.name.lower():
             engine.setProperty('voice', v.id)
             found = True
             break
    if not found and len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
                
    engine.say(sys.argv[1])
    engine.runAndWait()
except Exception as e:
    print(e, file=sys.stderr)
"""
        try:
            subprocess.run([sys.executable, "-c", code, text], capture_output=True)
        except Exception:
            pass
        finally:
            tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    tts_queue.put(text)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EMOVISTA_GUI")

# ---------------- GLOBAL STATE ----------------
current_emotion = "Neutral"
chat_lock = threading.Lock()
running = True

# ---------------- UI SETUP ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1200x800")
app.title("EMOVISTA – Emotion-Aware Medical Assistant")

def on_closing():
    global running
    running = False
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

# ---------------- LOAD MODELS ----------------
logger.info("Loading models...")
fer_model, speech_model, text_model, vectorizer, speech_le = load_all()
logger.info("Models loaded.")

# ---------------- SYSTEM MODULES ----------------
# tts = TTSEngine() # Replaced by web.backend.tts
severity_engine = SeverityEngine()
memory = EncryptedPatientMemory("patient_001")
emergency = EmergencyEscalation()
trend_analyzer = TrendAnalyzer()

if ollama:
    assistant = OllamaAssistant(model="gemma3:1b")
else:
    assistant = None

# ---------------- HELPERS ----------------
def get_fer_spec(model):
    try:
        shape = model.input_shape
        if len(shape) == 4:
            return int(shape[1]), int(shape[2]), int(shape[3])
    except Exception:
        pass
    return 96, 96, 3

FER_H, FER_W, FER_C = get_fer_spec(fer_model)

# ---------------- UI ELEMENTS ----------------
# Main Layout: Left (Video/Status), Right (Chat)
app.grid_columnconfigure(0, weight=1) # Video Area
app.grid_columnconfigure(1, weight=1) # Chat Area
app.grid_rowconfigure(0, weight=1)

# Frame for Video & Stats
video_frame = ctk.CTkFrame(app)
video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

title = ctk.CTkLabel(video_frame, text="Real-Time Monitoring", font=("Arial", 20, "bold"))
title.pack(pady=10)

# Webcam Display Label
video_label = ctk.CTkLabel(video_frame, text="")
video_label.pack(pady=10, padx=10)

result_label = ctk.CTkLabel(video_frame, text="Emotion: Neutral", font=("Arial", 18))
result_label.pack(pady=5)

severity_label = ctk.CTkLabel(video_frame, text="Severity: Normal", font=("Arial", 16))
severity_label.pack(pady=5)

status_label = ctk.CTkLabel(video_frame, text="System idle.", font=("Arial", 14))
status_label.pack(pady=5)

# Frame for Chat
chat_frame = ctk.CTkFrame(app)
chat_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
chat_frame.grid_rowconfigure(0, weight=1)
chat_frame.grid_columnconfigure(0, weight=1)

chat_history = ctk.CTkTextbox(chat_frame, font=("Roboto", 14))
chat_history.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
chat_history.insert("0.0", "EMOVISTA: Hello! I'm listening and watching. How are you feeling?\n\n")

input_frame = ctk.CTkFrame(chat_frame)
input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

text_input = ctk.CTkEntry(input_frame, placeholder_text="Type message...")
text_input.pack(side="left", fill="both", expand=True, padx=5)

send_btn = ctk.CTkButton(input_frame, text="Send", width=60)
send_btn.pack(side="right", padx=5)

# ---------------- CHAT LOGIC ----------------
import re
sentence_endings = re.compile(r'[.!?\n]')

def process_response(user_text, emotion):
    if not assistant:
        return

    full_response = ""
    buffer = ""
    
    # UI: Append User msg
    app.after(0, lambda: chat_history.insert("end", f"You: {user_text}\\n"))
    app.after(0, lambda: chat_history.insert("end", "EMOVISTA: ")) # Start bot msg
    
    try:
        for token in assistant.respond(emotion, user_text):
            full_response += token
            buffer += token
            
            # Stream to UI
            app.after(0, lambda t=token: chat_history.insert("end", t))
            app.after(0, lambda: chat_history.see("end"))
            
            # Speak chunks
            matches = list(sentence_endings.finditer(buffer))
            if matches:
                last_match = matches[-1]
                end_pos = last_match.end()
                
                to_speak = buffer[:end_pos].strip()
                remaining = buffer[end_pos:]
                
                if to_speak:
                    speak(to_speak)
                
                buffer = remaining
        
        # Flush buffer
        if buffer.strip():
            speak(buffer.strip())
            
        app.after(0, lambda: chat_history.insert("end", "\n\n"))
            
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        app.after(0, lambda err=str(e): chat_history.insert("end", f"[Error: {err}]\n\n"))

def on_submit(event=None):
    text = text_input.get().strip()
    if not text: return
    
    text_input.delete(0, "end")
    
    # Run in thread
    threading.Thread(target=process_response, args=(text, current_emotion), daemon=True).start()

text_input.bind("<Return>", on_submit)
send_btn.configure(command=on_submit)

# ---------------- CORE LOOP ----------------
def webcam_loop():
    global current_emotion
    
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot open webcam.")
        return

    previous_faces = []
    previous_labels = []
    frame_count = 0
    SKIP_FRAMES = 5
    DETECTION_SCALE = 0.5
    
    # Pre-allocate Image container to prevent GC churn if possible? 
    # Actually just creating new CTkImage is standard for this lib.

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        
        fer_label = "Neutral" # Default for this frame

        # Only run heavy detection/recognition heavily every SKIP_FRAMES
        if frame_count % SKIP_FRAMES == 0:
            # Optimize detection by resizing
            small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces on small image
            faces_small = cascade.detectMultiScale(gray, 1.3, 5)
            
            current_faces = []
            current_labels = []
            
            fer_pred_arr = None 
            
            for (sx, sy, sw, sh) in faces_small:
                # Scale back up
                x = int(sx / DETECTION_SCALE)
                y = int(sy / DETECTION_SCALE)
                w = int(sw / DETECTION_SCALE)
                h = int(sh / DETECTION_SCALE)
                
                # Extract ROI
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                try:
                    roi_resized = cv2.resize(roi, (FER_W, FER_H))
                    
                    if FER_C == 1:
                        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                        roi_input = roi_gray[..., np.newaxis]
                    else:
                        roi_input = roi_resized
                        
                    roi_input = roi_input.astype("float32") / 255.0
                    roi_input = np.expand_dims(roi_input, axis=0)

                    pred = fer_model.predict(roi_input, verbose=0)[0]
                    label = fer_labels[int(np.argmax(pred))]
                    
                    current_faces.append((x, y, w, h))
                    current_labels.append(label)
                    
                    # Use last face for system emotion state
                    fer_pred_arr = pred
                    fer_label = label
                    
                except Exception as e:
                    logger.error(f"FER Error: {e}")
                    pass

            # Update cache
            previous_faces = current_faces
            previous_labels = current_labels
            
            # -------- FUSION & STATE UPDATE --------
            # Create full vector for fusion
            
            # (Simplified: if no face, we don't really have a vector, but let's persist last known or neutral)
            fer_vector = np.zeros(len(fer_labels))
            if fer_pred_arr is not None:
                fer_vector = fer_pred_arr
            
            fused_label, combined = fuse(
                fer_vector if fer_pred_arr is not None else None,
                None,
                speech_le,
                None
            )
            
            # Update Global
            current_emotion = fused_label
            
            # Severity / Monitoring logic (Still run this in background)
            sev = severity_engine.evaluate(fused_label, combined)
            
            # Update UI Labels
            app.after(0, lambda l=fused_label: result_label.configure(text=f"Emotion: {l}"))
            app.after(0, lambda s=sev: severity_label.configure(text=f"Severity: {s['level']}"))
            
        else:
            # During skipped frames
            pass

        # Draw cached results
        for (x, y, w, h), label in zip(previous_faces, previous_labels):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # ---------------- GUI IMAGE UPDATE ----------------
        try:
            # 1. BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Maintain Aspect Ratio Resizing
            # Target width = 500, calculate height to keep aspect ratio
            h, w, _ = rgb_frame.shape
            target_w = 580 # Slightly larger to fill frame
            aspect_ratio = h / w
            target_h = int(target_w * aspect_ratio)
            
            rgb_frame = cv2.resize(rgb_frame, (target_w, target_h))
            
            # 3. Create PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # 4. Create CTkImage
            ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(target_w, target_h))
            
            # 5. Update Label
            app.after(0, lambda img=ctk_img: video_label.configure(image=img))
            
        except Exception as e:
            logger.error(f"Display Error: {e}")

    cap.release()

# ---------------- THREAD START ----------------
threading.Thread(target=webcam_loop, daemon=True).start()

# ---------------- RUN APP ----------------
app.mainloop()
