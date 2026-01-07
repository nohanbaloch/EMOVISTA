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

import argparse
import logging
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple
import sys
import threading
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import customtkinter as ctk
from tkinter import messagebox

# ---------------- PATH FIX ----------------
ROOT_SRC = Path(__file__).resolve().parent
SRC_DIR = ROOT_SRC / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------- CORE IMPORTS ----------------
from src.fusion.emotion_fusion import load_all, fuse, fer_labels
from src.analytics.trends import TrendAnalyzer
from src.safety.emergency import EmergencyEscalation
from src.memory.patient_memory import EncryptedPatientMemory
from src.severity.severity_engine import SeverityEngine

# Web-Feature Ports
try:
    from src.web.backend.llm_phi3 import Phi3Assistant
    from src.web.backend.tts import speak
    logger = logging.getLogger("EMOVISTA_GUI") # Define logger before usage in except block if needed
    logger.info("Web backend modules loaded successfully.")
except ImportError as e:
    logger = logging.getLogger("EMOVISTA_GUI")
    logger.error(f"Failed to import Web Backend modules: {e}")
    Phi3Assistant = None
    speak = lambda x: print(f"TTS (Mock): {x}")

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- GLOBAL STATE ----------------
current_emotion = "Neutral"
chat_lock = threading.Lock()

# ---------------- UI SETUP ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1000x800")
app.title("EMOVISTA – Emotion-Aware Medical Assistant")

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

if Phi3Assistant:
    assistant = Phi3Assistant()
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
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# Frame for Video & Stats
video_frame = ctk.CTkFrame(app)
video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

title = ctk.CTkLabel(video_frame, text="Real-Time Monitoring", font=("Arial", 20, "bold"))
title.pack(pady=10)

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
            
        app.after(0, lambda: chat_history.insert("end", "\\n\\n"))
            
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        app.after(0, lambda: chat_history.insert("end", f"[Error: {e}]\\n\\n"))

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

    while True:
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
            elif previous_labels:
                 # If we are using cached labels, we should probably construct a vector
                 # But for now, let's just stick to the fresh detection or skip
                 pass
            
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
            
            # Update UI
            app.after(0, lambda l=fused_label: result_label.configure(text=f"Emotion: {l}"))
            app.after(0, lambda s=sev: severity_label.configure(text=f"Severity: {s['level']}"))
            
            # Emergency Check
            # memory.add_entry... (Optional based on requirements)
            
        else:
            # During skipped frames, just use the last global current_emotion or cached labels
            pass

        # Draw cached results
        for (x, y, w, h), label in zip(previous_faces, previous_labels):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("EMOVISTA – Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    app.quit()

# ---------------- THREAD START ----------------
threading.Thread(target=webcam_loop, daemon=True).start()

# ---------------- RUN APP ----------------
app.mainloop()
