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
ROOT_SRC = Path(__file__).resolve().parents[1]
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

# ---------------- CORE IMPORTS ----------------
from fusion.emotion_fusion import load_all, fuse, fer_labels

from voice.tts_engine import TTSEngine
from analytics.trends import TrendAnalyzer
from safety.emergency import EmergencyEscalation
from memory.patient_memory import EncryptedPatientMemory
from severity.severity_engine import SeverityEngine

# ---------------- LOGGING ----------------
logger = logging.getLogger("EMOVISTA_GUI")
logging.basicConfig(level=logging.INFO)

# ---------------- UI SETUP ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1000x720")
app.title("EMOVISTA – Emotion-Aware Medical Assistant")

# ---------------- LOAD MODELS ----------------
logger.info("Loading models...")
fer_model, speech_model, text_model, vectorizer, speech_le = load_all()
logger.info("Models loaded.")

# ---------------- SYSTEM MODULES ----------------
tts = TTSEngine()
severity_engine = SeverityEngine()
memory = EncryptedPatientMemory("patient_001")
emergency = EmergencyEscalation()
trend_analyzer = TrendAnalyzer()

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
title = ctk.CTkLabel(app, text="Real-Time Emotion Monitoring", font=("Arial", 22, "bold"))
title.pack(pady=10)

text_input = ctk.CTkEntry(app, placeholder_text="Optional: type how you feel...")
text_input.pack(fill="x", padx=20, pady=8)

result_label = ctk.CTkLabel(app, text="", font=("Arial", 18))
result_label.pack(pady=8)

severity_label = ctk.CTkLabel(app, text="", font=("Arial", 16))
severity_label.pack(pady=6)

status_label = ctk.CTkLabel(app, text="System idle.", font=("Arial", 14))
status_label.pack(pady=6)

# ---------------- CORE LOOP ----------------
def webcam_loop():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        fer_pred = None

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (FER_W, FER_H))

            if FER_C == 1:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = roi[..., np.newaxis]

            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)

            fer_pred = fer_model.predict(roi, verbose=0)[0]
            label = fer_labels[int(np.argmax(fer_pred))]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # -------- TEXT --------
        text_pred = None
        text_val = text_input.get().strip()
        if text_model and vectorizer and text_val:
            try:
                vec = vectorizer.transform([text_val])
                text_pred = text_model.predict_proba(vec)[0]
            except Exception:
                text_pred = None

        # -------- FUSION --------
        fused_label, combined = fuse(
            fer_pred,
            None,
            speech_le,
            text_pred
        )

        # -------- SEVERITY --------
        sev = severity_engine.evaluate(fused_label, combined)

        # -------- MEMORY --------
        memory.add_entry(
            emotion=fused_label,
            severity=sev["level"],
            score=sev["score"]
        )

        # -------- EMERGENCY --------
        alert = emergency.check(memory.get_recent(5))

        # -------- UI UPDATE --------
        app.after(0, lambda: result_label.configure(
            text=f"Detected Emotion: {fused_label}"
        ))

        app.after(0, lambda: severity_label.configure(
            text=f"Severity: {sev['level']} ({sev['score']}/100)"
        ))

        if alert["escalate"]:
            tts.speak(alert["message"])
            app.after(0, lambda: status_label.configure(
                text="⚠ Emergency escalation triggered."
            ))
        else:
            tts.speak(sev["response"])
            app.after(0, lambda: status_label.configure(
                text="Monitoring normally."
            ))

        cv2.imshow("EMOVISTA – Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- THREAD START ----------------
threading.Thread(target=webcam_loop, daemon=True).start()

# ---------------- RUN APP ----------------
app.mainloop()
