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

import numpy as np
import joblib
import librosa
import tkinter as tk
from tkinter import simpledialog

# Optional audio capture
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False

# Fusion module
from fusion.emotion_fusion import load_all, fuse, fer_labels

# --- Constants ---
DEFAULT_H, DEFAULT_W, DEFAULT_C = 96, 96, 3
AUDIO_FEATURE_DIM = 94  # MFCC(40)+Mel(40)+Chroma(12)+ZCR+Centroid
AUDIO_QUEUE_MAXSIZE = 5

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger('EmotionAwareAI')

# --- Helpers ---
def get_model_input_spec(model) -> Optional[Tuple[int,int,Optional[int]]]:
    if model is None:
        return None
    try:
        shape = getattr(model, 'input_shape', None)
        if shape is None and hasattr(model, 'inputs'):
            shape = tuple(model.inputs[0].shape.as_list())
        if shape is None:
            return None
        if len(shape) == 4:
            return int(shape[1]), int(shape[2]), int(shape[3])
        if len(shape) == 3:
            return int(shape[1]), int(shape[2]), None
    except Exception:
        return None

def preprocess_face_roi(frame: np.ndarray, spec: Optional[Tuple[int,int,Optional[int]]]) -> np.ndarray:
    h, w, c = (DEFAULT_H, DEFAULT_W, DEFAULT_C) if spec is None else spec
    if h is None or w is None:
        h, w = DEFAULT_H, DEFAULT_W
    if c is None:
        c = DEFAULT_C

    resized = cv2.resize(frame, (w,h), interpolation=cv2.INTER_AREA)
    if c == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        arr = np.expand_dims(gray.astype('float32') / 255.0, axis=-1)
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def extract_audio_features(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    try:
        if y is None or len(y) == 0:
            return np.zeros(AUDIO_FEATURE_DIM, dtype=float)
        y = librosa.util.fix_length(y, int(sr*3))
        y, _ = librosa.effects.trim(y)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        mel = np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        return np.concatenate([mfcc, mel, chroma, [zcr, cent]])
    except Exception:
        return np.zeros(AUDIO_FEATURE_DIM, dtype=float)

def safe_predict(model, x):
    try:
        if model is None:
            return None
        out = model.predict(x)
        if out is None:
            return None
        return np.ravel(out)
    except Exception:
        return None

# --- GUI for real-time text input ---
class TextInputGUI(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.text_queue = queue.Queue()
        self.start()

    def run(self):
        self.root = tk.Tk()
        self.root.withdraw()  # hide main window
        while True:
            text = simpledialog.askstring("Text Input", "Enter text (press Cancel to skip):")
            if text is None:
                continue
            self.text_queue.put(text)

    def get_latest_text(self) -> Optional[str]:
        if self.text_queue.empty():
            return None
        try:
            text = self.text_queue.get_nowait()
            # drain the queue to always get latest
            while not self.text_queue.empty():
                text = self.text_queue.get_nowait()
            return text
        except Exception:
            return None

# ----------------  Main runtime  ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--audio-sr', type=int, default=22050)
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("Ignored unknown args: %s", unknown)

    # Load models
    fer_model, speech_model, text_model, vectorizer, speech_le = load_all(verbose=True)
    fer_spec = get_model_input_spec(fer_model)
    logger.info("FER spec: %s", fer_spec)

    # Audio queue
    audio_q: "queue.Queue" = queue.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if not audio_q.full():
            audio_q.put(indata.copy().flatten())

    def audio_thread():
        if not _HAS_SOUNDDEVICE:
            logger.info("sounddevice unavailable, skipping audio")
            return
        try:
            with sd.InputStream(channels=1, samplerate=args.audio_sr, callback=audio_callback):
                while not stop_event.is_set():
                    sd.sleep(200)
        except Exception as ex:
            logger.exception("Audio thread failed: %s", ex)

    if not args.no_audio and _HAS_SOUNDDEVICE:
        threading.Thread(target=audio_thread, daemon=True).start()
    else:
        logger.info("Audio capture disabled")

    # GUI for text
    text_gui = TextInputGUI() if text_model is not None else None

    # Webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Unable to open camera %s", args.camera)
        return

    logger.info("Starting Multimodal Emotion-Aware Assistant. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

            fer_pred = None
            text_pred = None

            # Face prediction
            for (x, y, w, h) in faces:
                roi_input = preprocess_face_roi(frame[y:y+h, x:x+w], fer_spec)
                fer_pred = safe_predict(fer_model, roi_input)
                label_text = fer_labels[int(np.argmax(fer_pred))] if fer_pred is not None else 'NoFER'
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            # Speech prediction
            speech_pred = None
            if speech_model is not None and not audio_q.empty():
                raw = audio_q.get()
                feats = extract_audio_features(raw, sr=args.audio_sr)
                X = feats.reshape(1, -1)
                if getattr(speech_model, 'input_shape', None) and len(speech_model.input_shape) == 3:
                    X = X.reshape(1,1,-1)
                speech_pred = safe_predict(speech_model, X)

            # Text prediction
            if text_model is not None and text_gui is not None:
                text_input = text_gui.get_latest_text()
                if text_input:
                    try:
                        vec = vectorizer.transform([text_input])
                        text_pred = text_model.predict_proba(vec)[0]  # assuming sklearn classifier
                    except Exception:
                        text_pred = None

            # Fusion
            fer_safe = fer_pred if fer_pred is not None else np.zeros(len(fer_labels))
            speech_safe = speech_pred if speech_pred is not None else np.zeros_like(fer_safe)
            text_safe = text_pred if text_pred is not None else np.zeros_like(fer_safe)

            fused_label, combined = fuse(fer_safe, speech_safe, speech_le, text_safe)
            cv2.putText(frame, f"Fused: {fused_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            cv2.imshow("EmotionAwareAI - Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutting down.")

if __name__ == "__main__":
    main()
