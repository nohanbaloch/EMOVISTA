
import customtkinter as ctk
from tkinter import messagebox
import threading, queue, os, time, cv2, numpy as np
from pathlib import Path
import sys
import logging

# Ensure the project `src/` directory is on sys.path so imports like
# `from fusion.emotion_fusion import ...` work when running this script
# directly (e.g., `python src/gui/main_gui.py`).
ROOT_SRC = Path(__file__).resolve().parents[1]
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from fusion.emotion_fusion import load_all, fuse, fer_labels
from tensorflow.keras.models import load_model
import joblib, librosa
from cv2 import data as cv2_data

# --- Logging setup ---
logger = logging.getLogger('emotion_gui')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Load models (non-blocking)
fer_model, speech_model, text_model, vectorizer, speech_le = load_all()

# Determine FER model expected input shape (height, width, channels or None for grayscale/no-channel)
def _get_fer_input_spec(model):
    if model is None:
        return None
    try:
        # Prefer model.input_shape if available
        shape = getattr(model, 'input_shape', None)
        if shape is None and hasattr(model, 'inputs'):
            shape = tuple(model.inputs[0].shape.as_list())
        if shape is None:
            return None
        # shape typically like (None, H, W, C) or (None, H, W)
        if len(shape) == 4:
            return (int(shape[1]), int(shape[2]), int(shape[3]))
        if len(shape) == 3:
            return (int(shape[1]), int(shape[2]), None)
    except Exception as ex:
        logger.warning('Failed to read FER model input shape: %s', ex)
    return None

fer_input_spec = _get_fer_input_spec(fer_model)
logger.info('FER input spec: %s', fer_input_spec)

app = ctk.CTk()
app.geometry("900x700")
app.title("Emotion-Aware AI Assistant - GUI")

text_input = ctk.CTkEntry(app, placeholder_text="Enter text here...")
text_input.pack(pady=10, padx=10, fill='x')

result_label = ctk.CTkLabel(app, text="", font=("Arial", 18))
result_label.pack(pady=8)

# Debug toggle (simple button that toggles debug mode)
debug_mode = {'enabled': False}
def _toggle_debug():
    debug_mode['enabled'] = not debug_mode['enabled']
    debug_button.configure(text=("Disable Debug" if debug_mode['enabled'] else "Enable Debug"))

debug_button = ctk.CTkButton(app, text="Enable Debug", command=_toggle_debug)
debug_button.pack(pady=6)

# Webcam preview frame using OpenCV
cv2_frame_label = ctk.CTkLabel(app, text="Webcam preview will open in a separate window.")
cv2_frame_label.pack(pady=6)

def run_realtime():
    # If models are not loaded, warn
    if fer_model is None:
        messagebox.showwarning("Model missing", "FER model not found in models/. Train or place fer_model.keras first.")
        return
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # audio capture omitted in GUI thread for simplicity; use main.py for full realtime with audio
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        fer_pred = None
        for (x,y,w,h) in faces:
            # Extract color ROI from the original frame and preprocess according to model spec.
            try:
                roi_color = frame[y:y+h, x:x+w]
                spec = fer_input_spec
                if spec is None:
                    # default to RGB 96x96
                    target_h, target_w, target_c = 96, 96, 3
                else:
                    target_h, target_w, target_c = spec
                    # if channels is None, treat as grayscale expected
                    if target_h is None:
                        target_h, target_w = 96, 96
                # resize first
                roi_resized = cv2.resize(roi_color, (target_w, target_h))

                if target_c is None or target_c == 1:
                    # convert to grayscale
                    roi_proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                    roi_proc = roi_proc.astype('float32') / 255.0
                    # if model expects shape (None, H, W, 1) expand channel dim
                    # detect whether model expects 3- or 4-D input by checking fer_model.input_shape len
                    ishape = getattr(fer_model, 'input_shape', None)
                    if ishape is None and hasattr(fer_model, 'inputs'):
                        try:
                            ishape = tuple(fer_model.inputs[0].shape.as_list())
                        except Exception:
                            ishape = None
                    if ishape is not None and len(ishape) == 4:
                        roi_input = np.expand_dims(roi_proc, axis=(0, -1))
                    else:
                        roi_input = np.expand_dims(roi_proc, axis=0)
                else:
                    # convert BGR->RGB and normalize
                    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                    roi_proc = roi_rgb.astype('float32') / 255.0
                    roi_input = np.expand_dims(roi_proc, axis=0)

                fer_pred = None
                if fer_model is not None:
                    fer_pred = fer_model.predict(roi_input)[0]
                    label_idx = int(np.argmax(fer_pred)) if fer_pred is not None else 0
                    label_text = fer_labels[label_idx]
                else:
                    label_text = 'NoFER'
            except Exception:
                logger.exception('FER preprocessing/prediction failed for ROI at %s,%s,%s,%s', x, y, w, h)
                fer_pred = None
                label_text = 'Error'

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # text infer
        text_val = text_input.get() or ""
        text_pred = None
        if text_model is not None and vectorizer is not None and text_val.strip():
            try:
                text_tfidf = vectorizer.transform([text_val])
                # LogisticRegression.predict_proba() returns probabilities for each class
                text_proba = text_model.predict_proba(text_tfidf)[0]
                # map to [negative, neutral, positive] if needed
                text_pred = text_proba
            except Exception as e:
                logger.exception('Text model prediction failed')
                # fallback: if predict_proba fails, try to predict class and convert to one-hot
                try:
                    pred = text_model.predict(text_tfidf)
                    # build one-hot vector based on classes_
                    classes = getattr(text_model, 'classes_', None)
                    if classes is not None and pred is not None:
                        vec = np.zeros(len(classes), dtype=float)
                        try:
                            idx = list(classes).index(pred[0])
                            vec[idx] = 1.0
                            text_pred = vec
                        except Exception:
                            text_pred = None
                    else:
                        text_pred = None
                except Exception:
                    text_pred = None
        # speech omitted here
        fused_label, combined = fuse(fer_pred, None, speech_le, text_pred)
        if debug_mode.get('enabled'):
            # show raw inputs when debugging
            result_label.configure(text=(f"DEBUG -> FER:{np.round(fer_pred,3) if fer_pred is not None else None} "
                                        f"TEXT:{np.round(text_pred,3) if text_pred is not None else None} "
                                        f"FUSED:{fused_label} {np.round(combined,3)}"))
        else:
            result_label.configure(text=f"Detected Emotion: {fused_label}  (scores: {np.round(combined,3)})")
        cv2.imshow("Webcam - EmotionAwareAI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

threading.Thread(target=run_realtime, daemon=True).start()
app.mainloop()
