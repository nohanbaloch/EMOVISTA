from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import sys

# Suppress TensorFlow logs (must be before importing tf)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
import cv2
import numpy as np
import logging
from pathlib import Path

# --- Path Setup for Fusion ---
# Current file: src/web/backend/app.py
# We need 'src' in sys.path to import 'fusion'
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR_PATH = CURRENT_FILE.parents[2] # src/web/backend -> src/web -> src
PROJECT_ROOT = CURRENT_FILE.parents[3] # src -> Project Root

if str(SRC_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_DIR_PATH))

try:
    from fusion.emotion_fusion import load_all, fuse, fer_labels
except ImportError as e:
    print(f"Error importing fusion: {e}. Make sure 'src' is in python path.")
    # Fallback/Warnings will happen later if modules missing
    pass

# Assuming these modules are available in your environment
from llm_phi3 import Phi3Assistant
from stt_vosk import VoskSTT
from tts import speak

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('web_backend')

# Calculate paths relative to this file
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(WEB_DIR, 'frontend')
STATIC_DIR = os.path.join(FRONTEND_DIR, 'static')
TEMPLATE_PATH = os.path.join(FRONTEND_DIR, 'templates', 'index.html')

MODELS_DIR = os.path.join(str(PROJECT_ROOT), 'models')
VOSK_MODEL_PATH = os.path.join(MODELS_DIR, 'vosk', 'vosk-model-en-us-0.22')

logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Models Dir: {MODELS_DIR}")
logger.info(f"Vosk Model Path: {VOSK_MODEL_PATH}")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')
CORS(app)

assistant = Phi3Assistant()

# STT Init
try:
    if not os.path.exists(VOSK_MODEL_PATH):
        logger.error(f"Vosk Model path does not exist: {VOSK_MODEL_PATH}")
        stt = None
    else:
        stt = VoskSTT(VOSK_MODEL_PATH)
        logger.info("STT Service Initialized")
except Exception as e:
    logger.error(f"STT Init Error: {e}")
    stt = None

# --- Fusion Model Loading ---
logger.info("Loading Fusion Models...")
fer_model, speech_model, text_model, vectorizer, speech_le = load_all()
if fer_model:
    logger.info("FER Model Loaded.")
else:
    logger.warning("FER Model NOT Loaded.")

# Helper: Determine FER input spec
def _get_fer_input_spec(model):
    if model is None:
        return None
    try:
        shape = getattr(model, 'input_shape', None)
        if shape is None and hasattr(model, 'inputs'):
            shape = tuple(model.inputs[0].shape.as_list())
        if shape is None:
            return None
        # shape typically (None, H, W, C)
        if len(shape) == 4:
            return (int(shape[1]), int(shape[2]), int(shape[3]))
        if len(shape) == 3:
            return (int(shape[1]), int(shape[2]), None)
    except Exception as ex:
        logger.warning('Failed to read FER model input spec: %s', ex)
    return None

fer_input_spec = _get_fer_input_spec(fer_model)
logger.info(f"FER Input Spec: {fer_input_spec}")

@app.route("/")
def index():
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Template not found at {TEMPLATE_PATH}"

@app.route("/stt", methods=["POST"])
def speech_to_text():
    if stt:
        text = stt.record_and_transcribe()
    else:
        text = "STT unavailable"
    return jsonify({"text": text})

@app.route("/analyze_face", methods=["POST"])
def analyze_face():
    """Receives a base64 encoded image frame and returns the detected emotion."""
    try:
        data = request.json
        image_data = data.get("image")
        
        if not image_data:
            return jsonify({"error": "No image data"}), 400

        # Decode base64
        if "," in image_data:
            header, encoded = image_data.split(",", 1)
        else:
            encoded = image_data

        decoded_data = base64.b64decode(encoded)
        np_data = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if fer_model is None:
            return jsonify({"emotion": "Model Not Loaded"})

        # Face Detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        fer_pred = None
        label_text = "Neutral"

        # Process first face found (simplification for web)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_color = frame[y:y+h, x:x+w]
            
            # Preprocess (Resize & Normalize)
            spec = fer_input_spec
            if spec is None:
                target_h, target_w, target_c = 96, 96, 3
            else:
                target_h, target_w, target_c = spec
                if target_h is None: target_h, target_w = 96, 96
            
            roi_resized = cv2.resize(roi_color, (target_w, target_h))

            if target_c is None or target_c == 1:
                # Grayscale path
                roi_proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                roi_proc = roi_proc.astype('float32') / 255.0
                
                # Expand dims based on model expectation
                ishape = getattr(fer_model, 'input_shape', None)
                if ishape is None and hasattr(fer_model, 'inputs'):
                     try: ishape = tuple(fer_model.inputs[0].shape.as_list())
                     except: ishape = None
                
                if ishape is not None and len(ishape) == 4:
                    roi_input = np.expand_dims(roi_proc, axis=(0, -1))
                else:
                    roi_input = np.expand_dims(roi_proc, axis=0)
            else:
                # RGB path
                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                roi_proc = roi_rgb.astype('float32') / 255.0
                roi_input = np.expand_dims(roi_proc, axis=0)

            # Predict
            fer_pred = fer_model.predict(roi_input, verbose=0)[0]
        
        # Fuse (Currently only FER provided from this endpoint)
        # Note: 'fuse' handles None for other modalities
        fused_label, combined_scores = fuse(fer_pred, None, speech_le, None)
        
        return jsonify({"emotion": fused_label})

    except Exception as e:
        logger.error(f"Face Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/consult", methods=["POST"])
def consult():
    data = request.json
    emotion = data.get("emotion", "Neutral") 
    text = data.get("text", "")

    # Text Emotion Analysis
    text_pred = None
    if text_model and vectorizer and text:
        try:
            vec = vectorizer.transform([text])
            text_pred = text_model.predict_proba(vec)[0] # Shape (3,) [Neg, Neu, Pos]
        except Exception as e:
            logger.error(f"Text Model Error: {e}")

    # Convert frontend emotion (String) to Probability Vector for Fusion
    # fer_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    fer_pred_vector = np.zeros(7) # 7 classes
    
    # Map the single label to a 1.0 probability
    if emotion in fer_labels:
        idx = fer_labels.index(emotion)
        fer_pred_vector[idx] = 1.0
    else:
        # Fallback if label unknown or "Neutral" default
        if "Neutral" in fer_labels:
            fer_pred_vector[fer_labels.index("Neutral")] = 1.0
    
    # Fuse (Speech component is None here as we only have Text + Face)
    fused_label, combined_scores = fuse(fer_pred_vector, None, speech_le, text_pred)
    
    logger.info(f"Frontend Emotion: {emotion} | Text: {text} | Fused: {fused_label}")

    def generate():
        for token in assistant.respond(fused_label, text):
            yield token

    return Response(generate(), mimetype="text/plain")

@app.route("/tts", methods=["POST"])
def tts_route():
    speak(request.json.get("text", ""))
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Speak only if this is the reloader worker (WERKZEUG_RUN_MAIN=true) or if debug is off
    # If debug=True, Flask spawns a child process. WERKZEUG_RUN_MAIN is set in the child.
    # We want to speak once.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
         try:
            speak("Emovista system online and ready.")
         except Exception as e:
            logger.error(f"Startup Speech Error: {e}")

    app.run(debug=True, port=5000)
