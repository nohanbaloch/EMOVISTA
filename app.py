"""
app.py

EMOVISTA Web Backend & API Server.
Flask server providing multimodal emotion analysis, offline STT, streaming TTS,
and local LLM consultation.

Security and Hardening:
- Flask debug mode OFF by default (enabled only via FLASK_DEBUG=1).
- CORS restricted to explicit local origins (configurable via EMOVISTA_ALLOWED_ORIGINS).
- Strict validation, size limits, and safe base64 decoding on /analyze_face.
- Safe payload parsing and length limits on /tts to prevent crashes.
- Deterministic crisis fallback on /consult.
- Centralized configuration with environment variable overrides.
"""

import os
import sys
import base64
import binascii
import logging
from pathlib import Path
from typing import Any, Callable, Sequence, cast

# Suppress TensorFlow logs before import
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'torch'

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# --- Centralized Configuration ---
class Config:
    CURRENT_FILE = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE.parent
    SRC_DIR_PATH = PROJECT_ROOT / 'src'
    HOST = os.getenv("EMOVISTA_HOST", "127.0.0.1")
    PORT = int(os.getenv("EMOVISTA_PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "t")
    ALLOWED_ORIGINS = [
        o.strip()
        for o in os.getenv("EMOVISTA_ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",")
        if o.strip()
    ]
    MAX_IMAGE_SIZE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(5 * 1024 * 1024)))  # 5 MB
    MAX_TTS_TEXT_LEN = int(os.getenv("MAX_TTS_TEXT_LEN", "1000"))
    MODELS_DIR = Path(os.getenv("EMOVISTA_MODELS_DIR", str(PROJECT_ROOT / "models")))
    VOSK_MODEL_PATH = Path(os.getenv(
        "VOSK_MODEL_PATH",
        str(MODELS_DIR / "vosk" / "vosk-model-small-en-us-0.15")
    ))


# Ensure src is in sys.path
if str(Config.SRC_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(Config.SRC_DIR_PATH))

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger('web_backend')

# Default emotion labels fallback
fer_labels: list[str] = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def _fallback_load_all() -> tuple[Any, Any, Any, Any, Any]:
    return (None, None, None, None, None)


def _fallback_fuse(
    fer_pred: Any,
    speech_pred: Any,
    speech_encoder: Any,
    text_pred: Any,
) -> tuple[str, np.ndarray[Any, Any]]:
    scores = np.zeros(len(fer_labels), dtype=np.float32)
    neutral_index = fer_labels.index('Neutral') if 'Neutral' in fer_labels else 6
    scores[neutral_index] = 1.0
    return ('Neutral', scores)


load_all: Callable[[], tuple[Any, Any, Any, Any, Any]] = _fallback_load_all
fuse: Callable[[Any, Any, Any, Any], tuple[str, np.ndarray[Any, Any]]] = _fallback_fuse

try:
    import src.fusion.emotion_fusion as fusion_module
    load_all = cast(Callable[[], tuple[Any, Any, Any, Any, Any]], getattr(fusion_module, 'load_all'))
    fuse = cast(Callable[[Any, Any, Any, Any], tuple[str, np.ndarray[Any, Any]]], getattr(fusion_module, 'fuse'))
    fer_labels = list(cast(Sequence[str], getattr(fusion_module, 'fer_labels')))
except Exception as e:
    logger.warning("Fusion import failed, using fallback behavior: %s", e)

from src.web.backend.llm_phi3 import (
    Phi3Assistant,
    list_ollama_models,
    is_crisis_text,
    DETERMINISTIC_CRISIS_RESPONSE
)
try:
    from src.web.backend.stt_vosk import VoskSTT
except Exception as e:
    logger.warning(f"VoskSTT module could not be imported: {e}")
    VoskSTT = None

import src.web.backend.tts as tts_module
from src.fusion.severity_engine import SeverityEngine

speak: Callable[[str], None] = cast(Callable[[str], None], getattr(tts_module, 'speak'))
severity_engine = SeverityEngine()

# Paths for web assets
FRONTEND_DIR = Config.SRC_DIR_PATH / 'web' / 'frontend'
STATIC_DIR = str(FRONTEND_DIR / 'static')
TEMPLATE_PATH = FRONTEND_DIR / 'templates' / 'index.html'

# Flask App Initialization
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_IMAGE_SIZE_BYTES

# Restrict CORS to explicit local origins
CORS(app, origins=Config.ALLOWED_ORIGINS)

logger.info(f"Project Root: {Config.PROJECT_ROOT}")
logger.info(f"Vosk Model Path: {Config.VOSK_MODEL_PATH}")
logger.info(f"CORS Allowed Origins: {Config.ALLOWED_ORIGINS}")
logger.info(f"Flask Debug Mode: {Config.DEBUG}")

assistant = Phi3Assistant()

# STT Initialization
stt = None
if VoskSTT is not None:
    try:
        if not Config.VOSK_MODEL_PATH.exists():
            logger.warning(f"Vosk model path does not exist: {Config.VOSK_MODEL_PATH}")
        else:
            stt = VoskSTT(str(Config.VOSK_MODEL_PATH))
            logger.info("STT Service Initialized successfully.")
    except Exception as e:
        logger.warning(f"STT Initialization skipped or failed: {e}")
        stt = None

# Load Fusion Models
logger.info("Loading Fusion Models...")
fer_model, speech_model, text_model, vectorizer, speech_le = load_all()
if fer_model:
    logger.info("FER Model Loaded.")
else:
    logger.warning("FER Model NOT Loaded.")


def _get_fer_input_spec(model: Any) -> tuple[int, int, int | None]:
    """Inspect model input shape, defaulting to 100x100 grayscale."""
    if model is None:
        return (100, 100, 1)
    try:
        shape = getattr(model, 'input_shape', None)
        if shape is None and hasattr(model, 'inputs'):
            shape = tuple(model.inputs[0].shape.as_list())
        if shape and len(shape) == 4:
            c = int(shape[3]) if shape[3] is not None else 1
            return (int(shape[1]), int(shape[2]), c)
    except Exception as ex:
        logger.warning('Failed to read FER model input spec: %s', ex)
    return (100, 100, 1)


fer_input_spec = _get_fer_input_spec(fer_model)
logger.info(f"FER Input Spec: {fer_input_spec}")


@app.route("/")
def index():
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Template not found at {TEMPLATE_PATH}", 404


@app.route("/stt", methods=["POST"])
def speech_to_text():
    """Vosk Speech-to-text route with explicit error reporting."""
    if stt is None:
        return jsonify({
            "error": "Speech-to-text service is currently unavailable.",
            "text": "",
            "details": f"Vosk model not found at '{Config.VOSK_MODEL_PATH}' or audio input unavailable."
        }), 503

    try:
        text = stt.record_and_transcribe()
        return jsonify({"text": text or ""})
    except Exception as e:
        logger.error(f"STT recording error: {e}")
        return jsonify({
            "error": "Failed to capture or transcribe audio.",
            "text": "",
            "details": str(e)
        }), 500


@app.route("/ollama_models", methods=["GET"])
def ollama_models():
    try:
        models = list_ollama_models()
        return jsonify({"models": models, "default": assistant.model})
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return jsonify({"models": [assistant.model], "default": assistant.model})


@app.route("/analyze_face", methods=["POST"])
def analyze_face():
    """Receives a base64 encoded image frame, detects face, and returns detected emotion."""
    try:
        payload_raw = request.get_json(silent=True)
        if not isinstance(payload_raw, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        image_data = payload_raw.get("image")
        if not image_data or not isinstance(image_data, str):
            return jsonify({"error": "Missing or invalid 'image' parameter"}), 400

        # Enforce size limit on base64 string
        if len(image_data) > Config.MAX_IMAGE_SIZE_BYTES * 2:
            return jsonify({"error": "Image payload exceeds maximum allowed size"}), 413

        # Strip Data URL prefix if present
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data

        try:
            decoded_data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as b64_err:
            return jsonify({"error": f"Invalid base64 image encoding: {b64_err}"}), 400

        if not decoded_data:
            return jsonify({"error": "Decoded image data is empty"}), 400

        np_data = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return jsonify({"error": "Unable to decode image frame"}), 400

        if fer_model is None:
            return jsonify({"emotion": "Model Not Loaded", "status": "unavailable"})

        # Face Detection via OpenCV Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        fer_pred = None

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_color = frame[y:y+h, x:x+w]
            target_h, target_w, target_c = fer_input_spec

            roi_resized = cv2.resize(roi_color, (target_w, target_h))

            if target_c == 1:
                roi_proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                roi_proc = roi_proc.astype('float32') / 255.0
                roi_input = np.expand_dims(roi_proc, axis=(0, -1))
            else:
                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                roi_proc = roi_rgb.astype('float32') / 255.0
                roi_input = np.expand_dims(roi_proc, axis=0)

            # Predict emotion probabilities
            pred = fer_model.predict(roi_input, verbose=0)
            fer_pred = pred[0]

        # Fuse (Visual only from this endpoint)
        fused_label, _ = fuse(fer_pred, None, speech_le, None)
        return jsonify({"emotion": fused_label})

    except Exception as e:
        logger.error(f"Face Analysis Error: {e}", exc_info=True)
        return jsonify({"error": "Internal face analysis error"}), 500


@app.route("/consult", methods=["POST"])
def consult():
    """Receives user text and detected emotion, yielding streaming conversational tokens."""
    payload_raw = request.get_json(silent=True)
    data: dict[str, Any] = cast(dict[str, Any], payload_raw) if isinstance(payload_raw, dict) else {}
    emotion = str(data.get("emotion", "Neutral")).strip()
    text = str(data.get("text", "")).strip()
    model_name = str(data.get("model", assistant.model)).strip() or assistant.model

    # 1. Deterministic safety check for crisis indicators
    if is_crisis_text(text):
        logger.warning(f"Deterministic crisis keyword detected in user input.")

        def generate_crisis():
            speak(DETERMINISTIC_CRISIS_RESPONSE)
            yield DETERMINISTIC_CRISIS_RESPONSE

        return Response(generate_crisis(), mimetype="text/plain")

    # 2. Text Emotion Analysis (Binary mapped to 3-way distribution)
    text_pred = None
    if text_model and vectorizer and text:
        try:
            if hasattr(vectorizer, 'transform'):
                vec = vectorizer.transform([text])
            elif hasattr(vectorizer, 'texts_to_sequences'):
                from keras.preprocessing.sequence import pad_sequences
                seq = vectorizer.texts_to_sequences([text])
                vec = pad_sequences(seq, maxlen=100)
            else:
                vec = None

            if vec is not None:
                if hasattr(text_model, 'predict_proba'):
                    text_pred = text_model.predict_proba(vec)[0]
                elif hasattr(text_model, 'predict'):
                    pred = text_model.predict(vec, verbose=0)
                    text_pred = pred[0]
        except Exception as e:
            logger.warning(f"Text Model inference failed: {e}")

    # 3. Create FER probability vector from frontend emotion
    fer_pred_vector = np.zeros(len(fer_labels), dtype=float)
    if emotion in fer_labels:
        fer_pred_vector[fer_labels.index(emotion)] = 1.0
    else:
        neutral_idx = fer_labels.index("Neutral") if "Neutral" in fer_labels else 6
        fer_pred_vector[neutral_idx] = 1.0

    # 4. Multimodal Fusion (FER + Text; speech is handled via /stt text)
    fused_label, combined_scores = fuse(fer_pred_vector, None, speech_le, text_pred)

    # 5. Non-diagnostic persistence heuristic logging
    try:
        severity_engine.evaluate(fused_label, combined_scores)
    except Exception:
        pass

    logger.info(f"Consult input - Emotion: {emotion} | Text: '{text[:40]}' | Fused: {fused_label}")

    def generate():
        import re
        buffer = ""
        sentence_endings = re.compile(r'[.!?\n]')

        respond_fn = cast(Callable[[str, str, str], Any], getattr(assistant, 'respond'))
        for token in respond_fn(fused_label, text, model_name):
            token_str = str(token)
            buffer += token_str
            yield token_str

            # Stream spoken sentence chunks safely
            if sentence_endings.search(buffer):
                matches = list(sentence_endings.finditer(buffer))
                if matches:
                    last_match = matches[-1]
                    end_pos = last_match.end()
                    to_speak = buffer[:end_pos].strip()
                    remaining = buffer[end_pos:]

                    if to_speak:
                        speak(to_speak)
                    buffer = remaining

        if buffer.strip():
            speak(buffer.strip())

    return Response(generate(), mimetype="text/plain")


@app.route("/tts", methods=["POST"])
def tts_route():
    """Queues text to be spoken asynchronously via isolated TTS subprocess."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing or empty 'text' field"}), 400

    if len(text) > Config.MAX_TTS_TEXT_LEN:
        return jsonify({
            "error": f"Text exceeds maximum permitted length of {Config.MAX_TTS_TEXT_LEN} characters"
        }), 400

    logger.info(f"TTS Request Queued: {text[:30]}...")
    speak(text.strip())
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    if Config.DEBUG:
        logger.warning("Running Flask in DEBUG mode. Do not use in production.")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False
    )
