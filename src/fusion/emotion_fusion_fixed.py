import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Optional imports handled safely
try:
    import joblib
except Exception:
    joblib = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# SAFER PROJECT ROOT DISCOVERY
# ---------------------------------------------------------------------
def _discover_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(5):
        if (p / "models").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parent  # fallback

BASE = _discover_root()

# ---------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------
FER_MODEL_PATHS = [
    BASE / "models" / "fer_model.keras",
    BASE / "models" / "fer_model.h5",
]

SPEECH_MODEL_PATHS = [
    BASE / "models" / "speech_model" / "speech_model.keras",
    BASE / "models" / "speech_model" / "speech_model.h5",
]

TEXT_MODEL_PATH = BASE / "models" / "text_model.pkl"
VECT_PATH = BASE / "models" / "vectorizer.pkl"
SPEECH_LE_PATH = BASE / "models" / "speech_model" / "speech_label_encoder.pkl"
FUSION_MODEL_PATH = BASE / "models" / "fusion_model" / "fusion_model.pkl"

fer_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Fusion adjustable weights
FER_WEIGHT = 0.5
SPEECH_WEIGHT = 0.3
TEXT_WEIGHT = 0.2


# ---------------------------------------------------------------------
# SAFE ARRAY HANDLER
# ---------------------------------------------------------------------
def _pad_or_trim(arr, length):
    arr = np.asarray(arr, dtype=float).ravel() if arr is not None else np.array([])
    if arr.size < length:
        return np.concatenate([arr, np.zeros(length - arr.size)])
    return arr[:length]


# ---------------------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------------------
def load_all(verbose: bool = False):
    fer_model = None
    speech_model = None
    text_model = None
    vectorizer = None
    speech_le = None

    # -------- FER MODEL LOADING --------
    def _load_fer():
        if load_model is None:
            return None

        # 1) Try defined paths
        for p in FER_MODEL_PATHS:
            if p.exists():
                try:
                    return load_model(str(p))
                except Exception as e:
                    logger.warning(f"Failed loading FER model: {p} → {e}")

        # 2) Scan 'models/' for files containing 'fer'
        for file in (BASE / "models").rglob("*"):
            name = file.name.lower()
            if file.suffix in (".keras", ".h5") and "fer" in name:
                try:
                    return load_model(str(file))
                except Exception:
                    continue

        return None

    fer_model = _load_fer()

    # -------- SPEECH MODEL LOADING --------
    if load_model:
        for p in SPEECH_MODEL_PATHS:
            if p.exists():
                try:
                    speech_model = load_model(str(p))
                    break
                except Exception as e:
                    logger.warning(f"Speech model load failed: {e}")

    # -------- TEXT MODEL --------
    if joblib and TEXT_MODEL_PATH.exists():
        try:
            text_model = joblib.load(TEXT_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Text model load failed: {e}")

    if joblib and VECT_PATH.exists():
        try:
            vectorizer = joblib.load(VECT_PATH)
        except Exception as e:
            logger.warning(f"Vectorizer load failed: {e}")

    if joblib and SPEECH_LE_PATH.exists():
        try:
            speech_le = joblib.load(SPEECH_LE_PATH)
        except Exception as e:
            logger.warning(f"Speech LE load failed: {e}")

    return fer_model, speech_model, text_model, vectorizer, speech_le


# ---------------------------------------------------------------------
# SPEECH → FER LABEL MAPPING
# ---------------------------------------------------------------------
def map_speech_to_fer(speech_label: str):
    mapping = {
        "angry": "Angry",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "neutral": "Neutral",
        "sad": "Sad",
    }
    key = str(speech_label).strip().lower()
    return mapping.get(key)


# ---------------------------------------------------------------------
# FUSION ENGINE
# ---------------------------------------------------------------------
def fuse(fer_pred, speech_pred, speech_le, text_pred):
    """
    Returns:
        (best_label, combined_probability_vector)
    """

    # Convert all predictions safely
    fer_arr = np.asarray(fer_pred).ravel() if fer_pred is not None else None
    speech_arr = np.asarray(speech_pred).ravel() if speech_pred is not None else None
    text_arr = np.asarray(text_pred).ravel() if text_pred is not None else None

    # -----------------------------------------------------------------
    # TRY LEARNED FUSION MODEL FIRST
    # -----------------------------------------------------------------
    if FUSION_MODEL_PATH.exists() and joblib:
        try:
            payload = joblib.load(FUSION_MODEL_PATH)

            if isinstance(payload, dict):
                model = payload.get("model")
                meta = payload.get("meta", {})
            else:
                model = payload
                meta = {}

            n_speech = meta.get("n_speech", speech_arr.size if speech_arr is not None else 0)
            n_text = meta.get("n_text", text_arr.size if text_arr is not None else 0)

            feat = []
            feat.extend(_pad_or_trim(fer_arr, 7))
            feat.extend(_pad_or_trim(speech_arr, n_speech))
            feat.extend(_pad_or_trim(text_arr, n_text))

            X = np.asarray(feat).reshape(1, -1)

            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[0]
                out = np.zeros(len(fer_labels))

                if hasattr(model, "classes_"):
                    for i, cls in enumerate(model.classes_):
                        if cls in fer_labels:
                            out[fer_labels.index(cls)] = p[i]

                # normalize
                if out.sum() > 0:
                    out = out / out.sum()
                    return fer_labels[int(np.argmax(out))], out

        except Exception as e:
            logger.warning(f"Fusion model failed: {e}")

    # -----------------------------------------------------------------
    # LEGACY WEIGHTED FUSION (FALLBACK)
    # -----------------------------------------------------------------
    combined = np.zeros(len(fer_labels))

    # FER weight
    if fer_arr is not None and fer_arr.size == 7:
        combined += fer_arr * FER_WEIGHT

    # Speech weight → mapped into FER labels
    if speech_arr is not None and speech_le is not None:
        for i, cls in enumerate(speech_le.classes_):
            mapped = map_speech_to_fer(cls)
            if mapped in fer_labels:
                idx = fer_labels.index(mapped)
                combined[idx] += speech_arr[i] * SPEECH_WEIGHT

    # Text → approximate mapping
    if text_arr is not None and text_arr.size >= 3:
        combined[fer_labels.index("Sad")] += text_arr[0] * TEXT_WEIGHT
        combined[fer_labels.index("Neutral")] += text_arr[1] * TEXT_WEIGHT
        combined[fer_labels.index("Happy")] += text_arr[2] * TEXT_WEIGHT

    # Normalize
    if combined.sum() > 0:
        combined = combined / combined.sum()

    best = fer_labels[int(np.argmax(combined))]
    return best, combined
