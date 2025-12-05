
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import joblib
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

# base repo root (two levels up: src/fusion -> src -> repo)
BASE = Path(__file__).resolve().parents[2]

# prefer Keras native format but fall back to legacy HDF5 for compatibility
FER_MODEL_PATHS = [
    BASE / 'models' / 'fer_model.keras',
    BASE / 'models' / 'fer_model.h5',
]
SPEECH_MODEL_PATHS = [
    BASE / 'models' / 'speech_model' / 'speech_model.keras',
    BASE / 'models' / 'speech_model' / 'speech_model.h5',
]
TEXT_MODEL_PATH = BASE / 'models' / 'text_model.pkl'
VECT_PATH = BASE / 'models' / 'vectorizer.pkl'
SPEECH_LE_PATH = BASE / 'models' / 'speech_model' / 'speech_label_encoder.pkl'
FUSION_MODEL_PATH = BASE / 'models' / 'fusion_model' / 'fusion_model.pkl'

fer_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_all(verbose: bool = False) -> Tuple[Optional[object], Optional[object], Optional[object], Optional[object], Optional[object]]:
    """Load FER, speech and text models (and related artifacts).

    Returns a 5-tuple: (fer_model, speech_model, text_model, vectorizer, speech_label_encoder)
    Any item may be None when the corresponding file is missing or fails to load.
    """
    fer_model = None
    speech_model = None
    text_model = None
    vectorizer = None
    speech_le = None

    # load FER model: prefer exact configured paths, but also attempt to
    # discover common variations (e.g. `models/Fer_model/fer_model.keras`).
    def _discover_and_load_fer():
        # try configured candidates first
        for p in FER_MODEL_PATHS:
            if p.exists():
                try:
                    if verbose:
                        logger.info("Attempting to load FER model from %s", p)
                    return load_model(str(p)), p
                except Exception as ex:
                    logger.warning("Failed to load FER model from %s: %s", p, ex)

        # attempt discovery in the models/ directory (case/spacing tolerant)
        models_dir = BASE / 'models'
        if models_dir.exists() and models_dir.is_dir():
            for child in models_dir.iterdir():
                # check directories and files that look like FER model containers
                name = child.name.lower()
                if 'fer' in name or 'face' in name:
                    # if it's a directory, look for common file names inside
                    if child.is_dir():
                        for candidate in ('fer_model.keras', 'fer_model.h5', 'fer_model.keras'.lower()):
                            cand_path = child / candidate
                            if cand_path.exists():
                                try:
                                    if verbose:
                                        logger.info("Attempting to load FER model from discovered path %s", cand_path)
                                    return load_model(str(cand_path)), cand_path
                                except Exception as ex:
                                    logger.warning("Failed to load discovered FER model %s: %s", cand_path, ex)
                    else:
                        # if it's a file whose name contains fer and looks like a Keras model
                        if child.is_file() and child.suffix.lower() in ('.keras', '.h5'):
                            try:
                                if verbose:
                                    logger.info("Attempting to load FER model from discovered file %s", child)
                                return load_model(str(child)), child
                            except Exception as ex:
                                logger.warning("Failed to load discovered FER model %s: %s", child, ex)

        # last resort: scan models dir recursively for files containing 'fer' and ending in .keras/.h5
        if models_dir.exists():
            for p in models_dir.rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.keras', '.h5') and 'fer' in p.name.lower():
                    try:
                        if verbose:
                            logger.info("Attempting to load FER model from fallback scan %s", p)
                        return load_model(str(p)), p
                    except Exception as ex:
                        logger.warning("Failed to load FER model from %s: %s", p, ex)

        return None, None

    fer_model, fer_path = _discover_and_load_fer()
    if verbose and fer_model is None:
        logger.info("FER model not found under %s; expected one of: %s", BASE / 'models', FER_MODEL_PATHS)

    # speech model
    for p in SPEECH_MODEL_PATHS:
        try:
            if p.exists():
                if verbose:
                    logger.info("Attempting to load speech model from %s", p)
                speech_model = load_model(str(p))
                if verbose:
                    logger.info("Loaded speech model from %s", p)
                break
        except Exception as ex:
            logger.warning("Failed to load speech model from %s: %s", p, ex)
            speech_model = None
            continue

    # text model / artifacts
    try:
        if TEXT_MODEL_PATH.exists():
            text_model = joblib.load(TEXT_MODEL_PATH)
    except Exception as ex:
        logger.warning("Failed to load text model from %s: %s", TEXT_MODEL_PATH, ex)
        text_model = None

    try:
        if VECT_PATH.exists():
            vectorizer = joblib.load(VECT_PATH)
    except Exception as ex:
        logger.warning("Failed to load vectorizer from %s: %s", VECT_PATH, ex)
        vectorizer = None

    try:
        if SPEECH_LE_PATH.exists():
            speech_le = joblib.load(SPEECH_LE_PATH)
    except Exception as ex:
        logger.warning("Failed to load speech label encoder from %s: %s", SPEECH_LE_PATH, ex)
        speech_le = None

    return fer_model, speech_model, text_model, vectorizer, speech_le

def map_speech_to_fer(speech_label: str) -> Optional[str]:
    """Map a speech-label (CREMA-D) to FER label string.

    Returns the FER label (capitalized) or None if no mapping exists.
    """
    if not speech_label:
        return None
    mapping = {
        'angry': 'Angry',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
    }
    key = str(speech_label).strip().lower()
    return mapping.get(key)

def fuse(fer_pred: Optional[np.ndarray], speech_pred: Optional[np.ndarray], speech_le, text_pred: Optional[np.ndarray]) -> Tuple[str, np.ndarray]:
    """Fuse modality predictions into a single FER label.

    - `fer_pred` is expected to be a length-7 probability vector (or None).
    - `speech_pred` is a probability vector aligned with `speech_le.classes_` (or None).
    - `text_pred` is expected to be length-3 probabilities [neg, neutral, pos] (or None).

    Returns: (best_label, combined_prob_vector_of_length_7)
    """
    # Try to use a learned fusion model if available. The learned model
    # (saved via joblib) should be a dict containing at least:
    #   - 'model': a sklearn-like estimator with predict_proba
    #   - 'meta': { 'n_speech': int, 'n_text': int, 'feature_columns': [...], ... }
    # If loading or prediction fails we fall back to the legacy weighted scheme.

    # Prepare numeric arrays for each modality (with safe padding/trimming helpers)
    def _pad_or_trim(arr, length):
        arr = np.asarray(arr, dtype=float).ravel() if arr is not None else np.array([], dtype=float)
        if arr.size < length:
            return np.concatenate([arr, np.zeros(length - arr.size, dtype=float)])
        if arr.size > length:
            return arr[:length]
        return arr

    fer_arr = None
    try:
        if fer_pred is not None:
            fer_arr = np.asarray(fer_pred, dtype=float).ravel()
    except Exception:
        fer_arr = None

    speech_arr = None
    try:
        if speech_pred is not None:
            speech_arr = np.asarray(speech_pred, dtype=float).ravel()
    except Exception:
        speech_arr = None

    text_arr = None
    try:
        if text_pred is not None:
            text_arr = np.asarray(text_pred, dtype=float).ravel()
    except Exception:
        text_arr = None

    # Learned fusion path
    if FUSION_MODEL_PATH.exists():
        try:
            payload = joblib.load(FUSION_MODEL_PATH)
            # payload may be the model itself or a dict
            if isinstance(payload, dict):
                model = payload.get('model')
                meta = payload.get('meta', {})
            else:
                model = payload
                meta = {}

            n_speech = int(meta.get('n_speech')) if meta.get('n_speech') is not None else (speech_arr.size if speech_arr is not None else 0)
            n_text = int(meta.get('n_text')) if meta.get('n_text') is not None else (text_arr.size if text_arr is not None else 0)

            feat = []
            feat.extend(_pad_or_trim(fer_arr, 7))
            feat.extend(_pad_or_trim(speech_arr, n_speech))
            feat.extend(_pad_or_trim(text_arr, n_text))

            X = np.asarray(feat, dtype=float).reshape(1, -1)

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                model_classes = getattr(model, 'classes_', None)
                # Build a length-7 probability vector aligned with module fer_labels
                prob_vector = np.zeros(len(fer_labels), dtype=float)
                if model_classes is not None:
                    for i, cls in enumerate(model_classes):
                        try:
                            if cls in fer_labels:
                                prob_vector[fer_labels.index(cls)] = proba[i]
                        except Exception:
                            continue
                else:
                    # If model did not expose classes_, try to assume the model was trained
                    # to output probabilities in the same fer_labels order (best-effort).
                    if proba.size == len(fer_labels):
                        prob_vector = proba

                ssum = prob_vector.sum()
                if ssum > 0:
                    prob_vector = prob_vector / ssum
                    best_idx = int(np.argmax(prob_vector))
                    return fer_labels[best_idx], prob_vector
        except Exception as ex:
            logger.warning("Learned fusion model load/predict failed: %s", ex)

    # Legacy weighted fusion (fallback)
    combined = np.zeros(len(fer_labels), dtype=float)

    # FER model contribution
    if fer_pred is not None:
        try:
            arr = np.asarray(fer_pred, dtype=float).ravel()
            if arr.size == len(fer_labels):
                combined += arr * 0.5
            else:
                logger.warning("fer_pred has unexpected size %s, expected %s", arr.size, len(fer_labels))
        except Exception as ex:
            logger.warning("Error processing fer_pred: %s", ex)

    # Speech -> project into FER label space
    if speech_pred is not None and speech_le is not None:
        try:
            sarr = np.asarray(speech_pred, dtype=float).ravel()
            classes = list(getattr(speech_le, 'classes_', []))
            if sarr.size != len(classes):
                logger.warning("speech_pred size (%s) does not match speech_le.classes_ (%s)", sarr.size, len(classes))
            for i, cls in enumerate(classes):
                if i >= sarr.size:
                    break
                mapped = map_speech_to_fer(cls)
                if mapped and mapped in fer_labels:
                    idx = fer_labels.index(mapped)
                    combined[idx] += sarr[i] * 0.3
        except Exception as ex:
            logger.warning("Error processing speech_pred: %s", ex)

    # Text mapping: approximate (negative->Sad, neutral->Neutral, positive->Happy)
    if text_pred is not None:
        try:
            tarr = np.asarray(text_pred, dtype=float).ravel()
            if tarr.size >= 3:
                combined[fer_labels.index('Sad')] += tarr[0] * 0.2
                combined[fer_labels.index('Neutral')] += tarr[1] * 0.2
                combined[fer_labels.index('Happy')] += tarr[2] * 0.2
            else:
                logger.warning("text_pred expected length>=3 but got %s", tarr.size)
        except Exception as ex:
            logger.warning("Error processing text_pred: %s", ex)

    # normalize
    s = combined.sum()
    if s > 0:
        combined = combined / s

    best_idx = int(np.argmax(combined))
    return fer_labels[best_idx], combined
