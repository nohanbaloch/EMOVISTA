"""
src/fusion/emotion_fusion.py

Multimodal Emotion Fusion Engine for EMOVISTA.
Fuses visual (FER), audio (SER), and text sentiment modalities into a
unified 7-emotion probability distribution:
['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'].

Fusion Methodology:
1. Base modality weights:
   - Visual (FER): 50% (0.50)
   - Speech (SER): 30% (0.30)
   - Text Sentiment: 20% (0.20)
2. Dynamic Renormalization:
   When one or two modalities are absent or invalid, weights are
   renormalized over the active modalities so their sum equals 1.0.
3. Modality Alignments:
   - Visual (FER): Direct 7-class prediction.
   - Speech (SER): 6 CREMA-D classes (angry, disgust, fear, happy, neutral, sad).
     *Surprise* is not present in speech audio and can only originate from visual FER.
   - Text Sentiment: Binary sentiment [P(neg), P(pos)] is mapped to a 3-way
     distribution [P(neg), P(neu), P(pos)] where Neutral represents model
     uncertainty (low confidence) rather than a confident neutral prediction:
       Confidence C = |P(pos) - P(neg)|
       P(neu) = 1.0 - C
       P(neg) = (1.0 - P(neu)) * (P(neg) / (P(neg) + P(pos)))
       P(pos) = (1.0 - P(neu)) * (P(pos) / (P(neg) + P(pos)))
     Then mapped: negative -> Sad, neutral -> Neutral, positive -> Happy.
4. Fallback:
   If all modalities are absent or malformed, the system gracefully falls back
   to Neutral with a deterministic probability vector.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Sequence

import numpy as np

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

try:
    import keras  # type: ignore
    load_model = keras.saving.load_model
except Exception:
    try:
        from tensorflow.keras.models import load_model  # type: ignore
    except Exception:
        load_model = None

logger = logging.getLogger(__name__)

# Base repo root (two levels up: src/fusion -> src -> repo)
BASE = Path(__file__).resolve().parents[2]

# Canonical model paths
FER_MODEL_PATHS = [
    BASE / 'models' / 'Fer_model' / 'fer_model.keras',
    BASE / 'models' / 'fer_model' / 'fer_model.keras',
]
SPEECH_MODEL_PATHS = [
    BASE / 'models' / 'speech_model' / 'speech_model.keras',
]
TEXT_MODEL_KERAS_PATH = BASE / 'models' / 'text_model' / 'text_model.keras'
TEXT_MODEL_PKL_PATH = BASE / 'models' / 'text_model.pkl'
VECT_PATH = BASE / 'models' / 'text_model' / 'vectorizer.pkl'
TEXT_LE_PATH = BASE / 'models' / 'text_model' / 'label_encoder.pkl'
SPEECH_LE_PATH = BASE / 'models' / 'speech_model' / 'speech_label_encoder.pkl'

# Fixed 7 target emotion labels
fer_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Base fusion weights
FER_WEIGHT = 0.50
SPEECH_WEIGHT = 0.30
TEXT_WEIGHT = 0.20

# Standard speech mapping: 6 CREMA-D classes -> FER labels
SPEECH_TO_FER = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'neutral': 'Neutral',
    'sad': 'Sad',
}


def load_all(verbose: bool = False) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Load FER, speech, and text models along with vectorizers and label encoders.

    Returns a 5-tuple:
        (fer_model, speech_model, text_model, vectorizer, speech_label_encoder)
    Any item may be None if its file is absent or fails to load.
    """
    fer_model = None
    speech_model = None
    text_model = None
    vectorizer = None
    speech_le = None

    # Load FER model
    if load_model is not None:
        for p in FER_MODEL_PATHS:
            if p.exists():
                try:
                    if verbose:
                        logger.info("Loading FER model from %s", p)
                    fer_model = load_model(str(p))
                    break
                except Exception as ex:
                    logger.warning("Failed to load FER model from %s: %s", p, ex)

    # Load Speech model
    if load_model is not None:
        for p in SPEECH_MODEL_PATHS:
            if p.exists():
                try:
                    if verbose:
                        logger.info("Loading speech model from %s", p)
                    speech_model = load_model(str(p))
                    break
                except Exception as ex:
                    logger.warning("Failed to load speech model from %s: %s", p, ex)

    # Load Text model
    if load_model is not None and TEXT_MODEL_KERAS_PATH.exists():
        try:
            if verbose:
                logger.info("Loading text model from %s", TEXT_MODEL_KERAS_PATH)
            text_model = load_model(str(TEXT_MODEL_KERAS_PATH))
        except Exception as ex:
            logger.warning("Failed to load text model from %s: %s", TEXT_MODEL_KERAS_PATH, ex)

    if text_model is None and joblib is not None and TEXT_MODEL_PKL_PATH.exists():
        try:
            text_model = joblib.load(TEXT_MODEL_PKL_PATH)
        except Exception as ex:
            logger.warning("Failed to load text model from %s: %s", TEXT_MODEL_PKL_PATH, ex)

    # Load Vectorizer and Label Encoders
    if joblib is not None:
        if VECT_PATH.exists():
            try:
                vectorizer = joblib.load(VECT_PATH)
            except Exception as ex:
                logger.warning("Failed to load vectorizer from %s: %s", VECT_PATH, ex)

        if SPEECH_LE_PATH.exists():
            try:
                speech_le = joblib.load(SPEECH_LE_PATH)
            except Exception as ex:
                logger.warning("Failed to load speech label encoder from %s: %s", SPEECH_LE_PATH, ex)

    return fer_model, speech_model, text_model, vectorizer, speech_le


def map_speech_to_fer(speech_label: str) -> Optional[str]:
    """Map a CREMA-D speech emotion label to the canonical FER label.

    Note: Speech dataset contains 6 classes (angry, disgust, fear, happy, neutral, sad).
    Surprise is not represented in CREMA-D and can only be detected via FER.
    """
    if not speech_label:
        return None
    return SPEECH_TO_FER.get(str(speech_label).strip().lower())


def map_binary_text_to_three_way(text_pred: Any) -> Optional[np.ndarray]:
    """Map text model output to a 3-way distribution: [negative, neutral, positive].

    Handles:
    - 2-way binary distribution: [P(neg), P(pos)]
      Neutral is interpreted as low classifier confidence (uncertainty),
      NOT a confident neutral prediction:
        Confidence C = |P(pos) - P(neg)|
        P(neutral) = 1.0 - C
        P(negative) = (1.0 - P(neutral)) * normalized(P(neg))
        P(positive) = (1.0 - P(neutral)) * normalized(P(pos))
    - Scalar probability p in [0, 1] representing P(positive), where P(neg) = 1 - p.
    - 3-way distribution: [P(neg), P(neutral), P(pos)] normalized to sum to 1.0.

    Returns:
        np.ndarray of length 3: [P(neg), P(neutral), P(pos)] or None if invalid.
    """
    if text_pred is None:
        return None

    try:
        arr = np.asarray(text_pred, dtype=float).ravel()
    except Exception:
        return None

    if arr.size == 0 or np.isnan(arr).any() or np.isinf(arr).any():
        return None

    # Scalar input: probability of positive class
    if arr.size == 1:
        p_pos = float(np.clip(arr[0], 0.0, 1.0))
        p_neg = 1.0 - p_pos
        arr = np.array([p_neg, p_pos], dtype=float)

    # Binary sentiment input: [p_neg, p_pos]
    if arr.size == 2:
        s = arr.sum()
        if s > 0:
            arr = arr / s
        else:
            arr = np.array([0.5, 0.5], dtype=float)

        p_neg, p_pos = float(arr[0]), float(arr[1])
        confidence = abs(p_pos - p_neg)  # in [0, 1]
        p_neutral = 1.0 - confidence     # high uncertainty => high neutral

        active_mass = 1.0 - p_neutral
        sum_polar = p_neg + p_pos
        if sum_polar > 0:
            p_neg_adj = active_mass * (p_neg / sum_polar)
            p_pos_adj = active_mass * (p_pos / sum_polar)
        else:
            p_neg_adj = active_mass * 0.5
            p_pos_adj = active_mass * 0.5

        three_way = np.array([p_neg_adj, p_neutral, p_pos_adj], dtype=float)
        total = three_way.sum()
        return three_way / total if total > 0 else np.array([0.0, 1.0, 0.0], dtype=float)

    # 3-way distribution input: [neg, neu, pos]
    if arr.size == 3:
        arr = np.clip(arr, 0.0, None)
        s = arr.sum()
        if s > 0:
            return arr / s
        return np.array([0.0, 1.0, 0.0], dtype=float)

    # Wrong vector length (e.g. size 4, 5, etc.)
    logger.warning("text_pred has unexpected length %s (expected 1, 2, or 3). Modality ignored.", arr.size)
    return None


def fuse(
    fer_pred: Optional[Any],
    speech_pred: Optional[Any],
    speech_le: Optional[Any],
    text_pred: Optional[Any]
) -> Tuple[str, np.ndarray]:
    """Fuse predictions from FER, Speech, and Text modalities.

    Arguments:
        fer_pred: Length-7 probability vector aligned with `fer_labels`, or None.
        speech_pred: Probability vector aligned with `speech_le.classes_` (6 classes), or None.
        speech_le: Scikit-learn LabelEncoder for speech with `.classes_`, or None.
        text_pred: Text sentiment probabilities (length 2 [neg, pos], 3 [neg, neu, pos], or scalar), or None.

    Returns:
        (predicted_emotion_label, combined_7_class_probabilities)
    """
    neutral_idx = fer_labels.index('Neutral') if 'Neutral' in fer_labels else 6
    fallback_vector = np.zeros(len(fer_labels), dtype=float)
    fallback_vector[neutral_idx] = 1.0

    active_modalities = {}

    # 1. Validate FER input
    if fer_pred is not None:
        try:
            f_arr = np.asarray(fer_pred, dtype=float).ravel()
            if f_arr.size == len(fer_labels) and not np.isnan(f_arr).any() and not np.isinf(f_arr).any():
                f_arr = np.clip(f_arr, 0.0, None)
                s = f_arr.sum()
                if s > 0:
                    active_modalities['fer'] = (f_arr / s, FER_WEIGHT)
                else:
                    logger.warning("fer_pred sum is 0; ignoring FER modality.")
            else:
                logger.warning("fer_pred has invalid length %s (expected %s) or contains NaN/Inf; ignoring.",
                               f_arr.size, len(fer_labels))
        except Exception as ex:
            logger.warning("Error processing fer_pred: %s; ignoring.", ex)

    # 2. Validate Speech input
    if speech_pred is not None:
        try:
            s_arr = np.asarray(speech_pred, dtype=float).ravel()
            classes = list(getattr(speech_le, 'classes_', [])) if speech_le is not None else list(SPEECH_TO_FER.keys())
            
            if s_arr.size == len(classes) and not np.isnan(s_arr).any() and not np.isinf(s_arr).any():
                s_arr = np.clip(s_arr, 0.0, None)
                s_sum = s_arr.sum()
                if s_sum > 0:
                    s_norm = s_arr / s_sum
                    # Project speech classes into 7-class FER space
                    speech_7 = np.zeros(len(fer_labels), dtype=float)
                    for i, cls in enumerate(classes):
                        mapped = map_speech_to_fer(cls)
                        if mapped and mapped in fer_labels:
                            idx = fer_labels.index(mapped)
                            speech_7[idx] += s_norm[i]
                    # Note: 'Surprise' receives 0 from speech (CREMA-D does not contain surprise)
                    active_modalities['speech'] = (speech_7, SPEECH_WEIGHT)
                else:
                    logger.warning("speech_pred sum is 0; ignoring Speech modality.")
            else:
                logger.warning("speech_pred size (%s) does not match expected classes (%s); ignoring.",
                               s_arr.size, len(classes))
        except Exception as ex:
            logger.warning("Error processing speech_pred: %s; ignoring.", ex)

    # 3. Validate Text input
    if text_pred is not None:
        try:
            mapped_text = map_binary_text_to_three_way(text_pred)
            if mapped_text is not None:
                # Map [neg, neu, pos] -> [Sad, Neutral, Happy] in 7-class space
                text_7 = np.zeros(len(fer_labels), dtype=float)
                text_7[fer_labels.index('Sad')] += mapped_text[0]
                text_7[fer_labels.index('Neutral')] += mapped_text[1]
                text_7[fer_labels.index('Happy')] += mapped_text[2]
                active_modalities['text'] = (text_7, TEXT_WEIGHT)
        except Exception as ex:
            logger.warning("Error processing text_pred: %s; ignoring.", ex)

    # 4. Fallback if no valid modalities
    if not active_modalities:
        return 'Neutral', fallback_vector

    # 5. Dynamic Weight Renormalization
    total_active_weight = sum(weight for _, weight in active_modalities.values())
    if total_active_weight <= 0:
        return 'Neutral', fallback_vector

    combined = np.zeros(len(fer_labels), dtype=float)
    for prob_vec, weight in active_modalities.values():
        normalized_weight = weight / total_active_weight
        combined += prob_vec * normalized_weight

    # Ensure final normalization
    c_sum = combined.sum()
    if c_sum > 0:
        combined = combined / c_sum
    else:
        return 'Neutral', fallback_vector

    best_idx = int(np.argmax(combined))
    return fer_labels[best_idx], combined
