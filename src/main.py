"""Real-time runner for Emotion-Aware AI Assistant.

This script opens the webcam and (optionally) microphone, performs per-modality
predictions using the project's trained artifacts and fuses them into a single
FER-style label for display.

Features:
- Uses `src/fusion/emotion_fusion.load_all()` to locate/load models/artifacts.
- Automatically adapts FER preprocessing to the model's expected input shape
  (grayscale vs RGB, and target HxW).
- Robust audio capture with feature extraction (MFCC+mel+chroma+ZCR+centroid)
  and a safe fallback when no audio device exists.
- Uses the learned fusion model if present, otherwise falls back to legacy
  weighted fusion implemented in `fuse()`.

Run from repo root:
    python src\main.py
"""

from pathlib import Path
import argparse
import logging
import sys
import time
import threading
import queue
from typing import Optional, Tuple

import cv2
import numpy as np
import joblib
import librosa

# third-party audio capture (optional)
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except Exception:
    _HAS_SOUNDDEVICE = False

from fusion.emotion_fusion import load_all, fuse, fer_labels

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger('emotion_main')

# ---- Helpers ----

def get_model_input_spec(model) -> Optional[Tuple[int, int, Optional[int]]]:
    """Return (H, W, C) where C may be None for single-channel models or None if unknown."""
    if model is None:
        return None
    try:
        shape = getattr(model, 'input_shape', None)
        if shape is None and hasattr(model, 'inputs'):
            try:
                shape = tuple(model.inputs[0].shape.as_list())
            except Exception:
                shape = None
        if shape is None:
            return None
        # shape examples: (None, H, W, C) or (None, H, W)
        if len(shape) == 4:
            return (int(shape[1]), int(shape[2]), int(shape[3]))
        if len(shape) == 3:
            return (int(shape[1]), int(shape[2]), None)
    except Exception as ex:
        logger.warning('Unable to read model input shape: %s', ex)
    return None


def preprocess_face_roi(frame: np.ndarray, spec: Optional[Tuple[int, int, Optional[int]]]) -> np.ndarray:
    """Given a BGR frame and ROI frame region, convert it to the model input.

    - `frame`: color ROI in BGR (as returned from OpenCV slicing)
    - `spec`: (H, W, C) where C==3 => RGB, C==1 or None => grayscale

    Returns: array shaped to be passed to Keras model (batch dim included)
    """
    if spec is None:
        h, w, c = 96, 96, 3
    else:
        h, w, c = spec
        if h is None:
            h, w, c = 96, 96, c
    try:
        resized = cv2.resize(frame, (w, h))
    except Exception:
        # fallback to safe size
        resized = cv2.resize(frame, (96, 96))
        h, w, c = 96, 96, 3

    if c is None or c == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        arr = gray.astype('float32') / 255.0
        # determine whether the model expects a channel axis
        # produce (1,H,W,1) for compatibility with typical Keras conv input
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype('float32') / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr


def extract_audio_features(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract a robust 1D feature vector from raw audio array.

    Returns a 1D numpy array (mean-pooled across time) ready for model input.
    """
    try:
        if y is None or len(y) == 0:
            return np.zeros(40, dtype=float)
        y = librosa.util.fix_length(y, int(sr * 3), mode='constant')
        y, _ = librosa.effects.trim(y)
        # MFCC (n_mfcc=40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        # Mel-spectrogram (mean over time)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        # ZCR and spectral centroid (single values)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        feat = np.concatenate([mfcc_mean, mel_mean, chroma_mean, [zcr, cent]])
        return feat
    except Exception as ex:
        logger.exception('Audio feature extraction failed: %s', ex)
        return np.zeros(40, dtype=float)


# ---- Main runtime ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Webcam index')
    parser.add_argument('--no-audio', action='store_true', help='Disable microphone capture')
    parser.add_argument('--audio-sr', type=int, default=22050, help='Audio sample rate')
    # Use parse_known_args so extra args injected by IPython/Jupyter kernels
    # do not cause argparse to call sys.exit() inside a notebook.
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug('Ignored unknown command-line args: %s', unknown)

    # Load models and artifacts
    fer_model, speech_model, text_model, vectorizer, speech_le = load_all(verbose=True)
    logger.info('FER model present: %s', bool(fer_model))
    logger.info('Speech model present: %s', bool(speech_model))
    logger.info('Text model present: %s', bool(text_model))

    fer_spec = get_model_input_spec(fer_model)
    logger.info('Detected FER input spec: %s', fer_spec)

    # Audio queue and capture thread
    audio_q: "queue.Queue" = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        try:
            audio_q.put(indata.copy().flatten())
        except Exception:
            pass

    def audio_thread():
        if not _HAS_SOUNDDEVICE:
            logger.info('sounddevice not available; skipping audio capture')
            return
        try:
            with sd.InputStream(channels=1, samplerate=args.audio_sr, callback=audio_callback):
                logger.info('Audio input stream started')
                while not stop_event.is_set():
                    sd.sleep(200)
        except Exception as ex:
            logger.exception('Audio listener failed or no input device: %s', ex)

    if not args.no_audio and _HAS_SOUNDDEVICE:
        t = threading.Thread(target=audio_thread, daemon=True)
        t.start()
    else:
        logger.info('Audio capture disabled')

    # Webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error('Unable to open camera index %s', args.camera)
        return

    logger.info("Starting Real-time Emotion-Aware Assistant. Press 'q' in the webcam window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            fer_pred = None
            text_pred = None

            for (x, y, w, h) in faces:
                try:
                    roi_color = frame[y:y + h, x:x + w]
                    roi_input = preprocess_face_roi(roi_color, fer_spec)
                    if fer_model is not None:
                        fer_out = fer_model.predict(roi_input)
                        # normalize to 1D prob vector if shape matches
                        if fer_out is not None:
                            fer_pred = np.asarray(fer_out[0]).ravel()
                        else:
                            fer_pred = None
                    else:
                        fer_pred = None

                    label_text = fer_labels[int(np.argmax(fer_pred))] if fer_pred is not None else 'NoFER'

                except Exception:
                    logger.exception('FER preprocessing/prediction failed for face at %s,%s,%s,%s', x, y, w, h)
                    label_text = 'Error'
                    fer_pred = None

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # audio inference: get latest chunk if present
            speech_pred = None
            if speech_model is not None and not audio_q.empty():
                try:
                    raw = audio_q.get()
                    feats = extract_audio_features(raw, sr=args.audio_sr)
                    # Try to adapt feature shape to the model input
                    in_shape = getattr(speech_model, 'input_shape', None)
                    X = feats.reshape(1, -1)
                    if in_shape is not None:
                        # if model expects 3D (None, timesteps, features), try to expand
                        if len(in_shape) == 3:
                            # attempt to place features on last axis with timesteps=1
                            X = X.reshape(1, 1, -1)
                    try:
                        out = speech_model.predict(X)
                        speech_pred = np.asarray(out[0]).ravel()
                    except Exception:
                        logger.exception('Speech model predict failed')
                        speech_pred = None
                except Exception:
                    logger.exception('Audio processing failed')
                    speech_pred = None

            # text inference: not interactive here; text input is provided by GUI or other frontends

            # fuse predictions
            try:
                fused_label, combined = fuse(fer_pred, speech_pred, speech_le, text_pred)
            except Exception:
                logger.exception('Fusion failed')
                fused_label, combined = ('Unknown', np.zeros(len(fer_labels)))

            # overlay fused result
            try:
                display_text = f"Fused: {fused_label}"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            except Exception:
                pass

            cv2.imshow('EmotionAwareAI - Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info('Interrupted by user')

    finally:
        stop_event.set()
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
