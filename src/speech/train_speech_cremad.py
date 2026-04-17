import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime

import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
CREMA_AUDIO_DIR = ROOT / 'data' / 'CREMA-D' / 'AudioWAV'
DATA_SPEECH_DIR = ROOT / 'data' / 'speech'
DATA_SPEECH_DIR.mkdir(parents=True, exist_ok=True)
SAVE_FEATURES = DATA_SPEECH_DIR / 'features_cremad.npy'
SAVE_LABELS = DATA_SPEECH_DIR / 'labels_cremad.npy'

SPEECH_MODEL_DIR = ROOT / 'models' / 'speech_model'
SPEECH_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVE_MODEL = SPEECH_MODEL_DIR / 'speech_model.keras'
SAVE_LE = SPEECH_MODEL_DIR / 'speech_label_encoder.pkl'

REPORT_DIR = SPEECH_MODEL_DIR / 'report_and_log'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = REPORT_DIR / 'speech_training.log'
REPORT_JSON = REPORT_DIR / 'speech_training_report.json'
REPORT_CSV = REPORT_DIR / 'speech_training_history.csv'
REPORT_PNG = REPORT_DIR / 'speech_training_history.png'
REPORT_TEXT = REPORT_DIR / 'speech_training_report.txt'

# --- Logging ---
logger = logging.getLogger('speech_train')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(TRAIN_LOG, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

# --- Emotion mapping ---
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# --- Feature extraction ---
def extract_features(file_path, sr=22050, n_mfcc=40):
    """MFCC + Mel + Chroma + ZCR + Spectral Centroid"""
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    audio, _ = librosa.effects.trim(audio)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    
    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(mel_spec_db.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # ZCR and spectral centroid
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(audio))
    spec_cent_mean = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    return np.concatenate([mfcc_mean, mel_mean, chroma_mean, [zcr_mean, spec_cent_mean]])

# --- Preprocessing ---
def preprocess_cremad(force_recompute=False):
    if SAVE_FEATURES.exists() and SAVE_LABELS.exists() and not force_recompute:
        logger.info("Features already exist; skipping extraction.")
        return

    if not CREMA_AUDIO_DIR.exists():
        raise FileNotFoundError(f"CREMA audio dir not found: {CREMA_AUDIO_DIR}")

    files = sorted([f for f in os.listdir(CREMA_AUDIO_DIR) if f.lower().endswith('.wav')])
    features, labels = [], []

    for file in tqdm(files, desc='Extracting features'):
        try:
            parts = file.split('_')
            emotion_code = parts[2]
            if emotion_code not in EMOTION_MAP:
                continue
            lbl = EMOTION_MAP[emotion_code]
            feats = extract_features(str(CREMA_AUDIO_DIR / file))
            features.append(feats)
            labels.append(lbl)
        except Exception as e:
            logger.warning(f"Skipping {file}: {e}")

    X = np.array(features)
    y = np.array(labels)
    np.save(SAVE_FEATURES, X)
    np.save(SAVE_LABELS, y)
    logger.info(f"Saved features ({X.shape}) and labels ({y.shape})")

# --- Training ---
def train_model(epochs=100, batch_size=32):
    X = np.load(SAVE_FEATURES)
    y = np.load(SAVE_LABELS)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)

    # 1D-CNN expects (samples, timesteps, features=1)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # --- Model ---
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=(X_train_cnn.shape[1], 1)),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(128, 3, activation='relu', padding='same'),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    logger.info(f"Training 1D-CNN with classes: {list(le.classes_)}")
    history = model.fit(X_train_cnn, y_train_cat, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val_cnn, y_val_cat))

    # --- Save model & label encoder ---
    model.save(str(SAVE_MODEL))
    joblib.dump(le, str(SAVE_LE))
    logger.info(f"Saved model to {SAVE_MODEL} and label encoder to {SAVE_LE}")

    # --- Reports ---
    hist = history.history
    try:
        with open(REPORT_JSON, 'w', encoding='utf-8') as fh:
            json.dump({'classes': list(le.classes_), 'history': hist}, fh, indent=2)
        logger.info(f"Saved JSON report to {REPORT_JSON}")
    except Exception as e:
        logger.warning(f"Failed to save JSON report: {e}")

    if plt is not None:
        try:
            plt.figure(figsize=(8,4))
            plt.plot(hist.get('loss', []), label='train_loss')
            plt.plot(hist.get('val_loss', []), label='val_loss')
            plt.plot(hist.get('accuracy', []), label='train_acc')
            plt.plot(hist.get('val_accuracy', []), label='val_acc')
            plt.legend()
            plt.grid(True)
            plt.savefig(REPORT_PNG)
            plt.close()
            logger.info(f"Saved plot to {REPORT_PNG}")
        except Exception as e:
            logger.warning(f"Failed to plot history: {e}")

    # Validation report
    try:
        y_pred_probs = model.predict(X_val_cnn)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_val, y_pred)
        cls_report = classification_report(y_val, y_pred, target_names=list(le.classes_))
        with open(REPORT_TEXT, 'w', encoding='utf-8') as fh:
            fh.write(f"Validation accuracy: {acc:.4f}\n\n{cls_report}")
        logger.info(f"Saved validation report to {REPORT_TEXT}")
    except Exception as e:
        logger.warning(f"Failed to save validation report: {e}")

def main():
    if SAVE_FEATURES.exists() and SAVE_LABELS.exists():
        logger.info("Using existing features.")
    else:
        preprocess_cremad()
    train_model()

if __name__ == '__main__':
    main()
