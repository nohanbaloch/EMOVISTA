
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path(__file__).resolve().parents[2]

# Accept either 'Data' or 'data' to be more flexible
POSSIBLE_DATA_DIRS = [ROOT / 'Data', ROOT / 'data']
# CREMA-D audio location (look under both Data/data)
CREMA_AUDIO_DIR = ROOT / 'Data' / 'CREMA-D' / 'AudioWAV'
for base in POSSIBLE_DATA_DIRS:
    cand = base / 'CREMA-D' / 'AudioWAV'
    if cand.exists():
        CREMA_AUDIO_DIR = cand
        break

# Features/labels storage (under repo data/speech)
DATA_SPEECH_DIR = ROOT / 'data' / 'speech'
if not DATA_SPEECH_DIR.exists():
    # try capitalized Data
    DATA_SPEECH_DIR = ROOT / 'Data' / 'speech'
DATA_SPEECH_DIR.mkdir(parents=True, exist_ok=True)

SAVE_FEATURES = DATA_SPEECH_DIR / 'features_cremad.npy'
SAVE_LABELS = DATA_SPEECH_DIR / 'labels_cremad.npy'

SPEECH_MODEL_DIR = ROOT / 'models' / 'speech_model'
SPEECH_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVE_MODEL = SPEECH_MODEL_DIR / 'speech_model.keras'
SAVE_LE = SPEECH_MODEL_DIR / 'speech_label_encoder.pkl'

# report and log directory for speech model
REPORT_DIR = SPEECH_MODEL_DIR / 'report_and_log'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = REPORT_DIR / 'speech_training.log'
REPORT_JSON = REPORT_DIR / 'speech_training_report.json'
REPORT_CSV = REPORT_DIR / 'speech_training_history.csv'
REPORT_PNG = REPORT_DIR / 'speech_training_history.png'
REPORT_TEXT = REPORT_DIR / 'speech_training_report.txt'

# Setup logging
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

logger.info(f"Root: {ROOT}")
logger.info(f"CREMA audio dir: {CREMA_AUDIO_DIR}")
logger.info(f"Features path: {SAVE_FEATURES}")

EMOTION_MAP = {
    "ANG":"angry",
    "DIS":"disgust",
    "FEA":"fear",
    "HAP":"happy",
    "NEU":"neutral",
    "SAD":"sad"
}

def extract_features(file_path, sr=22050, n_mfcc=40):
    """Extract richer features: MFCC + mel-spectrogram + chroma + zero-crossing rate."""
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    # Trim silence
    audio, _ = librosa.effects.trim(audio)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    
    # Mel-spectrogram (log scale)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(mel_spec_db.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)
    
    # Spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_cent_mean = np.mean(spec_cent)
    
    # Concatenate all features
    features = np.concatenate([
        mfcc_mean,
        mel_mean,
        chroma_mean,
        [zcr_mean, spec_cent_mean]
    ])
    return features

def preprocess_cremad(force_recompute=False):
    """Extract features from CREMA-D WAVs and save to disk.

    If `SAVE_FEATURES` already exists and `force_recompute` is False, this will skip extraction.
    """
    if SAVE_FEATURES.exists() and SAVE_LABELS.exists() and not force_recompute:
        logger.info(f"Features already exist at {SAVE_FEATURES}; skip preprocessing.")
        return

    if not CREMA_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"CREMA-D audio directory not found at {CREMA_AUDIO_DIR}.\n"
            f"Place WAV files there or precompute features to {SAVE_FEATURES}"
        )

    files = sorted([f for f in os.listdir(CREMA_AUDIO_DIR) if f.lower().endswith('.wav')])
    if not files:
        raise FileNotFoundError(f"No WAV files found in {CREMA_AUDIO_DIR}")

    features = []
    labels = []
    for file in tqdm(files, desc='Extracting CREMA-D features'):
        try:
            parts = file.split('_')
            emotion_code = parts[2]
            if emotion_code not in EMOTION_MAP:
                logger.debug(f"Skipping unknown emotion code {emotion_code} in file {file}")
                continue
            lbl = EMOTION_MAP[emotion_code]
            path = CREMA_AUDIO_DIR / file
            feats = extract_features(str(path))
            features.append(feats)
            labels.append(lbl)
        except Exception as e:
            logger.warning(f"Skipping {file} due to error: {e}")

    X = np.array(features)
    y = np.array(labels)
    np.save(SAVE_FEATURES, X)
    np.save(SAVE_LABELS, y)
    logger.info(f"Saved features to {SAVE_FEATURES} and labels to {SAVE_LABELS}")

def train_model():
    try:
        X = np.load(SAVE_FEATURES)
        y = np.load(SAVE_LABELS)
    except FileNotFoundError as e:
        logger.error(f"Features or labels not found: {e}")
        raise

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/validation split for explicit evaluation and reporting
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)

    # Reshape for 1D-CNN (add time dimension and channel)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Build 1D-CNN model for richer temporal feature learning
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

    logger.info(f"Starting 1D-CNN training: X_cnn={X_train_cnn.shape}, classes={list(le.classes_)}")
    history = model.fit(
        X_train_cnn, y_train_cat, epochs=100, batch_size=32, validation_data=(X_val_cnn, y_val_cat)
    )

    # Save model and label encoder
    model.save(str(SAVE_MODEL))
    joblib.dump(le, str(SAVE_LE))
    logger.info(f"Saved speech model to {SAVE_MODEL} and label encoder to {SAVE_LE}")

    # Reporting: history -> json/csv and optional plot
    hist = history.history
    timestamp = datetime.now().isoformat()
    report = {
        'timestamp': timestamp,
        'n_samples': int(X.shape[0]),
        'classes': [str(c) for c in le.classes_],
        'history': hist
    }
    try:
        with open(REPORT_JSON, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        logger.info(f"Wrote training report JSON to {REPORT_JSON}")
    except Exception as e:
        logger.warning(f"Failed to write JSON report: {e}")

    # CSV: write epoch rows
    try:
        keys = list(hist.keys())
        epochs = range(1, len(hist[keys[0]]) + 1)
        with open(REPORT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch'] + keys)
            for i in range(len(epochs)):
                row = [i + 1] + [hist[k][i] for k in keys]
                writer.writerow(row)
        logger.info(f"Wrote training history CSV to {REPORT_CSV}")
    except Exception as e:
        logger.warning(f"Failed to write CSV report: {e}")

    # Plot if matplotlib available
    if plt is not None:
        try:
            plt.figure(figsize=(8, 4))
            if 'loss' in hist:
                plt.plot(hist['loss'], label='train_loss')
            if 'val_loss' in hist:
                plt.plot(hist['val_loss'], label='val_loss')
            if 'accuracy' in hist:
                plt.plot(hist['accuracy'], label='train_acc')
            if 'val_accuracy' in hist:
                plt.plot(hist['val_accuracy'], label='val_acc')
            plt.legend()
            plt.title('Training History')
            plt.xlabel('Epoch')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(REPORT_PNG)
            plt.close()
            logger.info(f"Saved training plot to {REPORT_PNG}")
        except Exception as e:
            logger.warning(f"Failed to plot training history: {e}")

    # Validation classification report
    try:
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_val, y_pred)
        cls_report = classification_report(y_val, y_pred, target_names=list(le.classes_))
        with open(REPORT_TEXT, 'w', encoding='utf-8') as fh:
            fh.write(f"Validation accuracy: {acc:.4f}\n\n")
            fh.write(str(cls_report))
        logger.info(f"Wrote validation classification report to {REPORT_TEXT}")
    except Exception as e:
        logger.warning(f"Failed to create validation report: {e}")

def main():
    # If precomputed features exist, use them. Otherwise attempt to extract from CREMA audio dir.
    if SAVE_FEATURES.exists() and SAVE_LABELS.exists():
        logger.info(f"Found existing features at {SAVE_FEATURES}; skipping extraction.")
    else:
        # Need to have CREMA audio directory to build features
        if not CREMA_AUDIO_DIR.exists():
            raise FileNotFoundError(
                f"Neither features ({SAVE_FEATURES}) nor CREMA audio dir ({CREMA_AUDIO_DIR}) exist.\n"
                "Place CREMA-D WAVs in the audio directory or precompute features and save to the features path."
            )
        preprocess_cremad()
    train_model()

if __name__ == '__main__':
    main()
