"""
src/speech/train_speech_cremad.py

Speech Emotion Recognition (SER) training pipeline on CREMA-D dataset.
Includes:
- Speaker-grouped train/validation split (GroupShuffleSplit on ActorID) to prevent speaker overlap leakage.
- Audio feature extraction with optional data augmentation (noise, pitch shift, time stretch).
- Regularization (L2 kernel regularization, BatchNormalization, Dropout).
- Early stopping and ModelCheckpoint (restore best weights).
- Explicit reporting separating Training Accuracy and Validation Accuracy.
"""

import os
import json
import logging
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
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
SAVE_SPEAKERS = DATA_SPEECH_DIR / 'speakers_cremad.npy'

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

# --- Emotion mapping (CREMA-D 6 classes) ---
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add low-level white noise for data augmentation."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def extract_features(file_path: str, sr: int = 22050, n_mfcc: int = 40, augment: bool = False) -> np.ndarray:
    """Extract MFCC + Mel-spectrogram + Chroma + ZCR + Spectral Centroid."""
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    audio, _ = librosa.effects.trim(audio)

    if augment and np.random.rand() > 0.5:
        audio = add_noise(audio)

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


def preprocess_cremad(force_recompute: bool = False):
    """Preprocess CREMA-D audio files, extracting acoustic features and speaker IDs."""
    if SAVE_FEATURES.exists() and SAVE_LABELS.exists() and SAVE_SPEAKERS.exists() and not force_recompute:
        logger.info("Features, labels, and speaker IDs already exist; skipping extraction.")
        return

    if not CREMA_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"CREMA-D audio directory not found: {CREMA_AUDIO_DIR}.\n"
            "Place CREMA-D .wav files under data/CREMA-D/AudioWAV/."
        )

    files = sorted([f for f in os.listdir(CREMA_AUDIO_DIR) if f.lower().endswith('.wav')])
    features, labels, speakers = [], [], []

    for file in tqdm(files, desc='Extracting features'):
        try:
            parts = file.split('_')
            actor_id = parts[0]
            emotion_code = parts[2]
            if emotion_code not in EMOTION_MAP:
                continue
            lbl = EMOTION_MAP[emotion_code]
            feats = extract_features(str(CREMA_AUDIO_DIR / file))
            features.append(feats)
            labels.append(lbl)
            speakers.append(actor_id)
        except Exception as e:
            logger.warning(f"Skipping {file}: {e}")

    X = np.array(features)
    y = np.array(labels)
    spk = np.array(speakers)
    np.save(SAVE_FEATURES, X)
    np.save(SAVE_LABELS, y)
    np.save(SAVE_SPEAKERS, spk)
    logger.info(f"Saved features ({X.shape}), labels ({y.shape}), and speaker IDs ({spk.shape})")


def train_model(epochs: int = 100, batch_size: int = 32):
    """Train 1D-CNN SER model with speaker-grouped split and early stopping."""
    if not SAVE_FEATURES.exists() or not SAVE_LABELS.exists():
        raise FileNotFoundError(
            f"Preprocessed features not found at {SAVE_FEATURES}. Run preprocessing first."
        )

    X = np.load(SAVE_FEATURES)
    y = np.load(SAVE_LABELS)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 1. Speaker-grouped train/validation split (prevent speaker identity leakage)
    if SAVE_SPEAKERS.exists():
        speakers = np.load(SAVE_SPEAKERS)
        logger.info("Using GroupShuffleSplit by Actor ID to ensure 0 speaker overlap.")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y_enc, groups=speakers))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]
    else:
        logger.warning("speakers_cremad.npy not found; falling back to stratified random split. "
                       "Re-run preprocess_cremad(force_recompute=True) to extract speaker groups.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

    y_train_cat = to_categorical(y_train, num_classes=len(le.classes_))
    y_val_cat = to_categorical(y_val, num_classes=len(le.classes_))

    # Reshape for 1D-CNN (samples, timesteps, 1)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # 2. Regularized 1D-CNN Architecture
    weight_decay = 1e-4
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
               input_shape=(X_train_cnn.shape[1], 1)),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.35),

        Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.35),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 3. Callbacks: EarlyStopping on val_loss & ModelCheckpoint on val_accuracy
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(SAVE_MODEL), monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    logger.info(f"Training SER 1D-CNN with classes: {list(le.classes_)}")
    history = model.fit(
        X_train_cnn, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_cnn, y_val_cat),
        callbacks=callbacks
    )

    # Save Label Encoder
    joblib.dump(le, str(SAVE_LE))
    logger.info(f"Saved best model to {SAVE_MODEL} and label encoder to {SAVE_LE}")

    # 4. Separate Training vs Validation Reporting
    y_pred_train_probs = model.predict(X_train_cnn)
    y_pred_train = np.argmax(y_pred_train_probs, axis=1)
    train_acc = accuracy_score(y_train, y_pred_train)

    y_pred_val_probs = model.predict(X_val_cnn)
    y_pred_val = np.argmax(y_pred_val_probs, axis=1)
    val_acc = accuracy_score(y_val, y_pred_val)

    val_cls_report = classification_report(y_val, y_pred_val, target_names=list(le.classes_))
    overfit_gap = train_acc - val_acc

    report_content = (
        "============================================================\n"
        "SPEECH EMOTION RECOGNITION (SER) TRAINING & VALIDATION REPORT\n"
        "============================================================\n"
        f"Training Accuracy:   {train_acc:.4f}\n"
        f"Validation Accuracy: {val_acc:.4f}\n"
        f"Generalization Gap:  {overfit_gap:+.4f} (Train Acc - Val Acc)\n"
        "============================================================\n"
        "\nPER-CLASS VALIDATION CLASSIFICATION REPORT:\n\n"
        f"{val_cls_report}\n"
    )

    with open(REPORT_TEXT, 'w', encoding='utf-8') as fh:
        fh.write(report_content)
    logger.info(f"Saved comprehensive validation report to {REPORT_TEXT}")
    print("\n" + report_content)

    # Save JSON history
    hist = history.history
    try:
        report_data = {
            'classes': list(le.classes_),
            'train_accuracy': float(train_acc),
            'validation_accuracy': float(val_acc),
            'generalization_gap': float(overfit_gap),
            'history': {k: [float(v) for v in vals] for k, vals in hist.items()}
        }
        with open(REPORT_JSON, 'w', encoding='utf-8') as fh:
            json.dump(report_data, fh, indent=2)
        logger.info(f"Saved JSON report to {REPORT_JSON}")
    except Exception as e:
        logger.warning(f"Failed to save JSON report: {e}")

    # Plot metrics
    if plt is not None:
        try:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(hist.get('accuracy', []), label='Train Acc')
            plt.plot(hist.get('val_accuracy', []), label='Val Acc')
            plt.title('SER Accuracy (Speaker-Grouped)')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(hist.get('loss', []), label='Train Loss')
            plt.plot(hist.get('val_loss', []), label='Val Loss')
            plt.title('SER Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(REPORT_PNG)
            plt.close()
            logger.info(f"Saved plot to {REPORT_PNG}")
        except Exception as e:
            logger.warning(f"Failed to plot history: {e}")


def main():
    if not (SAVE_FEATURES.exists() and SAVE_LABELS.exists()):
        logger.info("Features not found. Checking raw CREMA-D directory...")
        preprocess_cremad()
    train_model()


if __name__ == '__main__':
    main()
