
import os
import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
try:
    from transformers import DistilBertTokenizer, TFDistilBertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / 'data' / 'text' / 'imdb.csv'

TEXT_MODEL_DIR = ROOT / 'models' / 'text_model'
TEXT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVE_MODEL = TEXT_MODEL_DIR / 'text_model.keras'
SAVE_VECT = TEXT_MODEL_DIR / 'vectorizer.pkl'

REPORT_DIR = TEXT_MODEL_DIR / 'report_and_log'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LOG = REPORT_DIR / 'text_training.log'
REPORT_JSON = REPORT_DIR / 'text_training_report.json'
REPORT_TEXT = REPORT_DIR / 'text_training_report.txt'

# Setup logger
logger = logging.getLogger('text_train')
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


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Text dataset not found at {DATA_CSV}. Place imdb.csv (columns: text,label) there.")
    logger.info(f"Loading text data from {DATA_CSV}")
    data = pd.read_csv(DATA_CSV)

    # Robust column detection: look for common text and label column names
    text_candidates = ['text', 'review', 'sentence', 'utterance', 'content']
    label_candidates = ['label', 'sentiment', 'emotion', 'target']

    text_col = None
    label_col = None
    cols_lower = {c.lower(): c for c in data.columns}

    for t in text_candidates:
        if t in cols_lower:
            text_col = cols_lower[t]
            break
    for l in label_candidates:
        if l in cols_lower:
            label_col = cols_lower[l]
            break

    if text_col is None or label_col is None:
        msg = (
            "Could not find expected 'text' and 'label' columns in the CSV. "
            f"Found columns: {list(data.columns)}.\n"
            "Please provide a CSV with a text column (e.g. 'text' or 'review') and a label column (e.g. 'label' or 'sentiment')."
        )
        logger.error(msg)
        raise KeyError(msg)

    logger.info(f"Using text column '{text_col}' and label column '{label_col}'")
    X_train, X_test, y_train, y_test = train_test_split(data[text_col], data[label_col], test_size=0.2, random_state=42)
    
    # Determine label encoding and number of classes
    le = LabelEncoder()
    le.fit(list(y_train) + list(y_test))
    n_classes = len(le.classes_)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Save label encoder for downstream use
    joblib.dump(le, TEXT_MODEL_DIR / 'label_encoder.pkl')
    logger.info(f"Detected {n_classes} classes: {list(le.classes_)}")

    # Use transformer-based embeddings if available, else fall back to LSTM with Keras Embedding
    if TRANSFORMERS_AVAILABLE:
        logger.info("Using DistilBERT transformer for text embeddings (high accuracy)")
        train_transformer_model(X_train, X_test, y_train_enc, y_test_enc, n_classes)
    else:
        logger.info("Transformers not available; using LSTM with Keras Embedding")
        train_lstm_model(X_train, X_test, y_train_enc, y_test_enc, n_classes)


def train_transformer_model(X_train, X_test, y_train, y_test, n_classes):
    """Train using DistilBERT transformer for better NLP accuracy."""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    
    # Tokenize and encode
    logger.info("Tokenizing training texts with DistilBERT")
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors='tf')
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128, return_tensors='tf')
    
    # Get embeddings from DistilBERT
    logger.info("Extracting embeddings from DistilBERT")
    train_embeddings = distilbert_model(train_encodings['input_ids'])[0][:, 0, :].numpy()  # [CLS] token
    test_embeddings = distilbert_model(test_encodings['input_ids'])[0][:, 0, :].numpy()
    
    # Build classifier on top of embeddings
    if n_classes == 2:
        output_units = 1
        output_activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        output_units = n_classes
        output_activation = 'softmax'
        loss = 'categorical_crossentropy'

    model = Sequential([
        Dense(512, activation='relu', input_shape=(train_embeddings.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_units, activation=output_activation)
    ])
    model.compile(optimizer=Adam(1e-3), loss=loss, metrics=['accuracy'])
    
    logger.info("Training classifier on DistilBERT embeddings")
    # Prepare labels for training
    if n_classes == 2:
        y_train_final = np.array(y_train)
    else:
        y_train_final = to_categorical(y_train, num_classes=n_classes)

    model.fit(train_embeddings, y_train_final, epochs=10, batch_size=32, validation_split=0.1)
    
    # Evaluate
    y_pred_proba = model.predict(test_embeddings)
    if n_classes == 2:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    cls_report_dict = classification_report(y_test, y_pred, output_dict=True)
    cls_report_text = classification_report(y_test, y_pred)
    
    # Save model
    model.save(str(SAVE_MODEL))
    joblib.dump(tokenizer, str(SAVE_VECT))
    logger.info(f"Saved transformer-based text model to {SAVE_MODEL}")
    
    write_reports(acc, cls_report_dict, cls_report_text, len(train_embeddings))


def train_lstm_model(X_train, X_test, y_train, y_test, n_classes):
    """Train using LSTM with Keras Embedding (fallback when transformers unavailable)."""
    # Tokenize
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, maxlen=100, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=100, padding='post')
    
    # Build LSTM model
    # Configure output for binary vs multiclass
    if n_classes == 2:
        output_units = 1
        output_activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        output_units = n_classes
        output_activation = 'softmax'
        loss = 'categorical_crossentropy'

    model = Sequential([
        Embedding(5000, 128),
        LSTM(128, return_sequences=True, input_shape=(100,)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(output_units, activation=output_activation)
    ])
    model.compile(optimizer=Adam(1e-3), loss=loss, metrics=['accuracy'])
    
    logger.info("Training LSTM text classifier")
    # Prepare labels for training
    if n_classes == 2:
        y_train_final = np.array(y_train)
    else:
        y_train_final = to_categorical(y_train, num_classes=n_classes)

    model.fit(X_train_padded, y_train_final, epochs=5, batch_size=32, validation_split=0.1)
    
    # Evaluate
    y_pred_proba = model.predict(X_test_padded)
    if n_classes == 2:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    cls_report_dict = classification_report(y_test, y_pred, output_dict=True)
    cls_report_text = classification_report(y_test, y_pred)
    
    # Save model
    model.save(str(SAVE_MODEL))
    joblib.dump(tokenizer, str(SAVE_VECT))
    logger.info(f"Saved LSTM text model to {SAVE_MODEL}")
    
    write_reports(acc, cls_report_dict, cls_report_text, len(X_train_padded))


def write_reports(acc, cls_report_dict, cls_report_text, n_samples):
    """Write JSON and text reports."""
    report = {
        'n_samples': int(n_samples),
        'accuracy': float(acc),
        'classification': cls_report_dict,
    }
    try:
        with open(REPORT_JSON, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        logger.info(f"Wrote JSON report to {REPORT_JSON}")
    except Exception as e:
        logger.warning(f"Failed writing JSON report: {e}")

    try:
        with open(REPORT_TEXT, 'w', encoding='utf-8') as fh:
            fh.write(f"Test accuracy: {acc:.4f}\n\n")
            fh.write(str(cls_report_text))
        logger.info(f"Wrote text report to {REPORT_TEXT}")
    except Exception as e:
        logger.warning(f"Failed writing text report: {e}")

    print(f"Saved text model and reports to {TEXT_MODEL_DIR}")


if __name__ == '__main__':
    main()
