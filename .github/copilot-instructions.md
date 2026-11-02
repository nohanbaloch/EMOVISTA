<!-- Copied/merged guidance for AI coding agents working on this repo -->
# Copilot / AI Agent Instructions — Emotion-Aware AI Assistant

Purpose: help an AI agent be immediately productive in this repository by documenting architecture, developer workflows, important file locations, and project-specific patterns.

- **Big picture**: this is a local, single-repo Emotion-Aware assistant that fuses three modalities:
  - Face-based FER (computer vision) in `src/fer/`.
  - Speech Emotion Recognition using CREMA-D in `src/speech/`.
  - Text classification in `src/text/`.
  - Runtime fusion and app entrypoint: `src/fusion/emotion_fusion.py` and `src/main.py` (real-time assistant using webcam + microphone).

- **Key directories & files**:
  - `data/` — datasets (FER, CREMA-D WAVs, text CSVs). Many scripts rely on `data/` relative to `src/`.
  - `models/` — trained models and saved artifacts (label encoders, vectorizers).
  - `src/fer/train_fer.py` — CNN training for FER. NOTE: current script saves `fer2013_cnn.h5` in its working dir.
  - `src/speech/train_speech_cremad.py` — feature extraction + training; saves features to `data/speech/` and model to `models/speech_model.h5`.
  - `src/text/train_text.py` — trains text model and saves to `models/text_model.pkl` and `models/vectorizer.pkl`.
  - `src/fusion/emotion_fusion.py` — loads models from `models/` and implements label mapping + fusion weights. `fer_labels` defines the FER output ordering used throughout.
  - `src/main.py` — runtime script that opens webcam, captures audio, performs predictions, and displays fused label.

- **Important conventions & discoverable quirks** (agents must respect these):
  - Many scripts use relative paths computed with `os.path.join(os.path.dirname(__file__), '../../...')`. Run scripts from the repository root (or ensure `__file__`-relative paths resolve correctly).
  - Model filenames expected by fusion:
    - FER: `models/fer_model.keras` (see `FER_MODEL_PATH` in `src/fusion/emotion_fusion.py`).
    - Speech: `models/speech_model.keras` and `models/speech_label_encoder.pkl`.
    - Text: `models/text_model.pkl` and `models/vectorizer.pkl`.
    - Action: if you train locally, copy/rename outputs to these exact paths or update the constants in `emotion_fusion.py`.
  - `src/fer/train_fer.py` currently saves `fer2013_cnn.h5` (not `models/fer_model.h5`) — update that script or move the file after training.
  - `src/fer/train_fer.py` uses a hard-coded `DATASET_PATH = "\Data\FER-2013"` which is likely incorrect on Windows (leading backslash) — prefer using a `data/` relative path or read from env/args.

- **Developer workflows / commands** (PowerShell examples):
  - Setup virtualenv (if needed):
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
  - Train FER (edit `DATASET_PATH` or run from root after placing data):
    ```powershell
    python src/fer/train_fer.py
    # then move/rename the model to models/fer_model.h5 if needed
    mv src\fer\fer2013_cnn.h5 models\fer_model.h5
    ```
  - Train speech model (CREMA-D):
    ```powershell
    # place WAV files into data/speech/crema-d
    python src/speech/train_speech_cremad.py
    ```
  - Train text model:
    ```powershell
    # ensure data/text/imdb.csv exists (columns: text,label)
    python src/text/train_text.py
    ```
  - Run real-time assistant (webcam + mic):
    ```powershell
    python src/main.py
    ```

- **Patterns an AI agent should follow when editing code**:
  - Prefer changing model paths/constants in one place: `src/fusion/emotion_fusion.py` or update training scripts to write into `models/` directly.
  - Keep relative path style consistent: use `os.path.join(os.path.dirname(__file__), '..', '..', 'models', '...')` rather than hard-coded backslashes.
  - Preserve label ordering: `fer_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`. Any change to ordering must be mirrored where predictions are interpreted (e.g., `src/main.py`, fusion code).
  - When adding new tests or CI, ensure hardware-dependent code (webcam/mic) is stubbed or guarded behind a flag so CI runs do not require devices.

- **Integration points & external deps**:
  - Audio: `librosa`, `sounddevice` (real-time mic input).
  - Vision: `opencv-python` (webcam and face detection using Haar cascades from OpenCV shipped data).
  - ML: `tensorflow` / `keras`, `scikit-learn` for vectorizers and label encoders.
  - Data files: CREMA-D WAVs and FER dataset expected under `data/` — scripts check existence and raise `FileNotFoundError` if absent.

- **Quick fixes agents commonly need to make** (actionable examples):
  - Update `src/fer/train_fer.py` to save to `models/fer_model.h5` instead of `fer2013_cnn.h5`.
  - Fix `DATASET_PATH` in `train_fer.py` to use a relative `data/FER-2013` path or accept a CLI arg.
  - If changing model I/O names, update `FER_MODEL_PATH`, `SPEECH_MODEL_PATH`, and `TEXT_MODEL_PATH` in `src/fusion/emotion_fusion.py`.

If something in this file is unclear or you'd like additional examples (e.g., an automated small patch to change the FER save path), tell me which area to expand and I will iterate.
