# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Tech stack (at a glance)
- Python project (no `pyproject.toml`); deps in `requirements.txt` and packaging via `setup.py`.
- ML/inference uses TensorFlow/Keras, scikit-learn/joblib, OpenCV, librosa.
- UIs:
  - OpenCV window + Tkinter text prompt (`src/main.py`)
  - Streamlit + WebRTC (`app.py`)
  - CustomTkinter GUI (`src/gui/main_gui.py`)

## Common commands (PowerShell)

### Environment setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Run the apps
- Real-time OpenCV multimodal assistant (camera + optional mic + Tkinter text prompt):
  ```powershell
  python .\src\main.py
  ```
  Useful flags (see `src/main.py`):
  ```powershell
  python .\src\main.py --camera 0
  python .\src\main.py --no-audio
  python .\src\main.py --audio-sr 22050
  ```

- Streamlit realtime app (browser UI via `streamlit-webrtc`):
  ```powershell
  streamlit run .\app.py
  ```

- Desktop GUI (CustomTkinter + OpenCV webcam window):
  ```powershell
  python .\src\gui\main_gui.py
  ```

### Tests
This repo uses `pytest` (see `tests/test_fusion.py`).
- Run all tests:
  ```powershell
  python -m pytest
  ```
- Run a single file:
  ```powershell
  python -m pytest .\tests\test_fusion.py
  ```
- Run a single test:
  ```powershell
  python -m pytest .\tests\test_fusion.py::test_fuse_with_fer_only
  ```
- Filter by name:
  ```powershell
  python -m pytest -k fuse
  ```

### Lint/format
No lint/format tooling is configured in-repo (no `ruff.toml`, `pyproject.toml`, `setup.cfg`, etc.).

## Training scripts (and expected data locations)
The README documents one set of dataset paths, but individual training scripts use different defaults—verify paths in each script before running.

- Face Emotion Recognition (FER):
  - Command:
    ```powershell
    python .\src\fer\train_fer.py
    ```
  - Default dataset path expected by `src/fer/train_fer.py`:
    - `data\FER-2013\train\...`
    - `data\FER-2013\test\...`
  - Outputs under `models\fer_model\` (e.g. `models\fer_model\fer_model.keras`).

- Speech Emotion Recognition (CREMA-D):
  - Command:
    ```powershell
    python .\src\speech\train_speech_cremad.py
    ```
  - Default audio directory expected by `src/speech/train_speech_cremad.py`:
    - `Data\CREMA-D\AudioWAV\...` (note the capital `Data/`)
  - Outputs under `models\speech_model\`:
    - `speech_model.keras`
    - `speech_label_encoder.pkl`

- Text model:
  - Command:
    ```powershell
    python .\src\text\train_text.py
    ```
  - Default dataset expected by `src/text/train_text.py`:
    - `data\text\imdb.csv` (must contain a text column and a label/sentiment column)
  - Outputs under `models\text_model\`.
  - Note: `src/fusion/emotion_fusion.py` currently tries to load text artifacts from `models\text_model.pkl` and `models\vectorizer.pkl` (top-level). If you train via `src/text/train_text.py`, you may need to align/copy/rename artifacts so `load_all()` can find them.

- Learned fusion meta-model (optional):
  - Command (input is a CSV of per-modality probability features):
    ```powershell
    python .\src\fusion\train_meta.py -i path\to\probs.csv -o .\models\fusion_model --model logreg
    ```
  - Expected CSV columns (see `src/fusion/train_meta.py`):
    - feature columns prefixed with `fer_`, `speech_`, `text_`
    - label column default `label`
  - Output used at runtime by the fusion layer:
    - `models\fusion_model\fusion_model.pkl`

## High-level architecture

### Core idea: fuse three modalities into a shared 7-class label space
The system treats facial emotion recognition as the “canonical” label space:
- `src/fusion/emotion_fusion.py` defines `fer_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`.

Speech and text are mapped/combined into that same 7-class space, then the final label is chosen by argmax.

### The fusion layer (`src/fusion/emotion_fusion.py`)
This is the central integration point used by the apps:
- `load_all()` loads model artifacts from `models/` (Keras for FER/speech; joblib for some artifacts).
- `fuse(fer_pred, speech_pred, speech_le, text_pred)` returns `(label, prob_vector_len_7)`.
  - If `models/fusion_model/fusion_model.pkl` exists, it attempts a learned fusion path (sklearn-like `predict_proba`).
  - Otherwise it falls back to a fixed-weight fusion:
    - FER weighted at 0.5
    - Speech projected into FER labels weighted at 0.3 (via `map_speech_to_fer()` and `speech_le.classes_`)
    - Text mapped to Sad/Neutral/Happy weighted at 0.2

### Runtime entrypoints
- `src/main.py` (CLI realtime assistant)
  - Loads all models via `fusion.emotion_fusion.load_all()`.
  - Face pipeline: webcam capture → Haar-cascade face detection → ROI resized/normalized based on model input shape → `fer_model.predict`.
  - Speech pipeline: optional `sounddevice` input stream → `extract_audio_features()` (MFCC+Mel+Chroma+ZCR+Centroid) → `speech_model.predict`.
  - Text pipeline: background Tk dialog (`TextInputGUI`) → vectorize + `text_model.predict_proba`.
  - Fuses per-frame and overlays the fused label onto the OpenCV window.

- `app.py` (Streamlit + WebRTC)
  - Injects `src/` into `sys.path` and imports `src.fusion.emotion_fusion`.
  - Video frames come from the browser; per-frame FER inference happens in a `VideoTransformerBase`.
  - Audio frames are buffered; a background worker periodically runs SER inference.
  - A second background loop periodically calls `fuse()` and updates Streamlit UI state.

- `src/gui/main_gui.py` (CustomTkinter)
  - Lightweight GUI; runs webcam processing in a background thread.
  - Does FER + (optional) text inference, then calls `fuse()` (speech is intentionally omitted here).

### Tests focus
`tests/test_fusion.py` validates fusion behavior in isolation (mapping + robustness + expected outputs) without requiring trained model files.