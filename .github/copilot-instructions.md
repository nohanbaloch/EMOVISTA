<!-- Guidance for AI coding agents and Copilot working on EMOVISTA -->
# Copilot / AI Agent Instructions — EMOVISTA Multimodal Assistant

Purpose: Help an AI agent or GitHub Copilot be immediately productive in this repository by documenting current architecture, developer workflows, important file locations, and project invariants.

- **Big picture**: EMOVISTA is a real-time multimodal emotion-aware assistant combining:
  - Face-based FER (computer vision) in `src/fer/`.
  - Speech Emotion Recognition using CREMA-D in `src/speech/`.
  - Text classification in `src/text/`.
  - Multimodal weighted fusion in `src/fusion/emotion_fusion.py`.
  - Flask web app in `app.py`, CustomTkinter desktop GUI in `customtkinter-main.py`, and real-time webcam demo in `src/main.py`.

- **Key directories & files**:
  - `data/` — datasets (FER-2013, CREMA-D WAVs, text CSVs). Gitignored.
  - `models/` — trained models and saved artifacts:
    - FER: `models/Fer_model/fer_model.keras` (Input: 100x100x1 grayscale).
    - Speech: `models/speech_model/speech_model.keras` and `models/speech_model/speech_label_encoder.pkl`.
    - Text: `models/text_model/text_model.keras`, `models/text_model/vectorizer.pkl`, and `models/text_model/label_encoder.pkl`.
    - Vosk STT: `models/vosk/vosk-model-small-en-us-0.15/`.
  - `src/fer/train_fer.py` — canonical CNN training script (mild augmentation, class weights, staged LR).
  - `src/speech/train_speech_cremad.py` — canonical SER trainer with `GroupShuffleSplit` across speakers.
  - `src/fusion/emotion_fusion.py` — weighted fusion engine (50% FER, 30% Speech, 20% Text), dynamic weight renormalization, and confidence-based 3-way sentiment mapping.
  - `src/fusion/severity_engine.py` — non-diagnostic persistence and intensity heuristic with disclaimer.
  - `src/memory/patient_memory.py` — encrypted session log store using Fernet (AES-128-CBC).
  - `scripts/evaluate_fer.py` — diagnostic evaluation script for FER model.
  - `tests/test_fusion.py` & `tests/test_web_app.py` — test suite (21 tests).

- **Important conventions & discoverable quirks**:
  - Always run scripts from the repository root or use repo-relative paths (`Path(__file__).resolve().parents[n]`).
  - Label ordering: `fer_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`.
  - CREMA-D speech does NOT contain *Surprise*. Surprise can only originate from visual FER.
  - Non-diagnostic boundaries: Never use clinical psychiatric labels (`depression`, `anxiety`). Use descriptive emotion terminology.
  - Deterministic crisis safety: Never rely on LLM prompts for crisis responses. High-risk keywords trigger `is_crisis_text()` in Python.

- **Developer workflows / commands**:
  - Run all tests:
    ```bash
    python -m unittest discover tests
    ```
  - Run FER evaluation:
    ```bash
    python scripts/evaluate_fer.py
    ```
  - Start Web Dashboard (debug OFF by default):
    ```bash
    python app.py
    ```
  - Start Desktop GUI:
    ```bash
    python customtkinter-main.py
    ```
  - Start Real-Time Webcam Demo:
    ```bash
    python src/main.py
    ```
