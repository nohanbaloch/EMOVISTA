# EMOVISTA - Real-Time Multimodal Emotion-Aware Assistant

EMOVISTA is a real-time multimodal emotion-aware conversational assistant. It fuses visual **Facial Expression Recognition (FER)**, acoustic **Speech Emotion Recognition (SER)**, and **Text Sentiment Analysis** into a unified 7-emotion representation:
**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

> [!IMPORTANT]
> **Non-Diagnostic Notice**: EMOVISTA is an AI conversational and emotional support companion, **NOT** a doctor, psychologist, or medical diagnostic device. It does not provide clinical diagnoses, psychiatric assessments, or medical treatment plans.

---

## 🚀 Key System Components

### 1. Multimodal Weighted Fusion (`src/fusion/emotion_fusion.py`)
Fuses three distinct modalities using a robust weighted combination:
- **Visual (FER)**: 50% contribution. Evaluates facial landmarks and expressions across all 7 emotions. *Surprise* is exclusively detected via visual FER.
- **Audio (SER)**: 30% contribution. Analyzes pitch, prosody, and spectral features across 6 CREMA-D emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`).
- **Text Sentiment**: 20% contribution. Evaluates user input. The binary sentiment probability vector `[P(neg), P(pos)]` is mapped to a 3-way distribution `[P(neg), P(neu), P(pos)]` where Neutral reflects low classifier confidence / uncertainty (`1 - |P(pos) - P(neg)|`).

When one or more modalities are unavailable, active weights are dynamically renormalized so they sum to 1.0. If all inputs are absent, the system gracefully defaults to Neutral.

### 2. Conversational Assistant & Voice
- **Local LLM**: Powered by **Phi-3 Mini** via Ollama for private, local conversational responses.
- **Deterministic Crisis Safety Protocol**: High-risk self-harm and crisis phrases trigger immediate, deterministic supportive guidance and referral to emergency lifelines (e.g. 988 in the US/Canada, 111/999 in the UK, 112 in the EU).
- **Streaming TTS**: Spoken responses generated chunk-by-chunk via an isolated subprocess engine (`pyttsx3`).
- **Offline STT**: Voice transcription powered by **Vosk** using the shipped local model (`models/vosk/vosk-model-small-en-us-0.15`).

### 3. Emotion Intensity & Privacy
- **Emotion Intensity Heuristic (`src/fusion/severity_engine.py`)**: A transparent 0–100 persistence and intensity metric tracking detected negative emotion duration. Explicitly non-diagnostic.
- **Encrypted Local Session Logs (`src/memory/patient_memory.py`)**: Local session history encrypted at rest using AES-128-CBC via Fernet (`cryptography`).

### 4. Interfaces
- **Web Dashboard**: Modern Flask web application (`app.py`).
- **Desktop Application**: High-performance GUI built with CustomTkinter (`customtkinter-main.py`).
- **Webcam CLI Entry Point**: Lightweight real-time OpenCV demo (`src/main.py`).

---

## 📂 Project Structure

```text
EMOVISTA/
├── app.py                     # ENTRY POINT: Flask Web Application & API
├── customtkinter-main.py      # ENTRY POINT: CustomTkinter Desktop GUI
├── requirements.txt           # Pinned project dependencies
├── README.md                  # System overview and architecture
├── SETUP.md                   # Installation and environment configuration
├── scripts/
│   └── evaluate_fer.py        # FER diagnostic evaluation tool
├── tests/
│   └── test_fusion.py         # Unit tests for multimodal fusion (TensorFlow-independent)
├── models/
│   ├── Fer_model/
│   │   ├── fer_model.keras    # Shipped FER CNN model (100x100 grayscale)
│   │   └── report_and_log/    # FER training history and metrics
│   ├── speech_model/
│   │   ├── speech_model.keras # Shipped 1D-CNN SER model
│   │   ├── speech_label_encoder.pkl
│   │   └── report_and_log/    # SER training history and metrics
│   ├── text_model/
│   │   ├── text_model.keras   # Shipped binary text sentiment model
│   │   ├── vectorizer.pkl     # Text tokenizer
│   │   ├── label_encoder.pkl  # Text label encoder
│   │   └── report_and_log/    # Text training history and metrics
│   └── vosk/
│       └── vosk-model-small-en-us-0.15/ # Shipped offline speech-to-text model
└── src/
    ├── main.py                # ENTRY POINT: Real-time webcam CLI assistant
    ├── fer/
    │   └── train_fer.py       # FER training pipeline with class weights
    ├── fusion/
    │   ├── emotion_fusion.py  # Canonical multimodal fusion engine
    │   ├── severity_engine.py # Heuristic emotion intensity tracking
    │   └── train_meta.py      # Meta-model training script
    ├── memory/
    │   ├── crypto_utils.py    # Fernet key management
    │   └── patient_memory.py  # Encrypted session persistence
    ├── speech/
    │   ├── speech_model.py    # Speech model architecture definitions
    │   └── train_speech_cremad.py # SER training pipeline with speaker grouping
    ├── text/
    │   └── train_text.py      # Text sentiment training script
    └── web/
        ├── backend/
        │   ├── llm_phi3.py    # Phi-3 integration & deterministic crisis checks
        │   ├── stt_vosk.py    # Vosk speech recognition wrapper
        │   ├── tts.py         # TTS async worker
        │   ├── independent_tts.py # Isolated TTS subprocess
        │   └── persona_config.json # Supportive assistant persona
        └── frontend/
            ├── static/        # Web assets (app.js, style.css)
            └── templates/     # index.html
```

---

## 🛠️ Quick Start

### 1. Installation
Ensure Python 3.12+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Verify Models & Run Tests
```bash
# Run fusion test suite
python -m unittest discover tests

# Run FER evaluation probe
python scripts/evaluate_fer.py
```

### 3. Launch the Application
- **Option A: Web Dashboard**
  ```bash
  python app.py
  ```
  Open `http://localhost:5000` in your browser.

- **Option B: Desktop GUI**
  ```bash
  python customtkinter-main.py
  ```

- **Option C: Webcam CLI Demo**
  ```bash
  python src/main.py
  ```
