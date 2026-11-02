# EMOVISTA - Setup & Operational Guide

## 📋 System Prerequisites
- **Python**: Version 3.12+ (tested on Windows and Linux).
- **Audio & Video Hardware**: A functional webcam (for FER) and microphone (for STT).
- **Ollama**: Optional for local conversational LLM (`phi3`).

---

## 🚀 Installation

### 1. Clone & Set Up Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venv

# Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Model Directory Configuration

The repository includes pre-trained model artifacts:

| Modality | Path | Input Dimensions | Target Output |
| :--- | :--- | :--- | :--- |
| **FER (Face)** | `models/Fer_model/fer_model.keras` | `(None, 100, 100, 1)` | 7 emotion classes |
| **SER (Speech)** | `models/speech_model/speech_model.keras` | `(None, 40, 1)` | 6 CREMA-D classes |
| **Text Sentiment** | `models/text_model/text_model.keras` | `(None, 100)` | Binary sentiment |
| **Offline STT** | `models/vosk/vosk-model-small-en-us-0.15/` | 16 kHz Mono PCM | Text transcription |

---

## ⚙️ Environment Variables

All runtime settings can be customized using environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `FLASK_DEBUG` | `false` | Set to `1` or `true` for local Flask reloader |
| `EMOVISTA_HOST` | `127.0.0.1` | Web server bind host |
| `EMOVISTA_PORT` | `5000` | Web server port |
| `EMOVISTA_ALLOWED_ORIGINS` | `http://localhost:5000,http://127.0.0.1:5000` | Allowed CORS origins (comma-separated) |
| `VOSK_MODEL_PATH` | `models/vosk/vosk-model-small-en-us-0.15` | Path to unpacked Vosk model directory |
| `MAX_IMAGE_SIZE_BYTES` | `5242880` (5 MB) | Maximum accepted image payload on `/analyze_face` |

---

## 🧪 Verification & Testing

### 1. Run Multimodal Fusion Unit Tests
```bash
python -m unittest tests/test_fusion.py
```
*Tests all modalities present, missing modalities, malformed vector lengths, and sentiment mappings.*

### 2. Run FER Diagnostic Evaluation
```bash
python scripts/evaluate_fer.py
```
*Evaluates the shipped FER model. If `data/FER-2013/test/` is present, computes per-class precision, recall, and accuracy; otherwise runs synthetic probe analysis.*

### 3. Verify Model Loading
```bash
python -c "from src.fusion.emotion_fusion import load_all; fer, spk, txt, vec, le = load_all(verbose=True); print('Models ready!')"
```

---

## 🖥️ Running the Application

### 1. Web Application (Flask Dashboard)
```bash
python app.py
```
Access the dashboard at `http://127.0.0.1:5000`. If the Vosk speech model or microphone is unavailable, `/stt` returns a clear 503 error status code while face and text analysis continue to work seamlessly.

### 2. Desktop GUI (CustomTkinter)
```bash
python customtkinter-main.py
```

### 3. Webcam CLI Demo
```bash
python src/main.py
```
Press `q` to quit the webcam view.
