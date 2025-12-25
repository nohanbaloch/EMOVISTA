# EMOVISTA - Real-Time Multimodal Emotion-Aware Assistant

EMOVISTA is an advanced, multimodal AI platform designed to detect and analyze human emotions in real-time. By fusing data from **Facial Expression Recognition (FER)**, **Speech Emotion Recognition (SER)**, and **Text Sentiment Analysis**, it provides a unified and highly accurate emotional profile of the user.

Designed with privacy and medical applications in mind, EMOVISTA features encrypted memory, severity tracking, and emergency escalation protocols.

---

## 🚀 Key Features

### 1. Multimodal Fusion Engine

The core of EMOVISTA is its intelligent fusion engine (`src/fusion/emotion_fusion.py`) that combines inputs from three distinct models:

- **Visual (FER)**: Analyzes facial landmarks and expressions (Contribution: ~50%).
- **Audio (SER)**: Analyzes tone, pitch, and prosody using CREMA-D trained models (Contribution: ~30%).
- **Text**: Analyzes spoken or typed content for sentiment (Contribution: ~20%).

The system uses a **weighted voting mechanism** by default but supports a **Learned Fusion Model** for higher accuracy if trained.

### 2. Medical & Safety Modules

EMOVISTA is equipped with features tailored for therapeutic and medical monitoring:

- **Encrypted Patient Memory**: Patient sessions and emotional history are stored securely using AES encryption (`src/memory/patient_memory.py`).
- **Severity Engine**: Calculates an aggregate "Severity Score" (0-100) based on negative emotion persistence and intensity.
- **Emergency Escalation**: Automatically flags high-risk states (e.g., prolonged distress) to trigger alerts.
- **Trend Analysis**: Tracks emotional trajectories over time to aid in diagnosis or progress monitoring.

### 3. Dual Interfaces

- **Desktop Application**: A high-performance GUI built with **CustomTkinter** for local, low-latency interaction.
- **Web Dashboard**: A **Flask**-based web server for remote monitoring or lightweight access.

---

## 📂 Project Structure

```text
EMOVISTA/
├── data/                   # Placeholder for datasets (FER2013, CREMA-D, etc.)
├── models/                 # Directory for trained model binaries
│   ├── fer_model.h5        # Facial Expression Recognition Model
│   ├── speech_model/       # Speech Emotion Recognition Model
│   ├── text_model.pkl      # Text Analysis Model
│   └── vosk/               # Offline Speech-to-Text Model
├── src/
│   ├── main.py             # ENTRY POINT: Desktop Application
│   ├── web/
│   │   └── backend/app.py  # ENTRY POINT: Web Application
│   ├── fusion/             # Fusion logic (Weighted & Learned)
│   ├── memory/             # Encrypted patient history usage
│   ├── analytics/          # Trend analysis algorithms
│   ├── safety/             # Emergency escalation logic
│   ├── voice/              # TTS and Audio processing
│   ├── fer/                # Face detection & processing
│   ├── speech/             # Audio feature extraction
│   └── text/               # NLP & Sentiment logic
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Webcam
- Microphone

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: You may need to install system-level dependencies for `pyaudio` or `opencv` depending on your OS (e.g., `portaudio` on macOS/Linux).*

### 2. Model Setup

EMOVISTA requires pre-trained models. Place them in the `models/` directory:

- **FER**: `models/fer_model.h5` or `.keras`
- **Speech**: `models/speech_model/`
- **Text**: `models/text_model.pkl` & `vectorizer.pkl`
- **Vosk**: Download a small English model from [Vosk Models](https://alphacephei.com/vosk/models) and unzip to `models/vosk/`

---

## 🖥️ Usage

### Option A: Desktop Application (Recommended)

Launch the full-featured GUI for the best experience (real-time video feedback, interactive charts).

```bash
python src/main.py
```

**Controls**:

- **Q**: Quit the application.
- **Text Input**: Type manually if voice input is not desired.

### Option B: Web Dashboard

Start the Flask server to access the web interface.

```bash
python src/web/backend/app.py
```

Access the dashboard at: `http://localhost:5000`

---

## 🔬 Architecture Details

### The Fusion Algorithm

The `fuse()` function in `src/fusion/emotion_fusion.py` orchestrates the decision making:

1. **Input**: Receives probability vectors from FER, Speech, and Text models.
2. **Normalization**: Aligns all modalities to a standard 7-emotion scale (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
3. **Weighting**: Applies confidence weights (Visual > Audio > Text).
4. **Decision**: Outputs the final predicted emotion and a fused confidence score.
