# EMOVISTA Project Presentation

---

## Slide 1: Title
**EMOVISTA** – Real‑Time Multimodal Emotion‑Aware Assistant

---

## Slide 2: Problem Statement
- Understanding human emotions in real‑time is critical for therapeutic, safety, and interactive AI applications.
- Existing solutions often rely on a single modality (vision or audio) and require cloud services.

---

## Slide 3: Key Features
- **Multimodal Fusion** – combines facial expression (FER), speech emotion (SER), and text sentiment.
- **Offline‑First** – all models run locally, no external API calls.
- **Medical‑Ready** – encrypted patient memory, severity scoring, emergency escalation.
- **Rich UI** – desktop (CustomTkinter) and web (Flask) interfaces with dark‑mode aesthetics.

---

## Slide 4: Architecture Overview
```
+-------------------+      +-------------------+
|   Front‑End UI    | ---> |   Flask Backend   |
+-------------------+      +-------------------+
                               |
                               v
                +-------------------------------+
                |   Fusion Engine (emotion_fusion) |
                +-------------------------------+
                 /        |          \        \
                v         v           v        v
          FER Model   Speech Model   Text Model   Vosk STT
```
- All models are stored under `models/`.
- `Vosk` provides offline speech‑to‑text.
- `tts.py` gives spoken feedback.

---

## Slide 5: Vosk Model Update
- Switched to **vosk‑model‑en‑us‑0.22** for higher accuracy.
- Updated `app.py` and README accordingly.

---

## Slide 6: Text Emotion Fusion
- `/consult` endpoint now:
  1. Predicts text sentiment using `text_model`.
  2. Converts frontend FER label to a one‑hot probability vector.
  3. Calls `fuse()` to obtain a fused emotion.
  4. Generates assistant response based on the fused label.

---

## Slide 7: Startup Greeting
- On server start, the system announces: *"Emovista system online and ready."*
- Uses `tts.speak()` with a guard (`WERKZEUG_RUN_MAIN`) to avoid double‑speaking during Flask reload.

---

## Slide 8: Demo Flow (Web UI)
1. User opens `http://localhost:5000`.
2. Webcam captures face → FER prediction.
3. User types or speaks text → text sentiment.
4. Fusion engine produces final emotion.
5. Assistant replies with voice feedback.

---

## Slide 9: Security & Privacy
- Patient data encrypted with AES (`patient_memory.py`).
- No cloud calls – all processing stays on‑device.
- Severity engine flags high‑risk emotional states.

---

## Slide 10: Practical Use in the Medical Field
- **Patient Monitoring**: Real‑time emotional cues during tele‑health sessions for clinicians.
- **Therapeutic Feedback**: Detect distress or anxiety and trigger calming interventions (audio, guided breathing).
- **Emergency Escalation**: Severity engine alerts caregivers when negative emotions persist.
- **Privacy‑First**: Local processing and encryption meet HIPAA‑like requirements.
- **Integration**: Backend can be called from EHR systems via REST APIs to log emotion scores alongside vital signs.

---

## Slide 11: Future Work
- Train a learned fusion model for higher accuracy.
- Add multilingual Vosk models.
- Integrate more expressive TTS voices.
- Deploy as a Docker container for easy distribution.

---

## Slide 12: Get Started
```bash
# Clone repo
git clone https://github.com/nohanbaloch/EMOVISTA.git
cd EMOVISTA

# Install dependencies
pip install -r requirements.txt

# Download Vosk model
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip -d models/vosk/

# Run the web backend
python src/web/backend/app.py
```

---

## Slide 13: Thank You
- Questions?
- Contact: **Nohan Baloch** – `nohan@example.com`

---
