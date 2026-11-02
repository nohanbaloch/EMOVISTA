# EMOVISTA Core Architecture

EMOVISTA is a multimodal emotion-aware assistant combining Facial Expression Recognition (FER), Speech Emotion Recognition (SER), and Text Sentiment Analysis into 7 emotion classes.

## Invariants & Architecture
- 7 target emotion labels: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral (`fer_labels` in `src/fusion/emotion_fusion.py`).
- Surprise is exclusively detected via visual FER (CREMA-D speech does not include surprise).
- Fusion engine combines FER (~50%), SER (~30%), Text (~20%) with dynamic weight renormalization when modalities are missing.
- Dual UI: Web dashboard in `app.py` (Flask) and Desktop GUI in `customtkinter-main.py`.
- Real-time webcam CLI entrypoint: `src/main.py`.
- Non-medical AI assistant boundary: EMOVISTA is an empathetic support tool, NOT a diagnostic medical device.
- Safety & Crisis: Deterministic keyword fallback with supportive messaging and referral to emergency/crisis support.

## Related Memories
- Tech stack details: `mem:tech_stack`
- Development and running commands: `mem:suggested_commands`
- Codebase style and conventions: `mem:conventions`
- Quality verification and completion checks: `mem:task_completion`
