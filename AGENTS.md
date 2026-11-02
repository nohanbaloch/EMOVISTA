# EMOVISTA — Agent Guidelines & Repository Directives

This document (`AGENTS.md`) is the definitive operational guide for AI agents (Antigravity, Copilot, Claude, etc.) and developers working in the **EMOVISTA** codebase. It outlines the core architecture, immutable invariants, model contracts, security boundaries, and verified commands.

---

## 1. Project Overview & Operational Boundaries

EMOVISTA is a real-time multimodal emotion-aware conversational assistant. It combines:
1. **Facial Expression Recognition (FER)**: Computer vision model (`(100, 100, 1)` grayscale CNN).
2. **Speech Emotion Recognition (SER)**: 1D-CNN on acoustic features extracted from CREMA-D audio.
3. **Text Sentiment Analysis**: Binary sentiment mapped to confidence-based 3-way distribution.
4. **Multimodal Weighted Fusion**: 50% Visual, 30% Speech, 20% Text, dynamically renormalized.
5. **Conversational AI**: Local Ollama Phi-3 Mini (`phi3`), offline STT (Vosk), and isolated subprocess TTS (`pyttsx3`).

> [!CAUTION]
> ### 🛑 Non-Diagnostic & Safety Invariant
> EMOVISTA is an empathetic AI conversational companion, **NOT** a doctor, clinical psychologist, or diagnostic medical device.
> - **Descriptive language only**: Never use clinical diagnoses (e.g. "depression", "anxiety disorder") for emotion states. Use descriptive terms (e.g. "feeling low / downhearted", "worried / heightened alertness").
> - **Deterministic Crisis Intercept**: Do not rely on LLM prompts for crisis management. Critical self-harm/suicide keywords MUST trigger the deterministic crisis protocol in Python (`is_crisis_text()`) before sending any prompt to an LLM.
> - **No medical claims**: Do not describe the severity score as a clinical measure. It is a non-diagnostic persistence and intensity heuristic.

---

## 2. Verified File Tree & Canonical Entry Points

```text
EMOVISTA/
├── app.py                         # Web Application & API (Flask)
├── customtkinter-main.py          # Desktop Application GUI (CustomTkinter)
├── requirements.txt               # Pinned dependencies
├── README.md                      # General user overview
├── SETUP.md                       # Environment setup & run guide
├── AGENTS.md                      # THIS FILE: Agent guidance and invariants
├── scripts/
│   └── evaluate_fer.py            # Diagnostic & evaluation script for FER model
├── tests/
│   ├── test_fusion.py             # 9 unit tests for multimodal fusion (TensorFlow-independent)
│   └── test_web_app.py            # 12 integration tests for web API, security & degradation
├── docs/
│   └── ARCHITECTURE.md            # In-depth architectural & mathematical specification
├── models/
│   ├── Fer_model/
│   │   ├── fer_model.keras        # Shipped FER CNN model (Input: 100x100x1 grayscale)
│   │   └── report_and_log/        # FER training CSVs and history
│   ├── speech_model/
│   │   ├── speech_model.keras     # Shipped 1D-CNN SER model (Input: 40 MFCC + acoustic features)
│   │   ├── speech_label_encoder.pkl # 6 CREMA-D classes
│   │   └── report_and_log/        # SER reports and history
│   ├── text_model/
│   │   ├── text_model.keras       # Shipped binary sentiment model
│   │   ├── vectorizer.pkl         # Text Tokenizer
│   │   ├── label_encoder.pkl      # Binary label encoder ('negative', 'positive')
│   │   └── report_and_log/        # Text training logs
│   └── vosk/
│       └── vosk-model-small-en-us-0.15/ # Shipped offline speech-to-text model
└── src/
    ├── main.py                    # Real-time webcam CLI assistant
    ├── fer/
    │   └── train_fer.py           # Canonical FER training script (class weights, staged LR)
    ├── fusion/
    │   ├── emotion_fusion.py      # Canonical weighted fusion engine
    │   ├── severity_engine.py     # Non-diagnostic persistence & intensity heuristic
    │   └── train_meta.py          # Optional meta-model training script
    ├── memory/
    │   ├── crypto_utils.py        # Fernet AES key loader (anchored to data/memory/)
    │   └── patient_memory.py      # Encrypted session interaction persistence
    ├── speech/
    │   ├── speech_model.py        # Model definitions
    │   └── train_speech_cremad.py # Canonical SER trainer with GroupShuffleSplit by ActorID
    ├── text/
    │   └── train_text.py          # Text sentiment training script
    └── web/
        ├── backend/
        │   ├── llm_phi3.py        # Phi-3 integration & deterministic crisis checks
        │   ├── stt_vosk.py        # Vosk STT wrapper with fallback error reporting
        │   ├── tts.py             # TTS async queue worker
        │   ├── independent_tts.py # Subprocess pyttsx3 runner (avoids COM lockups)
        │   └── persona_config.json # Non-diagnostic assistant persona
        └── frontend/
            ├── static/            # Frontend JS (app.js) and CSS (style.css)
            └── templates/         # index.html
```

---

## 3. Mathematical Contracts & Model Specifications

### Emotion Class Mapping
The unified emotion label list is fixed:
```python
fer_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
```

### Modality Alignments & Constraints
1. **Visual (FER)**:
   - Shipped Model: `models/Fer_model/fer_model.keras`
   - Input shape: `(None, 100, 100, 1)` (Grayscale normalized to `[0.0, 1.0]`).
   - Output: 7 softmax probabilities.
   - Directory class ordering is alphabetical: `0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise`.
2. **Speech Emotion Recognition (SER)**:
   - Shipped Model: `models/speech_model/speech_model.keras`
   - Input shape: `(None, 40, 1)`
   - Classes: 6 CREMA-D classes (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`).
   - **Asymmetry Invariant**: CREMA-D does NOT contain *Surprise*. Surprise probability from audio is always 0.0. Surprise can ONLY originate from visual FER.
3. **Text Sentiment Analysis**:
   - The text model outputs binary sentiment probabilities: $p_{\text{pos}}$ and $p_{\text{neg}}$.
   - Binary output is mapped to a 3-way distribution `[P(neg), P(neu), P(pos)]` where Neutral reflects **classifier uncertainty**, not a confident neutral assertion:
     $$\text{Confidence } C = |p_{\text{pos}} - p_{\text{neg}}| \in [0.0, 1.0]$$
     $$P(\text{neu}) = 1.0 - C$$
     $$P(\text{neg}) = (1.0 - P(\text{neu})) \times \frac{p_{\text{neg}}}{p_{\text{neg}} + p_{\text{pos}}}$$
     $$P(\text{pos}) = (1.0 - P(\text{neu})) \times \frac{p_{\text{pos}}}{p_{\text{neg}} + p_{\text{pos}}}$$
   - Mapped to FER space: Negative $\to$ Sad, Neutral $\to$ Neutral, Positive $\to$ Happy.

### Weighted Fusion Formulation
- Base Weights: $W_{\text{fer}} = 0.50$, $W_{\text{speech}} = 0.30$, $W_{\text{text}} = 0.20$.
- Dynamic Renormalization: When modality $k$ is missing or malformed, active weights re-normalize:
  $$W_i' = \frac{W_i}{\sum_{m \in \text{active}} W_m}$$
- Fallback: If all inputs are absent/invalid, returns `('Neutral', [0, 0, 0, 0, 0, 0, 1.0])`.

---

## 4. Hard Rules for AI Agents

1. **Do not retrain models blindly**:
   - Training datasets (`data/FER-2013`, `data/CREMA-D`) are intentionally gitignored and missing from the repo.
   - Do not download datasets or launch long training jobs without explicit user instruction.
   - Never claim an accuracy improvement over baseline (FER: 0.2471, SER: 0.4050) unless measured on actual test data.
2. **Never commit sensitive artifacts**:
   - Never commit `.key` files (e.g. `patient.key`) or `.enc` files (`patient_memory.enc`).
   - Never commit `__pycache__`, `.pyc`, or TensorBoard `.tfevents` files.
   - Never commit raw model binary weights without a Git LFS / storage plan.
3. **Preserve framework stack**:
   - Do not switch frameworks (Flask for Web, CustomTkinter for Desktop, OpenCV for CV, Keras/Torch for DL).
   - Do not introduce Streamlit (removed legacy dependency).
4. **Preserve deterministic crisis interception**:
   - Any modification to user consultation routes (`/consult` or desktop chat) must preserve the `is_crisis_text()` deterministic check before calling Ollama/Phi-3.
5. **Maintain test coverage**:
   - `tests/test_fusion.py` MUST remain runnable with zero TensorFlow dependency (pure NumPy/Python).

---

## 5. Developer & Agent Command Reference

### Running Tests
```bash
# Run all unit and integration tests (21 tests)
python -m unittest discover tests

# Run fusion tests only (fast, non-TensorFlow)
python -m unittest tests/test_fusion.py

# Run web app hardening & route tests
python -m unittest tests/test_web_app.py
```

### Running Diagnostics
```bash
# Evaluate shipped FER model and run synthetic probes
python scripts/evaluate_fer.py

# Evaluate with custom test dataset if available
python scripts/evaluate_fer.py --test-dir data/FER-2013/test
```

### Launching Applications
```bash
# Start Flask Web Dashboard (http://127.0.0.1:5000, Debug OFF by default)
python app.py

# Start Flask with Debug ON (local development only)
FLASK_DEBUG=1 python app.py

# Start CustomTkinter Desktop GUI
python customtkinter-main.py

# Start CLI Webcam Real-Time Demo
python src/main.py
```

---

## 6. Serena Integration
Serena project configuration is active at `.serena/project.yml`.
Memories stored in `.serena/memories/`:
- `mem:core` — High-level system map and invariants.
- `mem:tech_stack` — Exact frameworks and version bounds.
- `mem:suggested_commands` — Verified project commands.
- `mem:conventions` — Coding conventions, label orders, and boundaries.
- `mem:task_completion` — Pre-completion checklist.
To verify Serena memory links: `serena memories check`.
