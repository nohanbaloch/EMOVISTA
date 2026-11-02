# Conventions

- Emotion Representation:
  - Fixed 7 classes: `['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`.
  - Speech mapping (CREMA-D): maps 6 classes to the 7-class space; `surprise` is FER-exclusive.
  - Text mapping: maps 2-class sentiment probability vector `[P(neg), P(pos)]` to 3-way distribution `[P(neg), P(neu), P(pos)]` where neutral reflects model uncertainty `1 - |P(pos) - P(neg)|`.
- Safe Medical Boundaries:
  - EMOVISTA is an AI emotional support assistant, not a doctor or diagnostic tool.
  - Descriptive language only (e.g. "feeling low / sad / distressed") instead of clinical diagnoses ("depression", "anxiety disorder").
  - Deterministic safety fallback: when crisis keywords are detected, emit immediate supportive crisis guidance and contact recommendations without relying exclusively on LLM prompts.
- Module Imports:
  - Canonical files only: avoid duplicate `_fixed.py` variants.
  - Absolute/Relative paths anchored to repo root via `Path(__file__).resolve()`.
