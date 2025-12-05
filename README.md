# EMOVISTA---Emotion-Aware-AI-Assistant
EMOVISTA is an advanced emotion-analysis platform that fuses facial expressions, voice tone, and text sentiment into one unified emotional understanding. Built for researchers, developers, and human-centric AI applications, EMOVISTA provides a complete toolkit for real-time emotion recognition, visualization, and interaction


# EmotionAwareAI
Full Emotion-Aware AI Assistant (FER + Speech (CREMA-D) + Text Fusion)

## Structure
- `data/` - placeholder for datasets (FER2013, CREMA-D, IMDB)
- `models/` - trained model files (saved after training)
- `src/` - source code: training, utils, fusion, GUI, main
- `requirements.txt` - Python deps

## Quick start
1. Install requirements: `pip install -r requirements.txt`
2. Put datasets under `data/`:
   - FER: `data/fer/fer2013.csv`
   - CREMA-D: extracted WAV files in `data/speech/crema-d/`
   - Text (IMDB): `data/text/imdb.csv` (columns: text,label)
3. Train models:
   - FER: `python src/fer/train_fer.py`
   - SER (CREMA-D): `python src/speech/train_speech_cremad.py`
   - Text: `python src/text/train_text.py`
4. Run the real-time assistant: `python src/main.py`

Notes:
- The repository includes full, ready-to-run scripts for preprocessing and training.
- Models are not included due to size; train them or load your own into `models/`.
