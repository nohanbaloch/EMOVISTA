# Suggested Commands

## Running Applications
- Web Application:
  `python app.py` (Default: http://127.0.0.1:5000, debug off by default; use `FLASK_DEBUG=1` for dev)
- Desktop GUI:
  `python customtkinter-main.py`
- Real-time Webcam Demo:
  `python src/main.py`

## Testing & Diagnostics
- Run fusion unit tests (TensorFlow-independent):
  `python -m unittest discover tests` or `pytest tests/test_fusion.py`
- Run FER evaluation script:
  `python scripts/evaluate_fer.py`
