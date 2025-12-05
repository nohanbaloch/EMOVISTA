Web UI for Emotion-Aware AI Assistant

Run the Flask web app from the repository root:

```powershell
python src\web\app.py
```

Open http://localhost:5000 in your browser. Use the form to upload an image (face), optional audio (wav), and text, then click Predict.

Notes:
- The backend uses `src/fusion/emotion_fusion.load_all()` to find and load models.
- If models are missing, the API will still respond but predictions will be empty.

Example curl commands

- Predict (image + text):

```powershell
curl -X POST "http://localhost:5000/api/predict" -F "image=@face.jpg" -F "text=@-" -d "This is a happy message"
```

- Export a labeled row (image + optional audio/text) into the fusion training CSV:

```powershell
curl -X POST "http://localhost:5000/api/export_row" -F "image=@face.jpg" -F "audio=@example.wav" -F "text=hello" -F "label=Happy"
```

Notes on the exporter

- The `/api/export_row` endpoint appends rows to `models/fusion_model/training_data.csv`. It creates the file and header on first write using the observed prediction vector lengths. Use this CSV to train the fusion meta-model with `src/fusion/train_meta.py`.
