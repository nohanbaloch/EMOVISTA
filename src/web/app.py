from pathlib import Path
import sys
import logging
import io
from typing import Optional

from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from PIL import Image
import librosa

# ensure `src/` is on sys.path so we can import project modules
ROOT_SRC = Path(__file__).resolve().parents[1]
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from fusion.emotion_fusion import load_all, fuse, fer_labels
import csv
import os
import cv2

app = Flask(__name__, template_folder="templates", static_folder="static")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger('webapp')

# Load models once at startup
fer_model, speech_model, text_model, vectorizer, speech_le = load_all(verbose=True)


def get_model_input_spec(model) -> Optional[tuple]:
    if model is None:
        return None
    try:
        shape = getattr(model, 'input_shape', None)
        if shape is None and hasattr(model, 'inputs'):
            shape = tuple(model.inputs[0].shape.as_list())
        if shape is None:
            return None
        if len(shape) == 4:
            return (int(shape[1]), int(shape[2]), int(shape[3]))
        if len(shape) == 3:
            return (int(shape[1]), int(shape[2]), None)
    except Exception:
        return None


def preprocess_image_file(fp, spec=None):
    # fp: file-like object (binary)
    img = Image.open(fp).convert('RGB')
    arr = np.asarray(img)
    # arr is RGB HxWx3
    # Server-side face detection: detect and crop largest face if present
    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            # choose largest face
            faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
            x, y, w, h = faces[0]
            face_roi = arr[y:y+h, x:x+w]
            # attempt simple eye detection for alignment
            try:
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:
                    # pick two largest eyes
                    eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                    # compute eye centers in ROI coords
                    eye_centers = [(ex + ew/2.0, ey + eh/2.0) for (ex,ey,ew,eh) in eyes]
                    (x1,y1),(x2,y2) = eye_centers[0], eye_centers[1]
                    # ensure left->right ordering
                    if x2 < x1:
                        (x1,y1),(x2,y2) = (x2,y2),(x1,y1)
                    # compute angle
                    dy = y2 - y1
                    dx = x2 - x1
                    angle = np.degrees(np.arctan2(dy, dx))
                    # rotate the whole image around the center of the face_roi
                    center = (face_roi.shape[1] // 2, face_roi.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(face_roi, M, (face_roi.shape[1], face_roi.shape[0]), flags=cv2.INTER_CUBIC)
                    arr = rotated
                else:
                    arr = face_roi
            except Exception:
                arr = face_roi
    except Exception:
        # if face detection fails, continue with full image
        pass
    if spec is None:
        target_h, target_w, target_c = 96, 96, 3
    else:
        target_h, target_w, target_c = spec
        if target_h is None:
            target_h, target_w = 96, 96

    import cv2
    resized = cv2.resize(arr, (target_w, target_h))
    if target_c is None or target_c == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        proc = gray.astype('float32') / 255.0
        proc = np.expand_dims(proc, axis=(0, -1))
    else:
        proc = resized.astype('float32') / 255.0
        proc = np.expand_dims(proc, axis=0)
    return proc


def extract_audio_features_from_file(fp, sr=22050):
    try:
        data, _ = librosa.load(fp, sr=sr, mono=True)
        # reuse a compact feature set
        if data is None or len(data) == 0:
            return None
        data = librosa.util.fix_length(data, int(sr * 3), mode='constant')
        data, _ = librosa.effects.trim(data)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        feat = np.mean(mfcc, axis=1)
        return feat
    except Exception as ex:
        logger.exception('Audio load/feature extraction failed: %s', ex)
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Accept multipart/form-data: fields 'image', 'audio', 'text'
    fer_pred = None
    speech_pred = None
    text_pred = None

    # Image processing
    if 'image' in request.files and request.files['image'].filename:
        f = request.files['image']
        spec = get_model_input_spec(fer_model)
        try:
            arr = preprocess_image_file(f.stream, spec=spec)
            if fer_model is not None:
                out = fer_model.predict(arr)
                fer_pred = np.asarray(out[0]).ravel().tolist()
        except Exception as ex:
            logger.exception('FER prediction failed: %s', ex)

    # Audio processing
    if 'audio' in request.files and request.files['audio'].filename and speech_model is not None:
        g = request.files['audio']
        try:
            feats = extract_audio_features_from_file(g.stream)
            if feats is not None:
                X = feats.reshape(1, -1)
                try:
                    out = speech_model.predict(X)
                    speech_pred = np.asarray(out[0]).ravel().tolist()
                except Exception:
                    logger.exception('Speech model predict failed')
        except Exception:
            logger.exception('Audio handling failed')

    # Text processing
    text_val = request.form.get('text', '')
    if text_val and text_model is not None and vectorizer is not None:
        try:
            X = vectorizer.transform([text_val])
            out = text_model.predict_proba(X)[0]
            text_pred = np.asarray(out).ravel().tolist()
        except Exception:
            logger.exception('Text model predict failed')

    # Fuse
    try:
        fused_label, combined = fuse(np.array(fer_pred) if fer_pred is not None else None,
                                     np.array(speech_pred) if speech_pred is not None else None,
                                     speech_le,
                                     np.array(text_pred) if text_pred is not None else None)
        combined = combined.tolist()
    except Exception:
        logger.exception('Fusion failed')
        fused_label, combined = None, [0.0] * len(fer_labels)

    return jsonify({
        'fer_pred': fer_pred,
        'speech_pred': speech_pred,
        'text_pred': text_pred,
        'fused_label': fused_label,
        'combined': combined,
        'fer_labels': fer_labels,
    })


@app.route('/api/export_row', methods=['POST'])
def api_export_row():
    """Accepts same form as /api/predict but requires a 'label' field (ground-truth).
    Appends a CSV row to `models/fusion_model/training_data.csv` with columns:
    fer_0..fer_6, speech_0..speech_N, text_0..text_M, label
    The header is created on first write based on observed vector lengths.
    """
    label = request.form.get('label')
    if not label:
        return jsonify({'error': 'label is required'}), 400

    # reuse prediction code path
    res = api_predict().get_json()
    fer_p = res.get('fer_pred')
    sp_p = res.get('speech_pred')
    tx_p = res.get('text_pred')

    # prepare output dir
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / 'models' / 'fusion_model'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'training_data.csv'

    # Determine column counts
    n_fer = 7
    n_speech = len(sp_p) if sp_p is not None else 0
    n_text = len(tx_p) if tx_p is not None else 0

    # If file exists, read header to enforce column counts
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as fh:
            reader = csv.reader(fh)
            try:
                hdr = next(reader)
                # count existing speech/text columns
                n_fer = len([c for c in hdr if c.startswith('fer_')])
                n_speech = len([c for c in hdr if c.startswith('speech_')])
                n_text = len([c for c in hdr if c.startswith('text_')])
            except StopIteration:
                pass

    # Build header if missing
    if not csv_path.exists():
        cols = []
        cols += [f'fer_{i}' for i in range(7)]
        cols += [f'speech_{i}' for i in range(n_speech)]
        cols += [f'text_{i}' for i in range(n_text)]
        cols += ['label']
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)

    # Prepare row values, filling zeros for missing entries if necessary
    row = []
    # fer
    if fer_p is None:
        row += [0.0] * 7
    else:
        fp = list(fer_p)[:7]
        fp += [0.0] * (7 - len(fp))
        row += fp
    # speech
    if n_speech > 0:
        if sp_p is None:
            row += [0.0] * n_speech
        else:
            sp = list(sp_p)[:n_speech]
            sp += [0.0] * (n_speech - len(sp))
            row += sp
    # text
    if n_text > 0:
        if tx_p is None:
            row += [0.0] * n_text
        else:
            tp = list(tx_p)[:n_text]
            tp += [0.0] * (n_text - len(tp))
            row += tp

    row += [label]

    # Append
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(row)

    return jsonify({'ok': True, 'path': str(csv_path)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
