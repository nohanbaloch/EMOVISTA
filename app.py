"""
EMOVISTA - Streamlit Real-time Multimodal App (Option A)

Single-file Streamlit app that:
 - captures webcam video frames (FER)
 - captures microphone audio (SER)
 - takes text input
 - fuses FER+SER+Text via your fusion.emotion_fusion module
 - displays live results

Usage:
1) Install dependencies (see instructions below)
2) From project root run:
    streamlit run app.py

Notes:
 - streamlit-webrtc relies on PyAV (av) and browser permissions for webcam/mic.
 - For local dev, streamlit-webrtc works without HTTPS (it uses websockets).
 - This app expects your models & fusion module at src/fusion/emotion_fusion.py
"""



import os
import sys
import time
import threading
from collections import deque

import numpy as np
import streamlit as st

# WebRTC tools
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, AudioProcessorBase, RTCConfiguration

# AV/Frame handling
import av
import cv2
import librosa
import joblib
from pathlib import Path
import logging

# Make sure src/ is importable (so we can import fusion.emotion_fusion)
ROOT_SRC = Path(__file__).resolve().parents[0] / "src"
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

# import the project's fusion utilities
from src.fusion.emotion_fusion import load_all, fuse, fer_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emovista_streamlit")

st.set_page_config(page_title="EMOVISTA — Realtime", layout="wide")

# Sidebar controls
st.sidebar.title("EMOVISTA — Realtime")
st.sidebar.markdown(
    "Permissions: allow camera & microphone. "
    "This app runs inference locally in this process (server-side)."
)
model_info_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
debug = st.sidebar.checkbox("Debug mode", value=False)
ser_interval_sec = st.sidebar.slider("SER inference interval (s)", min_value=1, max_value=5, value=3)
fer_weight = st.sidebar.slider("FER weight", 0.0, 1.0, 0.5)
speech_weight = st.sidebar.slider("Speech weight", 0.0, 1.0, 0.3)
text_weight = st.sidebar.slider("Text weight", 0.0, 1.0, 0.2)

# Load models (may be large; do it once)
@st.cache_resource
def load_models_once():
    fer_model, speech_model, text_model, vectorizer, speech_le = load_all()
    return fer_model, speech_model, text_model, vectorizer, speech_le

fer_model, speech_model, text_model, vectorizer, speech_le = load_models_once()

model_info = [
    ("FER model", "loaded" if fer_model is not None else "missing"),
    ("Speech model", "loaded" if speech_model is not None else "missing"),
    ("Text model", "loaded" if text_model is not None else "missing"),
    ("Speech LabelEncoder", "loaded" if speech_le is not None else "missing"),
    ("Vectorizer", "loaded" if vectorizer is not None else "missing"),
]
model_info_placeholder.table(model_info)

# Shared state objects (to hold latest predictions)
if "last_fer_pred" not in st.session_state:
    st.session_state.last_fer_pred = None
if "last_speech_pred" not in st.session_state:
    st.session_state.last_speech_pred = None
if "last_text_pred" not in st.session_state:
    st.session_state.last_text_pred = None
if "last_fused" not in st.session_state:
    st.session_state.last_fused = ("Unknown", np.zeros(len(fer_labels)))

# Audio buffer to accumulate samples for SER
AUDIO_SR = 22050
AUDIO_QUEUE_MAX_SEC = 6  # keep last N seconds
_audio_buffer = deque(maxlen=AUDIO_SR * AUDIO_QUEUE_MAX_SEC)

# Helper: infer FER input spec and preprocess ROI
def _get_fer_input_spec(model):
    if model is None:
        return None
    try:
        shape = getattr(model, "input_shape", None)
        if shape is None and hasattr(model, "inputs"):
            shape = tuple(model.inputs[0].shape.as_list())
        if shape is None:
            return None
        if len(shape) == 4:
            return (int(shape[1]), int(shape[2]), int(shape[3]))
        if len(shape) == 3:
            return (int(shape[1]), int(shape[2]), None)
    except Exception as e:
        logger.exception("Couldn't determine FER input shape: %s", e)
    return None

fer_input_spec = _get_fer_input_spec(fer_model)
logger.info("FER input spec: %s", fer_input_spec)

# Audio processing: compute MFCCs for SER model
def extract_audio_features_from_buffer(buffer_np, sr=AUDIO_SR, n_mfcc=40):
    """
    buffer_np: 1D numpy float32 array (mono)
    returns: shape (1, n_mfcc)
    """
    try:
        # librosa expects float32 in range [-1,1]. If input is int16, convert before calling.
        y = buffer_np.astype(np.float32)
        # Trim silence and ensure length
        if len(y) < 512:
            return None
        # librosa trim
        y, _ = librosa.effects.trim(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        logger.exception("Audio feature extraction failed: %s", e)
        return None

# Video transformer: receives frames from the browser, returns annotated frames
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to numpy (BGR)
        img = frame.to_ndarray(format="bgr24")
        orig = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        fer_pred_for_frame = None

        for (x, y, w, h) in faces:
            try:
                # Extract ROI and preprocess depending on fer_input_spec
                roi_color = orig[y:y+h, x:x+w]
                spec = fer_input_spec
                if spec is None:
                    target_h, target_w, target_c = 48, 48, 1
                else:
                    target_h, target_w, target_c = spec
                    if target_h is None:
                        target_h, target_w = 48, 48

                # Resize
                roi_resized = cv2.resize(roi_color, (target_w, target_h))

                # Convert channels
                if target_c is None or target_c == 1:
                    roi_proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                    roi_proc = roi_proc.astype("float32") / 255.0
                    # Expand dims to (1, H, W, 1)
                    roi_input = np.expand_dims(roi_proc, axis=(0, -1))
                else:
                    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                    roi_proc = roi_rgb.astype("float32") / 255.0
                    roi_input = np.expand_dims(roi_proc, axis=0)

                if fer_model is not None:
                    pred = fer_model.predict(roi_input)
                    fer_pred_for_frame = pred[0]
                    idx = int(np.argmax(fer_pred_for_frame))
                    label_text = fer_labels[idx]
                else:
                    label_text = "NoFER"

            except Exception as e:
                logger.exception("FER processing error: %s", e)
                fer_pred_for_frame = None
                label_text = "ERR"

            # Draw
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # update session state
        st.session_state.last_fer_pred = fer_pred_for_frame

        # overlay fused label text on top-left (latest)
        fused_label, combined = st.session_state.last_fused
        overlay_text = f"Fused: {fused_label}"
        cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 220, 50), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Audio processor: accumulates audio frames into the shared buffer
class AudioAccumulator(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """
        This callback receives audio frames from the browser.
        We convert them to numpy and append to the circular buffer.
        """
        # Convert to numpy array (shape: (n_channels, n_samples))
        try:
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                # mixdown to mono
                mono = np.mean(arr, axis=0).astype(np.float32)
            else:
                mono = arr.astype(np.float32)
            # Normalize if int16
            if mono.dtype == np.int16:
                mono = mono / 32768.0
            # Append to buffer
            _audio_append(mono)
        except Exception as e:
            logger.exception("Audio receive error: %s", e)
        # We must return the frame (or None) to pass audio through; returning frame passes audio back to user
        return frame

# Append audio chunk to deque
def _audio_append(chunk: np.ndarray):
    try:
        # chunk is 1D float32
        _audio_buffer.extend(chunk.tolist())
        # Keep size <= maxlen already enforced by deque
    except Exception as e:
        logger.exception("Appending audio chunk failed: %s", e)

# Periodic speech inference worker (runs in background)
def _speech_worker():
    """
    Every ser_interval_sec seconds, read the last N seconds from buffer, extract features and run SER model.
    """
    while True:
        time.sleep(ser_interval_sec)
        try:
            # copy current buffer to numpy
            buf = np.array(_audio_buffer, dtype=np.float32)
            if buf.size == 0:
                continue
            # Resample or trim/pad to desired SR if needed (we assume incoming frames match AUDIO_SR or similar)
            # For safety, convert to float32
            feats = extract_audio_features_from_buffer(buf, sr=AUDIO_SR, n_mfcc=40)
            if feats is None or speech_model is None or speech_le is None:
                st.session_state.last_speech_pred = None
                continue
            preds = speech_model.predict(feats)
            st.session_state.last_speech_pred = preds[0]
            if debug:
                logger.info("SER preds: %s", preds[0])
        except Exception as e:
            logger.exception("Speech worker error: %s", e)

# Start the speech worker thread once
if "speech_thread_started" not in st.session_state:
    t = threading.Thread(target=_speech_worker, daemon=True)
    t.start()
    st.session_state.speech_thread_started = True

# UI: main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("EMOVISTA — Real-time Multimodal Emotion Assistant")
    st.write("Allow camera & microphone permissions in the browser. This runs inference on the server process.")
    # webrtc_streamer provides a video + audio UI element
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="emovista",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_transformer_factory=EmotionVideoTransformer,
        audio_processor_factory=AudioAccumulator,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

with col2:
    st.subheader("Inputs")
    text_val = st.text_area("Text input (optional)", placeholder="Type a message here...")
    if st.button("Run text inference"):
        # run text inference instantly
        if text_model is None or vectorizer is None:
            st.warning("Text model/vectorizer not found in models/; skip text inference.")
            st.session_state.last_text_pred = None
        else:
            try:
                tfidf = vectorizer.transform([text_val])
                text_proba = text_model.predict_proba(tfidf)[0]
                st.session_state.last_text_pred = text_proba
                st.success(f"Text probs: {np.round(text_proba,3)}")
            except Exception as e:
                logger.exception("Text inference failed: %s", e)
                st.session_state.last_text_pred = None

    st.subheader("Latest")
    fer_display = st.empty()
    speech_display = st.empty()
    text_display = st.empty()
    fused_display = st.empty()

# Fusion update loop: update fused result periodically
def _fusion_loop():
    """
    Periodically fuse the latest modality predictions and update UI placeholders.
    """
    while True:
        try:
            fer_pred = st.session_state.last_fer_pred
            speech_pred = st.session_state.last_speech_pred
            text_pred = st.session_state.last_text_pred
            # use the fusion.fuse helper (which maps speech labels to FER label-space)
            fused_label, combined = fuse(fer_pred, speech_pred, speech_le, text_pred)
            st.session_state.last_fused = (fused_label, combined)

            # Update UI elements in Streamlit (use experimental APIs - safe for simple dashboards)
            fer_display.markdown(f"**FER:** `{np.round(fer_pred,3) if fer_pred is not None else None}`")
            speech_display.markdown(f"**Speech:** `{np.round(speech_pred,3) if speech_pred is not None else None}`")
            text_display.markdown(f"**Text:** `{np.round(text_pred,3) if text_pred is not None else None}`")
            fused_display.markdown(f"### FUSED: **{fused_label}**\n\nScores: `{np.round(combined,3)}`")
            time.sleep(0.8)
        except Exception as e:
            logger.exception("Fusion loop error: %s", e)
            time.sleep(1.0)

# Start fusion thread once
if "fusion_thread_started" not in st.session_state:
    ft = threading.Thread(target=_fusion_loop, daemon=True)
    ft.start()
    st.session_state.fusion_thread_started = True

st.markdown("---")
st.caption("Notes: This demo performs server-side inference (models load into the Python process running Streamlit). "
           "Make sure sufficient CPU/GPU resources are available. For production, consider separating the inference "
           "service from the web frontend.")

st.caption("Developed as part of the EMOVISTA project.")