import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Standard FER-2013 Label Mapping (Common Convention)
# 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

class EmotionEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_error = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.load_error = f"File not found: {self.model_path}"
            print(f"Error: {self.load_error}")
            return
        try:
            self.model = load_model(self.model_path)
            print(f"Emotion model loaded from {self.model_path}")
        except Exception as e:
            self.load_error = str(e)
            print(f"Failed to load emotion model: {e}")

    def pad_image(self, image, target_shape=(48, 48)):
        # Assuming model expects 48x48 grayscale (FER-2013 standard)
        # Resize and pad/crop logic could go here if needed.
        # For now, simple resize.
        return cv2.resize(image, target_shape)

    def predict(self, image):
        """
        Predict emotion from an OpenCV image (numpy array).
        Expects BGR or Grayscale input.
        """
        if self.model is None:
            return f"Model Error: {self.load_error}" if self.load_error else "Model Not Loaded"

        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Face detection could be done here, but usually frontend sends a face or we process full frame.
            # Using Haar Cascade for face detection to be safe if full frame is sent.
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            target_face = gray
            if len(faces) > 0:
                # Take the largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                target_face = gray[y:y+h, x:x+w]

            # Preprocess for model (48x48, normalize 0-1, reshape)
            resized = cv2.resize(target_face, (48, 48))
            normalized = resized.astype('float32') / 255.0
            reshaped = np.expand_dims(normalized, axis=0) # (1, 48, 48)
            reshaped = np.expand_dims(reshaped, axis=-1)  # (1, 48, 48, 1)

            prediction = self.model.predict(reshaped, verbose=0)
            label_index = np.argmax(prediction)
            
            return EMOTION_LABELS.get(label_index, "Unknown")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error"
