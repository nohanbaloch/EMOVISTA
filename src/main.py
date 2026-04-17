#!/usr/bin/env python3
"""
Real-time Emotion-Aware Assistant
Fuses webcam (FER), microphone (Speech), and text input for emotion detection.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('emovista_main')

# Add src to path
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fusion.emotion_fusion import load_all, fuse, fer_labels


class EmotionAssistant:
    """Real-time emotion-aware assistant using webcam and microphone."""

    def __init__(self):
        """Initialize models and hardware."""
        logger.info("Initializing Emotion-Aware Assistant...")
        
        # Load all models
        self.fer_model, self.speech_model, self.text_model, \
            self.vectorizer, self.speech_le = load_all(verbose=True)
        
        # Setup face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.webcam = None
        self.running = False
        logger.info("Assistant initialized successfully")

    def get_fer_input_spec(self):
        """Get FER model input shape."""
        if self.fer_model is None:
            return (96, 96, 3)
        try:
            shape = getattr(self.fer_model, 'input_shape', None)
            if shape is None and hasattr(self.fer_model, 'inputs'):
                shape = tuple(self.fer_model.inputs[0].shape.as_list())
            if shape and len(shape) == 4:
                return (int(shape[1]), int(shape[2]), int(shape[3]))
        except Exception as ex:
            logger.warning(f"Could not determine FER input spec: {ex}")
        return (96, 96, 3)

    def process_frame(self, frame):
        """Process a single frame and return emotion prediction."""
        fer_pred = None
        
        if self.fer_model is None:
            logger.warning("FER model not loaded")
            return None, "Model unavailable"
        
        try:
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None, "No face detected"
            
            # Process first face
            (x, y, w, h) = faces[0]
            roi_color = frame[y:y+h, x:x+w]
            
            # Get model input specs
            target_h, target_w, target_c = self.get_fer_input_spec()
            roi_resized = cv2.resize(roi_color, (target_w, target_h))
            
            # Preprocess
            if target_c == 1 or target_c is None:
                roi_proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                roi_input = np.expand_dims(roi_proc, axis=(0, -1))
            else:
                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                roi_proc = roi_rgb.astype('float32') / 255.0
                roi_input = np.expand_dims(roi_proc, axis=0)
            
            # Predict
            fer_pred = self.fer_model.predict(roi_input, verbose=0)[0]
            
            # Fuse predictions
            fused_label, _ = fuse(fer_pred, None, self.speech_le, None)
            
            return fer_pred, fused_label
            
        except Exception as ex:
            logger.error(f"Error processing frame: {ex}")
            return None, f"Error: {str(ex)}"

    def run_webcam_demo(self):
        """Run real-time webcam emotion detection."""
        logger.info("Starting webcam demo (press 'q' to quit)...")
        
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            logger.error("Cannot open webcam")
            return
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.webcam.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Flip for selfie view
                frame = cv2.flip(frame, 1)
                
                # Process frame
                fer_pred, emotion = self.process_frame(frame)
                
                # Draw face detection box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display emotion label
                if isinstance(emotion, str) and emotion not in ["No face detected", "Model unavailable"]:
                    text = f"Emotion: {emotion}"
                    cv2.putText(frame, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display probabilities if available
                    if fer_pred is not None:
                        y_offset = 70
                        for i, label in enumerate(fer_labels):
                            prob_text = f"{label}: {fer_pred[i]:.2f}"
                            cv2.putText(frame, prob_text, (10, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            y_offset += 25
                else:
                    cv2.putText(frame, str(emotion), (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow("Emotion-Aware Assistant", frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User quit")
                    break
                    
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.webcam:
            self.webcam.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    try:
        assistant = EmotionAssistant()
        assistant.run_webcam_demo()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as ex:
        logger.error(f"Fatal error: {ex}", exc_info=True)


if __name__ == '__main__':
    main()
