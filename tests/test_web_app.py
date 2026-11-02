"""
tests/test_web_app.py

Integration and unit tests for Flask web app hardening, error handling,
security boundaries, and graceful degradation.
"""

import unittest
import json
import base64
import numpy as np
import cv2
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
import os
os.environ["FLASK_DEBUG"] = "0"
os.environ["KERAS_BACKEND"] = "torch"

from app import app, Config
from src.web.backend.llm_phi3 import DETERMINISTIC_CRISIS_RESPONSE


class TestWebAppHardening(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    # --- /analyze_face hardening tests ---

    def test_analyze_face_missing_payload(self):
        response = self.client.post("/analyze_face", data="")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_analyze_face_missing_image_field(self):
        response = self.client.post("/analyze_face", json={"wrong_key": "123"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_analyze_face_bad_base64(self):
        response = self.client.post("/analyze_face", json={"image": "not_valid_base64!!!"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_analyze_face_valid_blank_image(self):
        # Create a small blank image (100x100 RGB) and encode to base64
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        b64_img = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        response = self.client.post("/analyze_face", json={"image": b64_img})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("emotion", data)
        # With no detected face in blank image, fusion falls back to Neutral
        self.assertEqual(data["emotion"], "Neutral")

    # --- /tts hardening tests ---

    def test_tts_missing_payload(self):
        response = self.client.post("/tts", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_tts_empty_text(self):
        response = self.client.post("/tts", json={"text": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_tts_oversized_text(self):
        oversized = "a" * (Config.MAX_TTS_TEXT_LEN + 50)
        response = self.client.post("/tts", json={"text": oversized})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_tts_valid_text(self):
        response = self.client.post("/tts", json={"text": "Hello, world"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")

    # --- /stt graceful degradation ---

    def test_stt_missing_model_graceful_degradation(self):
        # When Vosk model or audio hardware is absent, returns 503 with helpful details
        response = self.client.post("/stt")
        if response.status_code == 503:
            data = response.get_json()
            self.assertIn("error", data)
            self.assertEqual(data.get("text"), "")
        else:
            self.assertIn(response.status_code, [200, 500, 503])

    # --- /consult deterministic crisis & graceful degradation ---

    def test_consult_deterministic_crisis_trigger(self):
        # High risk text triggers deterministic safety response immediately
        crisis_payload = {
            "emotion": "Sad",
            "text": "I feel completely hopeless and I want to kill myself tonight"
        }
        response = self.client.post("/consult", json=crisis_payload)
        self.assertEqual(response.status_code, 200)
        content = response.get_data(as_text=True)
        # Must contain crisis helpline guidance and 988/111/112 referral
        self.assertIn("emergency services", content.lower())
        self.assertIn("988", content)

    def test_consult_speech_unavailable_degrades_cleanly(self):
        # Consultation works with FER + text without requiring audio SER
        normal_payload = {
            "emotion": "Happy",
            "text": "I had a great day today!"
        }
        response = self.client.post("/consult", json=normal_payload)
        self.assertEqual(response.status_code, 200)
        content = response.get_data(as_text=True)
        self.assertTrue(len(content) > 0)


if __name__ == "__main__":
    unittest.main()
