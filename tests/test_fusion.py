"""
tests/test_fusion.py

Unit tests for multimodal emotion fusion (TensorFlow-independent).
Tests all modalities present, one missing, all missing, wrong vector lengths,
binary sentiment confidence mapping, and surprise source constraints.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fusion.emotion_fusion import (
    fuse,
    map_binary_text_to_three_way,
    map_speech_to_fer,
    fer_labels,
    SPEECH_TO_FER,
)


class DummySpeechLE:
    """Mock LabelEncoder with classes_ attribute matching CREMA-D."""
    def __init__(self, classes=None):
        self.classes_ = np.array(classes if classes is not None else [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'
        ])


class TestEmotionFusion(unittest.TestCase):
    def setUp(self):
        self.speech_le = DummySpeechLE()
        self.neutral_idx = fer_labels.index("Neutral")
        self.happy_idx = fer_labels.index("Happy")
        self.sad_idx = fer_labels.index("Sad")
        self.surprise_idx = fer_labels.index("Surprise")

    def test_all_modalities_present(self):
        # FER predicts Happy with 1.0 (weight 0.50)
        fer_pred = np.zeros(7)
        fer_pred[self.happy_idx] = 1.0

        # Speech predicts Sad with 1.0 (weight 0.30)
        speech_pred = np.zeros(6)
        speech_pred[list(self.speech_le.classes_).index('sad')] = 1.0

        # Text predicts positive with 1.0 -> maps to Happy (weight 0.20)
        text_pred = np.array([0.0, 1.0])

        best_label, scores = fuse(fer_pred, speech_pred, self.speech_le, text_pred)

        self.assertAlmostEqual(scores.sum(), 1.0, places=5)
        # Happy gets: 0.50 (FER) + 0.20 (Text) = 0.70
        # Sad gets: 0.30 (Speech)
        self.assertAlmostEqual(scores[self.happy_idx], 0.70, places=5)
        self.assertAlmostEqual(scores[self.sad_idx], 0.30, places=5)
        self.assertEqual(best_label, "Happy")

    def test_one_missing_fer_only(self):
        # Speech and Text missing
        fer_pred = np.zeros(7)
        fer_pred[self.surprise_idx] = 1.0

        best_label, scores = fuse(fer_pred, None, None, None)

        self.assertEqual(best_label, "Surprise")
        self.assertAlmostEqual(scores[self.surprise_idx], 1.0, places=5)
        self.assertAlmostEqual(scores.sum(), 1.0, places=5)

    def test_one_missing_speech_missing(self):
        # Speech missing: FER (0.50) and Text (0.20) active.
        # Renormalized weights: FER = 0.50 / 0.70 = 5/7, Text = 0.20 / 0.70 = 2/7
        fer_pred = np.zeros(7)
        fer_pred[self.happy_idx] = 1.0

        # Text predicts negative [1.0, 0.0] -> maps to Sad
        text_pred = np.array([1.0, 0.0])

        best_label, scores = fuse(fer_pred, None, None, text_pred)

        expected_happy = 0.50 / 0.70
        expected_sad = 0.20 / 0.70
        self.assertAlmostEqual(scores[self.happy_idx], expected_happy, places=5)
        self.assertAlmostEqual(scores[self.sad_idx], expected_sad, places=5)
        self.assertAlmostEqual(scores.sum(), 1.0, places=5)
        self.assertEqual(best_label, "Happy")

    def test_all_missing(self):
        best_label, scores = fuse(None, None, None, None)

        self.assertEqual(best_label, "Neutral")
        self.assertAlmostEqual(scores[self.neutral_idx], 1.0, places=5)
        self.assertAlmostEqual(scores.sum(), 1.0, places=5)

    def test_wrong_vector_length_fer(self):
        # Wrong FER length (5 instead of 7)
        bad_fer = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        # Text is valid: positive
        text_pred = np.array([0.0, 1.0])

        best_label, scores = fuse(bad_fer, None, None, text_pred)

        # Invalid FER ignored; Text becomes 100% active
        self.assertEqual(best_label, "Happy")
        self.assertAlmostEqual(scores[self.happy_idx], 1.0, places=5)
        self.assertAlmostEqual(scores.sum(), 1.0, places=5)

    def test_wrong_vector_length_speech(self):
        # Wrong Speech length (4 instead of 6)
        bad_speech = np.array([0.25, 0.25, 0.25, 0.25])
        fer_pred = np.zeros(7)
        fer_pred[self.neutral_idx] = 1.0

        best_label, scores = fuse(fer_pred, bad_speech, self.speech_le, None)

        # Speech ignored; FER is 100% active
        self.assertEqual(best_label, "Neutral")
        self.assertAlmostEqual(scores[self.neutral_idx], 1.0, places=5)

    def test_wrong_vector_length_text(self):
        # Wrong Text length (5 instead of 1, 2, or 3)
        bad_text = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        fer_pred = np.zeros(7)
        fer_pred[self.sad_idx] = 1.0

        best_label, scores = fuse(fer_pred, None, None, bad_text)

        # Text ignored; FER is 100% active
        self.assertEqual(best_label, "Sad")
        self.assertAlmostEqual(scores[self.sad_idx], 1.0, places=5)

    def test_binary_text_mapping_uncertainty_is_neutral(self):
        # Completely uncertain classifier: [0.5, 0.5] -> confidence = 0.0 -> Neutral = 1.0
        uncertain_res = map_binary_text_to_three_way(np.array([0.5, 0.5]))
        self.assertIsNotNone(uncertain_res)
        self.assertAlmostEqual(uncertain_res[1], 1.0, places=5)
        self.assertAlmostEqual(uncertain_res[0], 0.0, places=5)
        self.assertAlmostEqual(uncertain_res[2], 0.0, places=5)

        # Confident positive: [0.0, 1.0] -> confidence = 1.0 -> Pos = 1.0, Neutral = 0.0
        conf_pos = map_binary_text_to_three_way(np.array([0.0, 1.0]))
        self.assertAlmostEqual(conf_pos[2], 1.0, places=5)
        self.assertAlmostEqual(conf_pos[1], 0.0, places=5)

        # Confident negative: [1.0, 0.0] -> confidence = 1.0 -> Neg = 1.0, Neutral = 0.0
        conf_neg = map_binary_text_to_three_way(np.array([1.0, 0.0]))
        self.assertAlmostEqual(conf_neg[0], 1.0, places=5)
        self.assertAlmostEqual(conf_neg[1], 0.0, places=5)

        # Moderate positive: [0.2, 0.8] -> confidence = 0.6 -> Neutral = 0.4
        mod_pos = map_binary_text_to_three_way(np.array([0.2, 0.8]))
        self.assertAlmostEqual(mod_pos[1], 0.4, places=5)
        self.assertAlmostEqual(mod_pos[0] + mod_pos[1] + mod_pos[2], 1.0, places=5)

    def test_surprise_exclusive_to_fer(self):
        # Speech cannot produce Surprise: CREMA-D has 6 classes, excluding surprise
        self.assertNotIn("surprise", SPEECH_TO_FER)

        # Uniform speech prediction across all 6 classes
        uniform_speech = np.ones(6) / 6.0
        _, scores = fuse(None, uniform_speech, self.speech_le, None)

        # Surprise score must remain exactly 0.0
        self.assertEqual(scores[self.surprise_idx], 0.0)


if __name__ == "__main__":
    unittest.main()
