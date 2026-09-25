import time
import numpy as np

CLINICAL_MAP = {
    "angry": "stress",
    "fear": "anxiety",
    "sad": "depression",
    "happy": "positive",
    "neutral": "neutral",
    "disgust": "stress",
    "surprise": "anxiety"
}

class SeverityEngine:
    def __init__(self):
        self.history = []
        self.start_time = time.time()

    def _severity_label(self, score: int) -> str:
        if score <= 25:
            return "normal"
        elif score <= 50:
            return "mild"
        elif score <= 75:
            return "moderate"
        return "severe"

    def analyze(self, fer_probs: dict, ser_probs: dict, text_sentiment: float):
        """
        fer_probs / ser_probs:
        {"sad":0.6, "happy":0.1 ...}
        text_sentiment: -1 to +1
        """

        combined = {}

        for k, v in fer_probs.items():
            combined[k] = combined.get(k, 0) + v * 0.4

        for k, v in ser_probs.items():
            combined[k] = combined.get(k, 0) + v * 0.4

        # text sentiment influences sadness/anxiety
        if text_sentiment < -0.3:
            combined["sad"] = combined.get("sad", 0) + abs(text_sentiment) * 0.2
        elif text_sentiment > 0.3:
            combined["happy"] = combined.get("happy", 0) + text_sentiment * 0.2

        dominant_emotion = max(combined, key=combined.get)
        clinical_state = CLINICAL_MAP.get(dominant_emotion, "neutral")

        raw_score = int(min(100, combined[dominant_emotion] * 100))
        severity = self._severity_label(raw_score)

        record = {
            "timestamp": time.time(),
            "emotion": dominant_emotion,
            "state": clinical_state,
            "score": raw_score,
            "severity": severity
        }

        self.history.append(record)

        return record
