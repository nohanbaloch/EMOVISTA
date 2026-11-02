"""
src/fusion/severity_engine.py

Emotion Intensity and Persistence Tracking Engine for EMOVISTA.

DISCLAIMER:
This module implements a transparent heuristic based on detected emotion
intensity and temporal persistence. It is designed to assist empathetic
conversation pacing and supportive feedback.
IT DOES NOT PROVIDE A CLINICAL ASSESSMENT, PSYCHOLOGICAL DIAGNOSIS,
OR MEDICAL MONITORING METRIC.
"""

import time
from typing import Dict, Any, List

# Descriptive, non-clinical emotion state language
DESCRIPTIVE_STATE_MAP = {
    "angry": "distressed / tense",
    "fear": "worried / heightened alertness",
    "sad": "low / downhearted",
    "happy": "positive / content",
    "neutral": "settled / calm",
    "disgust": "unsettled / averse",
    "surprise": "startled / unexpected"
}

NON_DIAGNOSTIC_DISCLAIMER = (
    "This score is an automated behavioral heuristic tracking detected emotion intensity "
    "over time. It is NOT a clinical assessment, psychological evaluation, or medical diagnosis."
)


class SeverityEngine:
    """Calculates non-diagnostic intensity and persistence heuristics from emotion probabilities."""

    def __init__(self, persistence_window: int = 10):
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.persistence_window = persistence_window

    def _intensity_label(self, score: int) -> str:
        """Map a 0-100 heuristic intensity score to descriptive categories."""
        if score <= 25:
            return "low"
        elif score <= 50:
            return "mild"
        elif score <= 75:
            return "moderate"
        return "elevated"

    def analyze(
        self,
        fer_probs: Dict[str, float],
        ser_probs: Dict[str, float],
        text_sentiment: float
    ) -> Dict[str, Any]:
        """
        Analyze multimodal inputs and compute an emotion intensity heuristic.

        Arguments:
            fer_probs: Dictionary of emotion to probability (e.g. {"sad": 0.6, "happy": 0.1})
            ser_probs: Dictionary of audio emotion probabilities
            text_sentiment: Continuous sentiment score between -1.0 (negative) and +1.0 (positive)

        Returns:
            Dictionary containing dominant emotion, descriptive state, intensity score,
            and explicit non-diagnostic disclaimer.
        """
        combined: Dict[str, float] = {}

        # Visual contribution (40%)
        if fer_probs:
            for k, v in fer_probs.items():
                k_lower = str(k).lower()
                combined[k_lower] = combined.get(k_lower, 0.0) + float(v) * 0.40

        # Audio contribution (40%)
        if ser_probs:
            for k, v in ser_probs.items():
                k_lower = str(k).lower()
                combined[k_lower] = combined.get(k_lower, 0.0) + float(v) * 0.40

        # Text sentiment contribution (20%)
        if text_sentiment < -0.3:
            combined["sad"] = combined.get("sad", 0.0) + abs(text_sentiment) * 0.20
        elif text_sentiment > 0.3:
            combined["happy"] = combined.get("happy", 0.0) + text_sentiment * 0.20

        if not combined:
            combined["neutral"] = 1.0

        dominant_emotion = max(combined, key=combined.get)
        descriptive_state = DESCRIPTIVE_STATE_MAP.get(dominant_emotion, "neutral")

        # Intensity heuristic scaled 0 to 100
        raw_score = int(min(100, combined[dominant_emotion] * 100))
        intensity = self._intensity_label(raw_score)

        record = {
            "timestamp": time.time(),
            "emotion": dominant_emotion,
            "descriptive_state": descriptive_state,
            "intensity_score": raw_score,
            "intensity_level": intensity,
            "disclaimer": NON_DIAGNOSTIC_DISCLAIMER
        }

        self.history.append(record)
        if len(self.history) > self.persistence_window * 5:
            self.history = self.history[-self.persistence_window * 5:]

        return record

    def evaluate(self, emotion: str, probabilities: Any) -> Dict[str, Any]:
        """
        Evaluate emotion intensity from fused probabilities (UI helper).
        DISCLAIMER: Non-diagnostic heuristic only.
        """
        score = 0
        if probabilities is not None:
            try:
                score = int(np.max(probabilities) * 100)
            except Exception:
                score = 50
        level = self._intensity_label(score)
        state = DESCRIPTIVE_STATE_MAP.get(str(emotion).lower(), "neutral")
        return {
            "level": level.capitalize(),
            "score": score,
            "state": state,
            "response": f"Detected emotion state: {emotion} ({state}).",
            "disclaimer": NON_DIAGNOSTIC_DISCLAIMER
        }
