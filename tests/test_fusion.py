import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.emotion_fusion import map_speech_to_fer, fuse, fer_labels


class DummyLE:
    def __init__(self, classes):
        # mimic sklearn LabelEncoder.classes_
        self.classes_ = np.array(classes)


def test_map_speech_to_fer_variations():
    assert map_speech_to_fer('angry') == 'Angry'
    assert map_speech_to_fer('ANGRY') == 'Angry'
    assert map_speech_to_fer('  happy ') == 'Happy'
    assert map_speech_to_fer('unknown') is None


def test_fuse_with_fer_only():
    # happy index
    idx = fer_labels.index('Happy')
    fer_pred = np.zeros(len(fer_labels))
    fer_pred[idx] = 1.0
    label, combined = fuse(fer_pred, None, None, None)
    assert label == 'Happy'
    assert combined.shape[0] == len(fer_labels)
    assert np.isclose(combined[idx], 1.0)


def test_fuse_with_speech_only():
    # create speech classes aligning to 'happy'
    le = DummyLE(['angry', 'happy'])
    # speech_pred: second class 'happy' strong
    speech_pred = np.array([0.0, 1.0])
    label, combined = fuse(None, speech_pred, le, None)
    assert label == 'Happy'


def test_fuse_with_text_only():
    # text_pred [neg, neutral, pos] -> pos maps to Happy
    text_pred = np.array([0.0, 0.0, 1.0])
    label, combined = fuse(None, None, None, text_pred)
    assert label == 'Happy'


def test_fuse_handles_malformed_inputs():
    # fer_pred wrong size
    fer_pred = np.array([1, 2, 3])
    # speech_pred length mismatch
    le = DummyLE(['angry', 'disgust', 'happy'])
    speech_pred = np.array([0.1, 0.2])
    # should not raise
    label, combined = fuse(fer_pred, speech_pred, le, None)
    assert isinstance(label, str)
    assert combined.shape[0] == len(fer_labels)
