"""
src/memory/patient_memory.py

Encrypted Session Memory storage for EMOVISTA interactions.
Provides encrypted local storage of user emotion logs and conversation summaries.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from .crypto_utils import load_or_create_key, PROJECT_ROOT, DEFAULT_KEY_DIR


def get_memory_file_path() -> Path:
    mem_env = os.environ.get("EMOVISTA_MEMORY_PATH")
    if mem_env:
        return Path(mem_env)
    DEFAULT_KEY_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_KEY_DIR / "patient_memory.enc"


class PatientMemory:
    """Encrypted session interaction store using Fernet symmetric encryption."""

    def __init__(self):
        self.cipher = load_or_create_key()
        self.memory_path = get_memory_file_path()
        self.memory: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        try:
            with open(self.memory_path, "rb") as f:
                data = f.read()
            if not data:
                return []
            decrypted = self.cipher.decrypt(data)
            return json.loads(decrypted.decode("utf-8"))
        except Exception:
            return []

    def add_session(self, data: Dict[str, Any]):
        """Append an interaction entry and encrypt to disk."""
        self.memory.append(data)
        encrypted = self.cipher.encrypt(json.dumps(self.memory).encode("utf-8"))
        with open(self.memory_path, "wb") as f:
            f.write(encrypted)

    def add_entry(self, emotion: str, severity: str, score: int):
        """Backward-compatible entry helper."""
        self.add_session({
            "emotion": emotion,
            "severity": severity,
            "score": score
        })

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the n most recent session records."""
        return self.memory[-n:]
