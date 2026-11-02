"""
src/memory/crypto_utils.py

Cryptographic utilities for session log encryption using Fernet (AES-128-CBC).
DISCLAIMER: This encryption is intended for local user privacy of interaction logs.
It is NOT a certified HIPAA/health-record management system.
"""

from pathlib import Path
from cryptography.fernet import Fernet
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY_DIR = PROJECT_ROOT / "data" / "memory"


def get_key_path() -> Path:
    key_env = os.environ.get("EMOVISTA_KEY_PATH")
    if key_env:
        return Path(key_env)
    DEFAULT_KEY_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_KEY_DIR / "patient.key"


def load_or_create_key() -> Fernet:
    key_path = get_key_path()
    if not key_path.exists():
        key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(key)
    else:
        with open(key_path, "rb") as f:
            key = f.read().strip()
    return Fernet(key)
