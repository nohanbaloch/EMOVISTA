import json
import os
from .crypto_utils import load_or_create_key

MEMORY_FILE = "patient_memory.enc"

class PatientMemory:
    def __init__(self):
        self.cipher = load_or_create_key()
        self.memory = self._load()

    def _load(self):
        if not os.path.exists(MEMORY_FILE):
            return []
        with open(MEMORY_FILE, "rb") as f:
            decrypted = self.cipher.decrypt(f.read())
        return json.loads(decrypted.decode())

    def add_session(self, data: dict):
        self.memory.append(data)
        encrypted = self.cipher.encrypt(json.dumps(self.memory).encode())
        with open(MEMORY_FILE, "wb") as f:
            f.write(encrypted)

    def get_recent(self, n=5):
        return self.memory[-n:]
