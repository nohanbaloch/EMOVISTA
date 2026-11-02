"""
src/web/backend/llm_phi3.py

Phi-3 Mini Local LLM Assistant integration via Ollama.
Includes deterministic crisis detection fallback, empathetic prompt generation,
and robust model discovery.
"""

import os
import json
import re
from typing import Any, Iterator, Mapping, cast

try:
    import ollama
except ImportError:
    ollama = None

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self-harm", "hanging myself", "overdose", "slit my wrists",
    "take my own life", "better off dead"
]

DETERMINISTIC_CRISIS_RESPONSE = (
    "I hear that you are going through a painful moment, and I want you to be safe. "
    "I am an AI assistant and cannot provide medical or crisis care, but you do not have to carry this alone. "
    "Please reach out right now to someone you trust, a healthcare professional, or emergency services. "
    "If you are in the US or Canada, call or text 988. In the UK, call 111 or 999. In the EU, call 112. "
    "If you are elsewhere, please contact your local emergency response service immediately."
)


def is_crisis_text(text: str) -> bool:
    """Deterministic check for crisis or self-harm keywords."""
    if not text:
        return False
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    for kw in CRISIS_KEYWORDS:
        if kw in clean:
            return True
    return False


def load_system_prompt() -> str:
    config_path = os.path.join(os.path.dirname(__file__), "persona_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            c = cast(dict[str, Any], json.load(f))

        role = c.get("role", "You are Emovista, an empathetic emotional support AI assistant.")
        comm = cast(dict[str, Any], c.get("communication", {}))
        tone = ", ".join(comm.get("tone", ["Warm", "Supportive", "Calm"]))
        guidelines = "\n".join([f"- {r}" for r in comm.get("response_guidelines", [])])
        safety = cast(dict[str, Any], c.get("safety_and_ethics", {}))
        disclaimer = safety.get("disclaimer", "You are not a doctor.")

        return f"""{role}

Tone: {tone}

Response Rules:
{guidelines}

Important Boundary:
{disclaimer}
Always maintain a non-diagnostic, supportive stance. Keep responses concise (2-3 sentences max).
"""
    except Exception:
        return (
            "You are Emovista, an empathetic and supportive AI companion. "
            "You are NOT a doctor and do NOT diagnose medical or psychological conditions. "
            "Keep responses concise, warm, and helpful (max 2-3 sentences)."
        )


SYSTEM_PROMPT = load_system_prompt()


def list_ollama_models() -> list[str]:
    """Return installed Ollama model names in a robust, version-tolerant way."""
    if ollama is None:
        return ["phi3"]

    names: list[str] = []
    try:
        response = ollama.list()
        models_obj = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])

        for item in models_obj:
            name: str | None = None
            if isinstance(item, dict):
                raw = item.get("model") or item.get("name")
                name = str(raw) if raw else None
            else:
                raw = getattr(item, "model", None) or getattr(item, "name", None)
                name = str(raw) if raw else None

            if name:
                names.append(name)
    except Exception:
        return ["phi3"]

    deduped = list(dict.fromkeys(names))
    return deduped if deduped else ["phi3"]


class Phi3Assistant:
    def __init__(self, model: str = "phi3"):
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def respond(self, emotion: str, user_text: str, model: str | None = None) -> Iterator[str]:
        # 1. Deterministic safety check for crisis triggers
        if is_crisis_text(user_text):
            for token in DETERMINISTIC_CRISIS_RESPONSE.split(" "):
                yield token + " "
            return

        selected_model = (model or self.model).strip() or self.model

        prompt = f"""
Detected emotion: {emotion}
User statement: {user_text}

Respond with empathy and supportive active listening. Remember you are not a doctor.
"""
        self.messages.append({"role": "user", "content": prompt})

        if ollama is None:
            fallback = f"I hear you, and I can see you are feeling {emotion.lower()}. I am here to listen whenever you'd like to talk."
            yield fallback
            self.messages.append({"role": "assistant", "content": fallback})
            return

        try:
            stream = cast(Iterator[Mapping[str, Any]], ollama.chat(
                model=selected_model,
                messages=self.messages,
                stream=True
            ))

            full_reply = ""
            for chunk in stream:
                token = chunk["message"]["content"]
                full_reply += token
                yield token

            self.messages.append({"role": "assistant", "content": full_reply})
        except Exception as e:
            fallback = f"I understand you are feeling {emotion.lower()}. How can I support you right now?"
            yield fallback
            self.messages.append({"role": "assistant", "content": fallback})
