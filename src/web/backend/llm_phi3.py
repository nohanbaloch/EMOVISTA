import ollama
import json
import os
from typing import Any, Iterator, Mapping, cast

def load_system_prompt():
    config_path = os.path.join(os.path.dirname(__file__), "persona_config.json")
    try:
        with open(config_path, "r") as f:
            c = cast(dict[str, Any], json.load(f))
            
        # Extract nested configuration
        comm = cast(dict[str, Any], c.get('communication', {}))
        tone = ", ".join(comm.get('tone', ['Calm', 'Professional']))
        
        style_cfg = cast(dict[str, Any], comm.get('style', {}))
        length_pref = style_cfg.get('sentence_length', 'short')
        
        # Force strict length constraints
        length_instruction = ""
        if length_pref == "very-short":
            length_instruction = "CRITICAL: Keep your response extremely brief. Maximum 1-2 sentences. No fluff."
        elif length_pref == "short":
            length_instruction = "Keep your response concise. Maximum 3 sentences."

        style = f"Level: {style_cfg.get('language_level', 'Simple')}. {length_instruction}"
        
        # New Sections
        emergency = cast(dict[str, Any], c.get('emergency_resources', {}))
        boundaries = cast(dict[str, Any], c.get('topic_boundaries', {}))
        coping = cast(dict[str, Any], c.get('coping_mechanisms_library', {}))
        
        # Combine instructions
        rules: list[str] = []
        rules.append(f"Global Disclaimer: {emergency.get('global_disclaimer', '')}")
        rules.extend([str(item) for item in c.get('intended_use', [])])
        rules.extend([str(item) for item in comm.get('response_guidelines', [])])
        rules.extend([str(item) for item in c.get('medical_guidance_rules', [])])
        
        # Add Boundary Rules
        if boundaries:
                        rules.append(f"Refuse topics: {', '.join([str(item) for item in boundaries.get('refusal_topics', [])])}")
                        rules.append(f"Refusal phrase: '{boundaries.get('refusal_template', '')}'")

        instructions = "\n".join([f"- {r}" for r in rules])
        
        # Emergency Context
        hotlines = cast(dict[str, Any], emergency.get('us_hotlines', {}))
        emergency_text = f"Emergency Hotlines: Suicide={hotlines.get('suicide_prevention')}, Emergency={hotlines.get('emergency')}"

        return f"""
{c.get('role', 'You are a medical AI assistant.')}

Tone: {tone}.
Style: {style}

CRITICAL RULES:
{length_instruction}
- Do NOT be verbose.
- Use these Emergency Resources if needed: {emergency_text}

Instructions:
{instructions}

Coping Strategies (suggest if relevant):
{json.dumps(coping, indent=2)}
"""
    except Exception:
        return """
You are a calm, empathetic medical AI assistant.
You help users understand their emotions and guide them safely.
You do NOT diagnose.
You encourage professional help when needed.
"""

SYSTEM_PROMPT = load_system_prompt()


def list_ollama_models() -> list[str]:
    """Return installed Ollama model names in a robust, version-tolerant way."""
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

    # Preserve order while de-duplicating.
    deduped = list(dict.fromkeys(names))
    return deduped if deduped else ["phi3"]

class Phi3Assistant:
    def __init__(self, model: str = "phi3"):
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def respond(self, emotion: str, user_text: str, model: str | None = None):
        selected_model = (model or self.model).strip() or self.model

        prompt = f"""
Detected emotion: {emotion}
User says: {user_text}

Respond empathetically and medically responsibly.
"""
        self.messages.append({"role": "user", "content": prompt})

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
