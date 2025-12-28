import ollama
import json
import os

def load_system_prompt():
    config_path = os.path.join(os.path.dirname(__file__), "persona_config.json")
    try:
        with open(config_path, "r") as f:
            c = json.load(f)
            
        return f"""
{c.get('role', 'You are a medical AI assistant.')}
Tone: {c.get('tone', 'Cam and professional')}.
Style: {c.get('style', 'Concise')}.

Instructions:
{chr(10).join(['- ' + i for i in c.get('instructions', [])])}
"""
    except Exception:
        return """
You are a calm, empathetic medical AI assistant.
You help users understand their emotions and guide them safely.
You do NOT diagnose.
You encourage professional help when needed.
"""

SYSTEM_PROMPT = load_system_prompt()

class Phi3Assistant:
    def __init__(self, model="phi3"):
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def respond(self, emotion, user_text):
        prompt = f"""
Detected emotion: {emotion}
User says: {user_text}

Respond empathetically and medically responsibly.
"""
        self.messages.append({"role": "user", "content": prompt})

        stream = ollama.chat(
            model=self.model,
            messages=self.messages,
            stream=True
        )

        full_reply = ""
        for chunk in stream:
            token = chunk["message"]["content"]
            full_reply += token
            yield token

        self.messages.append({"role": "assistant", "content": full_reply})
