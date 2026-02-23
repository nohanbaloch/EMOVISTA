import ollama
import json
import os

def load_system_prompt():
    config_path = os.path.join(os.path.dirname(__file__), "persona_config.json")
    try:
        with open(config_path, "r") as f:
            c = json.load(f)
            
        # Extract nested configuration
        comm = c.get('communication', {})
        tone = ", ".join(comm.get('tone', ['Calm', 'Professional']))
        
        style_cfg = comm.get('style', {})
        length_pref = style_cfg.get('sentence_length', 'short')
        
        # Force strict length constraints
        length_instruction = ""
        if length_pref == "very-short":
            length_instruction = "CRITICAL: Keep your response extremely brief. Maximum 1-2 sentences. No fluff."
        elif length_pref == "short":
            length_instruction = "Keep your response concise. Maximum 3 sentences."

        style = f"Level: {style_cfg.get('language_level', 'Simple')}. {length_instruction}"
        
        # New Sections
        emergency = c.get('emergency_resources', {})
        boundaries = c.get('topic_boundaries', {})
        coping = c.get('coping_mechanisms_library', {})
        
        # Combine instructions
        rules = []
        rules.append(f"Global Disclaimer: {emergency.get('global_disclaimer', '')}")
        rules.extend(c.get('intended_use', []))
        rules.extend(comm.get('response_guidelines', []))
        rules.extend(c.get('medical_guidance_rules', []))
        
        # Add Boundary Rules
        if boundaries:
             rules.append(f"Refuse topics: {', '.join(boundaries.get('refusal_topics', []))}")
             rules.append(f"Refusal phrase: '{boundaries.get('refusal_template', '')}'")

        instructions = "\n".join([f"- {r}" for r in rules])
        
        # Emergency Context
        hotlines = emergency.get('us_hotlines', {})
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
