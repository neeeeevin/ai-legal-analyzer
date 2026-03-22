import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")


def get_risk_level(confidence):
    if confidence >= 0.85:
        return "High"
    elif confidence >= 0.7:
        return "Medium"
    return "Low"


def explain_clause(text):

    prompt = f"""
Explain this legal clause in simple English.

- What does it mean?
- Why is it important?
- Write 3–4 clear sentences

Clause:
{text}
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300
            },
            timeout=10
        )

        data = response.json()

        # 🔥 DEBUG (keep for now)
        print("EXPLAIN RESPONSE:", data)

        if "choices" not in data or not data["choices"]:
            return "⚠️ OpenRouter failed (check API key / limit)"

        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"⚠️ Explain error: {str(e)}"