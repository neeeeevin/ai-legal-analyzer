import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ FIXED MODEL NAME
model = genai.GenerativeModel("gemini-1.5-flash-latest")


def generate_summary(texts):

    if not texts:
        return "No content to summarize."

    combined_text = " ".join(texts)[:4000]

    prompt = f"""
Explain this legal document in simple English.

- Write 6–8 clear bullet points
- Each point must be complete
- Do NOT copy sentences directly
- Make it easy for a normal person

Text:
{combined_text}
"""

    try:
        response = model.generate_content(prompt)

        if not response or not hasattr(response, "text"):
            return "⚠️ Gemini returned empty response"

        return response.text.strip()

    except Exception as e:
        return f"⚠️ Summary error: {str(e)}"