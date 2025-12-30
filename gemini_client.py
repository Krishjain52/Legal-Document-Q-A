from google import genai
import os
from dotenv import load_dotenv

load_dotenv()  # 👈 THIS IS THE FIX

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text



