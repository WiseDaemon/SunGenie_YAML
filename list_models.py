import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/LLM/.env")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY is not set.")
    exit(1)

genai.configure(api_key=api_key)

print("Listing available models from API...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {str(e)}")
