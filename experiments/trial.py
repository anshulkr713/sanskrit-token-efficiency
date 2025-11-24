import google.generativeai as genai
import os

# Ensure API key is set
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Listing available models supporting 'countTokens':")
for m in genai.list_models():
    if 'countTokens' in m.supported_generation_methods:
        print(f"- {m.name}")