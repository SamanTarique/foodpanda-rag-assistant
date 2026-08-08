from google import genai
from config import GEMINI_API_KEY
 
client = genai.Client(api_key=GEMINI_API_KEY)
 
print("Available models for your API key:\n")
print(f"{'Model Name':<35} {'Supports':<40}")
print("-" * 75)
 
for model in client.models.list():
    name = model.name.replace("models/", "")
    methods = ", ".join(model.supported_actions) if hasattr(model, "supported_actions") else "?"
    print(f"{name:<35} {methods:<40}")
