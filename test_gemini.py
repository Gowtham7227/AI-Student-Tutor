import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY missing. Check your .env file.")

client = genai.Client(api_key=api_key)

print("Checking available Gemini models for this API key...")
available_models = []
try:
    for model in client.models.list():
        name = getattr(model, "name", "")
        if "gemini" in name:
            available_models.append(name)
except Exception as e:
    print(f"Error querying model list: {e}")

print(f"Total Gemini models found: {len(available_models)}")
for m in available_models:
    print(f"  - {m}")

candidate_models = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

test_prompt = "నమస్కారం! మీరు ఎవరు?"
print(f"Prompt: {test_prompt}")

success = False
for model_name in candidate_models:
    print(f"\nTrying model: {model_name}...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=test_prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful Telugu voice assistant. "
                    "Respond in natural, spoken Telugu. "
                    "Keep your response concise (1-2 sentences) suitable for speech synthesis. "
                    "Do not include markdown or asterisks."
                ),
                temperature=0.7,
            ),
        )
        if response and response.text:
            print(f"SUCCESS with {model_name}!")
            print(f"Gemini reply: {response.text.strip()}")
            success = True
            selected_model = model_name
            break
    except Exception as e:
        print(f"Failed with {model_name}: {e}")

if success:
    print(f"\nWorking model selected: {selected_model}")
    print("Gemini test completed successfully!")
else:
    print("\nAll candidate models failed.")
