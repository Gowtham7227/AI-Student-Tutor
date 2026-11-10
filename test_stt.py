
import os
import sys
from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

api_key = os.getenv("GNANI_API_KEY")

if not api_key:
    raise RuntimeError("GNANI_API_KEY missing. Check your .env file.")

audio_file = "test_tts.wav"

if not os.path.exists(audio_file):
    raise FileNotFoundError(
        "test_stt.wav not found. Save your recording in the project folder."
    )

client = GnaniSTTClient(api_key=api_key)

result = None
for attempt in range(3):
    try:
        result = client.transcribe(
            audio=audio_file,
            language_code="te-IN",
        )
        if result and result.get("success"):
            break
    except Exception as e:
        print(f"STT attempt {attempt + 1} encountered transient error: {e}")
        import time
        time.sleep(2)

print("STT result:")
print(result)
