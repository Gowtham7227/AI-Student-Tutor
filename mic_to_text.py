import os
import sys
from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient

# Ensure UTF-8 output for clean Telugu/Tanglish printing
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

api_key = os.getenv("GNANI_API_KEY")
if not api_key:
    raise RuntimeError("GNANI_API_KEY missing. Check your .env file.")

audio_file = "mic_test.wav"

if not os.path.exists(audio_file):
    raise FileNotFoundError("Run test_mic.py first to record audio.")

client = GnaniSTTClient(api_key=api_key)

print(f"🎙️ Transcribing '{audio_file}' with Gnani Prisma STT (indic-auto: Telugu & English)...")

try:
    result = client.transcribe(
        audio=audio_file,
        model="prisma-v3.0",
        language_code="indic-auto",
        language_candidates=["te-IN", "en-IN"],
        format="transcribe",
    )
except Exception as e:
    print(f"⚠️ Primary indic-auto STT error: {e}. Trying te-IN fallback...")
    result = client.transcribe(
        audio=audio_file,
        model="prisma-v3.0",
        language_code="te-IN",
        format="transcribe",
    )

print("\n==================================================")
print("🧠 GNANI STT RESULT:")
print("==================================================")
if result:
    req_id = result.get("request_id", "N/A")
    lang = result.get("detected_language_code", "unknown")
    srv_time = result.get("processing_time", 0.0)
    transcript = result.get("transcript", "")
    print(f"• Request ID       : {req_id}")
    print(f"• Detected Language: {lang}")
    print(f"• Processing Time  : {srv_time:.3f}s")
    print(f"• Transcript       : '{transcript}'")
    if not transcript:
        print("ℹ️ Note: Transcript is empty. The audio may contain silence, faint whispering, or background noise.")
else:
    print("❌ No response received from STT service.")
print("==================================================")
