
import os
from dotenv import load_dotenv
from gnani.tts import GnaniTTSClient

load_dotenv()

api_key = os.getenv("GNANI_API_KEY")

if not api_key:
    raise RuntimeError("GNANI_API_KEY missing. Check your .env file.")

client = GnaniTTSClient(api_key=api_key)

text = "నమస్కారం! నేను మీ తెలుగు వాయిస్ అసిస్టెంట్‌ను. మీకు ఎలా సహాయం చేయగలను?"

client.synthesize(
    text=text,
    voice="Pranav",
    model="timbre-v2.5",
    language="te-IN",
    output_file="test_tts.wav",
)

print("Success! Audio saved as test_tts.wav")
