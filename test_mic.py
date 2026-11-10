"""
Standalone Safe Microphone Recording & Audio Diagnostic Test
- Validates active microphone device.
- Records a 4-second audio test sample.
- Measures RMS, peak amplitude, noise floor, and clipping.
- Saves cleanly to mic_test.wav.
- Prepares audio for transcription testing via mic_to_text.py.
"""

import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

from voice_assistant import get_input_device, diagnose_microphone, DEFAULT_SAMPLE_RATE

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_mic_test():
    duration = 4.0
    sample_rate = DEFAULT_SAMPLE_RATE
    filename = "mic_test.wav"

    dev_idx, dev_info = get_input_device()
    print("=" * 70)
    print("🎙️  SAFE MICROPHONE RECORDING & DIAGNOSTIC TEST")
    print("=" * 70)
    print(f"• Active Device: [{dev_idx}] {dev_info.get('name', 'Default Microphone')}")
    print(f"• Input Channels: {dev_info.get('max_input_channels', 1)}")
    print(f"• Sample Rate: {sample_rate} Hz (Float32 Mono)")
    print(f"• Duration: {duration:.1f} seconds")
    print("=" * 70)
    print("\n👉 Please speak a sentence clearly (in English, Telugu, or Tanglish)...")
    print("   Example: 'I am studying in Class 8' or 'నాకు ఫిజిక్స్ చెప్పండి'")
    print("   Recording starts NOW!\n")

    diag = diagnose_microphone(
        duration=duration,
        device=dev_idx,
        sample_rate=sample_rate,
        output_filename=filename,
    )

    print("\n📋 Next Step:")
    if diag.get("stt_suitable"):
        print("✅ Audio recording is healthy and ready for speech recognition.")
        print("👉 Run this command to transcribe your voice:")
        print("   .\\.venv\\Scripts\\python.exe mic_to_text.py")
    else:
        print("⚠️ Audio may be too faint or silent. Check that your microphone is unmuted")
        print("   and speak directly towards the microphone, then run this test again.")


if __name__ == "__main__":
    run_mic_test()
