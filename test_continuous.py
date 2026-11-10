import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from voice_assistant import is_stop_command, generate_reply, stt, tts

def test_stop_command_detection():
    """Verify Telugu and English stop commands are recognized accurately."""
    print("=== Test 1: Stop Command Detection ===")
    test_cases = [
        ("ఆపు", True),
        ("ఆపండి", True),
        ("దయచేసి ఆపండి", True),
        ("ఆపివేయి", True),
        ("ముగించు", True),
        ("stop", True),
        ("STOP", True),
        ("Stop please", True),
        ("exit", True),
        ("EXIT", True),
        ("quit", True),
        ("bye", True),
        ("నమస్కారం నా పేరు గౌతమ్ రెడ్డి", False),
        ("ఈ రోజు వాతావరణం ఎలా ఉంది?", False),
        ("నాకు సహాయం కావాలి", False),
        ("", False),
    ]

    failed = 0
    for phrase, expected in test_cases:
        actual = is_stop_command(phrase)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            failed += 1
        print(f"  [{status}] Phrase: '{phrase}' -> Expected: {expected}, Got: {actual}")

    if failed == 0:
        print("--> Test 1 PASSED: All stop commands accurately classified.\n")
        return True
    else:
        print(f"--> Test 1 FAILED: {failed} test cases failed.\n")
        return False


def test_audio_stop_pipeline():
    """Synthesize 'ఆపండి' using Gnani TTS, transcribe with Gnani STT, and verify stop detection."""
    print("=== Test 2: Audio STT Stop Command Pipeline ===")
    stop_audio = "test_stop_command.wav"
    stop_text = "ఆపండి"

    print(f"1. Synthesizing stop phrase '{stop_text}' via Gnani TTS...")
    tts.synthesize(
        text=stop_text,
        voice="Pranav",
        model="timbre-v2.5",
        language="te-IN",
        output_file=stop_audio,
    )
    assert os.path.exists(stop_audio), "Stop audio file was not created"
    print(f"   Audio saved to {stop_audio}")

    print("2. Transcribing stop audio via Gnani STT...")
    stt_res = None
    for attempt in range(3):
        try:
            stt_res = stt.transcribe(audio=stop_audio, language_code="te-IN")
            if stt_res and stt_res.get("success"):
                break
        except Exception as e:
            print(f"   (Attempt {attempt + 1}) STT transient error: {e}")
            import time
            time.sleep(2)

    assert stt_res is not None, "STT failed after 3 attempts"
    transcript = stt_res.get("transcript", "").strip()
    print(f"   Transcribed text: '{transcript}'")

    print("3. Checking if transcribed text triggers stop command...")
    is_stop = is_stop_command(transcript)
    print(f"   is_stop_command: {is_stop}")

    if is_stop:
        print("--> Test 2 PASSED: Stop command audio synthesized, transcribed, and detected!\n")
        return True
    else:
        print("--> Test 2 FAILED: Stop command not detected in transcript.\n")
        return False


def test_multi_turn_simulation():
    """
    Simulate a multi-turn continuous conversation loop:
    Round 1: Real user question ('mic_test.wav') -> STT -> Gemini AI -> TTS synthesis
    Round 2: Empty audio -> STT returns empty -> graceful skip/continue
    Round 3: Stop audio ('test_stop_command.wav') -> STT -> Stop detected -> Goodbye message synthesized -> Loop breaks
    """
    print("=== Test 3: Multi-turn Continuous Conversation Simulation ===")
    
    # We will simulate 3 rounds
    rounds = [
        {"desc": "Round 1: User question", "audio": "test_tts.wav"},
        {"desc": "Round 2: Empty / silence", "audio": None},
        {"desc": "Round 3: Stop command", "audio": "test_stop_command.wav"},
    ]

    output_file = "assistant_reply.wav"
    loop_terminated_cleanly = False
    turn_count = 0

    for item in rounds:
        turn_count += 1
        print(f"\n--- [Simulation Turn {turn_count}: {item['desc']}] ---")
        audio_file = item["audio"]

        # Step 1: Simulate audio input
        if audio_file is None:
            print("ℹ️ Simulating no speech detected...")
            transcript = ""
        else:
            print(f"🧠 Transcribing speech from {audio_file}...")
            result = None
            for attempt in range(3):
                try:
                    result = stt.transcribe(audio=audio_file, language_code="te-IN")
                    if result and result.get("success"):
                        break
                except Exception as e:
                    print(f"   (Attempt {attempt + 1}) STT transient error: {e}")
                    import time
                    time.sleep(2)
            transcript = result.get("transcript", "").strip() if result else ""

        # Step 2: Handle empty transcript
        if not transcript:
            print("ℹ️ No speech detected. Loop continues to next question without error.")
            continue

        print(f"🗣️ You said: {transcript}")

        # Step 3: Check stop command before calling Gemini
        if is_stop_command(transcript):
            print("🛑 Stop command detected! Assistant is generating goodbye message...")
            goodbye_reply = "సరే, సంభాషణ ముగిస్తున్నాను. ధన్యవాదాలు, మళ్లీ కలుద్దాం!"
            print(f"🤖 Assistant: {goodbye_reply}")
            for attempt in range(3):
                try:
                    tts.synthesize(
                        text=goodbye_reply,
                        voice="Pranav",
                        model="timbre-v2.5",
                        language="te-IN",
                        output_file=output_file,
                    )
                    print("🔊 Goodbye audio synthesized successfully.")
                    break
                except Exception as e:
                    print(f"   (Attempt {attempt + 1}) TTS transient error: {e}")
                    import time
                    time.sleep(2)
            loop_terminated_cleanly = True
            break

        # Step 4: Call Gemini AI
        print("🤔 Calling Gemini AI...")
        reply = generate_reply(transcript)
        print(f"🤖 Assistant: {reply}")

        # Step 5: Synthesize TTS
        print("🔊 Synthesizing speech with Gnani TTS...")
        tts.synthesize(
            text=reply,
            voice="Pranav",
            model="timbre-v2.5",
            language="te-IN",
            output_file=output_file,
        )
        print("🔊 Reply audio synthesized successfully.")
        print("🔄 Ready for next question (continuing loop)...")

    if loop_terminated_cleanly and turn_count == 3:
        print("\n--> Test 3 PASSED: Multi-turn continuous conversation loop verified end-to-end!\n")
        return True
    else:
        print(f"\n--> Test 3 FAILED: loop_terminated_cleanly={loop_terminated_cleanly}, turn_count={turn_count}\n")
        return False


if __name__ == "__main__":
    t1 = test_stop_command_detection()
    t2 = test_audio_stop_pipeline()
    t3 = test_multi_turn_simulation()

    print("=" * 50)
    print("FINAL TEST SUMMARY:")
    print(f"  Test 1 (Stop Command Keywords): {'PASSED' if t1 else 'FAILED'}")
    print(f"  Test 2 (STT Stop Audio Pipeline): {'PASSED' if t2 else 'FAILED'}")
    print(f"  Test 3 (Multi-turn Loop Simulation): {'PASSED' if t3 else 'FAILED'}")
    print("=" * 50)

    if t1 and t2 and t3:
        sys.exit(0)
    else:
        sys.exit(1)
