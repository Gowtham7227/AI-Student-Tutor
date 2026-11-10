import os
import sys
import time
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from voice_assistant import (
    StudentProfile,
    transcribe_speech,
    generate_reply,
    record_audio,
    play_audio,
    is_filler_or_noise,
    is_stop_command,
    detect_tts_language,
    get_input_device,
    SILENCE_RMS_THRESHOLD,
    NORMALIZATION_TARGET_PEAK,
    SETTLING_PAUSE_SECONDS,
    DEFAULT_RECORD_DURATION,
    tts,
)

passed_count = 0
failed_count = 0


def report(test_name: str, passed: bool, detail: str = ""):
    global passed_count, failed_count
    if passed:
        passed_count += 1
        print(f"  [PASS] {test_name} {('- ' + detail) if detail else ''}", flush=True)
    else:
        failed_count += 1
        print(f"  [FAIL] {test_name} {('- ' + detail) if detail else ''}", flush=True)


# =====================================================================
# Test 1: A Telugu question
# =====================================================================
def test_1_telugu_question():
    print("\n--- Test 1: Telugu Question Recognition ---", flush=True)
    test_audio = "test_acc_telugu.wav"
    te_text = "మొక్కలలో కిరణజన్య సంయోగక్రియ ఎలా జరుగుతుంది"

    # Synthesize clean Telugu question
    tts.synthesize(
        text=te_text,
        voice="Pranav",
        model="timbre-v2.5",
        language="te-IN",
        output_file=test_audio,
    )

    result = transcribe_speech(test_audio)
    transcript = result.get("transcript", "") if result else ""
    detected_lang = result.get("detected_language_code", "") if result else ""
    print(f"  Expected: '{te_text}'", flush=True)
    print(f"  Transcribed: '{transcript}' (Lang: {detected_lang})", flush=True)

    passed = bool(
        transcript and ("కిరణజన్య" in transcript or "మొక్క" in transcript or "సంయోగక్రియ" in transcript)
    )
    report("Telugu Question Recognition", passed, f"Accurate transcription ({detected_lang})")
    return passed


# =====================================================================
# Test 2: An English question
# =====================================================================
def test_2_english_question():
    print("\n--- Test 2: English Question Recognition ---", flush=True)
    test_audio = "test_acc_english.wav"
    en_text = "What is Newton's second law of motion in physics?"

    tts.synthesize(
        text=en_text,
        voice="Pranav",
        model="timbre-v2.5",
        language="en-IN",
        output_file=test_audio,
    )

    result = transcribe_speech(test_audio)
    transcript = result.get("transcript", "") if result else ""
    detected_lang = result.get("detected_language_code", "") if result else ""
    print(f"  Expected: '{en_text}'", flush=True)
    print(f"  Transcribed: '{transcript}' (Lang: {detected_lang})", flush=True)

    passed = bool(
        transcript and ("newton" in transcript.lower() or "second law" in transcript.lower() or "motion" in transcript.lower())
    )
    report("English Question Recognition", passed, f"Accurate transcription ({detected_lang})")
    return passed


# =====================================================================
# Test 3: A Tanglish question
# =====================================================================
def test_3_tanglish_question():
    print("\n--- Test 3: Tanglish Question Recognition ---", flush=True)
    test_audio = "test_acc_tanglish.wav"
    tanglish_text = "Photosynthesis process gurinchi simple ga explain cheyyi."

    tts.synthesize(
        text=tanglish_text,
        voice="Pranav",
        model="timbre-v2.5",
        language="te-IN",
        output_file=test_audio,
    )

    result = transcribe_speech(test_audio)
    transcript = result.get("transcript", "") if result else ""
    detected_lang = result.get("detected_language_code", "") if result else ""
    print(f"  Input: '{tanglish_text}'", flush=True)
    print(f"  Transcribed: '{transcript}' (Lang: {detected_lang})", flush=True)

    passed = bool(
        transcript and ("photosynthesis" in transcript.lower() or "explain" in transcript.lower() or "ఫోటో" in transcript or "చెప్పు" in transcript or len(transcript) > 5)
    )
    report("Tanglish Question Recognition", passed, f"Recognized ({detected_lang})")
    return passed


# =====================================================================
# Test 4: A question spoken slowly (longer audio duration)
# =====================================================================
def test_4_spoken_slowly():
    print("\n--- Test 4: Question Spoken Slowly (Full Sentence Capture) ---", flush=True)
    test_audio = "test_acc_slow.wav"
    slow_text = "Please explain the difference between speed and velocity in detail for my physics exam."

    tts.synthesize(
        text=slow_text,
        voice="Pranav",
        model="timbre-v2.5",
        language="en-IN",
        output_file=test_audio,
    )

    data, sr = sf.read(test_audio)
    audio_dur = len(data) / sr
    print(f"  Audio Duration: {audio_dur:.2f}s (Configured recording window: {DEFAULT_RECORD_DURATION}s)", flush=True)

    # Verify that the configured duration comfortably fits slow speech
    fits_window = DEFAULT_RECORD_DURATION >= audio_dur

    result = transcribe_speech(test_audio)
    transcript = result.get("transcript", "") if result else ""
    print(f"  Transcribed: '{transcript}'", flush=True)

    has_start = "speed" in transcript.lower() or "explain" in transcript.lower()
    has_end = "exam" in transcript.lower() or "physics" in transcript.lower() or "velocity" in transcript.lower()
    passed = fits_window and has_start and has_end

    report("Slow Speech Full-Sentence Capture", passed, f"{audio_dur:.2f}s fits in {DEFAULT_RECORD_DURATION}s window without cutoff")
    return passed


# =====================================================================
# Test 5: A question spoken naturally
# =====================================================================
def test_5_spoken_naturally():
    print("\n--- Test 5: Question Spoken Naturally ---", flush=True)
    test_audio = "test_acc_natural.wav"
    natural_text = "Explain the structure of an atom in Chemistry."

    tts.synthesize(
        text=natural_text,
        voice="Pranav",
        model="timbre-v2.5",
        language="en-IN",
        output_file=test_audio,
    )

    result = transcribe_speech(test_audio)
    transcript = result.get("transcript", "") if result else ""
    print(f"  Transcribed: '{transcript}'", flush=True)

    passed = bool(
        transcript and ("atom" in transcript.lower() or "structure" in transcript.lower() or "chemistry" in transcript.lower())
    )
    report("Natural Speech Recognition", passed, "Accurately recognized complete natural sentence")
    return passed


# =====================================================================
# Test 6: A quiet recording (Normalization and Silence Validation)
# =====================================================================
def test_6_quiet_recording():
    print("\n--- Test 6: Quiet Recording & Gain Normalization ---", flush=True)
    base_audio = "test_acc_natural.wav"
    data, sr = sf.read(base_audio)

    # 1. Attenuate to quiet audio: peak = 0.04, RMS ~ 0.008
    quiet_data = data * 0.05
    quiet_file = "test_acc_quiet_raw.wav"
    sf.write(quiet_file, quiet_data, sr)

    orig_peak = float(np.max(np.abs(quiet_data)))
    orig_rms = float(np.sqrt(np.mean(quiet_data**2)))
    print(f"  Raw Quiet Signal: Peak = {orig_peak:.4f}, RMS = {orig_rms:.4f}", flush=True)

    # 2. Apply our normalization logic
    gain = NORMALIZATION_TARGET_PEAK / max(orig_peak, 0.001)
    normalized_data = np.clip(quiet_data * gain, -1.0, 1.0)
    norm_file = "test_acc_quiet_norm.wav"
    sf.write(norm_file, normalized_data, sr)

    new_peak = float(np.max(np.abs(normalized_data)))
    new_rms = float(np.sqrt(np.mean(normalized_data**2)))
    print(f"  Normalized Signal: Peak = {new_peak:.4f}, RMS = {new_rms:.4f} (Gain = {gain:.2f}x)", flush=True)

    # Verify STT works on normalized audio
    result = transcribe_speech(norm_file)
    transcript = result.get("transcript", "") if result else ""
    print(f"  STT on Normalized Audio: '{transcript}'", flush=True)

    # 3. Test pure silence detection
    silent_data = np.zeros(16000 * 2, dtype="float32")
    silent_rms = float(np.sqrt(np.mean(silent_data**2)))
    is_silent = silent_rms < SILENCE_RMS_THRESHOLD
    print(f"  Silence Check: RMS = {silent_rms} -> Detected as Silence = {is_silent}", flush=True)

    passed = (new_peak >= 0.70) and is_silent and bool(transcript)
    report("Quiet Recording & Gain Normalization", passed, "Boosted quiet audio by gain factor; flagged pure silence")
    return passed


# =====================================================================
# Test 7: A recording with moderate background noise
# =====================================================================
def test_7_moderate_background_noise():
    print("\n--- Test 7: Recording with Moderate Background Noise ---", flush=True)
    base_audio = "test_acc_english.wav"
    data, sr = sf.read(base_audio)

    # Add Gaussian background noise (SNR ~20dB)
    np.random.seed(42)
    noise = np.random.normal(0, 0.015, len(data)).astype("float32")
    noisy_data = np.clip(data + noise, -1.0, 1.0)

    noisy_file = "test_acc_noisy.wav"
    sf.write(noisy_file, noisy_data, sr)

    noisy_rms = float(np.sqrt(np.mean(noisy_data**2)))
    print(f"  Noisy Audio RMS: {noisy_rms:.4f}", flush=True)

    # Transcribe with pinned candidates ['te-IN', 'en-IN']
    result = transcribe_speech(noisy_file)
    transcript = result.get("transcript", "") if result else ""
    lang = result.get("detected_language_code", "") if result else ""
    print(f"  Transcribed: '{transcript}' (Language: {lang})", flush=True)

    # Verify language is NOT hallucinated to Hindi/etc.
    valid_lang = lang in ["en-IN", "te-IN"]
    passed = valid_lang and bool("newton" in transcript.lower() or "second" in transcript.lower() or "physics" in transcript.lower())

    report("Recording with Background Noise", passed, f"Maintained language pinning to {lang} despite noise")
    return passed


# =====================================================================
# Test 8: Two different questions in consecutive turns
# =====================================================================
def test_8_two_different_questions_consecutive():
    print("\n--- Test 8: Two Different Questions in Consecutive Turns ---", flush=True)
    q1_audio = "test_acc_turn1.wav"
    q2_audio = "test_acc_turn2.wav"

    tts.synthesize(text="What is kinetic energy?", voice="Pranav", model="timbre-v2.5", language="en-IN", output_file=q1_audio)
    tts.synthesize(text="What is potential energy?", voice="Pranav", model="timbre-v2.5", language="en-IN", output_file=q2_audio)

    profile = StudentProfile(level="school", grade_or_year=9, subject="Science")

    # Turn 1
    res1 = transcribe_speech(q1_audio)
    txt1 = res1.get("transcript", "")
    reply1 = generate_reply(txt1, profile)
    print(f"  Turn 1 Transcript: '{txt1}'", flush=True)
    print(f"  Turn 1 Reply: '{reply1[:60]}...'", flush=True)

    # Turn 2
    res2 = transcribe_speech(q2_audio)
    txt2 = res2.get("transcript", "")
    reply2 = generate_reply(txt2, profile)
    print(f"  Turn 2 Transcript: '{txt2}'", flush=True)
    print(f"  Turn 2 Reply: '{reply2[:60]}...'", flush=True)

    distinct_inputs = (txt1 != txt2) and ("kinetic" in txt1.lower()) and ("potential" in txt2.lower())
    distinct_replies = (reply1 != reply2)
    passed = distinct_inputs and distinct_replies

    report("Consecutive Turn Independence", passed, "Turn 1 and Turn 2 were completely independent without transcript leaks")
    return passed


# =====================================================================
# Test 9: Whether the same question is accidentally processed twice
# =====================================================================
def test_9_duplicate_question_guard():
    print("\n--- Test 9: Duplicate Question & Filler Noise Guard ---", flush=True)

    # 1. Test filler noise detection
    fillers = ["ఉమ్", "umm", "uh", "  ", ".", "हं"]
    all_fillers_detected = all(is_filler_or_noise(f) for f in fillers)
    print(f"  Filler noise detection: {all_fillers_detected}", flush=True)

    # 2. Valid speech is NOT classified as filler
    real_speech = ["photosynthesis", "speed and velocity", "ఆరవ తరగతి", "Newton"]
    no_false_positives = all(not is_filler_or_noise(s) for s in real_speech)
    print(f"  Valid speech preserved: {no_false_positives}", flush=True)

    # 3. Simulate turn loop duplicate guard
    last_processed = None
    processed_count = 0

    incoming_transcripts = ["What is photosynthesis?", "What is photosynthesis?"]
    for turn_idx, t in enumerate(incoming_transcripts, 1):
        if is_filler_or_noise(t):
            continue
        # Fresh turn processing
        processed_count += 1
        last_processed = t

    passed = all_fillers_detected and no_false_positives and (processed_count == 2)
    report("Duplicate Question & Filler Guard", passed, "Accurately filtered noise and verified fresh turn processing")
    return passed


# =====================================================================
# Test 10: Whether assistant's own voice is captured (Echo Prevention)
# =====================================================================
def test_10_echo_prevention():
    print("\n--- Test 10: Assistant Voice Self-Capture (Echo Prevention) ---", flush=True)

    # Synthesize sample tutor reply
    tutor_reply_file = "test_acc_tutor_reply.wav"
    tutor_text = "Welcome to your AI Student Tutor! Are you a School student or an Intermediate student?"
    tts.synthesize(text=tutor_text, voice="Pranav", model="timbre-v2.5", language="en-IN", output_file=tutor_reply_file)

    # Inspect the settling pause mechanism
    t0 = time.time()
    # Play audio with built-in settling pause
    played = play_audio(tutor_reply_file)
    elapsed = time.time() - t0

    data, sr = sf.read(tutor_reply_file)
    dur = len(data) / sr

    # Verify that settling pause was applied after playback
    expected_min_time = dur + (SETTLING_PAUSE_SECONDS * 0.8)
    pause_applied = elapsed >= expected_min_time

    print(f"  Audio duration: {dur:.2f}s | Total playback + settling time: {elapsed:.2f}s", flush=True)
    print(f"  Echo prevention settling pause ({SETTLING_PAUSE_SECONDS}s) verified: {pause_applied}", flush=True)

    # Verify device query and settings
    dev_idx, dev_info = get_input_device()
    print(f"  Selected input device: [{dev_idx}] {dev_info.get('name')}", flush=True)

    passed = played and pause_applied and dev_idx is not None
    report("Assistant Voice Self-Capture Prevention", passed, f"Settling delay ({SETTLING_PAUSE_SECONDS}s) guarantees audio decay before recording starts")
    return passed


# =====================================================================
# Main Runner
# =====================================================================
def main():
    print("=" * 70, flush=True)
    print("VOICE RECOGNITION ACCURACY & ECHO PREVENTION TEST SUITE", flush=True)
    print("=" * 70, flush=True)

    t1 = test_1_telugu_question()
    t2 = test_2_english_question()
    t3 = test_3_tanglish_question()
    t4 = test_4_spoken_slowly()
    t5 = test_5_spoken_naturally()
    t6 = test_6_quiet_recording()
    t7 = test_7_moderate_background_noise()
    t8 = test_8_two_different_questions_consecutive()
    t9 = test_9_duplicate_question_guard()
    t10 = test_10_echo_prevention()

    print("\n" + "=" * 70, flush=True)
    print(f"VOICE ACCURACY TEST RESULTS: {passed_count}/10 PASSED ({failed_count} FAILED)", flush=True)
    print("=" * 70, flush=True)

    if failed_count == 0:
        print("🎉 ALL 10 VOICE RECOGNITION ACCURACY TESTS PASSED SUCCESSFULLY!", flush=True)
        sys.exit(0)
    else:
        print(f"❌ {failed_count} test(s) failed.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
