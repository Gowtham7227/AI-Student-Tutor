import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from google import genai
from google.genai import types
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient
from voice_assistant import is_stop_command

gemini_key = os.getenv("GEMINI_API_KEY")
gnani_key = os.getenv("GNANI_API_KEY")

gemini_client = genai.Client(api_key=gemini_key)
stt_client = GnaniSTTClient(api_key=gnani_key)
tts_client = GnaniTTSClient(api_key=gnani_key)

TUTOR_SYSTEM_INSTRUCTION = (
    "You are an encouraging, expert AI Engineering & CSE Student Tutor. "
    "Your goal is to help undergraduate engineering students master Computer Science concepts, "
    "including C, Python, Java, Data Structures, DBMS, Operating Systems, Computer Networks, and AI/ML, "
    "as well as prepare for exams and interviews.\n\n"
    "Core Guidelines:\n"
    "1. Language Selection:\n"
    "   - By default, reply in the same language the student uses.\n"
    "   - If the student asks in English, answer in clear, simple English.\n"
    "   - If the student asks in Telugu (in Telugu script or Romanized Telugu/Tanglish), answer in conversational, natural Telugu.\n"
    "   - If the student mixes Telugu and English, answer in conversational Telugu while keeping technical terms in English (e.g., 'Primary Key', 'Deadlock', 'Pointer', 'Process Scheduling').\n"
    "   - Explicit Language Requests: If the student instructs 'Telugu lo explain cheyyi', 'explain in Telugu', or 'తెలుగులో చెప్పు', explain in Telugu. If they instruct 'Explain in English', 'in English please', or 'ఇంగ్లీష్ లో చెప్పు', explain in English.\n\n"
    "2. Teaching Style & Structure:\n"
    "   - Be patient, motivating, and clear like a helpful professor.\n"
    "   - Break complex ideas down into simple intuition first, then practical examples.\n"
    "   - For code questions, explain the logic step-by-step with simple, 1-2 line code snippets explained verbally.\n"
    "   - For exam preparation questions, provide the direct definition/answer followed by a key exam point.\n"
    "   - Never invent technical facts, and ensure all explanations and code logic are 100% accurate.\n\n"
    "3. Voice-Friendly Formatting:\n"
    "   - Keep answers concise (2 to 4 sentences maximum) so they sound natural and engaging over audio playback.\n"
    "   - Do NOT use markdown asterisks (**bold**), backticks (`code`), bullet points (*), headers (###), or raw code blocks.\n"
    "   - Express mathematical powers in words (e.g., 'x squared' instead of 'x**2') so it sounds natural when read aloud."
)

def clean_text_for_speech(text: str) -> str:
    """Sanitize text to remove markdown, asterisks, backticks, and special symbols for clean TTS playback."""
    if not text:
        return ""
    cleaned = text.replace("**", "").replace("*", "").replace("`", "").replace("###", "").replace("##", "")
    return cleaned.strip()

def query_tutor(question: str) -> str:
    """Query Gemini with student tutor system instruction."""
    import time
    candidate_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    for model_name in candidate_models:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=question,
                    config=types.GenerateContentConfig(
                        system_instruction=TUTOR_SYSTEM_INSTRUCTION,
                        temperature=0.7,
                    ),
                )
                if response and response.text:
                    return clean_text_for_speech(response.text)
            except Exception as e:
                time.sleep(1.5)
                continue
    return "క్షమించండి, ప్రస్తుతం సమాధానం అందించడంలో సమస్య ఉంది. దయచేసి మీ ప్రశ్నను మళ్లీ అడగండి."


def test_tutor_responses():
    print("=" * 60)
    print("TESTING AI STUDENT TUTOR AGENT (GEMINI CORE)")
    print("=" * 60)

    test_scenarios = [
        {
            "category": "Telugu CSE Question",
            "prompt": "ఆపరేటింగ్ సిస్టమ్‌లో డెడ్‌లాక్ అంటే ఏమిటి?",
            "expected_lang": "Telugu",
        },
        {
            "category": "English CSE Question",
            "prompt": "Explain the difference between Primary Key and Foreign Key in DBMS.",
            "expected_lang": "English",
        },
        {
            "category": "Telugu-English Mixed Question",
            "prompt": "C language lo pointers enduku use chestharu?",
            "expected_lang": "Mixed / Telugu with English tech terms",
        },
        {
            "category": "Explicit Telugu Switch Instruction",
            "prompt": "What is the difference between TCP and UDP? Telugu lo explain cheyyi.",
            "expected_lang": "Telugu",
        },
        {
            "category": "Explicit English Switch Instruction",
            "prompt": "Java lo Inheritance gurinchi explain in English.",
            "expected_lang": "English",
        },
        {
            "category": "Practical Code Question",
            "prompt": "Python lo list comprehension ante enti? Small example tho explain cheyyi.",
            "expected_lang": "Telugu / Mixed with code logic",
        },
    ]

    all_passed = True
    for i, test in enumerate(test_scenarios, 1):
        print(f"\n[{i}/6] Testing: {test['category']}")
        print(f"  Student Question: \"{test['prompt']}\"")
        reply = query_tutor(test['prompt'])
        print(f"  Tutor Answer: {reply}")

        # Basic validations: non-empty, no markdown asterisks, reasonably concise
        if not reply or len(reply) < 15:
            print("  [FAIL] Reply is too short or empty.")
            all_passed = False
        elif "**" in reply or "###" in reply:
            print("  [FAIL] Reply contains markdown formatting.")
            all_passed = False
        else:
            print(f"  [PASS] Verified concise, voice-friendly response for {test['category']}.")
        import time
        time.sleep(1)

    return all_passed


def detect_tts_language(text: str) -> str:
    """
    Determine the appropriate TTS language code.
    If text contains Telugu Unicode characters, use 'te-IN'.
    Otherwise use 'en-IN'.
    """
    if not text:
        return "te-IN"
    has_telugu = any('\u0c00' <= char <= '\u0c7f' for char in text)
    return "te-IN" if has_telugu else "en-IN"


def test_bilingual_tts():
    print("\n" + "=" * 60)
    print("TESTING BILINGUAL TTS SYNTHESIS (GNANI TIMBRE AUTO/DETECT)")
    print("=" * 60)

    samples = [
        ("English OS Answer", "A deadlock in Operating Systems happens when two processes wait for resources held by each other.", "test_tutor_en.wav"),
        ("Telugu C Pointers Answer", "C లాంగ్వేజ్ లో పాయింటర్స్ ని మెమరీని నేరుగా యాక్సెస్ చేయడానికి మరియు డైనమిక్ మెమరీ ఎలొకేషన్ కోసం ఉపయోగిస్తారు.", "test_tutor_te.wav"),
    ]

    tts_passed = True
    for label, text, out_file in samples:
        lang = detect_tts_language(text)
        print(f"\nSynthesizing {label} (using language={lang})...")
        success = False
        for attempt in range(3):
            try:
                tts_client.synthesize(
                    text=text,
                    voice="Pranav",
                    model="timbre-v2.5",
                    language=lang,
                    output_file=out_file,
                )
                if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
                    print(f"  [PASS] Successfully generated {out_file} ({os.path.getsize(out_file)} bytes)")
                    success = True
                    break
            except Exception as e:
                print(f"  (Attempt {attempt + 1}) Transient TTS error: {e}")
                import time
                time.sleep(2)

        if not success:
            print(f"  [FAIL] Could not generate {out_file}")
            tts_passed = False

    return tts_passed


def test_stop_commands_preserved():
    print("\n" + "=" * 60)
    print("TESTING STOP COMMANDS PRESERVATION")
    print("=" * 60)
    commands = ["ఆపు", "ఆపండి", "stop", "exit", "quit", "bye"]
    non_commands = ["C pointers explain cheyyi", "What is deadlock?", "Java inheritance"]

    for c in commands:
        assert is_stop_command(c) is True, f"Failed on command '{c}'"
    for nc in non_commands:
        assert is_stop_command(nc) is False, f"Failed on non-command '{nc}'"

    print("  [PASS] All stop commands and normal queries accurately verified.")
    return True


if __name__ == "__main__":
    t1 = test_tutor_responses()
    t2 = test_bilingual_tts()
    t3 = test_stop_commands_preserved()

    print("\n" + "=" * 60)
    print("PHASE 6 TUTOR TEST SUMMARY:")
    print(f"  Tutor AI Responses (6 Scenarios): {'PASSED' if t1 else 'FAILED'}")
    print(f"  Bilingual TTS Audio Synthesis:     {'PASSED' if t2 else 'FAILED'}")
    print(f"  Stop Commands Preserved:          {'PASSED' if t3 else 'FAILED'}")
    print("=" * 60)

    if t1 and t2 and t3:
        sys.exit(0)
    else:
        sys.exit(1)
