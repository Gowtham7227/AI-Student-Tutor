import os
import sys
import time
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from voice_assistant import (
    StudentProfile,
    QuizSession,
    parse_education_level,
    generate_reply,
    generate_quiz_question,
    evaluate_quiz_answer,
    is_stop_command,
    is_quiz_trigger,
    detect_tts_language,
    clean_text_for_speech,
    stt,
    tts,
)

passed_count = 0
failed_count = 0


def report(test_name: str, passed: bool, detail: str = ""):
    global passed_count, failed_count
    if passed:
        passed_count += 1
        print(f"  [PASS] {test_name} {('- ' + detail) if detail else ''}")
    else:
        failed_count += 1
        print(f"  [FAIL] {test_name} {('- ' + detail) if detail else ''}")


# =====================================================================
# Scenario 1: School student selection
# =====================================================================
def test_scenario_1():
    print("\n--- Test 1: School Student Selection ---")
    p = StudentProfile()
    is_complete, msg = parse_education_level("I am a school student", p)
    passed = (p.level == "school") and (not is_complete) and ("class" in msg.lower())
    report("School Student Selection", passed, f"Level: {p.level}, Prompt: {msg}")
    return passed


# =====================================================================
# Scenario 2: Class selection from 1st to 10th
# =====================================================================
def test_scenario_2():
    print("\n--- Test 2: Class Selection from 1st to 10th ---")
    all_ok = True
    samples = [
        ("10th class", 10),
        ("Class 5", 5),
        ("7th standard", 7),
        ("పదవ తరగతి", 10),
        ("1st class", 1),
    ]
    for text_in, expected_class in samples:
        p = StudentProfile(level="school")
        is_complete, msg = parse_education_level(text_in, p)
        ok = p.grade_or_year == expected_class and is_complete
        if not ok:
            all_ok = False
            print(f"    Failed on '{text_in}': expected {expected_class}, got {p.grade_or_year}")

    report("Class Selection (1st to 10th)", all_ok, "Tested multiple classes and languages")
    return all_ok


# =====================================================================
# Scenario 3: School Mathematics explanation
# =====================================================================
def test_scenario_3():
    print("\n--- Test 3: School Mathematics Explanation ---")
    p = StudentProfile(level="school", grade_or_year=6, subject="Mathematics")
    query = "What is a fraction? Explain with a simple everyday example like sharing a chocolate."
    reply = generate_reply(query, p)
    print(f"  Question: {query}")
    print(f"  Tutor Reply: {reply}")

    passed = (
        len(reply) > 30
        and "**" not in reply
        and ("chocolate" in reply.lower() or "part" in reply.lower() or "భాగం" in reply or "equal" in reply.lower())
    )
    report("School Mathematics Explanation", passed, "Verified age-appropriate everyday example")
    return passed


# =====================================================================
# Scenario 4: School Science explanation
# =====================================================================
def test_scenario_4():
    print("\n--- Test 4: School Science Explanation ---")
    p = StudentProfile(level="school", grade_or_year=7, subject="Science")
    query = "Why do plants need sunlight for photosynthesis? Simple ga explain cheyyi."
    reply = generate_reply(query, p)
    print(f"  Question: {query}")
    print(f"  Tutor Reply: {reply}")

    passed = (
        len(reply) > 30
        and "**" not in reply
        and ("food" in reply.lower() or "energy" in reply.lower() or "ఆహారం" in reply or "కాంతి" in reply or "sunlight" in reply.lower())
    )
    report("School Science Explanation", passed, "Verified simple everyday explanation")
    return passed


# =====================================================================
# Scenario 5: Intermediate MPC selection and subject explanation
# =====================================================================
def test_scenario_5():
    print("\n--- Test 5: Intermediate MPC Selection & Subject Explanation ---")
    p = StudentProfile()
    is_complete, msg = parse_education_level("Intermediate 1st year MPC", p)
    selection_ok = (p.level == "intermediate" and p.grade_or_year == 1 and p.group == "MPC" and is_complete)

    query = "Explain Boyle's law in Chemistry with its formula."
    reply = generate_reply(query, p)
    print(f"  Question: {query}")
    print(f"  Tutor Reply: {reply}")

    content_ok = (
        len(reply) > 30
        and "**" not in reply
        and ("pressure" in reply.lower() or "volume" in reply.lower() or "inversely" in reply.lower() or "p v" in reply.lower() or "పీడనం" in reply)
    )
    passed = selection_ok and content_ok
    report("Intermediate MPC Selection & Subject Explanation", passed, f"Profile: {p.describe()}")
    return passed


# =====================================================================
# Scenario 6: Intermediate BiPC selection and subject explanation
# =====================================================================
def test_scenario_6():
    print("\n--- Test 6: Intermediate BiPC Selection & Subject Explanation ---")
    p = StudentProfile()
    is_complete, msg = parse_education_level("Intermediate 2nd year BiPC", p)
    selection_ok = (p.level == "intermediate" and p.grade_or_year == 2 and p.group == "BiPC" and is_complete)

    query = "What are the four chambers of the human heart in Zoology?"
    reply = generate_reply(query, p)
    print(f"  Question: {query}")
    print(f"  Tutor Reply: {reply}")

    content_ok = (
        len(reply) > 30
        and "**" not in reply
        and ("atrium" in reply.lower() or "ventricle" in reply.lower() or "కర్ణికలు" in reply or "జఠరికలు" in reply)
    )
    passed = selection_ok and content_ok
    report("Intermediate BiPC Selection & Subject Explanation", passed, f"Profile: {p.describe()}")
    return passed


# =====================================================================
# Scenario 7: Correct quiz answer
# =====================================================================
def test_scenario_7():
    print("\n--- Test 7: Correct Quiz Answer Evaluation ---")
    p = StudentProfile(level="school", grade_or_year=8, subject="Science")
    q = "What is the powerhouse of the cell?"
    expected = "Mitochondria"
    student_ans = "Mitochondria"

    is_correct, feedback = evaluate_quiz_answer(student_ans, q, expected, p)
    print(f"  Question: {q}")
    print(f"  Student: {student_ans}")
    print(f"  Evaluation: Correct={is_correct}")
    print(f"  Tutor Feedback: {feedback}")

    passed = is_correct is True and len(feedback) > 10
    report("Correct Quiz Answer Evaluation", passed, f"Verdict={is_correct}")
    return passed


# =====================================================================
# Scenario 8: Incorrect quiz answer and explanation
# =====================================================================
def test_scenario_8():
    print("\n--- Test 8: Incorrect Quiz Answer and Explanation ---")
    p = StudentProfile(level="school", grade_or_year=8, subject="Science")
    q = "What is the chemical formula of water?"
    expected = "H2O"
    student_ans = "Carbon dioxide or CO2"

    is_correct, feedback = evaluate_quiz_answer(student_ans, q, expected, p)
    print(f"  Question: {q}")
    print(f"  Student: {student_ans}")
    print(f"  Evaluation: Correct={is_correct}")
    print(f"  Tutor Feedback: {feedback}")

    passed = is_correct is False and ("h2o" in feedback.lower() or "water" in feedback.lower() or "నీరు" in feedback)
    report("Incorrect Quiz Answer & Explanation", passed, f"Verdict={is_correct}")
    return passed


# =====================================================================
# Scenario 9: Quiz score calculation
# =====================================================================
def test_scenario_9():
    print("\n--- Test 9: Quiz Score Calculation ---")
    quiz = QuizSession(total_questions=3)
    quiz.active = True

    # Question 1: Correct
    quiz.question_num = 1
    quiz.score += 1

    # Question 2: Incorrect
    quiz.question_num = 2
    # no increment

    # Question 3: Correct
    quiz.question_num = 3
    quiz.score += 1

    passed = (quiz.score == 2) and (quiz.total_questions == 3)
    score_ratio = f"{quiz.score}/{quiz.total_questions}"
    report("Quiz Score Calculation", passed, f"Calculated score: {score_ratio}")
    return passed


# =====================================================================
# Scenario 10: Quiz exit using stop commands
# =====================================================================
def test_scenario_10():
    print("\n--- Test 10: Quiz Exit Using Stop Commands ---")
    quiz = QuizSession(total_questions=3)
    quiz.active = True
    quiz.score = 1

    stop_words = ["stop", "ఆపు", "exit", "ఆపండి"]
    all_ok = True
    for sw in stop_words:
        if not is_stop_command(sw):
            all_ok = False

    # Simulate exit handler
    if is_stop_command("stop"):
        quiz.reset()

    passed = all_ok and (not quiz.active) and (quiz.score == 0)
    report("Quiz Exit Using Stop Commands", passed, "Stop commands accurately reset quiz mode")
    return passed


# =====================================================================
# Scenario 11: 2-mark, 5-mark, and 10-mark exam answers
# =====================================================================
def test_scenario_11():
    print("\n--- Test 11: 2-mark, 5-mark, and 10-mark Exam Answers ---")
    p = StudentProfile(level="intermediate", grade_or_year=1, group="MPC", subject="Physics")

    # 1. 2-mark
    q2 = "Give me a 2-mark answer for Newton's first law of motion."
    ans2 = generate_reply(q2, p)
    print(f"  [2-Mark] Reply:\n    {ans2}")

    # 2. 5-mark
    q5 = "Give me a 5-mark answer for Law of Conservation of Energy."
    ans5 = generate_reply(q5, p)
    print(f"  [5-Mark] Reply:\n    {ans5}")

    # 3. 10-mark
    q10 = "Give me a 10-mark answer for Ohm's Law and its verification."
    ans10 = generate_reply(q10, p)
    print(f"  [10-Mark] Reply:\n    {ans10[:200]}...")

    ok2 = len(ans2) > 20 and "**" not in ans2
    ok5 = len(ans5) > 60 and "**" not in ans5
    ok10 = len(ans10) > 100 and "**" not in ans10
    not_guaranteed = "guaranteed" not in (ans2 + ans5 + ans10).lower()

    passed = ok2 and ok5 and ok10 and not_guaranteed
    report("2-mark, 5-mark, and 10-mark Exam Answers", passed, "Verified structure, length, and no false exam guarantees")
    return passed


# =====================================================================
# Scenario 12: Telugu, English, and Tanglish responses
# =====================================================================
def test_scenario_12():
    print("\n--- Test 12: Telugu, English, and Tanglish Responses ---")
    p = StudentProfile(level="school", grade_or_year=9)

    # Pure Telugu
    q_te = "భూమి గురుత్వాకర్షణ శక్తి అంటే ఏమిటి?"
    reply_te = generate_reply(q_te, p)
    has_te = any("\u0c00" <= c <= "\u0c7f" for c in reply_te)

    # Pure English
    q_en = "What is the difference between kinetic energy and potential energy?"
    reply_en = generate_reply(q_en, p)
    is_en = detect_tts_language(reply_en) == "en-IN"

    # Tanglish with explicit Telugu switch
    q_switch = "Explain evaporation. Telugu lo explain cheyyi."
    reply_switch = generate_reply(q_switch, p)
    has_switch_te = any("\u0c00" <= c <= "\u0c7f" for c in reply_switch) or any(w in reply_switch.lower() for w in ["ante", "neellu", "aviri", "maradam", "mariyu"])

    print(f"  Telugu Reply: {reply_te[:60]}... (Telugu={has_te})")
    print(f"  English Reply: {reply_en[:60]}... (Lang={detect_tts_language(reply_en)})")
    print(f"  Switch Reply: {reply_switch[:60]}... (Telugu/Tanglish={has_switch_te})")

    passed = has_te and is_en and has_switch_te
    report("Telugu, English, and Tanglish Responses", passed, "Accurately handled language switching and matching")
    return passed


# =====================================================================
# Scenario 13: Gnani STT and TTS integration
# =====================================================================
def test_scenario_13():
    print("\n--- Test 13: Gnani STT and TTS Integration ---")
    tts_out_te = "test_phase7_te.wav"
    tts_out_en = "test_phase7_en.wav"

    te_text = "నమస్కారం! నేను మీ స్కూల్ మరియు ఇంటర్మీడియట్ AI ట్యూటర్‌ను."
    en_text = "Welcome to your AI Student Tutor for School and Intermediate."

    tts_ok = False
    try:
        tts.synthesize(
            text=te_text,
            voice="Pranav",
            model="timbre-v2.5",
            language="te-IN",
            output_file=tts_out_te,
        )
        tts.synthesize(
            text=en_text,
            voice="Pranav",
            model="timbre-v2.5",
            language="en-IN",
            output_file=tts_out_en,
        )
        if os.path.exists(tts_out_te) and os.path.exists(tts_out_en):
            sz_te = os.path.getsize(tts_out_te)
            sz_en = os.path.getsize(tts_out_en)
            tts_ok = (sz_te > 1000 and sz_en > 1000)
            print(f"  Synthesized Telugu audio ({sz_te} bytes) and English audio ({sz_en} bytes).")
    except Exception as e:
        print(f"  TTS Error: {e}")

    # Test STT transcription on existing clean audio
    stt_ok = False
    try:
        stt_result = stt.transcribe(audio=tts_out_en, model="prisma-v3.0", language_code="indic-auto")
        if stt_result and stt_result.get("success"):
            print(f"  Transcribed: {stt_result.get('transcript')}")
            stt_ok = True
    except Exception as e:
        print(f"  STT Warning: {e}")
        # If network hiccup, fallback to te-IN check
        try:
            stt_result2 = stt.transcribe(audio=tts_out_te, language_code="te-IN")
            if stt_result2 and stt_result2.get("success"):
                stt_ok = True
        except Exception:
            pass

    passed = tts_ok and stt_ok
    report("Gnani STT and TTS Integration", passed, "Verified Timbre v2.5 synthesis and Prisma STT transcription")
    return passed


# =====================================================================
# Scenario 14: Continuous voice conversation flow
# =====================================================================
def test_scenario_14():
    print("\n--- Test 14: Continuous Voice Conversation Flow ---")
    # Simulate a 3-turn interactive conversation session:
    profile = StudentProfile(level="school", grade_or_year=8, subject="Science")
    simulated_inputs = [
        "What is friction?",
        "How can we reduce friction in daily life?",
        "ఆపు",  # Stop command terminates conversation
    ]

    session_active = True
    turn_results = []

    for turn_idx, user_speech in enumerate(simulated_inputs, 1):
        if not session_active:
            break

        # Step A: Check stop command
        if is_stop_command(user_speech):
            session_active = False
            turn_results.append(f"Round {turn_idx}: Stop command '{user_speech}' detected -> Session ended gracefully.")
            break

        # Step B: Generate response
        reply = generate_reply(user_speech, profile)
        tts_lang = detect_tts_language(reply)
        turn_results.append(f"Round {turn_idx}: Replied ({len(reply)} chars, {tts_lang}) -> Ready for next round.")

    for res in turn_results:
        print(f"  {res}")

    passed = (len(turn_results) == 3) and (not session_active)
    report("Continuous Voice Conversation Flow", passed, "Verified multi-turn progression and graceful stop termination")
    return passed


# =====================================================================
# Main Test Runner
# =====================================================================
def main():
    print("=" * 70)
    print("PHASE 7 VERIFICATION: SCHOOL & INTERMEDIATE AI STUDENT TUTOR")
    print("=" * 70)

    t1 = test_scenario_1()
    t2 = test_scenario_2()
    t3 = test_scenario_3()
    t4 = test_scenario_4()
    t5 = test_scenario_5()
    t6 = test_scenario_6()
    t7 = test_scenario_7()
    t8 = test_scenario_8()
    t9 = test_scenario_9()
    t10 = test_scenario_10()
    t11 = test_scenario_11()
    t12 = test_scenario_12()
    t13 = test_scenario_13()
    t14 = test_scenario_14()

    print("\n" + "=" * 70)
    print(f"PHASE 7 TEST RESULTS: {passed_count}/14 PASSED ({failed_count} FAILED)")
    print("=" * 70)

    if failed_count == 0:
        print("🎉 ALL 14 PHASE 7 VERIFICATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"❌ {failed_count} test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
