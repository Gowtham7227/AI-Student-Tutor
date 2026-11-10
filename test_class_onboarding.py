import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from voice_assistant import (
    StudentProfile,
    parse_education_level,
    extract_school_class,
    is_filler_or_noise,
    build_system_instruction,
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
# Test 1: Student says "Class 8"
# =====================================================================
def test_student_says_class_8():
    print("\n--- Test 1: Student says 'Class 8' ---")
    # Case A: Turn 1 is "School", Turn 2 is "Class 8"
    p1 = StudentProfile()
    done1, msg1 = parse_education_level("School", p1)
    assert p1.level == "school" and not done1, f"Expected school level, got {p1.level}"

    done2, msg2 = parse_education_level("Class 8", p1)
    passed_a = done2 and (p1.grade_or_year == 8) and ("Class 8 School student" in msg2)

    # Case B: Student says "Class 8" directly at first prompt
    p2 = StudentProfile()
    done_b, msg_b = parse_education_level("Class 8", p2)
    passed_b = done_b and (p2.level == "school") and (p2.grade_or_year == 8)

    passed = passed_a and passed_b
    report("Student says 'Class 8'", passed, f"Turn 2: Grade={p1.grade_or_year}, 1-shot: Grade={p2.grade_or_year}")
    return passed


# =====================================================================
# Test 2: Student says "I am studying 8th class"
# =====================================================================
def test_student_says_studying_8th_class():
    print("\n--- Test 2: Student says 'I am studying 8th class' ---")
    p = StudentProfile(level="school")
    done, msg = parse_education_level("I am studying 8th class", p)
    passed = done and (p.grade_or_year == 8) and p.is_complete()
    report("Student says 'I am studying 8th class'", passed, f"Grade={p.grade_or_year}, Complete={p.is_complete()}")
    return passed


# =====================================================================
# Test 3: Telugu class variations including "నేను 8వ తరగతి చదువుతున్నాను"
# =====================================================================
def test_student_says_telugu_variations():
    print("\n--- Test 3: Telugu Class Variations ---")
    telugu_samples = [
        ("నేను 8వ తరగతి చదువుతున్నాను", 8),
        ("ఎనిమిదో తరగతి", 8),
        ("ఎనిమిదవ తరగతి", 8),
        ("8వ తరగతి", 8),
        ("క్లాస్ 8", 8),
        ("పదవ తరగతి", 10),
        ("ఒకటో తరగతి", 1),
    ]

    all_passed = True
    for phrase, expected_grade in telugu_samples:
        p = StudentProfile(level="school")
        done, msg = parse_education_level(phrase, p)
        ok = done and (p.grade_or_year == expected_grade) and p.is_complete()
        if not ok:
            print(f"    Failed on '{phrase}': got Grade={p.grade_or_year}, Complete={done}")
            all_passed = False

    report("Telugu Class Variations ('నేను 8వ తరగతి చదువుతున్నాను', etc.)", all_passed, f"Tested {len(telugu_samples)} Telugu phrases")
    return all_passed


# =====================================================================
# Test 4: Student answers in Tanglish
# =====================================================================
def test_student_says_tanglish_variations():
    print("\n--- Test 4: Tanglish Variations ---")
    tanglish_samples = [
        ("nenu 8th class chaduvutunnanu", 8),
        ("8th class andi", 8),
        ("8va class", 8),
        ("class 8 lo chaduvutunna", 8),
        ("8th standard andi", 8),
        ("nenu 8th lo unnanu", 8),
    ]

    all_passed = True
    for phrase, expected_grade in tanglish_samples:
        p = StudentProfile(level="school")
        done, msg = parse_education_level(phrase, p)
        ok = done and (p.grade_or_year == expected_grade) and p.is_complete()
        if not ok:
            print(f"    Failed on '{phrase}': got Grade={p.grade_or_year}, Complete={done}")
            all_passed = False

    report("Tanglish Class Variations", all_passed, f"Tested {len(tanglish_samples)} Tanglish phrases")
    return all_passed


# =====================================================================
# Test 5: Profile persistence across subsequent conversation turns
# =====================================================================
def test_profile_persistence():
    print("\n--- Test 5: Profile Persistence Across Multiple Turns ---")
    # Simulate multi-turn session
    profile = StudentProfile()

    # Turn 1: Onboarding question 1
    done1, msg1 = parse_education_level("School student", profile)
    assert not done1 and profile.level == "school"

    # Turn 2: Student provides class
    done2, msg2 = parse_education_level("Class 8", profile)
    assert done2 and profile.grade_or_year == 8 and profile.is_complete()

    # Turns 3, 4, 5, 6: Conversation turns
    turns = [
        "What is photosynthesis in science?",
        "Can you explain fractions with a simple example?",
        "Telugu lo explain cheyyi",
        "Give me a 2-mark answer for cell definition",
    ]

    persisted = True
    for t_idx, user_query in enumerate(turns, start=3):
        # In main continuous loop, profile is used to build instructions
        instructions = build_system_instruction(profile)
        # Verify profile fields are untouched
        if profile.level != "school" or profile.grade_or_year != 8 or not profile.is_complete():
            persisted = False
            print(f"    Profile mutated in turn {t_idx}!")
            break
        if "Class 8 School student" not in instructions:
            persisted = False
            print(f"    Instruction missing profile in turn {t_idx}!")
            break

    report("Profile Persistence Across Turns", persisted, f"Verified across {len(turns)} turns without mutation")
    return persisted


# =====================================================================
# Test 6: Assistant does NOT repeat class question after saving class
# =====================================================================
def test_does_not_repeat_class_question():
    print("\n--- Test 6: No Repetition of Class Question ---")
    profile = StudentProfile(level="school", grade_or_year=8)

    # Calling parse_education_level when profile is already complete
    done, msg = parse_education_level("I want to learn science", profile)

    # Must be done and must NOT ask "Which class are you studying in?"
    question_repeated = "which class are you studying" in msg.lower() or "please say class 1" in msg.lower()
    passed = done and (not question_repeated) and ("registered as a Class 8 School student" in msg)
    report("No Repetition of Class Question", passed, f"Prompt returned: '{msg}'")
    return passed


# =====================================================================
# Test 7: Noise vs Valid Single Digit & Unclear Fallback
# =====================================================================
def test_noise_and_clarification():
    print("\n--- Test 7: Noise Filtering and Polite Clarification ---")
    # 1. Single digit "8" or "8." must NOT be flagged as noise
    not_noise_samples = ["8", "8.", "8th", "Class 8", "ఎనిమిదో తరగతి"]
    noise_samples = ["um", "uh", "hmm", ".", "-", "  "]

    clean_ok = all(not is_filler_or_noise(s) for s in not_noise_samples)
    noise_ok = all(is_filler_or_noise(s) for s in noise_samples)

    # 2. When student input is genuinely unclear in school mode, ask politely without repeating "Great! Which class..."
    p = StudentProfile(level="school")
    done, msg = parse_education_level("random unclear words", p)
    polite_ask = (not done) and ("catch your class clearly" in msg.lower())

    passed = clean_ok and noise_ok and polite_ask
    report("Noise Filtering & Clarification", passed, f"Clean OK={clean_ok}, Noise OK={noise_ok}, Polite Clarify={polite_ask}")
    return passed


def main():
    print("=" * 70)
    print("🧪  TEST SUITE: STUDENT CLASS ONBOARDING REPETITION FIX")
    print("=" * 70)

    test_student_says_class_8()
    test_student_says_studying_8th_class()
    test_student_says_telugu_variations()
    test_student_says_tanglish_variations()
    test_profile_persistence()
    test_does_not_repeat_class_question()
    test_noise_and_clarification()

    print("\n" + "=" * 70)
    print(f"📊 SUMMARY: Total Passed: {passed_count} | Total Failed: {failed_count}")
    print("=" * 70)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
