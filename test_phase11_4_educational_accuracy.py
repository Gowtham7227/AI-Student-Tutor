"""
Test Suite: Phase 11.4 Educational Accuracy and Voice Response Optimization
Tests:
  1. BiPC pronunciation variants normalization (BPC, BIPAC, bi p c, b.p.c, బైపిసి, etc.)
  2. MPC pronunciation variants normalization (MPC, m p c, m.p.c, ఎంపిసి)
  3. Standalone subject words alone (biology, botany, zoology) require clarification, not silent BiPC
  4. Unsupported groups (CEC, MEC, HEC, commerce, arts) rejected gracefully
  5. Ambiguous or unknown groups trigger clarification and are never silently saved
  6. Complete profile persistence without overwrite
  7. Cross-subject word problems/questions with science terms do NOT switch subject
  8. Explicit subject switches succeed while preserving profile level/year/group
  9. Markdown bullet removal preserves mathematical minus signs in formulas and coordinates
 10. Circle formulas and mathematical rigor instructions present in system prompt
 11. Concise normal answers (2-3 sentences), at most ONE follow-up, preserving exam/detail formats
"""

import sys
import unittest
from unittest.mock import patch

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from voice_assistant import (
    StudentProfile,
    LearningSession,
    parse_intermediate_group,
    parse_education_level,
    detect_subject_switch,
    classify_learning_intent,
    generate_reply,
    clean_text_for_speech,
    build_system_instruction,
    INTENT_SWITCH_TOPIC,
)


class TestIntermediateGroupParsing(unittest.TestCase):
    """Test 1 - 5: BiPC / MPC pronunciation variants, standalone subjects, unsupported groups."""

    def test_01_bipc_variants_normalization(self):
        """Test that common STT variants for BiPC normalize to 'BiPC'."""
        variants = [
            "bipc", "bpc", "BIPAC", "bi p c", "b.p.c", "b p c",
            "bi-pc", "b-p-c", "బైపీసీ", "బైపిసి", "బిపిసి", "బైపాక్", "బైపాసి"
        ]
        for variant in variants:
            group, err = parse_intermediate_group(variant)
            self.assertEqual(group, "BiPC", f"Failed to normalize BiPC variant: '{variant}'")
            self.assertIsNone(err, f"Unexpected clarification error for: '{variant}'")

    def test_02_mpc_variants_normalization(self):
        """Test that common STT variants for MPC normalize to 'MPC'."""
        variants = [
            "mpc", "m p c", "m.p.c", "m-p-c", "ఎంపీసీ", "ఎంపిసి"
        ]
        for variant in variants:
            group, err = parse_intermediate_group(variant)
            self.assertEqual(group, "MPC", f"Failed to normalize MPC variant: '{variant}'")
            self.assertIsNone(err, f"Unexpected clarification error for: '{variant}'")

    def test_03_standalone_subject_words_alone_require_confirmation(self):
        """Biology, botany, zoology, or maths alone must NOT confirm group without clarification."""
        subject_samples = ["biology", "botany", "zoology", "బయాలజీ", "బాటనీ", "జువాలజీ"]
        for sample in subject_samples:
            group, err = parse_intermediate_group(sample)
            self.assertIsNone(group, f"Group should not be confirmed for standalone subject: '{sample}'")
            self.assertIsNotNone(err, f"Clarification should be asked for: '{sample}'")
            self.assertIn("BiPC", err)

        math_samples = ["maths", "mathematics", "గణితం"]
        for sample in math_samples:
            group, err = parse_intermediate_group(sample)
            self.assertIsNone(group, f"Group should not be confirmed for standalone subject: '{sample}'")
            self.assertIsNotNone(err, f"Clarification should be asked for: '{sample}'")
            self.assertIn("MPC", err)

    def test_04_unsupported_groups_rejected(self):
        """Unsupported intermediate groups like CEC, MEC, HEC, arts, commerce must be flagged politely."""
        unsupported = ["CEC", "c.e.c", "MEC", "HEC", "commerce", "arts"]
        for group_name in unsupported:
            group, err = parse_intermediate_group(group_name)
            self.assertIsNone(group, f"Unsupported group '{group_name}' should not be accepted")
            self.assertIsNotNone(err, f"Clarification required for unsupported group: '{group_name}'")
            self.assertIn("MPC and BiPC", err)

    def test_05_ambiguous_and_unknown_groups(self):
        """Ambiguous (e.g. 'both mpc and bipc') or unknown queries must ask for clarification."""
        group, err = parse_intermediate_group("both mpc and bipc")
        self.assertIsNone(group)
        self.assertIsNotNone(err)

        # Onboarding flow test: student in Intermediate says ambiguous group
        p = StudentProfile(level="intermediate", grade_or_year=1)
        done, msg = parse_education_level("science", p)
        self.assertFalse(done)
        self.assertIsNone(p.group, "Group must not be silently assigned for 'science'")
        self.assertIn("Which group", msg)

    def test_06_bipc_full_onboarding_turn(self):
        """Multi-turn onboarding: Intermediate 1st year -> student says 'BPC' -> completes BiPC."""
        p = StudentProfile()
        done1, _ = parse_education_level("Intermediate", p)
        self.assertFalse(done1)
        done2, _ = parse_education_level("1st year", p)
        self.assertFalse(done2)
        done3, msg3 = parse_education_level("BPC", p)
        self.assertTrue(done3)
        self.assertEqual(p.level, "intermediate")
        self.assertEqual(p.grade_or_year, 1)
        self.assertEqual(p.group, "BiPC")
        self.assertIn("BiPC", msg3)


class TestSubjectContextAndCrossSubjectQueries(unittest.TestCase):
    """Test 7 - 8: Cross-subject word problems vs explicit subject switch."""

    def setUp(self):
        self.school_profile = StudentProfile(level="school", grade_or_year=8)
        self.session = LearningSession(self.school_profile)
        self.session.active_subject = "Mathematics"
        self.session.profile.subject = "Mathematics"
        self.session.active_topic = "Linear Equations"

        self.inter_bipc_profile = StudentProfile(level="intermediate", grade_or_year=1, group="BiPC")
        self.inter_session = LearningSession(self.inter_bipc_profile)
        self.inter_session.active_subject = "Botany"
        self.inter_session.profile.subject = "Botany"
        self.inter_session.active_topic = "Plant Morphology"

    def test_07_math_word_problems_with_science_terms_no_switch(self):
        """Cross-subject math word problems mentioning physics/science must NOT switch active subject."""
        queries = [
            "If a car travels at a speed of 60 km/h in physics, calculate the distance in math",
            "Calculate the force if mass is 5kg and acceleration is 2 m/s^2",
            "What is the speed of light in physics?",
            "ఫిజిక్స్ లో ఈ లెక్క చేయి: ఒక కారు వేగం 50 km/h",
            "కెమిస్ట్రీ లో వాటర్ ఫార్ములా ఏమిటి?",
            "Teach me how to calculate kinetic energy",
            "How does math help in science?",
            "What is chemical change?",
            "In physics, what is Newton's second law?",
        ]
        for query in queries:
            is_switch, target, _ = detect_subject_switch(query, self.school_profile)
            self.assertFalse(is_switch, f"Should NOT switch subject for problem/question: '{query}'")
            self.assertIsNone(target)

            # Intent should not be switch topic
            intent = classify_learning_intent(query, self.session)
            self.assertNotEqual(intent, INTENT_SWITCH_TOPIC, f"Intent should not be INTENT_SWITCH_TOPIC for: '{query}'")

    def test_08_explicit_subject_switch_preserves_profile(self):
        """Explicit subject switch changes active subject while preserving student profile."""
        queries = [
            ("Switch subject to Physics", "Physics"),
            ("Let's study Chemistry", "Chemistry"),
            ("బాటనీ చదువుదాం", "Botany"),
            ("బాటనీకి మార్చు", "Botany"),
        ]
        for query, expected_subj in queries:
            is_switch, target, _ = detect_subject_switch(query, self.inter_bipc_profile)
            self.assertTrue(is_switch, f"Failed to detect explicit switch: '{query}'")
            self.assertEqual(target, expected_subj, f"Target mismatch for: '{query}'")

            with patch("voice_assistant.call_gemini", return_value=f"Welcome to {expected_subj}!"):
                generate_reply(query, self.inter_bipc_profile, self.inter_session)
                self.assertEqual(self.inter_session.active_subject, expected_subj)
                # Verify student profile integrity: level, year, and group are preserved
                self.assertEqual(self.inter_bipc_profile.level, "intermediate")
                self.assertEqual(self.inter_bipc_profile.grade_or_year, 1)
                self.assertEqual(self.inter_bipc_profile.group, "BiPC")


class TestMathematicalAccuracyAndSpeechSanitization(unittest.TestCase):
    """Test 9 - 11: Clean text preserves minus signs, circle equations, prompt constraints."""

    def test_09_clean_text_preserves_mathematical_minus_signs(self):
        """Mathematical negative signs, formula minus signs, and list bullet stripping."""
        # 1. Negative coordinates
        text_coords = "The center of the circle is (-g, -f)."
        cleaned_coords = clean_text_for_speech(text_coords)
        self.assertIn("(-g, -f)", cleaned_coords)

        # 2. Formula minus signs and roots
        text_formula = "Radius is sqrt(g^2 + f^2 - c) where (x - h)^2 + (y - k)^2 = r^2."
        cleaned_formula = clean_text_for_speech(text_formula)
        self.assertIn("g^2 + f^2 - c", cleaned_formula)
        self.assertIn("(x - h)^2", cleaned_formula)
        self.assertIn("(y - k)^2", cleaned_formula)

        # 3. Simple math minus with spaces
        text_calc = "Solve x - 5 = 10"
        cleaned_calc = clean_text_for_speech(text_calc)
        self.assertIn("x - 5 = 10", cleaned_calc)

        # 4. Markdown bullet markers at line start are stripped without touching inline minus
        text_bullets = "- First point: Center is (-3, -4)\n- Second point: x - y = 2"
        cleaned_bullets = clean_text_for_speech(text_bullets)
        self.assertFalse(cleaned_bullets.startswith("- "))
        self.assertIn("(-3, -4)", cleaned_bullets)
        self.assertIn("x - y = 2", cleaned_bullets)

    def test_10_circle_formulas_in_system_instruction(self):
        """System prompt contains verified circle equations and explicit sign instructions."""
        profile = StudentProfile(level="intermediate", grade_or_year=1, group="MPC")
        session = LearningSession(profile)
        session.active_subject = "Mathematics"
        instructions = build_system_instruction(profile, session)

        # Check standard form
        self.assertIn("x^2 + y^2 = r^2", instructions)
        # Check center-radius form
        self.assertIn("(x - h)^2 + (y - k)^2 = r^2", instructions)
        # Check general form
        self.assertIn("x^2 + y^2 + 2gx + 2fy + c = 0", instructions)
        # Check center (-g, -f)
        self.assertIn("(-g, -f)", instructions)
        # Check radius sqrt(g^2 + f^2 - c)
        self.assertIn("sqrt(g^2 + f^2 - c)", instructions)
        # Check minus sign verbalization instruction
        self.assertIn("minus g, minus f", instructions)

    def test_11_concise_spoken_answer_rules_in_prompt(self):
        """System prompt enforces 2-3 sentences default, at most ONE follow-up, while preserving exam formats."""
        profile = StudentProfile(level="school", grade_or_year=8)
        instructions = build_system_instruction(profile)

        # Normal doubt conciseness
        self.assertIn("2 to 3 sentences", instructions)
        # Single follow-up question limit
        self.assertIn("AT MOST ONE", instructions)
        # No repetition / echoing
        self.assertIn("Do NOT repeat or echo back", instructions)
        # Exam formats preserved
        self.assertIn("2-mark answer", instructions)
        self.assertIn("5-mark answer", instructions)
        self.assertIn("10-mark answer", instructions)
        self.assertIn("explain in detail", instructions)


if __name__ == "__main__":
    unittest.main()
