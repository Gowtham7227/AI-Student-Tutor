"""
Test Suite: Phase 11.5 Live Tutor Intelligence & Conversational Quality Improvements
Covers:
  1. Context-aware subject handling (temporary turn context without profile mutation, ambiguous question guard)
  2. Concise educational answers (2-3 sentences default, at most ONE follow-up, preservation of 2/5/10 mark & detail)
  3. Unclear speech handling (cutoffs, bare starters vs preservation of valid short answers & contextual doubts)
  4. Student-guided problem solving (direct answer vs Socratic practice, assumption handling, missing values guard)
  5. Onboarding reliability (out-of-scope rejection, ambiguous selection, complete profile preservation)
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
    detect_question_subject,
    detect_subject_switch,
    is_incomplete_speech,
    parse_education_level,
    generate_reply,
    build_system_instruction,
    VALID_SHORT_ANSWERS,
)


class TestContextAwareSubjectHandling(unittest.TestCase):
    """Objective 1: Inferred question subject, temporary turn context, profile preservation."""

    def setUp(self):
        self.school_profile = StudentProfile(level="school", grade_or_year=8, subject="Mathematics")
        self.session = LearningSession(self.school_profile)
        self.session.active_subject = "Mathematics"
        self.session.profile.subject = "Mathematics"

        self.inter_profile = StudentProfile(level="intermediate", grade_or_year=1, group="BiPC", subject="Physics")
        self.inter_session = LearningSession(self.inter_profile)
        self.inter_session.active_subject = "Physics"
        self.inter_session.profile.subject = "Physics"

    def test_01_science_question_in_math_session_uses_turn_context(self):
        """A Science question in a Math session uses temporary Science context without changing profile."""
        query = "What is photosynthesis?"
        inferred = detect_question_subject(query, self.school_profile)
        self.assertEqual(inferred, "Science")

        with patch("voice_assistant.call_gemini", return_value="Photosynthesis is the process plants use to make food.") as mock_gem:
            reply = generate_reply(query, self.school_profile, self.session)
            # 1. Profile and permanent session active subject remain Mathematics
            self.assertEqual(self.session.active_subject, "Mathematics")
            self.assertEqual(self.school_profile.subject, "Mathematics")
            # 2. System instruction received Science as active turn subject
            called_sys_inst = mock_gem.call_args[0][1]
            self.assertIn("- Current Subject: Science", called_sys_inst)
            # 3. Prompt instruction specifies turn subject context
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("TURN SUBJECT CONTEXT", called_prompt)
            self.assertIn("Science", called_prompt)

    def test_02_botany_question_in_physics_session_bipc(self):
        """In BiPC, a plant question during Physics session infers Botany without altering active subject."""
        query = "Explain the differences between gymnosperms and angiosperms"
        inferred = detect_question_subject(query, self.inter_profile)
        self.assertEqual(inferred, "Botany")

        with patch("voice_assistant.call_gemini", return_value="Gymnosperms have naked seeds whereas angiosperms have enclosed seeds.") as mock_gem:
            reply = generate_reply(query, self.inter_profile, self.inter_session)
            self.assertEqual(self.inter_session.active_subject, "Physics")
            self.assertEqual(self.inter_profile.subject, "Physics")
            called_sys_inst = mock_gem.call_args[0][1]
            self.assertIn("- Current Subject: Botany", called_sys_inst)

    def test_03_safeguard1_ambiguous_question_does_not_force_subject(self):
        """Safeguard 1: Ambiguous questions do not force subject inference."""
        ambiguous_queries = [
            "Can you explain this concept?",
            "What is an experiment?",
            "Tell me about this topic",
            "Why does this change occur?",
        ]
        for query in ambiguous_queries:
            inferred = detect_question_subject(query, self.school_profile)
            self.assertIsNone(inferred, f"Should return None for ambiguous query: '{query}'")


class TestConciseEducationalAnswers(unittest.TestCase):
    """Objective 2: 2-3 sentences default, single follow-up, preservation of detail & exam formats."""

    def test_04_default_conciseness_prompt_injection(self):
        """Normal doubt mode injects strict 2-3 sentence and at most ONE follow-up constraint."""
        p = StudentProfile(level="school", grade_or_year=8)
        s = LearningSession(p)
        query = "What is friction?"

        with patch("voice_assistant.call_gemini", return_value="Friction is a force that opposes motion.") as mock_gem:
            generate_reply(query, p, s)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("DEFAULT CONCISENESS", called_prompt)
            self.assertIn("2 to 3 short, clear sentences", called_prompt)
            self.assertIn("at most ONE brief, friendly check question", called_prompt)

    def test_05_detailed_and_exam_requests_preserved(self):
        """Explicit detail and 2/5/10-mark requests do NOT inject normal 2-3 sentence constraint."""
        p = StudentProfile(level="intermediate", grade_or_year=1, group="MPC", subject="Physics")
        s = LearningSession(p)

        # 5-mark request
        with patch("voice_assistant.call_gemini", return_value="Structured 5-mark answer") as mock_gem:
            generate_reply("Give me a 5-mark answer for Newton's laws", p, s)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("5-MARK", called_prompt)
            self.assertNotIn("DEFAULT CONCISENESS", called_prompt)

        # Explicit detail request
        with patch("voice_assistant.call_gemini", return_value="Comprehensive detailed breakdown") as mock_gem:
            generate_reply("Explain in detail how transformers work", p, s)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("DETAILED explanation", called_prompt)
            self.assertNotIn("DEFAULT CONCISENESS", called_prompt)


class TestUnclearSpeechHandling(unittest.TestCase):
    """Objective 3 & Safeguard 2: Unclear speech, trailing cutoffs, valid short answers."""

    def test_06_trailing_cutoffs_and_bare_starters_detected(self):
        """Dangling prepositions/conjunctions and bare question starters are detected as incomplete."""
        incomplete_queries = [
            "because of",
            "and then the",
            "what is",
            "can you explain",
            "tell me",
            "చెప్పు",
            "కానీ",
            "యొక్క",
        ]
        for query in incomplete_queries:
            self.assertTrue(is_incomplete_speech(query), f"Failed to detect incomplete speech for: '{query}'")

    def test_07_safeguard2_preserve_valid_short_answers_and_mcq(self):
        """Safeguard 2: Digits, single-letter MCQ options, scientific terms, and affirmations are preserved."""
        valid_samples = [
            "A", "B", "C", "D", "a", "b", "c", "d",
            "ఏ", "బి", "సి", "డి",
            "8", "10", "42",
            "Oxygen", "Photosynthesis", "Cell", "బలం", "కణం",
            "yes", "no", "అవును", "కాదు", "సరే",
        ]
        for sample in valid_samples:
            self.assertFalse(is_incomplete_speech(sample), f"Valid short answer was wrongly flagged: '{sample}'")

    def test_08_safeguard2_preserve_contextual_doubt_interrogatives(self):
        """Safeguard 2: Short interrogatives ('Why?', 'ఎందుకు?') are valid when active context exists."""
        p = StudentProfile(level="school", grade_or_year=8)
        s = LearningSession(p)
        s.active_topic = "Photosynthesis"
        s.add_turn("student", "What is photosynthesis?")
        s.add_turn("tutor", "Photosynthesis is the process by which plants make food.")

        self.assertFalse(is_incomplete_speech("Why?", s))
        self.assertFalse(is_incomplete_speech("ఎందుకు?", s))
        self.assertFalse(is_incomplete_speech("How?", s))

        # Without session history, standalone 'Why?' is incomplete
        empty_session = LearningSession(p)
        self.assertTrue(is_incomplete_speech("Why?", empty_session))


class TestStudentGuidedProblemSolving(unittest.TestCase):
    """Objective 4 & Safeguard 3: Direct answer vs Socratic practice, assumptions, missing values."""

    def test_09_safeguard3_direct_answer_request(self):
        """Safeguard 3: Explicit direct answer requests trigger direct answer instructions."""
        p = StudentProfile(level="school", grade_or_year=8)
        s = LearningSession(p)

        queries = [
            "Give me the direct answer for what is the value of pi",
            "Just tell me the answer",
            "డైరెక్ట్ గా చెప్పు",
            "జవాబు చెప్పు",
        ]
        for query in queries:
            with patch("voice_assistant.call_gemini", return_value="The value is 3.14159.") as mock_gem:
                generate_reply(query, p, s)
                called_prompt = mock_gem.call_args[0][0]
                self.assertIn("DIRECT ANSWER REQUEST", called_prompt, f"Failed for query: '{query}'")

    def test_10_safeguard3_practice_mode_socratic_guidance(self):
        """Safeguard 3: Practice mode uses Socratic guidance and does not reveal final answer prematurely."""
        p = StudentProfile(level="school", grade_or_year=8)
        s = LearningSession(p)
        query = "Help me solve: 2x + 5 = 15"

        with patch("voice_assistant.call_gemini", return_value="Step 1: subtract 5 from both sides.") as mock_gem:
            generate_reply(query, p, s)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("PRACTICE MODE & SOCRATIC GUIDANCE", called_prompt)
            self.assertIn("Do NOT reveal the final numerical answer immediately", called_prompt)

    def test_11_assumption_and_missing_values_instructions(self):
        """Proposing an assumption asks which value to use; calculation adds missing values guard."""
        p = StudentProfile(level="school", grade_or_year=8)
        s = LearningSession(p)

        # Propose assumption
        with patch("voice_assistant.call_gemini", return_value="Sure! What value would you like to take?") as mock_gem:
            generate_reply("Can we assume a value for radius?", p, s)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("ASSUMPTION REQUEST", called_prompt)
            self.assertIn("ask which specific value they want to use", called_prompt)

        # Missing values guard in calculation
        with patch("voice_assistant.call_gemini", return_value="Formula is A = pi * r^2.") as mock_gem:
            generate_reply("Calculate the area of a circle", p, s)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("MISSING VALUES GUARD", called_prompt)
            self.assertIn("Never invent missing problem values without clearly stating they are an example", called_prompt)


class TestOnboardingReliability(unittest.TestCase):
    """Objective 5 & Safeguard 4: Out-of-scope rejection, ambiguous level, complete profile protection."""

    def test_12_safeguard4_out_of_scope_levels_rejected(self):
        """Safeguard 4: College, degree, B.Tech are rejected politely without setting profile level."""
        out_of_scope = ["I am in college", "Degree 2nd year", "B.Tech first year", "engineering"]
        for sample in out_of_scope:
            p = StudentProfile()
            done, msg = parse_education_level(sample, p)
            self.assertFalse(done)
            self.assertIsNone(p.level, f"Level should not be set for out-of-scope: '{sample}'")
            self.assertIn("exclusively for School", msg)

    def test_13_safeguard4_ambiguous_level_selection_clarified(self):
        """Safeguard 4: Mentioning both school and intermediate triggers clarification without setting level."""
        ambiguous = ["School or intermediate", "both school and inter"]
        for sample in ambiguous:
            p = StudentProfile()
            done, msg = parse_education_level(sample, p)
            self.assertFalse(done)
            self.assertIsNone(p.level, f"Level should not be set for ambiguous: '{sample}'")
            self.assertIn("Please clarify whether you are studying in School", msg)

    def test_14_safeguard4_never_overwrite_complete_profile(self):
        """Safeguard 4: Complete profile is preserved and never overwritten by out-of-scope utterances."""
        p = StudentProfile(level="intermediate", grade_or_year=2, group="MPC")
        self.assertTrue(p.is_complete())

        done, msg = parse_education_level("I am joining college B.Tech next", p)
        self.assertTrue(done)
        self.assertEqual(p.level, "intermediate")
        self.assertEqual(p.grade_or_year, 2)
        self.assertEqual(p.group, "MPC")
        self.assertIn("already registered", msg)


class TestPhase11_5_1_TargetedRefinements(unittest.TestCase):
    """Phase 11.5.1: Assumption handling distinction, conciseness, Socratic & direct preservation."""

    def setUp(self):
        self.school_profile = StudentProfile(level="school", grade_or_year=8, subject="Mathematics")
        self.session = LearningSession(self.school_profile)

    def test_15_student_proposes_assumption_asks_which_value(self):
        """When student asks if they can assume a value, first ask which value they want to choose."""
        student_queries = [
            "Can we assume a value?",
            "Can I assume a value for the radius?",
            "Let's assume a value",
            "Shall we assume a value?",
            "మనం ఒక విలువ అనుకుందామా?",
            "విలువ అనుకోవచ్చా?",
        ]
        for query in student_queries:
            with patch("voice_assistant.call_gemini", return_value="Sure! What value would you like to use?") as mock_gem:
                generate_reply(query, self.school_profile, self.session)
                called_prompt = mock_gem.call_args[0][0]
                self.assertIn("ASSUMPTION REQUEST", called_prompt, f"Failed for query: '{query}'")
                self.assertIn("ask which specific value they want to use", called_prompt, f"Failed for query: '{query}'")
                self.assertNotIn("TUTOR-CHOSEN EXAMPLE VALUES", called_prompt, f"Should not trigger tutor-chosen for: '{query}'")

    def test_16_student_asks_tutor_to_choose_example(self):
        """When student explicitly asks tutor to choose an example, provide clearly labeled illustrative values."""
        tutor_choose_queries = [
            "You choose an example",
            "You choose a value for me",
            "Give me an example value",
            "Can you choose an example?",
            "Tutor please choose a value",
            "You take a value and solve",
            "నువ్వే ఒక విలువ తీసుకో",
            "మీరే ఒక విలువ తీసుకోండి",
            "నువ్వే ఎగ్జాంపుల్ తీసుకో",
            "ఒక ఉదాహరణ విలువ చెప్పు",
        ]
        for query in tutor_choose_queries:
            with patch("voice_assistant.call_gemini", return_value="For example, let us take radius r = 7 cm.") as mock_gem:
                generate_reply(query, self.school_profile, self.session)
                called_prompt = mock_gem.call_args[0][0]
                self.assertIn("TUTOR-CHOSEN EXAMPLE VALUES", called_prompt, f"Failed for query: '{query}'")
                self.assertIn("Provide clearly labeled illustrative example values", called_prompt, f"Failed for query: '{query}'")
                self.assertNotIn("ASSUMPTION REQUEST", called_prompt, f"Should not trigger student assumption for: '{query}'")

    def test_17_direct_answer_overrides_assumption_and_socratic(self):
        """Explicit direct answer requests override assumption and Socratic modes."""
        queries = [
            "Just tell me the answer directly",
            "Give me the answer for radius 7",
            "డైరెక్ట్ గా చెప్పు",
            "జవాబు చెప్పు",
        ]
        for query in queries:
            with patch("voice_assistant.call_gemini", return_value="The area is 154 cm^2.") as mock_gem:
                generate_reply(query, self.school_profile, self.session)
                called_prompt = mock_gem.call_args[0][0]
                self.assertIn("DIRECT ANSWER REQUEST", called_prompt, f"Failed for query: '{query}'")
                self.assertNotIn("TUTOR-CHOSEN EXAMPLE VALUES", called_prompt)

    def test_18_socratic_practice_mode_preserved(self):
        """Socratic practice mode is preserved when problem solving is requested without assumptions."""
        query = "Help me solve this algebra problem"
        with patch("voice_assistant.call_gemini", return_value="Let's state the formula.") as mock_gem:
            generate_reply(query, self.school_profile, self.session)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("PRACTICE MODE & SOCRATIC GUIDANCE", called_prompt)
            self.assertNotIn("TUTOR-CHOSEN EXAMPLE VALUES", called_prompt)

    def test_19_conciseness_no_repetition_and_detail_preservation(self):
        """Normal doubt prompt enforces zero echoing and no repetitive phrasing; detail mode is preserved."""
        # 1. Normal doubt mode
        with patch("voice_assistant.call_gemini", return_value="Gravity is an attractive force.") as mock_gem:
            generate_reply("What is gravity?", self.school_profile, self.session)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("DEFAULT CONCISENESS", called_prompt)
            self.assertIn("Start directly with the core explanation without echoing", called_prompt)
            self.assertIn("Avoid repeating the same concept in different words", called_prompt)

        # 2. Detailed explanation request preserves detail mode
        with patch("voice_assistant.call_gemini", return_value="Detailed explanation.") as mock_gem:
            generate_reply("Explain gravity in detail", self.school_profile, self.session)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("DETAILED explanation", called_prompt)
            self.assertNotIn("DEFAULT CONCISENESS", called_prompt)

        # 3. 5-mark answer preserves exam rubric
        with patch("voice_assistant.call_gemini", return_value="5 mark structure.") as mock_gem:
            generate_reply("Give 5-mark answer for Newton's laws", self.school_profile, self.session)
            called_prompt = mock_gem.call_args[0][0]
            self.assertIn("5-MARK exam answer", called_prompt)
            self.assertNotIn("DEFAULT CONCISENESS", called_prompt)


if __name__ == "__main__":
    unittest.main()
