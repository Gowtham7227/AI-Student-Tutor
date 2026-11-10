"""
Test Suite for Phase 9: Intelligent Autonomous Student Tutor Agent
Validates:
1. Session Memory & Context-Aware Prompt Construction (StudentProfile + LearningSession + conversation history)
2. All 10 Student Learning Intents (classify_learning_intent)
3. Socratic Problem Solving & Pedagogical Prompt Specialization
4. Session Pause & Resume Flow
5. Subject Switching Flow & Context Reset
6. Adaptive Quiz Engine (Dynamic Difficulty Adjustment, Hint Generation & Tracking)
7. Scope Guardrail Enforcement (Reject B.Tech / Engineering / College Programming)
8. Telemetry & Phase 8 Adaptive VAD Preservation
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from voice_assistant import (
    StudentProfile,
    QuizSession,
    ConversationTurn,
    LearningSession,
    classify_learning_intent,
    build_system_instruction,
    generate_reply,
    generate_quiz_question,
    evaluate_quiz_answer,
    is_college_cse_query,
    is_stop_command,
    is_quiz_trigger,
    AudioRecordResult,
    print_latency_summary,
    INTENT_STOP,
    INTENT_PAUSE,
    INTENT_RESUME,
    INTENT_QUIZ_START,
    INTENT_EXAM_PREP,
    INTENT_REVISION,
    INTENT_STEP_BY_STEP,
    INTENT_HINT,
    INTENT_SWITCH_TOPIC,
    INTENT_DOUBT_FOLLOWUP,
    INTENT_EXPLANATION,
)


class TestPhase9Agent(unittest.TestCase):

    def setUp(self):
        self.profile = StudentProfile(level="school", grade_or_year=8, subject="Science")
        self.session = LearningSession(profile=self.profile)

    # -------------------------------------------------------------
    # Test 1: Session Memory & Context-Aware Dialogue History
    # -------------------------------------------------------------
    def test_session_memory_and_context(self):
        """Verifies that LearningSession tracks history (last 8 turns) and formats context correctly."""
        self.session.add_turn("student", "What is photosynthesis?", intent=INTENT_EXPLANATION)
        self.session.add_turn("tutor", "Photosynthesis is the process by which green plants make food.")
        self.session.add_turn("student", "Where does it happen?", intent=INTENT_DOUBT_FOLLOWUP)

        ctx = self.session.get_conversation_context()
        self.assertIn("Student: What is photosynthesis?", ctx)
        self.assertIn("Tutor: Photosynthesis is the process", ctx)
        self.assertIn("Student: Where does it happen?", ctx)

        # Verify sliding window keeps at most 8 turns
        for i in range(10):
            self.session.add_turn("student", f"Question {i}")
        self.assertLessEqual(len(self.session.history), 8)

    # -------------------------------------------------------------
    # Test 2: Intent Classification for All 10 Student Intents
    # -------------------------------------------------------------
    def test_intent_classification(self):
        """Verifies accurate intent classification across diverse student queries."""
        # 1. Stop
        self.assertEqual(classify_learning_intent("stop", self.session), INTENT_STOP)
        self.assertEqual(classify_learning_intent("ఆపు", self.session), INTENT_STOP)

        # 2. Pause
        self.assertEqual(classify_learning_intent("pause the lesson", self.session), INTENT_PAUSE)
        self.assertEqual(classify_learning_intent("కాసేపు ఆగు", self.session), INTENT_PAUSE)

        # 3. Resume
        self.assertEqual(classify_learning_intent("resume please", self.session), INTENT_RESUME)
        self.assertEqual(classify_learning_intent("continue", self.session), INTENT_RESUME)

        # 4. Quiz trigger
        self.assertEqual(classify_learning_intent("start quiz", self.session), INTENT_QUIZ_START)
        self.assertEqual(classify_learning_intent("రసప్రశ్నలు", self.session), INTENT_QUIZ_START)

        # 5. Exam prep
        self.assertEqual(classify_learning_intent("give me a 5-mark answer", self.session), INTENT_EXAM_PREP)
        self.assertEqual(classify_learning_intent("2 marks question", self.session), INTENT_EXAM_PREP)

        # 6. Revision / summary
        self.assertEqual(classify_learning_intent("revise this topic", self.session), INTENT_REVISION)
        self.assertEqual(classify_learning_intent("summarize what we learned", self.session), INTENT_REVISION)

        # 7. Step-by-step problem solving
        self.assertEqual(classify_learning_intent("help me solve this step by step", self.session), INTENT_STEP_BY_STEP)
        self.assertEqual(classify_learning_intent("solve 2x + 5 = 15", self.session), INTENT_STEP_BY_STEP)

        # 8. Hint request
        self.assertEqual(classify_learning_intent("give me a hint", self.session), INTENT_HINT)
        self.assertEqual(classify_learning_intent("I am stuck on this", self.session), INTENT_HINT)

        # 9. Switch topic/subject
        self.assertEqual(classify_learning_intent("switch to mathematics", self.session), INTENT_SWITCH_TOPIC)

        # 10. Doubt follow-up (when active topic and history exist)
        self.session.active_topic = "Photosynthesis"
        self.session.add_turn("student", "What is photosynthesis?")
        self.assertEqual(classify_learning_intent("where does it occur?", self.session), INTENT_DOUBT_FOLLOWUP)
        self.assertEqual(classify_learning_intent("why is this necessary?", self.session), INTENT_DOUBT_FOLLOWUP)

    # -------------------------------------------------------------
    # Test 3: Socratic Problem-Solving Instruction & Prompt Building
    # -------------------------------------------------------------
    @patch("voice_assistant.call_gemini")
    def test_socratic_pedagogical_prompt(self, mock_gemini):
        """Verifies Socratic problem solving prompt addition is supplied when solving equations."""
        mock_gemini.return_value = "First, subtract 5 from both sides. What do you get for 2x?"

        reply = generate_reply("Solve 2x + 5 = 15 step by step", profile=self.profile, session=self.session)
        self.assertIn("subtract 5", reply)

        # Verify call_gemini received the Socratic instruction in the prompt
        call_args = mock_gemini.call_args[0]
        prompt_passed = call_args[0]
        sys_inst_passed = call_args[1]

        self.assertIn("Socratic approach", prompt_passed)
        self.assertIn("Step 1", prompt_passed)
        self.assertIn("Socratic Problem Solving", sys_inst_passed)

    # -------------------------------------------------------------
    # Test 4: Session Pause and Resume Flow
    # -------------------------------------------------------------
    def test_pause_and_resume_flow(self):
        """Verifies state transitions and speech replies for session pause and resume."""
        # Pause
        pause_resp = generate_reply("pause please", profile=self.profile, session=self.session)
        self.assertTrue(self.session.is_paused)
        self.assertIn("paused", pause_resp.lower())

        # An attempt to study while paused is politely blocked
        blocked_resp = generate_reply("what is gravity?", profile=self.profile, session=self.session)
        self.assertIn("paused", blocked_resp.lower())

        # Resume
        resume_resp = generate_reply("resume", profile=self.profile, session=self.session)
        self.assertFalse(self.session.is_paused)
        self.assertIn("welcome back", resume_resp.lower())

    # -------------------------------------------------------------
    # Test 5: Subject Switching Flow
    # -------------------------------------------------------------
    @patch("voice_assistant.call_gemini")
    def test_subject_switching_flow(self, mock_gemini):
        """Verifies subject switches correctly within allowed student subjects."""
        mock_gemini.return_value = "Great! Let's explore Mathematics."
        self.assertEqual(self.session.active_subject, "Science")

        generate_reply("switch to mathematics", profile=self.profile, session=self.session)
        self.assertEqual(self.session.active_subject, "Mathematics")
        self.assertEqual(self.session.active_topic, "General")

    # -------------------------------------------------------------
    # Test 6: Adaptive Quiz Engine (Dynamic Difficulty & Hints)
    # -------------------------------------------------------------
    def test_adaptive_quiz_difficulty_scaling(self):
        """Verifies dynamic difficulty adjustment (easy -> medium -> challenging -> medium -> easy)."""
        quiz = self.session.quiz
        self.assertEqual(quiz.difficulty, "medium")

        # Correct answer steps up to challenging
        quiz.adjust_difficulty(was_correct=True)
        self.assertEqual(quiz.difficulty, "challenging")

        # Another correct answer caps at challenging
        quiz.adjust_difficulty(was_correct=True)
        self.assertEqual(quiz.difficulty, "challenging")

        # Incorrect answer steps down to medium
        quiz.adjust_difficulty(was_correct=False)
        self.assertEqual(quiz.difficulty, "medium")

        # Another incorrect answer steps down to easy
        quiz.adjust_difficulty(was_correct=False)
        self.assertEqual(quiz.difficulty, "easy")

    @patch("voice_assistant.call_gemini")
    def test_quiz_generation_with_hint(self, mock_gemini):
        """Verifies quiz question generator produces question, expected answer, and hint."""
        mock_gemini.return_value = (
            "QUESTION: What is the main gas absorbed by plants during photosynthesis?\n"
            "ANSWER: Carbon dioxide (CO2)\n"
            "HINT: It is what humans exhale and plants need to build glucose."
        )

        q, a, h = generate_quiz_question(
            self.profile,
            question_num=1,
            subject="Science",
            topic="Photosynthesis",
            difficulty="medium",
            return_hint=True,
        )

        self.assertIn("absorbed by plants", q)
        self.assertIn("Carbon dioxide", a)
        self.assertIn("humans exhale", h)

    # -------------------------------------------------------------
    # Test 7: Scope Guardrails (Exclusion of College/Engineering/CSE)
    # -------------------------------------------------------------
    def test_scope_guardrails_rejection(self):
        """Verifies strict refusal to teach B.Tech / Engineering / College-level programming."""
        self.assertTrue(is_college_cse_query("explain operating systems deadlock"))
        self.assertTrue(is_college_cse_query("how to write sql query with primary key"))
        self.assertTrue(is_college_cse_query("b.tech computer networks tcp/ip"))

        reply_eng = generate_reply("Explain B.Tech operating system deadlock", profile=self.profile, session=self.session)
        self.assertIn("exclusively specialized as an AI Student Tutor for School", reply_eng)

        reply_tel = generate_reply("deadlock gurinchi Telugu lo explain cheyyi", profile=self.profile, session=self.session)
        self.assertIn("కేవలం స్కూల్", reply_tel)

    # -------------------------------------------------------------
    # Test 8: Telemetry and Adaptive Audio Preservation
    # -------------------------------------------------------------
    def test_audio_result_and_telemetry_preservation(self):
        """Verifies AudioRecordResult tuple indexing and attributes for telemetry."""
        res = AudioRecordResult(
            success=True,
            is_silent=False,
            rms=0.045,
            peak=0.68,
            clipping_count=0,
            total_duration=3.2,
            speech_duration=2.1,
        )

        self.assertTrue(res.success)
        self.assertFalse(res.is_silent)
        self.assertEqual(res[0], True)
        self.assertEqual(res.rms, 0.045)
        self.assertEqual(res.peak, 0.68)
        self.assertEqual(res.total_duration, 3.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
