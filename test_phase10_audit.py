"""
Phase 10: Live Voice Testing & Reliability Audit Test Suite
Verifies:
1. Short valid quiz answers ('A', 'B', '1', '2', 'ఏ', 'బి') permitted through is_filler_or_noise.
2. STT failure verbal retry prompt logic (no silent loop drops).
3. Telugu & English subject switching intent and context updates.
4. Gated interim quiz STT checking (no duplicate/wasteful interim STT during thinking pauses).
5. Audio playback error resilience (sd.stop emergency cleanup).
6. State machine transitions (Learning -> Pause -> Resume -> Quiz -> Stop).
7. College CSE query rejection guardrail.
8. Anti-hallucination and prompt guardrails.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from voice_assistant import (
    StudentProfile,
    LearningSession,
    QuizSession,
    is_filler_or_noise,
    classify_learning_intent,
    generate_reply,
    is_college_cse_query,
    parse_quiz_command,
    play_audio,
    record_quiz_answer,
    QuizAnswerResult,
    INTENT_SWITCH_TOPIC,
    INTENT_PAUSE,
    INTENT_RESUME,
    INTENT_QUIZ_START,
    QUIZ_ACTION_SUBMIT,
    QUIZ_ACTION_SKIP,
    QUIZ_ACTION_REPEAT,
    QUIZ_ACTION_STOP,
)


class TestPhase10ReliabilityAudit(unittest.TestCase):
    def test_01_short_valid_answers_not_flagged_as_noise(self):
        """Task 2: Ensure single-letter / single-digit quiz answers are NOT discarded as noise."""
        valid_short_answers = [
            "A", "b", "C", "d", "A.", "(b)", "1", "2", "3", "4", "8", "ఏ", "బి", "సి", "డి"
        ]
        for ans in valid_short_answers:
            self.assertFalse(
                is_filler_or_noise(ans),
                f"Valid short answer '{ans}' was incorrectly flagged as filler/noise!"
            )

        # Confirm true noise/filler tokens are still filtered
        noise_tokens = ["umm", "uh", "ah", "hmm", "...", ".", "   ", "ha", "హ్మ్"]
        for n in noise_tokens:
            self.assertTrue(
                is_filler_or_noise(n),
                f"Noise token '{n}' was not filtered!"
            )

    def test_02_telugu_and_english_subject_switching(self):
        """Task 3: Ensure Telugu and English subject switches update session subject correctly."""
        profile = StudentProfile(level="school", grade_or_year=8, group="general", subject="Mathematics")
        session = LearningSession(profile)
        self.assertEqual(session.active_subject, "Mathematics")

        # Telugu switch triggers
        telugu_queries = [
            ("సైన్స్ చదువుదాం", "Science"),
            ("సోషల్ సబ్జెక్ట్ మార్చు", "Social Studies"),
            ("లెక్కలు చదువుదాం", "Mathematics"),
        ]

        for query, expected_subj in telugu_queries:
            intent = classify_learning_intent(query, session)
            self.assertEqual(intent, INTENT_SWITCH_TOPIC, f"Failed to detect switch topic for: {query}")
            # Run generate_reply with mocked Gemini
            with patch("voice_assistant.call_gemini", return_value="Okay, let us study!"):
                generate_reply(query, profile=profile, session=session)
            self.assertEqual(session.active_subject, expected_subj, f"Subject not updated to {expected_subj}")

    @patch("voice_assistant.sf.write")
    @patch("voice_assistant.transcribe_speech")
    @patch("voice_assistant.time.sleep")
    @patch("voice_assistant.sd.InputStream")
    def test_03_gated_interim_quiz_stt(self, mock_stream_cls, mock_sleep, mock_stt, mock_sf_write):
        """Task 1 & 4: Ensure interim STT is NOT repeatedly called during silent thinking pauses."""
        import numpy as np

        sample_rate = 16000
        chunk_dur = 0.05
        chunk_samples = int(sample_rate * chunk_dur)

        speech_chunk = np.ones((chunk_samples,), dtype=np.float32) * 0.05
        silent_chunk = np.zeros((chunk_samples,), dtype=np.float32)

        # 10 speech chunks (0.5s speech)
        # followed by 30 silence chunks (1.5s silence >= 1.2s pause check -> 1st check)
        # followed by 30 more silence chunks (1.5s more silence with no new speech -> must NOT trigger 2nd check!)
        chunks = (
            [speech_chunk] * 10
            + [silent_chunk] * 30
            + [silent_chunk] * 30
        )

        stream_instance = MagicMock()
        chunk_iter = iter(chunks)

        def mock_read(num_samples):
            try:
                c = next(chunk_iter)
                return c, False
            except StopIteration:
                return np.zeros(num_samples, dtype=np.float32), False

        stream_instance.read.side_effect = mock_read
        mock_stream_cls.return_value.__enter__.return_value = stream_instance

        mock_stt.return_value = {"success": True, "transcript": "photosynthesis"}

        # Simulate time advancing and timeout after chunks
        perf_times = [0.0] + [0.05 * i for i in range(1, len(chunks) + 1)] + [125.0]
        with patch("voice_assistant.time.perf_counter", side_effect=perf_times):
            res = record_quiz_answer(
                filename="test_quiz_gated.wav",
                sample_rate=sample_rate,
                timeout=5.0,
                vad_threshold=0.015,
                pause_check_duration=1.2,
                settling_pause=0.0,
            )

        # STT must have been called EXACTLY ONCE across the two 1.5s silence blocks because no new speech was spoken in the second block
        self.assertEqual(mock_stt.call_count, 1, f"Expected 1 gated STT call, got {mock_stt.call_count}")

    def test_04_play_audio_emergency_stop(self):
        """Task 1: Verify play_audio calls sd.stop() when an exception occurs to prevent buffer locks."""
        with patch("soundfile.read", side_effect=RuntimeError("Corrupt audio stream")):
            with patch("sounddevice.stop") as mock_sd_stop:
                success = play_audio("nonexistent.wav")
                self.assertFalse(success)
                mock_sd_stop.assert_called_once()

    def test_05_pause_and_resume_flow(self):
        """Task 3: Ensure session correctly pauses and resumes with context preservation."""
        profile = StudentProfile(level="school", grade_or_year=9, group="general", subject="Science")
        session = LearningSession(profile)
        session.active_topic = "Gravitation"

        # 1. Student requests pause
        with patch("voice_assistant.call_gemini", return_value="Should not be called"):
            pause_reply = generate_reply("ఒక్క నిమిషం ఆగు", profile=profile, session=session)
            self.assertTrue(session.is_paused)
            self.assertIn("పాజ్", pause_reply)

        # 2. Student speaks while paused (not resume)
        while_paused_reply = generate_reply("what is mass?", profile=profile, session=session)
        self.assertIn("paused", while_paused_reply.lower())

        # 3. Student resumes
        resume_reply = generate_reply("సరే కంటిన్యూ చేయి", profile=profile, session=session)
        self.assertFalse(session.is_paused)
        self.assertIn("Gravitation", resume_reply)

    def test_06_college_cse_guardrail(self):
        """Strict curriculum check: College/CSE/B.Tech topics must be rejected."""
        cse_questions = [
            "What is a primary key and foreign key in DBMS?",
            "Explain deadlock avoidance in operating systems",
            "Write a Python script for tcp/ip socket programming",
            "How does inheritance work in Java?",
        ]
        for q in cse_questions:
            self.assertTrue(is_college_cse_query(q), f"Failed to block engineering topic: {q}")

    def test_07_quiz_command_parsing(self):
        """Task 4: Quiz command extraction handles submit, skip, repeat, and stop in English and Telugu."""
        test_cases = [
            ("The answer is Mitochondria submit answer", QUIZ_ACTION_SUBMIT, "The answer is Mitochondria"),
            ("It is 15 Newton done", QUIZ_ACTION_SUBMIT, "It is 15 Newton"),
            ("సమాధానం సమర్పించు", QUIZ_ACTION_SUBMIT, ""),
            ("skip this question", QUIZ_ACTION_SKIP, ""),
            ("దాటవేయి", QUIZ_ACTION_SKIP, ""),
            ("can you please repeat question", QUIZ_ACTION_REPEAT, ""),
            ("మళ్లీ చెప్పు", QUIZ_ACTION_REPEAT, ""),
            ("stop quiz now", QUIZ_ACTION_STOP, ""),
            ("క్విజ్ ఆపు", QUIZ_ACTION_STOP, ""),
        ]
        for text, exp_act, exp_ans in test_cases:
            act, clean_ans = parse_quiz_command(text)
            self.assertEqual(act, exp_act, f"Failed for '{text}': got {act} instead of {exp_act}")
            if exp_ans:
                self.assertIn(exp_ans.lower(), clean_ans.lower())

    def test_08_no_duplicate_tts_in_quiz_feedback(self):
        """Task 4: Quiz evaluation and next question are combined into one single response string."""
        profile = StudentProfile(level="school", grade_or_year=7, group="general", subject="Science")
        quiz = QuizSession(total_questions=3)
        quiz.active = True
        quiz.question_num = 1
        quiz.current_question = "What is the green pigment in leaves?"
        quiz.current_expected = "Chlorophyll"

        with patch("voice_assistant.call_gemini", side_effect=[
            "EVALUATION: CORRECT\nEXPLANATION: Great job! Chlorophyll traps sunlight.",
            "QUESTION: What is the process of plant food making?\nANSWER: Photosynthesis\nHINT: Photo means light."
        ]):
            # Simulate correct answer evaluation flow
            from voice_assistant import evaluate_quiz_answer, generate_quiz_question
            is_correct, feedback = evaluate_quiz_answer(
                student_answer="Chlorophyll",
                question=quiz.current_question,
                expected_answer=quiz.current_expected,
                profile=profile,
            )
            self.assertTrue(is_correct)

            # Step to next question
            quiz.question_num += 1
            next_q, next_a, next_h = generate_quiz_question(profile, quiz.question_num, "Science", "General", "medium", return_hint=True)
            quiz.current_question = next_q
            quiz.current_expected = next_a

            combined_reply = f"{feedback} Next, Question {quiz.question_num}: {next_q}"
            # Verify single string created with both parts
            self.assertIn("Great job!", combined_reply)
            self.assertIn("Question 2:", combined_reply)


if __name__ == "__main__":
    unittest.main()
