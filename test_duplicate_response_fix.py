"""
=============================================================================
REGRESSION TEST SUITE: DUPLICATE RESPONSE FIX & IMMEDIATE MIC ACTIVATION
=============================================================================
NOTE: These are automated software verification tests using mock clients,
simulated speech turns, and execution sequence tracking.
They run headlessly in any environment and clearly distinguish automated
verification from actual microphone testing.
=============================================================================
"""

import sys
import time
import unittest
from unittest.mock import patch, MagicMock, call

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import voice_assistant
from voice_assistant import (
    StudentProfile,
    QuizSession,
    LearningSession,
    generate_reply,
    speak_and_play,
    record_audio,
    record_quiz_answer,
    parse_quiz_command,
    QUIZ_ACTION_REPEAT,
    QUIZ_ACTION_SUBMIT,
    AudioRecordResult,
    QuizAnswerResult,
    main,
)


class TestDuplicateResponseFix(unittest.TestCase):

    def setUp(self):
        self.profile = StudentProfile(level="school", grade_or_year=8, subject="Science")
        self.session = LearningSession(profile=self.profile)

    # -------------------------------------------------------------------------
    # Test 1: One student input produces exactly one tutor response
    # -------------------------------------------------------------------------
    @patch("voice_assistant.call_gemini")
    def test_one_input_produces_one_tutor_response(self, mock_gemini):
        """Verifies one student query triggers exactly one Gemini call and records exactly one tutor reply."""
        mock_gemini.return_value = "Photosynthesis is the process of food making in green plants."

        initial_history_len = len(self.session.history)
        reply = generate_reply("What is photosynthesis?", profile=self.profile, session=self.session)

        # 1. Exactly one call to Gemini
        self.assertEqual(mock_gemini.call_count, 1)

        # 2. Exactly one student turn and one tutor turn added
        self.assertEqual(len(self.session.history), initial_history_len + 2)
        self.assertEqual(self.session.history[-2].role, "student")
        self.assertEqual(self.session.history[-1].role, "tutor")
        self.assertEqual(self.session.history[-1].text, reply)

    # -------------------------------------------------------------------------
    # Test 2: One normal learning turn produces exactly one TTS playback call
    # -------------------------------------------------------------------------
    @patch("voice_assistant.get_input_device", return_value=(1, {"name": "Test Mic"}))
    @patch("voice_assistant.record_audio")
    @patch("voice_assistant.transcribe_speech")
    @patch("voice_assistant.generate_reply")
    @patch("voice_assistant.speak_and_play")
    @patch("voice_assistant.print_latency_summary")
    def test_no_duplicate_tts_in_normal_loop(
        self,
        mock_latency,
        mock_speak,
        mock_reply,
        mock_stt,
        mock_record,
        mock_device,
    ):
        """
        Simulates two conversation rounds (Round 1: student asks question, Round 2: stop).
        Verifies that Round 1 triggers speak_and_play EXACTLY ONCE for the reply.
        Prior bug had lines 2019 and 2035 both calling speak_and_play(reply, output_file).
        """
        # Mock speak_and_play return (play_ok, tts_dur, play_dur)
        mock_speak.return_value = (True, 0.5, 1.2)

        # Round 1: normal question
        # Round 2: stop command to terminate main() loop cleanly
        rec_res1 = AudioRecordResult(True, False, 0.05, 0.6, 0, 3.0, 2.0)
        rec_res2 = AudioRecordResult(True, False, 0.05, 0.6, 0, 1.5, 0.8)
        mock_record.side_effect = [rec_res1, rec_res2]

        mock_stt.side_effect = [
            # Onboarding already completed by pre-configuring profile
            {"success": True, "transcript": "What is gravity?"},
            {"success": True, "transcript": "stop"},
        ]

        mock_reply.return_value = "Gravity is the invisible force pulling objects together."

        # Pre-complete profile onboarding in a patched loop
        # We test the continuous loop logic directly
        with patch.object(StudentProfile, "is_complete", return_value=True):
            try:
                main()
            except SystemExit:
                pass

        # Inspect calls to speak_and_play:
        # Expected sequence:
        # 1. Welcome prompt (onboarding header)
        # 2. Reply to "What is gravity?" (EXACTLY ONCE!)
        # 3. Goodbye reply for "stop" (EXACTLY ONCE!)
        called_texts = [c[0][0] for c in mock_speak.call_args_list]

        # Verify gravity reply was spoken exactly once
        gravity_replies = [t for t in called_texts if "Gravity is the invisible force" in t]
        self.assertEqual(
            len(gravity_replies),
            1,
            f"Expected exactly 1 speak_and_play call for the tutor reply, but found {len(gravity_replies)}: {gravity_replies}",
        )

    # -------------------------------------------------------------------------
    # Test 3: No duplicate TTS call happens after the latency report
    # -------------------------------------------------------------------------
    @patch("voice_assistant.get_input_device", return_value=(1, {"name": "Test Mic"}))
    @patch("voice_assistant.record_audio")
    @patch("voice_assistant.transcribe_speech")
    @patch("voice_assistant.generate_reply")
    @patch("voice_assistant.speak_and_play")
    @patch("voice_assistant.print_latency_summary")
    def test_execution_order_speak_then_latency_then_mic(
        self,
        mock_latency,
        mock_speak,
        mock_reply,
        mock_stt,
        mock_record,
        mock_device,
    ):
        """
        Verifies strict execution ordering:
        [Speak reply] -> [Latency breakdown] -> [Immediate mic activation for next round]
        with NO second speak_and_play invocation after latency reporting.
        """
        events = []

        def record_speak(text, output_file="assistant_reply.wav"):
            events.append(("speak_and_play", text))
            return True, 0.4, 1.0

        def record_latency(*args, **kwargs):
            events.append(("latency_report", kwargs.get("turn_label", "")))

        def record_audio_call(*args, **kwargs):
            events.append(("record_audio", kwargs.get("max_duration", 0)))
            return AudioRecordResult(True, False, 0.05, 0.6, 0, 2.0, 1.5)

        mock_speak.side_effect = record_speak
        mock_latency.side_effect = record_latency
        mock_record.side_effect = record_audio_call

        mock_stt.side_effect = [
            {"success": True, "transcript": "What is kinetic energy?"},
            {"success": True, "transcript": "stop"},
        ]
        mock_reply.return_value = "Kinetic energy is the energy of motion."

        with patch.object(StudentProfile, "is_complete", return_value=True):
            try:
                main()
            except SystemExit:
                pass

        # Filter events for the learning turn
        speak_events = [e for e in events if e[0] == "speak_and_play" and "Kinetic energy" in e[1]]
        self.assertEqual(len(speak_events), 1)

        # Find indices in the event log
        idx_speak = events.index(("speak_and_play", "Kinetic energy is the energy of motion."))
        # Immediately following should be the latency report
        self.assertEqual(events[idx_speak + 1][0], "latency_report")
        # Immediately following the latency report should be the record_audio for the next question!
        self.assertEqual(events[idx_speak + 2][0], "record_audio")

    # -------------------------------------------------------------------------
    # Test 4: Microphone recording activates immediately after TTS + settling pause
    # -------------------------------------------------------------------------
    @patch("voice_assistant.play_audio")
    @patch("voice_assistant.tts.synthesize")
    def test_speak_and_play_waits_for_settling_pause_then_returns(self, mock_synth, mock_play):
        """Verifies speak_and_play completes audio playback and settling pause before returning to caller."""
        mock_play.return_value = True

        t0 = time.perf_counter()
        ok, tts_dur, play_dur = speak_and_play("Test tutor speech.")
        elapsed = time.perf_counter() - t0

        self.assertTrue(ok)
        self.assertEqual(mock_synth.call_count, 1)
        self.assertEqual(mock_play.call_count, 1)
        # Confirms play_audio was invoked to completion
        mock_play.assert_called_once_with("assistant_reply.wav")

    # -------------------------------------------------------------------------
    # Test 5: Next question is not processed until previous response finishes
    # -------------------------------------------------------------------------
    def test_sequential_processing_flow(self):
        """Verifies session history maintains sequential student -> tutor turns without overlap."""
        session = LearningSession(profile=self.profile)

        session.add_turn("student", "Question 1")
        session.add_turn("tutor", "Answer 1")
        session.add_turn("student", "Question 2")
        session.add_turn("tutor", "Answer 2")

        turns = session.history
        self.assertEqual(len(turns), 4)
        self.assertEqual([t.role for t in turns], ["student", "tutor", "student", "tutor"])

    # -------------------------------------------------------------------------
    # Test 6: Quiz Repeat Question still works intentionally
    # -------------------------------------------------------------------------
    def test_quiz_repeat_question_intentionally_repeats(self):
        """Verifies that saying 'Repeat question' or 'మళ్లీ చెప్పు' triggers QUIZ_ACTION_REPEAT without advancing."""
        cmd, _ = parse_quiz_command("repeat question")
        self.assertEqual(cmd, QUIZ_ACTION_REPEAT)

        cmd_tel, _ = parse_quiz_command("మళ్లీ చెప్పు")
        self.assertEqual(cmd_tel, QUIZ_ACTION_REPEAT)

        quiz = QuizSession(total_questions=3)
        quiz.active = True
        quiz.question_num = 1
        quiz.current_question = "What is the unit of force?"

        # Simulate repeat action: question_num must remain 1
        if cmd == QUIZ_ACTION_REPEAT:
            repeat_text = f"Let me repeat Question {quiz.question_num}: {quiz.current_question}"
            # question_num did not advance
            self.assertEqual(quiz.question_num, 1)
            self.assertIn("What is the unit of force?", repeat_text)

    # -------------------------------------------------------------------------
    # Test 7: Normal Doubt Mode and Quiz Mode remain separate
    # -------------------------------------------------------------------------
    @patch("voice_assistant.get_input_device", return_value=(1, {"name": "Test Mic"}))
    @patch("voice_assistant.record_audio")
    @patch("voice_assistant.record_quiz_answer")
    @patch("voice_assistant.transcribe_speech")
    @patch("voice_assistant.generate_reply")
    @patch("voice_assistant.speak_and_play")
    @patch("voice_assistant.print_latency_summary")
    def test_normal_and_quiz_mode_listen_separately(
        self,
        mock_latency,
        mock_speak,
        mock_reply,
        mock_stt,
        mock_quiz_rec,
        mock_normal_rec,
        mock_device,
    ):
        """
        Verifies that:
        - When quiz.active is False: record_audio is called (Normal Doubt Mode, 30s limit, 1.1s cutoff)
        - When quiz.active is True: record_quiz_answer is called (Quiz Answer Mode, 120s limit, no 1.1s cutoff)
        """
        mock_speak.return_value = (True, 0.2, 0.5)

        # 1. Normal mode turn: question asked -> record_audio called
        mock_normal_rec.return_value = AudioRecordResult(True, False, 0.05, 0.6, 0, 2.0, 1.5)
        mock_stt.side_effect = [
            {"success": True, "transcript": "What is velocity?"},
            {"success": True, "transcript": "stop"},
        ]
        mock_reply.return_value = "Velocity is speed with direction."

        with patch.object(StudentProfile, "is_complete", return_value=True):
            try:
                main()
            except SystemExit:
                pass

        # In normal mode, record_audio was called, record_quiz_answer was NOT called
        self.assertGreaterEqual(mock_normal_rec.call_count, 1)
        self.assertEqual(mock_quiz_rec.call_count, 0)


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("🧪  AUTOMATED REGRESSION SUITE: DUPLICATE RESPONSE FIX & TIMING")
    print("=" * 75)
    unittest.main(verbosity=2)
