"""
=============================================================================
AUTOMATED TEST SUITE: SEPARATE NORMAL RECORDING & QUIZ ANSWER LISTENING
=============================================================================
NOTE: These are automated software verification tests using deterministic
mock audio chunk streams and ASR parsing. They run headlessly in any environment
and do NOT require a live microphone or human speaker.
For physical microphone and voice testing, run `python voice_assistant.py`.
=============================================================================
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from voice_assistant import (
    # Normal Mode Configuration
    MAX_RECORD_DURATION,
    SILENCE_TIMEOUT_SECONDS,
    record_audio,
    # Quiz Answer Listening Mode Configuration & Functions
    QUIZ_ANSWER_TIMEOUT_SECONDS,
    QUIZ_PAUSE_CHECK_SECONDS,
    QUIZ_SETTLING_PAUSE,
    QUIZ_ACTION_SUBMIT,
    QUIZ_ACTION_SKIP,
    QUIZ_ACTION_REPEAT,
    QUIZ_ACTION_STOP,
    QUIZ_ACTION_TIMEOUT,
    QUIZ_ACTION_EMPTY,
    QuizAnswerResult,
    parse_quiz_command,
    record_quiz_answer,
    StudentProfile,
    QuizSession,
    LearningSession,
)


class TestQuizAnswerListening(unittest.TestCase):

    def setUp(self):
        self.profile = StudentProfile(level="school", grade_or_year=8, subject="Science")
        self.quiz = QuizSession(total_questions=3)

    # -------------------------------------------------------------------------
    # Test 1: Normal Doubt Mode Configuration & VAD Preservation
    # -------------------------------------------------------------------------
    def test_normal_doubt_mode_configuration(self):
        """Verifies normal doubt mode preserves 1.1s silence cutoff and configurable 30s max duration."""
        self.assertEqual(SILENCE_TIMEOUT_SECONDS, 1.1)
        self.assertEqual(MAX_RECORD_DURATION, 30.0)
        self.assertEqual(QUIZ_ANSWER_TIMEOUT_SECONDS, 120.0)
        self.assertGreater(QUIZ_ANSWER_TIMEOUT_SECONDS, MAX_RECORD_DURATION)

    # -------------------------------------------------------------------------
    # Test 2: Quiz Command Parsing - Submit Commands & Answer Extraction
    # -------------------------------------------------------------------------
    def test_quiz_submit_commands_and_extraction(self):
        """Verifies 'Submit answer', 'Done', and Telugu equivalents end recording and extract clean answer."""
        # 1. Trailing submit answer
        cmd, ans = parse_quiz_command("Mitochondria is the powerhouse of the cell. Submit answer")
        self.assertEqual(cmd, QUIZ_ACTION_SUBMIT)
        self.assertEqual(ans, "Mitochondria is the powerhouse of the cell")

        # 2. Trailing done
        cmd, ans = parse_quiz_command("It produces ATP energy. Done.")
        self.assertEqual(cmd, QUIZ_ACTION_SUBMIT)
        self.assertEqual(ans, "It produces ATP energy")

        # 3. Leading done
        cmd, ans = parse_quiz_command("Done, my answer is chlorophyll")
        self.assertEqual(cmd, QUIZ_ACTION_SUBMIT)
        self.assertEqual(ans, "my answer is chlorophyll")

        # 4. Telugu submit commands
        cmd, ans = parse_quiz_command("మైటోకాండ్రియా శక్తి కేంద్రం. సబ్మిట్")
        self.assertEqual(cmd, QUIZ_ACTION_SUBMIT)
        self.assertEqual(ans, "మైటోకాండ్రియా శక్తి కేంద్రం")

        cmd, ans = parse_quiz_command("మైటోకాండ్రియా. అయిపోయింది.")
        self.assertEqual(cmd, QUIZ_ACTION_SUBMIT)
        self.assertEqual(ans, "మైటోకాండ్రియా")

        # 5. Standalone submit
        cmd, ans = parse_quiz_command("Submit")
        self.assertEqual(cmd, QUIZ_ACTION_SUBMIT)
        self.assertEqual(ans, "")

    # -------------------------------------------------------------------------
    # Test 3: Quiz Command Parsing - Skip, Repeat, and Stop Commands
    # -------------------------------------------------------------------------
    def test_quiz_control_commands(self):
        """Verifies Skip, Repeat question, and Stop quiz commands are accurately detected."""
        # Skip commands
        cmd, _ = parse_quiz_command("skip")
        self.assertEqual(cmd, QUIZ_ACTION_SKIP)
        cmd, _ = parse_quiz_command("skip question please")
        self.assertEqual(cmd, QUIZ_ACTION_SKIP)
        cmd, _ = parse_quiz_command("స్కిప్ చేయండి")
        self.assertEqual(cmd, QUIZ_ACTION_SKIP)
        cmd, _ = parse_quiz_command("నాకు ఇది వద్దు")
        self.assertEqual(cmd, QUIZ_ACTION_SKIP)

        # Repeat commands
        cmd, _ = parse_quiz_command("repeat question")
        self.assertEqual(cmd, QUIZ_ACTION_REPEAT)
        cmd, _ = parse_quiz_command("can you repeat please")
        self.assertEqual(cmd, QUIZ_ACTION_REPEAT)
        cmd, _ = parse_quiz_command("మళ్లీ చెప్పు")
        self.assertEqual(cmd, QUIZ_ACTION_REPEAT)
        cmd, _ = parse_quiz_command("ఒక్కసారి మళ్లీ చెప్పు")
        self.assertEqual(cmd, QUIZ_ACTION_REPEAT)

        # Stop commands
        cmd, _ = parse_quiz_command("stop quiz")
        self.assertEqual(cmd, QUIZ_ACTION_STOP)
        cmd, _ = parse_quiz_command("exit quiz")
        self.assertEqual(cmd, QUIZ_ACTION_STOP)
        cmd, _ = parse_quiz_command("క్విజ్ ఆపు")
        self.assertEqual(cmd, QUIZ_ACTION_STOP)
        cmd, _ = parse_quiz_command("stop")
        self.assertEqual(cmd, QUIZ_ACTION_STOP)

        # Regular answer text (no command)
        cmd, ans = parse_quiz_command("Mitochondria produces cellular energy through aerobic respiration")
        self.assertIsNone(cmd)
        self.assertEqual(ans, "Mitochondria produces cellular energy through aerobic respiration")

    # -------------------------------------------------------------------------
    # Test 4: Listening Through Natural Pauses (No 1.1s Silence Cutoff in Quiz)
    # -------------------------------------------------------------------------
    @patch("voice_assistant.sf.write")
    @patch("voice_assistant.transcribe_speech")
    @patch("voice_assistant.time.sleep")
    @patch("voice_assistant.sd.InputStream")
    def test_listening_through_natural_pauses(self, mock_stream_cls, mock_sleep, mock_stt, mock_sf_write):
        """
        Verifies that during quiz mode, a natural pause of 1.5s does NOT abort or submit.
        The listener transcribes interim audio, notes no submit command, continues listening,
        and finalizes only when 'Submit answer' is spoken in phrase 2.
        """
        sample_rate = 16000
        chunk_samples = int(sample_rate * 0.05)  # 50ms chunk

        speech_chunk = np.ones(chunk_samples, dtype="float32") * 0.1
        silence_chunk = np.zeros(chunk_samples, dtype="float32")

        # Simulate audio chunks stream:
        # 1. Speech phrase 1: 10 chunks (0.5s)
        # 2. Natural pause: 30 chunks (1.5s) -> triggers interim check, no submit
        # 3. Speech phrase 2: 10 chunks (0.5s) with submit
        # 4. Final pause: 30 chunks (1.5s) -> triggers command check, submit found!
        chunks = (
            [speech_chunk] * 10
            + [silence_chunk] * 30
            + [speech_chunk] * 10
            + [silence_chunk] * 30
        )

        stream_instance = MagicMock()
        chunk_iter = iter(chunks)

        def mock_read(num_samples):
            try:
                c = next(chunk_iter)
                return c, False
            except StopIteration:
                # Return small silence after stream ends
                return np.zeros(num_samples, dtype="float32"), False

        stream_instance.read.side_effect = mock_read
        mock_stream_cls.return_value.__enter__.return_value = stream_instance

        # Mock STT responses:
        # First check (during pause): "Mitochondria" (no command)
        # Second check (after phrase 2): "Mitochondria produces ATP. Submit answer" (submit command found!)
        mock_stt.side_effect = [
            {"success": True, "transcript": "Mitochondria"},
            {"success": True, "transcript": "Mitochondria produces ATP. Submit answer"},
        ]

        result = record_quiz_answer(
            filename="mock_quiz.wav",
            sample_rate=sample_rate,
            timeout=10.0,
            settling_pause=0.0,
        )

        self.assertEqual(result.action, QUIZ_ACTION_SUBMIT)
        self.assertIn("Mitochondria produces ATP", result.answer_text)
        self.assertFalse(result.is_silent)
        # Confirms STT was called across the natural pause without aborting
        self.assertGreaterEqual(mock_stt.call_count, 2)

    # -------------------------------------------------------------------------
    # Test 5: Long Spoken Quiz Answers (Exceeding Normal 15s/30s Cap)
    # -------------------------------------------------------------------------
    @patch("voice_assistant.sf.write")
    @patch("voice_assistant.transcribe_speech")
    @patch("voice_assistant.time.sleep")
    @patch("voice_assistant.sd.InputStream")
    def test_long_answer_recording(self, mock_stream_cls, mock_sleep, mock_stt, mock_sf_write):
        """Verifies students can speak detailed answers beyond 30s without normal cap cutting them off."""
        sample_rate = 16000
        chunk_samples = int(sample_rate * 0.05)

        speech_chunk = np.ones(chunk_samples, dtype="float32") * 0.08
        silence_chunk = np.zeros(chunk_samples, dtype="float32")

        # Simulate 35 seconds of speech (700 chunks of 50ms) + 1.5s trailing silence (30 chunks)
        chunks = [speech_chunk] * 700 + [silence_chunk] * 30

        stream_instance = MagicMock()
        chunk_iter = iter(chunks)

        def mock_read(num_samples):
            try:
                return next(chunk_iter), False
            except StopIteration:
                return np.zeros(num_samples, dtype="float32"), False

        stream_instance.read.side_effect = mock_read
        mock_stream_cls.return_value.__enter__.return_value = stream_instance

        mock_stt.return_value = {
            "success": True,
            "transcript": "Cellular respiration takes place in three stages: glycolysis, Krebs cycle, and electron transport chain. Done.",
        }

        # Use 120s timeout
        result = record_quiz_answer(
            filename="mock_quiz_long.wav",
            sample_rate=sample_rate,
            timeout=120.0,
            settling_pause=0.0,
        )

        self.assertEqual(result.action, QUIZ_ACTION_SUBMIT)
        self.assertIn("glycolysis", result.answer_text)
        self.assertGreaterEqual(result.speech_duration, 30.0)

    # -------------------------------------------------------------------------
    # Test 6: Skip and Repeat Question Commands in Stream
    # -------------------------------------------------------------------------
    @patch("voice_assistant.sf.write")
    @patch("voice_assistant.transcribe_speech")
    @patch("voice_assistant.time.sleep")
    @patch("voice_assistant.sd.InputStream")
    def test_skip_command_in_stream(self, mock_stream_cls, mock_sleep, mock_stt, mock_sf_write):
        """Verifies saying 'Skip' immediately ends answer recording with QUIZ_ACTION_SKIP."""
        sample_rate = 16000
        chunk_samples = int(sample_rate * 0.05)
        chunks = [np.ones(chunk_samples, dtype="float32") * 0.1] * 10 + [np.zeros(chunk_samples, dtype="float32")] * 30

        stream_instance = MagicMock()
        chunk_iter = iter(chunks)
        stream_instance.read.side_effect = lambda n: (next(chunk_iter, np.zeros(n, dtype="float32")), False)
        mock_stream_cls.return_value.__enter__.return_value = stream_instance

        mock_stt.return_value = {"success": True, "transcript": "skip question"}

        result = record_quiz_answer(
            filename="mock_skip.wav",
            sample_rate=sample_rate,
            timeout=30.0,
            settling_pause=0.0,
        )

        self.assertEqual(result.action, QUIZ_ACTION_SKIP)

    # -------------------------------------------------------------------------
    # Test 7: Safety Timeout Handling (120s Reached with No Speech)
    # -------------------------------------------------------------------------
    @patch("voice_assistant.time.sleep")
    @patch("voice_assistant.sd.InputStream")
    def test_safety_timeout_with_no_speech(self, mock_stream_cls, mock_sleep):
        """Verifies when a student says nothing up to the safety timeout, TIMEOUT action is cleanly returned."""
        sample_rate = 16000
        chunk_samples = int(sample_rate * 0.05)

        stream_instance = MagicMock()
        # Return complete silence
        stream_instance.read.side_effect = lambda n: (np.zeros(n, dtype="float32"), False)
        mock_stream_cls.return_value.__enter__.return_value = stream_instance

        # Test with short simulated timeout of 0.2s
        result = record_quiz_answer(
            filename="mock_timeout.wav",
            sample_rate=sample_rate,
            timeout=0.2,
            settling_pause=0.0,
        )

        self.assertEqual(result.action, QUIZ_ACTION_TIMEOUT)
        self.assertTrue(result.is_silent)
        self.assertEqual(result.answer_text, "")

    # -------------------------------------------------------------------------
    # Test 8: Microphone Error Graceful Recovery
    # -------------------------------------------------------------------------
    @patch("voice_assistant.sd.InputStream")
    def test_microphone_error_graceful_recovery(self, mock_stream_cls):
        """Verifies that an audio device failure does not crash the application and returns QUIZ_ACTION_EMPTY."""
        mock_stream_cls.side_effect = RuntimeError("Microphone device disconnected")

        result = record_quiz_answer(
            filename="mock_err.wav",
            settling_pause=0.0,
        )

        self.assertEqual(result.action, QUIZ_ACTION_EMPTY)
        self.assertTrue(result.is_silent)

    # -------------------------------------------------------------------------
    # Test 9: Echo Prevention Settling Pause
    # -------------------------------------------------------------------------
    @patch("voice_assistant.time.sleep")
    @patch("voice_assistant.sd.InputStream")
    def test_settling_pause_applied(self, mock_stream_cls, mock_sleep):
        """Verifies settling pause (0.35s default) is applied to ensure tutor TTS echo does not bleed into mic."""
        mock_stream_cls.side_effect = RuntimeError("Abort after pause check")
        record_quiz_answer(settling_pause=0.35)

        mock_sleep.assert_called_with(0.35)


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("🧪  RUNNING AUTOMATED UNIT TESTS: SEPARATE NORMAL & QUIZ LISTENING")
    print("=" * 75)
    unittest.main(verbosity=2)
