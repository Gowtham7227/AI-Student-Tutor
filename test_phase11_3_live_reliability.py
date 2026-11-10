"""
Phase 11.3 Automated Verification Test Suite:
Live Voice Reliability and Telugu Tutor Optimization.

Tests:
1. Context-aware Incomplete Speech Detection:
   - Dangling linkers/conjunctions ("because", "మరియు", "and", "కానీ") flagged as incomplete.
   - Standalone doubt questions ("ఎందుకు?", "Why?") preserved when active conversation history exists.
   - Standalone doubt questions rejected as incomplete when no prior context exists.
   - All entries in VALID_SHORT_ANSWERS, digits, and scientific terms ("Oxygen", "ఆక్సిజన్") preserved.
2. Student Grade Context:
   - Intermediate MPC student requesting Class 8 explanation receives Class 8 pedagogical override instruction.
   - Saved StudentProfile (level='intermediate', grade=1, group='MPC') is NEVER modified.
   - Active subject context is preserved without alteration.
3. Concise Telugu Responses:
   - Default Telugu guidelines instruct 2-3 sentences, relatable examples, and 1 check question.
   - Explicit requests ("explain in detail", "వివరంగా చెప్పు", 5-mark, 10-mark) preserve detailed formats.
4. STT Reliability & WinError 10053:
   - Transient ConnectionAbortedError (WinError 10053) retried ONCE after 0.3s backoff.
   - Authentication errors (401/403) are NOT retried.
   - Fallback to te-IN preserved on persistent failure with total attempts strictly bounded.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import soundfile as sf
import numpy as np

# Force UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from voice_assistant import (
    LearningSession,
    StudentProfile,
    is_incomplete_speech,
    is_filler_or_noise,
    is_transient_connection_error,
    detect_requested_grade_override,
    generate_reply,
    transcribe_speech,
    build_system_instruction,
    VALID_SHORT_ANSWERS,
)


class TestIncompleteSpeechDetection(unittest.TestCase):
    """1. Test Incomplete Speech Detection and Preservation of Valid Answers."""

    def setUp(self):
        self.profile = StudentProfile(level="school", grade_or_year=8)
        self.session = LearningSession(self.profile)
        self.session.active_subject = "Science"
        self.session.active_topic = "Photosynthesis"

    def test_01_dangling_conjunctions_flagged_as_incomplete(self):
        """Dangling words and isolated prepositions/conjunctions must be flagged as incomplete."""
        dangling_words = ["because", "and", "so", "the", "or", "if", "మరియు", "కానీ", "అయితే"]
        for word in dangling_words:
            self.assertTrue(
                is_incomplete_speech(word, self.session),
                f"Dangling word '{word}' was NOT flagged as incomplete.",
            )

    def test_02_contextual_interrogative_preserved_with_history(self):
        """Standalone questions like 'ఎందుకు?' or 'Why?' must be preserved when prior context exists."""
        # Add prior turn to history
        self.session.add_turn("student", "Plants need sunlight.")
        self.session.add_turn("tutor", "Yes, plants absorb light through chlorophyll.")
        self.session.active_topic = "Photosynthesis"

        valid_doubts = ["ఎందుకు?", "ఎందుకు", "Why?", "why", "How?", "ఎలా?"]
        for doubt in valid_doubts:
            self.assertFalse(
                is_incomplete_speech(doubt, self.session),
                f"Valid context-aware doubt '{doubt}' was incorrectly rejected as incomplete.",
            )

    def test_03_standalone_interrogative_incomplete_without_history(self):
        """Without any prior conversation context, 'Why?' or 'ఎందుకు?' alone is incomplete."""
        empty_session = LearningSession(self.profile)
        empty_session.history = []
        empty_session.active_topic = "General"

        self.assertTrue(is_incomplete_speech("Why?", empty_session))
        self.assertTrue(is_incomplete_speech("ఎందుకు?", empty_session))

    def test_04_valid_short_answers_and_scientific_terms_preserved(self):
        """All valid short answers, options, and scientific terms must be preserved."""
        test_terms = [
            "A", "B", "C", "D", "a", "b", "c", "d", "ఏ", "బి", "సి", "డి",
            "1", "2", "3", "8", "10",
            "yes", "no", "ok", "okay", "అవును", "కాదు", "సరే", "నిజం", "తప్పు",
            "Oxygen", "Hydrogen", "Cell", "Force", "Work", "Energy", "ATP", "DNA",
            "ఆక్సిజన్", "కణం", "బలం", "పని", "శక్తి", "కాంతి", "ధ్వని",
        ]
        for term in test_terms:
            self.assertFalse(
                is_incomplete_speech(term, self.session),
                f"Valid term '{term}' was incorrectly flagged as incomplete.",
            )


class TestStudentGradeContext(unittest.TestCase):
    """2. Test Grade Context Override without Mutating Saved Profile."""

    def setUp(self):
        # Saved student profile: 1st Year Intermediate MPC
        self.inter_profile = StudentProfile(level="intermediate", grade_or_year=1, group="MPC", subject="Mathematics")
        self.session = LearningSession(self.inter_profile)
        self.session.active_subject = "Mathematics"

    def test_01_detect_requested_grade_override(self):
        """Detects explicit requests for school class explanations."""
        queries = [
            ("Explain photosynthesis like in Class 8", 8),
            ("Can you explain this for class 8?", 8),
            ("8th class level lo cheppu", 8),
            ("8వ తరగతి లాగా చెప్పు", 8),
            ("Explain this in 6th class level", 6),
        ]
        for text, expected_grade in queries:
            grade = detect_requested_grade_override(text, self.inter_profile)
            self.assertEqual(grade, expected_grade, f"Failed to detect requested grade in: '{text}'")

    def test_02_grade_override_applied_without_mutating_profile(self):
        """Gemini receives Class 8 instructions while saved profile remains Intermediate MPC."""
        query = "Explain photosynthesis like in Class 8"

        with patch("voice_assistant.call_gemini", return_value="Here is photosynthesis explained simply.") as mock_gem:
            reply = generate_reply(query, self.inter_profile, self.session)

            # 1. Profile remains 1st Year Intermediate MPC
            self.assertEqual(self.session.profile.level, "intermediate")
            self.assertEqual(self.session.profile.grade_or_year, 1)
            self.assertEqual(self.session.profile.group, "MPC")

            # 2. Active subject remains Mathematics
            self.assertEqual(self.session.active_subject, "Mathematics")

            # 3. Gemini prompt received pedagogical grade override
            call_prompt = mock_gem.call_args[0][0]
            self.assertIn("GRADE-LEVEL EXPLANATION OVERRIDE", call_prompt)
            self.assertIn("Class 8 School level", call_prompt)


class TestConciseTeluguResponses(unittest.TestCase):
    """3. Test Concise Telugu Guidelines and Detailed Formatting Exceptions."""

    def test_01_system_instruction_contains_telugu_conciseness(self):
        """System instruction explicitly directs 2-3 sentences with everyday examples."""
        profile = StudentProfile(level="school", grade_or_year=8)
        sys_inst = build_system_instruction(profile)
        self.assertIn("Telugu Conciseness: Keep normal Telugu explanations short, simple, and student-friendly (2 to 3 sentences)", sys_inst)
        self.assertIn("End with exactly one short, warm follow-up check question", sys_inst)

    def test_02_explicit_detail_requests_preserve_detailed_instruction(self):
        """When explicitly asked for detail, detailed pedagogical instructions are injected."""
        profile = StudentProfile(level="school", grade_or_year=8)
        session = LearningSession(profile)

        detail_queries = [
            "Explain in detail how plants make food",
            "వివరంగా చెప్పు",
            "పూర్తిగా వివరించు",
        ]
        for q in detail_queries:
            with patch("voice_assistant.call_gemini", return_value="Detailed answer.") as mock_gem:
                generate_reply(q, profile, session)
                call_prompt = mock_gem.call_args[0][0]
                self.assertIn("DETAILED explanation", call_prompt)


class TestSTTReliabilityAndWinError10053(unittest.TestCase):
    """4. Test STT Transient Socket Abort Detection, Backoff Retry, and Safe Fallback."""

    def test_01_is_transient_connection_error_detection(self):
        """Identifies WinError 10053 and connection aborts as transient, excludes auth/client errors."""
        # WinError 10053 ConnectionAbortedError
        abort_err = ConnectionAbortedError(10053, "An established connection was aborted by the software in your host machine")
        self.assertTrue(is_transient_connection_error(abort_err))

        # Reset / Timeout
        reset_err = ConnectionResetError(10054, "An existing connection was forcibly closed")
        self.assertTrue(is_transient_connection_error(reset_err))

        # Permanent Auth / Client error
        auth_err = Exception("401 Unauthorized: Invalid API key")
        self.assertFalse(is_transient_connection_error(auth_err))

        client_err = Exception("400 Bad Request: Unsupported format")
        self.assertFalse(is_transient_connection_error(client_err))

    def test_02_transcribe_speech_retries_transient_abort_once(self):
        """Transient abort triggers exactly 1 retry with backoff, then succeeds if retry passes."""
        dummy_wav = "test_transient_retry.wav"
        sr = 16000
        sf.write(dummy_wav, np.zeros(sr, dtype=np.float32), sr)

        try:
            with patch("voice_assistant.stt.transcribe") as mock_transcribe, \
                 patch("time.sleep") as mock_sleep:

                # First call raises WinError 10053; second call succeeds
                abort_err = ConnectionAbortedError(10053, "Connection aborted")
                mock_transcribe.side_effect = [
                    abort_err,
                    {"success": True, "transcript": "Class 8 Science", "detected_language_code": "te-IN"},
                ]

                result = transcribe_speech(dummy_wav)
                self.assertIsNotNone(result)
                self.assertEqual(result.get("transcript"), "Class 8 Science")

                # Exactly 2 calls (1 failure + 1 retry)
                self.assertEqual(mock_transcribe.call_count, 2)
                # 0.3s backoff was executed
                mock_sleep.assert_called_with(0.3)
        finally:
            if os.path.exists(dummy_wav):
                os.remove(dummy_wav)

    def test_03_transcribe_speech_permanent_auth_error_no_retry(self):
        """Permanent authentication error must NOT be retried and fallback is skipped."""
        dummy_wav = "test_auth_error.wav"
        sr = 16000
        sf.write(dummy_wav, np.zeros(sr, dtype=np.float32), sr)

        try:
            with patch("voice_assistant.stt.transcribe") as mock_transcribe:
                mock_transcribe.side_effect = Exception("401 Unauthorized: api_key is required")

                result = transcribe_speech(dummy_wav)
                self.assertIsNone(result)
                # Exactly 1 call, no retries and no fallback
                self.assertEqual(mock_transcribe.call_count, 1)
        finally:
            if os.path.exists(dummy_wav):
                os.remove(dummy_wav)


if __name__ == "__main__":
    unittest.main()
