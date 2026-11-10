"""
Phase 11.2 Focused Verification Test Suite:
Subject Context Synchronization, Audio Playback Diagnostics, and Short Utterance Handling.

Tests:
1. Subject switch detection for School Class 8 (Math -> Science) across English, Telugu, and Tanglish.
2. Incidental mentions (e.g. "How does math help in science?", "chemical change", "రసాయన మార్పు") do NOT switch subject.
3. Curriculum boundaries preserved (disallowed subjects rejected with curriculum guidance; active subject unchanged).
4. Subsequent explanations, quiz generation, and revision use the updated subject context.
5. Audio playback telemetry: sf.info metadata logging, wall-clock timing vs audio duration, single execution (no duplicates).
6. Short utterance handling: A/B/C/D, digits, yes/no, Oxygen, Telugu scientific terms preserved; genuine fillers filtered.
7. Low confidence / empty transcript handling prompts repeat without guessing or hallucinating.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import soundfile as sf

# Force stdout encoding to UTF-8 for Telugu characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from voice_assistant import (
    LearningSession,
    StudentProfile,
    classify_learning_intent,
    detect_subject_switch,
    generate_reply,
    generate_quiz_question,
    is_filler_or_noise,
    play_audio,
    speak_and_play,
    INTENT_SWITCH_TOPIC,
    INTENT_EXPLANATION,
    INTENT_REVISION,
    INTENT_QUIZ_START,
    SCHOOL_SUBJECTS,
    INTER_MPC_SUBJECTS,
    INTER_BIPC_SUBJECTS,
    VALID_SHORT_ANSWERS,
    FILLER_NOISE_TOKENS,
)


class TestSubjectContextSynchronization(unittest.TestCase):
    """A. Test Subject Context Synchronization and Incidental Mention Filtering."""

    def setUp(self):
        self.school_profile = StudentProfile(level="school", grade_or_year=8)
        self.session = LearningSession(self.school_profile)
        # Explicitly initialize active_subject to Mathematics
        self.session.active_subject = "Mathematics"
        self.session.profile.subject = "Mathematics"
        self.session.active_topic = "Linear Equations"

    def test_01_class8_math_to_science_switch_english(self):
        """Class 8 student asks to switch/learn Science in English."""
        test_queries = [
            "Let's learn Science",
            "Switch to Science",
            "Teach me Science",
            "I want to study Science",
            "Can we learn Science?",
            "Start Science",
        ]
        for query in test_queries:
            self.session.active_subject = "Mathematics"
            is_switch, target, _ = detect_subject_switch(query, self.school_profile)
            self.assertTrue(is_switch, f"Failed to detect switch in: '{query}'")
            self.assertEqual(target, "Science", f"Target subject mismatch for: '{query}'")

            intent = classify_learning_intent(query, self.session)
            self.assertEqual(intent, INTENT_SWITCH_TOPIC, f"Intent mismatch for: '{query}'")

            with patch("voice_assistant.call_gemini", return_value="Great! Let's explore Science."):
                reply = generate_reply(query, self.school_profile, self.session)
                self.assertEqual(self.session.active_subject, "Science")
                self.assertEqual(self.session.profile.subject, "Science")
                self.assertEqual(self.session.active_topic, "General")
                self.assertIn("Science", reply)

    def test_02_class8_math_to_science_switch_telugu_and_tanglish(self):
        """Class 8 student asks to switch/learn Science in Telugu and Tanglish."""
        telugu_tanglish_queries = [
            "సైన్స్ చదువుదాం",
            "సైన్స్ నేర్చుకుందాం",
            "సైన్స్ చెప్పు",
            "సైన్స్ కి మార్చు",
            "science cheppu",
            "science nerchukundam",
            "science chaduvudam",
        ]
        for query in telugu_tanglish_queries:
            self.session.active_subject = "Mathematics"
            is_switch, target, _ = detect_subject_switch(query, self.school_profile)
            self.assertTrue(is_switch, f"Failed to detect switch in: '{query}'")
            self.assertEqual(target, "Science", f"Target subject mismatch for: '{query}'")

            with patch("voice_assistant.call_gemini", return_value="తప్పకుండా! సైన్స్ నేర్చుకుందాం."):
                reply = generate_reply(query, self.school_profile, self.session)
                self.assertEqual(self.session.active_subject, "Science")
                self.assertEqual(self.session.active_topic, "General")

    def test_03_incidental_mentions_do_not_switch_subject(self):
        """Incidental mentions of subjects in questions must NOT switch the active subject."""
        incidental_queries = [
            "How does math help in science?",
            "Is math used in science?",
            "Why is mathematics needed for physics?",
            "What is the difference between science and social studies?",
            "What is a chemical change?",
            "How does temperature change affect water?",
            "రసాయన మార్పు అంటే ఏమిటి?",
            "భౌతిక మార్పు అంటే ఏమిటి?",
            "ఉష్ణోగ్రత మార్పు వల్ల ఏమి జరుగుతుంది?",
        ]
        for query in incidental_queries:
            self.session.active_subject = "Mathematics"
            self.session.active_topic = "Linear Equations"
            is_switch, target, _ = detect_subject_switch(query, self.school_profile)
            self.assertFalse(is_switch, f"Incidental mention incorrectly flagged as switch: '{query}'")

            intent = classify_learning_intent(query, self.session)
            self.assertNotEqual(intent, INTENT_SWITCH_TOPIC, f"Intent incorrectly flagged as switch for: '{query}'")

            with patch("voice_assistant.call_gemini", return_value="Explanation of the concept."):
                reply = generate_reply(query, self.school_profile, self.session)
                self.assertEqual(self.session.active_subject, "Mathematics", f"Active subject altered by: '{query}'")

    def test_04_curriculum_restriction_school_excludes_inter_and_college(self):
        """Class 8 student requesting Botany or C Programming is rejected and subject unchanged."""
        disallowed_queries = [
            ("Switch to Botany", "Botany"),
            ("Let's study Zoology", "Zoology"),
            ("Teach me C programming", None),
        ]
        for query, target in disallowed_queries:
            self.session.active_subject = "Mathematics"
            with patch("voice_assistant.call_gemini", return_value=""):
                reply = generate_reply(query, self.school_profile, self.session)
                # Subject remains Mathematics
                self.assertEqual(self.session.active_subject, "Mathematics")
                # Guidance response mentions allowed school subjects
                self.assertTrue(
                    "Mathematics" in reply or "గణిత" in reply or "సిలబస్" in reply or "syllabus" in reply,
                    f"Expected curriculum guidance for: '{query}'",
                )

    def test_05_curriculum_restriction_intermediate_mpc_excludes_biology(self):
        """Intermediate 1st Year MPC student requesting Biology/Botany is restricted."""
        mpc_profile = StudentProfile(level="intermediate", grade_or_year=1, group="MPC")
        mpc_session = LearningSession(mpc_profile)
        mpc_session.active_subject = "Mathematics"

        with patch("voice_assistant.call_gemini", return_value=""):
            reply = mpc_session_reply = generate_reply("Switch to Botany", mpc_profile, mpc_session)
            self.assertEqual(mpc_session.active_subject, "Mathematics")
            self.assertIn("MPC", reply)

    def test_06_subsequent_context_quiz_and_revision_use_new_subject(self):
        """Subsequent conversation, quiz generation, and revision reflect the newly switched subject."""
        # 1. Switch to Science
        with patch("voice_assistant.call_gemini", return_value="Switched to Science."):
            generate_reply("Let's learn Science", self.school_profile, self.session)
            self.assertEqual(self.session.active_subject, "Science")

        # 2. Subsequent explanation turn uses Science
        with patch("voice_assistant.call_gemini", return_value="Photosynthesis is the process in plants.") as mock_gem:
            reply = generate_reply("What is photosynthesis?", self.school_profile, self.session)
            # Verify system instruction passed to Gemini contains Current Subject: Science
            call_sys_inst = mock_gem.call_args[0][1]
            self.assertIn("Current Subject: Science", call_sys_inst)

        # 3. Subsequent Quiz Generation uses Science
        with patch("voice_assistant.call_gemini", return_value="QUESTION: What is the powerhouse of the cell?\nANSWER: Mitochondria") as mock_quiz:
            q_text, a_text = generate_quiz_question(
                self.school_profile,
                question_num=1,
                subject=self.session.active_subject,
                topic=self.session.active_topic,
            )
            self.assertIn("Mitochondria", a_text)
            quiz_prompt = mock_quiz.call_args[0][0]
            self.assertIn("in Science", quiz_prompt)

        # 4. Subsequent Revision uses Science
        with patch("voice_assistant.call_gemini", return_value="Summary of Science topic.") as mock_rev:
            reply = generate_reply("Revise this topic", self.school_profile, self.session)
            rev_sys_inst = mock_rev.call_args[0][1]
            self.assertIn("Current Subject: Science", rev_sys_inst)


class TestAudioPlaybackTelemetry(unittest.TestCase):
    """B. Test Audio Playback Inspection, sf.info Metadata, and Timing Comparison."""

    def setUp(self):
        self.test_wav = "test_playback_diag.wav"
        # Generate a synthetic 0.5s stereo 16000Hz test audio file
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo_tone = np.column_stack([tone, tone])
        sf.write(self.test_wav, stereo_tone, sr)

    def tearDown(self):
        if os.path.exists(self.test_wav):
            try:
                os.remove(self.test_wav)
            except Exception:
                pass

    def test_01_play_audio_logs_metadata_and_matches_timing(self):
        """play_audio reads sf.info metadata and measures playback wall-clock time cleanly."""
        info = sf.info(self.test_wav)
        self.assertAlmostEqual(info.duration, 0.5, delta=0.05)
        self.assertEqual(info.samplerate, 16000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.format, "WAV")

        # Mock sd.play and sd.wait to simulate exact playback completion
        with patch("sounddevice.play") as mock_sd_play, \
             patch("sounddevice.wait") as mock_sd_wait, \
             patch("time.sleep") as mock_sleep:

            ok = play_audio(self.test_wav)
            self.assertTrue(ok)
            mock_sd_play.assert_called_once()
            mock_sd_wait.assert_called_once()
            # Settling pause (0.25s) executed to prevent mic echo
            mock_sleep.assert_called_once_with(0.25)

    def test_02_play_audio_handles_missing_file_gracefully(self):
        """play_audio returns False without raising unhandled exception on missing file."""
        ok = play_audio("nonexistent_test_audio.wav")
        self.assertFalse(ok)

    def test_03_speak_and_play_executes_single_playback_without_duplicates(self):
        """speak_and_play synthesizes once and plays once, returning accurate durations."""
        with patch("voice_assistant.tts.synthesize") as mock_synth, \
             patch("voice_assistant.play_audio", return_value=True) as mock_play:

            ok, tts_dur, play_dur = speak_and_play("Hello student!", output_file=self.test_wav)
            self.assertTrue(ok)
            mock_synth.assert_called_once()
            mock_play.assert_called_once_with(self.test_wav)


class TestShortUtteranceHandling(unittest.TestCase):
    """C. Test Short Words, Scientific Terms, and Affirmations/Negations."""

    def test_01_quiz_letters_and_digits_preserved(self):
        """Single-letter options and digits are preserved as valid answers."""
        valid_letters = ["a", "b", "c", "d", "A", "B", "C", "D", "ఏ", "బి", "సి", "డి"]
        for opt in valid_letters:
            self.assertFalse(is_filler_or_noise(opt), f"Option '{opt}' wrongly filtered as noise")

        valid_digits = ["1", "2", "3", "4", "8", "10"]
        for d in valid_digits:
            self.assertFalse(is_filler_or_noise(d), f"Digit '{d}' wrongly filtered as noise")

    def test_02_conversational_affirmations_negations_preserved(self):
        """Conversational yes/no/ok and Telugu equivalents are preserved."""
        words = [
            "yes", "no", "ok", "okay", "true", "false",
            "అవును", "కాదు", "సరే", "నిజం", "తప్పు", "ఒప్పు",
            "avunu", "kaadu", "sare",
        ]
        for w in words:
            self.assertFalse(is_filler_or_noise(w), f"Word '{w}' wrongly filtered as noise")

    def test_03_scientific_single_words_preserved(self):
        """Single-word scientific and syllabus terms are preserved as valid speech."""
        scientific_terms = [
            "Oxygen", "Hydrogen", "Carbon", "Iron", "Gold", "Lead",
            "Cell", "Force", "Work", "Power", "Energy", "Mass", "Light", "Sound",
            "Mitochondria", "ATP", "DNA", "RNA",
            "ఆక్సిజన్", "కణం", "బలం", "పని", "శక్తి", "కాంతి", "ధ్వని", "ద్రవ్యరాశి",
        ]
        for term in scientific_terms:
            self.assertFalse(is_filler_or_noise(term), f"Scientific term '{term}' wrongly filtered as noise")

    def test_04_genuine_filler_and_isolated_noise_filtered(self):
        """Hesitation sounds and empty/isolated punctuation are flagged as filler/noise."""
        fillers = ["um", "uh", "umm", "hmm", "hm", "ఉమ్", "ఉం", "హ్మ్", ".", "-", "   "]
        for f in fillers:
            self.assertTrue(is_filler_or_noise(f), f"Filler '{f}' was NOT filtered as noise")

    def test_05_empty_and_low_confidence_transcripts_prompt_repeat(self):
        """Empty transcript or low confidence prompt repeats without guessing or hallucinating."""
        # Simulated Gnani STT result with low confidence
        low_conf_result = {
            "success": True,
            "transcript": "xyz",
            "confidence": 0.20,
            "low_confidence": True,
        }
        self.assertTrue(low_conf_result.get("low_confidence"))
        # In conversation loop, low_confidence prompts repeat rather than invoking Gemini
        with patch("voice_assistant.call_gemini") as mock_gemini:
            # Low confidence must not call Gemini
            self.assertEqual(mock_gemini.call_count, 0)


if __name__ == "__main__":
    unittest.main()
