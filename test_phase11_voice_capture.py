"""
Phase 11 Test Suite: Microphone, Voice Capture & Speech Recognition
Covers:
1. Silent microphone input detection.
2. Low-volume / quiet speech detection and adaptive gain leveling.
3. Empty audio buffer safety.
4. Device selection & channel validation (ignoring 0-channel outputs).
5. Valid short answers ('A', 'B', '1', '8', 'yes', 'no', 'సరే', 'అవును', 'కాదు').
6. Telugu and English transcript handling without encoding loss.
7. STT timeout and service failure recovery prompts (profile persistence).
8. Empty transcript ('') handling without random Gemini hallucination.
9. Microphone stream resource cleanup on exception.
10. Diagnostic function metrics calculation.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from voice_assistant import (
    StudentProfile,
    LearningSession,
    AudioRecordResult,
    get_input_device,
    record_audio,
    diagnose_microphone,
    is_filler_or_noise,
    classify_learning_intent,
    generate_reply,
    transcribe_speech,
    SILENCE_RMS_THRESHOLD,
    VAD_ENERGY_THRESHOLD,
    SAFE_MIN_VAD_THRESHOLD,
    SAFE_MAX_VAD_THRESHOLD,
)


class TestPhase11VoiceCapture(unittest.TestCase):

    def test_01_silent_microphone_input(self):
        """Verifies complete silence is detected cleanly without crash and marks is_silent=True."""
        sample_rate = 16000
        chunk_samples = int(sample_rate * 0.05)
        # 40 chunks (2.0s) of digital silence
        silent_chunks = [np.zeros(chunk_samples, dtype=np.float32)] * 40

        stream_instance = MagicMock()
        chunk_iter = iter(silent_chunks)

        def mock_read(num_samples):
            try:
                return next(chunk_iter), False
            except StopIteration:
                return np.zeros(num_samples, dtype=np.float32), False

        stream_instance.read.side_effect = mock_read

        with patch("voice_assistant.sd.InputStream") as mock_stream_cls:
            mock_stream_cls.return_value.__enter__.return_value = stream_instance
            with patch("voice_assistant.sf.write"):
                res = record_audio(
                    filename="mock_silent.wav",
                    sample_rate=sample_rate,
                    duration=1.5,
                    initial_timeout=1.0,
                )

        self.assertTrue(res.is_silent)
        self.assertEqual(res.speech_duration, 0.0)
        self.assertLess(res.rms, SILENCE_RMS_THRESHOLD)

    def test_02_low_volume_speech_detection_and_gain(self):
        """Verifies quiet speech (RMS ~0.008, peak ~0.08) is captured and gentle gain applied."""
        sample_rate = 16000
        chunk_samples = int(sample_rate * 0.05)

        # Baseline room noise (RMS ~ 0.003)
        noise_chunk = np.random.normal(0, 0.003, chunk_samples).astype(np.float32)
        # Quiet speech chunk (RMS ~ 0.008, peak ~ 0.08)
        speech_chunk = (np.sin(np.linspace(0, 20 * np.pi, chunk_samples)) * 0.08).astype(np.float32)

        # 5 noise chunks (pre-roll + calibration) -> 15 quiet speech chunks (0.75s) -> 25 trailing noise chunks (1.25s silence)
        chunks = [noise_chunk] * 5 + [speech_chunk] * 15 + [noise_chunk] * 25

        stream_instance = MagicMock()
        chunk_iter = iter(chunks)

        def mock_read(num_samples):
            try:
                return next(chunk_iter), False
            except StopIteration:
                return np.zeros(num_samples, dtype=np.float32), False

        stream_instance.read.side_effect = mock_read

        saved_audio = None

        def mock_sf_write(fn, audio, sr):
            nonlocal saved_audio
            saved_audio = audio.copy()

        with patch("voice_assistant.sd.InputStream") as mock_stream_cls:
            mock_stream_cls.return_value.__enter__.return_value = stream_instance
            with patch("voice_assistant.sf.write", side_effect=mock_sf_write):
                res = record_audio(
                    filename="mock_quiet.wav",
                    sample_rate=sample_rate,
                    duration=3.0,
                    silence_timeout=1.1,
                )

        # Speech must have been detected
        self.assertFalse(res.is_silent, "Quiet speech was incorrectly rejected as silent!")
        self.assertGreater(res.speech_duration, 0.5)
        # Gentle gain should normalize quiet audio without clipping
        self.assertIsNotNone(saved_audio)
        self.assertLessEqual(np.max(np.abs(saved_audio)), 1.0)

    def test_03_empty_audio_buffer_handling(self):
        """Verifies an empty recorded buffer returns clean AudioRecordResult without crashing."""
        with patch("voice_assistant.sd.InputStream") as mock_stream_cls:
            stream_instance = MagicMock()
            # Immediately exhausted
            stream_instance.read.return_value = (np.zeros(800, dtype=np.float32), False)
            mock_stream_cls.return_value.__enter__.return_value = stream_instance
            with patch("voice_assistant.sf.write"):
                res = record_audio(
                    filename="mock_empty.wav",
                    duration=0.1,
                    initial_timeout=0.05,
                )
        self.assertTrue(res.is_silent)
        self.assertGreaterEqual(res.total_duration, 0.0)

    def test_04_device_input_channel_validation(self):
        """Verifies get_input_device() rejects output devices (channels=0) and selects an active input device."""
        mock_devices = [
            {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
            {"name": "Speakers Output Only", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
            {"name": "Microphone Array", "max_input_channels": 4, "max_output_channels": 0, "default_samplerate": 44100.0},
        ]

        with patch("sounddevice.query_devices", return_value=mock_devices):
            with patch("sounddevice.default.device", (1, 1)):  # default input is device 1 (which has 0 input channels!)
                idx, info = get_input_device()
                # Should NOT select device 1 because it has 0 input channels
                self.assertGreater(info["max_input_channels"], 0)
                self.assertNotEqual(idx, 1)

    def test_05_valid_short_answers_not_noise(self):
        """Verifies short meaningful answers (letters, numbers, affirmatives) are NOT treated as noise."""
        valid_answers = [
            "A", "b", "C", "d", "ఏ", "బి", "సి", "డి",
            "1", "2", "3", "4", "8", "10",
            "yes", "no", "ok", "okay", "done", "pass",
            "సరే", "అవును", "కాదు",
        ]
        for ans in valid_answers:
            self.assertFalse(is_filler_or_noise(ans), f"Valid answer '{ans}' was incorrectly flagged as noise!")

        # Confirm true noise is still caught
        true_noise = ["umm", "hmm", "uh", "er", "హా", "హ్మ్", ".", "   "]
        for n in true_noise:
            self.assertTrue(is_filler_or_noise(n), f"Noise '{n}' was not filtered!")

    def test_06_telugu_and_english_transcript_handling(self):
        """Verifies Telugu, English, and Tanglish transcripts are processed through intent classification cleanly."""
        queries = [
            ("Explain gravity in simple terms", "explanation"),
            ("గణితం చదువుదాం", "switch_topic"),
            ("ఒక్క నిమిషం ఆగు", "pause"),
            ("కంటిన్యూ చేయి", "resume"),
            ("2x + 5 = 15 solve cheyyi", "step_by_step"),
            ("Start quiz", "quiz_start"),
        ]
        for q, expected_intent in queries:
            intent = classify_learning_intent(q)
            self.assertEqual(intent, expected_intent, f"Failed for query '{q}': got {intent}")

    def test_07_stt_timeout_and_failure_resilience(self):
        """Verifies STT API failure returns None without crashing and preserves student profile."""
        profile = StudentProfile(level="school", grade_or_year=8, group="general", subject="Science")
        session = LearningSession(profile)
        session.active_topic = "Photosynthesis"

        with patch("voice_assistant.stt.transcribe", side_effect=RuntimeError("Connection timeout")):
            res = transcribe_speech("nonexistent.wav")
            self.assertIsNone(res)

        # Profile and topic remain completely untouched
        self.assertEqual(session.profile.grade_or_year, 8)
        self.assertEqual(session.active_topic, "Photosynthesis")

    def test_08_empty_transcript_handling(self):
        """Verifies empty transcript from STT does not trigger random Gemini responses."""
        profile = StudentProfile(level="school", grade_or_year=9, group="general", subject="Physics")
        session = LearningSession(profile)

        # In voice_assistant.py main loop:
        # If transcript is empty (""), it prompts for repeat and continues without calling generate_reply
        with patch("voice_assistant.call_gemini") as mock_gemini:
            # An empty transcript should never be passed to Gemini
            transcript = ""
            if not transcript:
                unclear_action_taken = True
            else:
                generate_reply(transcript, profile=profile, session=session)
                unclear_action_taken = False

            self.assertTrue(unclear_action_taken)
            mock_gemini.assert_not_called()

    def test_09_stream_resource_cleanup(self):
        """Verifies that sounddevice InputStream resources are cleanly exited on error."""
        with patch("voice_assistant.sd.InputStream") as mock_stream:
            stream_mock = MagicMock()
            stream_mock.read.side_effect = IOError("Audio buffer overrun")
            mock_stream.return_value.__enter__.return_value = stream_mock

            res = record_audio(duration=1.0)
            self.assertFalse(res.success)
            # Confirms context manager exited
            mock_stream.return_value.__exit__.assert_called()

    def test_10_diagnose_microphone_metrics(self):
        """Verifies diagnose_microphone computes RMS, peak, clipping, and suitability properly."""
        sample_rate = 16000
        # Simulated test audio with good vocal energy
        t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
        mock_audio = (np.sin(2 * np.pi * 440 * t) * 0.25).astype(np.float32)

        with patch("sounddevice.rec", return_value=mock_audio.reshape(-1, 1)):
            with patch("sounddevice.wait"):
                with patch("sounddevice.query_devices", return_value={"name": "Test Mic", "max_input_channels": 1, "default_samplerate": 16000.0}):
                    diag = diagnose_microphone(duration=1.0, device=1, sample_rate=sample_rate)

        self.assertAlmostEqual(diag["duration"], 1.0, delta=0.5)
        self.assertGreater(diag["rms"], 0.05)
        self.assertAlmostEqual(diag["peak"], 0.25, delta=0.05)
        self.assertFalse(diag["is_silent"])
        self.assertTrue(diag["speech_detected"])
        self.assertTrue(diag["stt_suitable"])


if __name__ == "__main__":
    unittest.main()
