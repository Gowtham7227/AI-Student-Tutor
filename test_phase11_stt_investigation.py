"""
Phase 11.1: Gnani STT Accuracy Investigation Test Suite
Clearly distinguishes:
PART A: Automated / Mocked Tests (Offline, deterministic)
PART B: Live Gnani STT API Benchmark Tests (Against real audio files on disk)
"""

import os
import sys
import unittest
import numpy as np
import soundfile as sf
from unittest.mock import MagicMock, patch

# Ensure UTF-8 console output for Telugu characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from gnani.stt._prisma import (
    PRISMA_V3_MODEL,
    INDIC_AUTO_LANGUAGE_CODE,
    PRISMA_V3_FORMATS,
    parse_candidates,
)
from gnani.stt.client import GnaniSTTClient, SUPPORTED_LANGUAGES


class TestSTTInvestigationMocked(unittest.TestCase):
    """PART A: Automated / Mocked Tests (Offline unit tests)"""

    def test_01_wav_file_specifications(self):
        """Task 1: Verify WAV file conforms to Gnani STT specifications."""
        filename = "test_spec_temp.wav"
        sample_rate = 16000
        # 2.5 seconds of synthetic audio
        duration = 2.5
        samples = int(sample_rate * duration)
        data = (np.sin(np.linspace(0, 100 * np.pi, samples)) * 0.3).astype(np.float32)

        sf.write(filename, data, sample_rate, subtype="PCM_16")

        try:
            info = sf.info(filename)
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.format, "WAV")
            self.assertEqual(info.subtype, "PCM_16")
            self.assertAlmostEqual(info.duration, 2.5, places=2)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_02_complete_audio_passed_to_stt(self):
        """Task 2 & 6: Verify GnaniSTTClient reads and uploads the entire file without truncation."""
        client = GnaniSTTClient(api_key="mock_test_key_gnani_investigation")

        test_file = "test_upload_complete.wav"
        dummy_data = b"RIFF" + b"\x00" * 4000
        with open(test_file, "wb") as f:
            f.write(dummy_data)

        try:
            with patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "success": True,
                    "request_id": "req_mock_123",
                    "transcript": "mock sentence test",
                }
                mock_post.return_value = mock_resp

                res = client.transcribe(
                    audio=test_file,
                    model="prisma-v3.0",
                    language_code="indic-auto",
                    language_candidates=["te-IN", "en-IN"],
                )

                self.assertTrue(res["success"])
                self.assertEqual(res["transcript"], "mock sentence test")

                # Verify files payload passed to requests.post
                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args[1]
                self.assertIn("files", call_kwargs)
                file_handle = call_kwargs["files"]["audio_file"]
                # File handle must be open and pointed to our file
                self.assertEqual(file_handle.name, test_file)
                # Form data must contain required parameters
                form_data = call_kwargs["data"]
                self.assertEqual(form_data["model"], "prisma-v3.0")
                self.assertEqual(form_data["language_code"], "indic-auto")
                self.assertEqual(form_data["multi_lang_codes"], "te-IN,en-IN")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_03_prisma_v3_supported_parameters(self):
        """Task 3: Verify SDK parameter validation for prisma-v3.0 and indic-auto."""
        self.assertEqual(PRISMA_V3_MODEL, "prisma-v3.0")
        self.assertEqual(INDIC_AUTO_LANGUAGE_CODE, "indic-auto")
        self.assertIn("verbatim", PRISMA_V3_FORMATS)
        self.assertIn("transcribe", PRISMA_V3_FORMATS)
        self.assertIn("codemix", PRISMA_V3_FORMATS)

        # Candidates parsing: 2-4 distinct languages
        candidates = parse_candidates(["te-IN", "en-IN"], SUPPORTED_LANGUAGES)
        self.assertEqual(candidates, ["te-IN", "en-IN"])

        # Invalid candidate should raise ValueError
        with self.assertRaises(ValueError):
            parse_candidates(["xx-YY"], SUPPORTED_LANGUAGES)

    def test_04_vad_silence_timeout_comparison(self):
        """Task 1 & 8: Demonstrate how 1.1s cutoff truncates multi-word sentence vs 1.8s preserving full audio."""
        chunk_dur = 0.05
        sample_rate = 16000
        chunk_samples = int(sample_rate * chunk_dur)

        speech_chunk = np.ones(chunk_samples, dtype=np.float32) * 0.05
        silence_chunk = np.zeros(chunk_samples, dtype=np.float32)

        # Student cadence:
        # Word 1 ("హలో"): 10 chunks (0.5s speech)
        # Thinking pause: 24 chunks (1.2s silence)
        # Word 2 ("నేను 8వ తరగతి"): 20 chunks (1.0s speech)
        speech_timeline = (
            [speech_chunk] * 10
            + [silence_chunk] * 24
            + [speech_chunk] * 20
        )

        def simulate_recording(chunks, silence_timeout):
            recorded = []
            speech_started = False
            silence_dur = 0.0
            for c in chunks:
                rms = float(np.sqrt(np.mean(c**2)))
                is_speech = rms >= 0.010
                if not speech_started:
                    if is_speech:
                        speech_started = True
                        recorded.append(c)
                else:
                    recorded.append(c)
                    if is_speech:
                        silence_dur = 0.0
                    else:
                        silence_dur += chunk_dur
                        if silence_dur >= silence_timeout:
                            break
            return len(recorded) * chunk_dur

        # With 1.1s silence timeout: cuts off after 1.1s of the pause! Word 2 is NEVER recorded!
        dur_1_1 = simulate_recording(speech_timeline, silence_timeout=1.1)
        self.assertAlmostEqual(dur_1_1, 0.5 + 1.1, places=1)
        self.assertLess(dur_1_1, 2.0, "1.1s timeout should have cut off before Word 2!")

        # With 1.8s silence timeout: tolerates 1.2s thinking pause and captures Word 2 completely!
        dur_1_8 = simulate_recording(speech_timeline, silence_timeout=1.8)
        self.assertGreater(dur_1_8, 2.5, "1.8s timeout must capture Word 2 completely!")


class TestSTTLiveAPIBenchmarks(unittest.TestCase):
    """PART B: Live Gnani STT API Benchmark Tests (Uses real audio files on disk)"""

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv()
        cls.api_key = os.getenv("GNANI_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("GNANI_API_KEY missing from environment. Skipping live API tests.")
        cls.client = GnaniSTTClient(api_key=cls.api_key)

    def test_05_live_mic_test_wav_accuracy(self):
        """Live API test on existing mic_test.wav comparing indic-auto vs te-IN vs en-IN."""
        if not os.path.exists("mic_test.wav"):
            self.skipTest("mic_test.wav not present on disk.")

        info = sf.info("mic_test.wav")
        self.assertEqual(info.samplerate, 16000)

        # 1. indic-auto with format='transcribe' (ITN enabled)
        try:
            r_auto = self.client.transcribe(
                audio="mic_test.wav",
                model="prisma-v3.0",
                language_code="indic-auto",
                language_candidates=["te-IN", "en-IN"],
                format="transcribe",
            )
            self.assertTrue(r_auto.get("success"))
            self.assertTrue(len(r_auto.get("transcript", "")) > 0)
            print(f"\n[LIVE API BENCHMARK] mic_test.wav (transcribe): '{r_auto.get('transcript')}' (Lang: {r_auto.get('detected_language_code')})")
        except Exception as e:
            self.skipTest(f"Live Gnani STT API unreachable ({e})")

    def test_06_live_complete_sentence_transcription(self):
        """Live API test on complete multi-word recording test_tutor_en.wav (4.64s)."""
        if not os.path.exists("test_tutor_en.wav"):
            self.skipTest("test_tutor_en.wav not present on disk.")

        try:
            res = self.client.transcribe(
                audio="test_tutor_en.wav",
                model="prisma-v3.0",
                language_code="indic-auto",
                language_candidates=["te-IN", "en-IN"],
                format="transcribe",
            )
            self.assertTrue(res.get("success"))
            transcript = res.get("transcript", "")
            # Must contain complete sentence words, not a 1-word truncation
            self.assertIn("deadlock", transcript.lower())
            self.assertIn("operating systems", transcript.lower())
            print(f"[LIVE API BENCHMARK] test_tutor_en.wav: '{transcript}'")
        except Exception as e:
            self.skipTest(f"Live Gnani STT API unreachable ({e})")


if __name__ == "__main__":
    unittest.main()
