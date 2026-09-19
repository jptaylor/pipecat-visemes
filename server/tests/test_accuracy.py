"""A/B provenance and the untimed corpus text-input contract."""

import unittest

import numpy as np

from benchmarks.accuracy import analyze_clip, output_digest, validate_comparison
from benchmarks.common import Clip, Sentence, Voice
from tests.synth import synth_vowel


class TestAccuracy(unittest.IsolatedAsyncioTestCase):
    async def test_untimed_prior_has_byte_identical_output(self):
        pcm = (synth_vowel(700, 1200, secs=0.5) * 32767).astype(np.int16).tobytes()
        clip = Clip(
            Sentence("vowel", "Father.", [], {}), Voice("test", "test", 5000), 1, pcm, 16000
        )
        before, events_before, _ = await analyze_clip(clip)
        after, events_after, _ = await analyze_clip(clip, text_prior=True)
        self.assertTrue(before)
        self.assertEqual(output_digest(before, events_before), output_digest(after, events_after))

    def test_comparison_rejects_different_corpora_but_allows_text_and_dsp_changes(self):
        run = {
            "provider": "cartesia",
            "voices": ["v1", "v2"],
            "sentences": ["a", "b"],
            "takes": 2,
            "warm": False,
            "ceiling_override": None,
        }
        for key, value in (
            ("provider", "deepgram"),
            ("voices", ["v1"]),
            ("sentences", ["b"]),
            ("takes", 1),
            ("warm", True),
            ("ceiling_override", 4500),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_comparison(run, {"run": {**run, key: value}})
        validate_comparison(
            {**run, "text_prior": True, "overrides": {"dsp.LPC_ORDER": 14}},
            {"run": {**run, "voices": ["v2", "v1"], "text_prior": False}},
        )
