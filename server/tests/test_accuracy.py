"""A/B provenance and the untimed corpus text-input contract."""

import unittest

import numpy as np

from benchmarks.accuracy import (
    analyze_clip,
    output_digest,
    validate_audio_comparison,
    validate_comparison,
)
from benchmarks.common import Clip, Sentence, Voice
from benchmarks.text_timing import FixtureTextInputs
from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
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
            ("fixtures", "different-audio"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_comparison(run, {"run": {**run, key: value}})
        validate_comparison(
            {**run, "text_prior": True, "overrides": {"dsp.LPC_ORDER": 14}},
            {"run": {**run, "voices": ["v2", "v1"], "text_prior": False}},
        )

    def test_refreshed_pcm_cannot_pass_as_a_paired_comparison(self):
        with self.assertRaisesRegex(ValueError, "changed PCM"):
            validate_audio_comparison(
                {"clips": [{"label": "a", "pcm_sha256": "new"}]},
                {"clips": [{"label": "a", "pcm_sha256": "old"}]},
            )

    def test_fixture_words_are_not_exposed_before_their_captured_arrival(self):
        clip = Clip(
            Sentence("test", "Bob sees.", [], {}),
            Voice("test", "test", 5000),
            1,
            b"\0" * 32000,
            16000,
            {
                "anchors": [["Bob sees.", -0.2, 0]],
                "words": [["Bob", 0.1, 0.354], ["sees", 0.4, 0.554]],
            },
        )
        context = LipsyncAnalysisContext("test", 16000)
        inputs = FixtureTextInputs(clip, context)
        inputs.ingest(320)
        first = context.text_prior
        self.assertEqual(first.words, ())
        inputs.ingest(640)
        self.assertIs(context.text_prior, first)
        inputs.ingest(5760)
        self.assertEqual([w.text for w in context.text_prior.words], ["Bob"])
        self.assertEqual(context.text_prior.words[0].pts, 100_000_000)
        self.assertEqual(context.text_prior.word_start_pts, 0)
        self.assertIsNone(context.text_prior.audio_end)
        inputs.ingest(16000, final=True)
        self.assertEqual(context.text_prior.audio_end, 1.0)
        clip.text_timing = {"anchors": [], "words": []}
        FixtureTextInputs(clip, context).ingest(16000)
        self.assertIsNone(context.text_prior)  # explicit absence is not invented text
