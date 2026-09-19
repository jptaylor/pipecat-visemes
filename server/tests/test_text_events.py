"""Causal text priors, known-phone controls, and exact DSP fallback contracts."""

import unittest
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame

from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.frames import LipsyncUpdateSettingsFrame
from lipsync.lipsync_processor import LipsyncParams, LipsyncProcessor
from lipsync.pronunciation import load_lexicon, pronounce, supported_text, tokenize
from lipsync.text_events import TextEvents
from lipsync.text_prior import TextAnchor, TextPrior, TextWord
from lipsync.types import LipsyncEventKind
from tests.synth import synth_vowel
from tests.test_text_prior import anchor, word


def prior(text, *words, **kwargs):
    return TextPrior(anchors=(TextAnchor(text),), words=tuple(words), word_start_pts=0, **kwargs)


class TestPronunciation(unittest.TestCase):
    def test_packed_dictionary_and_hums(self):
        self.assertIs(load_lexicon(), load_lexicon())
        self.assertEqual(pronounce("bob").phones, ("B", "AA", "B"))
        self.assertEqual(pronounce("mama").phones, ("M", "AA", "M", "AH"))
        for text in ("hmm", "hmmmm", "mmm"):
            self.assertEqual(pronounce(text).phones, ("M",))
            self.assertTrue(pronounce(text).certain)
        self.assertFalse(pronounce("zxqvflorp").certain)
        self.assertTrue(pronounce("zxqvflorp").phones)

    def test_normalization_is_not_tts_number_or_markup_expansion(self):
        self.assertEqual(tokenize("It's Mama’s!"), ("it's", "mama's"))
        for text in ("Pay $5", "42", "<break/>", "café", "你好"):
            self.assertFalse(supported_text(text))
        self.assertTrue(supported_text('"Who knew?"'))


class TestTextEvents(unittest.TestCase):
    def test_negative_control_has_no_bilabial_or_nasal(self):
        model = TextEvents()
        model.prepare(prior("See the stars as they rise."))
        self.assertFalse(model.allow_closure(0.2))
        self.assertEqual(
            model.hint(0.2, voiced=True, f1=250, rms=0.2, silence_gate=0.001), (False, None)
        )
        self.assertFalse(model.inject_closures(0.2, 0.2, True))

    def test_timely_word_intervals_and_deduplication(self):
        model = TextEvents()
        model.prepare(prior("Bob sees.", TextWord("Bob", 0), TextWord("sees", 300_000_000)))
        events = []
        for t in np.arange(0.0125, 0.30, 0.02):
            events.extend(model.inject_closures(float(t), 0.2, True))
        self.assertEqual(len(events), 2)  # B AA B, two closures, no energy dip needed.
        self.assertAlmostEqual(events[0].offset, 0.0)
        self.assertAlmostEqual(events[1].offset, 0.2)
        self.assertFalse(model.allow_closure(0.2))

    def test_late_word_cannot_veto_already_analyzed_dsp_event(self):
        model = TextEvents()
        model.prepare(prior("Bob sees.", TextWord("Bob", 0), TextWord("sees", 300_000_000)), 0.5)
        self.assertTrue(model.allow_closure(0.15))
        self.assertFalse(model.inject_closures(0.51, 0.2, True))
        self.assertEqual(model.stats["late_closures"], 2)

    def test_later_snapshot_does_not_reclassify_timely_spans_as_late(self):
        model = TextEvents()
        words = (TextWord("Bob", 0), TextWord("sees", 300_000_000))
        model.prepare(prior("Bob sees.", *words))
        model.prepare(prior("Bob sees.", *words, audio_end=0.6), cursor=0.5)
        self.assertEqual(model.stats["late_closures"], 0)

    def test_equal_timestamps_are_grouped_without_zero_duration_words(self):
        model = TextEvents()
        model.prepare(
            prior(
                "My mother sees.",
                TextWord("my", 0),
                TextWord("mother", 0),
                TextWord("sees", 600_000_000),
            )
        )
        self.assertEqual(model.span_at(0.01).phone, "M")
        self.assertEqual(model.span_at(0.21).phone, "M")
        self.assertIsNone(model.span_at(0.65))  # last word duration is not invented

    def test_no_injected_events_in_silence(self):
        model = TextEvents()
        model.prepare(prior("Bob sees.", TextWord("Bob", 0), TextWord("sees", 300_000_000)))
        for t in np.arange(0.0125, 0.30, 0.02):
            self.assertFalse(model.inject_closures(float(t), 0, False))

    def test_contradictory_word_text_disables_the_context(self):
        model = TextEvents()
        model.prepare(prior("Mama.", TextWord("hello", 0)))
        self.assertIsNone(model.segment_at(0.2))
        self.assertEqual(model.stats["mismatched_contexts"], 1)

    def test_unknown_clock_keeps_only_inventory(self):
        model = TextEvents()
        model.prepare(
            prior("Bob sees.", TextWord("Bob", -100_000_000), TextWord("sees", 300_000_000))
        )
        self.assertIsNone(model.span_at(0.1))
        self.assertTrue(model.allow_closure(0.1))

    def test_open_vowel_contradicts_hum_and_n_does_not_force_lips_shut(self):
        model = TextEvents()
        model.prepare(prior("Hmm."))
        self.assertEqual(
            model.hint(0.2, voiced=True, f1=800, rms=0.2, silence_gate=0.001), (None, None)
        )
        model.prepare(prior("No sees.", TextWord("No", 0), TextWord("sees", 300_000_000)))
        self.assertEqual(
            model.hint(0.02, voiced=True, f1=250, rms=0.2, silence_gate=0.001), (None, "N")
        )

    def test_reset_clears_claimed_spans_and_rejection(self):
        model = TextEvents()
        good = prior("Bob sees.", TextWord("Bob", 0), TextWord("sees", 300_000_000))
        model.prepare(good)
        self.assertTrue(model.inject_closures(0.01, 0.2, True))
        model.reset()
        model.prepare(good)
        self.assertTrue(model.inject_closures(0.01, 0.2, True))


async def analyze(pcm, text=None, *, enabled=False):
    analyzer = FormantLipsyncAnalyzer(text_events_enabled=enabled)
    await analyzer.start(16_000)
    context = LipsyncAnalysisContext("test", 16_000, text_prior=text)
    keys, events = [], []
    for i in range(0, len(pcm), 320):
        result = await analyzer.analyze(pcm[i : i + 320], context)
        keys.extend(result.keyframes)
        events.extend(result.events)
    result = await analyzer.flush(context)
    keys.extend(result.keyframes)
    events.extend(result.events)
    return [asdict(k) for k in keys], [asdict(e) for e in events]


class TestTextAnalyzer(unittest.IsolatedAsyncioTestCase):
    async def test_disabling_lipsync_does_not_reset_upstream_word_clock(self):
        processor = LipsyncProcessor(params=LipsyncParams(text_events_enabled=True))
        processor._handle_tts_text(word("previous", ctx="old", pts=2_000_000_000))
        await processor._handle_update_settings(
            LipsyncUpdateSettingsFrame(settings={"enabled": False})
        )
        self.assertEqual(processor._last_word_pts, 2_000_000_000)

    async def test_exact_fallback_for_missing_untrusted_late_and_inconsistent_text(self):
        pcm = synth_vowel(280, 900, secs=0.6)
        expected = await analyze(pcm)
        variants = [
            None,
            prior("zxqvflorp."),
            prior("Pay $5."),
            prior("Mama.", TextWord("wrong", 0)),
            TextPrior(words=(TextWord("Mama", 0),)),
            TextPrior(anchors=(TextAnchor("Mama.", received_after_audio=0.4),)),
            TextPrior(anchors=(TextAnchor("Mama."), TextAnchor("Bob."))),
            prior("Bob " * 257),
            prior("x" * 65),
        ]
        for text in variants:
            with self.subTest(text=text):
                self.assertEqual(await analyze(pcm, text, enabled=True), expected)

    async def test_hum_override_and_silence_control(self):
        pcm = synth_vowel(250, 1500, secs=0.6)
        keys, events = await analyze(pcm, prior("Hmm."), enabled=True)
        self.assertTrue(any(e["kind"] == LipsyncEventKind.NASAL for e in events))
        settled = [k["openness"] for k in keys if k["offset"] >= 0.08]
        self.assertTrue(settled)
        self.assertLessEqual(max(settled), 0.15)
        _, events = await analyze(np.zeros(9600, dtype=np.float32), prior("Hmm."), enabled=True)
        self.assertFalse(any(e["kind"] != LipsyncEventKind.SILENCE for e in events))

    async def test_disabling_switch_preserves_all_output_even_with_text(self):
        pcm = synth_vowel(700, 1200, secs=0.6)
        self.assertEqual(await analyze(pcm, prior("Hmm.")), await analyze(pcm))

    async def test_word_clock_is_not_sentence_pts_or_queued_playout(self):
        processor = LipsyncProcessor(params=LipsyncParams(text_events_enabled=True))
        processor.get_clock = Mock(return_value=Mock(get_time=Mock(return_value=1_000_000_000)))
        processor._last_playout_end = 9_000_000_000
        processor._handle_tts_text(word("previous", ctx="old", pts=2_000_000_000))
        processor._handle_tts_text(anchor(pts=100))
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        processor._handle_tts_text(word("Mama", pts=2_100_000_000))  # flushed before first audio
        processor._handle_tts_audio(
            TTSAudioRawFrame(
                audio=b"\0" * 640, sample_rate=16_000, num_channels=1, context_id="ctx"
            )
        )
        context = processor._contexts[0]
        processor._snapshot_text(context)
        self.assertEqual(context.analysis.text_prior.word_start_pts, 2_000_000_000)
        self.assertEqual(context.analysis.text_prior.playout_start_pts, 9_000_000_000)
        await processor._handle_interruption()
        self.assertEqual(processor._last_word_pts, 0)
