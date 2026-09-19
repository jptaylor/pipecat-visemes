"""Text-input lifecycle and the DSP-only regression contract."""

import unittest
from dataclasses import FrozenInstanceError, asdict

from pipecat.frames.frames import (
    AggregatedTextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.tests.utils import run_test

from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.frames import LipsyncUpdateSettingsFrame
from lipsync.lipsync_processor import _MAX_CONTEXTS, LipsyncParams, LipsyncProcessor
from lipsync.text_prior import _MAX_TEXT_CHARS, _MAX_TEXT_RECORDS, TextAnchor, _TextAccumulator
from tests.synth import synth_vowel
from tests.test_lipsync_processor import lipsync_frames, make_tts_frames


def anchor(text="Mama made more.", ctx="ctx", pts=None):
    frame = AggregatedTextFrame(text, aggregated_by="sentence", context_id=ctx)
    frame.will_be_spoken = True
    frame.pts = pts
    return frame


def word(text="Mama", ctx="ctx", pts=0):
    frame = TTSTextFrame(text, aggregated_by="word", context_id=ctx)
    frame.pts = pts
    return frame


def observer():
    return LipsyncProcessor(params=LipsyncParams(text_prior_enabled=True))


class TestTextPrior(unittest.TestCase):
    def test_early_anchor_and_words_attach_once_without_subclass_confusion(self):
        processor = observer()
        processor._handle_tts_text(anchor(pts=100))
        processor._handle_tts_text(word(pts=0))  # zero is not missing
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        context = processor._contexts[0]
        processor._snapshot_text(context)
        prior = context.analysis.text_prior
        self.assertEqual([a.text for a in prior.anchors], ["Mama made more."])
        self.assertEqual([(w.text, w.pts) for w in prior.words], [("Mama", 0)])
        self.assertEqual(prior.anchors[0].pts, 100)
        self.assertFalse(processor._pending_text)

    def test_only_spoken_sentences_and_timed_word_frames_are_observed(self):
        processor = observer()
        unspoken = anchor()
        unspoken.will_be_spoken = False
        token = anchor()
        token.aggregated_by = "token"
        completion = TTSTextFrame("Mama made more.", aggregated_by="sentence", context_id="ctx")
        completion.will_be_spoken = True
        for frame in (unspoken, token, completion, word(pts=None), word(text="  ")):
            processor._handle_tts_text(frame)
        self.assertFalse(processor._pending_text)
        self.assertEqual(processor.stats["text_anchors"], 0)
        self.assertEqual(processor.stats["text_words"], 0)

    def test_late_anchors_and_queued_contexts_preserve_observations_not_fake_onsets(self):
        processor = observer()
        processor._handle_tts_text(anchor("First.", pts=100))
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        context = processor._contexts[0]
        context.sample_rate = 16_000
        context.analysis.samples_seen = 12_800
        context.t0 = 9_000_000_000  # queued behind other audio
        processor._snapshot_text(context)
        previous = context.analysis.text_prior
        processor._handle_tts_text(anchor("Second.", pts=200))
        processor._handle_tts_text(word("Second", pts=1_000_000_000))
        processor._snapshot_text(context)
        current = context.analysis.text_prior
        self.assertEqual(len(previous.anchors), 1)
        self.assertEqual(len(current.anchors), 2)
        self.assertEqual(current.anchors[1].received_after_audio, 0.8)
        self.assertEqual(current.words[0].received_after_audio, 0.8)
        self.assertEqual(current.words[0].pts, 1_000_000_000)
        self.assertEqual(current.playout_start_pts, 9_000_000_000)
        with self.assertRaises(FrozenInstanceError):
            current.anchors = ()
        processor._snapshot_text(context)
        self.assertIs(context.analysis.text_prior, current)

    def test_no_text_and_disabled_collection_remain_none(self):
        for processor in (LipsyncProcessor(), observer()):
            processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
            context = processor._contexts[0]
            processor._snapshot_text(context)
            self.assertIsNone(context.analysis.text_prior)
        processor = LipsyncProcessor()
        processor._handle_tts_text(anchor())
        self.assertFalse(processor._pending_text)
        self.assertEqual(processor.stats["text_anchors"], 0)

    def test_pending_contexts_are_bounded_and_stopped_orphans_are_removed(self):
        processor = observer()
        for n in range(_MAX_CONTEXTS + 2):
            processor._handle_tts_text(anchor(ctx=str(n)))
        self.assertEqual(len(processor._pending_text), _MAX_CONTEXTS)
        self.assertNotIn("0", processor._pending_text)
        self.assertNotIn("1", processor._pending_text)
        processor._handle_tts_stopped(TTSStoppedFrame(context_id="2"))
        self.assertNotIn("2", processor._pending_text)
        processor._handle_tts_started(TTSStartedFrame(context_id="0"))
        processor._snapshot_text(processor._contexts[0])
        self.assertIsNone(processor._contexts[0].analysis.text_prior)

    def test_reopening_context_does_not_inherit_closed_text(self):
        processor = observer()
        processor._handle_tts_text(anchor("Old."))
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        processor._handle_tts_stopped(TTSStoppedFrame(context_id="ctx"))
        processor._handle_tts_text(word("Stale"))
        self.assertFalse(processor._pending_text)
        processor._handle_tts_text(anchor("New."))
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        for context in processor._contexts:
            processor._snapshot_text(context)
        self.assertEqual(
            [c.analysis.text_prior.anchors[0].text for c in processor._contexts], ["Old.", "New."]
        )
        self.assertFalse(processor._contexts[1].analysis.text_prior.words)

    def test_oversized_context_drops_whole_prior_and_stays_bounded(self):
        text = _TextAccumulator()
        text.append(TextAnchor("First."))
        old = text.snapshot(None)
        self.assertFalse(text.append(TextAnchor("x" * _MAX_TEXT_CHARS)))
        self.assertIsNone(text.snapshot(None))
        self.assertFalse(text.append(TextAnchor("Later.")))
        self.assertEqual(old.anchors[0].text, "First.")
        self.assertFalse(text.anchors)
        self.assertFalse(text.words)
        text = _TextAccumulator()
        for _ in range(_MAX_TEXT_RECORDS):
            self.assertTrue(text.append(TextAnchor("x")))
        self.assertFalse(text.append(TextAnchor("x")))
        self.assertIsNone(text.snapshot(None))


class _ObservingAnalyzer(FormantLipsyncAnalyzer):
    def __init__(self):
        super().__init__()
        self.priors = []

    async def analyze(self, pcm, context):
        self.priors.append(context.text_prior)
        return await super().analyze(pcm, context)


class TestTextPriorPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_observation_does_not_change_output_or_forwarded_frames(self):
        outputs = []
        for enabled in (False, True):
            analyzer = _ObservingAnalyzer()
            processor = LipsyncProcessor(
                params=LipsyncParams(text_prior_enabled=enabled), analyzer=analyzer
            )
            audio = make_tts_frames(synth_vowel(700, 1200, secs=0.8), "ctx")
            frames = [anchor(), audio[0], word(), *audio[1:]]
            received, _ = await run_test(processor, frames_to_send=frames)
            sent = {f.id for f in frames}
            self.assertEqual([f.id for f in received if f.id in sent], [f.id for f in frames])
            batches = lipsync_frames(received)
            outputs.append(
                (
                    [asdict(k) for f in batches for k in f.keyframes],
                    [asdict(e) for f in batches for e in f.events],
                )
            )
            self.assertTrue(analyzer.priors)
            if enabled:
                self.assertTrue(any(p and p.anchors and p.words for p in analyzer.priors))
            else:
                self.assertTrue(all(p is None for p in analyzer.priors))
        self.assertEqual(outputs[0], outputs[1])

    async def test_interruption_cancellation_and_end_clear_parked_text(self):
        for method in ("_handle_interruption", "_cancel", "_stop"):
            processor = observer()
            processor._handle_tts_text(anchor())
            await getattr(processor, method)()
            self.assertFalse(processor._pending_text)
        processor = observer()
        processor._handle_tts_text(anchor())
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        await processor._handle_interruption()
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        processor._snapshot_text(processor._contexts[0])
        self.assertIsNone(processor._contexts[0].analysis.text_prior)

    async def test_disabling_drops_priors_and_reenabling_waits_for_next_context(self):
        processor = observer()
        processor._handle_tts_text(anchor())
        processor._handle_tts_started(TTSStartedFrame(context_id="ctx"))
        context = processor._contexts[0]
        processor._snapshot_text(context)
        self.assertIsNotNone(context.analysis.text_prior)
        await processor._handle_update_settings(
            LipsyncUpdateSettingsFrame(settings={"text_prior_enabled": False})
        )
        self.assertIsNone(context.text)
        self.assertIsNone(context.analysis.text_prior)
        await processor._handle_update_settings(
            LipsyncUpdateSettingsFrame(settings={"text_prior_enabled": True})
        )
        processor._handle_tts_text(anchor("Too late."))
        processor._snapshot_text(context)
        self.assertIsNone(context.analysis.text_prior)
        processor._handle_tts_text(anchor("Next.", ctx="next"))
        processor._handle_tts_started(TTSStartedFrame(context_id="next"))
        processor._snapshot_text(processor._contexts[1])
        self.assertEqual(processor._contexts[1].analysis.text_prior.anchors[0].text, "Next.")
