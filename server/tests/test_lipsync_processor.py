#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import numpy as np
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.tests.utils import SleepFrame, run_test

from lipsync.base_lipsync_analyzer import (
    BaseLipsyncAnalyzer,
    LipsyncFrameResult,
)
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.frames import LipsyncUpdateSettingsFrame, TTSLipsyncFrame
from lipsync.lipsync_processor import LipsyncParams, LipsyncProcessor
from lipsync.rtvi import LipsyncMessageRelay
from lipsync.types import LipsyncEvent, LipsyncEventKind, LipsyncKeyframe
from tests.synth import synth_vowel

SAMPLE_RATE = 16000


def make_tts_frames(pcm, context_id, chunk_ms=10, sample_rate=SAMPLE_RATE) -> list[Frame]:
    """TTSStartedFrame + audio chunks + TTSStoppedFrame for one utterance."""
    data = (pcm * 32767).astype(np.int16).tobytes()
    chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
    frames: list[Frame] = [TTSStartedFrame(context_id=context_id)]
    for i in range(0, len(data), chunk_bytes):
        frames.append(
            TTSAudioRawFrame(
                audio=data[i : i + chunk_bytes],
                sample_rate=sample_rate,
                num_channels=1,
                context_id=context_id,
            )
        )
    frames.append(TTSStoppedFrame(context_id=context_id))
    return frames


def lipsync_frames(received, context_id=None) -> list[TTSLipsyncFrame]:
    frames = [f for f in received if isinstance(f, TTSLipsyncFrame)]
    if context_id is not None:
        frames = [f for f in frames if f.context_id == context_id]
    return frames


def flat_keyframes(frames) -> list[tuple]:
    return [
        (round(k.offset, 6), k.openness, k.width, k.rounding) for f in frames for k in f.keyframes
    ]


class _FailingAnalyzer(BaseLipsyncAnalyzer):
    """Analyzer that blows up on first use, to test failure isolation."""

    async def start(self, sample_rate: int):
        pass

    async def analyze(self, pcm, context) -> LipsyncFrameResult:
        raise RuntimeError("boom")

    async def flush(self, context) -> LipsyncFrameResult:
        return LipsyncFrameResult()

    async def reset(self):
        pass


class TestLipsyncScaffolding(unittest.TestCase):
    """Smoke tests that the lipsync frames, params and processor construct."""

    def test_lipsync_frame_constructs(self):
        frame = TTSLipsyncFrame(
            context_id="ctx-1",
            window_start=0.0,
            window_end=0.2,
            keyframes=[
                LipsyncKeyframe(
                    offset=0.1,
                    openness=0.5,
                    width=0.5,
                    rounding=0.2,
                    energy=0.4,
                    pitch=0.3,
                    confidence=0.9,
                )
            ],
            events=[
                LipsyncEvent(
                    offset=0.15, kind=LipsyncEventKind.CLOSURE, duration=0.08, confidence=0.8
                )
            ],
        )
        self.assertIsNone(frame.pts)
        self.assertEqual(len(frame.keyframes), 1)
        self.assertEqual(frame.events[0].kind, "closure")

    def test_update_settings_frame_constructs(self):
        frame = LipsyncUpdateSettingsFrame(settings={"enabled": False})
        self.assertEqual(frame.settings["enabled"], False)

    def test_processor_constructs_with_defaults(self):
        processor = LipsyncProcessor(params=LipsyncParams(), analyzer=FormantLipsyncAnalyzer())
        self.assertTrue(processor._params.enabled)
        self.assertEqual(processor.stats["batches_emitted"], 0)


class TestLipsyncProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_passthrough_forwards_all_frames_unmodified(self):
        processor = LipsyncProcessor()
        frames = make_tts_frames(synth_vowel(700, 1200, secs=1.0), "ctx-1")
        received_down, _ = await run_test(processor, frames_to_send=frames)

        sent_ids = [f.id for f in frames]
        got_ids = [f.id for f in received_down if f.id in set(sent_ids)]
        self.assertEqual(got_ids, sent_ids)  # same objects, same order

        lipsync = lipsync_frames(received_down)
        self.assertTrue(lipsync)
        self.assertTrue(all(f.context_id == "ctx-1" for f in lipsync))
        self.assertTrue(all(f.pts for f in lipsync))
        self.assertEqual([f.pts for f in lipsync], sorted(f.pts for f in lipsync))
        self.assertEqual(processor.stats["batches_emitted"], len(lipsync))

    async def test_output_invariant_to_ingest_chunking(self):
        pcm = synth_vowel(700, 1200, secs=1.0)
        results = []
        for chunk_ms in (10, 320):
            processor = LipsyncProcessor()
            received_down, _ = await run_test(
                processor, frames_to_send=make_tts_frames(pcm, "ctx-1", chunk_ms=chunk_ms)
            )
            results.append(flat_keyframes(lipsync_frames(received_down)))

        self.assertEqual(len(results[0]), len(results[1]))
        for a, b in zip(results[0], results[1]):
            self.assertEqual(a[0], b[0])  # offsets exactly equal
            np.testing.assert_allclose(a[1:], b[1:], atol=1e-4)

    async def test_offsets_derived_from_sample_counts_under_jitter(self):
        pcm = synth_vowel(700, 1200, secs=0.8)
        plain = make_tts_frames(pcm, "ctx-1")
        jittered: list[Frame] = []
        for i, frame in enumerate(make_tts_frames(pcm, "ctx-1")):
            jittered.append(frame)
            if i % 20 == 10:
                jittered.append(SleepFrame(sleep=0.05))

        offsets = []
        for frames in (plain, jittered):
            processor = LipsyncProcessor()
            received_down, _ = await run_test(processor, frames_to_send=frames)
            offsets.append([o for (o, *_rest) in flat_keyframes(lipsync_frames(received_down))])

        self.assertTrue(offsets[0])
        self.assertEqual(offsets[0], offsets[1])

    async def test_interruption_clears_buffers_and_preserves_adaptive_state(self):
        # Half an utterance (too short for any batch to clear the event
        # horizon), interrupted, then a full utterance in a fresh context.
        half = make_tts_frames(synth_vowel(700, 1200, secs=0.5), "ctx-1")[:-1]  # no Stopped
        follow = make_tts_frames(synth_vowel(300, 2300, secs=1.0), "ctx-2")
        frames = [*half, SleepFrame(sleep=0.2), InterruptionFrame(), *follow]

        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)

        self.assertEqual(lipsync_frames(received_down, "ctx-1"), [])
        self.assertTrue(lipsync_frames(received_down, "ctx-2"))
        # Adaptation survived the interruption: voiced frames from both
        # utterances accumulated (ctx-2 alone contributes ~50).
        self.assertGreater(processor._analyzer._voiced_frames, 60)

    async def test_update_settings_disables_and_reenables(self):
        frames: list[Frame] = [
            *make_tts_frames(synth_vowel(700, 1200, secs=0.8), "ctx-1"),
            SleepFrame(sleep=0.2),  # let the analysis task drain ctx-1
            LipsyncUpdateSettingsFrame(settings={"enabled": False}),
            *make_tts_frames(synth_vowel(700, 1200, secs=0.8), "ctx-2"),
            SleepFrame(sleep=0.2),
            LipsyncUpdateSettingsFrame(settings={"enabled": True}),
            *make_tts_frames(synth_vowel(700, 1200, secs=0.8), "ctx-3"),
        ]
        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)

        self.assertTrue(lipsync_frames(received_down, "ctx-1"))
        self.assertEqual(lipsync_frames(received_down, "ctx-2"), [])
        self.assertTrue(lipsync_frames(received_down, "ctx-3"))

    async def test_two_sequential_contexts_isolated(self):
        first = synth_vowel(700, 1200, secs=0.7)
        second = synth_vowel(300, 2300, secs=0.7)
        frames = [
            *make_tts_frames(first, "ctx-1"),
            SleepFrame(sleep=0.1),
            *make_tts_frames(second, "ctx-2"),
        ]
        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)

        for context_id in ("ctx-1", "ctx-2"):
            frames_for_context = lipsync_frames(received_down, context_id)
            self.assertTrue(frames_for_context)
            for frame in frames_for_context:
                for keyframe in frame.keyframes:
                    self.assertLessEqual(keyframe.offset, 0.75)

        all_lipsync = lipsync_frames(received_down)
        self.assertEqual([f.pts for f in all_lipsync], sorted(f.pts for f in all_lipsync))

    async def test_stale_context_audio_ignored(self):
        frames = make_tts_frames(synth_vowel(700, 1200, secs=0.5), "ghost")[1:-1]  # audio only
        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)
        self.assertEqual(lipsync_frames(received_down), [])

    async def test_scheduling_lead_and_clamp(self):
        processor = LipsyncProcessor(params=LipsyncParams(scheduling_lead_ms=10_000))
        frames = make_tts_frames(synth_vowel(700, 1200, secs=1.0), "ctx-1")
        received_down, _ = await run_test(processor, frames_to_send=frames)

        lipsync = lipsync_frames(received_down)
        self.assertTrue(lipsync)
        stats = processor.stats
        self.assertEqual(stats["pts_clamped"], stats["batches_emitted"])
        self.assertEqual([f.pts for f in lipsync], sorted(f.pts for f in lipsync))

    async def test_analyzer_failure_degrades_to_passthrough(self):
        processor = LipsyncProcessor(analyzer=_FailingAnalyzer())
        frames = make_tts_frames(synth_vowel(700, 1200, secs=0.5), "ctx-1")
        received_down, received_up = await run_test(
            processor,
            frames_to_send=frames,
            expected_up_frames=[ErrorFrame],
        )

        sent_ids = [f.id for f in frames]
        got_ids = [f.id for f in received_down if f.id in set(sent_ids)]
        self.assertEqual(got_ids, sent_ids)
        self.assertEqual(lipsync_frames(received_down), [])
        error = next(f for f in received_up if isinstance(f, ErrorFrame))
        self.assertFalse(error.fatal)


class TestLipsyncMessageRelay(unittest.IsolatedAsyncioTestCase):
    """The relay sits after transport.output(), so every TTSLipsyncFrame it
    receives has already been released from the transport's clock queue at
    pts. Playout timing is therefore a pipeline-placement property — covered
    by live verification (message lead vs audio), not unit-testable here.
    """

    def _make_lipsync_frame(self):
        frame = TTSLipsyncFrame(
            context_id="ctx-1",
            window_start=0.0,
            window_end=0.2,
            keyframes=[
                LipsyncKeyframe(
                    offset=0.123456,
                    openness=0.512345,
                    width=0.448,
                    rounding=0.101,
                    energy=0.666,
                    pitch=0.333,
                    confidence=0.912,
                )
            ],
            events=[
                LipsyncEvent(
                    offset=0.15987, kind=LipsyncEventKind.CLOSURE, duration=0.08123, confidence=0.8
                )
            ],
        )
        frame.pts = 1
        return frame

    async def test_relays_quantized_server_message(self):
        frame = self._make_lipsync_frame()
        received_down, _ = await run_test(LipsyncMessageRelay(), frames_to_send=[frame])

        messages = [f for f in received_down if isinstance(f, RTVIServerMessageFrame)]
        self.assertEqual(len(messages), 1)
        # The original frame is forwarded, with the message following it.
        frame_index = next(i for i, f in enumerate(received_down) if f.id == frame.id)
        message_index = next(
            i for i, f in enumerate(received_down) if isinstance(f, RTVIServerMessageFrame)
        )
        self.assertGreater(message_index, frame_index)
        self.assertEqual(
            messages[0].data,
            {
                "type": "bot-tts-lipsync",
                "version": 1,
                "ctx": "ctx-1",
                "t0": 0.0,
                "kf": [[0.12, 0.51, 0.45, 0.1, 0.67, 0.33, 0.91]],
                "ev": [[0.16, "closure", 0.08, 0.8]],
            },
        )

    async def test_passthrough_forwards_all_frames(self):
        frames = make_tts_frames(synth_vowel(700, 1200, secs=0.1), "ctx-1")
        frames.insert(2, self._make_lipsync_frame())
        received_down, _ = await run_test(LipsyncMessageRelay(), frames_to_send=frames)

        sent_ids = [f.id for f in frames]
        got_ids = [f.id for f in received_down if f.id in set(sent_ids)]
        self.assertEqual(got_ids, sent_ids)  # same objects, same order
        messages = [f for f in received_down if isinstance(f, RTVIServerMessageFrame)]
        self.assertEqual(len(messages), 1)

    async def test_processor_to_relay_pipeline(self):
        pipeline = Pipeline([LipsyncProcessor(), LipsyncMessageRelay()])
        frames = make_tts_frames(synth_vowel(700, 1200, secs=1.0), "ctx-1")
        received_down, _ = await run_test(pipeline, frames_to_send=frames)

        messages = [f for f in received_down if isinstance(f, RTVIServerMessageFrame)]
        self.assertTrue(messages)
        for message in messages:
            self.assertEqual(message.data["type"], "bot-tts-lipsync")
            self.assertEqual(message.data["ctx"], "ctx-1")
            self.assertTrue(all(len(row) == 7 for row in message.data["kf"]))


if __name__ == "__main__":
    unittest.main()
