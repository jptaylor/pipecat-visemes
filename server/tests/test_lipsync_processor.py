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
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from lipsync.base_lipsync_analyzer import (
    BaseLipsyncAnalyzer,
    LipsyncFrameResult,
)
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.frames import LipsyncUpdateSettingsFrame, TTSLipsyncFrame
from lipsync.lipsync_processor import LipsyncParams, LipsyncProcessor
from lipsync.rtvi import LipsyncMessageRelay, lipsync_message_data
from lipsync.types import LipsyncEvent, LipsyncEventKind, LipsyncKeyframe
from tests.synth import synth_vowel

SAMPLE_RATE = 16000
NS = 1_000_000_000
LEAD_NS = 200_000_000  # LipsyncParams.scheduling_lead_ms default


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


def paced(frames: list[Frame], sleep: float) -> list[Frame]:
    """The same frames with a pause after every audio chunk (paced delivery)."""
    out: list[Frame] = []
    for frame in frames:
        out.append(frame)
        if isinstance(frame, TTSAudioRawFrame):
            out.append(SleepFrame(sleep=sleep))
    return out


def lipsync_frames(received, context_id=None) -> list[TTSLipsyncFrame]:
    frames = [f for f in received if isinstance(f, TTSLipsyncFrame)]
    if context_id is not None:
        frames = [f for f in frames if f.context_id == context_id]
    return frames


def server_messages(received) -> list[RTVIServerMessageFrame]:
    return [f for f in received if isinstance(f, RTVIServerMessageFrame)]


def flat_keyframes(frames) -> list[tuple]:
    return [
        (round(k.offset, 6), k.openness, k.width, k.rounding) for f in frames for k in f.keyframes
    ]


def make_lipsync_frame() -> TTSLipsyncFrame:
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
    frame.playout_ns = 10 * NS
    frame.release_ns = 10 * NS - LEAD_NS
    return frame


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


class _ClockTap(FrameProcessor):
    """Records the pipeline-clock time at which each downstream frame passes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.times: dict[int, int] = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.times[frame.id] = self.get_clock().get_time()
        await self.push_frame(frame, direction)


class _HeadlessOutputTransport(BaseOutputTransport):
    """An output transport with no device behind it.

    Audio is accepted and dropped. Lipsync batches are system frames, so the
    transport forwards them at once; word-timestamp frames would still go
    through its clock queue.
    """

    def __init__(self):
        super().__init__(TransportParams(audio_out_enabled=True))

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame) -> bool:
        return True


class TestLipsyncScaffolding(unittest.TestCase):
    """Smoke tests that the lipsync frames, params and processor construct."""

    def test_lipsync_frame_constructs(self):
        frame = make_lipsync_frame()
        self.assertIsNone(frame.pts)
        self.assertEqual(frame.release_ns, 10 * NS - LEAD_NS)
        self.assertEqual(frame.playout_offset, 0.0)
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
        self.assertTrue(all(f.release_ns and f.playout_ns for f in lipsync))
        self.assertTrue(all(f.pts is None for f in lipsync))
        self.assertTrue(all(f.playout_offset == 0.0 for f in lipsync))
        self.assertEqual([f.release_ns for f in lipsync], sorted(f.release_ns for f in lipsync))
        # Every emitted batch was delivered (those still scheduled when the
        # EndFrame arrived were released then).
        self.assertEqual(processor.stats["batches_emitted"], len(lipsync))
        self.assertEqual(processor.stats["playout_gaps"], 0)

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

    async def test_interruption_drops_scheduled_batches_and_preserves_adaptive_state(self):
        # A whole utterance arrives in a burst and is interrupted 50 ms in:
        # the batches already due were released, the ones scheduled for
        # later were dropped with the audio; then a fresh context.
        head = make_tts_frames(synth_vowel(700, 1200, secs=1.0), "ctx-1")[:-1]  # no Stopped
        follow = make_tts_frames(synth_vowel(300, 2300, secs=1.0), "ctx-2")
        frames = [*head, SleepFrame(sleep=0.05), InterruptionFrame(), *follow]

        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)

        interruption = next(
            i for i, f in enumerate(received_down) if isinstance(f, InterruptionFrame)
        )
        first = lipsync_frames(received_down, "ctx-1")
        self.assertTrue(first)
        self.assertLessEqual(len(first), 2)  # windows [0, 0.1) and [0.1, 0.3) were already due
        self.assertTrue(all(received_down.index(f) < interruption for f in first))
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
        self.assertEqual(
            [f.release_ns for f in all_lipsync], sorted(f.release_ns for f in all_lipsync)
        )

    async def test_queued_context_is_anchored_after_the_previous_audio(self):
        # ctx-2's audio arrives while ctx-1 (1.5 s) is still playing, so its
        # batches must be scheduled after ctx-1's audio ends, not at arrival.
        frames = [
            *make_tts_frames(synth_vowel(700, 1200, secs=1.5), "ctx-1"),
            *make_tts_frames(synth_vowel(300, 2300, secs=0.7), "ctx-2"),
        ]
        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)

        first = lipsync_frames(received_down, "ctx-1")
        second = lipsync_frames(received_down, "ctx-2")
        self.assertTrue(first and second)
        gap_ns = second[0].release_ns - first[0].release_ns
        # ctx-2's first batch is due 1.5 s after ctx-1's, less ctx-1's first
        # batch being clamped to its emission time (well under 0.3 s); if
        # ctx-2 were anchored at arrival instead, the gap would be ~0.
        self.assertGreater(gap_ns, 1.0e9)
        self.assertLess(gap_ns, 1.5e9)

    async def test_reopened_context_id_starts_a_new_segment(self):
        # pipecat >= 1.8 reuses one context id per turn and reopens it after
        # its idle timeout: Started/Stopped for the same id, twice.
        frames = [
            *make_tts_frames(synth_vowel(700, 1200, secs=0.8), "ctx-1"),
            SleepFrame(sleep=0.2),
            *make_tts_frames(synth_vowel(300, 2300, secs=0.8), "ctx-1"),
        ]
        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)

        lipsync = lipsync_frames(received_down, "ctx-1")
        self.assertEqual([f.window_start for f in lipsync].count(0.0), 2)
        self.assertEqual(processor.stats["contexts_opened"], 2)
        self.assertEqual([f.release_ns for f in lipsync], sorted(f.release_ns for f in lipsync))
        for frame in lipsync:
            for keyframe in frame.keyframes:
                self.assertLessEqual(keyframe.offset, 0.85)

    async def test_playout_gap_shifts_later_batches(self):
        # 0.5 s of audio, then a stall longer than its playout: the audio
        # after the stall plays when it arrives, so its batches are shifted
        # and reported with a nonzero playout offset.
        pcm = synth_vowel(700, 1200, secs=0.5)
        head = make_tts_frames(pcm, "ctx-1")[:-1]  # no Stopped
        tail = make_tts_frames(synth_vowel(300, 2300, secs=0.8), "ctx-1")[1:]  # no Started
        frames = [*head, SleepFrame(sleep=0.9), *tail]

        processor = LipsyncProcessor()
        received_down, _ = await run_test(processor, frames_to_send=frames)

        self.assertEqual(processor.stats["playout_gaps"], 1)
        lipsync = lipsync_frames(received_down, "ctx-1")
        before = [f for f in lipsync if f.window_end <= 0.5]
        after = [f for f in lipsync if f.window_start >= 0.5]
        self.assertEqual(len(before) + len(after), len(lipsync))  # none straddle the gap
        self.assertTrue(before and after)
        self.assertTrue(all(f.playout_offset == 0.0 for f in before))
        # The stall was ~0.4 s beyond the buffered audio (0.9 s sleep - 0.5 s playout).
        for frame in after:
            self.assertGreater(frame.playout_offset, 0.2)
            self.assertLess(frame.playout_offset, 1.0)
        self.assertEqual([f.release_ns for f in lipsync], sorted(f.release_ns for f in lipsync))

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
        self.assertEqual(stats["release_clamped"], stats["batches_emitted"])
        self.assertEqual([f.release_ns for f in lipsync], sorted(f.release_ns for f in lipsync))

    async def test_first_window_leaves_one_hop_after_its_audio(self):
        # At real-time delivery the first window's batch (a short 0.1 s one)
        # leaves once analysis is one hop past its end (0.14 s of audio in),
        # not after the 0.4 s closure-confirmation horizon.
        tap = _ClockTap()
        frames = paced(
            make_tts_frames(synth_vowel(700, 1200, secs=0.8), "ctx-1", chunk_ms=20), 0.02
        )
        received_down, _ = await run_test(
            Pipeline([LipsyncProcessor(), tap]), frames_to_send=frames
        )

        first_audio = next(f for f in received_down if isinstance(f, TTSAudioRawFrame))
        first_batch = lipsync_frames(received_down)[0]
        self.assertEqual(first_batch.window_start, 0.0)
        self.assertAlmostEqual(first_batch.window_end, 0.1)
        elapsed = (tap.times[first_batch.id] - tap.times[first_audio.id]) / NS
        self.assertGreater(elapsed, 0.1)
        self.assertLess(elapsed, 0.3)
        second = lipsync_frames(received_down)[1]
        self.assertAlmostEqual(second.window_start, 0.1)
        self.assertAlmostEqual(second.window_end, 0.3)

    async def test_batches_are_released_on_schedule(self):
        # A burst-delivered utterance: batches whose lead has not passed are
        # held by the processor and pushed at playout minus the lead; the
        # first ones (already due) leave at once.
        tap = _ClockTap()
        processor = LipsyncProcessor()
        frames = [
            *make_tts_frames(synth_vowel(700, 1200, secs=1.0), "ctx-1"),
            SleepFrame(sleep=1.0),
        ]
        received_down, _ = await run_test(
            Pipeline([processor, tap]), frames_to_send=frames, start_timeout=5.0
        )

        first_audio = next(f for f in received_down if isinstance(f, TTSAudioRawFrame))
        t0 = tap.times[first_audio.id]
        lipsync = lipsync_frames(received_down)
        self.assertGreaterEqual(len(lipsync), 4)
        for frame in lipsync:
            delivered = tap.times[frame.id]
            self.assertGreaterEqual(delivered, frame.release_ns - 2_000_000)  # never early
            due = t0 + int(frame.window_start * NS) - LEAD_NS
            if frame.window_start >= 0.3:  # not clamped: scheduled for its lead
                self.assertLess(abs(delivered - due), 60_000_000)
        self.assertEqual([f.release_ns for f in lipsync], sorted(f.release_ns for f in lipsync))
        self.assertLessEqual(processor.stats["release_clamped"], 2)

    async def test_late_events_ride_in_later_batches(self):
        # A closure is confirmed only once speech resumes after the dip. With
        # the dip straddling the end of the [0.1, 0.3) window, that window's
        # batch has left by then, so the event rides in a later batch with an
        # offset before that batch's window.
        vowel = synth_vowel(700, 1200, secs=0.26)
        pcm = np.concatenate(
            [
                vowel,
                np.zeros(int(0.1 * SAMPLE_RATE), dtype=vowel.dtype),
                synth_vowel(700, 1200, secs=0.64),
            ]
        )
        frames = paced(make_tts_frames(pcm, "ctx-1", chunk_ms=20), 0.002)
        received_down, _ = await run_test(LipsyncProcessor(), frames_to_send=frames)

        lipsync = lipsync_frames(received_down)
        for frame in lipsync:
            for keyframe in frame.keyframes:
                self.assertTrue(frame.window_start <= keyframe.offset < frame.window_end)
            for event in frame.events:
                self.assertLess(event.offset, frame.window_end)
        late = [
            event
            for frame in lipsync
            for event in frame.events
            if event.kind == LipsyncEventKind.CLOSURE and event.offset < frame.window_start
        ]
        self.assertTrue(late)
        self.assertTrue(any(0.2 <= event.offset <= 0.35 for event in late))

    async def test_idle_flush_emits_final_keyframes_during_a_stall(self):
        # 0.3 s of audio, then nothing for a while: the keyframes past the
        # first window, final but short of filling the second, are flushed
        # as a short batch instead of waiting for more audio.
        head = make_tts_frames(synth_vowel(700, 1200, secs=0.3), "ctx-1")[:-1]  # no Stopped
        tail = make_tts_frames(synth_vowel(300, 2300, secs=0.3), "ctx-1")[1:]  # no Started
        frames = [*head, SleepFrame(sleep=0.25), *tail]

        processor = LipsyncProcessor(params=LipsyncParams(heartbeat_ms=40))
        received_down, _ = await run_test(processor, frames_to_send=frames)

        self.assertEqual(processor.stats["idle_flushes"], 1)
        lipsync = lipsync_frames(received_down)
        # Windows [0, 0.1) and then [0.1, 0.3), cut short at what was final.
        partial = [f for f in lipsync if f.window_start == 0.1 and f.window_end < 0.3]
        self.assertEqual(len(partial), 1)
        self.assertGreater(partial[0].window_end, 0.2)
        # Flushed during the stall, before the tail's audio arrived.
        self.assertLess(received_down.index(partial[0]), received_down.index(tail[0]))
        for earlier, later in zip(lipsync, lipsync[1:]):
            self.assertGreaterEqual(later.window_start, earlier.window_end - 1e-9)

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
    """The relay sits after transport.output(); each TTSLipsyncFrame reaches it
    when the processor released it (see the end-to-end test below), and the
    message is stamped with the lead remaining at that moment.
    """

    EXPECTED_DATA = {
        "type": "bot-tts-lipsync",
        "version": 2,
        "ctx": "ctx-1",
        "t0": 0.0,
        "ws": 0.0,
        "lead": 0.15,
        "kf": [[0.123, 0.51, 0.45, 0.1, 0.67, 0.33, 0.91]],
        "ev": [[0.16, "closure", 0.081, 0.8]],
    }

    def test_message_data_is_quantized_and_stamped_with_lead(self):
        frame = make_lipsync_frame()
        data = lipsync_message_data(frame, now_ns=frame.playout_ns - 150_000_000)
        self.assertEqual(data, self.EXPECTED_DATA)
        # Sent after its window started playing: the lead goes negative.
        late = lipsync_message_data(frame, now_ns=frame.playout_ns + 35_000_000)
        self.assertEqual(late["lead"], -0.035)

    async def test_relays_server_message_after_the_frame(self):
        frame = make_lipsync_frame()
        received_down, _ = await run_test(LipsyncMessageRelay(), frames_to_send=[frame])

        messages = server_messages(received_down)
        self.assertEqual(len(messages), 1)
        # The original frame is forwarded, with the message following it.
        frame_index = next(i for i, f in enumerate(received_down) if f.id == frame.id)
        message_index = next(
            i for i, f in enumerate(received_down) if isinstance(f, RTVIServerMessageFrame)
        )
        self.assertGreater(message_index, frame_index)
        data = messages[0].data
        self.assertEqual(data["version"], 2)
        self.assertEqual(data["ws"], 0.0)
        self.assertIsInstance(data["lead"], float)
        self.assertEqual(data["kf"], self.EXPECTED_DATA["kf"])

    async def test_playout_offset_is_reported_as_t0(self):
        frame = make_lipsync_frame()
        frame.playout_offset = 0.4321
        received_down, _ = await run_test(LipsyncMessageRelay(), frames_to_send=[frame])
        self.assertEqual(server_messages(received_down)[0].data["t0"], 0.432)

    async def test_passthrough_forwards_all_frames(self):
        frames = make_tts_frames(synth_vowel(700, 1200, secs=0.1), "ctx-1")
        frames.insert(2, make_lipsync_frame())
        received_down, _ = await run_test(LipsyncMessageRelay(), frames_to_send=frames)

        sent_ids = [f.id for f in frames]
        got_ids = [f.id for f in received_down if f.id in set(sent_ids)]
        # Every frame is forwarded (the same objects) and the data frames keep
        # their order; the lipsync batch is a system frame and may overtake.
        self.assertEqual(sorted(got_ids), sorted(sent_ids))
        data_ids = [f.id for f in frames if not isinstance(f, TTSLipsyncFrame)]
        self.assertEqual([i for i in got_ids if i in set(data_ids)], data_ids)
        self.assertEqual(len(server_messages(received_down)), 1)

    async def test_processor_to_relay_pipeline(self):
        pipeline = Pipeline([LipsyncProcessor(), LipsyncMessageRelay()])
        frames = make_tts_frames(synth_vowel(700, 1200, secs=1.0), "ctx-1")
        received_down, _ = await run_test(pipeline, frames_to_send=frames)

        messages = server_messages(received_down)
        self.assertTrue(messages)
        for message in messages:
            self.assertEqual(message.data["type"], "bot-tts-lipsync")
            self.assertEqual(message.data["ctx"], "ctx-1")
            self.assertTrue(all(len(row) == 7 for row in message.data["kf"]))
        starts = [m.data["ws"] for m in messages]
        self.assertEqual(starts, sorted(starts))

    async def test_end_to_end_through_an_output_transport(self):
        # Processor → real output transport → relay: every batch becomes
        # exactly one server message, when the processor released it, and the
        # message carries the lead still remaining at that moment.
        transport = _HeadlessOutputTransport()
        pipeline = Pipeline([LipsyncProcessor(), transport, LipsyncMessageRelay()])
        frames = [
            *make_tts_frames(synth_vowel(700, 1200, secs=1.0), "ctx-1"),
            SleepFrame(sleep=1.0),  # let the scheduled batches go out on time
        ]
        received_down, _ = await run_test(pipeline, frames_to_send=frames, start_timeout=5.0)

        released = lipsync_frames(received_down)
        self.assertTrue(released)
        self.assertEqual([f.release_ns for f in released], sorted(f.release_ns for f in released))
        messages = server_messages(received_down)
        self.assertEqual(len(messages), len(released))
        for frame, message in zip(released, messages):
            frame_index = received_down.index(frame)
            message_index = received_down.index(message)
            self.assertGreater(message_index, frame_index)
            self.assertEqual(message.data["ctx"], "ctx-1")
            self.assertEqual(message.data["ws"], round(frame.window_start, 3))
            self.assertEqual(len(message.data["kf"]), len(frame.keyframes))
            if frame.window_start >= 0.3:
                # Released on schedule: the stamped lead is the configured one.
                self.assertAlmostEqual(message.data["lead"], 0.2, delta=0.06)
        # The first batch of a turn leaves once its window is analyzed, which
        # is after its lead would have been: a small negative lead.
        self.assertLess(messages[0].data["lead"], 0.05)


if __name__ == "__main__":
    unittest.main()
