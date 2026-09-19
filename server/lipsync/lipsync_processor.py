#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Provider-universal lipsync processor for TTS audio.

This module provides :class:`LipsyncProcessor`, a frame processor placed
between a TTS service and the output transport::

    ... → LLM → TTSService → LipsyncProcessor → transport.output()

It forwards every frame downstream immediately and unmodified (the audio path
gains zero latency), taps ``TTSAudioRawFrame`` payloads into per-context ring
buffers, analyzes them in a dedicated task, and emits ``TTSLipsyncFrame``
batches. Each batch is scheduled on the pipeline clock just ahead of its
window's playout and pushed at that time by the processor's own delivery
task; as a system frame it passes straight through the output transport
(whose clock queue would hold it behind any earlier-queued word-timestamp
frame with a later timestamp, and whose audio-sync path would release it
only after the audio queued ahead of it had played). Batches not yet
released are dropped on interruption, in step with the discarded audio.
:class:`~lipsync.rtvi.LipsyncMessageRelay`, placed after
``transport.output()``, then delivers each released batch to clients as a
standard RTVI ``server-message`` whose ``data.type`` is ``"bot-tts-lipsync"``.
Works with any ``TTSService``; no provider-specific requirements.
"""

import asyncio
import bisect

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregatedTextFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.time import nanoseconds_to_seconds, seconds_to_nanoseconds
from pydantic import BaseModel

from lipsync.base_lipsync_analyzer import (
    BaseLipsyncAnalyzer,
    LipsyncAnalysisContext,
    LipsyncFrameResult,
)
from lipsync.dsp import ANALYSIS_SAMPLE_RATE
from lipsync.formant_lipsync_analyzer import HOP_SECONDS, FormantLipsyncAnalyzer
from lipsync.frames import LipsyncUpdateSettingsFrame, TTSLipsyncFrame
from lipsync.text_prior import TextAnchor, TextWord, _TextAccumulator
from lipsync.types import LipsyncEvent, LipsyncEventKind, LipsyncKeyframe

# Ring buffers hold at most this much un-analyzed audio per context; beyond
# it, the oldest audio is dropped (backpressure) rather than growing memory.
_BUFFER_CAP_SECONDS = 2.0

# At most this many live TTS contexts are tracked; older ones are evicted.
_MAX_CONTEXTS = 8

# How long to wait for the analysis task to drain on EndFrame before
# cancelling it.
_END_DRAIN_TIMEOUT_SECS = 1.0

# Audio arriving later than this after the transport is expected to have run
# out of the context's audio counts as a playout gap (the transport chunks and
# paces its writes, so small overshoots are normal).
_PLAYOUT_GAP_TOLERANCE_SECS = 0.1

# Keyframes are final one hop after their own (zero-phase conditioning), so a
# window is emitted as soon as analysis is one hop past its end. Closures and
# silences are confirmed later (the analyzer's EVENT_FINALIZE_HORIZON_SEC) and
# ride in whichever batch is emitted once they are known.
_KEYFRAME_HORIZON_SEC = HOP_SECONDS

# The first window of each utterance is this short, so its batch leaves as
# soon as the first 0.14 s of audio has been analyzed; later windows use
# ``batch_window_ms``.
_FIRST_WINDOW_SEC = 0.1

# With no new audio for this long while a context is open, what is already
# final is emitted as a short batch instead of waiting for the window to fill:
# an LLM stalling mid-turn would otherwise hold the tail of the previous
# sentence until the next one arrived.
_IDLE_FLUSH_SECS = 0.1

_INT16_SCALE = 32768.0


class LipsyncParams(BaseModel):
    """Configuration parameters for :class:`LipsyncProcessor`.

    Parameters:
        batch_window_ms: Audio time covered by one emitted ``TTSLipsyncFrame``.
        scheduling_lead_ms: How far ahead of playout batches are released,
            giving the client scheduler runway. The lead actually remaining
            when a batch is sent goes on the wire, so clients anchor on it
            rather than assuming this value.
        dead_band: Minimum parameter delta since the last emitted keyframe
            required to emit a new one. Applied by the default analyzer;
            explicitly constructed analyzers own their conditioning settings.
        heartbeat_ms: Maximum time between keyframes even when parameters are
            static, so client interpolators stay pinned. Applied by the
            default analyzer, like ``dead_band``.
        emit_energy: Whether keyframes include the energy envelope.
        emit_pitch: Whether keyframes include normalized pitch.
        enabled: Whether analysis runs. When False the processor is
            passthrough-only and uses no CPU. Togglable at runtime via
            ``LipsyncUpdateSettingsFrame``; enabling mid-utterance takes
            effect from the next TTS context.
        text_prior_enabled: Collect stock sentence anchors and word timestamps
            as optional analyzer inputs. Experimental and off by default.
            Without text_events_enabled, these are observation only.
            Enabling takes effect on the next context; disabling drops priors.
        text_events_enabled: Opt in to experimental English text-informed
            events in the default analyzer. Implies text collection. Custom
            analyzers own this setting, like dead_band. Off by default.
    """

    batch_window_ms: int = 200
    scheduling_lead_ms: int = 200
    dead_band: float = 0.05
    heartbeat_ms: int = 240
    emit_energy: bool = True
    emit_pitch: bool = True
    enabled: bool = True
    text_prior_enabled: bool = False
    text_events_enabled: bool = False


class _Context:
    """Internal: per-TTS-context ingest and batching state.

    Offsets are seconds of audio from the context's first sample; ``t0`` is
    the clock time that sample plays. Whenever the transport runs out of the
    context's audio before more arrives (e.g. the LLM stalled mid-response),
    playout resumes when the next chunk arrives, so audio from that offset on
    plays later than ``t0 + offset``. ``gaps`` records each such shift so
    batches can be scheduled, and offsets reported, in playout time.
    """

    def __init__(self, context_id: str | None, text: _TextAccumulator | None = None):
        self.context_id = context_id
        self.buffer = bytearray()
        self.capacity = 0  # set at first audio, once the sample rate is known
        self.sample_rate = 0
        self.t0 = 0
        self.previous_word_pts = 0
        self.word_start_pts: int | None = None
        self.transport_destination: str | None = None
        self.closing = False
        self.analysis = LipsyncAnalysisContext(context_id=context_id, sample_rate=0)
        self.text = text
        self.resampler = create_stream_resampler()
        self.pending_keyframes: list[LipsyncKeyframe] = []
        self.pending_events: list[LipsyncEvent] = []
        self.cursor = 0.0  # seconds of this context fully analyzed
        self.window_start = 0.0
        self.skip_offset = 0.0  # audio time skipped by backpressure drops
        self.drop_silence_pending = False
        # (audio offset, cumulative playout shift) per playout gap, in seconds.
        self.gaps: list[tuple[float, float]] = []

    @property
    def has_work(self) -> bool:
        return bool(self.buffer) or self.closing

    @property
    def ingested_seconds(self) -> float:
        """Seconds of source audio ingested so far."""
        if not self.sample_rate:
            return 0.0
        return self.analysis.samples_seen / self.sample_rate

    def playout_shift(self, offset: float) -> float:
        """Seconds audio at ``offset`` plays later than ``t0 + offset``."""
        shift = 0.0
        for gap_offset, cumulative in self.gaps:
            if offset < gap_offset:
                break
            shift = cumulative
        return shift

    @property
    def playout_end(self) -> int:
        """Clock time (ns) at which the last ingested sample finishes playing."""
        offset = self.ingested_seconds
        return self.t0 + seconds_to_nanoseconds(offset + self.playout_shift(offset))


class LipsyncProcessor(FrameProcessor):
    """Generates a real-time mouth-articulation signal from streamed TTS audio.

    Copies TTS audio into per-context ring buffers and analyzes it off the
    frame path with a pluggable
    :class:`~lipsync.base_lipsync_analyzer.BaseLipsyncAnalyzer`
    (by default the formant analyzer, which works with any TTS provider).
    Emits ``TTSLipsyncFrame`` batches, each pushed by the processor's delivery
    task at its scheduled release time just ahead of audio playout; a
    :class:`~lipsync.rtvi.LipsyncMessageRelay` after ``transport.output()``
    turns each released batch into an RTVI ``server-message`` with
    ``data.type`` ``"bot-tts-lipsync"``.

    Analysis failures never propagate to the pipeline: on an unexpected
    error the processor reports a non-fatal ``ErrorFrame`` and degrades to
    passthrough for the rest of the session.

    Example::

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_aggregator,
                llm,
                tts,
                LipsyncProcessor(),
                transport.output(),
                LipsyncMessageRelay(),
                assistant_aggregator,
            ]
        )
    """

    def __init__(
        self,
        *,
        params: LipsyncParams | None = None,
        analyzer: BaseLipsyncAnalyzer | None = None,
        **kwargs,
    ):
        """Initialize the lipsync processor.

        Args:
            params: Batching, conditioning and scheduling parameters.
            analyzer: Analyzer to use. Defaults to
                :class:`~lipsync.formant_lipsync_analyzer.FormantLipsyncAnalyzer`,
                the provider-universal formant analyzer, configured with this
                processor's ``dead_band`` and ``heartbeat_ms``.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._params = params or LipsyncParams()
        self._owns_analyzer = analyzer is None
        self._analyzer = analyzer or FormantLipsyncAnalyzer(
            dead_band=self._params.dead_band,
            heartbeat_ms=self._params.heartbeat_ms,
            text_events_enabled=self._params.text_events_enabled,
        )

        # Contexts in creation order. A TTS service may reopen a context id it
        # already closed (e.g. when it reuses one id for a whole turn and the
        # LLM stalls past its idle timeout), so the same id can appear more
        # than once: the newest open entry is the live one.
        self._contexts: list[_Context] = []
        # Sentence anchors normally precede TTSStartedFrame. Bound orphaned
        # anchors by the same context limit as audio; words alone never open
        # a pending entry. Values are attached when their context starts.
        self._pending_text: dict[str | None, _TextAccumulator] = {}
        self._sample_rate = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        # Batches waiting for their release time, ordered by (release, seq).
        self._scheduled: list[tuple[int, int, TTSLipsyncFrame]] = []
        self._schedule_seq = 0
        self._deliver_wake = asyncio.Event()
        self._deliver_task: asyncio.Task | None = None
        self._generation = 0
        # Playout end of the most recently ingested audio, so a context that
        # queues behind another one is anchored where that audio ends rather
        # than at its own arrival time.
        self._last_playout_end = 0
        self._last_word_pts = 0
        self._stopping = False
        self._failed = False
        self._stats = {
            "batches_emitted": 0,
            "keyframes_emitted": 0,
            "events_emitted": 0,
            "release_clamped": 0,
            "idle_flushes": 0,
            "playout_gaps": 0,
            "bytes_dropped": 0,
            "contexts_opened": 0,
            "contexts_evicted": 0,
            "text_anchors": 0,
            "text_words": 0,
            "text_discarded": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Diagnostic counters (batches, drops, late releases); read-only snapshot."""
        return {
            **self._stats,
            **{f"prior_{k}": v for k, v in getattr(self._analyzer, "text_stats", {}).items()},
        }

    @property
    def _collect_text(self) -> bool:
        return self._params.text_prior_enabled or self._params.text_events_enabled

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._sample_rate = setup.audio_out_sample_rate

    async def cleanup(self):
        """Clean up the processor and cancel its tasks."""
        await super().cleanup()
        await self._cancel_tasks()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Forward all frames unmodified, tapping TTS frames for analysis.

        The frame path copies audio and, when opted in, appends bounded text
        observations. All analysis and text snapshots happen in its own task.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption()
        elif direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, StartFrame):
                await self._start(frame)
            elif isinstance(frame, TTSStartedFrame):
                self._handle_tts_started(frame)
            elif isinstance(frame, TTSAudioRawFrame):
                self._handle_tts_audio(frame)
            elif isinstance(frame, TTSStoppedFrame):
                self._handle_tts_stopped(frame)
            elif isinstance(frame, AggregatedTextFrame):
                self._handle_tts_text(frame)
            elif isinstance(frame, LipsyncUpdateSettingsFrame):
                await self._handle_update_settings(frame)
            elif isinstance(frame, EndFrame):
                await self._stop()
            elif isinstance(frame, CancelFrame):
                await self._cancel()

        await self.push_frame(frame, direction)

    #
    # Frame handling (all O(small), never blocks on analysis)
    #

    async def _start(self, frame: StartFrame):
        if self._task:
            return
        await self._analyzer.start(self._sample_rate)
        self._task = self.create_task(self._analysis_task_handler())
        self._deliver_task = self.create_task(self._delivery_task_handler())

    def _open_context(self, context_id: str | None) -> _Context | None:
        """The live (not yet closed) context for ``context_id``, if any."""
        for context in reversed(self._contexts):
            if context.context_id == context_id:
                return None if context.closing else context
        return None

    def _handle_tts_started(self, frame: TTSStartedFrame):
        if not self._params.enabled or self._failed:
            return
        if self._open_context(frame.context_id) is not None:
            return
        if len(self._contexts) >= _MAX_CONTEXTS:
            evicted = self._contexts.pop(0)
            self._stats["contexts_evicted"] += 1
            logger.warning(f"{self} evicted stale lipsync context {evicted.context_id}")
        text = None
        if self._collect_text:
            text = self._pending_text.pop(frame.context_id, None) or _TextAccumulator()
        context = _Context(frame.context_id, text)
        context.previous_word_pts = self._last_word_pts
        self._contexts.append(context)
        self._stats["contexts_opened"] += 1

    def _handle_tts_text(self, frame: AggregatedTextFrame):
        if (
            isinstance(frame, TTSTextFrame)
            and frame.aggregated_by == "word"
            and frame.pts is not None
        ):
            self._last_word_pts = frame.pts
        if not self._params.enabled or not self._collect_text or self._failed:
            return
        # TTSTextFrame subclasses AggregatedTextFrame: check it FIRST, or
        # every word (and the non-streaming completion sentence) becomes a
        # new sentence anchor.
        word = isinstance(frame, TTSTextFrame)
        if word:
            if frame.aggregated_by != "word" or frame.pts is None:
                return
        elif not frame.will_be_spoken or frame.aggregated_by != "sentence":
            return
        if not frame.text.strip():
            return

        context = self._open_context(frame.context_id)
        if context is not None:
            text = context.text
            received_after_audio = context.ingested_seconds
        else:
            text = self._pending_text.get(frame.context_id)
            if text is None and not word:
                if len(self._pending_text) >= _MAX_CONTEXTS:
                    self._pending_text.pop(next(iter(self._pending_text)))
                    self._stats["text_discarded"] += 1
                text = _TextAccumulator()
                self._pending_text[frame.context_id] = text
            received_after_audio = 0.0
        if text is None:
            return
        observation = (
            TextWord(frame.text, frame.pts, received_after_audio)
            if word
            else TextAnchor(frame.text, frame.pts, received_after_audio)
        )
        if text.append(observation):
            self._stats["text_words" if word else "text_anchors"] += 1
        else:
            self._stats["text_discarded"] += 1

    def _handle_tts_audio(self, frame: TTSAudioRawFrame):
        context = self._open_context(frame.context_id)
        if context is None or not self._params.enabled or self._failed:
            return
        now = self.get_clock().get_time()
        if context.t0 == 0:
            # Audio plays when it arrives unless earlier audio is still
            # queued, in which case it plays when that audio ends.
            context.t0 = max(now, self._last_playout_end)
            # Pipecat's word clock starts at first audio, but can inherit the
            # preceding context's last word PTS. Sentence-anchor PTS is earlier.
            context.word_start_pts = max(now, context.previous_word_pts)
            context.sample_rate = frame.sample_rate
            context.analysis.sample_rate = frame.sample_rate
            context.transport_destination = frame.transport_destination
            context.capacity = int(_BUFFER_CAP_SECONDS * frame.sample_rate) * 2
        else:
            expected = context.playout_end
            gap = nanoseconds_to_seconds(now - expected)
            if gap > _PLAYOUT_GAP_TOLERANCE_SECS:
                # The transport ran dry before this chunk arrived, so the
                # audio from here on plays later than its offset implies.
                previous = context.gaps[-1][1] if context.gaps else 0.0
                context.gaps.append((context.ingested_seconds, previous + gap))
                self._stats["playout_gaps"] += 1
        overflow = len(context.buffer) + len(frame.audio) - context.capacity
        if overflow > 0:
            del context.buffer[:overflow]
            context.skip_offset += overflow / 2.0 / context.sample_rate
            context.drop_silence_pending = True
            self._stats["bytes_dropped"] += overflow
            context.text = None  # Timing no longer matches the analyzer cursor.
        context.buffer += frame.audio
        context.analysis.samples_seen += frame.num_frames
        self._last_playout_end = max(self._last_playout_end, context.playout_end)
        self._wake.set()

    def _handle_tts_stopped(self, frame: TTSStoppedFrame):
        self._pending_text.pop(frame.context_id, None)
        context = self._open_context(frame.context_id)
        if context is None:
            return
        context.closing = True
        self._wake.set()

    async def _handle_interruption(self):
        # Batches not yet released go the way of the discarded audio.
        await self._discard_analysis_state(reset_word_clock=True)

    async def _handle_update_settings(self, frame: LipsyncUpdateSettingsFrame):
        for key, value in frame.settings.items():
            if key in LipsyncParams.model_fields:
                setattr(self._params, key, value)
            else:
                logger.warning(f"{self} unknown lipsync setting: {key}")
        if self._owns_analyzer:
            self._analyzer.set_text_events_enabled(self._params.text_events_enabled)
        if not self._params.enabled:
            await self._discard_analysis_state()
        elif not self._collect_text:
            self._pending_text.clear()
            for context in self._contexts:
                context.text = None
                context.analysis.text_prior = None

    async def _discard_analysis_state(self, *, reset_word_clock: bool = False):
        """Drop buffers and in-flight work; adaptive analyzer state survives."""
        self._generation += 1
        self._contexts.clear()
        self._pending_text.clear()
        self._scheduled.clear()
        self._deliver_wake.set()
        # Queued audio was discarded too, so nothing is playing after now.
        self._last_playout_end = 0
        # Disabling lipsync alone does not reset the upstream TTS word clock.
        if reset_word_clock:
            self._last_word_pts = 0
        if self._task:
            await self._analyzer.reset()

    async def _stop(self):
        self._pending_text.clear()
        if not self._task:
            return
        for context in self._contexts:
            context.closing = True
        self._stopping = True
        self._wake.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), _END_DRAIN_TIMEOUT_SECS)
        except TimeoutError:
            pass
        if not self._task.done():
            await self.cancel_task(self._task)
        self._task = None
        # The pipeline is ending: release whatever is still scheduled now (the
        # relay stamps each batch with its true lead, so early delivery is
        # harmless to clients), then retire the delivery task.
        if self._deliver_task:
            await self.cancel_task(self._deliver_task)
            self._deliver_task = None
        while self._scheduled:
            _, _, frame = self._scheduled.pop(0)
            await self.push_frame(frame)

    async def _cancel(self):
        await self._cancel_tasks()
        self._contexts.clear()

    async def _cancel_tasks(self):
        self._pending_text.clear()
        if self._task:
            await self.cancel_task(self._task)
            self._task = None
        if self._deliver_task:
            await self.cancel_task(self._deliver_task)
            self._deliver_task = None
        self._scheduled.clear()

    #
    # Analysis task
    #

    async def _analysis_task_handler(self):
        """Drain buffered audio, analyze, batch and schedule frames.

        Single long-lived task per processor. Parks on an event while there
        is no work (bot silence, or ``enabled`` False), so the idle cost is
        zero; while a context is open with final keyframes waiting for their
        window to fill, the park is bounded by the idle flush. Any unexpected
        error disables lipsync for the session and reports a non-fatal error
        upstream; the pipeline keeps running.
        """
        try:
            while True:
                while (context := self._active_context()) is not None:
                    await self._process_context(context)
                if self._stopping:
                    break
                idle = self._idle_context()
                if idle is None:
                    await self._wake.wait()
                else:
                    try:
                        await asyncio.wait_for(self._wake.wait(), _IDLE_FLUSH_SECS)
                    except TimeoutError:
                        await self._emit_batches(idle, self._generation, final=False, idle=True)
                        continue
                self._wake.clear()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._failed = True
            self._contexts.clear()
            self._pending_text.clear()
            await self.push_error(
                "Lipsync analysis failed; disabling lipsync for this session", exception=e
            )

    async def _delivery_task_handler(self):
        """Push each scheduled batch downstream at its release time.

        Sleeps until the earliest scheduled release; woken early when a batch
        is scheduled or the schedule is dropped (interruption).
        """
        clock = self.get_clock()
        try:
            while True:
                if not self._scheduled:
                    await self._deliver_wake.wait()
                    self._deliver_wake.clear()
                    continue
                wait = nanoseconds_to_seconds(self._scheduled[0][0] - clock.get_time())
                if wait > 0:
                    try:
                        await asyncio.wait_for(self._deliver_wake.wait(), wait)
                    except TimeoutError:
                        continue
                    self._deliver_wake.clear()
                    continue
                _, _, frame = self._scheduled.pop(0)
                await self.push_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._failed = True
            self._contexts.clear()
            self._pending_text.clear()
            self._scheduled.clear()
            await self.push_error(
                "Lipsync delivery failed; disabling lipsync for this session", exception=e
            )

    def _idle_context(self) -> _Context | None:
        """The active context, if it is open, idle, and has final keyframes waiting."""
        if not self._contexts:
            return None
        context = self._contexts[0]
        if context.closing or context.buffer:
            return None
        frontier = context.cursor - _KEYFRAME_HORIZON_SEC
        if frontier <= context.window_start + 1e-9:
            return None
        if any(k.offset < frontier for k in context.pending_keyframes) or any(
            e.offset < frontier for e in context.pending_events
        ):
            return context
        return None

    def _active_context(self) -> _Context | None:
        """The context currently being analyzed, if it has pending work.

        Contexts are processed strictly in creation order: the analyzer holds
        one utterance's state at a time, so a newer context waits until the
        active one is flushed and closed.
        """
        for context in self._contexts:
            return context if context.has_work else None
        return None

    async def _process_context(self, context: _Context):
        generation = self._generation
        if context.buffer:
            data = bytes(context.buffer)
            context.buffer.clear()
            resampled = await context.resampler.resample(
                data, context.sample_rate, ANALYSIS_SAMPLE_RATE
            )
            if generation != self._generation:
                return
            pcm = np.frombuffer(resampled, dtype=np.int16).astype(np.float32)
            pcm /= _INT16_SCALE
            self._snapshot_text(context)
            result = await self._analyzer.analyze(pcm, context.analysis)
            if generation != self._generation:
                return
            self._merge_result(context, result)
            await self._emit_batches(context, generation, final=False)
            if generation != self._generation:
                return

        if context.closing and not context.buffer:
            self._snapshot_text(context)
            result = await self._analyzer.flush(context.analysis)
            if generation != self._generation:
                return
            self._merge_result(context, result)
            await self._emit_batches(context, generation, final=True)
            if context in self._contexts:
                self._contexts.remove(context)

    @staticmethod
    def _snapshot_text(context: _Context):
        context.analysis.text_prior = (
            context.text.snapshot(
                context.t0 or None,
                word_start_pts=context.word_start_pts,
                audio_end=context.ingested_seconds if context.closing else None,
            )
            if context.text is not None
            else None
        )

    def _merge_result(self, context: _Context, result: LipsyncFrameResult):
        skip = context.skip_offset
        if skip:
            for keyframe in result.keyframes:
                keyframe.offset += skip
            for event in result.events:
                event.offset += skip
        context.pending_keyframes.extend(result.keyframes)
        context.pending_events.extend(result.events)
        context.cursor = max(context.cursor, result.processed_up_to + skip)
        if context.drop_silence_pending:
            # Backpressure dropped audio: send clients to neutral.
            context.drop_silence_pending = False
            context.pending_events.append(
                LipsyncEvent(
                    offset=context.cursor,
                    kind=LipsyncEventKind.SILENCE,
                    duration=0.0,
                    confidence=0.1,
                )
            )

    async def _emit_batches(
        self, context: _Context, generation: int, final: bool, idle: bool = False
    ):
        window = self._params.batch_window_ms / 1000.0
        while generation == self._generation:
            # Keyframes before the frontier are final; events confirmed later
            # ride in later batches (see TTSLipsyncFrame).
            frontier = context.cursor - _KEYFRAME_HORIZON_SEC
            first = context.window_start == 0.0
            window_end = context.window_start + (
                min(window, _FIRST_WINDOW_SEC) if first else window
            )
            if final:
                if not context.pending_keyframes and not context.pending_events:
                    return
                # Swallow the whole tail in one final batch.
                last = max(
                    [k.offset for k in context.pending_keyframes]
                    + [e.offset for e in context.pending_events]
                )
                window_end = max(window_end, context.cursor, last + 1e-6)
            elif window_end > frontier + 1e-9:
                if not idle or frontier <= context.window_start + 1e-9:
                    return
                # No audio is arriving to fill the window: emit what is final
                # as a short one.
                window_end = frontier
            # A batch never straddles a playout gap: the audio on each side
            # of it plays at different times.
            for gap_offset, _ in context.gaps:
                if context.window_start < gap_offset < window_end:
                    window_end = gap_offset
                    break

            keyframes = [k for k in context.pending_keyframes if k.offset < window_end]
            events = [e for e in context.pending_events if e.offset < window_end]
            context.pending_keyframes = [
                k for k in context.pending_keyframes if k.offset >= window_end
            ]
            context.pending_events = [e for e in context.pending_events if e.offset >= window_end]

            if keyframes or events:
                await self._push_batch(context, context.window_start, window_end, keyframes, events)
                if idle:
                    self._stats["idle_flushes"] += 1
            context.window_start = window_end

    async def _push_batch(
        self,
        context: _Context,
        window_start: float,
        window_end: float,
        keyframes: list[LipsyncKeyframe],
        events: list[LipsyncEvent],
    ):
        if not self._params.emit_energy or not self._params.emit_pitch:
            for keyframe in keyframes:
                if not self._params.emit_energy:
                    keyframe.energy = 0.0
                if not self._params.emit_pitch:
                    keyframe.pitch = 0.0
        events.sort(key=lambda e: e.offset)

        playout_offset = context.playout_shift(window_start)
        playout_ns = context.t0 + seconds_to_nanoseconds(window_start + playout_offset)
        release_ns = playout_ns - seconds_to_nanoseconds(self._params.scheduling_lead_ms / 1000.0)
        now = self.get_clock().get_time()
        if release_ns < now:
            release_ns = now
            self._stats["release_clamped"] += 1
        frame = TTSLipsyncFrame(
            context_id=context.context_id,
            window_start=window_start,
            window_end=window_end,
            playout_offset=playout_offset,
            playout_ns=playout_ns,
            release_ns=release_ns,
            keyframes=keyframes,
            events=events,
        )
        frame.transport_destination = context.transport_destination

        self._stats["batches_emitted"] += 1
        self._stats["keyframes_emitted"] += len(keyframes)
        self._stats["events_emitted"] += len(events)

        # Even a batch that is already due goes through the delivery task, so
        # batches always leave in release order.
        self._schedule_seq += 1
        bisect.insort(self._scheduled, (release_ns, self._schedule_seq, frame))
        self._deliver_wake.set()
