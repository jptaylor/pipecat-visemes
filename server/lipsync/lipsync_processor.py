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
buffers, analyzes them in a dedicated task, and emits playout-timed
``TTSLipsyncFrame`` batches. Because the batches carry ``pts``, the output
transport releases them at presentation time through the same clock-queue
mechanism used for word timestamps, including discarding unplayed batches on
interruption. :class:`~lipsync.rtvi.LipsyncMessageRelay`, placed after
``transport.output()``, then delivers each released batch to clients as a
standard RTVI ``server-message`` whose ``data.type`` is ``"bot-tts-lipsync"``.
Works with any ``TTSService``; no provider-specific requirements.
"""

import asyncio

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
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
from lipsync.formant_lipsync_analyzer import (
    EVENT_FINALIZE_HORIZON_SEC,
    FormantLipsyncAnalyzer,
)
from lipsync.frames import LipsyncUpdateSettingsFrame, TTSLipsyncFrame
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

_INT16_SCALE = 32768.0


class LipsyncParams(BaseModel):
    """Configuration parameters for :class:`LipsyncProcessor`.

    Parameters:
        batch_window_ms: Audio time covered by one emitted ``TTSLipsyncFrame``.
        scheduling_lead_ms: How far ahead of playout batches are released,
            giving the client scheduler runway.
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
    """

    batch_window_ms: int = 200
    scheduling_lead_ms: int = 200
    dead_band: float = 0.05
    heartbeat_ms: int = 240
    emit_energy: bool = True
    emit_pitch: bool = True
    enabled: bool = True


class _Context:
    """Internal: per-TTS-context ingest and batching state.

    Offsets are seconds of audio from the context's first sample; ``t0`` is
    the clock time that sample plays. Whenever the transport runs out of the
    context's audio before more arrives (e.g. the LLM stalled mid-response),
    playout resumes when the next chunk arrives, so audio from that offset on
    plays later than ``t0 + offset``. ``gaps`` records each such shift so
    batches can be scheduled, and offsets reported, in playout time.
    """

    def __init__(self, context_id: str | None):
        self.context_id = context_id
        self.buffer = bytearray()
        self.capacity = 0  # set at first audio, once the sample rate is known
        self.sample_rate = 0
        self.t0 = 0
        self.transport_destination: str | None = None
        self.closing = False
        self.analysis = LipsyncAnalysisContext(context_id=context_id, sample_rate=0)
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
    (by default the formant/DSP tier, which works with any TTS provider).
    Emits ``TTSLipsyncFrame`` batches whose ``pts`` schedules delivery just
    ahead of audio playout; a :class:`~lipsync.rtvi.LipsyncMessageRelay`
    after ``transport.output()`` turns each released batch into an RTVI
    ``server-message`` with ``data.type`` ``"bot-tts-lipsync"``.

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
            analyzer: Analysis tier to use. Defaults to
                :class:`~lipsync.formant_lipsync_analyzer.FormantLipsyncAnalyzer`,
                the provider-universal DSP tier, configured with this
                processor's ``dead_band`` and ``heartbeat_ms``.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._params = params or LipsyncParams()
        self._analyzer = analyzer or FormantLipsyncAnalyzer(
            dead_band=self._params.dead_band,
            heartbeat_ms=self._params.heartbeat_ms,
        )

        # Contexts in creation order. A TTS service may reopen a context id it
        # already closed (e.g. when it reuses one id for a whole turn and the
        # LLM stalls past its idle timeout), so the same id can appear more
        # than once: the newest open entry is the live one.
        self._contexts: list[_Context] = []
        self._sample_rate = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._generation = 0
        # Playout end of the most recently ingested audio, so a context that
        # queues behind another one is anchored where that audio ends rather
        # than at its own arrival time.
        self._last_playout_end = 0
        self._stopping = False
        self._failed = False
        self._stats = {
            "batches_emitted": 0,
            "keyframes_emitted": 0,
            "events_emitted": 0,
            "pts_clamped": 0,
            "playout_gaps": 0,
            "bytes_dropped": 0,
            "contexts_opened": 0,
            "contexts_evicted": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Diagnostic counters (batches, drops, clamps); read-only snapshot."""
        return dict(self._stats)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._sample_rate = setup.audio_out_sample_rate

    async def cleanup(self):
        """Clean up the processor and cancel the analysis task."""
        await super().cleanup()
        if self._task:
            await self.cancel_task(self._task)
            self._task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Forward all frames unmodified, tapping TTS frames for analysis.

        The frame path only ever copies audio bytes into a ring buffer; all
        analysis happens in the processor's own task.

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
        self._contexts.append(_Context(frame.context_id))
        self._stats["contexts_opened"] += 1

    def _handle_tts_audio(self, frame: TTSAudioRawFrame):
        context = self._open_context(frame.context_id)
        if context is None or not self._params.enabled or self._failed:
            return
        now = self.get_clock().get_time()
        if context.t0 == 0:
            # Audio plays when it arrives unless earlier audio is still
            # queued, in which case it plays when that audio ends.
            context.t0 = max(now, self._last_playout_end)
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
        context.buffer += frame.audio
        context.analysis.samples_seen += frame.num_frames
        self._last_playout_end = max(self._last_playout_end, context.playout_end)
        self._wake.set()

    def _handle_tts_stopped(self, frame: TTSStoppedFrame):
        context = self._open_context(frame.context_id)
        if context is None:
            return
        context.closing = True
        self._wake.set()

    async def _handle_interruption(self):
        # The transport discards unplayed batches from its clock queue itself.
        await self._discard_analysis_state()

    async def _handle_update_settings(self, frame: LipsyncUpdateSettingsFrame):
        for key, value in frame.settings.items():
            if key in LipsyncParams.model_fields:
                setattr(self._params, key, value)
            else:
                logger.warning(f"{self} unknown lipsync setting: {key}")
        if not self._params.enabled:
            await self._discard_analysis_state()

    async def _discard_analysis_state(self):
        """Drop buffers and in-flight work; adaptive analyzer state survives."""
        self._generation += 1
        self._contexts.clear()
        # Queued audio was discarded too, so nothing is playing after now.
        self._last_playout_end = 0
        if self._task:
            await self._analyzer.reset()

    async def _stop(self):
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

    async def _cancel(self):
        if self._task:
            await self.cancel_task(self._task)
            self._task = None
        self._contexts.clear()

    #
    # Analysis task
    #

    async def _analysis_task_handler(self):
        """Drain buffered audio, analyze, batch and emit playout-timed frames.

        Single long-lived task per processor. Parks on an event while there
        is no work (bot silence, or ``enabled`` False), so the idle cost is
        zero. Any unexpected error disables lipsync for the session and
        reports a non-fatal error upstream; the pipeline keeps running.
        """
        try:
            while True:
                while (context := self._active_context()) is not None:
                    await self._process_context(context)
                if self._stopping:
                    break
                await self._wake.wait()
                self._wake.clear()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._failed = True
            self._contexts.clear()
            await self.push_error(
                "Lipsync analysis failed; disabling lipsync for this session", exception=e
            )

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
            result = await self._analyzer.analyze(pcm, context.analysis)
            if generation != self._generation:
                return
            self._merge_result(context, result)
            await self._emit_batches(context, generation, final=False)
            if generation != self._generation:
                return

        if context.closing and not context.buffer:
            result = await self._analyzer.flush(context.analysis)
            if generation != self._generation:
                return
            self._merge_result(context, result)
            await self._emit_batches(context, generation, final=True)
            if context in self._contexts:
                self._contexts.remove(context)

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

    async def _emit_batches(self, context: _Context, generation: int, final: bool):
        window = self._params.batch_window_ms / 1000.0
        while generation == self._generation:
            window_end = context.window_start + window
            if final:
                if not context.pending_keyframes and not context.pending_events:
                    return
                # Swallow the whole tail in one final batch.
                last = max(
                    [k.offset for k in context.pending_keyframes]
                    + [e.offset for e in context.pending_events]
                )
                window_end = max(window_end, context.cursor, last + 1e-6)
            # A batch never straddles a playout gap: the audio on each side
            # of it plays at different times.
            for gap_offset, _ in context.gaps:
                if context.window_start < gap_offset < window_end:
                    window_end = gap_offset
                    break
            if not final and context.cursor < window_end + EVENT_FINALIZE_HORIZON_SEC:
                return

            keyframes = [k for k in context.pending_keyframes if k.offset < window_end]
            events = [e for e in context.pending_events if e.offset < window_end]
            context.pending_keyframes = [
                k for k in context.pending_keyframes if k.offset >= window_end
            ]
            context.pending_events = [e for e in context.pending_events if e.offset >= window_end]

            if keyframes or events:
                await self._push_batch(context, context.window_start, window_end, keyframes, events)
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
        frame = TTSLipsyncFrame(
            context_id=context.context_id,
            window_start=window_start,
            window_end=window_end,
            playout_offset=playout_offset,
            keyframes=keyframes,
            events=events,
        )
        frame.transport_destination = context.transport_destination

        lead = seconds_to_nanoseconds(self._params.scheduling_lead_ms / 1000.0)
        pts = context.t0 + seconds_to_nanoseconds(window_start + playout_offset) - lead
        now = self.get_clock().get_time()
        if pts < now:
            pts = now
            self._stats["pts_clamped"] += 1
        frame.pts = pts

        self._stats["batches_emitted"] += 1
        self._stats["keyframes_emitted"] += len(keyframes)
        self._stats["events_emitted"] += len(events)

        await self.push_frame(frame)
