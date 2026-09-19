#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base class for lipsync analyzers.

A lipsync analyzer turns raw TTS audio into an articulation signal
(:class:`~lipsync.types.LipsyncKeyframe` and
:class:`~lipsync.types.LipsyncEvent`).
:class:`~lipsync.lipsync_processor.LipsyncProcessor` owns
buffering, timing and batching; analyzers only measure. The formant analyzer
is the only implementation and the design of record; the interface exists so
measurement stays separable from delivery (a stub analyzer is enough to test
the processor) and any alternative would leave the wire format unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from lipsync.text_prior import TextPrior
from lipsync.types import LipsyncEvent, LipsyncKeyframe


@dataclass
class LipsyncAnalysisContext:
    """Per-utterance analysis state, keyed by TTS context.

    Parameters:
        context_id: TTS context this state belongs to.
        sample_rate: Source sample rate of the TTS audio in Hz.
        samples_seen: Total source samples ingested for this context. Event
            offsets are derived from this counter, never from arrival times,
            so they stay sample-accurate when TTS generates faster than real
            time.
        text_prior: Optional snapshot of sentence anchors and word timestamps.
            Text-unaware analyzers can ignore it. Raw clock timestamps are
            observations, not verified audio/phone alignment.
    """

    context_id: str | None
    sample_rate: int
    samples_seen: int = 0
    text_prior: TextPrior | None = None


@dataclass
class LipsyncFrameResult:
    """Keyframes and events produced by one analyzer call.

    Parameters:
        keyframes: Continuous articulation keyframes, ordered by offset.
        events: Discrete events, ordered by offset.
        processed_up_to: Seconds of the utterance fully analyzed so far. The
            processor uses this cursor to decide when a batch window can be
            finalized (deferred event confirmation may still add events with
            earlier offsets until the cursor passes them).
    """

    keyframes: list[LipsyncKeyframe] = field(default_factory=list)
    events: list[LipsyncEvent] = field(default_factory=list)
    processed_up_to: float = 0.0


class BaseLipsyncAnalyzer(ABC):
    """Abstract base class for lipsync analyzers.

    Implementations consume PCM audio incrementally and return articulation
    keyframes and events with offsets in seconds from utterance start. All
    methods are called from the lipsync processor's analysis task, never from
    the frame processing path.
    """

    @abstractmethod
    async def start(self, sample_rate: int):
        """Prepare the analyzer for a session.

        Called once at pipeline start, before any audio is analyzed. This is
        the place to allocate buffers and prime any per-session state.

        Args:
            sample_rate: Source sample rate of the TTS audio in Hz.
        """
        pass

    @abstractmethod
    async def analyze(self, pcm: np.ndarray, context: LipsyncAnalysisContext) -> LipsyncFrameResult:
        """Analyze a chunk of PCM audio from one TTS context.

        Args:
            pcm: Mono float32 samples at the fixed 16 kHz analysis rate. The
                caller (the lipsync processor) owns resampling from the
                context's source rate.
            context: Analysis state for the TTS context the audio belongs to.

        Returns:
            Keyframes and events measured from the chunk. May be empty when
            the chunk is smaller than one analysis hop.
        """
        pass

    @abstractmethod
    async def flush(self, context: LipsyncAnalysisContext) -> LipsyncFrameResult:
        """Flush any partial analysis window at the end of an utterance.

        Args:
            context: Analysis state for the TTS context being closed.

        Returns:
            Keyframes and events remaining in the analysis window.
        """
        pass

    @abstractmethod
    async def reset(self):
        """Reset per-utterance state after an interruption.

        Adaptive per-voice state (e.g. learned formant ranges) should be
        preserved: the voice has not changed, only the utterance was
        discarded.
        """
        pass
