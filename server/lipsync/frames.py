#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""App-local pipeline frames for lipsync.

These extend pipecat's public frame bases; they live here (rather than in
``pipecat.frames.frames``) because this project depends on released
pipecat-ai instead of carrying a fork.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import ControlFrame, SystemFrame, format_pts

from lipsync.types import LipsyncEvent, LipsyncKeyframe


@dataclass
class TTSLipsyncFrame(SystemFrame):
    """Lipsync keyframes and events covering one window of TTS audio.

    Emitted by :class:`~lipsync.lipsync_processor.LipsyncProcessor`, one
    frame per batch window of analyzed TTS audio. The processor schedules
    delivery itself on the pipeline clock and pushes the frame at
    ``release_ns`` (the window's playout time less the scheduling lead). It
    is a system frame so that the output transport forwards it at once: as a
    data frame it would either wait in the transport's clock queue behind an
    earlier-queued word-timestamp frame with a later timestamp, or be synced
    to the audio queue and released only after the audio ahead of it had
    played. Ordering (release times only advance) and interruption (frames
    not yet released are dropped with the discarded audio) are the
    processor's job.

    Offsets are seconds of audio from the first sample of ``context_id``. If
    the transport ran out of the context's audio partway through (e.g. the LLM
    stalled mid-response), audio after the gap plays later than its offset
    implies; ``playout_offset`` carries that shift for this window.

    Keyframes always lie inside the window. Events may precede it: a closure
    is confirmed only once speech resumes after it and a silence only once it
    has lasted long enough, so such events ride in the first batch emitted
    after they are known (append-only; clients key events by offset, not by
    window).

    Parameters:
        context_id: TTS context the audio was measured from.
        window_start: Window start in seconds from context start (inclusive).
        window_end: Window end in seconds from context start (exclusive).
        playout_offset: Seconds to add to this window's offsets to get playout
            time relative to the context's first sample; zero unless playout
            gapped earlier in the context.
        playout_ns: Pipeline clock time (ns) at which ``window_start`` plays.
        release_ns: Pipeline clock time (ns) the frame was scheduled to be
            pushed at: ``playout_ns`` less the scheduling lead, or the
            emission time when that had already passed.
        keyframes: Continuous articulation keyframes within the window.
        events: Discrete closure/nasal/silence events known when the window
            was emitted; offsets are below ``window_end`` but may precede
            ``window_start``.
    """

    context_id: str | None = None
    window_start: float = 0.0
    window_end: float = 0.0
    playout_offset: float = 0.0
    playout_ns: int = 0
    release_ns: int = 0
    keyframes: list[LipsyncKeyframe] = field(default_factory=list)
    events: list[LipsyncEvent] = field(default_factory=list)

    def __str__(self):
        release = format_pts(self.release_ns or None)
        return (
            f"{self.name}(release: {release}, context: {self.context_id}, "
            f"window: [{self.window_start:.3f}, {self.window_end:.3f}), "
            f"keyframes: {len(self.keyframes)}, events: {len(self.events)})"
        )


@dataclass
class LipsyncUpdateSettingsFrame(ControlFrame):
    """Frame for updating lipsync processor settings at runtime.

    Consumed by :class:`~lipsync.lipsync_processor.LipsyncProcessor`.
    Setting ``enabled`` to False parks analysis (passthrough only, zero CPU)
    without removing the processor from the pipeline.

    Parameters:
        settings: Setting name to value mappings; names match
            :class:`~lipsync.lipsync_processor.LipsyncParams`
            fields (e.g. ``{"enabled": False}``).
    """

    settings: Mapping[str, Any] = field(default_factory=dict)
