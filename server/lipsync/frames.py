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

from pipecat.frames.frames import ControlFrame, DataFrame, format_pts

from lipsync.types import LipsyncEvent, LipsyncKeyframe


@dataclass
class TTSLipsyncFrame(DataFrame):
    """Lipsync keyframes and events covering one window of TTS audio.

    Emitted by :class:`~lipsync.lipsync_processor.LipsyncProcessor`, one
    frame per batch window of analyzed TTS audio. ``pts`` is assigned before
    the frame is pushed, so the output transport releases it at presentation
    time via its clock queue — ahead of audio playout by the processor's
    scheduling lead — exactly like word timestamps. On interruption, unplayed
    frames are discarded along with the corresponding audio.

    Offsets are seconds of audio from the first sample of ``context_id``. If
    the transport ran out of the context's audio partway through (e.g. the LLM
    stalled mid-response), audio after the gap plays later than its offset
    implies; ``playout_offset`` carries that shift for this window.

    Parameters:
        context_id: TTS context the audio was measured from.
        window_start: Window start in seconds from context start (inclusive).
        window_end: Window end in seconds from context start (exclusive).
        playout_offset: Seconds to add to this window's offsets to get playout
            time relative to the context's first sample; zero unless playout
            gapped earlier in the context.
        keyframes: Continuous articulation keyframes within the window.
        events: Discrete closure/nasal/silence events within the window.
    """

    context_id: str | None = None
    window_start: float = 0.0
    window_end: float = 0.0
    playout_offset: float = 0.0
    keyframes: list[LipsyncKeyframe] = field(default_factory=list)
    events: list[LipsyncEvent] = field(default_factory=list)

    def __str__(self):
        pts = format_pts(self.pts)
        return (
            f"{self.name}(pts: {pts}, context: {self.context_id}, "
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
