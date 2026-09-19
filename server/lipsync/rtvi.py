#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Relays lipsync batches to clients as RTVI server-messages.

Place :class:`LipsyncMessageRelay` after ``transport.output()``::

    ... → TTSService → LipsyncProcessor → transport.output() → LipsyncMessageRelay → ...

The processor pushes each ``TTSLipsyncFrame`` at its scheduled release time
(just ahead of the window's playout), and the frame passes straight through
the output transport, so the relay sees every batch exactly once, when it is
due. The relay wraps the batch in an ``RTVIServerMessageFrame``, stamping it
with the window start and the lead still remaining at that moment, which the
stock ``RTVIObserver`` forwards to the client as
``{label: "rtvi-ai", type: "server-message", data: {...}}`` — no observer
subclass or observer params needed. Placing the relay after the transport
keeps that lead stamp as close to the client as the pipeline allows.
"""

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from lipsync.frames import TTSLipsyncFrame

# Discriminator clients use to pick lipsync batches out of the shared
# server-message stream.
LIPSYNC_MESSAGE_TYPE = "bot-tts-lipsync"

# Version 2 added ``ws`` and ``lead`` (and millisecond timing precision); the
# keyframe and event rows are unchanged from version 1.
LIPSYNC_MESSAGE_VERSION = 2


def lipsync_message_data(frame: TTSLipsyncFrame, now_ns: int) -> dict:
    """Pack a TTSLipsyncFrame as compact, JSON-safe server-message data.

    Keyframes and events are positional arrays rather than named fields to
    keep the wire size small. Keyframe order is ``[offset, openness, width,
    rounding, energy, pitch, confidence]``; event order is ``[offset, kind,
    duration, confidence]``. Timing values (offsets, durations, ``t0``,
    ``ws``, ``lead``) are quantized to milliseconds, pose values to two
    decimals (client-side interpolation makes finer precision meaningless).

    Offsets are seconds of audio from the first sample of ``ctx``; ``t0`` is
    a playout shift the client adds to all offsets and to ``ws`` (nonzero
    only after the bot's audio stalled partway through the context). ``ws``
    is the batch's window start and ``lead`` how far ahead of that window's
    playout the message is being sent, measured now (negative when analysis
    is behind playout, as at the start of a turn). A client anchors the
    utterance on them: audio offset ``ws + t0`` plays ``lead`` seconds after
    the message arrives, less network transit.

    Args:
        frame: The batch to pack.
        now_ns: Pipeline clock time at which the message is being sent.
    """

    def q(value: float) -> float:
        return round(value, 2)

    def ms(value: float) -> float:
        return round(value, 3)

    lead = (frame.playout_ns - now_ns) / 1e9 if frame.playout_ns else 0.0
    return {
        "type": LIPSYNC_MESSAGE_TYPE,
        "version": LIPSYNC_MESSAGE_VERSION,
        "ctx": frame.context_id,
        "t0": ms(frame.playout_offset),
        "ws": ms(frame.window_start),
        "lead": ms(lead),
        "kf": [
            [
                ms(k.offset),
                q(k.openness),
                q(k.width),
                q(k.rounding),
                q(k.energy),
                q(k.pitch),
                q(k.confidence),
            ]
            for k in frame.keyframes
        ],
        "ev": [[ms(e.offset), str(e.kind), ms(e.duration), q(e.confidence)] for e in frame.events],
    }


class LipsyncMessageRelay(FrameProcessor):
    """Emits one RTVI server-message per released lipsync batch.

    Forwards every frame unchanged; additionally, each downstream
    ``TTSLipsyncFrame`` is packed with :func:`lipsync_message_data` (stamped
    with the lead remaining at this moment) and pushed as an
    ``RTVIServerMessageFrame``.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSLipsyncFrame):
            now = self.get_clock().get_time()
            await self.push_frame(RTVIServerMessageFrame(data=lipsync_message_data(frame, now)))
