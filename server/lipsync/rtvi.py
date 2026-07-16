#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Relays playout-timed lipsync batches to clients as RTVI server-messages.

Place :class:`LipsyncMessageRelay` immediately AFTER ``transport.output()``::

    ... → TTSService → LipsyncProcessor → transport.output() → LipsyncMessageRelay → ...

The output transport holds each pts-carrying ``TTSLipsyncFrame`` in its clock
queue and re-pushes it downstream at presentation time (discarding unplayed
frames on interruption), so every frame this relay sees is already
playout-timed and seen exactly once. The relay wraps the batch in an
``RTVIServerMessageFrame``, which the stock ``RTVIObserver`` forwards to the
client as ``{label: "rtvi-ai", type: "server-message", data: {...}}`` — no
observer subclass or observer params needed.
"""

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from lipsync.frames import TTSLipsyncFrame

# Discriminator clients use to pick lipsync batches out of the shared
# server-message stream.
LIPSYNC_MESSAGE_TYPE = "bot-tts-lipsync"


def lipsync_message_data(frame: TTSLipsyncFrame) -> dict:
    """Pack a TTSLipsyncFrame as compact, JSON-safe server-message data.

    Keyframes and events are positional arrays rather than named fields to
    keep the wire size small, with floats quantized to two decimals (client-
    side smoothing makes finer precision meaningless). Keyframe order is
    ``[offset, openness, width, rounding, energy, pitch, confidence]``; event
    order is ``[offset, kind, duration, confidence]``. Offsets are seconds
    from the first audio of ``ctx``; ``t0`` is a base offset added to all
    offsets (0 in version 1). The schema is versioned: higher-fidelity
    analysis tiers may add fields under a ``version`` bump.
    """

    def q(value: float) -> float:
        return round(value, 2)

    return {
        "type": LIPSYNC_MESSAGE_TYPE,
        "version": 1,
        "ctx": frame.context_id,
        "t0": 0.0,
        "kf": [
            [
                q(k.offset),
                q(k.openness),
                q(k.width),
                q(k.rounding),
                q(k.energy),
                q(k.pitch),
                q(k.confidence),
            ]
            for k in frame.keyframes
        ],
        "ev": [[q(e.offset), str(e.kind), q(e.duration), q(e.confidence)] for e in frame.events],
    }


class LipsyncMessageRelay(FrameProcessor):
    """Emits one RTVI server-message per playout-timed lipsync batch.

    Forwards every frame unchanged; additionally, each downstream
    ``TTSLipsyncFrame`` is packed with :func:`lipsync_message_data` and pushed
    as an ``RTVIServerMessageFrame``.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSLipsyncFrame):
            await self.push_frame(RTVIServerMessageFrame(data=lipsync_message_data(frame)))
