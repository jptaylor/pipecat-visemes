#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Server-side lipsync for Pipecat voice bots.

Two processors, one on each side of the output transport::

    from lipsync import LipsyncMessageRelay, LipsyncProcessor

    pipeline = Pipeline(
        [..., tts, LipsyncProcessor(), transport.output(), LipsyncMessageRelay(), ...]
    )

``LipsyncProcessor(text_events=True)`` also reads the TTS service's own text
for closure and nasal events (experimental, English).
"""

from lipsync.frames import LipsyncUpdateSettingsFrame, TTSLipsyncFrame
from lipsync.lipsync_processor import LipsyncParams, LipsyncProcessor
from lipsync.rtvi import LipsyncMessageRelay

__all__ = [
    "LipsyncMessageRelay",
    "LipsyncParams",
    "LipsyncProcessor",
    "LipsyncUpdateSettingsFrame",
    "TTSLipsyncFrame",
]
