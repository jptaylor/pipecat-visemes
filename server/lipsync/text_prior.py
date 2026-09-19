#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Optional, immutable text observations for one TTS context.

These are inputs to a future text-informed analyzer, not phoneme labels or
assertions about what was spoken. In particular, ``received_after_audio`` is
the amount of audio ingested when a frame arrived, NOT that word's onset.
``pts`` preserves the provider's pipeline-clock timestamp in nanoseconds.

Keep both observations: Pipecat can stamp a sentence before synthesis and
initialize word timestamps only on the first audio chunk. Subtracting the
sentence's pts from each word's pts is not necessarily an audio offset.
No alignment or extra audio lookahead is introduced here.
"""

from dataclasses import dataclass

# Bound text independently of the processor's audio ring and context count.
# A pathological/very long context loses its whole prior, rather than feeding
# a silently truncated sentence to an eventual event detector.
_MAX_TEXT_CHARS = 32_768
_MAX_TEXT_RECORDS = 2_048


@dataclass(frozen=True)
class TextAnchor:
    """One will-be-spoken sentence, possibly arriving after some audio."""

    text: str
    pts: int | None = None
    received_after_audio: float = 0.0


@dataclass(frozen=True)
class TextWord:
    """One provider word timestamp; zero is a valid timestamp."""

    text: str
    pts: int
    received_after_audio: float = 0.0


@dataclass(frozen=True)
class TextPrior:
    """Snapshot supplied with an analysis call; absent text remains ``None``.

    Multiple anchors are allowed: a provider can reuse a context across
    sentences. ``playout_start_pts`` is the processor's clock anchor, which
    can be later than provider timestamps when audio queues behind a context.
    Corpus-only benchmarks have no clock, so leave it unset.
    """

    anchors: tuple[TextAnchor, ...] = ()
    words: tuple[TextWord, ...] = ()
    playout_start_pts: int | None = None


class _TextAccumulator:
    """Processor-owned collector: O(1) appends, snapshots off the frame path."""

    def __init__(self):
        self.anchors: list[TextAnchor] = []
        self.words: list[TextWord] = []
        self.rejected = False
        self._chars = 0
        self._snapshot: TextPrior | None = None

    def append(self, observation: TextAnchor | TextWord) -> bool:
        if self.rejected:
            return False
        if (
            self._chars + len(observation.text) > _MAX_TEXT_CHARS
            or len(self.anchors) + len(self.words) >= _MAX_TEXT_RECORDS
        ):
            self.anchors.clear()
            self.words.clear()
            self._snapshot = None
            self.rejected = True
            return False
        if isinstance(observation, TextAnchor):
            self.anchors.append(observation)
        else:
            self.words.append(observation)
        self._chars += len(observation.text)
        self._snapshot = None
        return True

    def snapshot(self, playout_start_pts: int | None) -> TextPrior | None:
        if self.rejected or not (self.anchors or self.words):
            return None
        if self._snapshot is None or self._snapshot.playout_start_pts != playout_start_pts:
            self._snapshot = TextPrior(
                anchors=tuple(self.anchors),
                words=tuple(self.words),
                playout_start_pts=playout_start_pts,
            )
        return self._snapshot
