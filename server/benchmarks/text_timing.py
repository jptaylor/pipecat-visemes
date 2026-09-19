"""Replay captured text availability, never future word timestamps, for a clip."""

from bisect import bisect_right

from lipsync.text_prior import TextAnchor, TextPrior, TextWord


class FixtureTextInputs:
    def __init__(self, clip, context):
        self.clip, self.context = clip, context
        self._key = None
        timing = clip.text_timing
        self._anchors = (
            tuple(
                TextAnchor(text, None if pts is None else round(pts * 1e9), at)
                for text, pts, at in timing.get("anchors", [])
            )
            if timing is not None
            else (TextAnchor(clip.sentence.text),)
        )
        self._words = (
            tuple(TextWord(text, round(pts * 1e9), at) for text, pts, at in timing.get("words", []))
            if timing is not None
            else ()
        )
        self._anchor_arrivals = [a.received_after_audio for a in self._anchors]
        self._word_arrivals = [w.received_after_audio for w in self._words]

    def ingest(self, samples: int, *, final: bool = False):
        seconds = samples / self.clip.sample_rate
        anchor_count = bisect_right(self._anchor_arrivals, seconds)
        word_count = bisect_right(self._word_arrivals, seconds)
        end = len(self.clip.pcm) / 2 / self.clip.sample_rate if final else None
        key = anchor_count, word_count, end
        if key != self._key:
            self._key = key
            self.context.text_prior = (
                TextPrior(
                    anchors=self._anchors[:anchor_count],
                    words=self._words[:word_count],
                    word_start_pts=0,
                    audio_end=end,
                )
                if anchor_count or word_count
                else None
            )
        self.context.samples_seen = samples
