"""Conservative stage-1 categorical priors, with zero extra audio lookahead.

Untimed sentences supply phone-inventory vetoes and explicit hum recognition.
Confirmed word intervals supply uniform phone placement. No total-duration
guess is made for text-only providers; no already-emitted keyframe is revised.
Late, untrusted and inconsistent observations leave those regions to DSP.
This is a pronunciation/timing prior, not speech recognition: a plausible but
incorrect transcript cannot always be identified acoustically.
"""

from bisect import bisect_right
from dataclasses import dataclass

from lipsync.pronunciation import BILABIALS, NASALS, pronounce, supported_text, tokenize
from lipsync.text_prior import TextPrior
from lipsync.types import LipsyncEvent, LipsyncEventKind

TOLERANCE = 0.060
MAX_WORDS = 256
MAX_PHONES = 2048
MAX_WORD_LENGTH = 64


@dataclass(frozen=True, slots=True)
class _Segment:
    start: float
    words: tuple[str, ...]
    trusted: bool
    has_bilabial: bool
    has_nasal: bool
    hum: bool


@dataclass(frozen=True, slots=True)
class _Span:
    start: float
    end: float
    phone: str
    identity: tuple[int, int]
    word_start: float
    word_end: float
    available_at: float


class TextEvents:
    def __init__(self):
        self.stats = dict.fromkeys(
            (
                "segments",
                "untrusted_segments",
                "mismatched_contexts",
                "late_contexts",
                "multi_anchor_contexts",
                "oversized_contexts",
                "timed_hops",
                "inventory_hops",
                "closures_injected",
                "closures_vetoed",
                "late_closures",
                "acoustic_rejected_hops",
            ),
            0,
        )
        self.reset()

    def reset(self):
        self._prior = None
        self._segments: list[_Segment] = []
        self._starts: list[float] = []
        self._spans: list[_Span] = []
        self._span_starts: list[float] = []
        self._claimed: set[tuple[int, int]] = set()
        self._late: set[tuple[int, int]] = set()
        self._rms: list[tuple[float, float]] = []
        self._rejected = False
        self._anchors_seen = 0
        self._available: dict[tuple[int, int], float] = {}

    def prepare(self, prior: TextPrior | None, cursor: float = 0.0):
        analysis_cursor = cursor
        if prior is self._prior:
            return
        self._prior = prior
        self._spans, self._span_starts = [], []
        self._segments, self._starts = [], []
        if prior is None or self._rejected or not prior.anchors:
            return
        if prior.anchors[0].received_after_audio > TOLERANCE:
            self.stats["late_contexts"] += 1
            self._rejected = True
            return
        if len(prior.anchors) > 1:
            # Samples-at-arrival are NOT sentence onsets. Shared contexts
            # require sentence alignment before inventory vetoes are safe.
            self.stats["multi_anchor_contexts"] += 1
            self._rejected = True
            return
        all_words = []
        for index, anchor in enumerate(prior.anchors):
            words = tokenize(anchor.text)
            if len(words) > MAX_WORDS or any(len(word) > MAX_WORD_LENGTH for word in words):
                # Do not let pathological words fill the shared decoded cache
                # with thousands of very large OOV strings/phone tuples.
                self.stats["oversized_contexts"] += 1
                self._rejected = True
                return
            pronunciations = [pronounce(w) for w in words] if supported_text(anchor.text) else []
            trusted = bool(pronunciations) and all(p.certain for p in pronunciations)
            phones = tuple(p for word in pronunciations for p in word.phones)
            if len(phones) > MAX_PHONES:
                self.stats["oversized_contexts"] += 1
                self._rejected = True
                return
            segment = _Segment(
                anchor.received_after_audio,
                words,
                trusted,
                bool(BILABIALS.intersection(phones)),
                bool(NASALS.intersection(phones)),
                trusted and bool(phones) and all(p == "M" for p in phones),
            )
            self._segments.append(segment)
            self._starts.append(segment.start)
            all_words.extend(words)
            if index >= self._anchors_seen:
                self.stats["segments"] += 1
                self.stats["untrusted_segments"] += not trusted
        self._anchors_seen = len(prior.anchors)
        # Word frames are post-transformation text; compare the observed prefix
        # before allowing any timed override. Equal starts are common ("my mother").
        observed = []
        for word in prior.words:
            for token in tokenize(word.text):
                observed.append((token, word.pts))
        if [word for word, _ in observed] != all_words[: len(observed)]:
            self.stats["mismatched_contexts"] += 1
            self._rejected = True
            self._segments, self._starts = [], []
            return
        if prior.word_start_pts is None or not observed:
            return
        points = [(word, (pts - prior.word_start_pts) / 1e9) for word, pts in observed]
        if any(t < -0.020 for _, t in points) or any(
            b[1] < a[1] for a, b in zip(points, points[1:])
        ):
            # Unknown clock origin: retain only the untimed inventory prior.
            return
        i = 0
        while i < len(points):
            end = i + 1
            while end < len(points) and abs(points[end][1] - points[i][1]) < 1e-6:
                end += 1
            start_time = max(0.0, points[i][1])
            end_time = points[end][1] if end < len(points) else prior.audio_end
            # Do not fabricate the last word's duration while audio is streaming.
            if end_time is not None and end_time > start_time:
                group = [pronounce(word).phones for word, _ in points[i:end]]
                count = sum(map(len, group))
                duration = end_time - start_time
                if count and 0.015 <= duration / count <= 0.35:
                    cursor = start_time
                    step = duration / count
                    for word_index, phones in enumerate(group, i):
                        word_end = cursor + len(phones) * step
                        word_start = cursor
                        for phone_index, phone in enumerate(phones):
                            identity = (word_index, phone_index)
                            available = self._available.setdefault(identity, analysis_cursor)
                            self._spans.append(
                                _Span(
                                    cursor,
                                    cursor + step,
                                    phone,
                                    identity,
                                    word_start,
                                    word_end,
                                    available,
                                )
                            )
                            if (
                                phone in BILABIALS
                                and available - cursor > TOLERANCE
                                and identity not in self._late
                            ):
                                self.stats["late_closures"] += 1
                                self._late.add(identity)
                            cursor += step
            i = end
        self._span_starts = [s.start for s in self._spans]

    def segment_at(self, offset: float) -> _Segment | None:
        i = bisect_right(self._starts, offset) - 1
        if i < 0 or not self._segments[i].trusted:
            return None
        return self._segments[i]

    def span_at(self, offset: float) -> _Span | None:
        i = bisect_right(self._span_starts, offset) - 1
        if i >= 0 and offset < self._spans[i].end:
            return self._spans[i]
        return None

    def hint(self, offset: float, *, voiced: bool, f1: float, rms: float, silence_gate: float):
        """Return (nasal evidence override or None, current supported phone)."""
        segment = self.segment_at(offset)
        if segment is None:
            return None, None
        span = self.span_at(offset)
        if span is None:
            self.stats["inventory_hops"] += 1
            if segment.hum:
                # Obvious open vowels contradict a written hum. This is only
                # a coarse gate; low vowels and nasalized speech remain ambiguous.
                if voiced and f1 > 500:
                    self.stats["acoustic_rejected_hops"] += 1
                    return None, None
                return voiced and rms >= silence_gate, "M"
            return (None if segment.has_nasal else False), None
        self.stats["timed_hops"] += 1
        if span.phone in NASALS:
            if voiced and f1 > 500:
                self.stats["acoustic_rejected_hops"] += 1
                return None, None
            # /n, ng/ do not imply closed lips. They permit DSP nasal evidence;
            # only /m/ and explicit hums supply a positive closure cue.
            return (voiced and rms >= silence_gate if span.phone == "M" else None), span.phone
        return False, span.phone

    def allow_closure(self, offset: float) -> bool:
        segment = self.segment_at(offset)
        if segment is None:
            return True
        if not segment.has_bilabial:
            self.stats["closures_vetoed"] += 1
            return False
        span = self.span_at(offset)
        if span is None or offset < span.available_at - TOLERANCE:
            return True  # timing unavailable: keep DSP recall
        candidates = [
            s
            for s in self._nearby(offset)
            if s.phone in BILABIALS and abs(s.start - offset) <= TOLERANCE
        ]
        if not candidates:
            self.stats["closures_vetoed"] += 1
            return False
        target = min(candidates, key=lambda s: abs(s.start - offset))
        if target.identity in self._claimed:
            return False
        self._claimed.add(target.identity)
        return True

    def _nearby(self, offset: float):
        start = max(0, bisect_right(self._span_starts, offset - TOLERANCE) - 1)
        end = bisect_right(self._span_starts, offset + TOLERANCE)
        return self._spans[start:end]

    def inject_closures(self, offset: float, rms: float, audible: bool) -> list[LipsyncEvent]:
        self._rms.append((offset, rms))
        if len(self._rms) > 4:
            self._rms.pop(0)
        if self.segment_at(offset) is None or not audible:
            return []
        events = []
        for span in self._nearby(offset):
            if span.phone not in BILABIALS or span.identity in self._claimed or span.start > offset:
                continue
            if offset - span.start > TOLERANCE:
                continue
            self._claimed.add(span.identity)
            # Refine to an observed energy minimum only if it is a real dip;
            # a shallow /m/ still gets its expected onset.
            nearby = [(t, energy) for t, energy in self._rms if abs(t - span.start) <= TOLERANCE]
            at = span.start
            if nearby:
                t, low = min(nearby, key=lambda row: row[1])
                peak = max(energy for _, energy in nearby)
                if low < 0.7 * peak:
                    at = t
            events.append(
                LipsyncEvent(
                    at, LipsyncEventKind.CLOSURE, max(0.02, min(0.12, span.end - span.start)), 0.6
                )
            )
            self.stats["closures_injected"] += 1
        return events
