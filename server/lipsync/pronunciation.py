"""Small, shared CMUdict lookup; no downloaded models or per-session dictionary.

The gzip payload is a sorted table of offsets followed by word\0phone-bytes\0
records. Binary search avoids creating 126k Python strings/tuples on startup.
Only the bounded lookup cache holds decoded words. See data/README.md.
"""

import gzip
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PHONES = tuple(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split()
)
BILABIALS = frozenset(("P", "B", "M"))
NASALS = frozenset(("M", "N", "NG"))
ROUNDED = frozenset(("UW", "OW", "W"))
LABIODENTALS = frozenset(("F", "V"))
_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)*")
_MAGIC = b"LPCMUD1\0"


@lru_cache(maxsize=1)
def load_lexicon() -> bytes:
    """Load once on opt-in analyzer startup, before the first audio arrives."""
    data = gzip.decompress((Path(__file__).parent / "data" / "cmudict.bin.gz").read_bytes())
    if data[:8] != _MAGIC:
        raise ValueError("unsupported packed CMUdict format")
    return data


@dataclass(frozen=True)
class Pronunciation:
    phones: tuple[str, ...]
    certain: bool


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_WORDS.findall(text.lower().replace("’", "'")))


def supported_text(text: str) -> bool:
    """Numbers, markup, symbols and non-English letters need TTS normalization."""
    return bool(tokenize(text)) and not re.search(r"[^a-zA-Z\s'’.,!?;:\-—…()\"]", text)


@lru_cache(maxsize=2048)
def pronounce(word: str) -> Pronunciation:
    word = word.lower().replace("’", "'")
    # CMUdict's lexical "hmm" is HH M; an extended written hum has no vowel.
    if re.fullmatch(r"h?m{2,}", word):
        return Pronunciation(("M",), True)
    data = load_lexicon()
    target = word.encode("ascii", errors="replace")
    lo, hi = 0, struct.unpack_from("<I", data, 8)[0]
    while lo < hi:
        mid = (lo + hi) // 2
        offset = struct.unpack_from("<I", data, 12 + mid * 4)[0]
        end = data.index(0, offset)
        candidate = data[offset:end]
        if candidate < target:
            lo = mid + 1
        elif candidate > target:
            hi = mid
        else:
            phone_end = data.index(0, end + 1)
            return Pronunciation(tuple(PHONES[n - 1] for n in data[end + 1 : phone_end]), True)
    return Pronunciation(_letter_to_sound(word), False)


def _letter_to_sound(word: str) -> tuple[str, ...]:
    """Conservative English OOV approximation, NEVER sufficient for event overrides.

    Keep place-bearing consonants and common digraphs. The confidence flag
    causes a sentence containing OOV words to retain DSP decisions; guessed
    names must not supply categorical vetoes or injected closures.
    """
    pairs = {"sh": "SH", "ch": "CH", "th": "TH", "ph": "F", "ng": "NG", "ee": "IY", "oo": "UW"}
    singles = dict(
        zip(
            "abcdefghijklmnopqrstuvwxyz",
            "AE B K D EH F G HH IH JH K L M N OW P K R S T AH V W K Y Z".split(),
        )
    )
    phones = []
    i = 0
    while i < len(word):
        pair = word[i : i + 2]
        if pair in pairs:
            phones.append(pairs[pair])
            i += 2
        else:
            if word[i] in singles:
                phones.append(singles[word[i]])
            i += 1
    return tuple(phones)
