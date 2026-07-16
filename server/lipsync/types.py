#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared types for server-side lipsync analysis.

Defines the articulation signal produced by lipsync analyzers: continuous
keyframes (mouth openness/width/rounding plus energy and pitch) and discrete
events (closures, nasals, silence). The server sends these parameters — never
mouth geometry — and clients map them onto their own mouth model, gating how
far from a neutral shape to commit by each sample's confidence.
"""

from dataclasses import dataclass
from enum import StrEnum


class LipsyncEventKind(StrEnum):
    """Kinds of discrete lipsync events.

    Parameters:
        CLOSURE: Bilabial closure (M/B/P): a brief energy dip inside speech.
        NASAL: Sustained nasal (/m/, /n/, "hmm"): voiced with closed lips;
            overrides the vowel trajectory while active.
        SILENCE: Sustained silence; the client should return to neutral.
    """

    CLOSURE = "closure"
    NASAL = "nasal"
    SILENCE = "silence"


@dataclass
class LipsyncKeyframe:
    """A continuous articulation sample at one instant of a TTS utterance.

    All values are normalized to 0..1. Clients map them to their own mouth
    geometry and interpolate between keyframes.

    Parameters:
        offset: Seconds from utterance (TTS context) start.
        openness: Vertical mouth openness (0 = closed, 1 = fully open).
        width: Horizontal lip spread (0 = back/rounded, 1 = spread /i/).
        rounding: Lip rounding (0 = unrounded, 1 = rounded /u/).
        energy: Log-compressed RMS envelope, for client secondary motion.
        pitch: Pitch normalized within the session range; 0 if unvoiced.
        confidence: Estimation confidence (0..1); gates how far from a
            neutral (schwa) shape the client should commit.
    """

    offset: float
    openness: float
    width: float
    rounding: float
    energy: float
    pitch: float
    confidence: float


@dataclass
class LipsyncEvent:
    """A discrete articulation event within a TTS utterance.

    Parameters:
        offset: Seconds from utterance (TTS context) start.
        kind: The event kind (closure, nasal or silence).
        duration: Duration in seconds; 0 means "until the next keyframe".
        confidence: Detection confidence (0..1).
    """

    offset: float
    kind: LipsyncEventKind
    duration: float
    confidence: float
