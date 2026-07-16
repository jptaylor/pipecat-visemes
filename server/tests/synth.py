#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Synthetic speech-signal helpers shared across the lipsync tests."""

import numpy as np

from lipsync.dsp import ANALYSIS_SAMPLE_RATE, PRE_EMPHASIS


def synth_vowel(
    f1, f2, f3=2900, f0=100, secs=0.5, fs=ANALYSIS_SAMPLE_RATE, bandwidths=None, extra=()
):
    """Impulse train through glottal tilt + cascaded 2nd-order resonators.

    The one-pole tilt stage models the natural spectral slope of voiced speech
    that the analyzer's pre-emphasis is designed to undo; without it, LPC on a
    flat impulse train is biased upward, especially for low F1. ``extra``
    appends additional (freq, bandwidth) resonators (e.g. a nasal pole).
    """
    bandwidths = bandwidths or (90, 110, 150)
    n = int(secs * fs)
    x = np.zeros(n, dtype=np.float32)
    x[:: int(fs / f0)] = 1.0
    y = np.zeros_like(x)
    for i in range(n):  # glottal tilt: one-pole lowpass matching PRE_EMPHASIS
        y[i] = x[i] + PRE_EMPHASIS * y[i - 1]
    x = y
    for fc, bw in list(zip((f1, f2, f3), bandwidths)) + list(extra):
        r = np.exp(-np.pi * bw / fs)
        c1, c2 = 2 * r * np.cos(2 * np.pi * fc / fs), -r * r
        y = np.zeros_like(x)
        for i in range(n):  # small-n IIR; test-only, clarity > speed
            y[i] = x[i] + c1 * y[i - 1] + c2 * y[i - 2]
        x = y
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)
