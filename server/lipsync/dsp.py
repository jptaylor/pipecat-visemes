#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NumPy-only DSP primitives for formant-based lipsync analysis.

Vendored implementations (no scipy/librosa dependency) of the signal
processing routines needed by
:class:`~pipecat.audio.lipsync.formant_lipsync_analyzer.FormantLipsyncAnalyzer`:
LPC via autocorrelation + Levinson-Durbin, formant extraction from LPC roots,
pitch by normalized cross-correlation (or the LPC residual), spectral features for nasal detection, and a
streaming quantile estimator for adaptive normalization.

Analysis always runs at 16 kHz mono (audio is resampled on ingest), which
fixes the LPC order and band constants below. All routines operate on float32.
Per-frame temporaries are kept small; ``np.roots`` and the FFTs allocate
internally and are the accepted exceptions to allocation-free operation.
"""

from functools import lru_cache
from typing import NamedTuple

import numpy as np

# Fixed internal analysis rate in Hz; TTS audio is resampled to this on ingest.
ANALYSIS_SAMPLE_RATE = 16000

# 25 ms analysis window, 20 ms hop (in samples at 16 kHz). This frame drives
# energy, closure/silence detection and the nasal spectral features, so event
# timing keeps its 25 ms resolution.
FRAME_SIZE = 400
HOP_SIZE = 320

# LPC/formant (and residual-pitch) window, centered on the same hop as the
# 25 ms frame. Decoupled so a longer LPC window cannot smear event timing;
# equal to FRAME_SIZE means one shared frame. Must be >= FRAME_SIZE with an
# even difference.
LPC_FRAME_SIZE = 400

# Raw frame read by the NCC pitch detector, centered on the same hop (same
# size constraints). 40 ms gives the NCC a 374-sample reference segment
# instead of 134: voicing agreement with Praat 0.926 -> 0.955 on the corpus.
PITCH_FRAME_SIZE = 640

# First-order pre-emphasis coefficient applied before LPC.
PRE_EMPHASIS = 0.97

# LPC model order at 16 kHz: two poles per kHz of bandwidth. Order 12 is
# pole-starved on real speech (resonances above F3 take poles, F1/F2 merge
# into broad roots and the slots empty out); see
# plans/deep-review-2026-09-results.md for the 12/14/16/18 A/B.
LPC_ORDER = 16

# LPC roots count as formants within these frequency/bandwidth bounds.
FORMANT_MIN_HZ = 200
FORMANT_MAX_HZ = 3500
FORMANT_MAX_BANDWIDTH_HZ = 500

# Formant slot bands: candidate roots are assigned to F1/F2/F3 by band
# membership, preferring the candidate closest to the previous estimate (or a
# nominal center) — never by pure ascending rank, so a damped F2 cannot
# promote F3 into its slot and a spurious low pole cannot shift every slot up.
F1_BAND_HZ = (200.0, 1000.0)
F2_BAND_HZ = (650.0, 2600.0)  # lower edge below male /u,o/ F2 (~700 Hz)
F3_BAND_HZ = (1800.0, 3500.0)
_SLOT_NOMINAL_HZ = (550.0, 1500.0, 2500.0)
# Per-slot (F1, F2, F3) weight of the absolute prior in slot assignment: each
# filled slot also pays ``weight * |f - nominal| / nominal``, so track
# continuity alone can never lock a slot onto the wrong root (F2 riding the
# F3 track) while the right one is among the candidates. 0.0 disables that
# slot's prior (pure previous-frame continuity).
SLOT_PRIOR_WEIGHTS = (0.0, 1.0, 1.0)
_F2_MIN_ABOVE_F1_HZ = 150.0
_F3_MIN_ABOVE_F2_HZ = 200.0
# F1 physically broadens with mouth opening: open /ɑ/ roots measure 600-730 Hz
# bandwidth on real TTS speech, beyond the strict formant cap. Such roots are
# reported separately as ``f1_broad`` — openness evidence for the caller's
# mapping, never a formant measurement (slots, adaptation and accuracy
# scoring stay on strict-bandwidth roots).
_F1_BROAD_MAX_BANDWIDTH_HZ = 800.0

# Pitch search range and voicing clarity threshold.
PITCH_MIN_HZ = 60
PITCH_MAX_HZ = 400
VOICED_CLARITY_THRESHOLD = 0.35

# Pitch/voicing estimator: "ncc" (normalized cross-correlation on the raw
# frame, :func:`ncc_pitch`) or "residual" (LPC-residual autocorrelation,
# :func:`lpc_residual_pitch` — coupled to the LPC order and window: it
# whitens with the same fit, so it degrades as either grows).
PITCH_METHOD = "ncc"
# NCC peak above which a frame is voiced. NCC alone over-voices low-level
# periodic tails; the analyzer adds an energy gate relative to the recent peak.
NCC_VOICED_THRESHOLD = 0.6
# The smallest lag whose NCC reaches this fraction of the global peak wins
# (octave guard: the NCC also peaks at period multiples).
_NCC_OCTAVE_GUARD = 0.85

# Guard added to divisors and log arguments.
EPSILON = 1e-9

_PITCH_LAG_MIN = ANALYSIS_SAMPLE_RATE // PITCH_MAX_HZ
_PITCH_LAG_MAX = ANALYSIS_SAMPLE_RATE // PITCH_MIN_HZ

# Lag-zero regularization: guards Levinson against singular systems on silence
# with negligible pole damping (1e-4 measurably shrinks high-gain pole radii).
_AUTOCORR_REGULARIZATION = 1.0 + 1e-6


@lru_cache(maxsize=4)
def _pitch_lag_gain(frame_size: int) -> np.ndarray:
    # The Hamming window attenuates autocorrelation peaks proportionally to
    # lag; compensate clarity by the window's own autocorrelation, clamped so
    # large lags don't over-amplify noise correlations. Keyed by frame length
    # so the table always follows the window the pitch frame was cut with.
    window = np.hamming(frame_size)
    autocorr = np.correlate(window, window, "full")[frame_size - 1 :][: _PITCH_LAG_MAX + 1]
    autocorr /= autocorr[0]
    return (1.0 / np.maximum(autocorr, 0.1)).astype(np.float32)


_SPECTRAL_NFFT = 512
_RFFT_FREQS = np.fft.rfftfreq(_SPECTRAL_NFFT, d=1.0 / ANALYSIS_SAMPLE_RATE).astype(np.float32)
# Bins strictly below 500 Hz at 16 kHz / 512-point FFT (31.25 Hz per bin).
_LOW_BAND_BINS = int(500 / (ANALYSIS_SAMPLE_RATE / _SPECTRAL_NFFT))

_TWO_PI = 2.0 * np.pi


class FormantEstimate(NamedTuple):
    """Formant frequencies estimated from one analysis frame.

    Parameters:
        f1: First formant frequency in Hz, or 0.0 if not found.
        f2: Second formant frequency in Hz, or 0.0 if not found.
        f3: Third formant frequency in Hz, or 0.0 if not found.
        plausible: Whether both F1 and F2 slots were filled this frame; when
            False the caller should hold the missing slots and lower
            confidence.
        f2_bandwidth: Bandwidth of the root filling the F2 slot in Hz (0.0
            when the slot is empty). A broad F2 is the nasal detector's
            "damped" signal.
        f1_broad: Frequency of a broad F1-band root (bandwidth between the
            strict formant cap and ``_F1_BROAD_MAX_BANDWIDTH_HZ``) present
            when the F1 slot is empty; 0.0 otherwise. Openness evidence for
            mapping — not a formant measurement.
    """

    f1: float
    f2: float
    f3: float
    plausible: bool
    f2_bandwidth: float = 0.0
    f1_broad: float = 0.0


class LpcResult(NamedTuple):
    """LPC solution for one analysis frame.

    Parameters:
        coefficients: LPC coefficients ``a[0..order]`` with ``a[0] == 1.0``.
        prediction_gain: Frame energy over the final prediction error — high
            for well-modeled (strongly resonant) frames, low for noise and
            transitions; 0.0 for degenerate frames.
    """

    coefficients: np.ndarray
    prediction_gain: float


class PitchEstimate(NamedTuple):
    """Pitch estimated from one analysis frame.

    Parameters:
        frequency: Pitch in Hz; 0.0 when unvoiced.
        voiced: Whether the frame is voiced.
        clarity: Autocorrelation peak clarity (0..1).
    """

    frequency: float
    voiced: bool
    clarity: float


def levinson_durbin(autocorr: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Solve the LPC normal equations by Levinson-Durbin recursion.

    Args:
        autocorr: Autocorrelation sequence with at least ``order + 1`` lags.
            Lag zero should be regularized by the caller (see
            ``_AUTOCORR_REGULARIZATION``) to avoid a singular system on silence.
        order: LPC model order.

    Returns:
        Tuple of (LPC coefficients ``a[0..order]`` with ``a[0] == 1.0``,
        final prediction error). On a numerically degenerate input the
        recursion stops early and returns the coefficients computed so far
        with an error of 0.0.
    """
    a = np.zeros(order + 1, dtype=np.float32)
    a[0] = 1.0
    err = float(autocorr[0])
    if err <= 0.0:
        return a, 0.0
    for i in range(1, order + 1):
        acc = float(autocorr[i])
        if i > 1:
            # sum_{j=1..i-1} a[j] * r[i-j]
            acc += float(np.dot(a[1:i], autocorr[i - 1 : 0 : -1]))
        k = -acc / err
        # a[j] += k * a[i-j] for j = 1..i-1, then a[i] = k.
        a[1:i] += (k * a[i - 1 : 0 : -1]).astype(np.float32)
        a[i] = k
        err *= 1.0 - k * k
        if err <= 0.0:
            return a, 0.0
    return a, err


def lpc_coefficients(frame: np.ndarray, order: int | None = None) -> LpcResult:
    """Compute the LPC solution for one pre-emphasized, windowed analysis frame.

    Autocorrelation method with lag-zero regularization, solved by
    :func:`levinson_durbin`. Shared by :func:`lpc_formants` (roots) and
    :func:`lpc_residual_pitch` (residual) so the recursion runs once per frame.

    Args:
        frame: float32 samples at 16 kHz (``LPC_FRAME_SIZE`` in the analyzer).
        order: LPC model order; ``LPC_ORDER`` when None (resolved per call, so
            the constant is never bound at import).

    Returns:
        The LPC coefficients and the frame's prediction gain.
    """
    if order is None:
        order = LPC_ORDER
    n = frame.shape[0]
    r = np.empty(order + 1, dtype=np.float32)
    r[0] = np.dot(frame, frame) * _AUTOCORR_REGULARIZATION + EPSILON
    for lag in range(1, order + 1):
        r[lag] = np.dot(frame[: n - lag], frame[lag:])
    a, err = levinson_durbin(r, order)
    gain = float(r[0] / err) if err > 0.0 else 0.0
    return LpcResult(a, gain)


def _assign_slots(
    candidates: list[tuple[float, float]], targets: tuple[float, float, float]
) -> tuple[tuple[float, float, float], float]:
    """Assign candidate (freq, bandwidth) roots to F1/F2/F3 slots.

    Exhaustive over per-band eligible candidates (a handful per frame):
    maximize filled slots first, then minimize total relative deviation from
    the targets plus the ``SLOT_PRIOR_WEIGHTS`` pull toward the nominal slot
    centers. Filled-count-first resolves the /u/ ambiguity (300+800 both
    fit the F1 band, but only F1=300 leaves an F2) without preferring
    spurious low poles when the true F1 is present.

    Returns:
        ((f1, f2, f3), f2_bandwidth) with 0.0 for empty slots.
    """
    bands = (F1_BAND_HZ, F2_BAND_HZ, F3_BAND_HZ)
    eligible = [
        [
            i
            for i, (f, bw) in enumerate(candidates)
            if band[0] <= f <= band[1] and bw <= FORMANT_MAX_BANDWIDTH_HZ
        ]
        + [None]
        for band in bands
    ]
    best = (-1, float("inf"), (0.0, 0.0, 0.0), 0.0)
    for i1 in eligible[0]:
        f1 = candidates[i1][0] if i1 is not None else 0.0
        for i2 in eligible[1]:
            if i2 is not None and i2 == i1:
                continue
            f2 = candidates[i2][0] if i2 is not None else 0.0
            if f1 and f2 and f2 <= f1 + _F2_MIN_ABOVE_F1_HZ:
                continue
            for i3 in eligible[2]:
                if i3 is not None and (i3 == i1 or i3 == i2):
                    continue
                f3 = candidates[i3][0] if i3 is not None else 0.0
                if f2 and f3 and f3 <= f2 + _F3_MIN_ABOVE_F2_HZ:
                    continue
                filled = (f1 > 0) + (f2 > 0) + (f3 > 0)
                deviation = sum(
                    abs(f - t) / t + weight * abs(f - nominal) / nominal
                    for f, t, nominal, weight in zip(
                        (f1, f2, f3), targets, _SLOT_NOMINAL_HZ, SLOT_PRIOR_WEIGHTS
                    )
                    if f > 0
                )
                if (filled, -deviation) > (best[0], -best[1]):
                    f2_bw = candidates[i2][1] if i2 is not None else 0.0
                    best = (filled, deviation, (f1, f2, f3), f2_bw)
    return best[2], best[3]


def lpc_formants(
    frame: np.ndarray,
    lpc: np.ndarray | None = None,
    prev: "FormantEstimate | None" = None,
) -> FormantEstimate:
    """Estimate F1/F2/F3 from one pre-emphasized, windowed analysis frame.

    Finds the roots of the LPC polynomial, keeps roots with positive imaginary
    part, bandwidth within ``(0, FORMANT_MAX_BANDWIDTH_HZ)`` and frequency
    within the formant band, then assigns them to F1/F2/F3 slots by band
    (``F1_BAND_HZ``..), preferring assignments close to the previous frame's
    estimate. A slot with no eligible candidate is reported as 0.0 — the
    caller holds its previous value rather than promoting a higher root.

    Args:
        frame: float32 samples of length ``FRAME_SIZE`` at 16 kHz.
        lpc: Precomputed LPC coefficients for the frame (from
            :func:`lpc_coefficients`). Computed from ``frame`` when None.
        prev: Previous frame's estimate, used as the slot-assignment target
            for track continuity. Nominal centers are used when None or when
            a previous slot is empty.

    Returns:
        The formant estimate, with ``plausible`` True only when both F1 and
        F2 slots were filled this frame.
    """
    a = lpc if lpc is not None else lpc_coefficients(frame).coefficients
    if not np.any(a[1:]):
        return FormantEstimate(0.0, 0.0, 0.0, False)

    roots = np.roots(a)
    candidates: list[tuple[float, float]] = []
    for root in roots:
        if root.imag <= 0.0:
            continue
        mag = abs(root)
        if mag <= EPSILON:
            continue
        bandwidth = -(ANALYSIS_SAMPLE_RATE / np.pi) * np.log(mag)
        if not 0.0 < bandwidth <= _F1_BROAD_MAX_BANDWIDTH_HZ:
            continue
        freq = np.arctan2(root.imag, root.real) * ANALYSIS_SAMPLE_RATE / _TWO_PI
        if FORMANT_MIN_HZ <= freq <= FORMANT_MAX_HZ:
            candidates.append((float(freq), float(bandwidth)))

    targets = tuple(
        prev_value if prev_value > 0.0 else nominal
        for prev_value, nominal in zip(
            (prev.f1, prev.f2, prev.f3) if prev else (0.0, 0.0, 0.0), _SLOT_NOMINAL_HZ
        )
    )
    (f1, f2, f3), f2_bandwidth = _assign_slots(candidates, targets)

    f1_broad = 0.0
    if f1 == 0.0:
        broad = [
            f
            for f, bw in candidates
            if F1_BAND_HZ[0] <= f <= F1_BAND_HZ[1] and bw > FORMANT_MAX_BANDWIDTH_HZ
        ]
        if broad:
            f1_broad = min(broad, key=lambda f: abs(f - targets[0]))
    return FormantEstimate(f1, f2, f3, f1 > 0.0 and f2 > 0.0, f2_bandwidth, f1_broad)


def lpc_residual_pitch(frame: np.ndarray, lpc: np.ndarray) -> PitchEstimate:
    """Estimate pitch by autocorrelation of the LPC residual.

    The residual (frame filtered by the LPC inverse filter) flattens the
    spectral envelope so the autocorrelation peak reflects glottal
    periodicity. Searches ``PITCH_MIN_HZ``..``PITCH_MAX_HZ``; a frame is
    voiced when the peak clarity exceeds ``VOICED_CLARITY_THRESHOLD``. Clarity
    is compensated for the Hamming window's lag decay, so the frame is
    expected to be Hamming-windowed over its full length.

    Args:
        frame: float32 samples at 16 kHz (the LPC frame).
        lpc: LPC coefficients used to compute the residual.

    Returns:
        The pitch estimate for the frame.
    """
    n = frame.shape[0]
    residual = np.convolve(frame, lpc)[:n]
    # Autocorrelation via FFT: nfft >= 2n - 1 keeps the lags linear.
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(residual, nfft)
    autocorr = np.fft.irfft(spectrum.real**2 + spectrum.imag**2)

    r0 = float(autocorr[0]) + EPSILON
    lag_max = min(_PITCH_LAG_MAX, n - 1)
    segment = (
        autocorr[_PITCH_LAG_MIN : lag_max + 1] * _pitch_lag_gain(n)[_PITCH_LAG_MIN : lag_max + 1]
    )
    peak = float(segment.max())
    # Octave guard: prefer the smallest lag whose peak is close to the global
    # maximum — the autocorrelation also peaks at period multiples, and a bare
    # argmax can land one octave down.
    candidates = np.nonzero(segment >= 0.85 * peak)[0]
    best = int(candidates[0]) if candidates.size else int(np.argmax(segment))
    clarity = float(segment[best]) / r0
    if clarity <= VOICED_CLARITY_THRESHOLD:
        return PitchEstimate(0.0, False, max(clarity, 0.0))
    frequency = ANALYSIS_SAMPLE_RATE / float(_PITCH_LAG_MIN + best)
    return PitchEstimate(frequency, True, clarity)


def ncc_pitch(frame: np.ndarray) -> PitchEstimate:
    """Estimate pitch by normalized cross-correlation of the raw frame.

    Correlates the frame's leading segment against the frame at every
    candidate lag, normalized by both segments' energies, so the peak is a
    periodicity measure in 0..1 that is independent of level, spectral
    envelope and the LPC fit. The frame must be raw: not windowed (a taper
    breaks the energy normalization) and not pre-emphasized (pre-emphasis
    boosts aspiration noise over the periodic low harmonics).

    Args:
        frame: float32 raw samples at 16 kHz; longer than ``_PITCH_LAG_MAX``.

    Returns:
        The pitch estimate for the frame; ``clarity`` is the NCC peak.
    """
    x = frame - np.mean(frame)
    n = x.shape[0]
    lag_max = min(_PITCH_LAG_MAX, n - 2 * _PITCH_LAG_MIN)
    seg = n - lag_max
    reference = x[:seg]
    cross = np.correlate(x[_PITCH_LAG_MIN:], reference, "valid")  # lags lag_min..lag_max
    energy = np.concatenate(([0.0], np.cumsum(x * x, dtype=np.float64)))
    lag_energy = (
        energy[_PITCH_LAG_MIN + seg : lag_max + seg + 1] - energy[_PITCH_LAG_MIN : lag_max + 1]
    )
    ncc = cross / np.sqrt(energy[seg] * lag_energy + EPSILON)

    peak = float(ncc.max())
    if peak <= NCC_VOICED_THRESHOLD:
        return PitchEstimate(0.0, False, max(peak, 0.0))
    # Octave guard: the local maximum inside the first contiguous run that
    # reaches the guard fraction of the peak (not the run's first sample).
    above = ncc >= _NCC_OCTAVE_GUARD * peak
    first = int(np.argmax(above))
    below = np.nonzero(~above[first:])[0]
    end = first + int(below[0]) if below.size else above.shape[0]
    best = first + int(np.argmax(ncc[first:end]))
    clarity = float(ncc[best])
    # Parabolic interpolation around the chosen lag.
    shift = 0.0
    if 0 < best < ncc.shape[0] - 1:
        left, mid, right = float(ncc[best - 1]), clarity, float(ncc[best + 1])
        curvature = left - 2.0 * mid + right
        if curvature < 0.0:
            shift = 0.5 * (left - right) / curvature
    frequency = ANALYSIS_SAMPLE_RATE / (_PITCH_LAG_MIN + best + shift)
    return PitchEstimate(float(frequency), clarity > NCC_VOICED_THRESHOLD, clarity)


def rms_energy(frame: np.ndarray) -> float:
    """Compute the RMS energy of one analysis frame.

    Computed on the raw (un-windowed, un-emphasized) frame so the envelope
    tracks loudness rather than spectral tilt.

    Args:
        frame: float32 samples.

    Returns:
        Root-mean-square energy of the frame.
    """
    return float(np.sqrt(np.dot(frame, frame) / frame.shape[0] + EPSILON))


def spectral_nasal_features(frame: np.ndarray) -> tuple[float, float]:
    """Compute the spectral features used for nasal detection.

    One 512-point rFFT per (windowed) frame.

    Args:
        frame: float32 samples of length ``FRAME_SIZE`` at 16 kHz.

    Returns:
        Tuple of (spectral centroid in Hz, ratio of energy below 500 Hz to
        total energy).
    """
    spectrum = np.fft.rfft(frame, _SPECTRAL_NFFT)
    power = spectrum.real**2 + spectrum.imag**2
    total = float(power.sum()) + EPSILON
    centroid = float(np.dot(power, _RFFT_FREQS)) / total
    low_band_ratio = float(power[:_LOW_BAND_BINS].sum()) / total
    return centroid, low_band_ratio


class P2QuantileEstimator:
    """Streaming quantile estimator using the P² algorithm (O(1) memory).

    Used to track running P5/P95 of voiced-frame F1/F2 for adaptive
    normalization, without storing samples. Reference: Jain & Chlamtac,
    "The P² algorithm for dynamic calculation of quantiles" (1985).
    """

    def __init__(self, quantile: float):
        """Initialize the estimator.

        Args:
            quantile: The quantile to track, in (0, 1).
        """
        self._quantile = quantile
        self._initial: list[float] = []
        # Marker heights, positions and desired positions (allocated after the
        # first five observations).
        self._q: list[float] = []
        self._n: list[float] = []
        self._np: list[float] = []
        self._dn = [0.0, quantile / 2.0, quantile, (1.0 + quantile) / 2.0, 1.0]

    def add(self, value: float):
        """Ingest one observation.

        Args:
            value: The observed value.
        """
        if len(self._initial) < 5:
            self._initial.append(value)
            if len(self._initial) == 5:
                self._q = sorted(self._initial)
                self._n = [1.0, 2.0, 3.0, 4.0, 5.0]
                p = self._quantile
                self._np = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0]
            return

        q, n = self._q, self._n

        # Locate the cell and update extreme markers.
        if value < q[0]:
            q[0] = value
            k = 0
        elif value >= q[4]:
            q[4] = value
            k = 3
        else:
            k = 0
            while value >= q[k + 1]:
                k += 1

        for i in range(k + 1, 5):
            n[i] += 1.0
        for i in range(5):
            self._np[i] += self._dn[i]

        # Adjust interior markers toward their desired positions.
        for i in range(1, 4):
            d = self._np[i] - n[i]
            if (d >= 1.0 and n[i + 1] - n[i] > 1.0) or (d <= -1.0 and n[i - 1] - n[i] < -1.0):
                d = 1.0 if d >= 0.0 else -1.0
                # Parabolic (P²) update, falling back to linear when it would
                # break marker monotonicity.
                qp = q[i] + d / (n[i + 1] - n[i - 1]) * (
                    (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
                    + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
                )
                if not q[i - 1] < qp < q[i + 1]:
                    j = i + int(d)
                    qp = q[i] + d * (q[j] - q[i]) / (n[j] - n[i])
                q[i] = qp
                n[i] += d

    def value(self) -> float:
        """Return the current quantile estimate.

        Returns:
            The estimated quantile. Before five observations have been
            ingested, returns the sample quantile of what has been seen
            (0.0 when empty).
        """
        if self._q:
            return self._q[2]
        if not self._initial:
            return 0.0
        ordered = sorted(self._initial)
        index = round(self._quantile * (len(ordered) - 1))
        return ordered[index]

    @property
    def count(self) -> int:
        """Number of observations ingested."""
        if self._q:
            return int(self._n[4])
        return len(self._initial)
