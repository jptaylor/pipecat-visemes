#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Formant-based lipsync analyzer.

The provider-universal analyzer and the design of record. Estimates a continuous articulation
signal directly from TTS audio using LPC formant tracking — no provider
timestamps, phoneme models or per-voice calibration required — with a bounded
CPU and memory footprint suitable for always-on use.
"""

from dataclasses import dataclass

import numpy as np

from lipsync import dsp
from lipsync.base_lipsync_analyzer import (
    BaseLipsyncAnalyzer,
    LipsyncAnalysisContext,
    LipsyncFrameResult,
)
from lipsync.types import LipsyncEvent, LipsyncEventKind, LipsyncKeyframe

# Analysis hop duration in seconds (20 ms).
HOP_SECONDS = dsp.HOP_SIZE / dsp.ANALYSIS_SAMPLE_RATE

# Events at an offset are final once the analysis cursor is this far past it:
# max closure duration (250 ms) + speech-confirmation window (150 ms). The
# processor holds batch emission by this horizon.
EVENT_FINALIZE_HORIZON_SEC = 0.4

# Neutral (schwa-like) articulation the signal decays toward when unvoiced;
# during true silence, openness rests nearly closed instead.
_NEUTRAL = 0.35
_SILENCE_REST_OPENNESS = 0.15
_UNVOICED_DECAY = 0.8

# Generic formant priors (Hz); shifted up for high-pitched voices.
_F1_PRIOR = (250.0, 900.0)
_F2_PRIOR = (800.0, 2500.0)
_HIGH_PITCH_HZ = 180.0
_PRIOR_SHIFT = 1.12
_PITCH_PROBE_FRAMES = 10

# Adaptive normalization: full trust in learned ranges after this many voiced
# frames (~1.2 s of speech — spike rejection guards the early estimates);
# learned spans narrower than the minimum are degenerate and ignored. The
# learned edges are P10/P90 (P5/P95 left too much slack: peak /a/ mapped to
# ~0.45 against a Praat-oracle ~0.94).
_CONVERGENCE_FRAMES = 60
_MIN_LEARNED_SPAN_HZ = 100.0
_MIN_ESTIMATOR_COUNT = 10

# Distribution shift: if the recent median voiced F1 deviates from the long
# ring median by more than this fraction, decay learned ranges back toward
# priors over ~2 s and reduce (not zero) convergence trust.
_SHIFT_RING = 100
_SHIFT_RECENT = 25
_SHIFT_DEVIATION = 0.25
_SHIFT_DECAY_PER_HOP = HOP_SECONDS / 2.0
_SHIFT_RETAINED_FRAMES = 30

# Energy tracking.
_NOISE_RING = 50  # min-statistics window (~1 s)
_NOISE_FLOOR_MIN = 1e-5
_NOISE_FLOOR_PEAK_CAP = 0.05  # floor never exceeds this fraction of the recent peak
_PEAK_DECAY = 0.98  # recent-peak decay per hop (~1 s time scale)
_ENERGY_MAX_DECAY = 0.998  # session energy max decay per hop (~10 s)
_LOG_COMPRESSION = 9.0

# Silence: sustained sub-threshold energy emits one SILENCE event.
_SILENCE_FLOOR_MULT = 2.5
_SILENCE_ABS = 1e-4
_SILENCE_EVENT_HOPS = 15  # 300 ms

# Closure (M/B/P): a short, bounded energy dip inside a speech region.
_CLOSURE_FLOOR_MULT = 3.0
_CLOSURE_PEAK_FRACTION = 0.22
_CLOSURE_MIN_HOPS = 2  # 40 ms hysteresis
_CLOSURE_MAX_SEC = 0.25
_SPEECH_WINDOW_HOPS = 8  # ±150 ms surrounding-speech requirement (in hops)
# Speech history must reach back past a maximum-length closure run.
_SPEECH_HIST_HOPS = _SPEECH_WINDOW_HOPS + int(_CLOSURE_MAX_SEC / HOP_SECONDS) + 2

# Nasal: voiced, spectrally dark, energy concentrated below 500 Hz, F2 damped.
# "Damped" means: no F2 root, a broad one, or one found above the murmur
# range while the spectrum is extremely dark — LPC fits narrow spurious poles
# mid-band on murmurs, and no real vowel pairs a high F2 with a ~0.9 low-band
# ratio (dark vowels have low F2).
_NASAL_CENTROID_MAX_HZ = 1000.0
_NASAL_LOW_RATIO_MIN = 0.6
_NASAL_F2_MAX_BANDWIDTH_HZ = 300.0
# When a missing F2 root counts as "damped": "always", "dark" (only above
# _NASAL_DARK_RATIO) or "never". Voice-dependent at LPC order 16: one corpus
# voice's murmur always has a narrow ~1.9 kHz root (caught by the spurious-F2
# rule, so "never" would cut false nasal closures on nasal-free speech from
# 28 % to 9 % of voiced hops), the other's has none and loses its hums
# without this rule. Keep "always" until a cue that separates murmurs from
# close vowels exists (plans/deep-review-2026-09-results.md).
_NASAL_MISSING_F2_DAMPED = "always"
_NASAL_DARK_RATIO = 0.9
_NASAL_SPURIOUS_F2_HZ = 1200.0
# A root at or below this above F1 (``FormantEstimate.f2_low``: any
# bandwidth, whichever root won the F2 slot) vetoes the murmur reading.
# Measured on both corpus voices, murmurs have no root between 500 and
# 1200 Hz (their F2, when found, sits at 1.5-1.9 kHz, which the spurious rule
# handles), while a back vowel's F2 lands at 500-1000 Hz — on one voice as a
# root too broad for the strict formant cap, which the missing-F2 rule then
# read as a murmur and shut the mouth on every /u/ (vowel-oo nasal duty
# 0.53-0.62 before the veto).
_NASAL_VOWEL_F2_MAX_HZ = 1200.0
# Enter after this many consecutive hops of nasal evidence, exit after as
# many without. Never a single hop: a 1-hop fast path on a very dark frame
# latched on /w u l ð/ and the voice bar of voiced stops (26–29 % of the
# voiced hops of a nasal-free sentence; 17–19 % without it). The pre-latch
# soft cap still closes the mouth within one hop. Longer entry or exit
# windows were measured: 3-hop entry loses hums, 3–4-hop exit brings the
# false latches back.
_NASAL_HYSTERESIS_HOPS = 2
# Extra murmur evidence: F1 (found or held) at most this, and at most this
# fraction of spectral energy in 500-1500 Hz (a murmur's antiformant empties
# that band; disabled). Neither separates murmurs from dark voiced consonants
# (measured 2026-09-18: ~0.01 of duty each), but the F1 cap does separate
# them from nasalized vowels — "moon", "new": nasal spectrum, F1 330-440 Hz,
# open mouth — which the spurious-F2 rule otherwise reads as murmurs. A
# murmur's F1 is the ~250 Hz nasal formant. 350 keeps the hum probes on both
# voices (320 buys a little more but halves one voice's hum margin).
_NASAL_F1_MAX_HZ = 350.0
_NASAL_MID_RATIO_MAX = float("inf")
# While the override is active the mapped openness is capped here rather than
# forced shut: a murmur's F1 sits at the learned floor and maps near zero
# anyway, while a nasalized vowel (nasal spectrum, open mouth — "moon", "new")
# keeps a small opening and its rounding instead of a shut mouth.
_NASAL_OPENNESS_MAX = 0.15
# A back rounded vowel (the nasal detector's veto: a root at 500-1200 Hz
# above F1) keeps at least this opening: its F1 (~210-300 Hz on the corpus
# voices) sits at the learned floor and would otherwise map to a shut mouth.
_ROUNDED_VOWEL_MIN_OPENNESS = 0.1
# Rounding is gated off only for wide-open mouths (a low-side gate on
# openness zeroed the rounding of /u/, whose opening is small by nature).
_ROUNDING_OPEN_GATE = (0.75, 0.9)
# Pre-latch soft cap: nasal-ish voiced frames cap openness before the event
# state machine latches, so the continuous signal reacts within one hop.
_NASAL_SOFT_CAP_OPENNESS = 0.2
_NASAL_SOFT_CAP_CENTROID_HZ = 800.0
_NASAL_SOFT_CAP_RATIO = 0.5

# NCC voicing gate: a voiced hop must also reach this fraction of the recent
# RMS peak. NCC is level-independent, so without the gate low-level periodic
# tails and room tone read as voiced.
_VOICED_MIN_PEAK_FRACTION = 0.05

# Conditioning. The median is zero-phase: hop t is conditioned (and its
# keyframe emitted) when hop t+1 arrives, so the 3-tap median is centered on
# it instead of trailing it — a trailing median plus the slew put the emitted
# track ~35 ms behind Praat on both corpus voices (openness r 0.49 at zero lag,
# 0.71–0.76 at the best lag). The cost is one hop of analysis latency, well
# inside the processor's emission horizon.
_MEDIAN_TAPS = 3
# Slew: 0.25 left the emitted track 12–20 ms late after the centered median;
# 0.4 halves that for +0.02 jitter (mean |second difference| 0.05 -> 0.07);
# no slew reads +3 ms but jitter 0.09–0.11.
_SLEW_MAX_PER_HOP = 0.4
# Anchor keyframe: when a parameter leaves the dead band, first emit the
# previous hop's value if it was suppressed, so the client interpolates the
# transition from where the mouth actually was rather than from a keyframe up
# to a heartbeat old. Off: on both corpus voices it changed openness/width r
# by 0.00–0.01 and added ~7 keyframes/s.
_ANCHOR_KEYFRAMES = False

# Hz-domain robustness: a cap on how long an empty slot may hold its last
# value before decaying toward the prior center, and a jump size treated as
# a spike when feeding the adaptive estimators.
_HOLD_MAX_HOPS = 3
_HOLD_DECAY = 0.1
_ADAPT_SPIKE_HZ = 400.0
_ADAPT_MAX_SKIPS = 2

# Confidence.
_FORMANT_DELTA_HZ = 300.0
_C_LPC_FLOOR = 0.3
_C_SLOT_PARTIAL = 0.5
_SNR_FULL_DB = 20.0
# log10(prediction gain) that earns full fit confidence. Tied to dsp.LPC_ORDER:
# gain rises with order (median log10 gain on Praat-voiced hops 1.27 -> 1.38
# and 1.44 -> 1.49 on the two corpus voices from order 12 to 16), so 3.0 at
# order 12 became 3.2 to keep c_fit's distribution where it was.
_C_FIT_LOG10_FULL = 3.2


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    t = _clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class LipsyncDebugFrame:
    """Raw per-hop analysis features (dev-only, for benchmarks and tests).

    Parameters:
        offset: Hop center in seconds from utterance start.
        f1: Raw first formant in Hz (0.0 = not found this hop).
        f2: Raw second formant in Hz.
        f3: Raw third formant in Hz.
        f2_bandwidth: Bandwidth of the F2 root in Hz (0.0 = no F2 this hop).
        f1_broad: Broad F1-band root used for mapping when the strict F1 slot
            is empty (0.0 = none).
        f2_broad: Broad F2-band root used for mapping when the strict F2
            slot is empty (0.0 = none).
        f2_low: Lowest root above F1 from 500 Hz up, any bandwidth; the nasal
            detector's murmur veto when at or below 1200 Hz (0.0 = none).
        pitch_hz: Raw pitch in Hz (0.0 when unvoiced).
        voiced: Whether the hop was classified voiced.
        clarity: Pitch-detector periodicity (NCC peak or residual clarity).
        rms: Frame RMS energy.
        centroid: Spectral centroid in Hz.
        low_band_ratio: Fraction of spectral energy below 500 Hz.
        mid_band_ratio: Fraction of spectral energy in 500-1500 Hz.
        prediction_gain: LPC prediction gain (0.0 below the silence gate).
        nasal_active: Whether the nasal override was closing the mouth.
        openness_mapped: Openness target from the F1 mapping (or unvoiced
            decay), before the nasal override and conditioning.
        openness_target: Openness target after the nasal override, before
            conditioning.
        openness_conditioned: Openness after conditioning (the value the
            keyframe gate sees); the offset it belongs to is ``offset``.
        c_lpc: Formant-plausibility confidence component.
        c_conv: Normalization-convergence confidence component.
        c_snr: SNR confidence component.
        c_fit: LPC prediction-gain confidence component.
        confidence: Final composite confidence for the hop.
        f1_lo: Effective F1 range low edge this hop.
        f1_hi: Effective F1 range high edge this hop.
        f2_lo: Effective F2 range low edge this hop.
        f2_hi: Effective F2 range high edge this hop.
    """

    offset: float
    f1: float
    f2: float
    f3: float
    f2_bandwidth: float
    f1_broad: float
    f2_broad: float
    f2_low: float
    pitch_hz: float
    voiced: bool
    clarity: float
    rms: float
    centroid: float
    low_band_ratio: float
    mid_band_ratio: float
    prediction_gain: float
    nasal_active: bool
    openness_mapped: float
    openness_target: float
    openness_conditioned: float
    c_lpc: float
    c_conv: float
    c_snr: float
    c_fit: float
    confidence: float
    f1_lo: float
    f1_hi: float
    f2_lo: float
    f2_hi: float


@dataclass
class _PendingHop:
    """Internal: one hop's keyframe fields, held until its centered median is known."""

    offset: float
    params: list[float]  # conditioned openness, width, rounding (filled when conditioned)
    energy: float
    pitch: float
    confidence: float
    event_fired: bool


@dataclass
class _PendingClosure:
    """Internal: a detected energy dip awaiting trailing-speech confirmation."""

    offset: float
    duration: float
    depth: float
    deadline_hop: int


class FormantLipsyncAnalyzer(BaseLipsyncAnalyzer):
    """Estimates mouth articulation from TTS audio via LPC formant analysis.

    Audio is analyzed at 16 kHz at a 20 ms hop: a 25 ms window for energy and
    events, and an LPC window (``dsp.LPC_FRAME_SIZE``) centered on the same
    hop for formants. Per frame, the analyzer maps:

    - F1 → openness (normalized within an adaptive F1 range)
    - F2 → width (high F2 → spread /i/, low F2 → back /u, o/)
    - F2 + F1 heuristic → rounding
    - RMS envelope → energy
    - Normalized cross-correlation of the raw frame → pitch and voicing

    Formant ranges start from generic priors and adapt online via streaming
    P5/P95 quantiles of voiced frames — no calibration step. Confidence
    combines formant plausibility, normalization convergence and SNR, so
    early or uncertain frames blend toward neutral client-side. Discrete
    closure/nasal/silence events are detected from energy dips, low spectral
    centroid with damped F2, and sustained sub-threshold energy respectively.

    Per-voice adaptive state is session-scoped (per analyzer instance);
    keying it by voice id across mid-session voice switches is a known
    follow-up (tech spec §13).
    """

    def __init__(
        self,
        *,
        dead_band: float = 0.05,
        heartbeat_ms: int = 240,
        collect_debug: bool = False,
    ):
        """Initialize the analyzer.

        Args:
            dead_band: Minimum openness/width/rounding delta since the last
                emitted keyframe required to emit a new one.
            heartbeat_ms: Maximum time between keyframes while speech is
                active, even when parameters are static. Suppressed during
                silence stretches (the SILENCE event parks the client).
            collect_debug: When True, collect one :class:`LipsyncDebugFrame`
                per hop in :attr:`debug_features` (dev/benchmark use only).
        """
        self._dead_band = dead_band
        self._heartbeat_sec = heartbeat_ms / 1000.0
        self._collect_debug = collect_debug
        self.debug_features: list[LipsyncDebugFrame] = []

        # Three framings share each hop center: the 25 ms frame (energy,
        # events, nasal spectral features), the LPC frame (formants, residual
        # pitch) and the NCC pitch frame. ``_pad`` is the widest frame's
        # reach either side of the 25 ms frame.
        self._lpc_frame_size = dsp.LPC_FRAME_SIZE
        self._pitch_frame_size = (
            dsp.PITCH_FRAME_SIZE if dsp.PITCH_METHOD == "ncc" else dsp.FRAME_SIZE
        )
        for size in (self._lpc_frame_size, self._pitch_frame_size):
            if size < dsp.FRAME_SIZE or (size - dsp.FRAME_SIZE) % 2:
                raise ValueError("frame sizes must be >= FRAME_SIZE with an even difference")
        self._lpc_pad = (self._lpc_frame_size - dsp.FRAME_SIZE) // 2
        self._pitch_pad = (self._pitch_frame_size - dsp.FRAME_SIZE) // 2
        self._pad = max(self._lpc_pad, self._pitch_pad)

        # Preallocated scratch; analysis frames are fixed-size at 16 kHz. The
        # ingest buffer covers the processor's largest drain (2 s ring cap)
        # plus the widest frame and the flush padding.
        self._window = np.hamming(self._lpc_frame_size).astype(np.float32)
        self._windowed = np.empty(self._lpc_frame_size, dtype=np.float32)
        if self._lpc_pad:
            self._window_short = np.hamming(dsp.FRAME_SIZE).astype(np.float32)
            self._windowed_short = np.empty(dsp.FRAME_SIZE, dtype=np.float32)
        else:
            self._windowed_short = self._windowed  # one shared frame
        self._buf = np.empty(
            2 * dsp.ANALYSIS_SAMPLE_RATE + dsp.FRAME_SIZE + 3 * self._pad, dtype=np.float32
        )

        self._reset_session_state()
        self._reset_utterance_state()

    #
    # BaseLipsyncAnalyzer
    #

    async def start(self, sample_rate: int):
        """Prepare the analyzer for a session.

        Args:
            sample_rate: Source sample rate of the TTS audio in Hz (analysis
                itself always runs at 16 kHz; the processor resamples).
        """
        self._reset_session_state()
        self._reset_utterance_state()

    async def analyze(self, pcm: np.ndarray, context: LipsyncAnalysisContext) -> LipsyncFrameResult:
        """Analyze a chunk of PCM audio from one TTS context.

        Args:
            pcm: Mono float32 samples at the fixed 16 kHz analysis rate.
            context: Analysis state for the TTS context the audio belongs to.

        Returns:
            Keyframes and events measured from the chunk.
        """
        result = LipsyncFrameResult()
        remaining = pcm
        while remaining.size:
            space = self._buf.size - self._buf_len
            take = min(space, remaining.size)
            self._buf[self._buf_len : self._buf_len + take] = remaining[:take]
            self._buf_len += take
            remaining = remaining[take:]
            self._drain(result)
        result.processed_up_to = self._hops * HOP_SECONDS
        return result

    async def flush(self, context: LipsyncAnalysisContext) -> LipsyncFrameResult:
        """Flush the utterance: analyze remaining full windows, drop the tail.

        Unconfirmed closure candidates are dropped (no trailing speech will
        arrive to confirm them).

        Args:
            context: Analysis state for the TTS context being closed.

        Returns:
            Keyframes and events remaining in the analysis window, with
            ``processed_up_to`` advanced past everything ingested.
        """
        result = LipsyncFrameResult()
        # The wider frames look ``_pad`` samples past the 25 ms frame: pad with
        # zeros so every complete 25 ms frame is still analyzed.
        if self._pad:
            self._buf[self._buf_len : self._buf_len + self._pad] = 0.0
            self._buf_len += self._pad
        self._drain(result)
        self._flush_conditioning(result)
        # Everything ingested is final: tail discarded, pending closures dead.
        result.processed_up_to = self._hops * HOP_SECONDS + EVENT_FINALIZE_HORIZON_SEC
        self._reset_utterance_state()
        return result

    async def reset(self):
        """Reset per-utterance state after an interruption.

        Preserves adaptive normalization state: the voice has not changed.
        """
        self._reset_utterance_state()

    #
    # State management
    #

    def _reset_session_state(self):
        self._f1_p5 = dsp.P2QuantileEstimator(0.10)
        self._f1_p95 = dsp.P2QuantileEstimator(0.90)
        self._f2_p5 = dsp.P2QuantileEstimator(0.10)
        self._f2_p95 = dsp.P2QuantileEstimator(0.90)
        self._pitch_p5 = dsp.P2QuantileEstimator(0.05)
        self._pitch_p95 = dsp.P2QuantileEstimator(0.95)
        self._voiced_frames = 0
        self._f1_prior = _F1_PRIOR
        self._f2_prior = _F2_PRIOR
        self._pitch_probe: list[float] = []
        self._priors_shifted = False
        self._shift_ring: list[float] = []
        self._prior_decay = 0.0
        self._noise_ring: list[float] = []
        self._noise_floor = _NOISE_FLOOR_MIN
        self._recent_peak = 0.0
        self._energy_max = _NOISE_FLOOR_MIN

    def _reset_utterance_state(self):
        # The buffer leads with the widest frame's left context (zeros at the
        # utterance start), so buffer index 0 is that frame's first start.
        self._buf[: self._pad] = 0.0
        self._buf_len = self._pad
        self._prev_sample = 0.0
        self._hops = 0
        self._prev_f1 = 0.0
        self._prev_f2 = 0.0
        self._prev_f3 = 0.0
        self._hold_counts = [0, 0]
        self._last_adapt = [0.0, 0.0]
        self._adapt_skips = [0, 0]
        self._prev_targets = [_NEUTRAL, _NEUTRAL, _NEUTRAL]  # openness, width, rounding
        self._median_hist: list[list[float]] = []  # last _MEDIAN_TAPS target vectors
        self._pending_hop: _PendingHop | None = None
        self._cond_prev: list[float] | None = None
        self._last_conditioned: _PendingHop | None = None  # previous hop, emitted or not
        self._last_emitted: list[float] | None = None
        self._last_emit_offset = -1.0
        self._speech_hist: list[bool] = []
        self._low_run_hops = 0
        self._low_run_start_offset = 0.0
        self._low_run_min_rms = 0.0
        self._pending_closures: list[_PendingClosure] = []
        self._silence_run_hops = 0
        self._silence_start_offset = 0.0
        self._silence_emitted = False
        self._nasal_active = False
        self._nasal_enter_count = 0
        self._nasal_exit_count = 0

    #
    # Analysis
    #

    def _drain(self, result: LipsyncFrameResult):
        """Process all complete analysis windows currently buffered."""
        start = 0
        pad = self._pad
        lpc_lead = pad - self._lpc_pad
        pitch_lead = pad - self._pitch_pad
        while self._buf_len - start >= dsp.FRAME_SIZE + 2 * pad:
            lpc_start = start + lpc_lead
            # Pre-emphasis needs the sample before the LPC frame: zero at the
            # utterance start, else carried across drains in ``_prev_sample``.
            prev = self._prev_sample if lpc_start == 0 else float(self._buf[lpc_start - 1])
            self._process_hop(
                self._buf[start + pad : start + pad + dsp.FRAME_SIZE],
                self._buf[lpc_start : lpc_start + self._lpc_frame_size],
                self._buf[start + pitch_lead : start + pitch_lead + self._pitch_frame_size],
                prev,
                result,
            )
            start += dsp.HOP_SIZE
        if start:
            self._prev_sample = float(self._buf[start - 1])
            remainder = self._buf_len - start
            self._buf[:remainder] = self._buf[start : self._buf_len]
            self._buf_len = remainder

    def _process_hop(
        self,
        raw: np.ndarray,
        lpc_raw: np.ndarray,
        pitch_raw: np.ndarray,
        prev_sample: float,
        result: LipsyncFrameResult,
    ):
        """Analyze one hop.

        Args:
            raw: The 25 ms frame (energy, events, nasal spectral features).
            lpc_raw: The LPC frame centered on the same hop (``raw`` itself
                when the framings are shared).
            pitch_raw: The NCC pitch frame centered on the same hop.
            prev_sample: The sample preceding ``lpc_raw`` (pre-emphasis).
            result: Accumulates keyframes and events.
        """
        offset = (self._hops * dsp.HOP_SIZE + dsp.FRAME_SIZE / 2) / dsp.ANALYSIS_SAMPLE_RATE
        self._hops += 1

        rms = dsp.rms_energy(raw)

        # Min-statistics noise floor, capped by the recent peak so pause-free
        # speech cannot inflate it into the speech range.
        self._noise_ring.append(rms)
        if len(self._noise_ring) > _NOISE_RING:
            self._noise_ring.pop(0)
        self._recent_peak = max(self._recent_peak * _PEAK_DECAY, rms)
        self._energy_max = max(self._energy_max * _ENERGY_MAX_DECAY, rms)
        self._noise_floor = max(
            min(min(self._noise_ring), _NOISE_FLOOR_PEAK_CAP * self._recent_peak),
            _NOISE_FLOOR_MIN,
        )

        silence_gate = max(_SILENCE_FLOOR_MULT * self._noise_floor, _SILENCE_ABS)

        # Per-frame DSP (skipped below the silence gate).
        voiced = False
        pitch_hz = 0.0
        clarity = 0.0
        centroid = 0.0
        low_ratio = 0.0
        mid_ratio = 0.0
        f1_found = f2_found = f3_found = False
        f1, f2, f3 = self._prev_f1, self._prev_f2, self._prev_f3
        prediction_gain = 0.0
        f2_bandwidth = f1_broad = 0.0
        if rms >= silence_gate:
            self._window_frames(lpc_raw, prev_sample)
            lpc = dsp.lpc_coefficients(self._windowed)
            prediction_gain = lpc.prediction_gain
            formants = dsp.lpc_formants(
                self._windowed,
                lpc.coefficients,
                prev=dsp.FormantEstimate(self._prev_f1, self._prev_f2, self._prev_f3, False),
            )
            if dsp.PITCH_METHOD == "ncc":
                pitch = dsp.ncc_pitch(pitch_raw)
                pitch_voiced = pitch.voiced and rms > _VOICED_MIN_PEAK_FRACTION * self._recent_peak
            else:
                pitch = dsp.lpc_residual_pitch(self._windowed, lpc.coefficients)
                pitch_voiced = pitch.voiced
            # Nasal features use the pre-emphasized spectrum: without the
            # tilt correction the glottal harmonics below 500 Hz dominate the
            # raw power spectrum of every voiced frame (measured ratio ~1.0
            # even for /a/), making the low-band ratio non-discriminative.
            centroid, low_ratio, mid_ratio = dsp.spectral_nasal_features(self._windowed_short)
            voiced = pitch_voiced
            pitch_hz = pitch.frequency if pitch_voiced else 0.0
            clarity = pitch.clarity
            # Per-slot hold: a found F1 is used even when F2 is missing this
            # frame; only the missing slot keeps its previous value. A broad
            # F1-band root (peak-open vowels) fills the mapped value only —
            # it is openness evidence, kept out of adaptation, confidence and
            # the debug tap.
            f1_found, f2_found, f3_found = formants.f1 > 0, formants.f2 > 0, formants.f3 > 0
            f2_bandwidth, f1_broad = formants.f2_bandwidth, formants.f1_broad
            f1_present = f1_found or formants.f1_broad > 0
            if f1_found:
                f1 = formants.f1
            elif formants.f1_broad > 0:
                f1 = formants.f1_broad
            else:
                f1 = self._prev_f1
            # A broad F2-band root fills the mapped F2 (rounding evidence)
            # like f1_broad fills F1; the strict slot stays empty for
            # adaptation, confidence and the debug tap.
            f2_broad, f2_low = formants.f2_broad, formants.f2_low
            if f2_found:
                f2 = formants.f2
            elif f2_broad > 0.0:
                f2 = f2_broad
            else:
                f2 = self._prev_f2
            f2_present = f2_found or f2_broad > 0.0
            f3 = formants.f3 if f3_found else self._prev_f3
            dark = low_ratio > _NASAL_DARK_RATIO
            f2_missing_damped = not f2_found and (
                _NASAL_MISSING_F2_DAMPED == "always"
                or (_NASAL_MISSING_F2_DAMPED == "dark" and dark)
            )
            vowel_f2 = 0.0 < formants.f2_low <= _NASAL_VOWEL_F2_MAX_HZ
            f2_damped = not vowel_f2 and (
                f2_missing_damped
                or formants.f2_bandwidth > _NASAL_F2_MAX_BANDWIDTH_HZ
                or (dark and formants.f2 > _NASAL_SPURIOUS_F2_HZ)
            )
        else:
            f1_present = False
            f2_present = False
            f2_broad = f2_low = 0.0
            f2_damped = False
            vowel_f2 = False

        # Bounded hold: after a few empty frames, drift a stale slot toward
        # its prior center instead of freezing an old shape indefinitely.
        # (A median over the Hz tracks was tried here and rejected: it lifted
        # track correlation but the added hop of lag raised MAE and hurt
        # trajectory shape — adaptation is protected by spike rejection
        # instead.)
        f1 = self._bound_hold(0, f1, f1_present, self._f1_prior)
        f2 = self._bound_hold(1, f2, f2_present, self._f2_prior)

        # Adaptive normalization updates (voiced frames only; found slots only,
        # so held values never pollute the learned ranges).
        if voiced:
            self._update_adaptation(f1, f2, pitch_hz, f1_found, f2_found)
        if self._prior_decay > 0.0:
            self._prior_decay = max(0.0, self._prior_decay - _SHIFT_DECAY_PER_HOP)

        f1_lo, f1_hi = self._effective_range(self._f1_p5, self._f1_p95, self._f1_prior)
        f2_lo, f2_hi = self._effective_range(self._f2_p5, self._f2_p95, self._f2_prior)

        # Continuous parameter targets.
        if voiced and f1 > 0.0:
            openness = _clamp01((f1 - f1_lo) / (f1_hi - f1_lo + dsp.EPSILON))
            width = _clamp01((f2 - f2_lo) / (f2_hi - f2_lo + dsp.EPSILON))
            f2_mid = (f2_lo + f2_hi) / 2.0
            rounding = _clamp01((f2_mid - f2) / (f2_mid - f2_lo + dsp.EPSILON)) * (
                1.0 - _smoothstep(openness, *_ROUNDING_OPEN_GATE)
            )
            if vowel_f2:
                openness = max(openness, _ROUNDED_VOWEL_MIN_OPENNESS)
        else:
            # Unvoiced speech (fricatives) decays toward neutral rather than
            # snapping shut; true silence rests nearly closed — otherwise the
            # first keyframe of a following utterance broadcasts a mid-open
            # mouth (measured on hum onsets).
            rest = _SILENCE_REST_OPENNESS if rms < silence_gate else _NEUTRAL
            openness = rest + (self._prev_targets[0] - rest) * _UNVOICED_DECAY
            width = _NEUTRAL + (self._prev_targets[1] - _NEUTRAL) * _UNVOICED_DECAY
            rounding = _NEUTRAL + (self._prev_targets[2] - _NEUTRAL) * _UNVOICED_DECAY

        openness_mapped = openness

        # Events (may override targets, e.g. nasal closes the mouth).
        speech = rms > 2.0 * max(
            _CLOSURE_FLOOR_MULT * self._noise_floor, _CLOSURE_PEAK_FRACTION * self._recent_peak
        )
        murmur_shape = (f1 <= _NASAL_F1_MAX_HZ or f1 == 0.0) and mid_ratio <= _NASAL_MID_RATIO_MAX
        event_fired = self._update_events(
            offset,
            rms,
            silence_gate,
            speech,
            voiced,
            centroid,
            low_ratio,
            f2_damped and murmur_shape,
            clarity,
            result,
        )
        if self._nasal_active:
            openness = min(openness, _NASAL_OPENNESS_MAX)
        elif (
            voiced
            and f2_damped
            and centroid < _NASAL_SOFT_CAP_CENTROID_HZ
            and low_ratio > _NASAL_SOFT_CAP_RATIO
        ):
            # Nasal-ish evidence before the state machine latches: cap the
            # continuous signal so the mouth starts closing within one hop.
            openness = min(openness, _NASAL_SOFT_CAP_OPENNESS)

        self._prev_targets = [openness, width, rounding]

        # Energy and pitch (normalized within session ranges).
        energy = float(
            np.log1p(_LOG_COMPRESSION * rms / (self._energy_max + dsp.EPSILON))
            / np.log1p(_LOG_COMPRESSION)
        )
        pitch_norm = 0.0
        if voiced:
            p_lo, p_hi = self._pitch_p5.value(), self._pitch_p95.value()
            if p_hi - p_lo >= 20.0:
                pitch_norm = _clamp01((pitch_hz - p_lo) / (p_hi - p_lo))
            else:
                pitch_norm = 0.5

        # Confidence: slot evidence × frame-to-frame stability × model fit ×
        # SNR, with convergence square-rooted so cold-start blending damps
        # (not flattens) the calibrated components.
        if f1_found and f2_found:
            c_slot = 1.0
        elif f1_found or f2_found:
            c_slot = _C_SLOT_PARTIAL
        else:
            c_slot = _C_LPC_FLOOR
        delta = max(abs(f1 - self._prev_f1), abs(f2 - self._prev_f2))
        delta_term = _clamp01(1.0 - delta / _FORMANT_DELTA_HZ)
        c_lpc = c_slot * (_C_LPC_FLOOR + (1.0 - _C_LPC_FLOOR) * delta_term)
        c_fit = _clamp01(float(np.log10(max(prediction_gain, 1.0))) / _C_FIT_LOG10_FULL)
        c_conv = min(1.0, self._voiced_frames / _CONVERGENCE_FRAMES)
        snr_db = 20.0 * np.log10((rms + dsp.EPSILON) / (3.0 * self._noise_floor + dsp.EPSILON))
        c_snr = _clamp01(float(snr_db) / _SNR_FULL_DB)
        confidence = c_lpc * c_fit * c_snr * float(np.sqrt(c_conv))

        self._prev_f1, self._prev_f2, self._prev_f3 = f1, f2, f3

        if self._collect_debug:
            self.debug_features.append(
                LipsyncDebugFrame(
                    offset=offset,
                    f1=f1 if f1_found else 0.0,
                    f2=f2 if f2_found else 0.0,
                    f3=f3 if f3_found else 0.0,
                    f2_bandwidth=f2_bandwidth,
                    f1_broad=f1_broad,
                    f2_broad=f2_broad,
                    f2_low=f2_low,
                    pitch_hz=pitch_hz,
                    voiced=voiced,
                    clarity=clarity,
                    rms=rms,
                    centroid=centroid,
                    low_band_ratio=low_ratio,
                    mid_band_ratio=mid_ratio,
                    prediction_gain=prediction_gain,
                    nasal_active=self._nasal_active,
                    openness_mapped=openness_mapped,
                    openness_target=openness,
                    openness_conditioned=float("nan"),  # filled when this hop is conditioned
                    c_lpc=c_lpc,
                    c_conv=c_conv,
                    c_snr=c_snr,
                    c_fit=c_fit,
                    confidence=confidence,
                    f1_lo=f1_lo,
                    f1_hi=f1_hi,
                    f2_lo=f2_lo,
                    f2_hi=f2_hi,
                )
            )

        # Conditioning: centered median-3 → slew clamp → dead-band/heartbeat
        # gate. This hop's targets complete the previous hop's median, so the
        # previous hop is what gets conditioned and emitted now.
        self._median_hist.append([openness, width, rounding])
        if len(self._median_hist) > _MEDIAN_TAPS:
            self._median_hist.pop(0)
        previous = self._pending_hop
        self._pending_hop = _PendingHop(offset, [], energy, pitch_norm, confidence, event_fired)
        if previous is not None:
            self._condition_and_emit(previous, self._median_hist, result, debug_index=-2)

    def _bound_hold(self, slot: int, value: float, found: bool, prior: tuple) -> float:
        if found:
            self._hold_counts[slot] = 0
            return value
        self._hold_counts[slot] += 1
        if self._hold_counts[slot] > _HOLD_MAX_HOPS and value > 0.0:
            center = (prior[0] + prior[1]) / 2.0
            return value + (center - value) * _HOLD_DECAY
        return value

    def _window_frames(self, lpc_raw: np.ndarray, prev_sample: float):
        """Pre-emphasize the LPC frame and window both framings into scratch."""
        windowed = self._windowed
        windowed[:] = lpc_raw
        windowed[1:] -= dsp.PRE_EMPHASIS * lpc_raw[:-1]
        windowed[0] -= dsp.PRE_EMPHASIS * prev_sample
        if self._lpc_pad:
            pad = self._lpc_pad
            np.multiply(
                windowed[pad : pad + dsp.FRAME_SIZE], self._window_short, out=self._windowed_short
            )
        windowed *= self._window

    def _update_adaptation(
        self, f1: float, f2: float, pitch_hz: float, f1_found: bool, f2_found: bool
    ):
        self._voiced_frames += 1

        # High-pitched voices sit higher in formant space: shift priors once.
        if not self._priors_shifted and len(self._pitch_probe) < _PITCH_PROBE_FRAMES:
            self._pitch_probe.append(pitch_hz)
            if len(self._pitch_probe) == _PITCH_PROBE_FRAMES:
                self._priors_shifted = True
                if float(np.median(self._pitch_probe)) > _HIGH_PITCH_HZ:
                    self._f1_prior = (_F1_PRIOR[0] * _PRIOR_SHIFT, _F1_PRIOR[1] * _PRIOR_SHIFT)
                    self._f2_prior = (_F2_PRIOR[0] * _PRIOR_SHIFT, _F2_PRIOR[1] * _PRIOR_SHIFT)

        if f1_found and self._accept_adaptation(0, f1):
            self._f1_p5.add(f1)
            self._f1_p95.add(f1)
            self._shift_ring.append(f1)
            if len(self._shift_ring) > _SHIFT_RING:
                self._shift_ring.pop(0)
            # Distribution shift (e.g. a voice change): fall back toward priors.
            if len(self._shift_ring) == _SHIFT_RING and self._prior_decay == 0.0:
                recent = float(np.median(self._shift_ring[-_SHIFT_RECENT:]))
                overall = float(np.median(self._shift_ring))
                if overall > 0.0 and abs(recent - overall) / overall > _SHIFT_DEVIATION:
                    self._prior_decay = 1.0
                    self._voiced_frames = _SHIFT_RETAINED_FRAMES
        if f2_found and self._accept_adaptation(1, f2):
            self._f2_p5.add(f2)
            self._f2_p95.add(f2)
        if pitch_hz > 0.0:
            self._pitch_p5.add(pitch_hz)
            self._pitch_p95.add(pitch_hz)

    def _accept_adaptation(self, slot: int, value: float) -> bool:
        """Reject isolated large jumps from the learned ranges; accept real shifts."""
        last = self._last_adapt[slot]
        if (
            last > 0.0
            and abs(value - last) > _ADAPT_SPIKE_HZ
            and self._adapt_skips[slot] < _ADAPT_MAX_SKIPS
        ):
            self._adapt_skips[slot] += 1
            return False
        self._adapt_skips[slot] = 0
        self._last_adapt[slot] = value
        return True

    def _effective_range(
        self,
        p5: dsp.P2QuantileEstimator,
        p95: dsp.P2QuantileEstimator,
        prior: tuple[float, float],
    ) -> tuple[float, float]:
        lo, hi = prior
        if p5.count >= _MIN_ESTIMATOR_COUNT:
            learned_lo, learned_hi = p5.value(), p95.value()
            if learned_hi - learned_lo >= _MIN_LEARNED_SPAN_HZ:
                weight = min(1.0, self._voiced_frames / _CONVERGENCE_FRAMES)
                lo = _lerp(prior[0], learned_lo, weight)
                hi = _lerp(prior[1], learned_hi, weight)
        if self._prior_decay > 0.0:
            lo = _lerp(lo, prior[0], self._prior_decay)
            hi = _lerp(hi, prior[1], self._prior_decay)
        return lo, hi

    #
    # Events
    #

    def _update_events(
        self,
        offset: float,
        rms: float,
        silence_gate: float,
        speech: bool,
        voiced: bool,
        centroid: float,
        low_ratio: float,
        f2_damped: bool,
        clarity: float,
        result: LipsyncFrameResult,
    ) -> bool:
        fired = False

        self._speech_hist.append(speech)
        if len(self._speech_hist) > _SPEECH_HIST_HOPS:
            self._speech_hist.pop(0)

        # Silence: sustained sub-gate energy emits one event per stretch.
        if rms < silence_gate:
            if self._silence_run_hops == 0:
                self._silence_start_offset = offset
            self._silence_run_hops += 1
            if self._silence_run_hops >= _SILENCE_EVENT_HOPS and not self._silence_emitted:
                result.events.append(
                    LipsyncEvent(
                        offset=self._silence_start_offset,
                        kind=LipsyncEventKind.SILENCE,
                        duration=0.0,
                        confidence=0.9,
                    )
                )
                self._silence_emitted = True
                fired = True
        else:
            self._silence_run_hops = 0
            self._silence_emitted = False

        # Closure candidates: bounded energy dip with speech right before it.
        closure_thresh = max(
            _CLOSURE_FLOOR_MULT * self._noise_floor, _CLOSURE_PEAK_FRACTION * self._recent_peak
        )
        low = rms < closure_thresh
        if low:
            if self._low_run_hops == 0:
                self._low_run_start_offset = offset
                self._low_run_min_rms = rms
            self._low_run_hops += 1
            self._low_run_min_rms = min(self._low_run_min_rms, rms)
        else:
            if self._low_run_hops:
                run_hops = self._low_run_hops
                duration = run_hops * HOP_SECONDS
                speech_before = any(self._speech_hist[: -run_hops - 1][-_SPEECH_WINDOW_HOPS:])
                if run_hops >= _CLOSURE_MIN_HOPS and duration <= _CLOSURE_MAX_SEC and speech_before:
                    depth = _clamp01(1.0 - self._low_run_min_rms / (closure_thresh + dsp.EPSILON))
                    self._pending_closures.append(
                        _PendingClosure(
                            offset=self._low_run_start_offset,
                            duration=duration,
                            depth=depth,
                            deadline_hop=self._hops + _SPEECH_WINDOW_HOPS,
                        )
                    )
            self._low_run_hops = 0

        # Confirm pending closures on trailing speech; expire the rest.
        if self._pending_closures:
            remaining = []
            for pending in self._pending_closures:
                if speech:
                    result.events.append(
                        LipsyncEvent(
                            offset=pending.offset,
                            kind=LipsyncEventKind.CLOSURE,
                            duration=pending.duration,
                            confidence=pending.depth,
                        )
                    )
                    fired = True
                elif self._hops <= pending.deadline_hop:
                    remaining.append(pending)
            self._pending_closures = remaining

        # Nasal: enter/exit with hysteresis; overrides openness while active.
        nasal_now = (
            voiced
            and centroid < _NASAL_CENTROID_MAX_HZ
            and low_ratio > _NASAL_LOW_RATIO_MIN
            and f2_damped
        )
        if nasal_now:
            self._nasal_enter_count += 1
            self._nasal_exit_count = 0
            if not self._nasal_active and self._nasal_enter_count >= _NASAL_HYSTERESIS_HOPS:
                self._nasal_active = True
                result.events.append(
                    LipsyncEvent(
                        offset=offset,
                        kind=LipsyncEventKind.NASAL,
                        duration=0.0,
                        confidence=min(low_ratio, max(clarity, 0.0)),
                    )
                )
                fired = True
        else:
            self._nasal_enter_count = 0
            if self._nasal_active:
                self._nasal_exit_count += 1
                if self._nasal_exit_count >= _NASAL_HYSTERESIS_HOPS:
                    self._nasal_active = False
                    self._nasal_exit_count = 0

        return fired

    #
    # Conditioning & emission
    #

    def _condition_and_emit(
        self,
        hop: _PendingHop,
        taps: list[list[float]],
        result: LipsyncFrameResult,
        debug_index: int,
    ):
        """Condition one hop from the target vectors around it, then gate/emit it."""
        conditioned = []
        for i in range(3):
            value = float(np.median([t[i] for t in taps]))
            if self._cond_prev is not None:
                prev = self._cond_prev[i]
                delta = value - prev
                if delta > _SLEW_MAX_PER_HOP:
                    value = prev + _SLEW_MAX_PER_HOP
                elif delta < -_SLEW_MAX_PER_HOP:
                    value = prev - _SLEW_MAX_PER_HOP
            conditioned.append(value)
        self._cond_prev = conditioned
        hop.params = conditioned
        if self._collect_debug and self.debug_features:
            self.debug_features[debug_index].openness_conditioned = conditioned[0]
        self._maybe_emit_keyframe(hop, result)
        self._last_conditioned = hop

    def _flush_conditioning(self, result: LipsyncFrameResult):
        """Condition and emit the last hop (edge-replicated median) at utterance end."""
        hop = self._pending_hop
        if hop is None:
            return
        self._pending_hop = None
        taps = self._median_hist[-2:] + self._median_hist[-1:]  # replicate the edge
        self._condition_and_emit(hop, taps, result, debug_index=-1)

    def _maybe_emit_keyframe(self, hop: _PendingHop, result: LipsyncFrameResult):
        params = hop.params
        last = self._last_emitted
        if not hop.event_fired and last is not None:
            moved = max(abs(params[i] - last[i]) for i in range(3))
            # Heartbeats only run while speech is active: during a silence
            # stretch the SILENCE event has already parked the client at
            # neutral, so heartbeats there are pure wire waste.
            silence_active = self._silence_run_hops >= _SILENCE_EVENT_HOPS
            heartbeat_due = (
                not silence_active and (hop.offset - self._last_emit_offset) >= self._heartbeat_sec
            )
            if moved <= self._dead_band and not heartbeat_due:
                return
            anchor = self._last_conditioned
            if (
                _ANCHOR_KEYFRAMES
                and moved > self._dead_band
                and anchor is not None
                and anchor.offset > self._last_emit_offset
            ):
                self._emit_keyframe(anchor, result)
        self._emit_keyframe(hop, result)

    def _emit_keyframe(self, hop: _PendingHop, result: LipsyncFrameResult):
        self._last_emitted = hop.params
        self._last_emit_offset = hop.offset
        result.keyframes.append(
            LipsyncKeyframe(
                offset=hop.offset,
                openness=hop.params[0],
                width=hop.params[1],
                rounding=hop.params[2],
                energy=hop.energy,
                pitch=hop.pitch,
                confidence=hop.confidence,
            )
        )
