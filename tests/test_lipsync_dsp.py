#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import numpy as np

from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
from lipsync.dsp import (
    ANALYSIS_SAMPLE_RATE,
    FRAME_SIZE,
    PRE_EMPHASIS,
    P2QuantileEstimator,
    levinson_durbin,
    lpc_coefficients,
    lpc_formants,
    lpc_residual_pitch,
    rms_energy,
    spectral_nasal_features,
)
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.types import LipsyncEventKind
from tests.synth import synth_vowel

#
# Synthetic-signal helpers (test-only, pure numpy).
#


def preemphasize(x):
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - PRE_EMPHASIS * x[:-1]
    return y


_HAMMING = np.hamming(FRAME_SIZE).astype(np.float32)


def analysis_frame(x, start):
    """One pre-emphasized, windowed frame, as the analyzer prepares them."""
    return (preemphasize(x)[start : start + FRAME_SIZE] * _HAMMING).astype(np.float32)


def median_formants(x, starts):
    """Median F1/F2 across several analysis frames of a signal."""
    f1s, f2s = [], []
    for start in starts:
        frame = analysis_frame(x, start)
        estimate = lpc_formants(frame)
        if estimate.plausible:
            f1s.append(estimate.f1)
            f2s.append(estimate.f2)
    return float(np.median(f1s)), float(np.median(f2s))


_FRAME_STARTS = range(1600, 6400, 400)  # 0.1 s .. 0.4 s, clear of onset transient


class TestLevinsonDurbin(unittest.TestCase):
    def test_reconstructs_ar_coefficients(self):
        # AR(4) process from two stable resonators; recover its coefficients.
        rng = np.random.default_rng(7)
        a_true = np.array([1.0], dtype=np.float64)
        for fc, r in ((500, 0.9), (1500, 0.85)):
            theta = 2 * np.pi * fc / ANALYSIS_SAMPLE_RATE
            a_true = np.convolve(a_true, [1.0, -2 * r * np.cos(theta), r * r])
        n = 8192
        e = rng.standard_normal(n)
        x = np.zeros(n)
        x[:4] = e[:4]
        for i in range(4, n):
            x[i] = e[i] - np.dot(a_true[1:], x[i - 4 : i][::-1])
        x = x.astype(np.float32)

        order = 4
        r = np.empty(order + 1, dtype=np.float32)
        r[0] = np.dot(x, x) * (1.0 + 1e-6)
        for lag in range(1, order + 1):
            r[lag] = np.dot(x[: n - lag], x[lag:])
        a_est, err = levinson_durbin(r, order)

        np.testing.assert_allclose(a_est, a_true, atol=0.05 * np.abs(a_true).max())
        self.assertGreater(r[0] / err, 10)  # strongly resonant: high prediction gain

    def test_degenerate_input_returns_identity(self):
        a, err = levinson_durbin(np.zeros(13, dtype=np.float32), 12)
        self.assertEqual(a[0], 1.0)
        self.assertFalse(np.any(a[1:]))
        self.assertEqual(err, 0.0)


class TestFormants(unittest.TestCase):
    def test_formants_recovered_from_synthetic_vowels(self):
        for name, f1, f2 in (("aa", 700, 1200), ("ii", 300, 2300), ("uu", 300, 800)):
            with self.subTest(vowel=name):
                x = synth_vowel(f1, f2)
                got_f1, got_f2 = median_formants(x, _FRAME_STARTS)
                self.assertAlmostEqual(got_f1, f1, delta=40, msg=f"F1 for /{name}/")
                self.assertAlmostEqual(got_f2, f2, delta=40, msg=f"F2 for /{name}/")

    def test_damped_f2_not_mis_slotted(self):
        # F2 resonator too broad to yield a valid root: the F2 slot must stay
        # empty rather than promoting F3 (2900 Hz) into it.
        x = synth_vowel(700, 1200, bandwidths=(90, 650, 150))
        prev = None
        f2s, f3s = [], []
        for start in _FRAME_STARTS:
            estimate = lpc_formants(analysis_frame(x, start), prev=prev)
            prev = estimate
            if estimate.f2 > 0:
                f2s.append(estimate.f2)
            if estimate.f3 > 0:
                f3s.append(estimate.f3)
        for f2 in f2s:  # any F2 that does appear must not be F3-region
            self.assertLess(f2, 2000)
        if f3s:
            self.assertAlmostEqual(float(np.median(f3s)), 2900, delta=100)

    def test_low_spurious_root_does_not_shift_slots(self):
        # An extra low pole (nasal-like) must not shift every slot up one.
        x = synth_vowel(700, 1200, extra=[(280, 200)])
        prev = None
        f1s, f2s = [], []
        for start in _FRAME_STARTS:
            estimate = lpc_formants(analysis_frame(x, start), prev=prev)
            prev = estimate
            if estimate.plausible:
                f1s.append(estimate.f1)
                f2s.append(estimate.f2)
        # The added pole biases the LPC fit slightly; the assertion is that
        # slots did not SHIFT (a shift would read F1≈280, F2≈700).
        self.assertGreater(float(np.median(f1s)), 450)
        self.assertAlmostEqual(float(np.median(f1s)), 700, delta=120)
        self.assertAlmostEqual(float(np.median(f2s)), 1200, delta=150)

    def test_implausible_on_silence_and_stable_on_noise(self):
        silence = np.zeros(FRAME_SIZE, dtype=np.float32)
        estimate = lpc_formants(silence)
        self.assertFalse(estimate.plausible)
        self.assertTrue(np.isfinite([estimate.f1, estimate.f2, estimate.f3]).all())

        rng = np.random.default_rng(3)
        noise = (rng.standard_normal(FRAME_SIZE) * _HAMMING).astype(np.float32)
        estimate = lpc_formants(noise)  # must not raise; values finite
        self.assertTrue(np.isfinite([estimate.f1, estimate.f2, estimate.f3]).all())


class TestPitch(unittest.TestCase):
    def _pitch_at(self, x, start):
        frame = analysis_frame(x, start)
        return lpc_residual_pitch(frame, lpc_coefficients(frame).coefficients)

    def test_pitch_on_synthetic_glottal_train(self):
        for f0, delta in ((120, 5), (280, 8)):
            with self.subTest(f0=f0):
                x = synth_vowel(700, 1200, f0=f0)
                estimates = [self._pitch_at(x, s) for s in _FRAME_STARTS]
                voiced = [e for e in estimates if e.voiced]
                self.assertGreaterEqual(len(voiced), 0.8 * len(estimates))
                median = float(np.median([e.frequency for e in voiced]))
                self.assertAlmostEqual(median, f0, delta=delta)

    def test_unvoiced_noise_not_voiced(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(ANALYSIS_SAMPLE_RATE // 2).astype(np.float32)
        estimates = [self._pitch_at(x, s) for s in _FRAME_STARTS]
        self.assertLess(sum(e.voiced for e in estimates), len(estimates) // 2)


class TestEnergyAndSpectral(unittest.TestCase):
    def test_rms_energy_scales(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(FRAME_SIZE).astype(np.float32)
        self.assertAlmostEqual(rms_energy(2.0 * x), 2.0 * rms_energy(x), delta=1e-3)
        self.assertLess(rms_energy(np.zeros(FRAME_SIZE, dtype=np.float32)), 1e-4)

    def test_spectral_low_band_ratio(self):
        t = np.arange(FRAME_SIZE, dtype=np.float32) / ANALYSIS_SAMPLE_RATE
        low = (np.sin(2 * np.pi * 200 * t) * _HAMMING).astype(np.float32)
        high = (np.sin(2 * np.pi * 3000 * t) * _HAMMING).astype(np.float32)

        low_centroid, low_ratio = spectral_nasal_features(low)
        high_centroid, high_ratio = spectral_nasal_features(high)

        self.assertGreater(low_ratio, 0.9)
        self.assertLess(high_ratio, 0.1)
        self.assertLess(low_centroid, high_centroid)


class TestP2QuantileEstimator(unittest.TestCase):
    def test_matches_numpy_percentile(self):
        rng = np.random.default_rng(42)
        samples = rng.lognormal(mean=6.0, sigma=0.4, size=10_000)
        for quantile in (0.05, 0.95):
            with self.subTest(quantile=quantile):
                estimator = P2QuantileEstimator(quantile)
                for sample in samples:
                    estimator.add(float(sample))
                expected = float(np.percentile(samples, quantile * 100))
                self.assertAlmostEqual(estimator.value(), expected, delta=0.03 * expected)
        self.assertEqual(estimator.count, len(samples))

    def test_early_values(self):
        estimator = P2QuantileEstimator(0.95)
        self.assertEqual(estimator.value(), 0.0)
        for value in (3.0, 1.0, 2.0):
            estimator.add(value)
        self.assertEqual(estimator.value(), 3.0)


#
# Analyzer-level tests: feed PCM directly, no pipeline.
#


def synth_nasal(f0=120, secs=0.8, fs=ANALYSIS_SAMPLE_RATE):
    """Voiced buzz through a single low resonator: dark spectrum, no F2."""
    n = int(secs * fs)
    x = np.zeros(n, dtype=np.float32)
    x[:: int(fs / f0)] = 1.0
    y = np.zeros_like(x)
    for i in range(n):  # glottal tilt
        y[i] = x[i] + PRE_EMPHASIS * y[i - 1]
    x = y
    r = np.exp(-np.pi * 100 / fs)
    c1, c2 = 2 * r * np.cos(2 * np.pi * 250 / fs), -r * r
    y = np.zeros_like(x)
    for i in range(n):
        y[i] = x[i] + c1 * y[i - 1] + c2 * y[i - 2]
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


async def run_analyzer(pcm, chunk_size=320, analyzer=None):
    """Feed PCM through an analyzer in chunks; return (keyframes, events, analyzer)."""
    if analyzer is None:
        analyzer = FormantLipsyncAnalyzer()
        await analyzer.start(ANALYSIS_SAMPLE_RATE)
    context = LipsyncAnalysisContext(context_id="test", sample_rate=ANALYSIS_SAMPLE_RATE)
    keyframes, events = [], []
    for i in range(0, len(pcm), chunk_size):
        result = await analyzer.analyze(pcm[i : i + chunk_size], context)
        keyframes += result.keyframes
        events += result.events
    result = await analyzer.flush(context)
    keyframes += result.keyframes
    events += result.events
    return keyframes, events, analyzer


def mean_late(keyframes, attr, after=0.3):
    values = [getattr(k, attr) for k in keyframes if k.offset >= after]
    return float(np.mean(values))


class TestFormantLipsyncAnalyzer(unittest.IsolatedAsyncioTestCase):
    async def test_vowel_openness_and_width_ordering(self):
        stats = {}
        for name, f1, f2 in (("aa", 700, 1200), ("ii", 300, 2300), ("uu", 300, 800)):
            keyframes, _, _ = await run_analyzer(synth_vowel(f1, f2, secs=1.0))
            self.assertTrue(keyframes)
            stats[name] = (
                mean_late(keyframes, "openness"),
                mean_late(keyframes, "width"),
                mean_late(keyframes, "rounding"),
            )
        self.assertGreater(stats["aa"][0], stats["ii"][0])
        self.assertGreater(stats["aa"][0], stats["uu"][0])
        self.assertGreater(stats["ii"][1], stats["uu"][1])
        self.assertGreater(stats["uu"][2], stats["ii"][2])

    async def test_nasal_override(self):
        keyframes, events, _ = await run_analyzer(synth_nasal())
        self.assertTrue(any(e.kind == LipsyncEventKind.NASAL for e in events))
        self.assertLess(mean_late(keyframes, "openness"), 0.15)
        # Peaks, not just the mean: onset keyframes must not broadcast an
        # open mouth either.
        openness = [k.openness for k in keyframes if k.offset > 0.1]
        self.assertLessEqual(float(np.percentile(openness, 90)), 0.25)

    async def test_silence_event_once(self):
        pcm = np.concatenate(
            [synth_vowel(700, 1200, secs=0.4), np.zeros(ANALYSIS_SAMPLE_RATE, dtype=np.float32)]
        )
        _, events, _ = await run_analyzer(pcm)
        silences = [e for e in events if e.kind == LipsyncEventKind.SILENCE]
        self.assertEqual(len(silences), 1)

    async def test_closure_detection(self):
        vowel = synth_vowel(700, 1200, secs=0.3)
        gap = np.zeros(int(0.08 * ANALYSIS_SAMPLE_RATE), dtype=np.float32)
        _, events, _ = await run_analyzer(np.concatenate([vowel, gap, vowel]))
        closures = [e for e in events if e.kind == LipsyncEventKind.CLOSURE]
        self.assertEqual(len(closures), 1)
        self.assertGreaterEqual(closures[0].duration, 0.02)
        self.assertLessEqual(closures[0].duration, 0.16)
        self.assertAlmostEqual(closures[0].offset, 0.3, delta=0.08)

        long_gap = np.zeros(int(0.5 * ANALYSIS_SAMPLE_RATE), dtype=np.float32)
        _, events, _ = await run_analyzer(np.concatenate([vowel, long_gap, vowel]))
        closures = [e for e in events if e.kind == LipsyncEventKind.CLOSURE]
        self.assertEqual(len(closures), 0)

    async def test_conditioning_caps_keyframe_rate(self):
        keyframes, _, _ = await run_analyzer(synth_vowel(700, 1200, secs=3.0))
        steady = [k for k in keyframes if k.offset >= 2.0]
        self.assertLessEqual(len(steady), 6)  # heartbeat-dominated: ~4/s over the last second

    async def test_no_heartbeats_during_silence(self):
        pcm = np.concatenate(
            [
                synth_vowel(700, 1200, secs=0.5),
                np.zeros(int(1.2 * ANALYSIS_SAMPLE_RATE), dtype=np.float32),
            ]
        )
        keyframes, events, _ = await run_analyzer(pcm)
        self.assertTrue(any(e.kind == LipsyncEventKind.SILENCE for e in events))
        # After the silence event fires (~0.8 s) no heartbeat keyframes.
        self.assertEqual([k.offset for k in keyframes if k.offset > 1.0], [])

    async def test_analyzer_chunk_invariance(self):
        pcm = synth_vowel(700, 1200, secs=1.0)
        small_kf, small_ev, _ = await run_analyzer(pcm, chunk_size=160)
        large_kf, large_ev, _ = await run_analyzer(pcm, chunk_size=5120)

        self.assertEqual(len(small_kf), len(large_kf))
        for a, b in zip(small_kf, large_kf):
            self.assertAlmostEqual(a.offset, b.offset, delta=1e-6)
            self.assertAlmostEqual(a.openness, b.openness, delta=1e-5)
            self.assertAlmostEqual(a.width, b.width, delta=1e-5)
        self.assertEqual(
            [(e.kind, round(e.offset, 6)) for e in small_ev],
            [(e.kind, round(e.offset, 6)) for e in large_ev],
        )

    async def test_confidence_discriminates_signal_from_noise(self):
        rng = np.random.default_rng(31)
        vowel = synth_vowel(700, 1200, secs=1.0)
        noise = (rng.standard_normal(ANALYSIS_SAMPLE_RATE) * 0.3).astype(np.float32)
        analyzer = FormantLipsyncAnalyzer(collect_debug=True)
        await analyzer.start(ANALYSIS_SAMPLE_RATE)
        await run_analyzer(np.concatenate([vowel, noise]), analyzer=analyzer)

        frames = analyzer.debug_features
        vowel_conf = float(np.mean([f.confidence for f in frames if 0.2 < f.offset < 0.95]))
        noise_conf = float(np.mean([f.confidence for f in frames if f.offset > 1.1]))
        self.assertGreater(vowel_conf - noise_conf, 0.2)

    async def test_adaptation_ignores_single_frame_spikes(self):
        rng = np.random.default_rng(23)
        clean = synth_vowel(700, 1200, secs=1.0)
        corrupted = clean.copy()
        corrupted[8000:8400] = rng.standard_normal(400).astype(np.float32) * 0.5

        _, _, clean_analyzer = await run_analyzer(clean)
        _, _, corrupted_analyzer = await run_analyzer(corrupted)
        self.assertAlmostEqual(
            clean_analyzer._f1_p95.value(), corrupted_analyzer._f1_p95.value(), delta=25
        )

    async def test_reset_preserves_adaptation(self):
        _, _, analyzer = await run_analyzer(synth_vowel(700, 1200, secs=1.0))
        voiced_before = analyzer._voiced_frames
        self.assertGreater(voiced_before, 0)
        await analyzer.reset()
        self.assertEqual(analyzer._voiced_frames, voiced_before)
        self.assertEqual(analyzer._hops, 0)

    async def test_debug_tap_collects_features(self):
        analyzer = FormantLipsyncAnalyzer(collect_debug=True)
        await analyzer.start(ANALYSIS_SAMPLE_RATE)
        await run_analyzer(synth_vowel(700, 1200, secs=0.5), analyzer=analyzer)
        self.assertTrue(analyzer.debug_features)
        voiced = [d for d in analyzer.debug_features if d.voiced]
        self.assertTrue(voiced)
        f1_median = float(np.median([d.f1 for d in voiced if d.f1 > 0]))
        self.assertAlmostEqual(f1_median, 700, delta=60)


if __name__ == "__main__":
    unittest.main()
