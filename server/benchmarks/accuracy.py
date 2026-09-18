"""Accuracy benchmark: score the lipsync analyzer against Praat + corpus expectations.

Three diagnostic layers (plans/benchmark-harness-accuracy.md):

- L1 (DSP): our raw per-hop F1/F2 in Hz vs Praat reference tracks.
- L2 (trajectory): our normalized openness/width shape vs per-clip-normalized
  Praat tracks (scale-invariant Pearson r).
- L3 (expectations): event counts and distribution bounds from corpus.yaml.

Run: uv run python -m benchmarks.accuracy [--offline] [--compare]
"""

import argparse
import ast
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import parselmouth
from loguru import logger

from benchmarks.common import (
    RESULTS_DIR,
    Clip,
    Sentence,
    Voice,
    chunks_16k_float32,
    get_clip,
    load_corpus,
    teardowns_pending,
)
from lipsync import dsp, formant_lipsync_analyzer
from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.types import LipsyncEventKind

BASELINE_PATH = RESULTS_DIR / "baseline.json"


def baseline_path(provider: str) -> Path:
    """One baseline per provider (``baseline.json`` is the Cartesia corpus)."""
    return BASELINE_PATH if provider == "cartesia" else RESULTS_DIR / f"baseline-{provider}.json"


# Post-convergence window start: metrics are reported for the full clip and
# for t >= this, isolating adaptive-normalization warmup cost.
POST_CONV_SECS = 1.5

# Timing lag search: our interpolated track is slid against the reference on
# this grid; the lag is reported only when it improves r by at least
# LAG_MIN_GAIN and lies strictly inside the window (else NaN). Positive = our
# track is late.
LAG_GRID_MS = np.arange(-120, 121, 5)
LAG_MIN_GAIN = 0.05

# Composite score: component -> (weight, full-marks value, zero-marks value).
# For higher-is-better metrics full > zero; linear ramp between. PROVISIONAL:
# frozen after the first eyeballed run — never tune these and the
# implementation in the same change. Deltas (--compare) are the real signal.
COMPOSITE = {
    "f1_mae_hz": (20, 50.0, 200.0),
    "f2_mae_hz": (20, 80.0, 300.0),
    "openness_r": (15, 0.85, 0.0),
    "width_r": (15, 0.85, 0.0),
    "checks_closure": (10, 1.0, 0.0),
    "checks_nasal": (10, 1.0, 0.0),
    "checks_silence": (5, 1.0, 0.0),
    "convergence_s": (5, 3.0, 8.0),
}

# Expectation key -> (metric it reads, satisfied predicate, composite bucket).
# Vowel-probe peak checks have no composite bucket of their own (they overlap
# L2); they still count toward event_satisfaction and clip failures.
_GE = lambda m, v: m >= v  # noqa: E731
_LE = lambda m, v: m <= v  # noqa: E731
CHECKS = {
    "closures_min": ("closures", _GE, "closure"),
    "closures_max": ("closures", _LE, "closure"),
    "nasals_min": ("nasals", _GE, "nasal"),
    "nasals_max": ("nasals", _LE, "nasal"),
    "nasal_fraction_max": ("nasal_fraction", _LE, "nasal"),
    "silences_min": ("silences", _GE, "silence"),
    "silences_max": ("silences", _LE, "silence"),
    "openness_p90_max": ("openness_p90", _LE, "nasal"),
    "openness_p90_min": ("openness_p90_pc", _GE, None),
    "width_p90_min": ("width_p90_pc", _GE, None),
    "width_p90_max": ("width_p90_pc", _LE, None),
    "rounding_p90_min": ("rounding_p90_pc", _GE, None),
}


@dataclass
class ClipResult:
    clip_label: str
    sentence_id: str
    voice_id: str
    take: int
    metrics: dict[str, float]
    checks: dict[str, bool] = field(default_factory=dict)


#
# Reference tracks (Praat via parselmouth), sampled at our hop offsets
#


@dataclass
class Reference:
    f1: np.ndarray
    f2: np.ndarray
    voiced: np.ndarray
    intensity_db: np.ndarray


def compute_reference(clip: Clip, offsets: np.ndarray, ceiling: float) -> Reference:
    samples = np.frombuffer(clip.pcm, dtype=np.int16).astype(np.float64) / 32768.0
    sound = parselmouth.Sound(samples, sampling_frequency=clip.sample_rate)
    formant = sound.to_formant_burg(
        time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling
    )
    pitch = sound.to_pitch(time_step=0.01)
    intensity = sound.to_intensity(time_step=0.01)

    f1 = np.array([formant.get_value_at_time(1, t) for t in offsets])
    f2 = np.array([formant.get_value_at_time(2, t) for t in offsets])
    pitch_vals = np.array([pitch.get_value_at_time(t) for t in offsets])
    db = np.array(
        [parselmouth.praat.call(intensity, "Get value at time", t, "cubic") for t in offsets]
    )
    return Reference(f1=f1, f2=f2, voiced=~np.isnan(pitch_vals), intensity_db=db)


#
# Analyzer driver (mirrors LipsyncProcessor ingest; cold by default)
#


async def analyze_clip(clip: Clip, warm_pcm: bytes | None = None, warm_rate: int = 0):
    analyzer = FormantLipsyncAnalyzer(collect_debug=True)
    await analyzer.start(clip.sample_rate)

    if warm_pcm:
        warm_ctx = LipsyncAnalysisContext(context_id="warm", sample_rate=warm_rate)
        async for chunk in chunks_16k_float32(warm_pcm, warm_rate):
            await analyzer.analyze(chunk, warm_ctx)
        await analyzer.flush(warm_ctx)
    warm_frames = len(analyzer.debug_features)

    context = LipsyncAnalysisContext(context_id="bench", sample_rate=clip.sample_rate)
    keyframes, events = [], []
    async for chunk in chunks_16k_float32(clip.pcm, clip.sample_rate):
        result = await analyzer.analyze(chunk, context)
        keyframes += result.keyframes
        events += result.events
    result = await analyzer.flush(context)
    keyframes += result.keyframes
    events += result.events
    return keyframes, events, analyzer.debug_features[warm_frames:]


#
# Metrics
#


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _norm_track(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = values[mask]
    if masked.size < 8:
        return np.full_like(values, np.nan)
    p5, p95 = np.nanpercentile(masked, 5), np.nanpercentile(masked, 95)
    if p95 - p5 < 1e-9:
        return np.full_like(values, np.nan)
    return np.clip((values - p5) / (p95 - p5), 0.0, 1.0)


def _convergence_secs(debug) -> float:
    final = debug[-1]
    spans = (final.f1_hi - final.f1_lo, final.f2_hi - final.f2_lo)
    if min(spans) < 1e-9:
        return float("nan")

    def deviation(d) -> float:
        return max(
            abs(d.f1_lo - final.f1_lo) / spans[0],
            abs(d.f1_hi - final.f1_hi) / spans[0],
            abs(d.f2_lo - final.f2_lo) / spans[1],
            abs(d.f2_hi - final.f2_hi) / spans[1],
        )

    last_bad = -1
    for i, d in enumerate(debug):
        if deviation(d) > 0.10:
            last_bad = i
    if last_bad + 1 >= len(debug):
        return debug[-1].offset  # never settled
    return debug[last_bad + 1].offset if last_bad >= 0 else 0.0


def compute_metrics(clip: Clip, keyframes, events, debug, ref: Reference) -> dict[str, float]:
    m: dict[str, float] = {}
    offs = np.array([d.offset for d in debug])
    ours_f1 = np.array([d.f1 for d in debug])
    ours_f2 = np.array([d.f2 for d in debug])
    ours_voiced = np.array([d.voiced for d in debug])
    conf = np.array([d.confidence for d in debug])
    nasal_active = np.array([d.nasal_active for d in debug])

    def add_windowed(name: str, values_fn):
        """Store metric for the full clip and the post-convergence window."""
        m[name] = values_fn(np.ones_like(offs, dtype=bool))
        m[f"{name}_pc"] = values_fn(offs >= POST_CONV_SECS)

    # L1 — raw formants vs Praat (masked to frames where both sides commit;
    # per-slot: F2 may be absent on frames where F1 was found).
    l1 = ref.voiced & ours_voiced & np.isfinite(ref.f1) & (ours_f1 > 0)
    l1_f2 = ref.voiced & ours_voiced & np.isfinite(ref.f2) & (ours_f2 > 0)
    err_f1 = np.abs(ours_f1 - ref.f1)
    err_f2 = np.abs(ours_f2 - ref.f2)
    add_windowed(
        "f1_mae_hz",
        lambda w: float(np.mean(err_f1[l1 & w])) if (l1 & w).sum() >= 8 else float("nan"),
    )
    add_windowed(
        "f2_mae_hz",
        lambda w: float(np.mean(err_f2[l1_f2 & w])) if (l1_f2 & w).sum() >= 8 else float("nan"),
    )
    m["f1_r"] = _pearson(ours_f1[l1], ref.f1[l1])
    m["voicing_agreement"] = float(np.mean(ours_voiced == ref.voiced))

    # Coverage (reported, not scored): MAE above only sees hops where we
    # committed a slot, so a change that finds F1 on more frames can raise
    # f1_mae while improving the mouth. ``*_coverage`` is the fraction of
    # Praat-voiced hops L1 actually scores; ``*_slot_coverage`` ignores our own
    # voicing decision, isolating the LPC/slot path from the pitch detector.
    # Both are 0.0 (never NaN) when nothing commits, so a zero-coverage clip
    # cannot vanish from an aggregate.
    ref_f1_hops = ref.voiced & np.isfinite(ref.f1)
    ref_f2_hops = ref.voiced & np.isfinite(ref.f2)

    def coverage(scored: np.ndarray, population: np.ndarray, w: np.ndarray) -> float:
        n = int((population & w).sum())
        return float((scored & w).sum() / n) if n else 0.0

    add_windowed("f1_coverage", lambda w: coverage(l1, ref_f1_hops, w))
    add_windowed("f2_coverage", lambda w: coverage(l1_f2, ref_f2_hops, w))
    add_windowed(
        "f1_slot_coverage", lambda w: coverage(ref_f1_hops & (ours_f1 > 0), ref_f1_hops, w)
    )
    add_windowed(
        "f2_slot_coverage", lambda w: coverage(ref_f2_hops & (ours_f2 > 0), ref_f2_hops, w)
    )
    m["f1_scored_n"] = float(l1.sum())
    m["f2_scored_n"] = float(l1_f2.sum())
    m["ref_voiced_n"] = float(ref.voiced.sum())

    # Nasal override duty cycle over Praat-voiced hops (reported; bounded by
    # ``nasal_fraction_max`` on low-nasal sentences). The nasal event count
    # alone cannot see a detector that latches on close vowels.
    ref_voiced_n = int(ref.voiced.sum())
    m["nasal_fraction"] = (
        float((nasal_active & ref.voiced).sum() / ref_voiced_n) if ref_voiced_n else 0.0
    )

    # L2 — normalized trajectory shape (scale-invariant).
    if len(keyframes) >= 2:
        kf_offs = np.array([k.offset for k in keyframes])
        kf_open = [k.openness for k in keyframes]
        kf_width = [k.width for k in keyframes]
        ours_open = np.interp(offs, kf_offs, kf_open)
        ours_width = np.interp(offs, kf_offs, kf_width)
        ours_round = np.interp(offs, kf_offs, [k.rounding for k in keyframes])
        ours_energy = np.interp(offs, kf_offs, [k.energy for k in keyframes])

        l2 = ref.voiced & np.isfinite(ref.f1)
        ref_open = _norm_track(ref.f1, l2)
        ref_width = _norm_track(ref.f2, ref.voiced & np.isfinite(ref.f2))
        add_windowed("openness_r", lambda w: _pearson(ours_open[l2 & w], ref_open[l2 & w]))
        add_windowed(
            "width_r",
            lambda w: _pearson(
                ours_width[ref.voiced & np.isfinite(ref.f2) & w],
                ref_width[ref.voiced & np.isfinite(ref.f2) & w],
            ),
        )
        m["openness_mae"] = (
            float(np.nanmean(np.abs(ours_open[l2] - ref_open[l2])))
            if l2.sum() >= 8
            else float("nan")
        )

        # Timing: best cross-correlation shift of our track against the
        # reference (review §6.3). Pearson r at zero lag cannot tell a late
        # track from a wrong one; ``*_r_best`` is r at the best lag.
        def lag(kf_values, ref_track, mask):
            if mask.sum() < 8:
                return float("nan"), float("nan")
            r_by_lag = [
                _pearson(np.interp(offs - tau / 1000.0, kf_offs, kf_values)[mask], ref_track[mask])
                for tau in LAG_GRID_MS
            ]
            r0 = r_by_lag[len(LAG_GRID_MS) // 2]
            best = int(np.nanargmax(r_by_lag)) if not np.all(np.isnan(r_by_lag)) else -1
            if best < 0:
                return float("nan"), float("nan")
            r_best = r_by_lag[best]
            significant = (
                np.isfinite(r0) and r_best - r0 >= LAG_MIN_GAIN and 0 < best < len(LAG_GRID_MS) - 1
            )
            # ours(t - tau) matches ref(t) at tau = -lag, so a late track has
            # a negative tau; report lag = -tau (positive = our track is late).
            return (-float(LAG_GRID_MS[best]) if significant else float("nan")), float(r_best)

        l2_width = ref.voiced & np.isfinite(ref.f2)
        m["openness_lag_ms"], m["openness_r_best"] = lag(kf_open, ref_open, l2)
        m["width_lag_ms"], m["width_r_best"] = lag(kf_width, ref_width, l2_width)

        # Jitter: mean |second difference| of the interpolated openness over
        # speech hops (review §6.4 guard; a rougher track is not a better one).
        speech = ref.intensity_db > np.nanmax(ref.intensity_db) - 25
        if speech.sum() >= 8:
            m["openness_jitter"] = float(np.mean(np.abs(np.diff(ours_open, 2))[speech[1:-1]]))

        db_ok = np.isfinite(ref.intensity_db)
        m["energy_r"] = _pearson(ours_energy[db_ok], _norm_track(ref.intensity_db, db_ok)[db_ok])

        # L3 peak-reaching stats over our voiced hops. The nasal guard
        # (openness_p90) covers the full clip including onsets; the vowel-probe
        # minima (_pc) cover the post-convergence window, matching the oracle's
        # steady-state framing — cold-start cost is measured by convergence_s
        # and gated client-side by confidence.
        if ours_voiced.sum() >= 4:
            m["openness_p90"] = float(np.percentile(ours_open[ours_voiced], 90))
        pc = ours_voiced & (offs >= POST_CONV_SECS)
        if pc.sum() >= 4:
            m["openness_p90_pc"] = float(np.percentile(ours_open[pc], 90))
            m["width_p90_pc"] = float(np.percentile(ours_width[pc], 90))
            m["rounding_p90_pc"] = float(np.percentile(ours_round[pc], 90))

        # Confidence calibration (reported, not scored): confident hops should
        # be the accurate ones. Uses the per-hop debug confidence, not the
        # sparse keyframe track.
        good, bad = l1 & (err_f1 < 75), l1 & (err_f1 > 150)
        m["conf_on_accurate"] = float(np.mean(conf[good])) if good.sum() >= 4 else float("nan")
        m["conf_on_inaccurate"] = float(np.mean(conf[bad])) if bad.sum() >= 4 else float("nan")

    # Events and rates.
    m["closures"] = sum(e.kind == LipsyncEventKind.CLOSURE for e in events)
    m["nasals"] = sum(e.kind == LipsyncEventKind.NASAL for e in events)
    m["silences"] = sum(e.kind == LipsyncEventKind.SILENCE for e in events)
    # Duration-based rate is comparable to live-bot observations; the
    # speech-only variant shows density during actual speech.
    speech_secs = float((ref.intensity_db > np.nanmax(ref.intensity_db) - 25).sum()) * 0.02
    m["keyframe_rate"] = (
        len(keyframes) / clip.duration_secs if clip.duration_secs > 0.2 else float("nan")
    )
    m["keyframe_rate_speech"] = len(keyframes) / speech_secs if speech_secs > 0.2 else float("nan")
    m["convergence_s"] = _convergence_secs(debug)
    m["mean_confidence"] = float(np.mean(conf))
    return m


def _smoothstep(value: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def oracle_stats(ref: Reference, offsets: np.ndarray) -> dict[str, float]:
    """What a Praat-perfect tracker would score on the vowel-probe p90 checks.

    Openness/width oracles are the per-clip-normalized reference tracks; the
    rounding oracle applies the analyzer's own rounding formula (in normalized
    terms) to those tracks. Stats cover the post-convergence window, matching
    the checks. Used by ``--calibrate-expectations`` to set corpus bars —
    never computed from our analyzer's output.
    """
    mask = ref.voiced & np.isfinite(ref.f1) & np.isfinite(ref.f2) & (offsets >= POST_CONV_SECS)
    if mask.sum() < 8:
        return {}
    ref_open = _norm_track(ref.f1, mask)
    ref_width = _norm_track(ref.f2, mask)
    # Analyzer rounding in normalized terms: (f2_mid - F2)/(f2_mid - f2_lo)
    # == (0.5 - width)/0.5, gated by the openness window.
    window = _smoothstep(ref_open, 0.05, 0.15) * (1.0 - _smoothstep(ref_open, 0.75, 0.9))
    ref_round = np.clip((0.5 - ref_width) / 0.5, 0.0, 1.0) * window
    return {
        "openness_p90": float(np.nanpercentile(ref_open[mask], 90)),
        "width_p90": float(np.nanpercentile(ref_width[mask], 90)),
        "rounding_p90": float(np.nanpercentile(ref_round[mask], 90)),
    }


def run_checks(sentence: Sentence, metrics: dict[str, float]) -> dict[str, bool]:
    results = {}
    for key, value in sentence.expect.items():
        metric_name, predicate, _ = CHECKS[key]
        metric = metrics.get(metric_name)
        results[key] = bool(metric is not None and np.isfinite(metric) and predicate(metric, value))
    return results


#
# Composite score
#


def _ramp(value: float, full: float, zero: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    t = (value - zero) / (full - zero)
    return float(np.clip(t, 0.0, 1.0) * 100.0)


def _component_values(results: list[ClipResult]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("f1_mae_hz", "f2_mae_hz", "openness_r", "width_r", "convergence_s"):
        vals = [r.metrics.get(name, float("nan")) for r in results]
        values[name] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
    for bucket in ("closure", "nasal", "silence"):
        outcomes = [ok for r in results for key, ok in r.checks.items() if CHECKS[key][2] == bucket]
        values[f"checks_{bucket}"] = (sum(outcomes) / len(outcomes)) if outcomes else float("nan")
    return values


def composite_score(results: list[ClipResult]) -> tuple[float, dict[str, float]]:
    values = _component_values(results)
    total_weight = 0.0
    total = 0.0
    scores: dict[str, float] = {}
    for name, (weight, full, zero) in COMPOSITE.items():
        score = _ramp(values.get(name, float("nan")), full, zero)
        scores[name] = score
        if np.isfinite(score):
            total += weight * score
            total_weight += weight
    return (total / total_weight if total_weight else float("nan")), scores


#
# Reporting
#


def _fmt(value, digits=2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _aggregate(results: list[ClipResult], name: str) -> tuple[float, float]:
    vals = np.array([r.metrics.get(name, float("nan")) for r in results])
    if np.all(np.isnan(vals)):
        return float("nan"), float("nan")
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def report(results: list[ClipResult], run_meta: dict, baseline: dict | None):
    composite, comp_scores = composite_score(results)
    base_line = ""
    if baseline:
        delta = composite - baseline["composite"]
        base_line = f"  (baseline {baseline['composite']:.1f}, {delta:+.1f})"
    print(
        f"\nACCURACY  {run_meta['provider']} · {len(set(r.voice_id for r in results))} voice(s) · "
        f"{len(set(r.sentence_id for r in results))} sentences · {run_meta['takes']} take(s)"
        f"{' · warm' if run_meta['warm'] else ''}"
        f"\ncomposite {composite:.1f}{base_line}\n"
    )

    rows = []
    metric_names = [
        "f1_mae_hz",
        "f2_mae_hz",
        "f1_coverage",
        "f2_coverage",
        "f1_slot_coverage",
        "f2_slot_coverage",
        "f1_r",
        "voicing_agreement",
        "openness_r",
        "openness_r_best",
        "openness_lag_ms",
        "openness_mae",
        "openness_jitter",
        "width_r",
        "width_r_best",
        "width_lag_ms",
        "energy_r",
        "nasal_fraction",
        "convergence_s",
        "keyframe_rate",
        "keyframe_rate_speech",
        "mean_confidence",
        "conf_on_accurate",
        "conf_on_inaccurate",
    ]
    for name in metric_names:
        mean, std = _aggregate(results, name)
        pc_mean, _ = _aggregate(results, f"{name}_pc")
        base = ""
        if baseline and name in baseline["aggregate"]:
            b = baseline["aggregate"][name]
            base = _fmt(mean - b) if np.isfinite(mean) and b is not None else ""
        rows.append((name, f"{_fmt(mean)} ± {_fmt(std)}", _fmt(pc_mean), base))

    check_totals: dict[str, list[int]] = {}
    for r in results:
        for key, ok in r.checks.items():
            bucket = CHECKS[key][2] or "other"
            check_totals.setdefault(bucket, [0, 0])
            check_totals[bucket][0] += ok
            check_totals[bucket][1] += 1
    for bucket, (ok, n) in sorted(check_totals.items()):
        rows.append((f"checks: {bucket}", f"{ok}/{n}", "", ""))

    # Per-clip L1: MAE next to the number of hops it was computed over
    # (committed / Praat-voiced), so a coverage change is never read as an
    # accuracy change. Baseline values in parentheses when comparing.
    base_clips = {c["label"]: c["metrics"] for c in baseline["clips"]} if baseline else {}

    def l1_cell(r: ClipResult, slot: str) -> str:
        n, ref_n = int(r.metrics[f"{slot}_scored_n"]), int(r.metrics["ref_voiced_n"])
        cell = f"{_fmt(r.metrics[f'{slot}_mae_hz'], 0)} Hz · {n}/{ref_n}"
        base = base_clips.get(r.clip_label)
        if base and base.get(f"{slot}_scored_n") is not None:
            cell += f"  ({_fmt(base.get(f'{slot}_mae_hz'), 0)} · {int(base[f'{slot}_scored_n'])})"
        return cell

    clip_cols = (
        "clip",
        "f1_mae · committed/voiced",
        "f1_cov",
        "f2_mae · committed/voiced",
        "f2_cov",
    )
    clip_rows = [
        (
            r.clip_label,
            l1_cell(r, "f1"),
            _fmt(r.metrics["f1_coverage"]),
            l1_cell(r, "f2"),
            _fmt(r.metrics["f2_coverage"]),
        )
        for r in results
    ]

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        for cols, table_rows in (
            (("metric", "mean ± std", "post-conv", "Δ baseline"), rows),
            (clip_cols, clip_rows),
        ):
            table = Table(show_header=True, header_style="bold")
            for col in cols:
                table.add_column(col)
            for row in table_rows:
                table.add_row(*row)
            console.print(table)
    except ImportError:
        for row in rows:
            print(f"  {row[0]:<24} {row[1]:>16} {row[2]:>10} {row[3]:>10}")
        for row in clip_rows:
            print(f"  {row[0]:<30} {row[1]:>30} {row[2]:>6} {row[3]:>30} {row[4]:>6}")

    scored = sorted(results, key=lambda r: composite_score([r])[0])
    worst = ", ".join(f"{r.clip_label} ({composite_score([r])[0]:.1f})" for r in scored[:3])
    print(f"\nworst clips: {worst}")
    failed = [f"{r.clip_label}:{key}" for r in results for key, ok in r.checks.items() if not ok]
    if failed:
        print(f"failed checks: {', '.join(failed)}")


#
# Run orchestration
#

# Modules whose constants ``--set`` may override for A/B runs.
_OVERRIDE_MODULES = {"dsp": dsp, "analyzer": formant_lipsync_analyzer}


def apply_overrides(specs: list[str]) -> dict[str, object]:
    """Apply ``module.CONSTANT=value`` overrides before any analyzer is built.

    Equivalent to editing the constant in the source: the lipsync modules read
    their tunables at call time (nothing binds them at import). Unknown names
    fail loudly so a typo cannot silently A/B nothing.
    """
    applied: dict[str, object] = {}
    for spec in specs:
        target, _, raw = spec.partition("=")
        module_name, _, name = target.partition(".")
        module = _OVERRIDE_MODULES.get(module_name)
        if module is None or not raw or not hasattr(module, name):
            raise SystemExit(
                f"bad --set {spec!r}: expected {{dsp,analyzer}}.EXISTING_CONSTANT=value"
            )
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                value = float(raw)  # "inf", "nan"
            except ValueError:
                value = raw
        setattr(module, name, value)
        applied[target] = value
    return applied


def _src_sha() -> str:
    # Lipsync code lives in this repo (formerly the pipecat fork at src/).
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


async def run(args) -> dict:
    sentences, voice_map = load_corpus()
    if args.sentences:
        wanted = set(args.sentences.split(","))
        missing = wanted - {s.id for s in sentences}
        if missing:
            raise SystemExit(f"unknown sentence ids: {missing}")
        sentences = [s for s in sentences if s.id in wanted]
    voices = voice_map.get(args.provider, [])
    if args.voices:
        voices = [Voice(args.provider, vid, 5500) for vid in args.voices.split(",")]
    if not voices:
        raise SystemExit(f"no voices configured for provider {args.provider}")
    if args.ceiling:
        voices = [Voice(v.provider, v.id, args.ceiling) for v in voices]

    results: list[ClipResult] = []
    for voice in voices:
        warm_pcm, warm_rate = None, 0
        if args.warm:
            warm_clip = await get_clip(
                next(s for s in sentences if s.id.startswith("harvard")),
                voice,
                1,
                offline=args.offline,
                refresh=False,
            )
            warm_pcm, warm_rate = warm_clip.pcm, warm_clip.sample_rate
        for sentence in sentences:
            for take in range(1, args.takes + 1):
                clip = await get_clip(
                    sentence, voice, take, offline=args.offline, refresh=args.refresh
                )
                keyframes, events, debug = await analyze_clip(clip, warm_pcm, warm_rate)
                if not debug:
                    print(f"  ! no analysis frames for {clip.label}; skipping")
                    continue
                offsets = np.array([d.offset for d in debug])
                ref = compute_reference(clip, offsets, voice.formant_ceiling)
                metrics = compute_metrics(clip, keyframes, events, debug, ref)
                results.append(
                    ClipResult(
                        clip_label=clip.label,
                        sentence_id=sentence.id,
                        voice_id=voice.id,
                        take=take,
                        metrics=metrics,
                        checks=run_checks(sentence, metrics),
                    )
                )

    composite, comp_scores = composite_score(results)
    aggregate = {}
    for r in results:
        for name in r.metrics:
            aggregate.setdefault(name, None)
    aggregate = {name: _aggregate(results, name)[0] for name in aggregate}

    return {
        "run": {
            "timestamp": datetime.now(UTC).isoformat(),
            "src_sha": _src_sha(),
            "provider": args.provider,
            "voices": [v.id for v in voices],
            "takes": args.takes,
            "warm": args.warm,
            "sentences": [s.id for s in sentences],
            "overrides": getattr(args, "overrides", {}),
            "ceiling_override": args.ceiling,
        },
        "composite": composite,
        "component_scores": comp_scores,
        "aggregate": {k: (v if np.isfinite(v) else None) for k, v in aggregate.items()},
        "clips": [
            {
                "label": r.clip_label,
                "sentence": r.sentence_id,
                "voice": r.voice_id,
                "take": r.take,
                "metrics": {k: (v if np.isfinite(v) else None) for k, v in r.metrics.items()},
                "checks": r.checks,
            }
            for r in results
        ],
    }, results


# Corpus bars = ORACLE_DISCOUNT x the Praat-oracle value (min over takes) —
# one global constant, applied uniformly; never fitted per check.
ORACLE_DISCOUNT = 0.8


async def calibrate(args):
    """Print oracle p90 stats per clip and suggested corpus bars."""
    sentences, voice_map = load_corpus()
    if args.sentences:
        sentences = [s for s in sentences if s.id in set(args.sentences.split(","))]
    voices = voice_map.get(args.provider, [])
    print(f"oracle p90 stats (suggested bar = {ORACLE_DISCOUNT} × min over takes)\n")
    for voice in voices:
        per_sentence: dict[str, list[dict[str, float]]] = {}
        for sentence in sentences:
            for take in range(1, args.takes + 1):
                clip = await get_clip(sentence, voice, take, offline=args.offline, refresh=False)
                _, _, debug = await analyze_clip(clip)
                if not debug:
                    continue
                offsets = np.array([d.offset for d in debug])
                ref = compute_reference(clip, offsets, voice.formant_ceiling)
                stats = oracle_stats(ref, offsets)
                if stats:
                    per_sentence.setdefault(sentence.id, []).append(stats)
                    row = "  ".join(f"{k}={v:.3f}" for k, v in stats.items())
                    print(f"  {clip.label:28} {row}")
        print()
        for sentence_id, stats_list in per_sentence.items():
            bars = {
                k: round(ORACLE_DISCOUNT * min(s[k] for s in stats_list), 2) for k in stats_list[0]
            }
            print(f"  {sentence_id:12} suggested bars: {bars}")


def main():
    parser = argparse.ArgumentParser(description="Lipsync accuracy benchmark")
    parser.add_argument("--provider", default="cartesia", choices=["cartesia", "deepgram"])
    parser.add_argument("--voices", help="comma-separated voice ids (overrides corpus)")
    parser.add_argument("--sentences", help="comma-separated sentence ids to run")
    parser.add_argument("--takes", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="never synthesize; cache only")
    parser.add_argument("--refresh", action="store_true", help="re-synthesize fixtures")
    parser.add_argument("--warm", action="store_true", help="pre-converge analyzer per voice")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument(
        "--compare",
        nargs="?",
        const="",
        default=None,
        help="baseline JSON to diff against (default: this provider's saved baseline)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="MODULE.CONST=VALUE",
        help="override a lipsync tunable for this run, e.g. dsp.LPC_ORDER=16 "
        "(modules: dsp, analyzer); recorded in the results JSON",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        help="override the Praat formant ceiling for every voice (reference sensitivity check)",
    )
    parser.add_argument("--tag", help="results file name suffix (accuracy-<tag>.json)")
    parser.add_argument(
        "--calibrate-expectations",
        action="store_true",
        help="print Praat-oracle p90 stats per clip (for setting corpus bars) and exit",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    args.overrides = apply_overrides(args.set)
    if args.overrides:
        print("overrides: " + ", ".join(f"{k}={v}" for k, v in args.overrides.items()))

    if args.calibrate_expectations:
        asyncio.run(calibrate(args))
        return

    # Not asyncio.run: closing the loop would wait on detached TTS teardowns
    # (see common.synthesize), which can take minutes after a synthesis run.
    loop = asyncio.new_event_loop()
    payload, results = loop.run_until_complete(run(args))

    baseline = None
    if args.compare is not None:
        compare_path = Path(args.compare) if args.compare else baseline_path(args.provider)
        if compare_path.exists():
            baseline = json.loads(compare_path.read_text())
        else:
            print(f"no baseline at {compare_path}; reporting without comparison")

    report(results, payload["run"], baseline)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"accuracy-{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nresults: {out_path.relative_to(Path.cwd())}")
    if args.save_baseline:
        save_path = baseline_path(args.provider)
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"baseline: {save_path.relative_to(Path.cwd())}")

    if teardowns_pending():
        sys.stdout.flush()
        os._exit(0)
    loop.close()


if __name__ == "__main__":
    main()
