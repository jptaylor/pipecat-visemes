"""Where does openness correlation go? Stage-by-stage r against Praat F1.

On Praat-voiced hops, per clip then averaged: Pearson r between the
per-clip-normalized Praat F1 (the harness's openness reference) and
  f1_committed   our raw F1, committed hops only (what L1 sees)
  f1_held        our raw F1 zero-order-held across uncommitted hops
  mapped         the openness target out of the F1 mapping / unvoiced decay
  +nasal         after the nasal override
  conditioned    after median-3 + slew (per hop, before the keyframe gate)
  emitted        the keyframe track the client interpolates (harness openness_r)
For every stage after f1_committed the best lag (5 ms grid, positive = ours
late) and r at that lag are printed too, so the lag can be attributed.

Run from server/:  uv run python ../plans/experiments/openness_stages.py [--provider deepgram] [--set ...]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from benchmarks.accuracy import (  # noqa: E402
    _norm_track,
    _pearson,
    analyze_clip,
    apply_overrides,
    compute_reference,
)
from benchmarks.common import get_clip, load_corpus  # noqa: E402

LAG_GRID = np.arange(-120, 121, 5)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="cartesia")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    apply_overrides(args.set)

    sentences, voice_map = load_corpus()
    voice = voice_map[args.provider][0]
    stage_names = ("f1_committed", "f1_held", "mapped", "+nasal", "conditioned", "emitted")
    stages = {k: [] for k in stage_names}
    lags = {k: [] for k in stage_names}
    r_best = {k: [] for k in stage_names}
    nasal_frac, unvoiced_frac = [], []
    for sentence in sentences:
        for take in (1, 2):
            clip = await get_clip(sentence, voice, take, offline=True, refresh=False)
            keyframes, _, debug = await analyze_clip(clip)
            offs = np.array([d.offset for d in debug])
            ref = compute_reference(clip, offs, voice.formant_ceiling)
            mask = ref.voiced & np.isfinite(ref.f1)
            ref_open = _norm_track(ref.f1, mask)
            f1 = np.array([d.f1 for d in debug])
            committed = mask & (f1 > 0) & np.array([d.voiced for d in debug])
            held = f1.copy()
            for i in range(1, held.size):
                if held[i] == 0:
                    held[i] = held[i - 1]
            kf_offs = np.array([k.offset for k in keyframes])
            kf_open = [k.openness for k in keyframes]
            emitted = np.interp(offs, kf_offs, kf_open)
            stages["f1_committed"].append(_pearson(f1[committed], ref_open[committed]))
            tracks = {"f1_held": held}
            for name, attr in (
                ("mapped", "openness_mapped"),
                ("+nasal", "openness_target"),
                ("conditioned", "openness_conditioned"),
            ):
                tracks[name] = np.array([getattr(d, attr) for d in debug])
            tracks["emitted"] = emitted
            for name, track in tracks.items():
                stages[name].append(_pearson(track[mask], ref_open[mask]))
                # Lag search: sample the per-hop track at t - tau by linear
                # interpolation on the hop grid (keyframes: on their own grid).
                grid_offs, grid_vals = (kf_offs, kf_open) if name == "emitted" else (offs, track)
                r_by_lag = [
                    _pearson(
                        np.interp(offs - tau / 1000.0, grid_offs, grid_vals)[mask], ref_open[mask]
                    )
                    for tau in LAG_GRID
                ]
                best = int(np.nanargmax(r_by_lag))
                lags[name].append(-LAG_GRID[best])
                r_best[name].append(r_by_lag[best])
            nasal_frac.append(np.mean([d.nasal_active for d, m in zip(debug, mask) if m]))
            unvoiced_frac.append(np.mean([not d.voiced for d, m in zip(debug, mask) if m]))
    print(f"{args.provider} {args.set}")
    print("  r at zero lag:  " + "  ".join(f"{k} {np.nanmean(v):.2f}" for k, v in stages.items()))
    print(
        "  best lag (ms):  "
        + "  ".join(f"{k} {np.nanmean(lags[k]):+.0f}" for k in stage_names if k != "f1_committed")
    )
    print(
        "  r at best lag:  "
        + "  ".join(f"{k} {np.nanmean(r_best[k]):.2f}" for k in stage_names if k != "f1_committed")
    )
    print(
        f"  on Praat-voiced hops: nasal override active {np.mean(nasal_frac):.2f}, "
        f"ours unvoiced {np.mean(unvoiced_frac):.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
