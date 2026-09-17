"""Is "order 16 wins" a property of the tracker or of the Praat reference?

Order 16 at 16 kHz has exactly the pole density of Praat's Burg at ceiling
5000 (10 poles over 5 kHz = 2 per kHz), so agreement with that reference could
be by construction. This scores every LPC order against Praat at several
ceilings, on (a) each order's own committed hops and (b) the hops every order
committed (matched frames, so coverage cannot move the error), with medians
next to means (MAE is outlier-dominated).

Run from server/:  uv run python ../plans/experiments/order_vs_ceiling.py
Needs the cached fixtures (benchmarks/fixtures); no API keys.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from benchmarks.accuracy import analyze_clip, compute_reference  # noqa: E402
from benchmarks.common import get_clip, load_corpus  # noqa: E402
from lipsync import dsp  # noqa: E402

ORDERS = (12, 14, 16, 18)
CEILINGS = (4500.0, 5000.0, 5500.0)


async def main():
    sentences, voice_map = load_corpus()
    voice = voice_map["cartesia"][0]
    clips = [
        await get_clip(s, voice, take, offline=True, refresh=False)
        for s in sentences
        for take in (1, 2)
    ]

    # tracks[order][clip_index] = (offsets, f1, f2, voiced)
    tracks: dict[int, list] = {}
    for order in ORDERS:
        dsp.LPC_ORDER = order
        tracks[order] = []
        for clip in clips:
            _, _, debug = await analyze_clip(clip)
            tracks[order].append(
                (
                    np.array([d.offset for d in debug]),
                    np.array([d.f1 for d in debug]),
                    np.array([d.f2 for d in debug]),
                    np.array([d.voiced for d in debug]),
                )
            )

    for ceiling in CEILINGS:
        refs = [
            compute_reference(clip, tracks[ORDERS[0]][i][0], ceiling)
            for i, clip in enumerate(clips)
        ]
        print(f"\n=== Praat ceiling {ceiling:.0f} Hz (pooled over {len(clips)} clips)")
        print(
            f"{'order':>5} | {'F1 own: n':>9} {'mean':>6} {'med':>5} | {'F1 matched: n':>13} {'mean':>6} {'med':>5}"
            f" | {'F2 own: n':>9} {'mean':>6} {'med':>5} | {'F2 matched: n':>13} {'mean':>6} {'med':>5}"
        )
        for slot, name in ((1, "f1"), (2, "f2")):
            pass
        rows = {order: [] for order in ORDERS}
        for slot in (1, 2):
            own_err = {order: [] for order in ORDERS}
            matched_err = {order: [] for order in ORDERS}
            for i, ref in enumerate(refs):
                ref_track = ref.f1 if slot == 1 else ref.f2
                base = ref.voiced & np.isfinite(ref_track)
                masks = {}
                for order in ORDERS:
                    _, f1, f2, voiced = tracks[order][i]
                    ours = f1 if slot == 1 else f2
                    masks[order] = base & voiced & (ours > 0)
                    own_err[order].append(np.abs(ours - ref_track)[masks[order]])
                common = np.logical_and.reduce([masks[order] for order in ORDERS])
                for order in ORDERS:
                    ours = tracks[order][i][slot]
                    matched_err[order].append(np.abs(ours - ref_track)[common])
            for order in ORDERS:
                own = np.concatenate(own_err[order])
                matched = np.concatenate(matched_err[order])
                rows[order] += [
                    own.size,
                    own.mean(),
                    np.median(own),
                    matched.size,
                    matched.mean(),
                    np.median(matched),
                ]
        for order in ORDERS:
            r = rows[order]
            print(
                f"{order:>5} | {r[0]:>9d} {r[1]:>6.1f} {r[2]:>5.1f} | {r[3]:>13d} {r[4]:>6.1f} {r[5]:>5.1f}"
                f" | {r[6]:>9d} {r[7]:>6.1f} {r[8]:>5.1f} | {r[9]:>13d} {r[10]:>6.1f} {r[11]:>5.1f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
