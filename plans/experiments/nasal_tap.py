"""Nasal detector features from the debug tap: hum takes vs everything else.

The nasal-hum takes are (almost) all nasal murmur; harvard-03 ("It's easy to
tell the depth of a well") contains no nasal at all, so its NASAL activity is
pure false positive. Prints, per clip: NASAL events, the fraction of our-voiced
hops with the nasal override active, and how the F2-damping evidence breaks
down (no F2 / broad F2 / spurious high F2 on a dark frame) on hops that pass
the spectral gates (centroid, low-band ratio).

Run from server/:  uv run python ../plans/experiments/nasal_tap.py [--set analyzer._X=v ...]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from benchmarks.accuracy import analyze_clip, apply_overrides  # noqa: E402
from benchmarks.common import get_clip, load_corpus  # noqa: E402
from lipsync import formant_lipsync_analyzer as fla  # noqa: E402
from lipsync.types import LipsyncEventKind  # noqa: E402

# True nasal phones per sentence (counted by hand from the corpus text).
TRUE_NASALS = {
    "vowel-aa": 2, "vowel-ee": 2, "vowel-oo": 6, "bilabial-mama": 6, "bilabial-bob": 1,
    "nasal-hum": 3, "pause-probe": 5, "harvard-01": 3, "harvard-02": 1, "harvard-03": 0,
    "harvard-04": 1, "harvard-05": 3,
}  # fmt: skip


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--hist", action="store_true", help="F2 bandwidth/frequency quantiles")
    args = parser.parse_args()
    apply_overrides(args.set)

    sentences, voice_map = load_corpus()
    voice = voice_map["cartesia"][0]
    print(
        f"{'clip':18} {'true':>4} {'events':>6} {'active':>7} | gated hops: "
        f"{'n':>4} {'noF2':>5} {'broad':>5} {'spur':>5} {'clean':>5}"
    )
    pools = {"hum": [], "other": []}
    for sentence in sentences:
        for take in (1, 2):
            clip = await get_clip(sentence, voice, take, offline=True, refresh=False)
            _, events, debug = await analyze_clip(clip)
            voiced = [d for d in debug if d.voiced]
            gated = [
                d
                for d in voiced
                if d.centroid < fla._NASAL_CENTROID_MAX_HZ
                and d.low_band_ratio > fla._NASAL_LOW_RATIO_MIN
            ]
            no_f2 = sum(d.f2 == 0 for d in gated)
            broad = sum(d.f2 > 0 and d.f2_bandwidth > fla._NASAL_F2_MAX_BANDWIDTH_HZ for d in gated)
            spurious = sum(
                d.f2 > 0
                and d.f2_bandwidth <= fla._NASAL_F2_MAX_BANDWIDTH_HZ
                and d.low_band_ratio > fla._NASAL_DARK_RATIO
                and d.f2 > fla._NASAL_SPURIOUS_F2_HZ
                for d in gated
            )
            n_events = sum(e.kind == LipsyncEventKind.NASAL for e in events)
            active = np.mean([d.nasal_active for d in voiced]) if voiced else 0.0
            print(
                f"{sentence.id + '/t' + str(take):18} {TRUE_NASALS[sentence.id]:>4} {n_events:>6} "
                f"{active:>7.2f} | {'':11} {len(gated):>4} {no_f2:>5} {broad:>5} {spurious:>5} "
                f"{len(gated) - no_f2 - broad - spurious:>5}"
            )
            pools["hum" if sentence.id == "nasal-hum" else "other"] += gated

    if args.hist:
        for name, hops in pools.items():
            with_f2 = [d for d in hops if d.f2 > 0]
            print(f"\n{name}: {len(hops)} gated voiced hops, {len(with_f2)} with an F2 root")
            for label, values in (
                ("f2_bandwidth", [d.f2_bandwidth for d in with_f2]),
                ("f2", [d.f2 for d in with_f2]),
                ("low_band_ratio", [d.low_band_ratio for d in hops]),
                ("centroid", [d.centroid for d in hops]),
            ):
                q = np.percentile(values, [10, 25, 50, 75, 90])
                print(f"  {label:15} p10/25/50/75/90: " + "  ".join(f"{v:8.2f}" for v in q))


if __name__ == "__main__":
    asyncio.run(main())
