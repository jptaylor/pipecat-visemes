"""Matched-frame A/B of two analyzer configurations against Praat.

The harness's f1/f2_mae only score hops where the analyzer committed a slot,
so a variant that commits on more (harder) hops can read a *higher* MAE while
being strictly better. This pools |error| over the corpus on: A's own hops,
B's own hops, the hops both committed (matched), and the hops only one side
committed — mean and median for each.

Run from server/:
  uv run python ../plans/experiments/ab_matched.py \
      --a dsp.PITCH_METHOD=residual --b dsp.PITCH_METHOD=ncc
Overrides use the harness's --set syntax; several per side are allowed.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from benchmarks.accuracy import analyze_clip, apply_overrides, compute_reference  # noqa: E402
from benchmarks.common import get_clip, load_corpus  # noqa: E402


def _stats(err: np.ndarray) -> str:
    if err.size == 0:
        return f"{0:>5d} {'—':>7} {'—':>7}"
    return f"{err.size:>5d} {err.mean():>7.1f} {np.median(err):>7.1f}"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", action="append", default=[])
    parser.add_argument("--b", action="append", default=[])
    parser.add_argument("--ceiling", type=float)
    args = parser.parse_args()

    sentences, voice_map = load_corpus()
    voice = voice_map["cartesia"][0]
    ceiling = args.ceiling or voice.formant_ceiling
    clips = [
        await get_clip(s, voice, take, offline=True, refresh=False)
        for s in sentences
        for take in (1, 2)
    ]

    tracks = {}
    for side, overrides in (("a", args.a), ("b", args.b)):
        # Later sides only see their own overrides if they name the same
        # constants; pass every constant explicitly on both sides.
        apply_overrides(overrides)
        tracks[side] = [(await analyze_clip(clip))[2] for clip in clips]

    print(f"A: {args.a}\nB: {args.b}\nPraat ceiling {ceiling:.0f} Hz, {len(clips)} clips\n")
    print(f"{'':14} {'n':>5} {'mean':>7} {'median':>7}")
    for slot in ("f1", "f2"):
        pools = {k: [] for k in ("A own", "B own", "A matched", "B matched", "A only", "B only")}
        for i, clip in enumerate(clips):
            offsets = np.array([d.offset for d in tracks["a"][i]])
            ref = compute_reference(clip, offsets, ceiling)
            ref_track = getattr(ref, slot)
            base = ref.voiced & np.isfinite(ref_track)
            err, mask = {}, {}
            for side in ("a", "b"):
                ours = np.array([getattr(d, slot) for d in tracks[side][i]])
                voiced = np.array([d.voiced for d in tracks[side][i]])
                mask[side] = base & voiced & (ours > 0)
                err[side] = np.abs(ours - ref_track)
            both = mask["a"] & mask["b"]
            pools["A own"].append(err["a"][mask["a"]])
            pools["B own"].append(err["b"][mask["b"]])
            pools["A matched"].append(err["a"][both])
            pools["B matched"].append(err["b"][both])
            pools["A only"].append(err["a"][mask["a"] & ~mask["b"]])
            pools["B only"].append(err["b"][mask["b"] & ~mask["a"]])
        print(f"--- {slot.upper()} |error| Hz")
        for name, chunks in pools.items():
            print(f"{name:14} {_stats(np.concatenate(chunks))}")


if __name__ == "__main__":
    asyncio.run(main())
