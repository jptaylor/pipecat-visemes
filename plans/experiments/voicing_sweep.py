"""Tune the NCC voicing threshold and energy gate against Praat voicing.

Runs the analyzer once per clip with the NCC threshold and gate disabled,
records per-hop NCC clarity and RMS from the debug tap, then sweeps
(threshold, gate) offline. Reports agreement, precision and recall pooled over
all hops of the corpus (agreement alone is flat and hides the trade-off).

Run from server/:  uv run python ../plans/experiments/voicing_sweep.py [--pitch-frame=640]
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from benchmarks.accuracy import analyze_clip, compute_reference  # noqa: E402
from benchmarks.common import get_clip, load_corpus  # noqa: E402
from lipsync import dsp  # noqa: E402
from lipsync import formant_lipsync_analyzer as fla  # noqa: E402

THRESHOLDS = (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
GATES = (0.0, 0.02, 0.03, 0.05, 0.08)


async def main():
    dsp.PITCH_METHOD = "ncc"
    dsp.NCC_VOICED_THRESHOLD = 0.0
    fla._VOICED_MIN_PEAK_FRACTION = 0.0
    for arg in sys.argv[1:]:
        if arg.startswith("--pitch-frame="):
            dsp.PITCH_FRAME_SIZE = int(arg.split("=")[1])

    sentences, voice_map = load_corpus()
    voice = voice_map["cartesia"][0]
    clarity, rms, peak, ref_voiced = [], [], [], []
    for sentence in sentences:
        for take in (1, 2):
            clip = await get_clip(sentence, voice, take, offline=True, refresh=False)
            _, _, debug = await analyze_clip(clip)
            offsets = np.array([d.offset for d in debug])
            ref = compute_reference(clip, offsets, voice.formant_ceiling)
            recent = 0.0
            for d in debug:
                recent = max(recent * fla._PEAK_DECAY, d.rms)
                peak.append(recent)
            clarity += [d.clarity for d in debug]
            rms += [d.rms for d in debug]
            ref_voiced += list(ref.voiced)
    clarity, rms, peak = np.array(clarity), np.array(rms), np.array(peak)
    ref_voiced = np.array(ref_voiced)

    print(f"{len(ref_voiced)} hops, Praat voiced fraction {ref_voiced.mean():.2f}")
    print(f"{'gate':>5} {'thr':>5} | {'agree':>6} {'prec':>6} {'recall':>6} {'ours voiced':>11}")
    for gate in GATES:
        for thr in THRESHOLDS:
            ours = (clarity > thr) & (rms > gate * peak)
            tp = (ours & ref_voiced).sum()
            print(
                f"{gate:>5.2f} {thr:>5.2f} | {(ours == ref_voiced).mean():>6.3f} "
                f"{tp / max(ours.sum(), 1):>6.3f} {tp / ref_voiced.sum():>6.3f} {ours.mean():>11.2f}"
            )
        print()


if __name__ == "__main__":
    asyncio.run(main())
