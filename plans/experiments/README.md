# Review experiments (2026-09-17)

Prototype scripts behind the numbers in `plans/deep-review-2026-09.md`. They are
**not** part of the benchmark harness and are not maintained with the code; they exist
so the measurements can be re-run.

Requirements: the server venv (`cd server && uv sync --all-groups`) and, for the
espeak scripts, `libespeak-ng` on the system (`apt-get install espeak-ng`; or
`pip install espeakng-loader` and point `espeak_synth.py` at its library).

| Script | What it measures | Run |
|---|---|---|
| `espeak_synth.py` | ctypes binding to libespeak-ng that returns int16 audio plus IPA phoneme events with audio positions (exact phoneme boundaries) | `uv run --directory server python ../plans/experiments/espeak_synth.py` |
| `l4_proto.py` | "L4" phoneme-target layer: synthesizes 17 sentences × 4 espeak voices, maps IPA phones to (class, openness, width, rounding) targets, runs `FormantLipsyncAnalyzer`, and scores openness/width/rounding MAE and correlation, lag, closure precision/recall against p/b/m onsets (±60 ms), openness during bilabials, nasal-closed fraction, and a per-class mean-shape table | `uv run --directory server python ../plans/experiments/l4_proto.py -v` |
| `pacing_sim.py` | Emission lead of each lipsync batch relative to its audio playout (`(t0 + window_start) − emit time`) under burst, 10×, 2× and 1× real-time TTS delivery, through a real `Pipeline` with the processor | `uv run --directory server python ../plans/experiments/pacing_sim.py` |

espeak-ng is a formant synthesizer: its numbers are a proxy for detector logic, timing and
mapping behaviour, never for neural-TTS spectra. Every proposal in the review that rests on
these scripts is marked "must validate on real fixtures".
