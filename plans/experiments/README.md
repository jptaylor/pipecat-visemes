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

## Real-fixture validation (2026-09-18)

Scripts behind `plans/deep-review-2026-09-results.md`. They need the cached benchmark
fixtures (`server/benchmarks/fixtures/`, gitignored) and no API keys; run from `server/`.

| Script | What it measures | Run |
|---|---|---|
| `ab_table.py` | The whole A/B ladder (LPC order, slot prior, LPC window, NCC voicing, nasal rule) through the accuracy harness, every variant stated as explicit `--set` overrides; prints the results table | `uv run python ../plans/experiments/ab_table.py [--provider deepgram] [--ceiling 5500] [--only 0,4b]` |
| `order_vs_ceiling.py` | LPC order 12/14/16/18 scored against Praat at ceilings 4500/5000/5500, on own and matched hops, mean and median — shows the "best" order follows the reference's pole density | `uv run python ../plans/experiments/order_vs_ceiling.py [deepgram]` |
| `ab_matched.py` | Matched-frame A/B of two configurations: error on the hops both committed vs the hops only one did (the f1/f2_mae masking artefact) | `uv run python ../plans/experiments/ab_matched.py --a dsp.PITCH_METHOD=residual --b dsp.PITCH_METHOD=ncc` |
| `voicing_sweep.py` | NCC threshold × energy-gate grid against Praat voicing: agreement, precision, recall | `uv run python ../plans/experiments/voicing_sweep.py [--pitch-frame=400]` |
| `nasal_tap.py` | Per clip: NASAL events vs hand-counted true nasals, nasal-override active fraction, and which F2-damping rule supplied the evidence | `uv run python ../plans/experiments/nasal_tap.py [--hist] [--set …]` |
| `openness_stages.py` | Openness correlation against Praat stage by stage: raw F1 → held → mapped → after nasal override → emitted keyframes | `uv run python ../plans/experiments/openness_stages.py [--provider deepgram] [--set …]` |

espeak-ng is a formant synthesizer: its numbers are a proxy for detector logic, timing and
mapping behaviour, never for neural-TTS spectra. Every proposal in the review that rests on
these scripts is marked "must validate on real fixtures".
