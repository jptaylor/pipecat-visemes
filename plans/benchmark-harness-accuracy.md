# Benchmark Harness — Accuracy

**Status:** Ready for implementation (after implementation plan M1+M2)
**Location:** Root project (`benchmarks/`), never upstream — the pipecat PR stays free of harness code.
**Goal:** A lightweight, repeatable score of how accurately the lipsync implementation turns
TTS audio into articulation signals, across multiple voices and multiple synthesis takes,
against a corpus of sentences with known expectations. The score's job is to point at
*where* the implementation needs work (DSP? normalization? events?) and to show whether a
change moved accuracy up or down.
**Companions:** [technical-specification.md](technical-specification.md) §11.1–§11.2 (this is the lightweight local stand-in for the heavier CI corpus), [pipecat-implementation.md](pipecat-implementation.md).

## 1. Non-goals

- No Montreal Forced Aligner / phoneme-boundary ground truth (spec §11.2's heavy path).
  MFA needs conda/kaldi tooling — the opposite of lightweight. See §3 for what replaces it.
- No perceptual studies, no client rendering, no CI gating (upstream concerns, later).
- No timing/latency measurement — that's [benchmark-harness-performance.md](benchmark-harness-performance.md).
- Not a pipeline test: the analyzer is driven directly, offline (unit tests already cover
  processor plumbing; the performance harness covers the live pipeline).

## 2. Prerequisites

- Implementation plan **M1 (dsp) + M2 (analyzer)** complete. M3/M4 not required — this
  harness exists precisely so DSP/analyzer tuning can start before the processor lands.
- **Debug feature tap** on `FormantLipsyncAnalyzer` (see §9 "Required seam" — a small,
  dev-only addition now recorded in the implementation plan): per-hop raw features
  (F1/F2/F3 Hz, pitch, voiced, rms, confidence components, effective normalization ranges).
  Without it we can't score formants in Hz — keyframes only carry normalized 0..1 values.
- Root dev deps (root `pyproject.toml` `[dependency-groups].dev` — pipecat's own
  pyproject stays untouched, consistent with the no-new-deps decision):
  - `praat-parselmouth` — reference formant/pitch/intensity tracks (spec §10 names it as
    the dev/test reference tracker).
  - `pyyaml` — corpus file (already installed transitively via pipecat; declared because
    the harness imports it directly).
- `.env` with `CARTESIA_API_KEY` (and optionally `DEEPGRAM_API_KEY` — `DeepgramTTSService`
  exists in `services/deepgram/tts.py`, so a second provider costs zero new extras).

## 3. Ground-truth strategy (the core design decision)

There is no free source of "true mouth shapes" for arbitrary TTS audio. The harness layers
three cheap, complementary truths instead:

| Layer | Truth source | What it scores | Why it's trustworthy |
|---|---|---|---|
| **L1 — DSP** | Praat (via parselmouth) formant tracks on the *same* audio | Our per-hop F1/F2 in Hz (MAE, correlation) | Praat's Burg tracker is the de-facto phonetics reference; disagreement in Hz localizes bugs to `dsp.py` |
| **L2 — Trajectory** | Praat F1/F2 normalized per-clip → reference openness/width | Shape of our normalized openness/width over time (Pearson r, MAE) | Scale-invariant comparison; disagreement *here but not L1* localizes bugs to normalization/conditioning |
| **L3 — Expectations** | The corpus itself: sentences *designed* so we know what must happen | Events (closure/nasal/silence counts), qualitative bounds (e.g. "hum keeps mouth closed") | Categorical, human-verifiable by reading the sentence; catches exactly the spec's worst-case failures |

The layering is diagnostic by construction: L1 bad → fix `dsp.py`; L1 good + L2 bad → fix
adaptive normalization or conditioning; L1+L2 good + L3 bad → fix event state machines or
the nasal override.

## 4. File layout (root project)

```
benchmarks/
  __init__.py
  common.py          # corpus loading, fixture cache, TTS synthesis, WAV I/O   (~180 LOC)
  accuracy.py        # analyzer driver, reference tracks, metrics, score, CLI  (~420 LOC)
  corpus.yaml        # sentences + expectations + default voices
  fixtures/          # cached TTS audio (gitignored)
  results/           # JSON run results (gitignored)
```

Two Python modules total. `common.py` is shared with the performance harness. Add
`benchmarks/fixtures/` and `benchmarks/results/` to the root `.gitignore`.

## 5. Corpus (`corpus.yaml`)

Schema:

```yaml
voices:
  cartesia:
    - id: "71a7ad14-091c-4e8e-a314-022ece01c121"   # quickstart default voice
      formant_ceiling: 5500                        # Praat ceiling: 5500 female / 5000 male
    # add more voice ids here; --voices overrides
  deepgram:
    - id: "aura-2-asteria-en"
      formant_ceiling: 5500

sentences:
  - id: vowel-aa
    text: "Father was calm as he walked past the palm trees."
    tags: [vowel-probe]
    expect: { openness_mean_voiced_min: 0.45 }
  - id: vowel-ee
    text: "See the green trees? Please believe me, these seeds are free."
    tags: [vowel-probe]
    expect: { width_mean_voiced_min: 0.5 }
  - id: vowel-oo
    text: "Soon the new moon grew blue, and the room felt cool."
    tags: [vowel-probe]
    expect: { rounding_mean_voiced_min: 0.35, width_mean_voiced_max: 0.5 }
  - id: bilabial-mama
    text: "Mama made more mashed potatoes for my mother."
    tags: [events]
    expect: { closures_min: 4, closures_max: 14 }
  - id: bilabial-bob
    text: "Bob put a big pepper in the paper bag."
    tags: [events]
    expect: { closures_min: 5, closures_max: 14 }
  - id: nasal-hum
    text: "Hmm. Hmm, hmm."
    tags: [events, nasal-guard]
    expect: { nasals_min: 2, openness_p90_max: 0.30 }
  - id: pause-probe
    text: "Wait for it. Now continue talking normally."
    tags: [events]
    expect: { silences_min: 1 }
  # Phonetically balanced sentences (Harvard set 1, public domain) — L1/L2 scoring only:
  - id: harvard-01
    text: "The birch canoe slid on the smooth planks."
  - id: harvard-02
    text: "Glue the sheet to the dark blue background."
  - id: harvard-03
    text: "It's easy to tell the depth of a well."
  - id: harvard-04
    text: "These days a chicken leg is a rare dish."
  - id: harvard-05
    text: "Rice is often served in round bowls."
```

Design notes:

- **Vowel probes use real words** (father/see/soon), not sustained "aaah" — TTS pronounces
  real words reliably; nonsense strings synthesize unpredictably.
- **Event expectations are deliberately loose ranges.** The energy-dip closure detector
  fires on *all* stop consonants (p/b/t/d/k/g), not only bilabials, and TTS coarticulation
  merges some closures. Exact counts would be brittle; ranges catch gross failures
  ("no closures detected in a plosive-heavy sentence") without punishing coarticulation.
- `expect` keys form a tiny closed DSL (see §8.3); anything unspecified simply isn't
  checked for that sentence. L1/L2 always run on every sentence.
- Praat's `formant_ceiling` matters per voice (5000 Hz for typical male ranges, 5500 for
  female); it rides with the voice entry so reference tracks stay honest across voices.

## 6. Fixture pipeline (`common.py`)

### 6.1 Synthesis — a headless mini-pipeline per (provider, voice, sentence, take)

Reuses pipecat's real TTS services (same code path as a bot, exercising real streaming
chunk sizes), driven by `TTSSpeakFrame` — no transport, no LLM:

```python
class _AudioCollector(FrameProcessor):
    """Collects TTSAudioRawFrames (+ start/stop) for one synthesis run."""
    # on TTSAudioRawFrame: append (frame.audio, frame.sample_rate, frame.context_id)
    # on TTSStoppedFrame: set an asyncio.Event

async def synthesize(provider: str, voice_id: str, text: str) -> Clip:
    tts = _make_tts(provider, voice_id)            # factory: cartesia | deepgram
    collector = _AudioCollector()
    pipeline = Pipeline([tts, collector])
    worker = PipelineWorker(pipeline, params=PipelineParams(), enable_rtvi=False)
    runner = WorkerRunner(handle_sigint=False)
    await worker.queue_frames([TTSSpeakFrame(text), EndFrame()])
    await runner.add_workers(worker)
    await runner.run()
    return collector.clip                          # pcm bytes + sample_rate
```

(`_make_tts` mirrors the quickstart constructor for Cartesia —
`CartesiaTTSService(api_key=..., settings=CartesiaTTSService.Settings(voice=voice_id))` —
and the Deepgram equivalent. Exact worker teardown details settled at implementation; the
collector + `TTSSpeakFrame` + `EndFrame` shape is the design.)

### 6.2 Cache layout & takes

```
benchmarks/fixtures/{provider}/{voice_id}/{sentence_id}/take-{n}.wav   # 16-bit PCM, native rate
benchmarks/fixtures/{provider}/{voice_id}/{sentence_id}/take-{n}.json  # {text, sample_rate, created_at, provider}
```

- WAV I/O via stdlib `wave` (no new dependency).
- `--takes N` (default 2): TTS is non-deterministic, so each take is a distinct synthesis;
  the score reports mean ± std across takes — variance itself is signal (a metric that
  swings wildly across takes is not measuring the implementation).
- Cache-first: a (provider, voice, sentence, take) tuple already on disk is never
  re-synthesized unless `--refresh`. `--offline` errors instead of synthesizing (pure
  re-scoring runs are free, fast, and API-key-independent — the common case while tuning).
- Sidecar JSON guards staleness: if `corpus.yaml` text changed for a sentence id, the
  cached fixture is invalid → warn and (unless `--offline`) re-synthesize.

## 7. Reference tracks (`accuracy.py`, parselmouth)

Per clip, computed once and cached in memory for the run:

```python
snd = parselmouth.Sound(str(wav_path))
formants = snd.to_formant_burg(time_step=0.01, maximum_formant=ceiling)   # 25 ms window default
pitch = snd.to_pitch(time_step=0.01)
# Sampled AT OUR HOP TIMES for alignment (our offsets: (k·320 + 200) / 16000):
f1_ref[t] = formants.get_value_at_time(1, t);  f2_ref[t] = formants.get_value_at_time(2, t)
voiced_ref[t] = pitch.get_value_at_time(t) is not NaN
```

- **Alignment rule:** the reference is always sampled at *our* hop offsets — no
  interpolation of our sparse data onto Praat's grid.
- **Scoring mask:** a hop participates in L1 iff `voiced_ref AND our voiced AND f1_ref is
  finite`. Voicing *agreement* is scored separately (it would otherwise double-penalize).
- **Reference trajectory for L2:** `openness_ref = clamp01((f1_ref − P5) / (P95 − P5))`
  with P5/P95 over the clip's masked frames; same for `width_ref` from F2. Per-clip
  normalization makes L2 scale-invariant — it scores *shape*, deliberately ignoring
  whether our adaptive ranges match Praat's absolute ranges (L1 covers absolute).

## 8. Driving the analyzer & metrics

### 8.1 Offline analyzer driver

Mirrors the processor's ingest contract exactly (analyzer receives 16 kHz float32; the
resample step is the harness's copy of the processor's ingest, per the implementation
plan's §5.2 decision):

```python
async def analyze_clip(clip: Clip, *, warm_pcm: np.ndarray | None = None) -> AnalyzerOutput:
    analyzer = FormantLipsyncAnalyzer(collect_debug=True)      # the §9 seam
    await analyzer.start(clip.sample_rate)
    ctx = LipsyncAnalysisContext(context_id="bench", sample_rate=clip.sample_rate)
    if warm_pcm is not None:                                   # --warm: pre-converge, unscored
        await feed(analyzer, warm_pcm, ctx_warm); await analyzer.flush(ctx_warm)
    keyframes, events = [], []
    for chunk in chunks_16k_float32(clip, chunk_ms=20):        # resample + convert like processor
        r = await analyzer.analyze(chunk, ctx)
        keyframes += r.keyframes; events += r.events
    r = await analyzer.flush(ctx); keyframes += r.keyframes; events += r.events
    return AnalyzerOutput(keyframes, events, analyzer.debug_features)
```

- Fresh analyzer per clip (default, "cold") measures convergence honestly every clip;
  `--warm` feeds ~3 s of the same voice first (excluded from scoring) to measure
  steady-state accuracy. Both modes matter: cold ↔ first-utterance UX, warm ↔ long-session UX.
- Fixed 20 ms chunking (chunk-boundary invariance is already a unit test; not re-proven here).

### 8.2 Metric definitions

Dense reconstruction for L2: linearly interpolate emitted keyframes onto the 10 ms grid
(clients interpolate too; linear is the neutral choice). All metrics computed twice:
**full-clip** and **post-convergence** (t ≥ 1.5 s) — the gap between them isolates
convergence cost.

| Metric | Layer | Definition |
|---|---|---|
| `f1_mae_hz`, `f2_mae_hz` | L1 | mean abs error vs Praat, masked hops (debug-tap raw Hz) |
| `f1_r`, `f2_r` | L1 | Pearson r of raw Hz tracks, masked hops |
| `voicing_agreement` | L1 | fraction of hops where our voiced flag == Praat's |
| `openness_r`, `openness_mae` | L2 | reconstructed openness vs `openness_ref`, masked hops |
| `width_r`, `width_mae` | L2 | same for width |
| `energy_r` | L2 | reconstructed energy vs Praat intensity (dB, normalized) — sanity only |
| `event_satisfaction` | L3 | fraction of `expect` checks satisfied (per sentence → mean) |
| `convergence_s` | — | first t where both effective ranges (from debug tap) stay within 10% of their final values (spec §11.2 metric) |
| `keyframe_rate` | — | emitted keyframes / speech-seconds (sanity: 4–25 expected) |
| `confidence_calibration` | — | mean confidence on hops where L1 error is low vs high (high-error hops should carry lower confidence; reported, not scored, v1) |

### 8.3 Expectation DSL (L3) — the full v1 set

`closures_min/max`, `nasals_min/max`, `silences_min/max` (event counts);
`openness_p90_max`, `openness_mean_voiced_min`, `width_mean_voiced_min/max`,
`rounding_mean_voiced_min` (distribution bounds on the reconstructed tracks over voiced
hops). Each check is a ~3-line function in a registry dict; unknown keys in the corpus fail
loudly at load time.

### 8.4 Composite score

Per clip → per (voice, sentence) mean over takes → aggregate. Each component maps to
0–100 via a linear ramp between a "full marks" and a "zero marks" threshold, then a
weighted sum:

| Component | Weight | 100 at | 0 at |
|---|---|---|---|
| F1 accuracy (`f1_mae_hz`) | 20 | ≤ 50 Hz | ≥ 200 Hz |
| F2 accuracy (`f2_mae_hz`) | 20 | ≤ 80 Hz | ≥ 300 Hz |
| Openness shape (`openness_r`) | 15 | ≥ 0.85 | ≤ 0.0 |
| Width shape (`width_r`) | 15 | ≥ 0.85 | ≤ 0.0 |
| Closure events | 10 | all satisfied | none |
| Nasal guard | 10 | all satisfied | none |
| Silence events | 5 | all satisfied | none |
| Convergence (`convergence_s`) | 5 | ≤ 3 s | ≥ 8 s |

**The thresholds are provisional** — calibrated against nothing yet. The rule: freeze them
after the first run whose output has been eyeballed (plots or spot-checked numbers), then
never tune thresholds and implementation in the same change. The absolute number matters
less than the delta between runs; `--compare` (§10) is the primary consumption mode.
Weights/thresholds live as constants at the top of `accuracy.py` with this warning attached.

## 9. Required seam in pipecat (recorded in the implementation plan)

`FormantLipsyncAnalyzer(collect_debug: bool = False)` → when set, appends one
`LipsyncDebugFrame` per hop to `analyzer.debug_features`:

```python
@dataclass
class LipsyncDebugFrame:      # dev-only; lives in formant_lipsync_analyzer.py
    offset: float             # hop center, seconds
    f1: float; f2: float; f3: float          # raw Hz (0.0 = not found)
    pitch_hz: float; voiced: bool
    rms: float; centroid: float; low_band_ratio: float
    c_lpc: float; c_conv: float; c_snr: float
    f1_lo: float; f1_hi: float; f2_lo: float; f2_hi: float   # effective ranges this hop
```

Cost when disabled: one `if` per hop. Never enabled by the processor; harness/test-only.
This replaces the alternative (re-implementing hop framing inside the harness), keeping a
single source of truth for framing/windowing.

## 10. CLI & workflow

```bash
# First run: synthesize fixtures (needs API keys), score, save results
uv run python -m benchmarks.accuracy --provider cartesia --takes 2

# Tuning loop (free, offline, seconds):
uv run python -m benchmarks.accuracy --offline --save-baseline        # pin current state
# ...change dsp.py / analyzer...
uv run python -m benchmarks.accuracy --offline --compare              # delta vs baseline

# More voices / providers / modes
uv run python -m benchmarks.accuracy --voices <id1>,<id2> --takes 3
uv run python -m benchmarks.accuracy --provider deepgram
uv run python -m benchmarks.accuracy --offline --warm
uv run python -m benchmarks.accuracy --offline --sentences nasal-hum,bilabial-mama   # focus
```

Flags: `--provider` `--voices` `--sentences` `--takes` `--offline` `--refresh` `--warm`
`--save-baseline` `--compare [path]` (default `benchmarks/results/baseline.json`).

Console output (rich table — available via the `cli` extra; plain-text fallback):

```
ACCURACY  cartesia · 2 voices · 12 sentences · 2 takes          composite 71.4  (baseline 68.9, +2.5)
metric                 mean ± std      post-conv     baseline Δ
f1_mae_hz              78.2 ± 6.1      61.0          −9.3   ✓
f2_mae_hz             141.5 ± 12.0    120.2          −15.1  ✓
openness_r             0.71 ± 0.05     0.79          +0.04  ✓
width_r                0.62 ± 0.08     0.66          −0.01  ≈
events: closure        9/10 checks     —             +1     ✓
events: nasal          2/2 checks      —             =
convergence_s          2.4 ± 0.3       —             −0.6   ✓
worst clips: harvard-03/voiceB/t2 (54.1), vowel-oo/voiceA/t1 (58.8)
```

JSON result (`benchmarks/results/accuracy-<ts>.json`): run metadata (src git sha, params),
per-clip metrics, aggregates per voice and overall — the schema is whatever the report
table needs, plus per-clip raw metric values so regressions can be traced to specific
clips without re-running.

The "worst clips" line is deliberate: the tuning loop is *look at the worst clip, listen
to its WAV, plot its tracks, fix the cause* — not chase the aggregate.

## 11. Implementation checklist

1. Root `pyproject.toml`: add `praat-parselmouth`, `pyyaml` to `[dependency-groups].dev`;
   `uv sync`. Add `benchmarks/fixtures/`, `benchmarks/results/` to `.gitignore`.
2. `benchmarks/common.py`: corpus loader (+ DSL key validation), WAV cache, `synthesize()`
   mini-pipeline, `chunks_16k_float32()` ingest mirror. Smoke: synthesize one sentence,
   inspect the WAV manually.
3. Debug tap in `FormantLipsyncAnalyzer` (implementation-plan seam) if not already landed.
4. `benchmarks/accuracy.py`: reference tracks → metrics → expectations registry →
   composite → report/JSON → argparse CLI (in that order; each stage testable by running).
5. First full run (2 takes × default voice); eyeball worst clips; freeze thresholds;
   `--save-baseline`.
6. Optional (only if a metric looks suspicious): `--plot clip_id` dumping a PNG of
   ours-vs-Praat tracks via matplotlib *if installed* — guard the import, don't add the dep.

## 12. Decisions log

| Decision | Why |
|---|---|
| Praat/parselmouth as continuous truth, not MFA | pip-installable, zero setup, per-frame reference; MFA is conda/kaldi-heavy and only adds phoneme *labels*, which the designed corpus covers categorically |
| Event scoring by loose count ranges, not aligned boundaries | Alignment needs MFA; count ranges on designed sentences catch every gross failure mode and are human-verifiable by reading the sentence |
| Analyzer driven directly, offline | Deterministic, seconds-fast, isolates the signal model; pipeline behavior is unit-tested and perf-tested elsewhere |
| Fixtures cached, gitignored | Lightweight repo; deterministic re-scoring via cache; committing fixtures is a later upstream-CI decision (spec §11.2), not a harness concern |
| Reference sampled at our hop times | One alignment rule, no resampling of sparse keyframe data |
| Cold analyzer per clip + `--warm` mode | Measures both first-utterance and steady-state accuracy; convergence becomes a first-class metric instead of noise |
| Thresholds provisional, frozen after first eyeballed run | Prevents circular calibration; deltas (`--compare`) are the primary signal. Frozen 2026-07-14 at baseline composite 28.8 |
| Reference = raw Burg tracks (no Praat `Track...`, no stability masking) | Tested both: Track requires ≥N formants on every frame (fails on real clips) and neither materially changed L1 — the disagreement is genuine implementation signal |
| L2 evaluated at our hop centers, not a 10 ms grid | One time base everywhere; reference is always sampled at our offsets anyway |
| stdlib `wave`, argparse; only parselmouth + pyyaml added (root dev group) | "As lightweight as possible", and pipecat's own dependency set stays pristine |
