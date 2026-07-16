# pipecat-visemes

Working repo for building a **server-side lipsync / viseme processor** for
[Pipecat](https://github.com/pipecat-ai/pipecat), to be contributed upstream as a PR.
The processor (`LipsyncProcessor`) sits between any `TTSService` and the output
transport, analyzes streamed TTS audio, and delivers playout-timed articulation
keyframes to clients as RTVI `bot-tts-lipsync` messages — no provider-specific
requirements.

Full design: [plans/technical-specification.md](plans/technical-specification.md)
Implementation guide: [plans/pipecat-implementation.md](plans/pipecat-implementation.md) — milestones, guardrails, per-module specs, unit tests, bot.py integration
Benchmarks: [plans/benchmark-harness-accuracy.md](plans/benchmark-harness-accuracy.md) (signal accuracy vs Praat + corpus expectations) · [plans/benchmark-harness-performance.md](plans/benchmark-harness-performance.md) (TTS→RTVI latency, CPU/RSS budgets)
Accuracy tuning: [plans/accuracy-improvements.md](plans/accuracy-improvements.md) (pass 1, complete: 28.8 → 70.1) · [plans/accuracy-improvements-2.md](plans/accuracy-improvements-2.md) (pass 2: oracle-calibrated corpus, peak compression, hum onset)

## Layout

| Path | Purpose |
| --- | --- |
| `plans/` | Technical specification |
| `src/` | Clone of pipecat `main`, on branch `feat/lipsync-processor`, where the feature is developed |
| `bot.py` | Official `pipecat init quickstart` starter bot, used to test the processor end-to-end |
| `pyproject.toml` | uv project; installs `pipecat-ai` **editable from `./src`** so changes there apply immediately |

## Current status

Implemented (milestones M1–M4 of the implementation plan), with 33 unit tests green
and zero regressions against pipecat's suite:

- `src/src/pipecat/audio/lipsync/` — types, vendored DSP (LPC/Levinson, formants, pitch, P² quantiles), analyzer base class, formant (Tier 0) analyzer with adaptive normalization, conditioning, closure/nasal/silence events, and a `collect_debug` tap for benchmarks
- `src/src/pipecat/processors/audio/lipsync_processor.py` — `LipsyncProcessor` + `LipsyncParams`: passthrough frame path, background analysis task, sample-accurate offsets, playout-timed `pts` batching, interruption handling, `stats` counters
- `src/src/pipecat/frames/frames.py` — `TTSLipsyncFrame`, `LipsyncUpdateSettingsFrame`
- `src/src/pipecat/processors/frameworks/rtvi/` — `bot-tts-lipsync` message (quantized positional arrays), `bot_lipsync_enabled` observer param, transport-gated dispatch
- `src/tests/test_lipsync_dsp.py`, `src/tests/test_lipsync_processor.py` — DSP, analyzer, processor and observer coverage
- `bot.py` — lipsync wired in (processor between TTS and transport output, observer param enabled)

Remaining before PR: live end-to-end run with API keys, foundational example, towncrier changelog fragment (needs PR number). Benchmark harnesses (see plans) not yet built.

## Setup

```bash
uv sync
cp .env.example .env   # add DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY
uv run bot.py          # then open http://localhost:7860
```

## Tests

```bash
uv run pytest src/tests/test_lipsync_processor.py -v
```

## Accuracy benchmark

```bash
uv run python -m benchmarks.accuracy                      # first run synthesizes fixtures (needs keys)
uv run python -m benchmarks.accuracy --offline --compare  # free re-score vs baseline while tuning
```

Scores the analyzer against Praat reference tracks (L1 formant Hz, L2 trajectory shape)
and corpus expectations (L3 events) — see [plans/benchmark-harness-accuracy.md](plans/benchmark-harness-accuracy.md).
Baseline: `benchmarks/results/baseline.json`.
