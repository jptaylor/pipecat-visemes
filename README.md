# pipecat-visemes

A standalone example application: **server-side lipsync / viseme analysis** for
[Pipecat](https://github.com/pipecat-ai/pipecat) voice bots, built on released
`pipecat-ai` (no fork). `LipsyncProcessor` sits between any `TTSService` and the
output transport, analyzes streamed TTS audio, and emits playout-timed
articulation keyframes; `LipsyncMessageRelay` (placed after
`transport.output()`) delivers each batch to clients over the stock RTVI
`server-message` channel with `data.type: "bot-tts-lipsync"` — no
provider-specific requirements, no pipecat patches.

Full design: [plans/technical-specification.md](plans/technical-specification.md)
Implementation guide: [plans/pipecat-implementation.md](plans/pipecat-implementation.md) — milestones, guardrails, per-module specs, unit tests, bot.py integration (historical: written for the in-tree pipecat layout; module paths now map to `lipsync/`)
Benchmarks: [plans/benchmark-harness-accuracy.md](plans/benchmark-harness-accuracy.md) (signal accuracy vs Praat + corpus expectations) · [plans/benchmark-harness-performance.md](plans/benchmark-harness-performance.md) (TTS→RTVI latency, CPU/RSS budgets; not yet built)
Accuracy tuning: [plans/accuracy-improvements.md](plans/accuracy-improvements.md) (pass 1, complete: 28.8 → 70.1) · [plans/accuracy-improvements-2.md](plans/accuracy-improvements-2.md) (pass 2, complete: 70.9, oracle-calibrated corpus, peak compression, hum onset)

## Layout

| Path | Purpose |
| --- | --- |
| `server/lipsync/` | The lipsync package: types, vendored DSP (LPC/Levinson, formants, pitch, P² quantiles), formant (Tier 0) analyzer, `LipsyncProcessor`, app-local frames, RTVI server-message relay |
| `server/bot.py` | Official `pipecat init quickstart` starter bot with the lipsync processor + relay wired in |
| `server/tests/` | Unit tests (DSP, analyzer, processor, relay) |
| `server/benchmarks/` | Accuracy harness (Praat reference + designed corpus) |
| `client/` | Vite + React web client: connects over SmallWebRTC, parses lipsync server-messages, renders an animated mouth with timing/event inspectors |
| `plans/` | Technical specification and implementation/tuning docs |

## How delivery works

`LipsyncProcessor` assigns each keyframe batch a `pts`, so the output
transport's clock queue releases it at presentation time (~200 ms ahead of the
matching audio), discarding unplayed batches on interruption. The relay sits
immediately after `transport.output()`, so every batch it sees is already
playout-timed; it wraps the batch in an `RTVIServerMessageFrame`, which the
stock `RTVIObserver` forwards to the client as a standard `server-message`.
Clients subscribe with the SDK's `onServerMessage` callback and demux on
`data.type === "bot-tts-lipsync"` (see `client/src/lipsync/protocol.ts`).

## Setup

```bash
cd server
uv sync
cp .env.example .env   # add DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY
uv run bot.py          # bot + SmallWebRTC on http://localhost:7860

npm --prefix client install
npm --prefix client run dev   # viseme client on http://localhost:5173
```

## Tests

```bash
cd server && uv run pytest
```

## Accuracy benchmark

```bash
cd server
uv run python -m benchmarks.accuracy                      # first run synthesizes fixtures (needs keys)
uv run python -m benchmarks.accuracy --offline --compare  # free re-score vs baseline while tuning
```

Scores the analyzer against Praat reference tracks (L1 formant Hz, L2 trajectory shape)
and corpus expectations (L3 events) — see [plans/benchmark-harness-accuracy.md](plans/benchmark-harness-accuracy.md).
Baseline: `server/benchmarks/results/baseline.json` (composite 70.9).

## History

This started as a pipecat fork (branch `feat/lipsync-processor`) targeting an
upstream PR, then was extracted into this standalone app. The only fork-coupled
pieces — the two frame classes and the RTVI observer dispatch — were
re-expressed as `lipsync/frames.py` and `lipsync/rtvi.py` on public pipecat
API; the analyzer/DSP/processor modules moved verbatim. The fork snapshot is
preserved in `~/Sites/pipecat-lipsync-fork.bundle`. A future upstream
contribution is a mechanical copy-back (module filenames were kept identical).
