# pipecat-visemes

Server-side lipsync for [Pipecat](https://github.com/pipecat-ai/pipecat) voice
bots: the server analyzes streamed TTS audio and sends playout-timed
articulation keyframes to the client, which renders an animated mouth — no
audio analysis in the browser, no provider timestamp APIs.

Goals:

- **Provider-agnostic** — works with any `TTSService`; analysis runs on the PCM
  stream itself, with no reliance on provider word/phoneme timestamps, viseme
  events, or other side-channel metadata.
- **No fork** — built on released `pipecat-ai` (~1.10.0) using only stock
  extension points (frame processors, RTVI `server-message`); nothing in the
  framework is subclassed or patched.
- **Playout-accurate** — keyframes ride the output transport's clock, staying
  in sync with audio and discarded on interruption.

Implementation: `LipsyncProcessor` sits between the TTS service and the output
transport, running a formant-based analyzer over the audio and emitting
keyframe batches. `LipsyncMessageRelay`, placed after `transport.output()`,
delivers each batch to clients as a standard RTVI `server-message` with
`data.type: "bot-tts-lipsync"`.

The analyzer is numpy-only DSP at 16 kHz on a 20 ms hop: order-16 LPC
formants (F1 → openness, F2 → width/rounding) with adaptive per-voice
normalization, normalized-cross-correlation voicing on a 40 ms frame, and
energy/spectral event detectors for closures, nasals and silence. Keyframes
are conditioned by a zero-phase median and a slew limit, so the emitted track
sits within ~10 ms of a Praat reference; the whole thing costs ~190 µs per
hop (about 1 % of one core per bot).

## Layout

| Path                 | Purpose                                                                                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `server/lipsync/`    | The lipsync package: types, vendored DSP (LPC/Levinson, formants, pitch, P² quantiles), formant (Tier 0) analyzer, `LipsyncProcessor`, app-local frames, RTVI server-message relay |
| `server/bot.py`      | Official `pipecat init quickstart` starter bot with the lipsync processor + relay wired in                                                                                         |
| `server/tests/`      | Unit tests (DSP, analyzer, processor, relay, end-to-end through a headless output transport)                                                                                       |
| `server/benchmarks/` | Accuracy harness (Praat reference + designed corpus, two provider voices, committed baselines)                                                                                    |
| `client/`            | Vite + React web client: connects over SmallWebRTC, parses lipsync server-messages, renders an animated mouth with timing/event inspectors                                         |
| `plans/`             | Technical specification, tuning/review notes, and the experiment scripts behind them                                                                                               |

## How delivery works

`LipsyncProcessor` assigns each keyframe batch a `pts`, so the output
transport's clock queue releases it at presentation time (~200 ms ahead of the
matching audio), discarding unplayed batches on interruption. The relay sits
immediately after `transport.output()`, so every batch it sees is already
playout-timed; it wraps the batch in an `RTVIServerMessageFrame`, which the
stock `RTVIObserver` forwards to the client as a standard `server-message`.
Clients subscribe with the SDK's `onServerMessage` callback and demux on
`data.type === "bot-tts-lipsync"` (see `client/src/lipsync/protocol.ts`).

Timing follows the audio, not the arrival of frames:

- A context's first sample is anchored at `max(now, end of already-queued
  audio)`, so an utterance queued behind another one is scheduled where that
  audio ends.
- If the transport runs out of a context's audio before more arrives (the LLM
  stalled mid-response; pipecat ≥ 1.8 keeps one TTS context per turn), later
  batches are shifted by the gap and never straddle it. The shift travels on
  the wire as `t0`, which the client adds to that batch's offsets.
- A TTS service that reopens a context id it already closed (the same id after
  its idle timeout) starts a new segment with offsets from zero; the client
  re-anchors when offsets regress within one `ctx`.

## Setup

```bash
cd server
uv sync                # add --all-groups for the tests and the benchmark
cp .env.example .env   # add DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY
uv run bot.py          # bot + SmallWebRTC on http://localhost:7860

npm --prefix client install
npm --prefix client run dev   # viseme client on http://localhost:5173
```

## Tests

```bash
cd server && uv run pytest    # 46 tests; ruff check . / ruff format . for lint
```

## Accuracy benchmark

```bash
cd server
uv run python -m benchmarks.accuracy                      # first run synthesizes fixtures (CARTESIA_API_KEY)
uv run python -m benchmarks.accuracy --offline --compare  # free re-score vs the committed baseline
uv run python -m benchmarks.accuracy --provider deepgram --offline --compare   # second voice (DEEPGRAM_API_KEY once)
```

Scores the analyzer against Praat reference tracks (L1 formant Hz with coverage, L2
trajectory shape and timing lag) and corpus expectations (L3 events and duty-cycle guards)
— see [plans/benchmark-harness-accuracy.md](plans/benchmark-harness-accuracy.md).
Baselines are committed (`server/benchmarks/results/baseline.json` for the Cartesia voice,
composite 87.5; `baseline-deepgram.json`, 90.7) so `--compare` reads +0.0 on a clean
checkout once the fixtures exist; fixtures themselves are not committed. While tuning,
`--set dsp.LPC_ORDER=14` overrides any `dsp`/`analyzer` constant for one run and `--tag`
names the results file; `plans/experiments/ab_table.py` runs whole A/B ladders.

## Further development

A deep review of the whole project (verified findings, a benchmark revision and a
ranked roadmap) is in [plans/deep-review-2026-09.md](plans/deep-review-2026-09.md);
its §8 roadmap supersedes the tier list below. Its DSP findings were validated on
real fixtures in [plans/deep-review-2026-09-results.md](plans/deep-review-2026-09-results.md).

The formant analyzer is Tier 0 of a planned analyzer ladder; all tiers emit the
same keyframe/event wire format, so clients are unaffected by tier choice.

- **Tier 1 — provider visemes:** consume Azure viseme events / Polly speech
  marks via a TTS service hook; highest confidence where providers support it.
- **Tier 2 — timestamp + G2P:** derive phonemes from word/char timestamps
  (ElevenLabs, Cartesia) with grapheme-to-phoneme lookup.
- **Tier 3 — phoneme model:** ONNX CTC phoneme recognition over the audio
  stream (`onnxruntime` optional extra).
- **Performance benchmark:** TTS→RTVI latency and CPU/RSS budget harness
  (designed, not yet built).

Nearest open items from the validation notes: the NASAL event also fires on
dark voiced consonants (it needs a place cue or a broader name before clients
style it), keyframe economy (28–31/s), and energy-gated closures.

## License

BSD 2-Clause, the same license as [Pipecat](https://github.com/pipecat-ai/pipecat).
See [LICENSE](LICENSE).
