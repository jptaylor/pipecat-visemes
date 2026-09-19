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
- **Playout-accurate** — batches are scheduled on the pipeline clock just
  ahead of playout, carry their exact lead so the client anchors on facts, and
  are dropped on interruption.

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
| `server/lipsync/`    | The lipsync package: types, vendored DSP (LPC/Levinson, formants, pitch, P² quantiles), formant analyzer, `LipsyncProcessor`, app-local frames, RTVI server-message relay |
| `server/bot.py`      | Official `pipecat init quickstart` starter bot with the lipsync processor + relay wired in                                                                                         |
| `server/tests/`      | Unit tests (DSP, analyzer, processor, relay, end-to-end through a headless output transport)                                                                                       |
| `server/benchmarks/` | Accuracy harness (Praat reference + designed corpus, seven Cartesia voices and a Deepgram voice, committed baselines) and the recorder behind the client's Eval tab           |
| `client/`            | Vite + React web client: a Live tab (SmallWebRTC to the bot, animated mouth with timing/event inspectors) and an Eval tab that replays recorded clips                              |
| `plans/`             | Plan of record ([plans/README.md](plans/README.md)): status, findings, what is and is not planned; the specification, tuning/review notes and experiment scripts behind them        |

## How delivery works

`LipsyncProcessor` emits a batch for each window of analyzed audio (the first
100 ms of an utterance, then 200 ms windows) as soon as analysis is one hop past
the window's end, and schedules it on the pipeline clock for 200 ms before the
window plays. Its delivery task pushes the batch at that time; when the time has
already passed (the first windows of a turn, whose audio has to be analyzed
first), the batch goes out at once. Batches not yet released are dropped on
interruption, in step with the discarded audio. A batch is a system frame, so
the output transport forwards it immediately instead of queueing it: the
transport's clock queue would hold it behind any earlier-queued word-timestamp
frame with a later timestamp (for up to a word's length), and its audio-sync
path would release it only once the audio queued ahead of it had played.

`LipsyncMessageRelay` sits after `transport.output()` and wraps each batch in an
`RTVIServerMessageFrame`, stamped with the window start (`ws`) and the lead
still remaining at that moment (`lead`); the stock `RTVIObserver` forwards it
as a standard `server-message`. Clients subscribe with the SDK's
`onServerMessage` callback, demux on `data.type === "bot-tts-lipsync"`, and
anchor the utterance on `now + lead − ws`: exact but for network transit, with
no assumed lead (see `client/src/lipsync/protocol.ts` and `feed.ts`). Closure
and silence events are confirmed after their window's batch has left and ride
in the next one, keyed by offset, so an event may precede its batch's window.

Timing follows the audio, not the arrival of frames:

- A context's first sample is anchored at `max(now, end of already-queued
  audio)`, so an utterance queued behind another one is scheduled where that
  audio ends.
- If the transport runs out of a context's audio before more arrives (the LLM
  stalled mid-response; pipecat ≥ 1.8 keeps one TTS context per turn), later
  batches are shifted by the gap and never straddle it. The shift travels on
  the wire as `t0`, which the client adds to that batch's offsets and `ws`.
  Keyframes that are already final when audio stops arriving are flushed after
  100 ms rather than held until the window fills.
- A TTS service that reopens a context id it already closed (the same id after
  its idle timeout) starts a new segment with offsets from zero; the client
  re-anchors when `ws` regresses within one `ctx`.
- On barge-in the client cuts the utterance at its playhead (plus a short grace
  for its own audio latency) when the bot-stopped-speaking event arrives, and
  ignores that utterance's batches still in flight.

Measured on the eval corpus (19 lines, live Cartesia at 6–11× real time, the
same audio and keyframes replayed through both paths): the client's anchor
error at utterance start went from +309 ms median (worst +631; it assumed a
200 ms lead the first batch never had) to −3 ms, the settled anchor from
−90 ms to −3 ms, the first batch now leaves as playout starts (was 0.1–0.4 s
after), and the longest a batch left late is 21 ms (was 840 ms, waiting behind
a word-timestamp frame). Details in [plans/README.md](plans/README.md).

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
cd server && uv run pytest    # ruff check . / ruff format . for lint
```

## Accuracy benchmark

```bash
cd server
uv run python -m benchmarks.accuracy                      # first run synthesizes fixtures for seven voices, ~3 min (CARTESIA_API_KEY)
uv run python -m benchmarks.accuracy --offline --compare  # free re-score vs the committed baseline
uv run python -m benchmarks.accuracy --provider deepgram --offline --compare   # second voice (DEEPGRAM_API_KEY once)
```

Scores the analyzer against Praat reference tracks (L1 formant Hz with coverage, L2
trajectory shape and timing lag) and corpus expectations (L3 events and duty-cycle guards)
— see [plans/benchmark-harness-accuracy.md](plans/benchmark-harness-accuracy.md).
Baselines are committed (`server/benchmarks/results/baseline.json`, composite 86.0 pooled over
seven Cartesia voices, 80–94 per voice; `baseline-deepgram.json`, 91.9) so `--compare` reads +0.0 on a clean
checkout once the fixtures exist; fixtures themselves are not committed. While tuning,
`--set dsp.LPC_ORDER=14` overrides any `dsp`/`analyzer` constant for one run and `--tag`
names the results file; `plans/experiments/ab_table.py` runs whole A/B ladders.

Text-informed work is isolated on `codex/text-informed-events`; `main` at `8ea78f0`
is the original comparison point. The branch's first checkpoint adds optional text
observations and A/B provenance, with the existing DSP analyzer and default behavior.
`--text-prior` supplies untimed corpus text to the accuracy driver (observation only for
now); results include full-precision output hashes for exact regression comparisons.
Use `--tag` for experiments and `--compare <results.json>` for matching runs. Commands,
status and the remaining event work are in
[the text-informed plan](plans/text-informed-events.md#current-branch-checkpoint--inputs-and-comparison).

## Eval tab

The client's **Eval** tab plays a fixed corpus of clips (`server/benchmarks/eval_corpus.yaml`:
probes mirroring the accuracy corpus plus bot-style lines) with their recorded audio and
lipsync, to judge how things look without running the bot or talking to it.

```bash
cd server
uv run python -m benchmarks.record                 # record the corpus (~1 min, CARTESIA_API_KEY)
uv run python -m benchmarks.record --reanalyze     # offline: retained audio through the current lipsync code
uv run python -m benchmarks.record --reanalyze --set dsp.LPC_ORDER=14 --tag order14   # A/B a tunable
```

Recording speaks each example in the bot's voice through the bot's output path (TTS →
`LipsyncProcessor` → output transport → `LipsyncMessageRelay`), in real time, with a headless
transport paced like SmallWebRTC's. It keeps the audio as played, every lipsync server-message
with its release and due times, word timings, and the TTS arrival timeline, which
`--reanalyze` replays (audio, sentence anchors and word-timestamp frames) without calling the TTS, adding a run
to the newest recording so changes compare on identical audio. Files land in
`client/public/eval/` (gitignored), served as-is by the Vite dev server.

On the text-informed branch, `--text-prior` enables observation of those frames by the
processor. Old recordings did not capture sentence anchors: `--assume-early-text` explicitly
supplies one from each example's text for those takes and marks the assumption in the run.
New recordings preserve the real anchor arrivals automatically, including late or absent text.

In the tab, **as delivered** hands each batch to the stock feed at its recorded release time,
so it anchors exactly as a connected client would (minus network); **ideal** puts every batch
on time, isolating the analysis. The timeline shows the waveform, the rendered pose, events,
batch arrivals (late ones in red) and words; per-clip stats cover the start lag, the settled
anchor, batches that arrived after their audio began, and how late any batch was released.

## Further development

The aims are fixed: fast (keyframes reach the client ahead of playout), light
on CPU (numpy-only DSP, about 1 % of one core per speaking bot) and drop-in
for any TTS provider (nothing but the PCM stream, on released pipecat). The
formant analyzer is the design of record and is judged close enough. The next
step in fidelity would be a trained model, or provider-specific tiers (viseme
events, timestamp + grapheme-to-phoneme); both are deliberately not planned.
[plans/README.md](plans/README.md) is the plan of record: the status of every
design and tuning note, the 2026-09 findings, and what is open.

Open items, in order:

- **NASAL event** also fires on dark voiced consonants (ð, /w l/, voice bars);
  it needs a place cue or a broader name before clients style it.
- **Keyframe economy** (28–31/s against the 25/s guard) and **energy-gated
  closures** (the mouth should close with the energy dip, not only badge it).
- **Client:** events are shown as badges but do not shape the pose.

## License

BSD 2-Clause, the same license as [Pipecat](https://github.com/pipecat-ai/pipecat).
See [LICENSE](LICENSE).
