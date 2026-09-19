# Text-input checkpoint — 2026-09-19

Branch: `codex/text-informed-events`, starting at `8ea78f0` (`main`). This is the
input and measurement checkpoint of [the text-informed plan](text-informed-events.md),
not the CMUdict/event implementation. The formant analyzer and wire format are unchanged.
Text observation is opt-in; no quality improvement is claimed yet.

## Accuracy and regression

Reused the local fixtures, with no provider calls: 12 sentences × 2 takes ×
7 Cartesia voices, plus the Deepgram voice. The original checkout reproduced
the committed baselines before work began. After the changes, **every per-clip
metric and check matched that original run**, not just the rounded composite.

| Corpus | Original DSP | Branch DSP | Text observation | Checks | Exact output matches, off vs on |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cartesia, 7 voices | 86.0425 | 86.0425 | 86.0425 | 204/224 | 168/168 |
| Deepgram, 1 voice | 91.9202 | 91.9202 | 91.9202 | 30/32 | 24/24 |

Exact matches use SHA-256 over full-precision keyframes and events before wire
rounding. Source fingerprint for the branch runs:
`268d6a38fc35a998c70811f0e2f0351dadb47113317d981d6106f1c003798342`.
The source hash includes uncommitted/new Python files, so it identifies these
runs even though the checkpoint had not yet been committed when they ran.

Local result files (gitignored under `server/benchmarks/results/`):

- `accuracy-text-start-before-{cartesia,deepgram}.json`: original checkout.
- `accuracy-text-input-dsp-{cartesia,deepgram}.json`: branch, observation off.
- `accuracy-text-input-observe-{cartesia,deepgram}.json`: branch, untimed corpus text attached.

Use the commands in the plan to regenerate named A/B results. No saved DSP
baseline was replaced. The accuracy corpus does not retain word arrivals;
this test exercises optional inputs and regression, not phone alignment.

## Delivery replay

Replayed all 19 examples of `cartesia-71a7ad14-20260919-123250` with the retained
audio, arrival schedule and word PTS. The control used a temporary, clean
checkout of `main` at `8ea78f0` with the same Python environment. These runs
are selectable in the existing Eval tab:

| Run | Code / input | Median start lag | Longest release delay | Batches | Keyframes |
| --- | --- | ---: | ---: | ---: | ---: |
| `text-input-main` | Clean `main` | −4.3 ms | 18.9 ms | 354 | 2025 |
| `text-input-dsp` | Branch, text off | −4.8 ms | 29.5 ms | 355 | 2026 |
| `text-input-observe` | Branch, text observation | −4.8 ms | 21.6 ms | 355 | 2026 |

All three emitted **142 closures, 82 nasals and 12 silences**. Observation
collected 19 anchors and 225 words, with no text or audio dropped. This is a
legacy recording: its sentence anchors were explicitly supplied using
`--assume-early-text`, and the run records all 19 affected examples. It does
not establish the provider's real sentence/word-clock relationship.

These are individual real-time replays, not evidence of a timing improvement.
Per-clip keyframes vary by up to two across runs; resampler/task chunking and
idle-flush scheduling already vary between replays (see the plan of record's
measurement caveats). The deterministic accuracy driver above is the exact
output regression gate. No extra analysis lookahead was introduced.

## Tests

`uv run --offline python -m pytest -q`: **70 passed**, 35 subtests. Before this
checkpoint: 55 passed, 29 subtests. `ruff check lipsync benchmarks tests` passes.

New coverage includes early/late anchors, `TTSTextFrame` subclass discrimination,
zero PTS, non-streaming completion sentences, multiple sentences per context,
queued/reused contexts, bounded orphan and oversized text, interruption/cancel/end,
runtime disabling, immutable snapshots, unchanged forwarded frames and output,
record/replay of text, explicit legacy assumptions, and A/B corpus compatibility.

## Timing finding and next work

The installed Pipecat 1.10.0 initializes its word clock on first audio, whereas
the sentence announcement may have been timestamped before synthesis. Those
timestamps share a clock but are not guaranteed to share an origin. The input
contract preserves their raw PTS and samples-at-observation, and the recorder
retains actual arrival times. The plan's claim that anchor PTS directly reveals
the word origin was corrected against the installed implementation.

Next: validate conversion to context audio time on captured frame streams,
then implement the vendored CMUdict prior and opt-in stage-1 event decisions.
Viterbi and a learned scorer remain conditional on measured stage-1 results.
