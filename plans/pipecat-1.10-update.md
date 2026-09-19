# Update to pipecat-ai 1.10.0 (2026-09-17)

The app stays a standalone example on released `pipecat-ai` (`~=1.10.0`); the
lipsync package is not being upstreamed into the framework. This note records
what changed between the 1.5.0 pin and 1.10.0 that mattered, and what was
changed here in response.

## What changed upstream that affects lipsync

- **One TTS context per turn.** `TTSService(reuse_context_id_within_turn=True)`
  is the default since 1.8: every sentence of an LLM response shares one
  `context_id`, with one `TTSStartedFrame`/`TTSStoppedFrame` pair per turn
  (the stop arrives after `stop_frame_timeout_s`, 3 s of no audio, or at turn
  end). Two consequences:
  - Audio for a context can arrive with gaps (the LLM is slow to produce the
    next sentence). Under the 1.5.0 per-sentence contexts each sentence was
    anchored at its own arrival, which hid this; with per-turn contexts the
    processor must notice when the transport has run dry.
  - If the LLM stalls past the idle timeout, the service closes the context
    and later **reopens the same id**. The old processor ignored a
    `TTSStartedFrame` for a known id and dropped audio for a closed one.
- **`StartFrame.audio_out_sample_rate` is deprecated (1.8).** Processors read
  `FrameProcessorSetup.audio_out_sample_rate` in `setup()` instead.
- Everything else the code touches is unchanged: `create_stream_resampler`,
  `seconds_to_nanoseconds`, `TTSAudioRawFrame.context_id`, the output
  transport's clock queue (`handle_timed_frame` for any non-system frame with
  a `pts`, re-pushed downstream at `pts`, dropped on interruption),
  `RTVIServerMessageFrame` → `server-message`, `pipecat.tests.utils.run_test`.
  The client SDK still routes `server-message` to `onServerMessage`
  (client-js 1.13).

## Changes here

`server/lipsync/lipsync_processor.py`

- Contexts are an ordered list keyed by creation, not a dict by id: a
  `TTSStartedFrame` for an id whose context is closed opens a new segment
  (offsets restart at 0); audio and `TTSStoppedFrame` go to the newest open
  context with that id.
- **Playout anchoring.** A context's first sample is anchored at
  `max(now, playout end of already-ingested audio)` (was: `max(now, last
  emitted batch pts + 1)`), so a queued utterance is scheduled where the
  previous audio ends. Reset on interruption (the transport drops its queue).
- **Playout gaps.** When audio arrives more than 100 ms after the ingested
  audio should have finished playing, the gap is recorded at that offset and
  later windows are shifted by the cumulative gap; batches never straddle a
  gap. `TTSLipsyncFrame.playout_offset` carries the shift and goes on the wire
  as `t0` (previously always 0).
- Sample rate is taken from `setup()`.
- New stat: `playout_gaps`.

`server/lipsync/frames.py`, `rtvi.py`: `playout_offset` field; `t0` is quantized
like everything else.

`client/src/lipsync/feed.ts`: offsets regressing within one `ctx` mean the
server reopened the context, so the feed re-anchors as for a new utterance.
`t0` was already applied by the parser.

`server/bot.py`: matches the current `pipecat init quickstart` shape
(`WorkerRunner(handle_sigint=runner_args.handle_sigint)`, `runner.cancel()` on
disconnect, no explicit `observers=[]`).

Client SDKs bumped to `@pipecat-ai/client-js` ^1.13, `client-react` ^1.8.2,
`small-webrtc-transport` ^1.10.6.

## Verification

- `uv run pytest` in `server/`: 43 tests (new: queued-context anchoring,
  reopened context id, playout gap, `t0` on the wire, processor → headless
  `BaseOutputTransport` → relay end to end).
- Accuracy benchmark `--offline --compare`: composite 70.9, +0.0 versus the
  baseline (analyzer and DSP untouched apart from removing a dead assignment).
- `ruff` (0.15.14) clean; client `tsc -b && vite build` and `oxlint` clean.

## Not done / follow-ups

Tracked in [README.md](README.md) (the idle flush is part of the utterance-start latency item).

- Tail latency during a mid-turn stall: keyframes for the last ~0.6 s before a
  gap are held by the event-finalize horizon until more audio arrives, so the
  mouth eases to rest slightly early; an idle flush timer would fix it.
- The in-framework variant explored on 2026-09-17 (observer branch handling
  `TTSLipsyncFrame` from the output transport, `LipsyncMessageData` model,
  frames in `pipecat.frames.frames`) was set aside; the standalone relay is the
  design of record.
