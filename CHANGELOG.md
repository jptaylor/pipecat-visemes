# Changelog

Notable changes to pipecat-visemes. Earlier history is in the git log, and the design notes
that lived in `plans/` until 2026-10-14 remain in it (`git show 9c85db4:plans/README.md`).

## 2026-10-14

### Added

- `runtimes/typescript` (`@pipecat-visemes/typescript`): the web runtime as its own package,
  with the wire-format parser, `LipsyncFeed` and `lipsyncCallbacks(feed)` for `PipecatClient`.
  The example client is built on it.
- `runtimes/swift` (`PipecatVisemes`): a Swift package for on-device lipsync on iOS 17 and
  macOS 14. It holds a port of the formant analyzer with its resampler and DSP, and a
  timeline that samples the pose on the audio player's clock. Moved from
  gradium-phonon-demo.
- `LipsyncProcessor(text_events=True)` turns on text-informed events in one argument, and
  `from lipsync import LipsyncProcessor, LipsyncMessageRelay` imports both processors.

### Changed

- Updated to `pipecat-ai` 1.12.0 (from 1.10.0). No code changes were needed, and the
  analyzer's benchmark output is unchanged.
- The client's `src/lipsync/protocol.ts` and `feed.ts` moved to `runtimes/typescript/src/`.

### Removed

- `plans/`: the specification, tuning and review notes, and experiment scripts. They remain
  in the git history.
