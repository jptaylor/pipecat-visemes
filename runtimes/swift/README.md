# PipecatVisemes (Swift)

On-device lipsync for Apple platforms: a Swift port of the server's formant analyzer, for
apps that play TTS audio locally and drive a mouth from it without a server round trip. It
follows `server/lipsync/formant_lipsync_analyzer.py` hop for hop (the default DSP path; the
opt-in text-informed events are not ported) and samples the pose the way the web runtime's
feed does.

| File | Purpose |
| --- | --- |
| `LipsyncTimeline.swift` | Entry point: analyzes audio as it is scheduled for playback and serves the pose at the player's position |
| `FormantLipsyncAnalyzer.swift` | The 16 kHz analyzer: keyframes (openness, width, rounding, energy, pitch) and closure, nasal and silence events |
| `LipsyncDSP.swift` | LPC formants (roots through LAPACK's `dgeev_`), NCC pitch, nasal spectrum, P² quantiles |
| `LipsyncResampler.swift` | 24 kHz to 16 kHz windowed-sinc resampler with no group delay |

```swift
import PipecatVisemes

let timeline = LipsyncTimeline()  // @MainActor, @Observable

// As each chunk of 24 kHz mono audio is scheduled on the player, with the time it
// starts on the player's clock in seconds:
timeline.schedule(samples, at: startTime)

// When playback drains, or is interrupted:
timeline.finish()
timeline.cancel()

// Every display frame, at the audible position (output latency included):
let pose = timeline.pose(at: playbackTime)  // openness, width, rounding in 0...1
```

Detected closures shut the lips with 30 ms ramps, and `isAnimating` turns false once the
mouth has settled, so a view can pause its display link. Analysis runs on a serial
background queue; release builds analyze 30 s of audio in under 0.1 s on Apple silicon.
`FormantLipsyncAnalyzer` and `LipsyncResampler` can also be used directly.

Requires iOS 17 or macOS 14 (Observation). Add it as a local package
(`.package(path: "…/runtimes/swift")`, or Add Local… in Xcode). The target passes
`-Xcc -DACCELERATE_NEW_LAPACK`, which SwiftPM allows for local packages; an app that copies
the sources instead needs the same flag in Other Swift Flags.

Moved from gradium-phonon-demo (`src/ios/PipecatVoice`, `character-rig` at `11b7f82`),
where it drives each character's mouth and its checks against the Python analyzer live.
