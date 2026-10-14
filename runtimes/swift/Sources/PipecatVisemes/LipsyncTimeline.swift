//
// Copyright (c) 2026, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

// Pose sampling follows the web runtime's feed (runtimes/typescript/src/feed.ts).
import Foundation
import Observation

/// Analyzes TTS audio as it is scheduled for playback and serves the mouth pose at the
/// playback position. Audio never leaves the device, so keyframes are placed on the
/// player's own clock instead of the network anchoring the web runtime needs.
/// Sampling follows that client's feed: interpolate between keyframes, blend in from
/// rest, then hold and ease back to rest after the last keyframe. Keyframes start at the
/// utterance's first sound: the analyzer begins every utterance from a half-open neutral
/// pose that takes about 0.1 s to decay to rest, and TTS engines often lead a sentence with
/// near-silence (0.1–0.25 s on Phonon), so earlier keyframes would open the mouth in it.
/// Analysis runs off the main actor; chunks are scheduled ahead of the playhead, so
/// keyframes arrive in time.
@MainActor @Observable
public final class LipsyncTimeline {
    public struct Pose: Equatable, Sendable {
        public var openness: Double
        public var width: Double
        public var rounding: Double

        public init(openness: Double, width: Double, rounding: Double) {
            self.openness = openness
            self.width = width
            self.rounding = rounding
        }

        public func blended(to other: Pose, by t: Double) -> Pose {
            Pose(openness: openness + (other.openness - openness) * t,
                 width: width + (other.width - width) * t,
                 rounding: rounding + (other.rounding - rounding) * t)
        }
    }

    /// The analyzer's silence rest (`_REST_POSE` in server/lipsync/formant_lipsync_analyzer.py),
    /// which silence decays toward: lips nearly closed, width and rounding neutral.
    public static let rest = Pose(openness: 0.15, width: 0.35, rounding: 0.35)
    /// The lips start to part this long before an utterance's first keyframe: early enough
    /// to read as anticipation, too short to show an open mouth in silence.
    private static let prerollSeconds = 0.1
    /// An utterance's first sound is its first 10 ms block (at 24 kHz) whose RMS reaches
    /// −40 dBFS. Phonon's leading silence sits near −80 dBFS, with a click near −52 dBFS in
    /// its first 40 ms; speech passes −40 dBFS within a few milliseconds of starting.
    private static let onsetLevel: Float = 0.01
    private static let onsetBlock = 240
    private static let holdSeconds = 0.25
    private static let easeSeconds = 0.3
    /// Lips close fully during a detected closure (M/B/P), easing over this long at each edge.
    private static let closureRampSeconds = 0.03
    private static let closedOpenness = 0.02

    /// Whether the mouth may still move; the view pauses its animation otherwise.
    public private(set) var isAnimating = false

    @ObservationIgnored private let analysis = LipsyncAnalysis()
    /// Identifies the current utterance, so results for a stopped one are dropped.
    @ObservationIgnored private var generation = 0
    /// Where each scheduled chunk starts in utterance audio and on the player's clock (seconds).
    @ObservationIgnored private var segments: [(audio: Double, player: Double)] = []
    @ObservationIgnored private var scheduledAudio = 0.0
    /// Where the utterance's first sound starts in its audio (seconds), once scheduled.
    @ObservationIgnored private var onset: Double?
    @ObservationIgnored private var keyframes: [(time: Double, pose: Pose)] = []
    @ObservationIgnored private var closures: [(start: Double, end: Double)] = []
    @ObservationIgnored private var cursor = 0
    @ObservationIgnored private var inUtterance = false
    @ObservationIgnored private var restTask: Task<Void, Never>?

    public init() {}

    /// Analyze `samples` (24 kHz mono) that will start playing at `playerTime` seconds.
    public func schedule(_ samples: [Float], at playerTime: Double) {
        guard !samples.isEmpty else { return }
        if !inUtterance {
            generation += 1
            clearTrack()
            inUtterance = true
            restTask?.cancel()
            isAnimating = true
        }
        segments.append((scheduledAudio, playerTime))
        if onset == nil, let index = Self.firstSound(in: samples) {
            onset = scheduledAudio + Double(index) / LipsyncResampler.inputRate
        }
        scheduledAudio += Double(samples.count) / LipsyncResampler.inputRate
        analysis.analyze(samples, deliver: recorder())
    }

    /// Playback drained: analyze the tail, then let the mouth settle before pausing.
    public func finish() {
        guard inUtterance else { return }
        analysis.analyze(nil, deliver: recorder())
        inUtterance = false
        let settle = Self.holdSeconds + 5 * Self.easeSeconds
        restTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(settle))
            guard !Task.isCancelled, let self, !self.inUtterance else { return }
            self.isAnimating = false
        }
    }

    /// Playback was interrupted: drop the utterance and rest at once.
    public func cancel() {
        generation += 1
        analysis.reset()
        inUtterance = false
        restTask?.cancel()
        clearTrack()
        isAnimating = false
    }

    /// The pose at `playerTime` seconds on the player's clock, or rest without playback.
    public func pose(at playerTime: Double?) -> Pose {
        guard isAnimating, let time = playerTime, let first = keyframes.first else { return Self.rest }
        var pose: Pose
        if time < first.time {
            let until = first.time - time
            if until > Self.prerollSeconds { return Self.rest }
            pose = Self.rest.blended(to: first.pose, by: 1 - until / Self.prerollSeconds)
        } else {
            while cursor + 1 < keyframes.count && keyframes[cursor + 1].time <= time { cursor += 1 }
            while cursor > 0 && keyframes[cursor].time > time { cursor -= 1 }
            let current = keyframes[cursor]
            if cursor + 1 < keyframes.count {
                let next = keyframes[cursor + 1]
                let span = next.time - current.time
                pose = current.pose.blended(to: next.pose, by: span > 0 ? min(1, (time - current.time) / span) : 1)
            } else {
                let over = time - current.time
                pose = over <= Self.holdSeconds ? current.pose
                    : current.pose.blended(to: Self.rest, by: 1 - exp(-(over - Self.holdSeconds) / Self.easeSeconds))
            }
        }
        for closure in closures where closure.start - Self.closureRampSeconds < time && time < closure.end + Self.closureRampSeconds {
            let entering = (time - (closure.start - Self.closureRampSeconds)) / Self.closureRampSeconds
            let leaving = (closure.end + Self.closureRampSeconds - time) / Self.closureRampSeconds
            let amount = min(1, entering, leaving)
            pose.openness += (min(pose.openness, Self.closedOpenness) - pose.openness) * amount
        }
        return pose
    }

    private func recorder() -> @MainActor @Sendable (FormantLipsyncAnalyzer.Result) -> Void {
        let generation = generation
        return { [weak self] result in
            guard let self, self.generation == generation else { return }
            record(result)
        }
    }

    /// Place analyzer output from the utterance's first sound on the player's clock. Results
    /// only cover audio already scheduled, so the onset is known by the time they arrive.
    private func record(_ result: FormantLipsyncAnalyzer.Result) {
        guard let onset else { return }
        for keyframe in result.keyframes where keyframe.offset >= onset {
            keyframes.append((playerTime(forAudio: keyframe.offset),
                              Pose(openness: keyframe.openness, width: keyframe.width, rounding: keyframe.rounding)))
        }
        for event in result.events where event.kind == .closure && event.offset >= onset {
            let start = playerTime(forAudio: event.offset)
            closures.append((start, start + event.duration))
        }
    }

    /// The start of the first 10 ms block of `samples` whose RMS reaches `onsetLevel`.
    private static func firstSound(in samples: [Float]) -> Int? {
        var start = 0
        while start < samples.count {
            let end = min(start + onsetBlock, samples.count)
            var energy: Float = 0
            for i in start..<end { energy += samples[i] * samples[i] }
            if energy >= onsetLevel * onsetLevel * Float(end - start) { return start }
            start = end
        }
        return nil
    }

    /// Map seconds of utterance audio to the player's clock, across any gaps between chunks.
    private func playerTime(forAudio seconds: Double) -> Double {
        guard let segment = segments.last(where: { $0.audio <= seconds }) ?? segments.first else { return seconds }
        return segment.player + (seconds - segment.audio)
    }

    private func clearTrack() {
        segments = []
        scheduledAudio = 0
        onset = nil
        keyframes = []
        closures = []
        cursor = 0
    }
}

/// Owns the resampler and analyzer on a serial queue, so work runs in scheduling order.
private final class LipsyncAnalysis: @unchecked Sendable {
    private let queue = DispatchQueue(label: "ai.pipecat.visemes.lipsync", qos: .userInitiated)
    // Touched only on `queue`.
    private let analyzer = FormantLipsyncAnalyzer()
    private var resampler = LipsyncResampler()

    /// Discard the current utterance's state.
    func reset() {
        queue.async { [self] in
            analyzer.resetUtterance()
            resampler.reset()
        }
    }

    /// Analyze 24 kHz `samples`, or with nil flush the utterance's tail, then deliver
    /// the result on the main actor. The main queue is FIFO, so results stay in order.
    func analyze(_ samples: [Float]?, deliver: @escaping @MainActor @Sendable (FormantLipsyncAnalyzer.Result) -> Void) {
        queue.async { [self] in
            var result: FormantLipsyncAnalyzer.Result
            if let samples {
                result = analyzer.analyze(resampler.process(samples))
            } else {
                result = analyzer.analyze(resampler.flush())
                let tail = analyzer.flush()
                result.keyframes += tail.keyframes
                result.events += tail.events
                resampler.reset()
            }
            DispatchQueue.main.async { MainActor.assumeIsolated { deliver(result) } }
        }
    }
}
