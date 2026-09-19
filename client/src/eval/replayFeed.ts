import { LipsyncFeed, SCHEDULING_LEAD_SEC, type ArticulationSample } from "../lipsync/feed";
import type { LipsyncEvent } from "../lipsync/protocol";
import type { TimedBatch } from "./recording";

/**
 * How a replay delivers a clip's batches to the feed.
 *
 * - `delivered`: at their recorded release times, so the feed anchors the
 *   utterance exactly as a live client would have (minus network): late
 *   first batches start the mouth late, held batches arrive in bursts.
 * - `ideal`: each batch exactly SCHEDULING_LEAD_SEC ahead of its window, so
 *   the mouth is locked to the audio and only the analysis is on show.
 */
export type TimingMode = "delivered" | "ideal";

function windowStart(batch: TimedBatch["batch"]): number {
  return batch.windowStart ?? batch.keyframes[0]?.offset ?? batch.events[0]?.offset ?? 0;
}

/** The batch as the feed should see it in `mode`, and when. */
function timed({ batch, at }: TimedBatch, mode: TimingMode): { arrival: number; batch: TimedBatch["batch"] } {
  if (mode === "delivered") return { arrival: at * 1000, batch };
  // Arriving exactly one lead ahead of its window: say so in the batch too,
  // since the feed anchors on the lead it carries.
  const start = windowStart(batch);
  const ideal = batch.lead === undefined ? batch : { ...batch, lead: SCHEDULING_LEAD_SEC };
  return { arrival: (start - SCHEDULING_LEAD_SEC) * 1000, batch: ideal };
}

/**
 * A LipsyncFeed driven by a playback clock instead of the network.
 *
 * `now()` is the clip's playhead (ms from its first sample), and sampling at
 * a time first ingests every batch that had arrived by then, with its
 * arrival time — so the stock anchoring, interpolation and event logic run
 * unchanged, and the stock Mouth/Meters components render it. Seeking
 * backwards replays the batches from scratch, which is deterministic.
 */
export class ReplayFeed extends LipsyncFeed {
  private readonly clock: () => number;
  private queue: { arrival: number; batch: TimedBatch["batch"] }[] = [];
  private next = 0;
  private horizon = -Infinity;

  constructor(clock: () => number) {
    super();
    this.clock = clock;
  }

  override now(): number {
    return this.clock();
  }

  load(batches: TimedBatch[], mode: TimingMode): void {
    this.queue = batches.map((b) => timed(b, mode)).sort((a, b) => a.arrival - b.arrival);
    this.rewind();
  }

  override sample(nowMs: number): ArticulationSample {
    this.deliver(nowMs);
    return super.sample(nowMs);
  }

  override activeEvents(nowMs: number): LipsyncEvent[] {
    this.deliver(nowMs);
    return super.activeEvents(nowMs);
  }

  private deliver(nowMs: number): void {
    if (nowMs < this.horizon) this.rewind();
    this.horizon = nowMs;
    while (this.next < this.queue.length && this.queue[this.next].arrival <= nowMs) {
      const { arrival, batch } = this.queue[this.next++];
      this.ingest(batch, arrival);
    }
  }

  private rewind(): void {
    this.clear();
    this.next = 0;
    this.horizon = -Infinity;
  }
}

/** The pose a replay renders over a whole clip, sampled on a fixed grid. */
export interface ClipTrace {
  stepSec: number;
  openness: Float32Array;
  width: Float32Array;
  rounding: Float32Array;
  energy: Float32Array;
}

const TRACE_STEP_SEC = 0.01;

/** Runs a replay over the clip offline: exactly what the mouth will show. */
export function traceClip(batches: TimedBatch[], mode: TimingMode, durationSec: number): ClipTrace {
  let clockMs = 0;
  const feed = new ReplayFeed(() => clockMs);
  feed.load(batches, mode);
  const n = Math.max(2, Math.ceil(durationSec / TRACE_STEP_SEC) + 1);
  const trace: ClipTrace = {
    stepSec: TRACE_STEP_SEC,
    openness: new Float32Array(n),
    width: new Float32Array(n),
    rounding: new Float32Array(n),
    energy: new Float32Array(n),
  };
  for (let i = 0; i < n; i++) {
    clockMs = i * TRACE_STEP_SEC * 1000;
    const s = feed.sample(clockMs);
    trace.openness[i] = s.openness;
    trace.width[i] = s.width;
    trace.rounding[i] = s.rounding;
    trace.energy[i] = s.energy;
  }
  return trace;
}
