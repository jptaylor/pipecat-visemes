import type {
  LipsyncBatch,
  LipsyncEvent,
  LipsyncEventKind,
  LipsyncKeyframe,
} from "./protocol";

/**
 * Server releases each batch this far ahead of its window's audio playout
 * (LipsyncParams.scheduling_lead_ms default).
 */
export const SCHEDULING_LEAD_SEC = 0.2;

/** Mouth pose when no keyframes apply (mirrors the analyzer's silence rest). */
export const REST_POSE = {
  openness: 0.15,
  width: 0.35,
  rounding: 0.1,
  energy: 0,
  pitch: 0,
  confidence: 0,
};

const REST_HOLD_SEC = 0.25; // hold the last pose this long past the final keyframe
const REST_EASE_SEC = 0.3; // then ease toward rest with this time constant
const PREROLL_SEC = 0.2; // blend rest -> first keyframe over this window
const PRUNE_HORIZON_SEC = 30; // drop keyframes this far behind the playhead
const RATE_WINDOW_MS = 5000; // sliding window for msg/s + kf/s rates
const MIN_EVENT_ACTIVE_SEC = 0.25; // floor so zero-duration events still flash

export interface ArticulationSample {
  openness: number;
  width: number;
  rounding: number;
  energy: number;
  pitch: number;
  confidence: number;
  /** True while keyframes are actively driving the pose. */
  live: boolean;
  /** Playhead position in utterance time, or null before any batch. */
  relTime: number | null;
}

export interface LoggedEvent extends LipsyncEvent {
  ctx: string | null;
  seq: number;
  wallMs: number;
}

export interface FeedStats {
  msgsPerSec: number;
  kfsPerSec: number;
  totalMessages: number;
  totalKeyframes: number;
  totalEvents: number;
  ctx: string | null;
  queueLength: number;
  lastLeadMs: number | null;
  resyncs: number;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpPose(
  a: ArticulationSample,
  k: LipsyncKeyframe,
  t: number,
): ArticulationSample {
  return {
    openness: lerp(a.openness, k.openness, t),
    width: lerp(a.width, k.width, t),
    rounding: lerp(a.rounding, k.rounding, t),
    energy: lerp(a.energy, k.energy, t),
    pitch: lerp(a.pitch, k.pitch, t),
    confidence: lerp(a.confidence, k.confidence, t),
    live: true,
    relTime: null,
  };
}

/**
 * Buffers lipsync batches and plays them back on a wall-clock timeline.
 *
 * Keyframe offsets are utterance-relative, and the server releases each batch
 * SCHEDULING_LEAD_SEC ahead of its audio, so a batch's arrival time implies
 * where the utterance's t=0 sits on the wall clock ("anchor"). Batches are
 * stored in offset space; `sample()` maps the wall clock through the anchor
 * and interpolates between the bracketing keyframes.
 */
export class LipsyncFeed {
  /** User-adjustable A/V trim in ms (positive delays the mouth). */
  offsetTrimMs = 0;

  lastBatch: LipsyncBatch | null = null;
  eventLog: LoggedEvent[] = [];

  private anchorMs: number | null = null;
  private ctx: string | null = null;
  private kfs: LipsyncKeyframe[] = [];
  private evs: LipsyncEvent[] = [];
  private cursor = 0;
  private msgTimes: number[] = [];
  private kfTimes: number[] = [];
  private totalMessages = 0;
  private totalKeyframes = 0;
  private totalEvents = 0;
  private resyncs = 0;
  private lastLeadMs: number | null = null;
  private listeners = new Set<() => void>();

  now(): number {
    return performance.now();
  }

  ingest(batch: LipsyncBatch): void {
    const now = this.now();
    const first = batch.keyframes[0]?.offset ?? batch.events[0]?.offset;
    if (first === undefined) return;

    const impliedAnchor = now + SCHEDULING_LEAD_SEC * 1000 - first * 1000;
    let anchor = this.anchorMs;
    if (batch.ctx !== this.ctx || anchor === null) {
      // New utterance (or first ever): drop the old queue and re-anchor.
      this.ctx = batch.ctx;
      anchor = impliedAnchor;
      this.kfs = [];
      this.evs = [];
      this.cursor = 0;
    } else if (impliedAnchor < anchor - 1) {
      // A batch arriving with more lead implies an earlier true anchor;
      // late batches (network jitter, pts clamping) never move it.
      if (anchor - impliedAnchor > 120) this.resyncs++;
      anchor = impliedAnchor;
    }
    this.anchorMs = anchor;
    this.lastLeadMs = anchor + first * 1000 - now;

    this.kfs.push(...batch.keyframes);
    for (const e of batch.events) {
      this.evs.push(e);
      // wallMs is the event's scheduled playback time, not its arrival time.
      this.eventLog.push({
        ...e,
        ctx: batch.ctx,
        seq: this.totalEvents,
        wallMs: anchor + e.offset * 1000,
      });
      this.totalEvents += 1;
    }
    if (this.eventLog.length > 200) this.eventLog.splice(0, this.eventLog.length - 200);

    this.totalMessages += 1;
    this.totalKeyframes += batch.keyframes.length;
    this.msgTimes.push(now);
    for (let i = 0; i < batch.keyframes.length; i++) this.kfTimes.push(now);
    this.trimRates(now);
    this.prune();

    this.lastBatch = batch;
    this.notify();
  }

  /** Utterance-relative playhead time for a wall-clock timestamp. */
  relTime(nowMs: number): number | null {
    if (this.anchorMs === null) return null;
    return (nowMs - this.anchorMs - this.offsetTrimMs) / 1000;
  }

  sample(nowMs: number): ArticulationSample {
    const rest: ArticulationSample = { ...REST_POSE, live: false, relTime: null };
    const rel = this.relTime(nowMs);
    if (rel === null) return rest;
    rest.relTime = rel;
    if (this.kfs.length === 0) return rest;

    // Advance the cursor to the last keyframe at or before the playhead.
    while (this.cursor + 1 < this.kfs.length && this.kfs[this.cursor + 1].offset <= rel) {
      this.cursor += 1;
    }
    while (this.cursor > 0 && this.kfs[this.cursor].offset > rel) {
      this.cursor -= 1;
    }

    const first = this.kfs[0];
    if (rel < first.offset) {
      // Not started yet: blend from rest into the first keyframe.
      const until = first.offset - rel;
      if (until > PREROLL_SEC) return rest;
      return { ...lerpPose(rest, first, 1 - until / PREROLL_SEC), relTime: rel };
    }

    const k0 = this.kfs[this.cursor];
    const k1 = this.kfs[this.cursor + 1];
    if (k1 !== undefined) {
      const span = k1.offset - k0.offset;
      const t = span > 0 ? Math.min(1, (rel - k0.offset) / span) : 1;
      const k0Pose: ArticulationSample = { ...k0, live: true, relTime: rel };
      return { ...lerpPose(k0Pose, k1, t), relTime: rel };
    }

    // Past the last keyframe: hold briefly, then ease to rest.
    const over = rel - k0.offset;
    if (over <= REST_HOLD_SEC) return { ...k0, live: true, relTime: rel };
    const fade = 1 - Math.exp(-(over - REST_HOLD_SEC) / REST_EASE_SEC);
    const k0Pose: ArticulationSample = { ...k0, live: over < 2, relTime: rel };
    const restKf: LipsyncKeyframe = { ...REST_POSE, offset: 0 };
    return { ...lerpPose(k0Pose, restKf, fade), relTime: rel };
  }

  /** Events whose span covers the playhead (for badges/flashes). */
  activeEvents(nowMs: number): LipsyncEvent[] {
    const rel = this.relTime(nowMs);
    if (rel === null) return [];
    return this.evs.filter(
      (e) => rel >= e.offset && rel <= e.offset + Math.max(e.duration, MIN_EVENT_ACTIVE_SEC),
    );
  }

  activeEventKinds(nowMs: number): Set<LipsyncEventKind> {
    return new Set(this.activeEvents(nowMs).map((e) => e.kind));
  }

  statsSnapshot(nowMs: number): FeedStats {
    this.trimRates(nowMs);
    const windowSec = RATE_WINDOW_MS / 1000;
    return {
      msgsPerSec: this.msgTimes.length / windowSec,
      kfsPerSec: this.kfTimes.length / windowSec,
      totalMessages: this.totalMessages,
      totalKeyframes: this.totalKeyframes,
      totalEvents: this.totalEvents,
      ctx: this.ctx,
      queueLength: this.kfs.length,
      lastLeadMs: this.lastLeadMs,
      resyncs: this.resyncs,
    };
  }

  /** Notified whenever a batch is ingested (event log, stats, inspector). */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  reset(): void {
    this.anchorMs = null;
    this.ctx = null;
    this.kfs = [];
    this.evs = [];
    this.cursor = 0;
    this.lastLeadMs = null;
    this.lastBatch = null;
    this.notify();
  }

  private prune(): void {
    const rel = this.relTime(this.now());
    if (rel === null) return;
    const horizon = rel - PRUNE_HORIZON_SEC;
    let drop = 0;
    while (drop < this.kfs.length - 1 && this.kfs[drop + 1].offset < horizon) drop += 1;
    if (drop > 0) {
      this.kfs.splice(0, drop);
      this.cursor = Math.max(0, this.cursor - drop);
    }
    this.evs = this.evs.filter((e) => e.offset + e.duration >= horizon);
  }

  private trimRates(nowMs: number): void {
    const cutoff = nowMs - RATE_WINDOW_MS;
    while (this.msgTimes.length > 0 && this.msgTimes[0] < cutoff) this.msgTimes.shift();
    while (this.kfTimes.length > 0 && this.kfTimes[0] < cutoff) this.kfTimes.shift();
  }

  private notify(): void {
    for (const listener of this.listeners) listener();
  }
}
