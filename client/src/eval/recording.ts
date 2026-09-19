/**
 * Eval recordings: the files `python -m benchmarks.record` writes under
 * client/public/eval/ (served as static files, so no bot is needed), and the
 * derived per-clip data the Eval tab shows.
 *
 * A recording is one pass of the eval corpus through the bot's TTS and
 * lipsync pipeline: per example, the audio as played plus word timings. Each
 * run is one analysis of that audio (the live pass, or an offline
 * `--reanalyze` with newer lipsync code): every lipsync server-message with
 * the time it was released and the time it was due, in seconds from the
 * first played sample.
 */

import { SCHEDULING_LEAD_SEC } from "../lipsync/feed";
import {
  parseLipsyncData,
  type LipsyncBatch,
  type LipsyncEvent,
  type LipsyncEventKind,
} from "../lipsync/protocol";

const EVAL_BASE = `${import.meta.env.BASE_URL}eval`;

export interface RecordingSummary {
  id: string;
  label: string;
  provider: string;
  voice: string;
  created_at: string;
  examples: number;
  runs: number;
}

export interface EvalIndex {
  version: number;
  recordings: RecordingSummary[];
}

export interface Example {
  id: string;
  text: string;
  tags: string[];
  look_for: string;
  /** Audio path relative to the recording directory. */
  audio: string;
  duration: number;
  /** [start seconds, word] from the TTS's word timestamps. */
  words: [number, string][];
  word_timestamps: boolean;
}

export interface RunSummary {
  id: string;
  label: string;
  mode: "live" | "reanalyze";
  created_at: string;
  file: string;
}

export interface Recording {
  version: number;
  id: string;
  label: string;
  provider: string;
  voice: string;
  sample_rate: number;
  created_at: string;
  examples: Example[];
  runs: RunSummary[];
}

export interface RecordedMessage {
  /** Seconds from the first played sample when the message was released. */
  at: number;
  /** When it was due: its batch's scheduled release (the emission time if analysis ran late). */
  due: number;
  /** The server-message data, exactly as a client receives it. */
  data: unknown;
}

export interface RunExample {
  messages: RecordedMessage[];
  /** LipsyncProcessor counters for this example. */
  stats: Record<string, number>;
}

export interface Run {
  version: number;
  id: string;
  label: string;
  mode: "live" | "reanalyze";
  created_at: string;
  git: { sha: string; dirty: boolean };
  overrides: Record<string, unknown>;
  examples: Record<string, RunExample>;
}

/** A lipsync batch with its recorded delivery times. */
export interface TimedBatch {
  batch: LipsyncBatch;
  at: number;
  due: number;
}

export interface ClipStats {
  messages: number;
  keyframes: number;
  keyframesPerSec: number;
  events: Record<LipsyncEventKind, number>;
  /** How late a live client's mouth starts (ms): the first batch's implied anchor. */
  startLagMs: number | null;
  /** The anchor once every batch has landed (ms; ~0 when delivery is sound). */
  settledLagMs: number | null;
  /** When the anchor last moved (seconds into the clip). */
  settledAt: number | null;
  /** Batches released after their window had started playing. */
  lateBatches: number;
  /** Longest a batch was released after its scheduled time (ms). */
  maxHoldMs: number;
  /** Batches released at emission because their lead had already passed. */
  releaseClamped: number;
}

/** One example as the Eval tab lists it: data from the recording and the run. */
export interface Clip {
  example: Example;
  batches: TimedBatch[];
  stats: ClipStats;
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${EVAL_BASE}/${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} fetching eval/${path}`);
  return (await res.json()) as T;
}

/** The recordings index, or null when nothing has been recorded yet. */
export async function loadIndex(): Promise<EvalIndex | null> {
  const res = await fetch(`${EVAL_BASE}/index.json`, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} fetching eval/index.json`);
  // Vite's SPA fallback answers a missing file with index.html.
  if (!res.headers.get("content-type")?.includes("json")) return null;
  return (await res.json()) as EvalIndex;
}

export function loadRecording(recordingId: string): Promise<Recording> {
  return fetchJson<Recording>(`${recordingId}/recording.json`);
}

export function loadRun(recordingId: string, run: RunSummary): Promise<Run> {
  return fetchJson<Run>(`${recordingId}/${run.file}`);
}

export function audioUrl(recordingId: string, example: Example): string {
  return `${EVAL_BASE}/${recordingId}/${example.audio}`;
}

/** Examples with results in this run, in recording order. */
export function clipsForRun(recording: Recording, run: Run): Clip[] {
  return recording.examples.flatMap((example) => {
    const result = run.examples[example.id];
    if (!result) return [];
    const batches = result.messages.flatMap(({ at, due, data }) => {
      const batch = parseLipsyncData(data);
      return batch ? [{ batch, at, due }] : [];
    });
    return [{ example, batches, stats: clipStats(example, batches, result.stats) }];
  });
}

function firstOffset(batch: LipsyncBatch): number | null {
  return batch.keyframes[0]?.offset ?? batch.events[0]?.offset ?? null;
}

function clipStats(
  example: Example,
  batches: TimedBatch[],
  counters: Record<string, number>,
): ClipStats {
  const events: Record<LipsyncEventKind, number> = { closure: 0, nasal: 0, silence: 0 };
  let keyframes = 0;
  for (const { batch } of batches) {
    keyframes += batch.keyframes.length;
    for (const e of batch.events) events[e.kind] += 1;
  }

  // Anchor the way LipsyncFeed.ingest does: a version-2 batch says how far
  // ahead of its window's playout it was sent (so its implied anchor is
  // exact here, with no network); a version-1 batch was assumed to arrive
  // SCHEDULING_LEAD_SEC before its first offset. The first batch sets the
  // anchor and later ones only ever move it earlier. Relative to the audio,
  // the true anchor is 0.
  let anchor: number | null = null;
  let startLagMs: number | null = null;
  let settledAt: number | null = null;
  let lateBatches = 0;
  let maxHoldMs = 0;
  for (const { batch, at, due } of batches) {
    const exact = batch.windowStart !== undefined && batch.lead !== undefined;
    const start = exact ? batch.windowStart! : firstOffset(batch);
    if (start === null) continue;
    const implied = exact
      ? (at + batch.lead! - batch.windowStart!) * 1000
      : (at + SCHEDULING_LEAD_SEC - start) * 1000;
    if (anchor === null) {
      anchor = startLagMs = implied;
      settledAt = at;
    } else if (implied < anchor - 1) {
      anchor = implied;
      settledAt = at;
    }
    if (at > start) lateBatches += 1;
    maxHoldMs = Math.max(maxHoldMs, (at - due) * 1000);
  }

  return {
    messages: batches.length,
    keyframes,
    keyframesPerSec: example.duration > 0 ? keyframes / example.duration : 0,
    events,
    startLagMs,
    settledLagMs: anchor,
    settledAt,
    lateBatches,
    maxHoldMs,
    releaseClamped: counters.release_clamped ?? counters.pts_clamped ?? 0,
  };
}

/** Every event in the clip, in offset order. */
export function clipEvents(clip: Clip): LipsyncEvent[] {
  return clip.batches.flatMap(({ batch }) => batch.events).sort((a, b) => a.offset - b.offset);
}

export interface DecodedAudio {
  /** Object URL of the WAV, for an <audio> element. */
  url: string;
  samples: Float32Array;
  sampleRate: number;
  duration: number;
}

/** Fetches a recorded WAV once: an object URL to play, plus samples to draw. */
export async function loadAudio(url: string): Promise<DecodedAudio> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} fetching ${url}`);
  const buffer = await res.arrayBuffer();
  const { samples, sampleRate } = parseWav(buffer);
  return {
    url: URL.createObjectURL(new Blob([buffer], { type: "audio/wav" })),
    samples,
    sampleRate,
    duration: samples.length / sampleRate,
  };
}

/** 16-bit PCM WAV → mono float samples (the recorder writes mono). */
function parseWav(buffer: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const view = new DataView(buffer);
  const tag = (offset: number) =>
    String.fromCharCode(...new Uint8Array(buffer, offset, 4));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("not a WAV file");

  let channels = 0;
  let sampleRate = 0;
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      const format = view.getUint16(body, true);
      const bits = view.getUint16(body + 14, true);
      if (format !== 1 || bits !== 16) throw new Error("expected 16-bit PCM WAV");
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
    } else if (id === "data") {
      if (!channels) throw new Error("WAV data before fmt");
      const frames = Math.floor(size / (2 * channels));
      const samples = new Float32Array(frames);
      for (let i = 0; i < frames; i++) {
        samples[i] = view.getInt16(body + i * 2 * channels, true) / 32768;
      }
      return { samples, sampleRate };
    }
    offset = body + size + (size & 1);
  }
  throw new Error("WAV has no data chunk");
}
