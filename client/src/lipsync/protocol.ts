/**
 * Wire format of the lipsync payload, delivered inside the standard RTVI
 * `server-message`.
 *
 * Mirrors `lipsync_message_data` in the bot's lipsync/rtvi.py: the payload
 * arrives as the `data` object passed to `onServerMessage`, with
 * `type: "bot-tts-lipsync"` as the demux discriminator (other server
 * messages are ignored by `parseLipsyncData`). Keyframes and events are
 * positional arrays for wire compaction, offsets are seconds from the first
 * audio of `ctx`, and `t0` is a base offset added to all offsets (0 in
 * version 1).
 */

export const LIPSYNC_MESSAGE_TYPE = "bot-tts-lipsync";

export type LipsyncEventKind = "closure" | "nasal" | "silence";

export interface LipsyncKeyframe {
  /** Seconds from the utterance's first audio (t0 already applied). */
  offset: number;
  openness: number;
  width: number;
  rounding: number;
  energy: number;
  pitch: number;
  confidence: number;
}

export interface LipsyncEvent {
  offset: number;
  kind: LipsyncEventKind;
  duration: number;
  confidence: number;
}

export interface LipsyncBatch {
  version: number;
  ctx: string | null;
  keyframes: LipsyncKeyframe[];
  events: LipsyncEvent[];
  /** Original message data, for the raw inspector. */
  raw: unknown;
}

type KeyframeTuple = [number, number, number, number, number, number, number];
type EventTuple = [number, string, number, number];

export function parseLipsyncData(data: unknown): LipsyncBatch | null {
  if (typeof data !== "object" || data === null) return null;
  const d = data as Record<string, unknown>;
  if (d.type !== LIPSYNC_MESSAGE_TYPE) return null;
  const t0 = typeof d.t0 === "number" ? d.t0 : 0;
  const kf = Array.isArray(d.kf) ? (d.kf as KeyframeTuple[]) : [];
  const ev = Array.isArray(d.ev) ? (d.ev as EventTuple[]) : [];
  return {
    version: typeof d.version === "number" ? d.version : 1,
    ctx: typeof d.ctx === "string" ? d.ctx : null,
    keyframes: kf.map(
      ([offset, openness, width, rounding, energy, pitch, confidence]) => ({
        offset: t0 + offset,
        openness,
        width,
        rounding,
        energy,
        pitch,
        confidence,
      }),
    ),
    events: ev.map(([offset, kind, duration, confidence]) => ({
      offset: t0 + offset,
      kind: kind as LipsyncEventKind,
      duration,
      confidence,
    })),
    raw: data,
  };
}
