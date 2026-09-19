import type { LipsyncEvent } from "../lipsync/protocol";
import {
  audioUrl,
  clipEvents,
  clipsForRun,
  loadAudio,
  loadIndex,
  loadRecording,
  loadRun,
  type Clip,
  type DecodedAudio,
  type Recording,
  type RecordingSummary,
  type Run,
} from "./recording";
import { ReplayFeed, traceClip, type ClipTrace, type TimingMode } from "./replayFeed";

const ADVANCE_DELAY_MS = 800; // pause between clips in play-all, so the mouth settles
const RUNOUT_MS = 2000; // the clock runs on this far past a clip's end (mouth eases to rest)
const STEP_SEC = 0.02; // one analysis hop

/** The selected clip, ready to play and draw. */
export interface LoadedClip {
  clip: Clip;
  audio: DecodedAudio;
  events: LipsyncEvent[];
  /** What the mouth renders over the clip in `mode`. */
  trace: ClipTrace;
  mode: TimingMode;
}

export interface EvalState {
  status: "loading" | "empty" | "ready" | "error";
  error: string | null;
  recordings: RecordingSummary[];
  recording: Recording | null;
  run: Run | null;
  /** Examples with results in the selected run. */
  clips: Clip[];
  /** Selected clip, or -1. */
  index: number;
  /** The selected clip once its audio has loaded. */
  loaded: LoadedClip | null;
  playing: boolean;
  playAll: boolean;
  rate: number;
  mode: TimingMode;
  trimMs: number;
}

const INITIAL: EvalState = {
  status: "loading",
  error: null,
  recordings: [],
  recording: null,
  run: null,
  clips: [],
  index: -1,
  loaded: null,
  playing: false,
  playAll: false,
  rate: 1,
  mode: "delivered",
  trimMs: 0,
};

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * The Eval tab's model: loads recordings and runs, plays the selected clip's
 * retained audio, and drives a ReplayFeed from the audio clock so the stock
 * mouth components render the recorded lipsync in sync with it. Components
 * read `getState()` through useSyncExternalStore; canvas loops read
 * `position` and `feed` directly every frame.
 */
export class EvalSession {
  readonly feed: ReplayFeed;
  private readonly audio = new Audio();
  private state: EvalState = INITIAL;
  private listeners = new Set<() => void>();
  private audioCache = new Map<string, Promise<DecodedAudio>>();
  private audioSrc: string | null = null;
  private endedAt: number | null = null;
  private advanceTimer: number | null = null;
  private dataToken = 0;
  private clipToken = 0;

  constructor() {
    this.feed = new ReplayFeed(() => this.clockMs());
    this.audio.preservesPitch = true;
    this.audio.addEventListener("play", () => this.update({ playing: true }));
    this.audio.addEventListener("pause", () => this.update({ playing: false }));
    this.audio.addEventListener("ended", () => this.onEnded());
    void this.reload();
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getState = (): EvalState => this.state;

  /** Playhead in seconds from the clip's first sample. */
  get position(): number {
    const duration = this.state.loaded?.audio.duration ?? 0;
    return Math.min(this.clockMs() / 1000, duration);
  }

  /** Re-reads the recordings (after a new record/reanalyze run), keeping the selection. */
  async reload(): Promise<void> {
    const { recording, run, clips, index } = this.state;
    this.stop();
    for (const pending of this.audioCache.values()) {
      pending.then((a) => URL.revokeObjectURL(a.url)).catch(() => {});
    }
    this.audioCache.clear();
    this.audioSrc = null;
    const token = ++this.dataToken;
    this.update({ status: "loading", error: null });
    try {
      const index_ = await loadIndex();
      if (token !== this.dataToken) return;
      if (!index_ || index_.recordings.length === 0) {
        this.update({ ...INITIAL, status: "empty" });
        return;
      }
      const recordingId =
        index_.recordings.find((r) => r.id === recording?.id)?.id ?? index_.recordings[0].id;
      this.update({ recordings: index_.recordings });
      await this.openRecording(token, recordingId, run?.id, clips[index]?.example.id);
    } catch (e) {
      if (token === this.dataToken) this.update({ status: "error", error: message(e) });
    }
  }

  async selectRecording(id: string): Promise<void> {
    this.stop();
    const token = ++this.dataToken;
    try {
      await this.openRecording(token, id);
    } catch (e) {
      if (token === this.dataToken) this.update({ status: "error", error: message(e) });
    }
  }

  /** Switches runs on the same audio: the clip, playhead and playback carry over. */
  async selectRun(id: string): Promise<void> {
    const { recording, clips, index, playing } = this.state;
    const summary = recording?.runs.find((r) => r.id === id);
    if (!recording || !summary) return;
    const token = ++this.dataToken;
    try {
      const run = await loadRun(recording.id, summary);
      if (token !== this.dataToken) return;
      this.showRun(recording, run, clips[index]?.example.id, playing);
    } catch (e) {
      if (token === this.dataToken) this.update({ status: "error", error: message(e) });
    }
  }

  select(index: number, play = true): void {
    void this.load(index, { play });
  }

  prev(): void {
    if (this.state.index > 0) this.select(this.state.index - 1);
  }

  next(): void {
    if (this.state.index + 1 < this.state.clips.length) this.select(this.state.index + 1);
  }

  play(): void {
    if (!this.state.loaded) return;
    this.cancelAdvance();
    if (this.endedAt !== null) this.seek(0);
    this.audio.play().catch(() => this.update({ playing: false }));
  }

  pause(): void {
    this.cancelAdvance();
    this.audio.pause();
  }

  toggle(): void {
    if (this.state.playing) this.pause();
    else this.play();
  }

  /** Stops playback, e.g. when the tab is hidden. */
  stop(): void {
    this.update({ playAll: false });
    this.pause();
  }

  seek(seconds: number): void {
    const duration = this.state.loaded?.audio.duration ?? 0;
    this.endedAt = null;
    this.audio.currentTime = Math.max(0, Math.min(duration, seconds));
  }

  /** Pauses and moves the playhead by one analysis hop. */
  step(direction: -1 | 1): void {
    const position = this.position;
    this.pause();
    this.seek(position + direction * STEP_SEC);
  }

  /** Plays through every clip from the selected one (or stops doing so). */
  setPlayAll(on: boolean): void {
    this.update({ playAll: on });
    if (on && !this.state.playing) this.play();
  }

  setRate(rate: number): void {
    // A new src resets playbackRate to defaultPlaybackRate.
    this.audio.defaultPlaybackRate = rate;
    this.audio.playbackRate = rate;
    this.update({ rate });
  }

  setMode(mode: TimingMode): void {
    const { loaded } = this.state;
    if (loaded) {
      this.feed.load(loaded.clip.batches, mode);
      const trace = traceClip(loaded.clip.batches, mode, loaded.audio.duration);
      this.update({ mode, loaded: { ...loaded, trace, mode } });
    } else {
      this.update({ mode });
    }
  }

  setTrim(ms: number): void {
    this.feed.offsetTrimMs = ms;
    this.update({ trimMs: ms });
  }

  private async openRecording(
    token: number,
    recordingId: string,
    runId?: string,
    clipId?: string,
  ): Promise<void> {
    this.update({ status: "loading", error: null });
    const recording = await loadRecording(recordingId);
    if (token !== this.dataToken) return;
    const summary =
      recording.runs.find((r) => r.id === runId) ?? recording.runs[recording.runs.length - 1];
    if (!summary) throw new Error(`recording ${recordingId} has no runs`);
    const run = await loadRun(recordingId, summary);
    if (token !== this.dataToken) return;
    this.showRun(recording, run, clipId, false);
  }

  private showRun(recording: Recording, run: Run, clipId: string | undefined, play: boolean) {
    const clips = clipsForRun(recording, run);
    const index = Math.max(
      0,
      clips.findIndex((c) => c.example.id === clipId),
    );
    const sameRecording = this.state.recording?.id === recording.id;
    this.update({ status: "ready", error: null, recording, run, clips });
    if (clips.length === 0) {
      this.update({ index: -1, loaded: null });
      return;
    }
    const keepPosition = sameRecording && clips[index].example.id === clipId;
    void this.load(index, { play, keepPosition });
  }

  private async load(
    index: number,
    { play, keepPosition = false }: { play: boolean; keepPosition?: boolean },
  ): Promise<void> {
    const { clips, recording } = this.state;
    const clip = clips[index];
    if (!clip || !recording) return;
    const token = ++this.clipToken;
    this.cancelAdvance();
    const url = audioUrl(recording.id, clip.example);
    const sameAudio = url === this.audioSrc;
    if (!sameAudio) {
      this.audio.pause();
      this.endedAt = null;
      this.update({ index, loaded: null });
    } else {
      this.update({ index });
    }

    let audio: DecodedAudio;
    try {
      audio = await this.fetchAudio(url);
    } catch (e) {
      if (token === this.clipToken) this.update({ error: message(e) });
      return;
    }
    if (token !== this.clipToken) return;
    if (!sameAudio) {
      this.audio.src = audio.url;
      this.audioSrc = url;
    }
    const { mode } = this.state; // may have changed while the audio loaded
    this.feed.load(clip.batches, mode);
    this.update({
      error: null,
      loaded: {
        clip,
        audio,
        events: clipEvents(clip),
        trace: traceClip(clip.batches, mode, audio.duration),
        mode,
      },
    });
    if (!keepPosition) this.seek(0);
    if (play) this.play();
  }

  private fetchAudio(url: string): Promise<DecodedAudio> {
    let pending = this.audioCache.get(url);
    if (!pending) {
      pending = loadAudio(url);
      pending.catch(() => this.audioCache.delete(url));
      this.audioCache.set(url, pending);
    }
    return pending;
  }

  /**
   * Replay clock in ms: the audio element's position while it has audio;
   * after the clip ends it runs on briefly (at the playback rate) so the
   * mouth eases to rest just as it does after a live utterance.
   */
  private clockMs(): number {
    if (this.endedAt !== null) {
      const duration = this.state.loaded?.audio.duration ?? 0;
      const runout = (performance.now() - this.endedAt) * this.state.rate;
      return duration * 1000 + Math.min(runout, RUNOUT_MS);
    }
    return this.audio.currentTime * 1000;
  }

  private onEnded(): void {
    this.endedAt = performance.now();
    const { playAll, index, clips } = this.state;
    if (!playAll) return;
    if (index + 1 >= clips.length) {
      this.update({ playAll: false });
      return;
    }
    this.advanceTimer = window.setTimeout(() => {
      this.advanceTimer = null;
      void this.load(index + 1, { play: true });
    }, ADVANCE_DELAY_MS);
  }

  private cancelAdvance(): void {
    if (this.advanceTimer !== null) {
      window.clearTimeout(this.advanceTimer);
      this.advanceTimer = null;
    }
  }

  private update(patch: Partial<EvalState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }
}

let shared: EvalSession | null = null;

/** The page's eval session: created on first use, kept across tab switches. */
export function evalSession(): EvalSession {
  shared ??= new EvalSession();
  return shared;
}
