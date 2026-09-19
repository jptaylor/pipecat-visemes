import { useEffect, useRef, type PointerEvent } from "react";

import type { EvalSession, LoadedClip } from "../eval/session";

const PAD_X = 8;
const EVENT_Y = 2; // event spans: a colored band along the top...
const EVENT_H = 8;
const ARRIVAL_Y = 13; // ...then batch arrival ticks...
const ARRIVAL_H = 8;
const PLOT_TOP = 26; // ...then waveform and pose traces...
const WORD_BAND = 18; // ...and word labels along the bottom.
const MIN_EVENT_SEC = 0.03; // so instant events still show

const TRACES = [
  { key: "openness", cssVar: "--openness" },
  { key: "width", cssVar: "--width" },
  { key: "rounding", cssVar: "--rounding" },
] as const;

interface Peaks {
  loaded: LoadedClip;
  columns: number;
  minMax: Float32Array;
  gain: number;
}

function computePeaks(loaded: LoadedClip, columns: number): Peaks {
  const { samples } = loaded.audio;
  const minMax = new Float32Array(columns * 2);
  const perColumn = samples.length / columns;
  let loudest = 1e-6;
  for (let c = 0; c < columns; c++) {
    let lo = 0;
    let hi = 0;
    const end = Math.min(samples.length, Math.floor((c + 1) * perColumn));
    for (let i = Math.floor(c * perColumn); i < end; i++) {
      const v = samples[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    minMax[2 * c] = lo;
    minMax[2 * c + 1] = hi;
    loudest = Math.max(loudest, -lo, hi);
  }
  return { loaded, columns, minMax, gain: 1 / loudest };
}

/**
 * The whole selected clip: audio waveform, the pose the mouth renders over it
 * (in the current timing mode), event spans, when each lipsync batch arrived
 * (red when it landed after its audio had started), word timings, and a
 * playhead. Click or drag to seek.
 */
export function ClipTimeline({ session }: { session: EvalSession }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const readoutRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    const styles = getComputedStyle(canvas);
    const color = (cssVar: string) => styles.getPropertyValue(cssVar).trim() || "#888";
    const eventColors: Record<string, string> = {
      closure: color("--closure"),
      nasal: color("--nasal"),
      silence: color("--silence"),
    };
    const mono = color("--mono");

    let peaks: Peaks | null = null;
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { width: w, height: h } = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, w, h);
      const loaded = session.getState().loaded;
      const position = session.position;
      if (readoutRef.current) {
        readoutRef.current.textContent = loaded
          ? `${position.toFixed(2)} / ${loaded.audio.duration.toFixed(2)} s`
          : "";
      }
      if (!loaded) return;

      const { audio, trace, events, clip, mode } = loaded;
      const duration = audio.duration || 1;
      const plotBottom = h - WORD_BAND;
      const x = (t: number) => PAD_X + (t / duration) * (w - 2 * PAD_X);
      const y = (v: number) => plotBottom - Math.max(0, Math.min(1, v)) * (plotBottom - PLOT_TOP);

      // Grid: a line every half second, stronger on whole seconds.
      ctx.strokeStyle = color("--grid");
      ctx.lineWidth = 1;
      for (let t = 0; t <= duration; t += 0.5) {
        ctx.globalAlpha = Number.isInteger(t) ? 1 : 0.45;
        ctx.beginPath();
        ctx.moveTo(x(t), PLOT_TOP);
        ctx.lineTo(x(t), plotBottom);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Waveform, normalized to the clip's loudest sample.
      const columns = Math.max(1, Math.floor(w - 2 * PAD_X));
      if (!peaks || peaks.loaded !== loaded || peaks.columns !== columns) {
        peaks = computePeaks(loaded, columns);
      }
      const mid = (PLOT_TOP + plotBottom) / 2;
      const amp = ((plotBottom - PLOT_TOP) / 2) * 0.92 * peaks.gain;
      ctx.fillStyle = color("--waveform");
      for (let c = 0; c < columns; c++) {
        const lo = peaks.minMax[2 * c];
        const hi = peaks.minMax[2 * c + 1];
        ctx.fillRect(PAD_X + c, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
      }

      // Events: a tint over the plot plus a solid band along the top.
      for (const e of events) {
        const x0 = x(e.offset);
        const x1 = Math.max(x0 + 2, x(e.offset + Math.max(e.duration, MIN_EVENT_SEC)));
        ctx.fillStyle = eventColors[e.kind] ?? "#888";
        ctx.globalAlpha = 0.08;
        ctx.fillRect(x0, PLOT_TOP, x1 - x0, plotBottom - PLOT_TOP);
        ctx.globalAlpha = 0.9;
        ctx.fillRect(x0, EVENT_Y, x1 - x0, EVENT_H);
      }
      ctx.globalAlpha = 1;

      // Batch arrivals (as delivered): a tick when each batch reached the
      // client; a late one also gets a line back to where its audio began.
      if (mode === "delivered") {
        for (const { batch, at } of clip.batches) {
          const first = batch.windowStart ?? batch.keyframes[0]?.offset ?? batch.events[0]?.offset ?? at;
          const late = at > first;
          ctx.fillStyle = late ? eventColors.closure : color("--muted");
          ctx.fillRect(x(at) - 0.75, ARRIVAL_Y, 1.5, ARRIVAL_H);
          if (late) ctx.fillRect(x(first), ARRIVAL_Y + ARRIVAL_H / 2 - 0.5, x(at) - x(first), 1);
        }
      }

      // Energy as a filled area, then the pose traces.
      const n = trace.openness.length;
      ctx.beginPath();
      ctx.moveTo(x(0), plotBottom);
      for (let i = 0; i < n; i++) ctx.lineTo(x(i * trace.stepSec), y(trace.energy[i]));
      ctx.lineTo(x((n - 1) * trace.stepSec), plotBottom);
      ctx.closePath();
      ctx.fillStyle = color("--energy-area");
      ctx.fill();
      ctx.lineWidth = 1.6;
      for (const { key, cssVar } of TRACES) {
        const values = trace[key];
        ctx.strokeStyle = color(cssVar);
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          if (i === 0) ctx.moveTo(x(0), y(values[0]));
          else ctx.lineTo(x(i * trace.stepSec), y(values[i]));
        }
        ctx.stroke();
      }

      // Words at their start times; labels that would collide are skipped.
      ctx.font = `10px ${mono}`;
      ctx.fillStyle = color("--muted");
      let labelEnd = -Infinity;
      for (const [t, word] of clip.example.words) {
        const wx = x(t);
        ctx.fillRect(wx, plotBottom, 1, 4);
        if (wx >= labelEnd + 4) {
          ctx.fillText(word, wx + 2, h - 5);
          labelEnd = wx + 2 + ctx.measureText(word).width;
        }
      }

      // Playhead.
      ctx.fillStyle = color("--accent");
      ctx.fillRect(x(position) - 0.75, 0, 1.5, h);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, [session]);

  const seekTo = (e: PointerEvent<HTMLCanvasElement>) => {
    const loaded = session.getState().loaded;
    if (!loaded) return;
    const rect = e.currentTarget.getBoundingClientRect();
    session.seek(((e.clientX - rect.left - PAD_X) / (rect.width - 2 * PAD_X)) * loaded.audio.duration);
  };

  return (
    <div className="timeline">
      <div className="timeline-legend">
        <span style={{ color: "var(--openness)" }}>openness</span>
        <span style={{ color: "var(--width)" }}>width</span>
        <span style={{ color: "var(--rounding)" }}>rounding</span>
        <span style={{ color: "var(--energy)" }}>energy</span>
        <span className="legend-events">
          <span style={{ color: "var(--closure)" }}>closure</span>
          <span style={{ color: "var(--nasal)" }}>nasal</span>
          <span style={{ color: "var(--silence)" }}>silence</span>
        </span>
        <span className="timeline-window" ref={readoutRef} />
      </div>
      <canvas
        ref={canvasRef}
        className="timeline-canvas clip-timeline"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          seekTo(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons & 1) seekTo(e);
        }}
      />
    </div>
  );
}
