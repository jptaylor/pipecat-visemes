import { useEffect, useRef } from "react";

import type { LipsyncFeed } from "../lipsync/feed";

const WINDOW_SEC = 8;
const TRACES = [
  { key: "openness", cssVar: "--openness" },
  { key: "width", cssVar: "--width" },
  { key: "rounding", cssVar: "--rounding" },
] as const;

interface Point {
  wallMs: number;
  openness: number;
  width: number;
  rounding: number;
  energy: number;
}

interface Marker {
  wallMs: number;
  kind: string;
}

/** Scrolling strip chart of the sampled articulation trajectory. */
export function Timeline({ feed }: { feed: LipsyncFeed }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const points: Point[] = [];
    const markers: Marker[] = [];
    let seenEvents = 0;
    let raf = 0;

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

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const now = feed.now();
      const s = feed.sample(now);
      points.push({
        wallMs: now,
        openness: s.openness,
        width: s.width,
        rounding: s.rounding,
        energy: s.energy,
      });
      const cutoff = now - WINDOW_SEC * 1000;
      while (points.length > 0 && points[0].wallMs < cutoff) points.shift();

      // Pick up newly logged events; anchor their marker at playback time.
      for (; seenEvents < feed.eventLog.length; seenEvents++) {
        const e = feed.eventLog[seenEvents];
        markers.push({ wallMs: e.wallMs, kind: e.kind });
      }
      while (markers.length > 0 && markers[0].wallMs < cutoff) markers.shift();

      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      ctx.clearRect(0, 0, w, h);

      const x = (wallMs: number) => ((wallMs - cutoff) / (WINDOW_SEC * 1000)) * w;
      const y = (v: number) => h - 4 - Math.max(0, Math.min(1, v)) * (h - 12);

      // Grid: one vertical line per second.
      ctx.strokeStyle = color("--grid");
      ctx.lineWidth = 1;
      const firstSec = Math.ceil(cutoff / 1000) * 1000;
      for (let t = firstSec; t <= now; t += 1000) {
        ctx.beginPath();
        ctx.moveTo(x(t), 0);
        ctx.lineTo(x(t), h);
        ctx.stroke();
      }

      // Energy as a filled area behind the traces.
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(x(points[0].wallMs), h);
        for (const p of points) ctx.lineTo(x(p.wallMs), y(p.energy));
        ctx.lineTo(x(points[points.length - 1].wallMs), h);
        ctx.closePath();
        ctx.fillStyle = color("--energy-area");
        ctx.fill();
      }

      for (const { key, cssVar } of TRACES) {
        if (points.length < 2) break;
        ctx.beginPath();
        ctx.strokeStyle = color(cssVar);
        ctx.lineWidth = 1.8;
        points.forEach((p, i) => {
          if (i === 0) ctx.moveTo(x(p.wallMs), y(p[key]));
          else ctx.lineTo(x(p.wallMs), y(p[key]));
        });
        ctx.stroke();
      }

      // Event markers as ticks along the top.
      for (const m of markers) {
        ctx.strokeStyle = color(`--${m.kind}`);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x(m.wallMs), 0);
        ctx.lineTo(x(m.wallMs), 14);
        ctx.stroke();
      }
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, [feed]);

  return (
    <div className="timeline">
      <div className="timeline-legend">
        <span style={{ color: "var(--openness)" }}>openness</span>
        <span style={{ color: "var(--width)" }}>width</span>
        <span style={{ color: "var(--rounding)" }}>rounding</span>
        <span style={{ color: "var(--energy)" }}>energy</span>
        <span className="timeline-window">last {WINDOW_SEC}s</span>
      </div>
      <canvas ref={canvasRef} className="timeline-canvas" />
    </div>
  );
}
