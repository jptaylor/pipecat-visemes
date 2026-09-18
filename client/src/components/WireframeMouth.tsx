import { useEffect, useRef } from "react";

import type { LipsyncFeed } from "../lipsync/feed";
import type { LipsyncEventKind } from "../lipsync/protocol";

const CX = 200;
const CY = 130;
const VIEW_W = 400;
const VIEW_H = 260;
const SEGS = 12; // samples per cubic -> 2*SEGS vertices per ring; multiple of 6 lands samples on the bow peaks/dip
const DOT_OUTER = 3.2;
const DOT_INNER = 2.6;
const DOT_DETAIL = 2; // teeth/tongue vertices
const TEETH_COUNT = 6;
const TONGUE_SEGS = 16;

const EVENT_LABELS: Record<LipsyncEventKind, string> = {
  closure: "CLOSURE",
  nasal: "NASAL",
  silence: "SILENCE",
};

interface Point {
  x: number;
  y: number;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function bez(a: number, b: number, c: number, d: number, t: number): number {
  const u = 1 - t;
  return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d;
}

function gauss(t: number, center: number, sigma: number): number {
  const d = (t - center) / sigma;
  return Math.exp(-0.5 * d * d);
}

/**
 * Cupid's bow: two peaks (up) at t=1/3 and t=2/3 with a philtrum dip (down)
 * between them, as a y-offset over the top curve's parameter. Zero at the
 * corners so ring endpoints stay put.
 */
function bowOffset(t: number, amp: number): number {
  return amp * (0.9 * gauss(t, 0.5, 0.07) - gauss(t, 1 / 3, 0.09) - gauss(t, 2 / 3, 0.09));
}

/**
 * One closed lip contour: the same two mirrored cubics as the classic Mouth,
 * sampled at SEGS points each (t=1 endpoints excluded so each corner is
 * contributed once, by the other curve's t=0), shaped by a cupid's bow on
 * the top edge and center fullness on the bottom edge.
 */
function ringPoints(
  hw: number,
  top: number,
  bot: number,
  yc: number,
  bow: number,
  full: number,
): Point[] {
  const xL = CX - hw;
  const xR = CX + hw;
  const cxOff = hw * 0.52;
  const pts: Point[] = [];
  for (let i = 0; i < SEGS; i++) {
    const t = i / SEGS;
    pts.push({
      x: bez(xL, CX - cxOff, CX + cxOff, xR, t),
      y: bez(yc, top, top, yc, t) + bowOffset(t, bow),
    });
  }
  for (let i = 0; i < SEGS; i++) {
    const t = i / SEGS;
    pts.push({
      x: bez(xR, CX + cxOff, CX - cxOff, xL, t),
      y: bez(yc, bot, bot, yc, t) + full * gauss(t, 0.5, 0.16),
    });
  }
  return pts;
}

function ringPath(pts: Point[]): Path2D {
  const path = new Path2D();
  path.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) path.lineTo(pts[i].x, pts[i].y);
  path.closePath();
  return path;
}

function fillDot(ctx: CanvasRenderingContext2D, p: Point, size: number): void {
  ctx.fillRect(p.x - size / 2, p.y - size / 2, size, size);
}

/**
 * Wireframe mouth: outer + inner lip contour rings of square vertices joined
 * by thin lines, radial spokes between the rings, and wireframe teeth/tongue
 * contours clipped inside the inner ring. Same articulation semantics as the
 * classic Mouth (openness -> aperture, width -> spread, rounding -> pucker,
 * teeth/tongue fade thresholds), plus energy -> vertex glow/size and
 * pitch -> subtle corner-lift bias. Confidence is a per-hop evidence value
 * (it tracks loudness on real speech, mean ~0.15) shown in the meters only;
 * it does not modulate alpha. Canvas drawing
 * driven by the feed at display refresh rate; React never re-renders on
 * animation.
 */
export function WireframeMouth({ feed }: { feed: LipsyncFeed }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);
  const clockRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let lastBadge = "";

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      // Uniform logical scale: draw in the fixed 400x260 space regardless of
      // CSS size (aspect-ratio locks rect.height to the same factor).
      const k = rect.width / VIEW_W;
      ctx.setTransform(dpr * k, 0, 0, dpr * k, 0, 0);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    const styles = getComputedStyle(canvas);
    const color = (cssVar: string) => styles.getPropertyValue(cssVar).trim() || "#888";
    const lips = color("--lips");
    const vertex = color("--teeth");
    const glow = color("--energy");
    const tongue = color("--tongue");

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const now = feed.now();
      const s = feed.sample(now);

      // Geometry: keep in sync with Mouth.tsx (pitch smile bias is new here).
      const halfW = (44 + 62 * s.width) * (1 - 0.42 * s.rounding);
      const h = (6 + 118 * s.openness) * (1 + 0.32 * s.rounding);
      const cornerLift = Math.max(0, s.width - 0.4) * 14 + s.pitch * 6;
      const yCorner = CY - cornerLift;
      const yTop = CY - h * 0.54;
      const yBot = CY + h * 0.66;
      const lipT = (9.5 - 3 * s.openness + 3 * s.rounding) * 0.9;

      // Rounding relaxes the bow (puckered lips round out); the inner ring's
      // shaping fades as the mouth closes so the seam stays a clean line.
      const bowAmp = (3 + halfW * 0.05) * (1 - 0.55 * s.rounding);
      const closeScale = clamp01((h - 10) / 28);
      const outer = ringPoints(halfW + lipT, yTop - lipT, yBot + lipT, yCorner, bowAmp, 2.5);
      // Clamps keep the inner ring from inverting when the mouth closes; it
      // flattens to a near-line seam instead.
      const inner = ringPoints(
        Math.max(halfW - lipT, 6),
        Math.min(yTop + lipT, CY - 1),
        Math.max(yBot - lipT, CY + 1),
        yCorner,
        bowAmp * 0.6 * closeScale,
        1.5 * closeScale,
      );
      const innerPath = ringPath(inner);

      ctx.clearRect(0, 0, VIEW_W, VIEW_H);

      // Teeth: same fade conditions as the classic mouth, drawn as an
      // outlined band with per-tooth separators, clipped to the inner ring.
      const teethAlpha =
        clamp01((h - 26) / 12) * clamp01((0.55 - s.rounding) / 0.18);
      if (teethAlpha > 0.01) {
        const tx0 = CX - halfW * 0.78;
        const tw = halfW * 1.56;
        const ty = yTop + 2;
        const th = Math.min(16, h * 0.3);
        ctx.save();
        ctx.clip(innerPath);
        ctx.globalAlpha = teethAlpha * 0.8;
        ctx.strokeStyle = vertex;
        ctx.lineWidth = 1;
        ctx.strokeRect(tx0, ty, tw, th);
        ctx.beginPath();
        for (let i = 1; i < TEETH_COUNT; i++) {
          const x = tx0 + (tw * i) / TEETH_COUNT;
          ctx.moveTo(x, ty);
          ctx.lineTo(x, ty + th);
        }
        ctx.stroke();
        ctx.fillStyle = vertex;
        for (let i = 0; i <= TEETH_COUNT; i++) {
          fillDot(ctx, { x: tx0 + (tw * i) / TEETH_COUNT, y: ty + th }, DOT_DETAIL);
        }
        ctx.restore();
      }

      // Tongue: the classic ellipse as a sampled contour with a center
      // ridge, clipped to the inner ring, fading in on big apertures.
      const tongueAlpha = clamp01((h - 42) / 16);
      if (tongueAlpha > 0.01) {
        const tcy = CY + h * 0.34;
        const trx = halfW * 0.58;
        const trY = h * 0.26;
        ctx.save();
        ctx.clip(innerPath);
        ctx.globalAlpha = tongueAlpha * 0.9;
        ctx.strokeStyle = tongue;
        ctx.lineWidth = 1;
        ctx.beginPath();
        const tonguePts: Point[] = [];
        for (let i = 0; i < TONGUE_SEGS; i++) {
          const a = (i / TONGUE_SEGS) * Math.PI * 2;
          const p = { x: CX + trx * Math.cos(a), y: tcy + trY * Math.sin(a) };
          tonguePts.push(p);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.moveTo(CX, tcy - trY);
        ctx.lineTo(CX, tcy + trY);
        ctx.stroke();
        ctx.fillStyle = tongue;
        for (let i = 0; i < TONGUE_SEGS; i += 2) fillDot(ctx, tonguePts[i], DOT_DETAIL);
        ctx.restore();
      }

      // Spokes under the rings.
      ctx.strokeStyle = lips;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      for (let i = 0; i < outer.length; i++) {
        ctx.moveTo(outer[i].x, outer[i].y);
        ctx.lineTo(inner[i].x, inner[i].y);
      }
      ctx.stroke();

      // Ring contours.
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 1.2;
      ctx.stroke(ringPath(outer));
      ctx.lineWidth = 1;
      ctx.stroke(innerPath);

      // Square vertices on top; energy drives glow and size.
      ctx.globalAlpha = 1;
      ctx.fillStyle = vertex;
      ctx.shadowColor = glow;
      ctx.shadowBlur = 14 * s.energy;
      const grow = 1 + 0.35 * s.energy;
      for (const p of outer) fillDot(ctx, p, DOT_OUTER * grow);
      for (const p of inner) fillDot(ctx, p, DOT_INNER * grow);
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;

      // Event badge + playhead clock (plain DOM, no React churn).
      const kinds = feed.activeEventKinds(now);
      const badge =
        (["closure", "nasal", "silence"] as const).find((k) => kinds.has(k)) ?? "";
      if (badge !== lastBadge && badgeRef.current) {
        lastBadge = badge;
        badgeRef.current.textContent = badge ? EVENT_LABELS[badge] : "";
        badgeRef.current.dataset.kind = badge;
      }
      if (clockRef.current) {
        clockRef.current.textContent =
          s.relTime === null ? "idle" : `t=${s.relTime.toFixed(2)}s${s.live ? "" : " (rest)"}`;
      }
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, [feed]);

  return (
    <div className="mouth-card">
      <div className="mouth-badge" ref={badgeRef} />
      <canvas ref={canvasRef} className="mouth-canvas" />
      <div className="mouth-clock" ref={clockRef} />
    </div>
  );
}
