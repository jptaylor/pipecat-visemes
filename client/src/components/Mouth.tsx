import { useEffect, useRef } from "react";

import type { LipsyncFeed } from "../lipsync/feed";
import type { LipsyncEventKind } from "../lipsync/protocol";

const CX = 200;
const CY = 130;

const EVENT_LABELS: Record<LipsyncEventKind, string> = {
  closure: "CLOSURE",
  nasal: "NASAL",
  silence: "SILENCE",
};

/**
 * Parametric mouth driven by the feed at display refresh rate.
 *
 * openness -> vertical aperture, width -> corner spread, rounding -> pucker
 * (narrower + taller + rounder), energy -> glow. Confidence is a per-hop
 * evidence value (it tracks loudness on real speech, mean ~0.15) and is shown
 * in the meters only — it must not fade or flatten the mouth.
 * DOM updates go through refs so React never re-renders on animation.
 */
export function Mouth({ feed }: { feed: LipsyncFeed }) {
  const groupRef = useRef<SVGGElement>(null);
  const innerRef = useRef<SVGPathElement>(null);
  const lipsRef = useRef<SVGPathElement>(null);
  const clipRef = useRef<SVGPathElement>(null);
  const teethRef = useRef<SVGRectElement>(null);
  const tongueRef = useRef<SVGEllipseElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);
  const clockRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let raf = 0;
    let lastBadge = "";

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const now = feed.now();
      const s = feed.sample(now);

      // Geometry: rounding shrinks the corners and stretches the aperture.
      const halfW = (44 + 62 * s.width) * (1 - 0.42 * s.rounding);
      const h = (6 + 118 * s.openness) * (1 + 0.32 * s.rounding);
      const cornerLift = Math.max(0, s.width - 0.4) * 14;
      const yCorner = CY - cornerLift;
      const xL = CX - halfW;
      const xR = CX + halfW;
      const cxOff = halfW * 0.52;
      const yTop = CY - h * 0.54;
      const yBot = CY + h * 0.66;

      const path =
        `M ${xL.toFixed(1)} ${yCorner.toFixed(1)} ` +
        `C ${(CX - cxOff).toFixed(1)} ${yTop.toFixed(1)}, ${(CX + cxOff).toFixed(1)} ${yTop.toFixed(1)}, ${xR.toFixed(1)} ${yCorner.toFixed(1)} ` +
        `C ${(CX + cxOff).toFixed(1)} ${yBot.toFixed(1)}, ${(CX - cxOff).toFixed(1)} ${yBot.toFixed(1)}, ${xL.toFixed(1)} ${yCorner.toFixed(1)} Z`;

      innerRef.current?.setAttribute("d", path);
      clipRef.current?.setAttribute("d", path);
      lipsRef.current?.setAttribute("d", path);
      lipsRef.current?.setAttribute(
        "stroke-width",
        (9.5 - 3 * s.openness + 3 * s.rounding).toFixed(1),
      );

      // Teeth fade in on wide-open unrounded shapes; tongue on big apertures.
      if (teethRef.current) {
        const teethOn = h > 26 && s.rounding < 0.55;
        teethRef.current.setAttribute("opacity", teethOn ? "0.9" : "0");
        teethRef.current.setAttribute("x", (CX - halfW * 0.78).toFixed(1));
        teethRef.current.setAttribute("width", (halfW * 1.56).toFixed(1));
        teethRef.current.setAttribute("y", (yTop + 2).toFixed(1));
        teethRef.current.setAttribute("height", Math.min(16, h * 0.3).toFixed(1));
      }
      if (tongueRef.current) {
        tongueRef.current.setAttribute("opacity", h > 42 ? "0.85" : "0");
        tongueRef.current.setAttribute("cx", String(CX));
        tongueRef.current.setAttribute("cy", (CY + h * 0.34).toFixed(1));
        tongueRef.current.setAttribute("rx", (halfW * 0.58).toFixed(1));
        tongueRef.current.setAttribute("ry", (h * 0.26).toFixed(1));
      }

      if (groupRef.current) {
        groupRef.current.style.filter = `drop-shadow(0 0 ${(s.energy * 22).toFixed(0)}px var(--energy))`;
      }

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
    return () => cancelAnimationFrame(raf);
  }, [feed]);

  return (
    <div className="mouth-card">
      <div className="mouth-badge" ref={badgeRef} />
      <svg viewBox="0 0 400 260" className="mouth-svg">
        <defs>
          <clipPath id="mouth-clip">
            <path ref={clipRef} d="" />
          </clipPath>
        </defs>
        <g ref={groupRef}>
          <path ref={innerRef} d="" fill="var(--mouth-cavity)" />
          <g clipPath="url(#mouth-clip)">
            <rect ref={teethRef} fill="var(--teeth)" rx="3" opacity="0" />
            <ellipse ref={tongueRef} fill="var(--tongue)" opacity="0" />
          </g>
          <path
            ref={lipsRef}
            d=""
            fill="none"
            stroke="var(--lips)"
            strokeLinejoin="round"
          />
        </g>
      </svg>
      <div className="mouth-clock" ref={clockRef} />
    </div>
  );
}
