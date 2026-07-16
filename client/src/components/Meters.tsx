import { useEffect, useRef } from "react";

import type { ArticulationSample, LipsyncFeed } from "../lipsync/feed";

const PARAMS = [
  { key: "openness", label: "Openness", color: "var(--openness)" },
  { key: "width", label: "Width", color: "var(--width)" },
  { key: "rounding", label: "Rounding", color: "var(--rounding)" },
  { key: "energy", label: "Energy", color: "var(--energy)" },
  { key: "pitch", label: "Pitch", color: "var(--pitch)" },
  { key: "confidence", label: "Confidence", color: "var(--confidence)" },
] as const;

type ParamKey = (typeof PARAMS)[number]["key"];

/** Live bars for the six keyframe parameters, updated outside React. */
export function Meters({ feed }: { feed: LipsyncFeed }) {
  const barRefs = useRef<Partial<Record<ParamKey, HTMLDivElement>>>({});
  const valueRefs = useRef<Partial<Record<ParamKey, HTMLSpanElement>>>({});

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const s: ArticulationSample = feed.sample(feed.now());
      for (const { key } of PARAMS) {
        const v = Math.max(0, Math.min(1, s[key]));
        const bar = barRefs.current[key];
        const value = valueRefs.current[key];
        if (bar) bar.style.width = `${(v * 100).toFixed(1)}%`;
        if (value) value.textContent = s[key].toFixed(2);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [feed]);

  return (
    <div className="meters">
      {PARAMS.map(({ key, label, color }) => (
        <div className="meter" key={key}>
          <span className="meter-label">{label}</span>
          <div className="meter-track">
            <div
              className="meter-bar"
              style={{ background: color }}
              ref={(el) => {
                barRefs.current[key] = el ?? undefined;
              }}
            />
          </div>
          <span
            className="meter-value"
            ref={(el) => {
              valueRefs.current[key] = el ?? undefined;
            }}
          >
            0.00
          </span>
        </div>
      ))}
    </div>
  );
}
