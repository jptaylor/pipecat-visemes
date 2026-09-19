import { useEffect, useRef } from "react";

import type { Clip } from "../eval/recording";
import type { EvalSession } from "../eval/session";

/** The run's clips; click one to play it. */
export function ClipList({
  session,
  clips,
  index,
}: {
  session: EvalSession;
  clips: Clip[];
  index: number;
}) {
  const listRef = useRef<HTMLUListElement>(null);

  // Keep the selected clip in view as play-all or the arrow keys move on,
  // scrolling only the list (scrollIntoView would scroll the page too).
  useEffect(() => {
    const list = listRef.current;
    const row = list?.querySelector<HTMLElement>(".active");
    if (!list || !row) return;
    if (row.offsetTop < list.scrollTop) {
      list.scrollTop = row.offsetTop;
    } else if (row.offsetTop + row.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTop = row.offsetTop + row.offsetHeight - list.clientHeight;
    }
  }, [index]);

  return (
    <div className="clip-list">
      <div className="panel-title">Clips · {clips.length}</div>
      <ul ref={listRef}>
        {clips.map((clip, i) => {
          const lag = clip.stats.startLagMs;
          return (
            <li
              key={clip.example.id}
              className={i === index ? "active" : undefined}
              onClick={() => session.select(i)}
            >
              <span className="clip-id">{clip.example.id}</span>
              <span className="clip-text">{clip.example.text}</span>
              <span className="clip-meta">{clip.example.duration.toFixed(1)}s</span>
              <span
                className={lag !== null && lag > 100 ? "clip-meta clip-lag-warn" : "clip-meta"}
                title="Start lag as delivered"
              >
                {lag === null ? "—" : `${lag >= 0 ? "+" : ""}${lag.toFixed(0)}ms`}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
