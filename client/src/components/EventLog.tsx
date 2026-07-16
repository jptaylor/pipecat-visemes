import { useSyncExternalStore } from "react";

import type { LipsyncFeed, LoggedEvent } from "../lipsync/feed";

const MAX_ROWS = 30;

function shortCtx(ctx: string | null): string {
  return ctx ? ctx.slice(0, 8) : "—";
}

/** Feed of discrete lipsync events (closure / nasal / silence). */
export function EventLog({ feed }: { feed: LipsyncFeed }) {
  const version = useSyncExternalStore(
    (cb) => feed.subscribe(cb),
    () => feed.eventLog.length + (feed.eventLog[feed.eventLog.length - 1]?.seq ?? 0),
  );
  void version;

  const rows: LoggedEvent[] = feed.eventLog.slice(-MAX_ROWS).reverse();
  return (
    <div className="event-log">
      <div className="panel-title">Events</div>
      {rows.length === 0 ? (
        <div className="event-empty">No events yet — connect and let the bot speak.</div>
      ) : (
        <ul>
          {rows.map((e) => (
            <li key={`${e.ctx}-${e.seq}`}>
              <span className={`event-chip event-${e.kind}`}>{e.kind}</span>
              <span className="event-time">+{e.offset.toFixed(2)}s</span>
              <span className="event-detail">
                {e.duration > 0 ? `${e.duration.toFixed(2)}s` : "instant"} · conf{" "}
                {e.confidence.toFixed(2)} · ctx {shortCtx(e.ctx)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
