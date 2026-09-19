import { useEffect, useState } from "react";

import type { FeedStats, LipsyncFeed } from "../lipsync/feed";
import { Stat } from "./Stat";

/** Message/keyframe rates, scheduler health, and the A/V trim control. */
export function StatsBar({ feed }: { feed: LipsyncFeed }) {
  const [stats, setStats] = useState<FeedStats | null>(null);
  const [trim, setTrim] = useState(0);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => setStats(feed.statsSnapshot(feed.now())), 500);
    return () => clearInterval(timer);
  }, [feed]);

  const onTrim = (value: number) => {
    setTrim(value);
    feed.offsetTrimMs = value;
  };

  return (
    <div className="stats">
      <div className="stats-row">
        <Stat label="msg/s" value={stats ? stats.msgsPerSec.toFixed(1) : "—"} />
        <Stat label="kf/s" value={stats ? stats.kfsPerSec.toFixed(1) : "—"} />
        <Stat label="messages" value={stats ? String(stats.totalMessages) : "—"} />
        <Stat label="keyframes" value={stats ? String(stats.totalKeyframes) : "—"} />
        <Stat label="events" value={stats ? String(stats.totalEvents) : "—"} />
        <Stat
          label="lead"
          value={stats?.lastLeadMs != null ? `${stats.lastLeadMs.toFixed(0)}ms` : "—"}
          title="How far ahead of its window's playout the last batch arrived (the server aims for 200ms; the first batches of a turn are late by design, as their audio has to be analyzed first)"
        />
        <Stat label="resyncs" value={stats ? String(stats.resyncs) : "—"} />
        <Stat label="ctx" value={stats?.ctx ? stats.ctx.slice(0, 8) : "—"} />
        <label
          className="trim"
          title="Delays (+) or advances (−) the mouth against the audio. The server track is zero-phase (within ~10 ms of the audio), so 0 is the right default; use this for your own playout path."
        >
          A/V trim {trim >= 0 ? "+" : ""}
          {trim}ms
          <input
            type="range"
            min={-300}
            max={300}
            step={10}
            value={trim}
            onChange={(e) => onTrim(Number(e.target.value))}
          />
        </label>
        <button className="link-button" onClick={() => setShowRaw((v) => !v)}>
          {showRaw ? "hide raw" : "raw message"}
        </button>
      </div>
      {showRaw && (
        <pre className="raw-message">
          {feed.lastBatch ? JSON.stringify(feed.lastBatch.raw, null, 1) : "No messages yet."}
        </pre>
      )}
    </div>
  );
}
