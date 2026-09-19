import type { Clip } from "../eval/recording";
import { Stat } from "./Stat";

const LAG_WARN_MS = 100; // noticeably out of sync

function ms(value: number | null): string {
  return value === null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(0)}ms`;
}

/** Analysis and delivery numbers for the selected clip. */
export function ClipStats({ clip }: { clip?: Clip }) {
  if (!clip) return null;
  const { stats } = clip;
  return (
    <div className="stats">
      <div className="stats-row">
        <Stat label="keyframes" value={`${stats.keyframes} (${stats.keyframesPerSec.toFixed(0)}/s)`} />
        <Stat label="closures" value={String(stats.events.closure)} />
        <Stat label="nasals" value={String(stats.events.nasal)} />
        <Stat label="silences" value={String(stats.events.silence)} />
        <Stat
          label="start lag"
          value={ms(stats.startLagMs)}
          warn={Math.abs(stats.startLagMs ?? 0) > LAG_WARN_MS}
          title="How late a live client's mouth starts: the utterance anchor implied by the first batch (exact but for network with wire version 2; version-1 clients assumed a fixed lead)"
        />
        <Stat
          label="settled"
          value={stats.settledAt === null ? "—" : `${ms(stats.settledLagMs)} @${stats.settledAt.toFixed(2)}s`}
          warn={Math.abs(stats.settledLagMs ?? 0) > LAG_WARN_MS}
          title="The anchor once later batches have pulled it in, and when that happened"
        />
        <Stat
          label="late batches"
          value={`${stats.lateBatches}/${stats.messages}`}
          warn={stats.lateBatches > 1}
          title="Batches that reached the client after their audio had started playing"
        />
        <Stat
          label="max late release"
          value={`${stats.maxHoldMs.toFixed(0)}ms`}
          warn={stats.maxHoldMs > 50}
          title="Longest a batch left the server after its scheduled time (before wire version 2 batches could sit in the transport's clock queue behind a word-timestamp frame)"
        />
        <Stat
          label="clamped"
          value={String(stats.releaseClamped)}
          title="Batches released as soon as they were emitted because their lead had already passed: the first windows of a turn, whose audio has to be analyzed first"
        />
      </div>
    </div>
  );
}
