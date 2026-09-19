import type { TimingMode } from "../eval/replayFeed";
import type { EvalSession, EvalState } from "../eval/session";

const RATES = [1, 0.5, 0.25];

const MODES: { mode: TimingMode; label: string; title: string }[] = [
  {
    mode: "delivered",
    label: "as delivered",
    title:
      "Batches reach the mouth when the server released them, anchored the way the live " +
      "client anchors them: what you would see connected (minus network)",
  },
  {
    mode: "ideal",
    label: "ideal",
    title: "Every batch on time: the mouth is locked to the audio, showing the analysis alone",
  },
];

function Segmented<T extends string | number>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string; title?: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <span className="segmented">
      {options.map((o) => (
        <button
          key={String(o.value)}
          className={o.value === value ? "active" : undefined}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </span>
  );
}

/** Transport controls, playback rate, timing mode and A/V trim. */
export function PlaybackBar({ session, state }: { session: EvalSession; state: EvalState }) {
  const { playing, playAll, rate, mode, trimMs, index, clips, loaded } = state;
  return (
    <div className="playback">
      <div className="playback-row">
        <button onClick={() => session.prev()} disabled={index <= 0} title="Previous clip (↑)">
          ◀◀
        </button>
        <button
          className="play-button"
          onClick={() => session.toggle()}
          disabled={!loaded}
          title="Play / pause (space)"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          onClick={() => session.next()}
          disabled={index + 1 >= clips.length}
          title="Next clip (↓)"
        >
          ▶▶
        </button>
        <label className="play-all" title="Keep going through the rest of the clips">
          <input
            type="checkbox"
            checked={playAll}
            onChange={(e) => session.setPlayAll(e.target.checked)}
          />
          play all
        </label>
        <Segmented
          options={RATES.map((r) => ({ value: r, label: `${r}×`, title: "Playback rate" }))}
          value={rate}
          onChange={(r) => session.setRate(r)}
        />
      </div>
      <div className="playback-row">
        <span className="playback-label">timing</span>
        <Segmented
          options={MODES.map((m) => ({ value: m.mode, label: m.label, title: m.title }))}
          value={mode}
          onChange={(m) => session.setMode(m)}
        />
        <label
          className="trim"
          title="Delays (+) or advances (−) the mouth against the audio, e.g. to cancel Bluetooth output latency"
        >
          A/V trim {trimMs >= 0 ? "+" : ""}
          {trimMs}ms
          <input
            type="range"
            min={-300}
            max={300}
            step={10}
            value={trimMs}
            onChange={(e) => session.setTrim(Number(e.target.value))}
          />
        </label>
      </div>
    </div>
  );
}
