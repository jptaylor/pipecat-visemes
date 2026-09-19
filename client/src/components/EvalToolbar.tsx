import type { EvalSession, EvalState } from "../eval/session";

/** Recording and run pickers, plus reload for freshly recorded data. */
export function EvalToolbar({ session, state }: { session: EvalSession; state: EvalState }) {
  const { recordings, recording, run, status } = state;
  return (
    <div className="eval-toolbar">
      <label title="One pass of the eval corpus through the bot's TTS and lipsync pipeline">
        Recording
        <select
          value={recording?.id ?? ""}
          onChange={(e) => void session.selectRecording(e.target.value)}
        >
          {recordings.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label} · {r.examples} clips
            </option>
          ))}
        </select>
      </label>
      <label title="The lipsync analysis of that audio: the live pass, or a --reanalyze with newer code">
        Run
        <select value={run?.id ?? ""} onChange={(e) => void session.selectRun(e.target.value)}>
          {recording?.runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label}
            </option>
          ))}
        </select>
      </label>
      {status === "loading" && <span className="eval-loading">loading…</span>}
      <button className="link-button" onClick={() => void session.reload()}>
        reload
      </button>
      <span className="eval-keys">space play · ↑↓ clip · ←→ step 20 ms</span>
    </div>
  );
}
