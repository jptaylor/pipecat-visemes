import { useEffect, useSyncExternalStore } from "react";

import { evalSession } from "../eval/session";
import { ClipList } from "./ClipList";
import { ClipStats } from "./ClipStats";
import { ClipTimeline } from "./ClipTimeline";
import { ClipTranscript } from "./ClipTranscript";
import { EvalToolbar } from "./EvalToolbar";
import { Meters } from "./Meters";
import { MouthCard } from "./MouthCard";
import { PlaybackBar } from "./PlaybackBar";

const RECORD_COMMAND = "uv --directory server run python -m benchmarks.record";

/**
 * Offline eval: plays clips recorded through the bot's TTS and lipsync
 * pipeline (`python -m benchmarks.record`) with their retained audio, driving
 * the same mouth and meters as the live tab — no bot or connection needed.
 */
export function EvalView() {
  const session = evalSession();
  const state = useSyncExternalStore(session.subscribe, session.getState);
  // Dev-only console handle, like window.lipsyncFeed on the live tab.
  if (import.meta.env.DEV) Object.assign(window, { evalSession: session });

  // Keyboard shortcuts while the tab is shown; leaving the tab stops playback
  // (the session itself, and so the selection, outlives the tab).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLElement && e.target.closest("input, select, textarea")) return;
      if (e.key === " ") session.toggle();
      else if (e.key === "ArrowUp") session.prev();
      else if (e.key === "ArrowDown") session.next();
      else if (e.key === "ArrowLeft") session.step(-1);
      else if (e.key === "ArrowRight") session.step(1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      session.stop();
    };
  }, [session]);

  if (state.status === "empty" || (state.status === "error" && !state.recording)) {
    return (
      <div className="eval-empty">
        {state.status === "error" ? (
          <p className="eval-error">{state.error}</p>
        ) : (
          <p>No recordings yet.</p>
        )}
        <p>
          Record the eval corpus through the bot&rsquo;s TTS and lipsync pipeline (about a
          minute; needs <code>CARTESIA_API_KEY</code>), then reload:
        </p>
        <pre>{RECORD_COMMAND}</pre>
        <button onClick={() => session.reload()}>Reload</button>
      </div>
    );
  }
  if (!state.recording) return <div className="eval-empty">Loading recordings…</div>;

  return (
    <>
      <EvalToolbar session={session} state={state} />
      {state.error && <div className="eval-error">{state.error}</div>}
      <main className="grid">
        <section className="col">
          <ClipTranscript session={session} clip={state.clips[state.index]} />
          <MouthCard feed={session.feed} />
          <Meters feed={session.feed} />
        </section>
        <section className="col">
          <PlaybackBar session={session} state={state} />
          <ClipTimeline session={session} />
          <ClipStats clip={state.clips[state.index]} />
          <ClipList session={session} clips={state.clips} index={state.index} />
        </section>
      </main>
    </>
  );
}
