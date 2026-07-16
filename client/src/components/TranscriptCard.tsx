import { Fragment } from "react";

import { useKaraokeTranscript } from "../hooks/useKaraokeTranscript";

/**
 * Karaoke transcript of the bot utterance currently being spoken: the full
 * sentence appears as soon as the bot responds, with the not-yet-spoken
 * portion muted, brightening word by word in sync with audio playback.
 */
export function TranscriptCard() {
  const { segments } = useKaraokeTranscript();
  const empty = segments.every((s) => s.spoken === "" && s.unspoken === "");
  return (
    <div className="transcript">
      <div className="panel-title">Transcript</div>
      {empty ? (
        <div className="transcript-empty">
          Connect and the bot&rsquo;s words will appear here.
        </div>
      ) : (
        <p className="transcript-text">
          {segments.map((s, i) => (
            <Fragment key={i}>
              {s.needsSeparator && " "}
              {s.spoken}
              {s.unspoken && <span className="transcript-unspoken">{s.unspoken}</span>}
            </Fragment>
          ))}
        </p>
      )}
    </div>
  );
}
