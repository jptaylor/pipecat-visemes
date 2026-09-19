import type { Clip } from "../eval/recording";
import type { EvalSession } from "../eval/session";
import { useAnimationFrameValue } from "../hooks/useAnimationFrameValue";

/**
 * The selected clip's text, brightening word by word with the audio when the
 * TTS provided word timestamps, plus what to look for in it.
 */
export function ClipTranscript({ session, clip }: { session: EvalSession; clip?: Clip }) {
  const words = clip?.example.word_timestamps ? clip.example.words : [];
  const spoken = useAnimationFrameValue(() => {
    const t = session.position;
    let n = 0;
    while (n < words.length && words[n][0] <= t) n++;
    return n;
  });

  return (
    <div className="transcript">
      <div className="panel-title">
        {clip ? clip.example.id : "Transcript"}
        {clip?.example.tags.map((tag) => (
          <span key={tag} className="clip-tag">
            {tag}
          </span>
        ))}
      </div>
      {!clip ? (
        <div className="transcript-empty">Pick a clip.</div>
      ) : (
        <>
          <p className="transcript-text">
            {words.length > 0 ? (
              <>
                {words.slice(0, spoken).map(([, w]) => w).join(" ")}
                {spoken > 0 && spoken < words.length && " "}
                <span className="transcript-unspoken">
                  {words.slice(spoken).map(([, w]) => w).join(" ")}
                </span>
              </>
            ) : (
              clip.example.text
            )}
          </p>
          {clip.example.look_for && <p className="look-for">{clip.example.look_for}</p>}
        </>
      )}
    </div>
  );
}
