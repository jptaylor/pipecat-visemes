import { useState } from "react";

import type { LipsyncFeed } from "../lipsync/feed";
import { Mouth } from "./Mouth";
import { WireframeMouth } from "./WireframeMouth";

/**
 * Hosts the active mouth visual and a toggle between the wireframe (default)
 * and classic SVG renderers. Only the active visual is mounted; both share
 * the same feed, so switching mid-utterance is seamless.
 */
export function MouthCard({ feed }: { feed: LipsyncFeed }) {
  const [classic, setClassic] = useState(false);
  return (
    <div className="mouth-switcher">
      {classic ? <Mouth feed={feed} /> : <WireframeMouth feed={feed} />}
      <button className="mouth-toggle" onClick={() => setClassic((v) => !v)}>
        {classic ? "wire" : "classic"}
      </button>
    </div>
  );
}
