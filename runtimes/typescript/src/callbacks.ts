import type { LipsyncFeed } from "./feed";
import { parseLipsyncData } from "./protocol";

/**
 * The two `PipecatClient` callbacks that drive a feed: lipsync batches arrive
 * as RTVI server-messages (`data.type === "bot-tts-lipsync"`; any other server
 * message is ignored), and bot-stopped-speaking cuts the utterance at the
 * playhead.
 *
 *     const feed = new LipsyncFeed();
 *     const client = new PipecatClient({ transport, callbacks: lipsyncCallbacks(feed) });
 *
 * An app with callbacks of its own spreads these in, and one that handles
 * server messages too calls `onServerMessage` from its own handler.
 */
export function lipsyncCallbacks(feed: LipsyncFeed) {
  return {
    onServerMessage: (data: unknown) => {
      const batch = parseLipsyncData(data);
      if (batch) feed.ingest(batch);
    },
    // Fires at once on barge-in (the server drops the unplayed audio and the
    // batches not yet sent) and as the last audio leaves at a natural turn
    // end; the feed keeps a short grace window either way.
    onBotStoppedSpeaking: () => feed.cut(),
  };
}
