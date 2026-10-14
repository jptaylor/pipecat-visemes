/**
 * TypeScript runtime for pipecat-visemes web clients: turns the bot's `bot-tts-lipsync`
 * server-messages into a mouth pose at any instant. Rendering is the app's:
 * sample the feed once per animation frame and draw.
 */

export { lipsyncCallbacks } from "./callbacks";
export {
  LipsyncFeed,
  REST_POSE,
  SCHEDULING_LEAD_SEC,
  type ArticulationSample,
  type FeedStats,
  type LoggedEvent,
} from "./feed";
export {
  LIPSYNC_MESSAGE_TYPE,
  parseLipsyncData,
  type LipsyncBatch,
  type LipsyncEvent,
  type LipsyncEventKind,
  type LipsyncKeyframe,
} from "./protocol";
