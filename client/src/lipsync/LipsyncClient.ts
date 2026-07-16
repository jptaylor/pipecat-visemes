import {
  PipecatClient,
  RTVIMessage,
  type PipecatClientOptions,
} from "@pipecat-ai/client-js";

import {
  LIPSYNC_MESSAGE_TYPE,
  parseLipsyncData,
  type LipsyncBatch,
} from "./protocol";

/**
 * PipecatClient subclass that intercepts the custom `bot-tts-lipsync` server
 * message. The stock dispatcher drops unrecognized message types, so until
 * the client libraries know about lipsync we hook `handleMessage` directly.
 */
export class LipsyncClient extends PipecatClient {
  onLipsyncBatch?: (batch: LipsyncBatch) => void;

  constructor(options: PipecatClientOptions) {
    super(options);
  }

  protected override handleMessage(ev: RTVIMessage): void {
    if (ev.type === LIPSYNC_MESSAGE_TYPE) {
      const batch = parseLipsyncData(ev.data);
      if (batch) this.onLipsyncBatch?.(batch);
      return;
    }
    super.handleMessage(ev);
  }
}
