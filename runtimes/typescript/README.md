# @pipecat-visemes/typescript

The TypeScript runtime for web clients; the example client (`client/`) is built on it. It turns the bot's
`bot-tts-lipsync` RTVI server-messages into a mouth pose at any instant, anchored on the wall
clock from each batch's lead; drawing the mouth is left to the app.

| File | Purpose |
| --- | --- |
| `src/protocol.ts` | The wire format (version 2) and `parseLipsyncData` |
| `src/feed.ts` | `LipsyncFeed`: buffers batches, anchors them on the wall clock, interpolates the pose, holds and eases to rest, cuts on barge-in |
| `src/callbacks.ts` | `lipsyncCallbacks(feed)`: the `PipecatClient` callbacks that drive a feed |

```ts
import { LipsyncFeed, lipsyncCallbacks } from "@pipecat-visemes/typescript";

const feed = new LipsyncFeed();
const client = new PipecatClient({ transport, callbacks: lipsyncCallbacks(feed) });

// Every animation frame:
const pose = feed.sample(feed.now()); // openness, width, rounding, energy, pitch in 0..1
```

`client/src/components/Mouth.tsx` is a complete SVG mouth over it. Confidence is a
diagnostic: it tracks loudness, so never fade or scale the mouth by it.

Plain TypeScript with no dependencies, and not published: the client depends on it as
`"@pipecat-visemes/typescript": "file:../runtimes/typescript"`, and Vite and `tsc` compile its source
directly, with no build step.
