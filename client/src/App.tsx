import {
  PipecatClientAudio,
  PipecatClientProvider,
} from "@pipecat-ai/client-react";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { useState, type ComponentProps } from "react";

import "./App.css";
import { ConnectBar } from "./components/ConnectBar";
import { EventLog } from "./components/EventLog";
import { Meters } from "./components/Meters";
import { Mouth } from "./components/Mouth";
import { StatsBar } from "./components/StatsBar";
import { Timeline } from "./components/Timeline";
import { LipsyncFeed } from "./lipsync/feed";
import { LipsyncClient } from "./lipsync/LipsyncClient";

interface Session {
  client: LipsyncClient;
  feed: LipsyncFeed;
}

function createSession(): Session {
  const feed = new LipsyncFeed();
  const client = new LipsyncClient({
    transport: new SmallWebRTCTransport(),
    enableMic: true,
    enableCam: false,
  });
  client.onLipsyncBatch = (batch) => feed.ingest(batch);
  return { client, feed };
}

// client-react's rolled-up d.ts inlines its own PipecatClient declaration,
// nominally incompatible with the real class (protected members). Same class
// at runtime; cast at the provider boundary only.
type ProviderClient = ComponentProps<typeof PipecatClientProvider>["client"];

export default function App() {
  const [{ client, feed }] = useState(createSession);

  return (
    <PipecatClientProvider client={client as unknown as ProviderClient} autoInitDevices>
      <div className="app">
        <ConnectBar feed={feed} />
        <main className="grid">
          <section className="col">
            <Mouth feed={feed} />
            <Meters feed={feed} />
          </section>
          <section className="col">
            <Timeline feed={feed} />
            <EventLog feed={feed} />
          </section>
        </main>
        <StatsBar feed={feed} />
      </div>
      <PipecatClientAudio />
    </PipecatClientProvider>
  );
}
