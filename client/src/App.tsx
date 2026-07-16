import { PipecatClient } from "@pipecat-ai/client-js";
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
import { MouthCard } from "./components/MouthCard";
import { StatsBar } from "./components/StatsBar";
import { Timeline } from "./components/Timeline";
import { TranscriptCard } from "./components/TranscriptCard";
import { LipsyncFeed } from "./lipsync/feed";
import { parseLipsyncData } from "./lipsync/protocol";

interface Session {
  client: PipecatClient;
  feed: LipsyncFeed;
}

function createSession(): Session {
  const feed = new LipsyncFeed();
  const client = new PipecatClient({
    transport: new SmallWebRTCTransport(),
    enableMic: true,
    enableCam: false,
    callbacks: {
      // Lipsync batches arrive as standard RTVI server-messages;
      // parseLipsyncData demuxes on data.type === "bot-tts-lipsync".
      onServerMessage: (data: unknown) => {
        const batch = parseLipsyncData(data);
        if (batch) feed.ingest(batch);
      },
    },
  });
  // Dev-only hook for driving the mouth with synthetic batches from the console.
  if (import.meta.env.DEV) Object.assign(window, { lipsyncFeed: feed });
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
            <TranscriptCard />
            <MouthCard feed={feed} />
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
