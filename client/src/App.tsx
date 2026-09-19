import { PipecatClient } from "@pipecat-ai/client-js";
import {
  PipecatClientAudio,
  PipecatClientProvider,
} from "@pipecat-ai/client-react";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { useState, type ComponentProps } from "react";

import "./App.css";
import { ConnectBar } from "./components/ConnectBar";
import { EvalView } from "./components/EvalView";
import { LiveView } from "./components/LiveView";
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
      // Fires at once on barge-in (the server drops the unplayed audio and
      // the batches not yet sent) and as the last audio leaves at a natural
      // turn end; the feed keeps a short grace window either way.
      onBotStoppedSpeaking: () => feed.cut(),
    },
  });
  return { client, feed };
}

// client-react's rolled-up d.ts inlines its own PipecatClient declaration,
// nominally incompatible with the real class (protected members). Same class
// at runtime; cast at the provider boundary only.
type ProviderClient = ComponentProps<typeof PipecatClientProvider>["client"];

type Tab = "live" | "eval";

const TABS: { id: Tab; label: string; title: string }[] = [
  { id: "live", label: "Live", title: "Talk to the bot" },
  { id: "eval", label: "Eval", title: "Play back recorded clips, no bot needed" },
];

export default function App() {
  const [{ client, feed }] = useState(createSession);
  // The tab lives in the URL hash so a reload (or a bookmark) keeps it.
  const [tab, setTab] = useState<Tab>(() => (window.location.hash === "#eval" ? "eval" : "live"));
  // Dev-only hook for driving the mouth with synthetic batches from the
  // console. Assigned here rather than in createSession so it always points
  // at the session React kept (StrictMode runs the initializer twice).
  if (import.meta.env.DEV) Object.assign(window, { lipsyncFeed: feed });

  const selectTab = (next: Tab) => {
    setTab(next);
    const { pathname, search } = window.location;
    window.history.replaceState(null, "", next === "eval" ? "#eval" : pathname + search);
  };

  return (
    <PipecatClientProvider client={client as unknown as ProviderClient} autoInitDevices>
      <div className="app">
        <header className="app-header">
          <h1>
            Pipecat <span className="accent">Lipsync</span>
          </h1>
          <nav className="tabs">
            {TABS.map(({ id, label, title }) => (
              <button
                key={id}
                className={tab === id ? "tab tab-active" : "tab"}
                title={title}
                onClick={() => selectTab(id)}
              >
                {label}
              </button>
            ))}
          </nav>
          {tab === "live" && <ConnectBar feed={feed} />}
        </header>
        {tab === "live" ? <LiveView feed={feed} /> : <EvalView />}
      </div>
      <PipecatClientAudio />
    </PipecatClientProvider>
  );
}
