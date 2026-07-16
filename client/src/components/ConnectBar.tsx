import {
  usePipecatClient,
  usePipecatClientMediaDevices,
  usePipecatClientTransportState,
} from "@pipecat-ai/client-react";
import { useState } from "react";

import type { LipsyncFeed } from "../lipsync/feed";

const OFFER_URL = "/api/offer"; // proxied to the Pipecat dev runner by Vite

/** Connection controls + transport state. */
export function ConnectBar({ feed }: { feed: LipsyncFeed }) {
  const client = usePipecatClient();
  const transportState = usePipecatClientTransportState();
  const { availableMics, selectedMic, updateMic } = usePipecatClientMediaDevices();
  const [error, setError] = useState<string | null>(null);
  const [micEnabled, setMicEnabled] = useState(true);

  const busy = ["initializing", "authenticating", "connecting"].includes(transportState);
  const connected = ["connected", "ready"].includes(transportState);

  const onConnect = async () => {
    if (!client) return;
    setError(null);
    try {
      await client.connect({ connectionUrl: OFFER_URL });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onDisconnect = async () => {
    if (!client) return;
    await client.disconnect();
    feed.reset();
  };

  const onToggleMic = () => {
    if (!client) return;
    const next = !micEnabled;
    setMicEnabled(next);
    client.enableMic(next);
  };

  return (
    <header className="connect-bar">
      <h1>
        Pipecat <span className="accent">Lipsync</span>
      </h1>
      <span className={`state state-${transportState}`}>{transportState}</span>
      {error && <span className="connect-error">{error}</span>}
      <div className="connect-actions">
        <select
          className="mic-select"
          value={selectedMic?.deviceId ?? ""}
          onChange={(e) => updateMic(e.target.value)}
          disabled={availableMics.length === 0}
          title="Microphone"
        >
          {availableMics.length === 0 ? (
            <option value="">no mics — grant permission</option>
          ) : (
            availableMics.map((mic) => (
              <option key={mic.deviceId} value={mic.deviceId}>
                {mic.label || `Mic ${mic.deviceId.slice(0, 8)}`}
              </option>
            ))
          )}
        </select>
        <label className="mic-toggle">
          <input type="checkbox" checked={micEnabled} onChange={onToggleMic} />
          mic
        </label>
        {connected ? (
          <button onClick={onDisconnect}>Disconnect</button>
        ) : (
          <button onClick={onConnect} disabled={busy || !client}>
            {busy ? "Connecting…" : "Connect"}
          </button>
        )}
      </div>
    </header>
  );
}
