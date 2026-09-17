# Viseme client

Vite + React web client for the lipsync bot in `../server`. It connects over
SmallWebRTC with the stock `PipecatClient`, receives lipsync batches as RTVI
`server-message`s (`data.type === "bot-tts-lipsync"`), and renders an animated
mouth with timing and event inspectors and a word-synced transcript.

```bash
npm install
npm run dev      # http://localhost:5173 (proxies /api to the bot on :7860)
npm run build    # tsc -b && vite build
npm run lint     # oxlint
```

Start the bot first (`cd ../server && uv run bot.py`).

## Layout

| Path                              | Purpose                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `src/App.tsx`                     | Client setup (`onServerMessage` → `parseLipsyncData` → `LipsyncFeed`) and page layout |
| `src/lipsync/protocol.ts`         | Wire format of the lipsync payload and its parser                                     |
| `src/lipsync/feed.ts`             | Buffers batches, anchors them on the wall clock, interpolates the mouth pose          |
| `src/components/`                 | Mouth, timeline, meters, event log, stats bar, transcript card                        |
| `src/hooks/useKaraokeTranscript.ts` | Word-level spoken progress from the SDK's `bot-output` events                       |

In development `window.lipsyncFeed` exposes the feed so synthetic batches can be
fed from the console.
