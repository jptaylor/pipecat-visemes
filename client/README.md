# Viseme client

Vite + React web client for the lipsync bot in `../server`. The **Live** tab
connects over SmallWebRTC with the stock `PipecatClient`, receives lipsync
batches as RTVI `server-message`s (`data.type === "bot-tts-lipsync"`), and
renders an animated mouth with timing and event inspectors and a word-synced
transcript. The **Eval** tab (`#eval`) needs no bot: it replays clips recorded
by `python -m benchmarks.record` (served from `public/eval/`) with their audio,
through the same mouth — see the root README.

```bash
npm install
npm run dev      # http://localhost:5173 (proxies /api to the bot on :7860)
npm run build    # tsc -b && vite build
npm run lint     # oxlint
```

Start the bot first for the Live tab (`cd ../server && uv run bot.py`).

## Layout

| Path                                | Purpose                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------ |
| `src/App.tsx`                       | Client setup (`onServerMessage` → `parseLipsyncData` → `LipsyncFeed`), Live/Eval tabs      |
| `src/lipsync/protocol.ts`           | Wire format of the lipsync payload and its parser                                          |
| `src/lipsync/feed.ts`               | Buffers batches, anchors them on the wall clock, interpolates the mouth pose               |
| `src/eval/`                         | Eval recordings (format, loaders), `ReplayFeed` (a feed on the audio clock), `EvalSession` |
| `src/components/`                   | Mouth, timeline, meters, event log, stats bar, transcript card; the Eval tab's views       |
| `src/hooks/useKaraokeTranscript.ts` | Word-level spoken progress from the SDK's `bot-output` events                              |

In development `window.lipsyncFeed` exposes the live feed so synthetic batches
can be fed from the console, and `window.evalSession` the Eval tab's session.
