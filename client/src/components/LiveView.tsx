import type { LipsyncFeed } from "../lipsync/feed";
import { EventLog } from "./EventLog";
import { Meters } from "./Meters";
import { MouthCard } from "./MouthCard";
import { StatsBar } from "./StatsBar";
import { Timeline } from "./Timeline";
import { TranscriptCard } from "./TranscriptCard";

/** The live tab: the connected bot's lipsync, as it arrives. */
export function LiveView({ feed }: { feed: LipsyncFeed }) {
  return (
    <>
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
    </>
  );
}
