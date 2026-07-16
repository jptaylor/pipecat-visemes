import type {
  BotOutputText,
  ConversationMessagePart,
} from "@pipecat-ai/client-react";
import { usePipecatConversation } from "@pipecat-ai/client-react";

/** One karaoke segment: the spoken/unspoken split of a message part. */
export interface KaraokeSegment {
  spoken: string;
  unspoken: string;
  /** Render a leading space before this segment (RTVI 2.0 inter-segment separator). */
  needsSeparator: boolean;
}

export interface KaraokeTranscript {
  segments: KaraokeSegment[];
  /** True once the bot has finished speaking the utterance. */
  final: boolean;
}

function isBotOutputText(
  text: ConversationMessagePart["text"],
): text is BotOutputText {
  return (
    text !== null && typeof text === "object" && "spoken" in text && "unspoken" in text
  );
}

function toSegment(part: ConversationMessagePart): KaraokeSegment {
  const needsSeparator = part.needsSeparator ?? false;
  if (isBotOutputText(part.text)) {
    return { spoken: part.text.spoken, unspoken: part.text.unspoken, needsSeparator };
  }
  return {
    spoken: typeof part.text === "string" ? part.text : "",
    unspoken: "",
    needsSeparator,
  };
}

const EMPTY: KaraokeTranscript = { segments: [], final: false };

/**
 * Karaoke split of the bot utterance currently being spoken (the latest
 * assistant message). The spoken/unspoken boundary comes from the server's
 * bot-output progress events, which are released at audio presentation time,
 * so it advances word by word in sync with playback.
 */
export function useKaraokeTranscript(): KaraokeTranscript {
  const { messages } = usePipecatConversation();
  const current = messages.findLast((m) => m.role === "assistant");
  if (!current) return EMPTY;
  return {
    segments: current.parts.map(toSegment),
    final: current.final ?? false,
  };
}
