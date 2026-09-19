"""Record the eval corpus through the bot's lipsync pipeline for the client's Eval tab.

Each example in eval_corpus.yaml is spoken by the bot's TTS service and runs,
in real time, through the same output path as bot.py::

    TTS → LipsyncProcessor → output transport → LipsyncMessageRelay

The output transport is a headless stand-in for SmallWebRTC's: a simulated
audio device that plays each write in real time (a write returns once its
audio has been consumed), so lipsync batches leave the transport's clock queue
exactly when a live client would receive them. No WebRTC, STT or LLM is
involved: the corpus text is spoken verbatim, one TTS context per example, all
in one session (the analyzer's adaptive state carries across examples as it
does in a call).

Per example the recording keeps:

- the audio as played (WAV; silence wherever playout stalled),
- every lipsync server-message with its release time and the time it was due
  (its batch's scheduled release), in seconds from the first played sample,
- word timings (TTSTextFrame release times) when the TTS has word timestamps,
- the TTS output timeline (sentence anchors, audio chunks, and word-timestamp frames, which
  go through the transport's clock queue like they do live), so
  ``--reanalyze`` can replay the retained audio through the current lipsync
  code (no TTS calls) with the same timing.

Layout (static files, served by the client's Vite dev server)::

    client/public/eval/
      index.json              recordings, newest first
      <recording>/recording.json   examples, audio paths, word timings, runs
      <recording>/audio/<id>.wav   audio as played
      <recording>/arrivals.json    TTS arrival timelines (replay input)
      <recording>/runs/<run>.json  lipsync messages + processor stats per example

Run (from server/):
    uv run python -m benchmarks.record                     # new recording (CARTESIA_API_KEY)
    uv run python -m benchmarks.record --reanalyze         # new run over the newest recording
    uv run python -m benchmarks.record --reanalyze --set dsp.LPC_ORDER=14 --tag order14
    uv run python -m benchmarks.record --provider deepgram
    uv run python -m benchmarks.record --reanalyze --text-prior --tag text-observe

Text priors currently collect observations only; the formant analyzer remains
DSP-only. Old recordings without anchors can explicitly opt into
``--assume-early-text``; the run records that assumption. Recordings with an
empty anchors list preserve the observed absence of text.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger
from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frames import (
    AggregatedTextFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.time import nanoseconds_to_seconds, seconds_to_nanoseconds
from pipecat.workers.runner import WorkerRunner

import lipsync
from benchmarks.common import (
    BENCH_DIR,
    apply_overrides,
    lipsync_digest,
    load_corpus,
    make_tts,
    read_wav,
    src_sha,
    write_wav,
)
from lipsync.frames import TTSLipsyncFrame
from lipsync.lipsync_processor import LipsyncParams, LipsyncProcessor
from lipsync.rtvi import LIPSYNC_MESSAGE_TYPE, LipsyncMessageRelay

EVAL_CORPUS_PATH = BENCH_DIR / "eval_corpus.yaml"
EVAL_DIR = BENCH_DIR.parent.parent / "client" / "public" / "eval"
FORMAT_VERSION = 1

# A write returns this long before its audio finishes playing. SmallWebRTC's
# track hands audio out in 10 ms slices (a write returns when its last slice
# is taken); a little more slack keeps event-loop jitter (a burst of analysis)
# from reading as a playout stall.
_WRITE_LOOKAHEAD_SECS = 0.02
# After an example's audio has played out, wait this long for stragglers
# (a final batch released late) before starting the next example.
_SETTLE_SECS = 0.5
_READY_TIMEOUT_SECS = 30
_LIVE_TIMEOUT_SECS = 60
_TEARDOWN_TIMEOUT_SECS = 5
# How far ahead of its window the processor aims to release a batch. Wire
# version 2 carries the lead actually remaining; version-1 clients assumed
# exactly this value.
_LEAD_SECS = LipsyncParams().scheduling_lead_ms / 1000.0


@dataclass
class Example:
    id: str
    text: str
    tags: list[str]
    look_for: str


def load_examples() -> list[Example]:
    """Load eval_corpus.yaml; duplicate ids fail loudly."""
    data = yaml.safe_load(EVAL_CORPUS_PATH.read_text())
    examples = [
        Example(
            id=entry["id"],
            text=entry["text"],
            tags=entry.get("tags", []),
            look_for=entry.get("look_for", ""),
        )
        for entry in data["examples"]
    ]
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("eval_corpus.yaml: duplicate example ids")
    return examples


#
# Capture: taps around the lipsync path, and a paced headless transport
#


@dataclass
class _Capture:
    """Everything observed for one example, stamped with the pipeline clock (ns)."""

    example: Example
    ctx: str | None = None
    started_ns: int | None = None
    stopped_ns: int | None = None
    # TTS audio as it reached the lipsync processor.
    arrivals: list[tuple[int, bytes]] = field(default_factory=list)
    tts_sample_rate: int = 0
    # (arrived, pts, text) of each word-timestamp frame from the TTS.
    word_arrivals: list[tuple[int, int, str]] = field(default_factory=list)
    # (arrived, pts or None, text); anchors can precede TTSStartedFrame.
    anchor_arrivals: list[tuple[int, int | None, str]] = field(default_factory=list)
    # (play start, play end, bytes) for every write to the simulated device.
    played: list[tuple[int, int, bytes]] = field(default_factory=list)
    # [released, due, data]: due is the batch's scheduled release (already
    # clamped to its emission time if analysis ran late).
    messages: list[list] = field(default_factory=list)
    # A batch and its server-message reach the output tap in batch order, but
    # the message (a SystemFrame) may overtake its batch: pair them FIFO.
    unpaired_pts: list[int] = field(default_factory=list)
    unpaired_messages: list[list] = field(default_factory=list)
    words: list[tuple[int, str, bool]] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class _Recorder:
    """Routes frames observed around the lipsync path into the current capture.

    Frames are matched on the TTS context id, so stragglers from a previous
    example can never land in the next one.
    """

    def __init__(self):
        self.current: _Capture | None = None

    def begin(self, example: Example, ctx: str | None = None) -> _Capture:
        self.current = _Capture(example=example, ctx=ctx)
        return self.current

    def end(self):
        self.current = None

    def on_input(self, frame: Frame, now: int):
        """Frames entering the lipsync processor (straight from the TTS)."""
        capture = self.current
        if capture is None:
            return
        if (
            isinstance(frame, AggregatedTextFrame)
            and not isinstance(frame, TTSTextFrame)
            and frame.will_be_spoken
            and frame.aggregated_by == "sentence"
        ):
            if capture.ctx is None:
                capture.ctx = frame.context_id
            if frame.context_id == capture.ctx:
                capture.anchor_arrivals.append((now, frame.pts, frame.text))
        elif isinstance(frame, TTSStartedFrame):
            if capture.ctx is None:
                capture.ctx = frame.context_id
            if frame.context_id == capture.ctx and capture.started_ns is None:
                capture.started_ns = now
        elif isinstance(frame, TTSAudioRawFrame) and frame.context_id == capture.ctx:
            capture.arrivals.append((now, frame.audio))
            capture.tts_sample_rate = frame.sample_rate
        elif (
            isinstance(frame, TTSTextFrame)
            and frame.context_id == capture.ctx
            and frame.aggregated_by == "word"
            and frame.pts is not None
        ):
            capture.word_arrivals.append((now, frame.pts, frame.text))
        elif isinstance(frame, TTSStoppedFrame) and frame.context_id == capture.ctx:
            capture.stopped_ns = now

    def on_played(self, start: int, end: int, audio: bytes):
        """Audio handed to the simulated device, with its playout span."""
        if self.current is not None:
            self.current.played.append((start, end, audio))

    def on_output(self, frame: Frame, now: int):
        """Frames leaving the relay: what the RTVI observer would send a client."""
        capture = self.current
        if capture is None:
            return
        if isinstance(frame, TTSLipsyncFrame) and frame.context_id == capture.ctx:
            due = frame.release_ns or now
            if capture.unpaired_messages:
                capture.unpaired_messages.pop(0)[1] = due
            else:
                capture.unpaired_pts.append(due)
        elif isinstance(frame, RTVIServerMessageFrame):
            data = frame.data
            if (
                isinstance(data, dict)
                and data.get("type") == LIPSYNC_MESSAGE_TYPE
                and data.get("ctx") == capture.ctx
            ):
                message = [now, None, data]
                capture.messages.append(message)
                if capture.unpaired_pts:
                    message[1] = capture.unpaired_pts.pop(0)
                else:
                    capture.unpaired_messages.append(message)
        elif isinstance(frame, TTSTextFrame) and frame.context_id == capture.ctx:
            capture.words.append((now, frame.text, frame.pts is not None))
        elif isinstance(frame, TTSStoppedFrame) and frame.context_id == capture.ctx:
            # The transport forwards TTSStoppedFrame once all preceding audio
            # has been written, i.e. once the example has played out.
            capture.done.set()


class _Tap(FrameProcessor):
    """Passes every frame through, reporting downstream ones with the clock time."""

    def __init__(self, on_frame: Callable[[Frame, int], None], **kwargs):
        super().__init__(**kwargs)
        self._on_frame = on_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self._on_frame(frame, self.get_clock().get_time())
        await self.push_frame(frame, direction)


class _PlayoutTransport(BaseOutputTransport):
    """Headless output transport with a simulated real-time audio device.

    Mirrors SmallWebRTC's output track: a write returns once its audio has
    been taken for playback, so the transport's audio queue drains at playback
    speed, and audio arriving after the device ran dry plays on arrival.
    Timed frames (lipsync batches, word timestamps) go through the real clock
    queue.
    """

    def __init__(self, on_played: Callable[[int, int, bytes], None]):
        super().__init__(TransportParams(audio_out_enabled=True, audio_out_end_silence_secs=0))
        self._on_played = on_played
        self._play_end = 0
        self.ready = asyncio.Event()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        self.ready.set()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        clock = self.get_clock()
        start = max(clock.get_time(), self._play_end)
        self._play_end = start + seconds_to_nanoseconds(frame.num_frames / frame.sample_rate)
        self._on_played(start, self._play_end, frame.audio)
        wait = nanoseconds_to_seconds(self._play_end - clock.get_time()) - _WRITE_LOOKAHEAD_SECS
        if wait > 0:
            await asyncio.sleep(wait)
        return True


#
# Takes: turning a capture into files
#


@dataclass
class _Take:
    example: Example
    pcm: bytes  # audio as played, int16 mono
    sample_rate: int
    words: list[list]
    word_timestamps: bool
    messages: list[dict]
    stats: dict[str, int]
    arrival: dict

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


def _strip_gaps(played: bytes, gaps: list[list[int]], size: int) -> bytes:
    """Recover the TTS audio from the as-played audio (drop stall silence and padding)."""
    out = bytearray()
    pos = 0
    for offset, length in gaps:
        out += played[pos:offset]
        pos = offset + length
    out += played[pos:]
    return bytes(out[:size])


def _finish(capture: _Capture, sample_rate: int, stats: dict[str, int]) -> _Take:
    if not capture.played:
        raise RuntimeError("no audio was played")
    if capture.started_ns is None or capture.stopped_ns is None:
        raise RuntimeError("TTSStartedFrame/TTSStoppedFrame not seen")
    if capture.tts_sample_rate != sample_rate:
        raise RuntimeError(
            f"TTS audio is {capture.tts_sample_rate} Hz but the transport plays "
            f"{sample_rate} Hz; the retained audio would not replay bit-exact"
        )

    # Audio as played: device writes on the playout timeline, with silence
    # wherever the device ran dry before more audio arrived.
    start = capture.played[0][0]
    pcm = bytearray()
    gaps: list[list[int]] = []
    end = start
    for play_start, play_end, audio in capture.played:
        if play_start > end:
            silence = round((play_start - end) * sample_rate / 1e9) * 2
            gaps.append([len(pcm), silence])
            pcm += bytes(silence)
        pcm += audio
        end = play_end

    arrived = b"".join(audio for _, audio in capture.arrivals)
    if _strip_gaps(bytes(pcm), gaps, len(arrived)) != arrived:
        raise RuntimeError("played audio does not match the TTS audio")

    def since_start(ns: int) -> float:
        return round((ns - start) / 1e9, 4)

    def since_started(ns: int) -> float:
        return round((ns - capture.started_ns) / 1e9, 4)

    return _Take(
        example=capture.example,
        pcm=bytes(pcm),
        sample_rate=sample_rate,
        words=[[since_start(ns), text] for ns, text, _ in capture.words],
        word_timestamps=bool(capture.words) and all(timed for _, _, timed in capture.words),
        messages=[
            {"at": since_start(ns), "due": since_start(ns if due is None else due), "data": data}
            for ns, due, data in capture.messages
        ],
        stats=stats,
        arrival={
            "ctx": capture.ctx,
            "sample_rate": capture.tts_sample_rate,
            "chunks": [[since_started(ns), len(audio)] for ns, audio in capture.arrivals],
            # Presence matters: [] means observed NO anchors; older files
            # without this key did not record them at all.
            "anchors": [
                [since_started(ns), since_started(pts) if pts is not None else None, text]
                for ns, pts, text in capture.anchor_arrivals
            ],
            "words": [
                [since_started(ns), since_started(pts), text]
                for ns, pts, text in capture.word_arrivals
            ],
            "stopped": since_started(capture.stopped_ns),
            "gaps": gaps,
        },
    )


def _start_lag_ms(messages: list[dict]) -> float | None:
    """How late a live client's mouth starts: the first batch's implied anchor.

    With wire version 2 the client anchors on the window start and the lead
    stamped into the message, so the error is network transit only (zero
    here). A version-1 client assumed every batch arrived exactly
    ``_LEAD_SECS`` before its first offset, so a batch released late (analysis
    still catching up) made its anchor late by this much.
    """
    if not messages:
        return None
    at, data = messages[0]["at"], messages[0]["data"]
    t0 = data.get("t0", 0.0)
    if "lead" in data and "ws" in data:
        return (at + data["lead"] - (t0 + data["ws"])) * 1000.0
    offsets = [row[0] for row in data["kf"]] + [row[0] for row in data["ev"]]
    if not offsets:
        return None
    return (at + _LEAD_SECS - (t0 + min(offsets))) * 1000.0


def _summary(take: _Take) -> str:
    keyframes = sum(len(m["data"]["kf"]) for m in take.messages)
    kinds = [row[1] for m in take.messages for row in m["data"]["ev"]]
    held = max((m["at"] - m["due"] for m in take.messages), default=0.0)
    lag = _start_lag_ms(take.messages)
    return (
        f"{take.duration:5.2f} s  {len(take.messages):3} msgs  {keyframes:4} kf  "
        f"closure {kinds.count('closure'):2}  nasal {kinds.count('nasal'):2}  "
        f"silence {kinds.count('silence'):2}  start lag "
        + (f"{lag:+4.0f} ms" if lag is not None else "   n/a")
        + f"  max late release {held * 1000:3.0f} ms"
    )


#
# Driving the pipeline
#


def _replay_frames(
    arrival: dict, pcm: bytes, base_ns: int, *, assumed_text: str | None = None
) -> list[tuple[float, Frame]]:
    """Restore text, audio and lifecycle frames without inventing word timing.

    ``assumed_text`` is an explicit legacy-recording experiment. It inserts
    an untimestamped anchor at context start only when anchors were never
    recorded, not when a recording observed none (TOKEN/no-text providers).
    """
    ctx = arrival["ctx"]
    sample_rate = arrival["sample_rate"]
    frames: list[tuple[float, Frame]] = []
    anchors = arrival.get("anchors", [])
    if "anchors" not in arrival and assumed_text:
        anchors = [[0.0, None, assumed_text]]
    for t, pts, text in anchors:
        anchor = AggregatedTextFrame(text, aggregated_by="sentence", context_id=ctx)
        anchor.will_be_spoken = True
        anchor.append_to_context = False
        anchor.pts = base_ns + seconds_to_nanoseconds(pts) if pts is not None else None
        frames.append((t, anchor))
    # An anchor rounded to t=0 still precedes start (stable sort). Late
    # TOKEN-mode anchors retain their position after the relevant audio.
    frames.append((0.0, TTSStartedFrame(context_id=ctx)))
    offset = 0
    for t, size in arrival["chunks"]:
        audio = pcm[offset : offset + size]
        offset += size
        frames.append(
            (t, TTSAudioRawFrame(audio, sample_rate=sample_rate, num_channels=1, context_id=ctx))
        )
    for t, pts, text in arrival.get("words", []):
        word = TTSTextFrame(text, aggregated_by="word", context_id=ctx)
        word.pts = base_ns + seconds_to_nanoseconds(pts)
        frames.append((t, word))
    frames.append((arrival["stopped"], TTSStoppedFrame(context_id=ctx)))
    frames.sort(key=lambda item: item[0])
    return frames


async def _replay(
    worker: PipelineWorker,
    clock: BaseClock,
    arrival: dict,
    pcm: bytes,
    *,
    assumed_text: str | None = None,
):
    """Replay the retained arrival timeline, including pre-start/late anchors.

    Shift all PTS by the same clock origin. Never use sentence PTS as the
    word-clock origin: those baselines need not be the same.
    """
    loop = asyncio.get_running_loop()
    first = min([0.0] + [a[0] for a in arrival.get("anchors", [])])
    base = loop.time() - first
    base_ns = clock.get_time() - seconds_to_nanoseconds(first)
    frames = _replay_frames(arrival, pcm, base_ns, assumed_text=assumed_text)

    for t, frame in frames:
        delay = base + t - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        await worker.queue_frames([frame])


@dataclass
class _Source:
    """Where example audio comes from: a live TTS service, or retained takes."""

    tts: FrameProcessor | None = None
    arrivals: dict[str, dict] = field(default_factory=dict)
    pcm: dict[str, bytes] = field(default_factory=dict)
    assume_early_text: bool = False


async def _run_examples(
    examples: list[Example], source: _Source, sample_rate: int, *, text_prior: bool = False
) -> tuple[list[_Take], bool]:
    """Speak every example through one pipeline session.

    Returns the takes and whether the pipeline shut down cleanly.
    """
    recorder = _Recorder()
    lipsync_processor = LipsyncProcessor(params=LipsyncParams(text_prior_enabled=text_prior))
    transport = _PlayoutTransport(recorder.on_played)
    processors = [
        _Tap(recorder.on_input),
        lipsync_processor,
        transport,
        LipsyncMessageRelay(),
        _Tap(recorder.on_output),
    ]
    if source.tts is not None:
        processors.insert(0, source.tts)
    worker = PipelineWorker(
        Pipeline(processors),
        params=PipelineParams(audio_out_sample_rate=sample_rate),
        enable_rtvi=False,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    run_task = asyncio.create_task(runner.run())

    takes: list[_Take] = []
    try:
        await asyncio.wait_for(transport.ready.wait(), timeout=_READY_TIMEOUT_SECS)
        await asyncio.sleep(_SETTLE_SECS)
        for n, example in enumerate(examples, 1):
            label = f"  [{n:2}/{len(examples)}] {example.id:14}"
            arrival = source.arrivals.get(example.id)
            capture = recorder.begin(example, ctx=arrival["ctx"] if arrival else None)
            before = lipsync_processor.stats
            feeder = None
            if arrival is not None:
                feeder = asyncio.create_task(
                    _replay(
                        worker,
                        transport.get_clock(),
                        arrival,
                        source.pcm[example.id],
                        assumed_text=example.text if source.assume_early_text else None,
                    )
                )
                prestart = -min([0.0] + [a[0] for a in arrival.get("anchors", [])])
                timeout = (
                    prestart
                    + arrival["stopped"]
                    + len(source.pcm[example.id]) / 2 / sample_rate
                    + 15
                )
            else:
                await worker.queue_frames([TTSSpeakFrame(example.text)])
                timeout = _LIVE_TIMEOUT_SECS
            try:
                await asyncio.wait_for(capture.done.wait(), timeout=timeout)
                if feeder is not None:
                    await feeder
            except TimeoutError:
                recorder.end()
                if feeder is not None:
                    feeder.cancel()
                print(f"{label} timed out waiting for playout; skipped")
                continue
            await asyncio.sleep(_SETTLE_SECS)
            recorder.end()
            after = lipsync_processor.stats
            stats = {key: value - before.get(key, 0) for key, value in after.items()}
            try:
                take = _finish(capture, sample_rate, stats)
            except RuntimeError as e:
                print(f"{label} {e}; skipped")
                continue
            print(f"{label} {_summary(take)}")
            takes.append(take)
    finally:
        clean = await _shutdown(worker, run_task)
    return takes, clean


async def _shutdown(worker: PipelineWorker, run_task: asyncio.Task) -> bool:
    """Stop the pipeline; False if it would not stop (see common.synthesize)."""
    if run_task.done():
        return True
    try:
        await worker.queue_frames([EndFrame()])
        await asyncio.wait_for(asyncio.shield(run_task), timeout=_TEARDOWN_TIMEOUT_SECS)
        return True
    except TimeoutError:
        pass
    try:
        await asyncio.wait_for(worker.cancel(), timeout=_TEARDOWN_TIMEOUT_SECS)
        await asyncio.wait_for(asyncio.shield(run_task), timeout=_TEARDOWN_TIMEOUT_SECS)
        return True
    except TimeoutError:
        return False


#
# Files
#


def _lipsync_dirty() -> bool:
    """Whether the lipsync package differs from HEAD (e.g. an uncommitted DSP change)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", str(Path(lipsync.__file__).parent)],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return bool(out.strip())
    except Exception:
        return False


def _write_json(path: Path, payload: dict, *, compact: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(payload, separators=(",", ":")))
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_index(out_dir: Path):
    """Rebuild index.json from the recordings on disk, newest first."""
    recordings = []
    for path in out_dir.glob("*/recording.json"):
        rec = json.loads(path.read_text())
        recordings.append(
            {
                "id": rec["id"],
                "label": rec["label"],
                "provider": rec["provider"],
                "voice": rec["voice"],
                "created_at": rec["created_at"],
                "examples": len(rec["examples"]),
                "runs": len(rec["runs"]),
            }
        )
    recordings.sort(key=lambda r: r["created_at"], reverse=True)
    _write_json(out_dir / "index.json", {"version": FORMAT_VERSION, "recordings": recordings})


def _find_recording(out_dir: Path, which: str) -> Path:
    if which != "latest":
        path = out_dir / which
        if not (path / "recording.json").exists():
            raise SystemExit(f"no recording at {path}")
        return path
    candidates = [json.loads(p.read_text()) for p in out_dir.glob("*/recording.json")]
    if not candidates:
        raise SystemExit(f"no recordings under {out_dir}; record one first (drop --reanalyze)")
    newest = max(candidates, key=lambda rec: rec["created_at"])
    return out_dir / newest["id"]


def _run_payload(
    run_id: str,
    mode: str,
    takes: list[_Take],
    overrides: dict,
    *,
    text_prior: bool = False,
    assumed_text_examples: list[str] | None = None,
) -> dict:
    sha, dirty = src_sha(), _lipsync_dirty()
    label = " · ".join(
        [run_id, sha + ("+dirty" if dirty else ""), "text: observe" if text_prior else "text: off"]
        + (["assumed early text"] if assumed_text_examples else [])
        + [f"{k}={v}" for k, v in overrides.items()]
    )
    return {
        "version": FORMAT_VERSION,
        "id": run_id,
        "label": label,
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": {"sha": sha, "dirty": dirty},
        "lipsync_sha256": lipsync_digest(),
        "text_prior": text_prior,
        "assumed_text_examples": assumed_text_examples or [],
        "overrides": overrides,
        "examples": {
            take.example.id: {"messages": take.messages, "stats": take.stats} for take in takes
        },
    }


def _run_summary(run: dict) -> dict:
    return {
        "id": run["id"],
        "label": run["label"],
        "mode": run["mode"],
        "created_at": run["created_at"],
        "file": f"runs/{run['id']}.json",
    }


def _default_voice(provider: str) -> str:
    # The bot reads CARTESIA_VOICE_ID and falls back to the quickstart voice,
    # which is also corpus.yaml's first Cartesia voice.
    if provider == "cartesia" and os.getenv("CARTESIA_VOICE_ID"):
        return os.environ["CARTESIA_VOICE_ID"]
    _, voices = load_corpus()
    return voices[provider][0].id


def _select(examples: list[Example], wanted: str | None) -> list[Example]:
    if not wanted:
        return examples
    ids = wanted.split(",")
    missing = set(ids) - {e.id for e in examples}
    if missing:
        raise SystemExit(f"unknown example ids: {', '.join(sorted(missing))}")
    return [e for e in examples if e.id in ids]


async def record_new(args, overrides: dict) -> bool:
    """Record a new set of takes through the live TTS service."""
    examples = _select(load_examples(), args.examples)
    voice = args.voice or _default_voice(args.provider)
    sample_rate = PipelineParams().audio_out_sample_rate
    print(f"recording {len(examples)} examples: {args.provider} voice {voice}")
    takes, clean = await _run_examples(
        examples,
        _Source(tts=make_tts(args.provider, voice)),
        sample_rate,
        text_prior=args.text_prior,
    )
    if not takes:
        raise SystemExit("nothing recorded")

    voice_tag = voice[:8] if re.fullmatch(r"[0-9a-f]{8}-.*", voice) else voice
    now = datetime.now()
    rec_id = f"{args.provider}-{voice_tag}-{now.strftime('%Y%m%d-%H%M%S')}"
    rec_dir = args.out / rec_id
    for take in takes:
        write_wav(rec_dir / "audio" / f"{take.example.id}.wav", take.pcm, take.sample_rate)
    _write_json(
        rec_dir / "arrivals.json",
        {
            "version": FORMAT_VERSION,
            "examples": {take.example.id: take.arrival for take in takes},
        },
        compact=True,
    )
    run = _run_payload(args.tag or "live", "live", takes, overrides, text_prior=args.text_prior)
    _write_json(rec_dir / "runs" / f"{run['id']}.json", run, compact=True)
    _write_json(
        rec_dir / "recording.json",
        {
            "version": FORMAT_VERSION,
            "id": rec_id,
            "label": f"{args.provider.title()} {voice_tag} · {now.strftime('%Y-%m-%d %H:%M')}",
            "provider": args.provider,
            "voice": voice,
            "sample_rate": sample_rate,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "examples": [
                {
                    "id": take.example.id,
                    "text": take.example.text,
                    "tags": take.example.tags,
                    "look_for": take.example.look_for,
                    "audio": f"audio/{take.example.id}.wav",
                    "duration": round(take.duration, 4),
                    "words": take.words,
                    "word_timestamps": take.word_timestamps,
                }
                for take in takes
            ],
            "runs": [_run_summary(run)],
        },
    )
    _write_index(args.out)
    print(f"\nrecording: {rec_dir}  (run {run['label']})")
    return clean


async def reanalyze(args, overrides: dict) -> bool:
    """Add a run: replay a recording's retained audio through the current lipsync code."""
    rec_dir = _find_recording(args.out, args.reanalyze)
    recording = json.loads((rec_dir / "recording.json").read_text())
    arrivals = json.loads((rec_dir / "arrivals.json").read_text())["examples"]
    examples = _select(
        [
            Example(id=e["id"], text=e["text"], tags=e["tags"], look_for=e["look_for"])
            for e in recording["examples"]
        ],
        args.examples,
    )
    run_id = args.tag or f"reanalyze-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if any(r["id"] == run_id and r["mode"] == "live" for r in recording["runs"]):
        raise SystemExit(f"--tag {run_id} would overwrite the recording's live run")
    sample_rate = recording["sample_rate"]
    source = _Source(arrivals=arrivals, assume_early_text=args.assume_early_text)
    assumed = [
        e.id
        for e in examples
        if args.assume_early_text and "anchors" not in arrivals[e.id] and e.text
    ]
    for entry in recording["examples"]:
        if any(e.id == entry["id"] for e in examples):
            played, _ = read_wav(rec_dir / entry["audio"])
            arrival = arrivals[entry["id"]]
            size = sum(chunk_size for _, chunk_size in arrival["chunks"])
            source.pcm[entry["id"]] = _strip_gaps(played, arrival["gaps"], size)

    print(f"reanalyzing {len(examples)} examples of {recording['id']}")
    if assumed:
        print(f"  assuming an early, untimestamped sentence anchor for {len(assumed)} legacy takes")
    takes, clean = await _run_examples(examples, source, sample_rate, text_prior=args.text_prior)
    if not takes:
        raise SystemExit("nothing analyzed")

    run = _run_payload(
        run_id,
        "reanalyze",
        takes,
        overrides,
        text_prior=args.text_prior,
        assumed_text_examples=assumed,
    )
    _write_json(rec_dir / "runs" / f"{run_id}.json", run, compact=True)
    recording["runs"] = [r for r in recording["runs"] if r["id"] != run_id] + [_run_summary(run)]
    _write_json(rec_dir / "recording.json", recording)
    _write_index(args.out)
    print(f"\nrun: {run['label']}  ({rec_dir.name})")
    return clean


def main():
    parser = argparse.ArgumentParser(description="Record the eval corpus for the client's Eval tab")
    parser.add_argument("--provider", default="cartesia", choices=["cartesia", "deepgram"])
    parser.add_argument(
        "--voice", help="voice id (default: the bot's Cartesia voice, or corpus.yaml's)"
    )
    parser.add_argument("--examples", help="comma-separated example ids (default: all)")
    parser.add_argument(
        "--reanalyze",
        nargs="?",
        const="latest",
        metavar="RECORDING",
        help="replay a recording's retained audio through the current lipsync code instead "
        "of calling the TTS (default: the newest recording)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="MODULE.CONST=VALUE",
        help="override a lipsync tunable for this run, e.g. dsp.LPC_ORDER=14",
    )
    parser.add_argument("--tag", help="run name (default: live, or reanalyze-<timestamp>)")
    parser.add_argument(
        "--text-prior",
        action="store_true",
        help="attach text observations to analyzer contexts (experimental; currently observation only)",
    )
    parser.add_argument(
        "--assume-early-text",
        action="store_true",
        help="replay only: supply an untimestamped sentence anchor from corpus text for legacy "
        "takes that did not record anchors; marks the run as an assumption",
    )
    parser.add_argument(
        "--out", type=Path, default=EVAL_DIR, help=f"output directory (default: {EVAL_DIR})"
    )
    args = parser.parse_args()
    if args.assume_early_text and not args.reanalyze:
        parser.error("--assume-early-text requires --reanalyze")

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    overrides = apply_overrides(args.set)
    if overrides:
        print("overrides: " + ", ".join(f"{k}={v}" for k, v in overrides.items()))

    # Not asyncio.run: closing the loop would wait on a TTS service that never
    # drains its EndFrame (DeepgramTTSService on pipecat 1.10).
    loop = asyncio.new_event_loop()
    job = reanalyze(args, overrides) if args.reanalyze else record_new(args, overrides)
    clean = loop.run_until_complete(job)
    if not clean:
        sys.stdout.flush()
        os._exit(0)
    loop.close()


if __name__ == "__main__":
    main()
