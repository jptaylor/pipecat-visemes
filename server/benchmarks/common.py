"""Shared harness plumbing: corpus, TTS fixture cache, analyzer ingest mirror.

Used by the accuracy benchmark and the eval recorder (``benchmarks.record``).
Lives in the test harness only — never upstream. See
plans/benchmark-harness-accuracy.md.
"""

import ast
import asyncio
import hashlib
import json
import os
import subprocess
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from lipsync import dsp, formant_lipsync_analyzer
from lipsync.dsp import ANALYSIS_SAMPLE_RATE

load_dotenv(override=True)

BENCH_DIR = Path(__file__).parent
FIXTURES_DIR = BENCH_DIR / "fixtures"
RESULTS_DIR = BENCH_DIR / "results"
CORPUS_PATH = BENCH_DIR / "corpus.yaml"

SYNTHESIS_TIMEOUT_SECS = 60
TEARDOWN_TIMEOUT_SECS = 3

# Every expectation key the corpus may use (validated at load so typos fail
# loudly). Grouped for composite scoring: closure / nasal / silence. Vowel
# probes use p90 peak-reaching checks (sentence means measure dilution, not
# fidelity), with bars oracle-calibrated from the Praat reference — see
# `--calibrate-expectations`.
EXPECT_KEYS = {
    "closures_min",
    "closures_max",
    "nasals_min",
    "nasals_max",
    "nasal_fraction_max",
    "silences_min",
    "silences_max",
    "openness_p90_max",
    "openness_p90_min",
    "width_p90_min",
    "width_p90_max",
    "rounding_p90_min",
}


@dataclass
class Sentence:
    id: str
    text: str
    tags: list[str]
    expect: dict[str, float]


@dataclass
class Voice:
    provider: str
    id: str
    formant_ceiling: float


@dataclass
class Clip:
    """One cached synthesis: PCM plus its provenance."""

    sentence: Sentence
    voice: Voice
    take: int
    pcm: bytes  # int16 mono
    sample_rate: int

    @property
    def label(self) -> str:
        return f"{self.sentence.id}/{self.voice.id[:8]}/t{self.take}"

    @property
    def duration_secs(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


def load_corpus() -> tuple[list[Sentence], dict[str, list[Voice]]]:
    """Load and validate corpus.yaml; unknown expectation keys fail loudly."""
    data = yaml.safe_load(CORPUS_PATH.read_text())
    sentences = []
    for entry in data["sentences"]:
        expect = entry.get("expect", {})
        unknown = set(expect) - EXPECT_KEYS
        if unknown:
            raise ValueError(f"corpus.yaml sentence {entry['id']}: unknown expect keys {unknown}")
        sentences.append(
            Sentence(
                id=entry["id"],
                text=entry["text"],
                tags=entry.get("tags", []),
                expect=expect,
            )
        )
    voices = {
        provider: [
            Voice(provider=provider, id=v["id"], formant_ceiling=v.get("formant_ceiling", 5500))
            for v in entries
        ]
        for provider, entries in data["voices"].items()
    }
    return sentences, voices


#
# TTS synthesis (headless mini-pipeline) and fixture cache
#


class _AudioCollector(FrameProcessor):
    """Collects one utterance's TTSAudioRawFrames; signals on TTSStoppedFrame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunks: list[bytes] = []
        self.sample_rate = 0
        self.done = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.chunks.append(frame.audio)
            self.sample_rate = frame.sample_rate
        elif isinstance(frame, TTSStoppedFrame):
            self.done.set()
        await self.push_frame(frame, direction)


def make_tts(provider: str, voice_id: str):
    if provider == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService

        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaTTSService.Settings(voice=voice_id),
        )
    if provider == "deepgram":
        from pipecat.services.deepgram.tts import DeepgramTTSService

        return DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice=voice_id)
    raise ValueError(f"unknown provider: {provider}")


_teardowns: set[asyncio.Task] = set()


def teardowns_pending() -> bool:
    """Whether a detached TTS teardown is still running (see ``synthesize``)."""
    return bool(_teardowns)


def _detach_teardown(worker, run_task: asyncio.Task):
    async def teardown():
        try:
            await worker.cancel()
            await run_task
        except BaseException:
            pass

    task = asyncio.create_task(teardown())
    _teardowns.add(task)
    task.add_done_callback(_teardowns.discard)


async def synthesize(provider: str, voice_id: str, text: str) -> tuple[bytes, int]:
    """Synthesize one utterance through a real pipecat TTS service.

    Returns (int16 mono PCM, sample rate).
    """
    tts = make_tts(provider, voice_id)
    collector = _AudioCollector()
    pipeline = Pipeline([tts, collector])
    worker = PipelineWorker(pipeline, params=PipelineParams(), enable_rtvi=False)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    run_task = asyncio.create_task(runner.run())
    try:
        await worker.queue_frames([TTSSpeakFrame(text)])
        await asyncio.wait_for(collector.done.wait(), timeout=SYNTHESIS_TIMEOUT_SECS)
        # The audio is complete at TTSStoppedFrame; teardown is best-effort and
        # never awaited past a short grace (DeepgramTTSService on pipecat 1.10
        # does not drain the EndFrame and its cancel takes ~30 s) — a slow
        # teardown finishes detached while the next clip synthesizes.
        await worker.queue_frames([EndFrame()])
        try:
            await asyncio.wait_for(asyncio.shield(run_task), timeout=TEARDOWN_TIMEOUT_SECS)
        except TimeoutError:
            _detach_teardown(worker, run_task)
    except BaseException:
        run_task.cancel()
        raise
    if not collector.chunks:
        raise RuntimeError(f"TTS produced no audio for: {text!r}")
    return b"".join(collector.chunks), collector.sample_rate


def _fixture_paths(sentence: Sentence, voice: Voice, take: int) -> tuple[Path, Path]:
    base = FIXTURES_DIR / voice.provider / voice.id / sentence.id
    return base / f"take-{take}.wav", base / f"take-{take}.json"


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def write_wav(path: Path, pcm: bytes, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


async def get_clip(
    sentence: Sentence, voice: Voice, take: int, *, offline: bool, refresh: bool
) -> Clip:
    """Fetch one clip, cache-first. Sidecar text mismatch invalidates the cache."""
    wav_path, meta_path = _fixture_paths(sentence, voice, take)

    if wav_path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text())
        if meta.get("text") == sentence.text:
            pcm, sample_rate = read_wav(wav_path)
            return Clip(sentence, voice, take, pcm, sample_rate)
        print(f"  stale fixture (text changed): {wav_path}")

    if offline:
        raise FileNotFoundError(f"--offline but no fixture for {sentence.id} take {take}")

    print(f"  synthesizing {voice.provider}/{voice.id[:12]}… {sentence.id} take {take}")
    pcm, sample_rate = await synthesize(voice.provider, voice.id, sentence.text)
    write_wav(wav_path, pcm, sample_rate)
    meta_path.write_text(
        json.dumps(
            {
                "text": sentence.text,
                "sample_rate": sample_rate,
                "provider": voice.provider,
                "voice": voice.id,
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    return Clip(sentence, voice, take, pcm, sample_rate)


#
# Tunable overrides and provenance
#

# Modules whose constants ``--set`` may override for A/B runs.
_OVERRIDE_MODULES = {"dsp": dsp, "analyzer": formant_lipsync_analyzer}


def apply_overrides(specs: list[str]) -> dict[str, object]:
    """Apply ``module.CONSTANT=value`` overrides before any analyzer is built.

    Equivalent to editing the constant in the source: the lipsync modules read
    their tunables at call time (nothing binds them at import). Unknown names
    fail loudly so a typo cannot silently A/B nothing.
    """
    applied: dict[str, object] = {}
    for spec in specs:
        target, _, raw = spec.partition("=")
        module_name, _, name = target.partition(".")
        module = _OVERRIDE_MODULES.get(module_name)
        if module is None or not raw or not hasattr(module, name):
            raise SystemExit(
                f"bad --set {spec!r}: expected {{dsp,analyzer}}.EXISTING_CONSTANT=value"
            )
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                value = float(raw)  # "inf", "nan"
            except ValueError:
                value = raw
        setattr(module, name, value)
        applied[target] = value
    return applied


def src_sha() -> str:
    """Short git sha of the checkout (the lipsync code lives in this repo)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def lipsync_digest() -> str:
    """Fingerprint runtime sources, including uncommitted/untracked Python files.

    Git HEAD alone cannot distinguish two experiments in the same worktree.
    Hash names as well as content, in a stable order, ignoring bytecode.
    """
    root = Path(dsp.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


#
# Analyzer ingest mirror
#


async def chunks_16k_float32(
    pcm: bytes, sample_rate: int, chunk_ms: int = 20
) -> AsyncIterator[np.ndarray]:
    """Yield 16 kHz float32 chunks exactly as LipsyncProcessor ingests audio.

    Same path as the processor: per-clip stream resampler on int16 bytes,
    then float32 conversion.
    """
    resampler = create_stream_resampler()
    chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2
    for i in range(0, len(pcm), chunk_bytes):
        resampled = await resampler.resample(
            pcm[i : i + chunk_bytes], sample_rate, ANALYSIS_SAMPLE_RATE
        )
        if resampled:
            samples = np.frombuffer(resampled, dtype=np.int16).astype(np.float32)
            samples /= 32768.0
            yield samples
