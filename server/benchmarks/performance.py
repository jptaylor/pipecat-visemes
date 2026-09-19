"""Analyzer-only CPU and allocation benchmark, usable against a clean old checkout.

Run this file by absolute path with --source-path pointing to a server directory.
Audio conversion, imports, fixture I/O and dictionary cold start are outside the
steady-state CPU measurement. Context construction, text snapshots, analysis and
flush are inside it. Use separate processes and alternating mode order; never
run CPU comparisons concurrently. No Praat or debug feature collection.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
import wave
from pathlib import Path
from types import SimpleNamespace


async def run(args):
    # Select BEFORE any lipsync import: the DSP control really can be main.
    sys.path.insert(0, str(args.source_path.resolve()))
    import numpy as np
    from pipecat.audio.utils import create_stream_resampler

    from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
    from lipsync.formant_lipsync_analyzer import EVENT_FINALIZE_HORIZON_SEC, FormantLipsyncAnalyzer

    enabled = args.mode != "dsp"
    inputs_type = None
    if args.mode == "text":
        from benchmarks.text_timing import FixtureTextInputs

        inputs_type = FixtureTextInputs

    def factory():
        return FormantLipsyncAnalyzer(**({"text_events_enabled": True} if enabled else {}))

    start = time.perf_counter_ns()
    probe = factory()
    await probe.start(24_000)
    cold_plain_ms = (time.perf_counter_ns() - start) / 1e6
    del probe
    if enabled:
        from lipsync.pronunciation import load_lexicon

        load_lexicon.cache_clear()
    # Cold shared allocation: tracemalloc is OFF for steady-state timings.
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter_ns()
    probe = factory()
    await probe.start(24_000)
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    gc.collect()
    retained, cold_peak = tracemalloc.get_traced_memory()
    start_bytes = retained
    instances = [factory() for _ in range(16)]
    for instance in instances:
        await instance.start(24_000)
    gc.collect()
    instance_bytes = (tracemalloc.get_traced_memory()[0] - start_bytes) / len(instances)
    tracemalloc.stop()
    del instances, instance, probe

    prepared = []
    corpus = hashlib.sha256()
    for path in sorted(args.fixtures.glob("*/*/*/take-*.wav")):
        if path.parent.name.startswith("control-"):
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        with wave.open(str(path), "rb") as wav:
            rate, pcm = wav.getframerate(), wav.readframes(wav.getnframes())
        corpus.update(str(path.relative_to(args.fixtures)).encode())
        corpus.update(pcm)
        clip = SimpleNamespace(
            pcm=pcm,
            sample_rate=rate,
            sentence=SimpleNamespace(text=meta["text"]),
            text_timing=meta.get("text_timing"),
        )
        resampler, chunks = create_stream_resampler(), []
        size = round(rate * 0.02) * 2
        for i in range(0, len(pcm), size):
            converted = await resampler.resample(pcm[i : i + size], rate, 16_000)
            if converted:
                chunks.append(
                    (
                        np.frombuffer(converted, dtype=np.int16).astype(np.float32) / 32768,
                        min(i + size, len(pcm)) // 2,
                    )
                )
        prepared.append((clip, chunks))
    if not prepared:
        raise SystemExit("no fixtures found")

    async def once():
        cpu, hops, keyframes, events = 0, 0, 0, 0
        call_us = []
        for clip, chunks in prepared:
            analyzer = factory()
            await analyzer.start(clip.sample_rate)
            begin = time.process_time_ns()
            context = LipsyncAnalysisContext("performance", clip.sample_rate)
            inputs = inputs_type(clip, context) if inputs_type else None
            for pcm, seen in chunks:
                call_start = time.process_time_ns()
                if inputs:
                    inputs.ingest(seen)
                result = await analyzer.analyze(pcm, context)
                call_us.append((time.process_time_ns() - call_start) / 1000)
                keyframes += len(result.keyframes)
                events += len(result.events)
            if inputs:
                inputs.ingest(len(clip.pcm) // 2, final=True)
            result = await analyzer.flush(context)
            cpu += time.process_time_ns() - begin
            hops += round((result.processed_up_to - EVENT_FINALIZE_HORIZON_SEC) / 0.02)
            keyframes += len(result.keyframes)
            events += len(result.events)
        return {
            "cpu_seconds": cpu / 1e9,
            "us_per_20ms_hop": cpu / 1000 / hops,
            "call_p50_us": float(np.percentile(call_us, 50)),
            "call_p95_us": float(np.percentile(call_us, 95)),
            "call_p99_us": float(np.percentile(call_us, 99)),
            "rtf": cpu / 1e9 / sum(len(c.pcm) / 2 / c.sample_rate for c, _ in prepared),
            "hops": hops,
            "keyframes": keyframes,
            "events": events,
        }

    await once()  # Warm imports, FFT paths and the bounded pronunciation cache.
    trials = [await once() for _ in range(args.repeats)]
    payload = {
        "mode": args.mode,
        "source": str(args.source_path),
        "lipsync_sha256": source_digest(args.source_path / "lipsync"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "clips": len(prepared),
        "pcm_corpus_sha256": corpus.hexdigest(),
        "audio_seconds": sum(len(c.pcm) / 2 / c.sample_rate for c, _ in prepared),
        "cold_start_with_tracemalloc_ms": cold_ms,
        "cold_start_ms": cold_plain_ms,
        "cold_retained_bytes": retained,
        "cold_peak_bytes": cold_peak,
        "additional_idle_instance_bytes": instance_bytes,
        "max_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "median_us_per_hop": statistics.median(t["us_per_20ms_hop"] for t in trials),
        "median_rtf": statistics.median(t["rtf"] for t in trials),
        "trials": trials,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in {".py", ".gz", ".json"}):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--mode", choices=("dsp", "no-text", "text"), default="dsp")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
