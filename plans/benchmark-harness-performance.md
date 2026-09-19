# Benchmark Harness — Performance

**Status:** Not built; superseded (2026-09-19). What it was designed to measure is covered elsewhere: CPU cost is reported by the accuracy harness as µs per 20 ms hop and RTF (~185 µs, RTF ≈ 0.009, about 1 % of one core per speaking bot); delivery timing — release time against playout, first-batch lag, clock-queue holds, the `pts_clamped`/`bytes_dropped`/`playout_gaps` counters — is recorded by the eval recorder (`server/benchmarks/record.py`), which drives a real `BaseOutputTransport` with a real-time simulated audio device (the §4.2 idea, with honest pacing). Not measured anywhere: RSS per session and the `--sessions` leak fit; the differential-CPU design in §4.5 is the way to do that if it is ever needed. The text below also predates the standalone layout — `RTVIObserverParams(bot_lipsync_enabled=True)` and a `CapturingRTVIObserver` do not exist; the relay path is the design of record — and is kept as the design record only. See [README.md](README.md).
**Location:** would have been `server/benchmarks/performance.py`.
**Goal:** Measure the server-side path **TTS audio → `bot-tts-lipsync` RTVI message** —
latency at each stage, delivery-timing correctness, and CPU/RSS cost — so we know whether
the implementation is performing as designed and where it isn't. No client, no network:
everything terminates in-process at the exact point a message would leave the server.
**Companions:** [technical-specification.md](technical-specification.md) §1.1/§9/§11.4 (budgets), [pipecat-implementation.md](pipecat-implementation.md) (§2.1 clock-queue path being measured), [benchmark-harness-accuracy.md](benchmark-harness-accuracy.md) (shares `benchmarks/common.py` + fixtures).

## 1. Non-goals

- No client-side skew measurement (spec §11.3's loopback harness — later, upstream).
- No accuracy claims — signal quality is the accuracy harness's job.
- Not a load/concurrency test (one pipeline at a time; Pipecat Cloud-style multi-session
  density is out of scope).
- Not a profiler. When this harness says "too slow", reach for `py-spy`/cProfile manually;
  the harness's job is to *detect and localize*, cheaply and repeatably.

## 2. What "the server-side path" is (and where we tap it)

```
FixtureTTSSource ──TTSAudioRawFrame──▶ LipsyncProcessor ──TTSLipsyncFrame(pts)──▶ NullOutputTransport
      │                                     │                                        │ (real MediaSender
      │ t_audio (per chunk)                 │ t_emit                                 │  clock queue)
      ▼                                     ▼                                        ▼ t_release
   [PerfObserver records FramePushed.timestamp — already pipeline-clock based]       │
                                                                                     ▼
                                                              CapturingRTVIObserver → t_msg (message built;
                                                              would-be network write point — recorded, dropped)
```

Verified plumbing this relies on (already confirmed against `main` for the implementation
plan): the transport routes any pts frame into `MediaSender._clock_queue` and releases it
at `pts` via the pipeline clock; `FramePushed.timestamp` is pipeline-clock time, directly
comparable to `pts`; `SystemClock` is `time.monotonic_ns()`-based.

**Offline by default:** TTS audio comes from the accuracy harness's cached fixtures,
replayed by a source processor — deterministic, free, API-key-independent. `--live` swaps
in a real TTS service when provider streaming cadence matters (§7).

## 3. Prerequisites

- Implementation plan **M1–M4** complete (processor emitting, observer wired).
- Accuracy harness `common.py` + at least one synthesized fixture set (or `--live`).
- **Processor stats seam** (recorded in the implementation plan): a read-only
  `LipsyncProcessor.stats` dict — `{batches_emitted, keyframes_emitted, events_emitted,
  pts_clamped, bytes_dropped, contexts_opened, contexts_evicted}` — plain int counters,
  incremented where the work happens. The harness reads it post-run; `bytes_dropped > 0`
  is how backpressure shows up without log-scraping.
- Root dev dep: `psutil` (RSS/CPU sampling; cross-platform, unlike `resource.ru_maxrss`
  unit quirks). Everything else is stdlib (`resource`, `tracemalloc`, `time`, `argparse`).

## 4. Harness components (`benchmarks/performance.py`, ~350 LOC)

### 4.1 `FixtureTTSSource(FrameProcessor)`

Replays cached clips as a TTS service would:

- Per clip: `TTSStartedFrame(context_id)` → `TTSAudioRawFrame` chunks (20 ms, native rate,
  `context_id` set) → `TTSStoppedFrame(context_id)`; 500 ms gap between clips.
- Pacing modes:
  - `realtime` (default for timing metrics): sleep so audio is pushed at 1× — steady-state
    behavior, honest jitter/lead numbers.
  - `burst`: push each utterance as fast as the pipeline accepts — mimics real TTS
    (providers typically deliver much faster than real time) and stresses the ring-buffer
    cap + pts clamp path. Timing metrics are reported but flagged (lead is meaningless in
    burst; RTF and correctness counters are the point).
- `--barge-in S`: every S seconds mid-utterance, `await self.broadcast_interruption()`
  (the real interruption API), then start the next clip as a fresh context. Exercises
  buffer clearing, clock-task recreation, and generation-counter discards under load.

### 4.2 `NullOutputTransport(BaseOutputTransport)`

The real transport base class — real `MediaSender`, real audio pacing, real clock queue —
with the device writes stubbed:

```python
class NullOutputTransport(BaseOutputTransport):
    """Real BaseOutputTransport machinery; media goes nowhere."""
    async def write_audio_frame(self, frame) -> bool:  return True
    async def write_video_frame(self, frame) -> bool:  return True
```

`TransportParams(audio_out_enabled=True, audio_out_sample_rate=<fixture rate>)`. The
transport's own 10 ms-chunk pacing loop is what makes `t0`/pts semantics real, so this is
the honest measurement path (verify the exact minimal abstract surface when implementing —
the base class provides `start`/`stop`; only the two writes should need bodies).

### 4.3 `PerfObserver(BaseObserver)`

`on_push_frame(data)`: O(1) appends, analysis after the run.

- `data.frame` is `TTSAudioRawFrame` and `data.source` is the fixture source → record
  `(context_id, cumulative_audio_secs[context], data.timestamp)`.
- `data.frame` is `TTSLipsyncFrame` → record `(id(frame), context_id, window_start,
  window_end, pts, source_kind, data.timestamp)` where `source_kind` distinguishes
  processor-emit vs transport-release by `isinstance(data.source, ...)`.

### 4.4 `CapturingRTVIObserver(RTVIObserver)`

Subclass with `send_rtvi_message` overridden to record
`(msg.type, clock.get_time(), batch correlation via the just-seen frame)` and **not**
forward to any transport (there is no client). Constructed with
`RTVIObserverParams(bot_lipsync_enabled=True)`. This is the "message leaves the server"
point; everything beyond it is network/client, out of scope by design.
(Implementation note: correlate message → batch by having `_handle_lipsync` called
synchronously from the observer's frame handling — record the active frame around the
super() call, or match on `(ctx, first kf offset)`.)

### 4.5 Resource meter

- **Counters:** `resource.getrusage(RUSAGE_SELF)` — `ru_utime + ru_stime` delta over the
  measured window (exact CPU seconds, no sampling error).
- **Sampler:** psutil thread at 100 ms — RSS series (start, peak, end) and CPU% timeline.
- **Differential CPU (the key isolation trick):** run the *identical* workload twice —
  once with `LipsyncParams(enabled=False)`, once enabled. The pipeline, transport, pacing
  and observers are identical; the delta is lipsync's cost:
  `rtf = (cpu_enabled − cpu_disabled) / audio_seconds`, `%core = delta_cpu / wall_time`.
  This kills the noise from transport pacing loops and asyncio overhead that a single-run
  measurement would misattribute to lipsync.
- **`--allocs`:** wrap the steady-state segment (skip first 2 s) in `tracemalloc`
  snapshots; report top-10 allocation sites filtered to `pipecat/audio/lipsync` +
  `lipsync_processor.py`. Validates the allocation-free guardrail; `np.roots`'s internal
  allocation is the one documented exemption.
- **`--sessions N`:** N sequential full workloads, fresh `PipelineWorker` each, same
  process; RSS delta per session (linear fit) is the leak detector.

## 5. Metrics (all times pipeline-clock ns unless noted)

For batch `b` in context `c`:
`audio_avail(s)` = first PerfObserver timestamp where context `c`'s cumulative pushed
audio ≥ `s` seconds. `HORIZON` = `EVENT_FINALIZE_HORIZON_SEC` (0.4 s — the by-design
event-finalization hold: max closure duration + confirmation window; import it from
`pipecat.audio.lipsync.formant_lipsync_analyzer`, see implementation plan §6.5).

| Metric | Formula | What it tells you |
|---|---|---|
| `analysis_latency` | `t_emit(b) − audio_avail(window_end(b) + HORIZON)` | Analyzer + task scheduling cost per batch; the raw (un-adjusted) variant is also reported |
| `emit_headroom` | `pts(b) − t_emit(b)` | Slack before the release deadline; ~0 ⇒ clamped ⇒ analysis barely keeping up |
| `clamp_rate` | fraction of batches with `pts` clamped (from `stats.pts_clamped`) | Systematic lateness (expected ≈ utterance-start only) |
| `release_jitter` | `t_release(b) − pts(b)` | Clock-task fidelity; asyncio sleep granularity |
| `msg_overhead` | `t_msg(b) − t_release(b)` | Observer dispatch + pydantic serialization cost |
| `playout_lead` | `(t0(c) + first_kf_offset(b)) − t_msg(b)` | **The headline number:** how far ahead of audio playout the message exists server-side; the client's runway |
| `late_rate` | fraction with `playout_lead < 0` | Spec principle "never late" — must be 0 |
| `rtf` | differential CPU / audio seconds | Spec §11.4 budget ≤ 0.03 |
| `%core` | differential CPU / wall time (speech segments) | Spec §1.1 budget < 3% |
| `rss_session_mb` | enabled-run peak RSS − disabled-run peak RSS | Spec §1.1 budget < 20 MB |
| `rss_leak_mb_per_session` | slope across `--sessions N` | Soak-lite leak guard (spec §11.4) |
| `offset_integrity` | max over batches of `|nanoseconds_to_seconds(pts + lead − t0) − window_start|` | Offsets/pts arithmetic drift, esp. under `--barge-in` (must stay < 1 ms) |
| `msg_rate`, `bytes_per_msg` | from captured messages (JSON-encoded size) | Spec §6.4 wire budget: ≤ 5 msg/s, ≈ ≤ 350 B |

Distributions reported as median / p95 / max; per-context and aggregate.

## 6. Budgets (pass/fail in the report)

| Metric | Budget | Source |
|---|---|---|
| `rtf` | ≤ 0.03 | spec §11.4 |
| `%core` | ≤ 3% | spec §1.1 |
| `rss_session_mb` | < 20 MB | spec §1.1 |
| `rss_leak_mb_per_session` | < 1 MB | soak guard |
| `late_rate` | = 0 | spec §1.3 "never late" |
| `playout_lead` median (realtime pacing) | `scheduling_lead − 60 ms … scheduling_lead` | derived from §2.3 timing model |
| `release_jitter` p95 | ≤ 10 ms | asyncio sleep granularity allowance |
| `msg_overhead` p95 | ≤ 2 ms | serialization should be trivial |
| `clamp_rate` (realtime, post-first-batch) | < 5% | clamp is an utterance-start phenomenon |
| `analysis_latency` p95 (realtime) | ≤ 50 ms | task should track ingest closely |
| `bytes_dropped` (realtime) | = 0 | backpressure must not trigger at 1× |
| `--allocs` steady-state | ≈ 0 B/s from lipsync modules (np.roots exempt) | guardrail 12 |
| `msg_rate` | ≤ 5 msg/s per context | spec §6.4 |

Budgets are constants at the top of `performance.py`. A failed budget exits non-zero
(usable as a local pre-PR gate), with the failing rows highlighted.

## 7. Workloads & CLI

```bash
# Standard run: fixtures, realtime pacing, differential CPU, one voice
uv run python -m benchmarks.performance

# Stress / soak-lite
uv run python -m benchmarks.performance --pacing burst
uv run python -m benchmarks.performance --barge-in 3 --sessions 5
uv run python -m benchmarks.performance --minutes 10          # loop fixtures for a duration

# Allocation audit (slower; tracemalloc overhead is real — CPU numbers not comparable)
uv run python -m benchmarks.performance --allocs

# Live provider cadence (needs API key; timing includes provider streaming behavior)
uv run python -m benchmarks.performance --live --provider cartesia --voice <id>

# Regression tracking (same pattern as the accuracy harness)
uv run python -m benchmarks.performance --save-baseline
uv run python -m benchmarks.performance --compare
```

Flags: `--pacing realtime|burst`, `--sessions N`, `--minutes M`, `--barge-in S`,
`--allocs`, `--live --provider --voice`, `--voices/--sentences` (fixture selection),
`--save-baseline`, `--compare [path]`. `--live` uses the real TTS service in place of
`FixtureTTSSource` (driven by `TTSSpeakFrame`s via the accuracy harness's synthesis
plumbing) — the only mode where `analysis_latency`/lead include true provider chunk cadence.

Console output:

```
PERFORMANCE  fixtures · realtime · 12 clips · 61.4 s audio            PASS (13/13 budgets)
metric                    median      p95        max       budget         Δ baseline
playout_lead (ms)         182         158        —         140…200   ✓    +3
release_jitter (ms)       1.2         4.8        9.1       ≤10       ✓    −0.3
analysis_latency (ms)     11.4        26.0       41.2      ≤50       ✓    −2.1
msg_overhead (ms)         0.3         0.7        1.9       ≤2        ✓    =
late_rate                 0           —          —         =0        ✓
rtf                       0.019                            ≤0.03     ✓    −0.002
%core (speech)            1.9%                             ≤3%       ✓
rss_session (MB)          11.2 (peak Δ)                    <20       ✓    +0.4
leak (MB/session, n=5)    0.1                              <1        ✓
msg_rate (/s)             4.2        bytes/msg 291         ≤5, ~350  ✓
stats: batches 302 · pts_clamped 6 (2.0%) · bytes_dropped 0 · contexts 12/0 evicted
```

JSON to `benchmarks/results/performance-<ts>.json` (run metadata incl. src git sha,
pacing, machine info from `platform`/psutil; all distributions; budgets with pass/fail;
raw per-batch rows behind `--full-json` for offline digging).

## 8. Implementation checklist

1. Root `pyproject.toml`: add `psutil` to `[dependency-groups].dev`; `uv sync`.
2. `NullOutputTransport` — verify minimal abstract surface, run a bare pipeline
   (source → transport) pushing one clip; confirm clock task releases a hand-made pts
   frame at the right time (this validates the harness against known transport behavior
   before lipsync enters the picture).
3. `FixtureTTSSource` (both pacings) + `PerfObserver`; wire pipeline with
   `LipsyncProcessor`; print raw per-batch rows.
4. `CapturingRTVIObserver` + message correlation; `playout_lead`/`late_rate` land.
5. Resource meter: differential CPU runs, psutil sampler, `--sessions` leak fit.
6. Budgets table, report, JSON, `--save-baseline`/`--compare`, non-zero exit on fail.
7. `--barge-in`, `--minutes`, `--allocs`, `--live` (in that order; each optional beyond MVP).
8. First run on the dev machine; sanity-check lead ≈ scheduling_lead and rtf against a
   `time python -c "analyze 60s of audio"` back-of-envelope; `--save-baseline`.

## 9. Decisions log

| Decision | Why |
|---|---|
| Terminate at `send_rtvi_message`, no client/socket | User-defined scope: server-side only; everything after is network/client and would add weight without adding signal about *our* code |
| Real `BaseOutputTransport` subclass, not a mock | The clock queue, pacing and `t0` semantics ARE the thing being measured; a mock would measure the mock. ~10 LOC cost |
| Fixture replay by default, `--live` optional | Deterministic, free, offline; provider cadence is a separate question answered by one flag |
| Differential CPU (enabled vs disabled runs) | Isolates lipsync cost from transport/asyncio noise without a profiler; directly yields the spec's RTF and %core budgets |
| `FramePushed.timestamp` for all pipeline timings | Already pipeline-clock based (verified) — pts-comparable, no clock-domain conversion, no monkey-patching |
| Read-only `stats` counters on the processor (seam) | Backpressure/clamps/evictions are otherwise invisible without log scraping; plain ints, zero hot-path cost |
| psutil (dev group) despite stdlib `resource` | `ru_maxrss` units differ macOS/Linux and there's no portable RSS *timeline*; psutil is one small wheel in the harness-only dev group |
| Non-zero exit on budget fail | Makes the harness usable as a local pre-PR gate with zero extra machinery |
