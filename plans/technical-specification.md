# Pipecat Server-Side Lipsync / Viseme Generation — Technical Specification

**Status:** Design of record, as built (revised 2026-09-19). Written 2026-07 for an in-tree pipecat implementation; the code now lives in `server/lipsync/` on released `pipecat-ai` (~1.10.0) and is not being upstreamed. The table below lists where the implementation departs from the text; the sections are otherwise left as written, and §8–§13 are rewritten to reflect that the formant analyzer is the final design (no further tiers).
**Scope:** Server-side only. Client-side rendering, shape tables, and interpolation are out of scope.
**Companions:** [README.md](README.md) (plan of record, open items), [pipecat-implementation.md](pipecat-implementation.md) §11 (2026-07 decisions), [deep-review-2026-09-results.md](deep-review-2026-09-results.md) (2026-09 tuning), [pipecat-1.10-update.md](pipecat-1.10-update.md) (timing model on per-turn TTS contexts).

### As built — where the implementation departs from this document

| Section | Spec says | As built | Where decided |
|---|---|---|---|
| §2.2, §7 | `RTVIObserver` emits `bot-tts-lipsync` when it sees `TTSLipsyncFrame`, gated by `RTVIObserverParams.bot_lipsync_enabled`; the transport's clock queue releases the frame at `pts` | No observer changes: `LipsyncMessageRelay`, placed after `transport.output()`, wraps each released batch in an `RTVIServerMessageFrame`; the stock observer sends it as a `server-message` with `data.type = "bot-tts-lipsync"`. Since 2026-09-19 the processor's own delivery task schedules each batch on the pipeline clock and pushes it (a system frame, forwarded at once by the transport) at playout minus the lead; the clock queue is not used, because it holds a frame behind any earlier-queued word-timestamp frame with a later pts | pipecat-1.10-update.md; README.md (delivery timing) |
| §2.3 | `t0` = clock at first audio; continuity via the last emitted pts | First sample anchored at `max(now, playout end of already-ingested audio)`; audio arriving after the transport ran dry shifts later windows by the gap (`playout_offset`, wire `t0`); a reopened context id starts a new segment with offsets from zero | pipecat-1.10-update.md |
| §3.1 | `DataFrame` with `pts` | `SystemFrame` with `playout_ns` and `release_ns` (no `pts`); `playout_offset`; events may precede the window (late closures/silences); contexts kept as an ordered list, since one id can reopen | pipecat-1.10-update.md; README.md |
| §4.2 nasal | 1-hop entry on strong evidence | 2 consecutive hops; nasal features from the pre-emphasized spectrum; "missing F2 counts as damped" kept (needed on the Deepgram voice) | accuracy-improvements.md, deep-review-2026-09-results.md |
| §4.3 | P5/P95, 150 voiced frames to converge | P10/P90, 60 frames; adaptation spike rejection instead of an Hz median | accuracy-improvements-2.md |
| §4.4 | `c_lpc × c_conv × c_snr`; the client lerps toward neutral by confidence | `c_lpc × c_fit × c_snr × √c_conv` (`c_fit` from LPC prediction gain, full at log10 gain 3.2); diagnostic only — on real speech it mostly tracks loudness (~0.15), so clients scale neither pose nor opacity by it | pipecat-implementation.md §11, deep review [CONF-1] |
| §4.5 | Trailing 3-frame median, slew 0.25/hop, dead band 0.04 | Zero-phase median (hop t is emitted when t+1 arrives), slew 0.4/hop, dead band 0.05; heartbeats suppressed during silence | deep-review-2026-09-results.md |
| §5.2 | LPC order 12; pitch from LPC-residual autocorrelation, clarity 0.35 | Order 16; F2/F3 slot prior in assignment; voicing by normalized cross-correlation on a separate 40 ms raw frame (threshold 0.6, peak-fraction gate 0.05), residual method kept as a fallback; `f1_broad` (a broad F1-band root) feeds the openness mapping when the F1 slot is empty | deep-review-2026-09-results.md |
| §5.3 | `r[0] *= 1.0001` | `1 + 1e-6` | pipecat-implementation.md §11 |
| §6.1 | `LipsyncParams` dataclass | pydantic `BaseModel`; `dead_band`/`heartbeat_ms` are forwarded to the analyzer | pipecat-implementation.md §11 |
| §6.4 | batch at window close, 200 ms windows | keyframes and NASAL leave one hop (20 ms) after the window end; CLOSURE and SILENCE ride late in the next batch; the first window of an utterance is 100 ms; final keyframes are flushed after 100 ms without audio (mid-turn stall). The 0.4 s event-finalize horizon of 2026-07 held every batch and made the first one of each turn late | README.md (delivery timing) |
| §5.1, §10 | `soxr` resampling in the analyzer front end | pipecat's stream resampler in the processor's ingest; the analyzer contract is 16 kHz float32 | pipecat-implementation.md §11 |
| §11 | MFA phoneme truth, CI gates, a loopback timing rig, a performance harness | Praat + designed-corpus harness with committed baselines; delivery timing from the eval recorder; no CI, no performance harness | README.md |

---

## 1. Overview

Generate a real-time mouth-articulation signal from streamed TTS audio and deliver it to clients as timestamped RTVI data messages, synchronized with audio playout. Clients map the signal to their own mouth geometry (e.g. a 32-point 2D contour) and interpolate between shapes.

### 1.1 Goals

- Provider-universal: works with any `TTSService` producing `TTSAudioRawFrame`s, with zero provider-specific requirements.
- Continuous articulation signal (openness/width/rounding) rather than discrete viseme IDs as the primary channel; discrete events only for closures/nasals.
- Sample-accurate offsets server-side; playout-aligned delivery via the existing `pts` / clock-queue mechanism in `BaseOutputTransport`.
- No per-voice calibration step. Online adaptive normalization from generic priors.
- Zero cost when disabled; bounded, small cost when enabled (< 3% of one core, < 20 MB RSS per session).
- One analyzer interface between measurement and delivery, so the timing/batching path does not depend on how the signal is measured.

### 1.2 Non-Goals

- Consonant-accurate viseme classification in v1 (only closure + nasal detection).
- Emotion/expression classification (pitch/energy are exposed; interpretation is client-side).
- Forced alignment, phoneme recognition or a trained model — not planned (see [README.md](README.md)).
- Client SDK implementation.

### 1.3 Design principles

1. **Geometry stays client-side.** Server sends articulation parameters, never mouth points.
2. **Continuous > categorical.** Misestimation of a float drifts; misclassification of an ID snaps. Drift is masked by client smoothing; snaps are not.
3. **Never late.** Events are delivered ahead of playout with explicit offsets; a late event is dropped, not played.
4. **Uncertain → neutral.** Confidence gates how far from schwa the client should commit.

---

## 2. Architecture

### 2.1 Pipeline placement

```
... → LLM → TTSService → LipsyncProcessor → transport.output()
```

`LipsyncProcessor` is a `FrameProcessor` inserted between TTS and the output transport. It:

- Forwards **all** frames downstream immediately and unmodified (audio path adds zero latency).
- Copies `TTSAudioRawFrame` payloads into a per-context ring buffer.
- Runs analysis in a dedicated task (created via `self.create_task()` so it participates in processor lifecycle/cancellation).
- Emits `TTSLipsyncFrame`s (batched keyframes, `pts` set) downstream.

### 2.2 Data flow

```
TTSAudioRawFrame ──▶ passthrough ──────────────────────────▶ transport (audio path)
        │
        └─▶ ring buffer ─▶ analysis task ─▶ conditioning ─▶ batcher ─▶ TTSLipsyncFrame(pts)
                                                                            │
                                                    BaseOutputTransport._handle_frame
                                                                            │
                                                          frame.pts → MediaSender._clock_queue
                                                                            │
                                              _clock_task_handler releases at pts (pipeline clock)
                                                                            │
                                                     LipsyncMessageRelay (after transport.output()) → RTVIServerMessageFrame → RTVIObserver → server-message
```

This reuses the exact release mechanism used by word timestamps (`TTSTextFrame.pts`): frames with `pts` are queued in `MediaSender._clock_queue` and pushed downstream at presentation time relative to the pipeline clock (`transport.get_clock().get_time()`, nanoseconds). Interruptions cancel/recreate the clock task, discarding unplayed batches in lockstep with discarded audio — no new interruption machinery needed on the transport side.

### 2.3 Timing model

- **Baseline:** on the first `TTSAudioRawFrame` of a context, record `t0 = get_clock().get_time()`. This mirrors `TTSService._add_word_timestamps` semantics (`_initial_word_timestamp`): playout begins approximately when the first chunk reaches the transport, because the transport writes in real-time-paced 10 ms chunks from that moment.
- **Offsets:** per context, maintain `samples_seen`. Event offset (seconds) = `sample_index / sample_rate`. Sample-accurate by construction; immune to wall-clock jitter in frame delivery (TTS generates faster than real-time — offsets must be derived from sample counts, never from arrival times).
- **pts assignment:** for a batch covering audio `[w_start, w_end)`:
  `pts = t0 + seconds_to_nanoseconds(w_start) − seconds_to_nanoseconds(scheduling_lead)`
  where `scheduling_lead` (default 200 ms) releases the batch ahead of playout so the client scheduler has runway. Clamp `pts ≥ now` at emission.
- **Continuity across contexts:** adopt the `_word_last_pts` pattern — if a new context's `t0` would precede the last emitted pts (overlapping contexts), use the later value.
- **Client anchor semantics (contract, defined here even though client impl is out of scope):** offsets in the message are relative to `utterance start` = first audio of the given `context_id`. The RTVI schema carries `context_id` so clients anchor per-utterance, with a single client-side tunable delay knob to absorb jitter-buffer skew.

---

## 3. Frame Types

### 3.1 `TTSLipsyncFrame`

```python
@dataclass
class LipsyncKeyframe:
    offset: float            # seconds from utterance (context) start
    openness: float          # 0..1
    width: float             # 0..1
    rounding: float          # 0..1
    energy: float            # 0..1, RMS envelope (client secondary motion)
    pitch: float             # normalized 0..1 within session range; 0 if unvoiced
    confidence: float        # 0..1

@dataclass
class LipsyncEvent:
    offset: float
    kind: str                # "closure" | "nasal" | "silence"
    duration: float          # seconds; 0 = until next keyframe
    confidence: float

@dataclass
class TTSLipsyncFrame(DataFrame):
    context_id: str | None
    window_start: float      # seconds, inclusive
    window_end: float        # seconds, exclusive
    keyframes: list[LipsyncKeyframe]
    events: list[LipsyncEvent]
    # pts set before push; routed via clock queue in BaseOutputTransport
```

Notes:

- One frame per batch window (~200 ms), not per keyframe.
- `DataFrame` (not `ControlFrame`): it is content, ordered, and interruptible.
- Not `UninterruptibleFrame`: on interruption it must be discarded with the audio.

### 3.2 Consumed frames

| Frame | Action |
|---|---|
| `TTSStartedFrame` | open analysis context (`frame.context_id`), reset sample counter |
| `TTSAudioRawFrame` | copy into ring buffer keyed by `frame.context_id`; note `sample_rate` |
| `TTSStoppedFrame` | flush partial analysis window, close context |
| `InterruptionFrame` | drop buffers/counters, cancel in-flight batch, reset adaptive state decay timer |
| `StartFrame` | warm up (allocate buffers, prime FFT plans) |
| `EndFrame` / `CancelFrame` | teardown analysis task |

All are also forwarded unmodified.

---

## 4. Functional Specification — Signal Model

### 4.1 Parameters

| Param | Source | Mapping |
|---|---|---|
| `openness` | F1 | normalized within adaptive F1 range (low F1 → closed, high F1 → open) |
| `width` | F2 | normalized within adaptive F2 range (high F2 → spread /i/, low F2 → back /u,o/) |
| `rounding` | F2 + F3 heuristic | low F2 with moderate F1 → rounded; coarse is acceptable |
| `energy` | RMS envelope | log-compressed, normalized to session running max |
| `pitch` | LPC-residual autocorrelation | normalized to session p5–p95 range |
| `confidence` | composite | see §4.4 |

### 4.2 Discrete events

- **`closure`** (M/B/P): energy dip below adaptive threshold while inside a speech region, ≥ 2 consecutive analysis frames (40 ms hysteresis), bounded duration ≤ 250 ms. Distinguished from `silence` by surrounding speech energy within ±150 ms.
- **`nasal`** (hmm, /m/ /n/ sustained): voiced frame AND low spectral centroid AND strong energy concentration < 500 Hz AND damped/absent F2 peak. Overrides vowel trajectory while active (client shows closed/near-closed lips despite voicing). This is the sole guard against the worst-case failure (open mouth during "hmmm").
- **`silence`**: sustained sub-threshold energy > 300 ms → client returns to neutral.

### 4.3 Adaptive normalization (no calibration)

- Priors: F1 ∈ [250, 900] Hz, F2 ∈ [800, 2500] Hz; shifted +12% if median pitch of first 10 voiced frames > 180 Hz.
- Per-session running P5/P95 of voiced-frame F1/F2 via streaming quantile estimator (P² algorithm; O(1) memory).
- Effective range = `lerp(prior, learned, min(1, voiced_frames / 150))` → full convergence ≈ 3 s of speech.
- Confidence penalty during convergence (see §4.4) so early frames blend toward neutral client-side.
- Cache learned ranges in-process keyed by a voice fingerprint if available (TTS service voice_id via `TTSStartedFrame` context is **not** currently carried — v1 keys by processor instance = session; acceptable since sessions rarely switch voices). On detected distribution shift (learned median jumps > 25%), decay learned state back toward priors over 2 s.

### 4.4 Confidence

`confidence = c_lpc × c_conv × c_snr` where:

- `c_lpc`: formant plausibility — roots found within expected bands, bandwidths < 500 Hz, frame-to-frame formant delta < 300 Hz/frame. Merged/missing roots → 0.3.
- `c_conv`: `min(1, voiced_frames / 150)` normalization convergence.
- `c_snr`: energy margin above noise floor estimate.

Contract: client target = `lerp(neutral_schwa, estimated_shape, confidence)`.

### 4.5 Conditioning (server-side, pre-emission)

1. 3-frame median filter per continuous param (adds one hop = 20 ms latency; inside `scheduling_lead`).
2. Slew clamp: max param delta per hop = 0.25 (physical plausibility; kills residual spikes).
3. Dead-band: emit keyframe only if any param moved > 0.04 since last emitted keyframe, or > 240 ms elapsed (heartbeat keyframe so client springs stay pinned). Typical output: 15–25 keyframes/s speaking, 4/s sustained vowel.

---

## 5. DSP Implementation

All NumPy-only. Vendored in `server/lipsync/dsp.py`. No runtime dependency beyond numpy (pipecat's own resampler is used on ingest).

### 5.1 Front end

- **Analysis rate:** resample to 16 kHz mono via `soxr` (already a core dependency) regardless of TTS output rate. Fixes LPC order and band constants; removes the sample-rate variance pitfall.
- **Framing:** 25 ms window (400 samples), 20 ms hop (320). Hamming window. Pre-emphasis `y[n] = x[n] − 0.97·x[n−1]`.
- **Ring buffer:** per-context `bytearray`, capacity 2 s; analysis consumes at hop granularity; carries 5 ms overlap across TTS chunk boundaries (10 ms transport chunks never align with 20 ms hops — buffer indices must be sample-count-driven, not chunk-driven).

### 5.2 Per-frame features

- **RMS energy** + running noise floor (min-statistics over 1 s).
- **LPC:** order 12 @ 16 kHz. Autocorrelation method + Levinson-Durbin (implement directly; ~30 LOC; avoids `scipy`). Formants from `numpy.roots` on the LPC polynomial: keep roots with `imag > 0`, bandwidth `−(fs/π)·ln|r| < 500 Hz`, freq ∈ [200, 3500] Hz; sort ascending → F1, F2, (F3).
  - Failure handling: < 2 valid roots → hold previous formants, set `c_lpc = 0.3`.
  - Perf note: `numpy.roots` on order-12 poly ≈ 20 µs; at 50 fps ≈ 1 ms/s audio. Negligible. (Eigenvalue-free root refinement unnecessary.)
- **Pitch:** autocorrelation of LPC residual, search 60–400 Hz, peak clarity threshold 0.35 → voiced flag.
- **Spectral:** one 512-pt rFFT per frame → spectral centroid, band-energy ratio (< 500 Hz / total) for nasal detection.

### 5.3 Numeric guidance

- float32 throughout; convert from int16 PCM once at buffer ingest.
- Preallocate all per-frame arrays at `StartFrame`; zero allocations in steady state (relevant for GC pauses in long sessions).
- Add ε = 1e−9 guards on all log/div operations.
- Regularize autocorrelation: `r[0] *= 1.0001` (avoids singular Toeplitz on silence).

---

## 6. Processor Design

### 6.1 Class sketch

```python
class LipsyncProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        params: LipsyncParams | None = None,
        analyzer: BaseLipsyncAnalyzer | None = None,   # extension seam, §8
    ): ...

@dataclass
class LipsyncParams:
    batch_window_ms: int = 200
    scheduling_lead_ms: int = 200
    dead_band: float = 0.04
    heartbeat_ms: int = 240
    emit_energy: bool = True
    emit_pitch: bool = True
    enabled: bool = True          # runtime-togglable
```

### 6.2 Concurrency

- `process_frame` does passthrough + `buffer.write(frame.audio)` only — O(memcpy), never blocks on analysis.
- One analysis task per processor (not per context), created via `self.create_task()`. Loop: await new-audio event → drain hops → condition → batch → `await self.push_frame(lipsync_frame)`.
- Push from the analysis task is safe (FrameProcessor push is task-agnostic) but must go through `push_frame` on the processor to preserve ordering guarantees downstream.
- Backpressure: if the analysis task lags > 1 s behind ingest (pathological), drop oldest buffered audio and emit a `silence` event with low confidence — never stall the pipeline.

### 6.3 Interruption & lifecycle

- `InterruptionFrame`: clear ring buffers, reset per-context counters, abandon current batch. Do **not** reset adaptive normalization (voice unchanged). Downstream, the transport clock-task recreation drops already-emitted unplayed batches automatically.
- Overlapping contexts: buffers and sample counters keyed by `context_id`; a batch always carries the context it was measured from. Stale context (closed + drained) → drop residue silently.
- `TTSStoppedFrame`: process remaining full hops, discard tail < 1 hop, emit final partial batch, close context.

### 6.4 Batching & emission

- Accumulate keyframes/events until `batch_window_ms` of audio-time covered or context closes.
- Assign `pts` per §2.3; push one `TTSLipsyncFrame`.
- Wire-size budget: ≤ 6 keyframes + events per batch ≈ ≤ 350 bytes JSON; ≤ 5 msg/s. No socket-flooding risk; no binary encoding needed in v1.

---

## 7. RTVI Integration

### 7.1 Message

A standard RTVI `server-message` whose `data.type` is `bot-tts-lipsync`, emitted by `LipsyncMessageRelay` for each `TTSLipsyncFrame` the output transport releases (already playout-timed by the clock queue, minus `scheduling_lead`); the stock `RTVIObserver` forwards it unchanged.

```json
{
  "type": "bot-tts-lipsync",
  "data": {
    "version": 1,
    "ctx": "<context_id>",
    "t0": 0.0,
    "kf": [[0.42, 0.61, 0.35, 0.1, 0.7, 0.44, 0.9], ...],
    "ev": [[0.55, "closure", 0.12, 0.8], ...]
  }
}
```

- `kf` tuple order: `[offset, openness, width, rounding, energy, pitch, confidence]` — positional arrays over named fields for size.
- `version` field mandatory so the schema can evolve additively. Version 2 (2026-09-19) added `ws` (window start, audio seconds) and `lead` (seconds until that window plays, measured when the message is sent, negative when analysis is behind playout) and millisecond precision on timing fields; the client anchors the utterance on `now + lead − (ws + t0)`.
- No observer gating: the message exists only when `LipsyncMessageRelay` is in the pipeline.
- Follows the existing `bot-tts-text` precedent: separate channel, not attached to transcription messages, independent granularity.

### 7.2 Enable/disable

- Static: processor and relay present in the pipeline.
- Runtime: `LipsyncParams.enabled` togglable via a `LipsyncUpdateSettingsFrame` (pattern: `TTSUpdateSettingsFrame`); when disabled, passthrough only — analysis task idles on the event, zero CPU.

---

## 8. Analyzer Seam

```python
class BaseLipsyncAnalyzer(ABC):
    @abstractmethod
    async def start(self, sample_rate: int): ...
    @abstractmethod
    async def analyze(self, pcm: np.ndarray, ctx: AnalysisContext) -> LipsyncFrameResult: ...
    @abstractmethod
    async def flush(self, ctx) -> LipsyncFrameResult: ...
    @abstractmethod
    async def reset(self): ...
```

`FormantLipsyncAnalyzer` (§4–§5) is the only implementation and the design of record. The interface stays because it separates measurement from timing and batching — the processor and relay never look inside a keyframe — which keeps the processor testable with a stub analyzer and leaves an alternative possible without committing to one.

The tiers this section originally listed — provider viseme events (Azure/Polly), word/char timestamps + G2P, and an ONNX phoneme model — are not planned (2026-09-19):

- Provider-specific sources contradict the first goal in §1.1. Pipecat 1.10 exposes no viseme data for any service (Azure subscribes only to word boundaries; ElevenLabs collapses character alignment to words), so such a tier would need per-provider service subclasses anyway.
- A trained model (the deep review's "Tier 0.5": ~46 features from the existing DSP, 20–50 k parameters, forced-aligned TTS data, +10–25 µs/hop) is the obvious next step in consonant fidelity. It is not worth its training pipeline for clients that render a continuous mouth; the DSP signal is judged close enough.
- Text-informed alignment (the sentence text is already in the frame stream) is the cheapest route to bilabial precision if a client ever needs it; parked, see [README.md](README.md).

---

## 9. Optimizations

- **Zero-cost disabled path:** no processor → no cost; `enabled=False` → passthrough + parked task.
- **Steady-state allocation-free DSP** (§5.3).
- **Adaptive idle:** analysis task blocks on `asyncio.Event`; no polling during bot silence.
- **Batch pts clamping** prevents burst release when TTS outruns playout at utterance start.
- **Wire compaction:** positional arrays, 2-decimal float quantization at serialization (client springs make finer precision meaningless).

---

## 10. Dependencies

Runtime: **none new.** `numpy` (a pipecat core dependency); resampling on ingest through pipecat's stream resampler. LPC/Levinson-Durbin vendored.

Rejected for runtime: `librosa` (heavy transitive deps: numba/llvmlite — known cross-platform/py-version friction inside Pipecat installs), `scipy` (avoidable for the LPC in use; keep out of core).

Dev/test only (never runtime): `praat-parselmouth` (reference formant tracker for the accuracy harness), `pyyaml` (corpus files). Montreal Forced Aligner was never adopted — not pip-installable in practice (deep review §6.6) — and no phoneme truth is planned.

---

## 11. Testing & Benchmarking

As built: §11.1 is `server/tests/` (54 tests, including an end-to-end run through a headless output transport and the relay, the delivery schedule, and the eval recorder's replay). §11.2 became the Praat + designed-corpus accuracy harness with committed baselines ([benchmark-harness-accuracy.md](benchmark-harness-accuracy.md)) rather than MFA truth and CI gates. §11.3's loopback rig is the eval recorder (`server/benchmarks/record.py`), which measures each batch's release time against playout through a real output transport with a real-time simulated device. §11.4 is measured as µs per hop by the accuracy harness; the standalone performance harness was designed and not built. §11.5 is manual, against the client's eval playback. The original text follows.

### 11.1 Unit

- LPC/Levinson vs synthetic vowels with known formants (generate via all-pole filter excitation; assert F1/F2 within ±40 Hz).
- Chunk-boundary invariance: identical output for 10 ms vs 320 ms ingest chunking of the same PCM.
- Offset determinism: offsets derived purely from sample counts under artificially jittered frame delivery.
- Interruption: buffers cleared, no cross-context leakage, adaptive state preserved.

### 11.2 Accuracy regression (CI)

- Corpus: fixed sentence set × N voices × main providers, TTS output committed as fixtures.
- Ground truth: MFA phoneme boundaries → mapped to openness/width targets via phoneme→parameter table.
- Metrics: frame-wise MAE per parameter (voiced frames), closure precision/recall, nasal precision/recall, convergence time to 90% of final range.
- Gate: MAE regression > 10% fails CI.

### 11.3 Timing

- Loopback harness: instrumented client records played audio + received `bot-tts-lipsync` with local timestamps; cross-correlate energy envelope vs `kf.energy` track → end-to-end skew distribution. Target: |median skew| ≤ 60 ms, p95 ≤ 120 ms after client delay knob.

### 11.4 Performance

- RTF and RSS per session via `pipecat.evals` harness; target RTF ≤ 0.03, steady-state alloc rate ≈ 0.
- Soak: 30 min session, rapid barge-in every 5 s → assert no offset drift (sample counter vs wall clock), no buffer growth.

### 11.5 Perceptual (pre-GA, manual)

- Reference client rig; A/B graded artificial offsets (0/±50/±100/±150 ms) and injected error rates → establishes empirical perceptual budget and pass bar for §11.2 metrics.

---

## 12. Rollout (as it happened)

1. 2026-07: `TTSLipsyncFrame`, `LipsyncProcessor`, the formant analyzer and the RTVI wiring built in a pipecat fork (`pipecat-implementation.md`), then extracted into this standalone app on released pipecat-ai, with the relay replacing the observer branch and the frames kept app-local.
2. 2026-07: accuracy harness and two tuning passes (composite 28.8 → 70.9 on the Cartesia corpus).
3. 2026-09-17: pipecat-ai 1.10.0 — per-turn TTS contexts, playout gaps and `t0` on the wire (`pipecat-1.10-update.md`); upstreaming considered and declined.
4. 2026-09-18: deep review; the DSP bundle, zero-phase conditioning and 2-hop nasal entry validated on two voices (87.5 / 90.7).
5. Open: utterance-start latency and the other items in [README.md](README.md).

## 13. Open Questions

- Carry a voice key so adaptive state survives a mid-session voice switch? Still open. The stock `TTSService` consumes `TTSUpdateSettingsFrame`, so the key would have to come from the app (deep review [ADAPT-2]); the example bot does not switch voices.
- Should `scheduling_lead` adapt to measured transport latency? Not pursued; the actual lead is on the wire (version 2), so the client anchors on it and needs no assumption.
- Binary encoding (msgpack)? Not needed at 28–31 keyframes/s and ≤ 5 messages/s; JSON stays.
- Multi-destination transports: per-destination clock queues isolate timing, but `transport_destination` propagation on `TTSLipsyncFrame` has not been exercised.
