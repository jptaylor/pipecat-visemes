# Pipecat Lipsync — Implementation Plan

**Status:** Ready for implementation
**Scope:** The Pipecat side only — everything needed to take the stubbed files in `src/` to a
mergeable PR. Client SDKs, the eval/benchmark harness (tech spec §11.2–§11.5), and higher
analyzer tiers (§8 tiers 1–3) are out of scope; unit tests per Pipecat conventions are in scope.
**Companion:** [technical-specification.md](technical-specification.md) (referenced throughout as "spec §N").
**Verified against:** pipecat `main` @ `d9706519a` (the `src/` checkout, branch `feat/lipsync-processor`).

---

## 1. What exists, what gets built

All files are stubbed on `feat/lipsync-processor` with signatures, docstrings and
`TODO(lipsync)` markers. Implementation fills bodies; public interfaces are already settled
and should not change without updating this plan.

| File | Status | Work | Est. LOC |
|---|---|---|---|
| `src/pipecat/audio/lipsync/types.py` | **Done** | None — `LipsyncKeyframe`, `LipsyncEvent`, `LipsyncEventKind` are final | — |
| `src/pipecat/audio/lipsync/dsp.py` | Stub | M1: all DSP primitives (§4 below) | ~450 |
| `src/pipecat/audio/lipsync/base_lipsync_analyzer.py` | **Done** | None — ABC is final | — |
| `src/pipecat/audio/lipsync/formant_lipsync_analyzer.py` | Stub | M2: feature→signal mapping, adaptive normalization, conditioning, events (§5) | ~350 |
| `src/pipecat/processors/audio/lipsync_processor.py` | Stub | M3: buffering, analysis task, timing, batching, lifecycle (§6) | ~300 |
| `src/pipecat/frames/frames.py` | **Done** | None — `TTSLipsyncFrame`, `LipsyncUpdateSettingsFrame` are final | — |
| `src/pipecat/processors/frameworks/rtvi/models.py` | **Done** | None — `LipsyncMessageData`, `BotTTSLipsyncMessage` are final | — |
| `src/pipecat/processors/frameworks/rtvi/observer.py` | Partial | M4: dispatch branch + `_handle_lipsync` (param field exists; TODO marks the spot) | ~40 |
| `src/tests/test_lipsync_dsp.py` | New | M1/M2 tests (§8) | ~250 |
| `src/tests/test_lipsync_processor.py` | Stub | M3/M4 tests — implement the skipped placeholders (§8) | ~350 |
| `src/pipecat/transports/base_output.py` | **No change** | See §2.1 — pts routing already generic | 0 |
| `src/pyproject.toml` / `uv.lock` | **No change** | Tier 0 rides on core `numpy` + `soxr` (decided; no `lipsync` extra) | 0 |

### Milestones (bottom-up; each lands with its tests green)

1. **M1 — DSP primitives** (`dsp.py` + `test_lipsync_dsp.py`). Pure functions, no pipecat coupling. Everything downstream depends on these being right.
2. **M2 — FormantLipsyncAnalyzer** (+ analyzer-level tests feeding PCM arrays directly). The whole signal model, testable without a pipeline.
3. **M3 — LipsyncProcessor** (+ `run_test()` pipeline tests). Buffering, concurrency, timing, batching, lifecycle.
4. **M4 — RTVI observer wiring** (+ observer test). ~40 lines; the message models already exist.
5. **M5 — Integration polish**: bot.py example validated end-to-end, foundational example, changelog fragment, docs, PR checklist (§10).

---

## 2. Verified integration facts (do not rediscover these)

These were confirmed by reading `main`; the implementation relies on them.

### 2.1 The transport already routes pts frames — zero transport changes

`BaseOutputTransport._handle_frame` (base_output.py:361) dispatches:

```python
elif frame.pts:
    await sender.handle_timed_frame(frame)   # → MediaSender._clock_queue
```

Any non-system downstream frame with `pts` set lands in `MediaSender._clock_queue`
(an `asyncio.PriorityQueue` ordered by `(pts, counter)`), and `_clock_task_handler`
sleeps until `pts` vs `transport.get_clock().get_time()` then calls
`self._transport.push_frame(frame)` — pushing it downstream of the transport, where
observers see it with `src` = the transport. On interruption the clock task is cancelled
and recreated (base_output.py:570–575), discarding queued frames in lockstep with the
audio. **`TTSLipsyncFrame` inherits all of this by simply having `pts` set.**

Consequences:

- `pts` must be truthy (the dispatch is `elif frame.pts:`) — the clamp-to-now rule (§6.4)
  guarantees a positive value.
- `frame.transport_destination` must match a registered media sender or the frame is
  dropped with a warning (base_output.py:363). The processor must copy
  `transport_destination` from the `TTSAudioRawFrame`s it measured (resolves spec §13's
  multi-destination question — propagation is our responsibility, routing is free).
- After clock release the frame continues downstream (e.g. through the assistant
  aggregator in the quickstart pipeline). It's a plain `DataFrame`; unknown-frame handling
  in existing processors forwards it. No action needed.

### 2.2 Word-timestamp baseline semantics to mirror (tts_service.py:1180–1288)

- Baseline: `_initial_word_timestamp = max(get_clock().get_time(), _word_last_pts)` at
  first audio — the max preserves monotonicity across overlapping contexts.
- `pts = baseline + seconds_to_nanoseconds(offset_seconds)`.
- Reset (`_initial_word_timestamp = -1`) on interruption; last emitted pts is tracked to
  anchor the next context.

The processor reimplements this pattern privately (`_t0`, `_last_emitted_pts`); it must not
touch the TTS service's fields.

### 2.3 Other confirmed APIs

- `FrameProcessor` inherits `create_task(coro, name)` / `cancel_task(task, timeout)` from
  `BaseObject`; `get_clock()` exists on `FrameProcessor` (frame_processor.py:568);
  `push_error(msg, exception=None, fatal=False)` pushes upstream.
- `InterruptionFrame` is a **SystemFrame** — it can overtake queued data frames; the
  framework discards queued data frames on interruption, our job is only our own state.
- `StartFrame` carries `audio_out_sample_rate` (may be None until transport resolves it);
  `TTSAudioRawFrame.sample_rate` is always authoritative per frame.
- Resampling: `pipecat.audio.utils.create_stream_resampler()` →
  `await resampler.resample(pcm_bytes, in_rate, out_rate)` (soxr-backed, streaming, int16 bytes).
- `pipecat.utils.time.seconds_to_nanoseconds` / `nanoseconds_to_seconds` exist.
- RTVI: `PipelineWorker(..., rtvi_observer_params=RTVIObserverParams(...))` configures the
  auto-added observer (pipeline/worker.py:251); an explicit `RTVIProcessor` in the pipeline
  + `RTVIObserver` in `observers=[...]` overrides it.
- Tests: `pipecat.tests.utils.run_test(processor, frames_to_send=..., expected_down_frames=...,
  ignore_start=True, send_end_frame=True)` and `SleepFrame(sleep=N)` for inter-frame delays.

---

## 3. Global guardrails (apply to every milestone)

**Pipeline safety — the prime directive.** A lipsync bug must never degrade a working voice
bot:

1. `process_frame` always forwards every frame, first-touch cheap: passthrough + `bytes`
   copy into a ring buffer only. No DSP, no allocation beyond the copy, no awaiting the
   analysis task.
2. Any exception inside the analysis task: catch at the task's top level, log via
   `push_error(..., fatal=False)`, set an internal `_failed` flag that parks analysis
   permanently for the session. Passthrough continues. Never re-raise into the pipeline.
3. Bounded memory: ring buffers capped at 2 s per context; max 8 live contexts (evict
   oldest, log once); conditioning/history buffers fixed-size.
4. Backpressure (spec §6.2): if un-analyzed audio exceeds 1 s in a context, drop the oldest
   audio, emit one `silence` event with `confidence=0.1`, and log throttled. Never stall.

**Timing correctness.**

5. Offsets derive **only** from sample counts (never wall-clock arrival) — TTS generates
   faster than real time.
6. Never late: `pts = max(computed_pts, get_clock().get_time())` at emission. A frame that
   would be late releases immediately (still ordered); clients decide staleness from offsets.
7. Never block the frame path on analysis; the analysis task never blocks on the frame path.

**Signal correctness.**

8. Uncertain → neutral: every keyframe carries honest `confidence`; the client contract is
   `lerp(neutral, estimate, confidence)`. When in doubt, lower confidence rather than
   guessing shape.
9. Continuous params drift, never snap: median-3 + slew clamp before emission (spec §4.5).
10. Geometry stays client-side: nothing in the wire format names mouth points.

**Numerics & performance (spec §5.3, §9).**

11. float32 end-to-end after int16 ingest; `EPSILON = 1e-9` guards on every log/divide;
    lag-zero regularization `r[0] *= (1 + 1e-6)` before Levinson (the spec's 1e-4 factor
    measurably damps poles on high-prediction-gain frames — see decisions log).
12. Steady-state allocation-free: all per-frame arrays preallocated at `start()`; hot loops
    use `out=` / in-place ops. (Budget: < 3% of one core, < 20 MB RSS per session.)
13. Adaptive idle: the analysis task blocks on an `asyncio.Event`; zero CPU during bot
    silence or when `enabled=False`.

**Code conventions (repo AGENTS.md).**

14. Google-style docstrings everywhere (`Args:`/`Returns:`; `Parameters:` on dataclasses and
    pydantic models); license header `Copyright (c) 2026, Daily`.
15. Lint/format with the locked toolchain: `uvx ruff@0.15.14 check` / `format` (older ruff
    false-flags pre-existing code).
16. No new dependencies. `numpy` and `soxr` only; no scipy/librosa (spec §10 rejects them).

---

## 4. M1 — `dsp.py` implementation spec

Pure, stateless functions (plus one small stateful class), operating on preallocated
float32 arrays. All constants already stubbed (`ANALYSIS_SAMPLE_RATE=16000`,
`FRAME_SIZE=400`, `HOP_SIZE=320`, `PRE_EMPHASIS=0.97`, `LPC_ORDER=12`, formant/pitch bands,
`VOICED_CLARITY_THRESHOLD=0.35`).

### 4.1 `levinson_durbin(autocorr, order) -> np.ndarray`

Standard recursion solving Toeplitz normal equations:

```
a = [1, 0, ..., 0]; err = r[0]
for i in 1..order:
    k = -(r[i] + Σ_{j=1}^{i-1} a[j]·r[i−j]) / err
    a[1..i] = a[1..i] + k·a[i−1..0]   (reflected update, use a scratch buffer)
    err *= (1 − k²)
    if err <= 0: break                 # numerically degenerate → return current a
```

Caller responsibilities (documented, asserted in tests): pass `order+1` lags; regularize
`r[0] *= 1.0001` beforehand so silence doesn't produce a singular system. Returns
`a[0..order]`, `a[0] == 1.0`. ~30 LOC.

### 4.2 `lpc_formants(frame) -> FormantEstimate`

Input: one **already pre-emphasized, Hamming-windowed** frame (the analyzer owns
windowing so the same windowed buffer feeds LPC and FFT). Steps:

1. Autocorrelation lags 0..12: `np.correlate(frame, frame, 'full')` center-out slice, or
   rFFT power-spectrum → irFFT (either is fine at N=400; pick one, keep it allocation-free).
2. Regularize, Levinson → `a`.
3. `roots = np.roots(a)`; keep `imag(r) > 0`.
4. Per root: `freq = atan2(imag, real) · fs/(2π)`;
   `bandwidth = −(fs/π)·ln(|r|)` (guard `|r|` with EPSILON).
5. Filter: `FORMANT_MIN_HZ ≤ freq ≤ FORMANT_MAX_HZ` and `bandwidth < FORMANT_MAX_BANDWIDTH_HZ`.
6. Sort ascending → F1, F2, F3 (0.0 for missing). `plausible = (count ≥ 2)`.

Perf note (spec §5.2): `np.roots` on an order-12 polynomial ≈ 20 µs; at 50 fps that is
~1 ms of CPU per second of audio — fine, no root-refinement needed. `np.roots` allocates
internally; this is the one accepted allocation per frame (documented exception to
guardrail 12 — it is small, and replacing it is not worth the complexity).

### 4.3 `lpc_residual_pitch(frame, lpc) -> PitchEstimate`

1. Residual = FIR filter of the (pre-emphasized, windowed) frame by `a[0..12]`
   (`np.convolve(frame, a)[:FRAME_SIZE]`, or a preallocated manual loop).
2. Autocorrelate residual; search integer lags `fs/PITCH_MAX_HZ .. fs/PITCH_MIN_HZ`
   (40..266 samples @16 kHz).
3. `clarity = r[best_lag] / (r[0] + EPSILON)`; `voiced = clarity > VOICED_CLARITY_THRESHOLD`;
   `frequency = fs / best_lag` if voiced else 0.0.

### 4.4 `rms_energy(frame) -> float`

`sqrt(mean(frame²) + EPSILON)`. Computed on the **raw** (un-windowed, un-emphasized) frame
so the envelope tracks loudness, not spectral tilt. The noise-floor tracker lives in the
analyzer (it's stateful), not here.

### 4.5 `spectral_nasal_features(frame) -> (centroid_hz, low_band_ratio)`

One 512-point `np.fft.rfft` per frame (window already applied): power spectrum `P`;
`centroid = Σ f·P / (ΣP + EPSILON)`; `low_band_ratio = Σ_{f<500Hz} P / (ΣP + EPSILON)`.
Bin 500 Hz = index 16 at 16 kHz/512 — precompute the slice index as a module constant.

### 4.6 `P2QuantileEstimator`

Textbook P² (Jain & Chlamtac 1985): 5 markers (min, q/2, q, (1+q)/2, max), parabolic/linear
marker adjustment, O(1) memory, `add(value)` + `value()`. First 5 samples: store & sort;
`value()` before 5 samples returns the running sample quantile. ~60 LOC. Used by the
analyzer for F1/F2 P5/P95 and pitch P5/P95.

**Definition of done (M1):** `test_lipsync_dsp.py` green (§8.1); every function
allocation-checked by eye; docstring bodies replace the TODO markers.

---

## 5. M2 — `FormantLipsyncAnalyzer` implementation spec

The analyzer is a **pure consumer of 16 kHz float32 samples** and producer of
keyframes/events with offsets in seconds from context start. It owns: resampling to 16 kHz,
hop framing, feature extraction (via `dsp`), adaptive normalization, conditioning, and event
detection. It does **not** know about pts, batching, or frames — that's the processor.

### 5.1 Internal state (allocated in `start()`)

```python
# Per-session (survives interruptions; reset only on start()):
_resampler                     # create_stream_resampler()
_win: np.ndarray               # Hamming window, FRAME_SIZE, float32
_frame_buf, _windowed, _autocorr, _residual, _fft_buf   # preallocated scratch
_f1_p5, _f1_p95, _f2_p5, _f2_p95: P2QuantileEstimator   # adaptive ranges
_pitch_p5, _pitch_p95: P2QuantileEstimator
_voiced_frames: int            # convergence counter (c_conv)
_energy_max: float             # running max, exponential decay τ≈10 s
_noise_floor: float            # min-statistics over ~1 s (50 frames)
_pitch_shift_applied: bool     # +12% prior shift decision (first 10 voiced frames)
_prior_decay: float            # 0..1 blend toward priors after distribution shift

# Per-utterance (cleared by reset() and at context close):
_pending: np.ndarray + length  # 16 kHz tail < HOP_SIZE carried across analyze() calls
_hops_processed: int           # drives offsets: offset = (hops·HOP + FRAME/2) / 16000
_prev_formants: (f1, f2, f3)   # hold-last on implausible frames
_median_hist: 3×N float32      # conditioning history per continuous param
_last_emitted: keyframe params # dead-band reference
_last_keyframe_offset: float   # heartbeat reference
_event_state: closure/nasal/silence state machines (§5.5)
_recent_energy: ring of ~16 frames (±150 ms context for closure confirmation)
```

Keep per-context state **inside the processor's context dict** (the analyzer is called with
one `LipsyncAnalysisContext` at a time and stores per-utterance state keyed by nothing —
see §6.2: the processor serializes calls per context and calls `flush()`/`reset()` between
contexts, so the analyzer holds exactly one utterance's state at a time).

### 5.2 `analyze(pcm, context)` pipeline

1. **Ingest:** input arrives as float32 mono at `context.sample_rate` (the processor
   converts int16→float32 once). Resample to 16 kHz via `_resampler` (bytes API: the
   processor actually passes int16 bytes; see §6.3 — the analyzer's public `pcm: np.ndarray`
   receives the already-resampled 16 kHz float32; keep the resampler in the processor's
   ingest path so `analyze()` stays pure 16 kHz).
   *Decision:* resampling lives in the **processor ingest** (it has the bytes and the rate);
   the analyzer contract is: `pcm` is 16 kHz float32. Update the ABC docstring accordingly
   when implementing (one-line change, keeps the analyzer deterministic for tests).
2. **Frame:** append to `_pending`; while `len ≥ FRAME_SIZE`: process one frame at
   `HOP_SIZE` stride (25 ms window, 20 ms hop → overlapping tail carried, satisfying the
   spec's 5 ms overlap requirement across arbitrary chunk boundaries; buffer indices are
   sample-count-driven by construction).
3. **Per-frame features:** raw RMS → energy path; pre-emphasize + window into `_windowed`;
   `lpc_formants`; `lpc_residual_pitch`; `spectral_nasal_features` (skip FFT when frame is
   below the silence threshold — cheap early-out).
4. **Adaptive normalization update** (voiced frames only): feed F1/F2/pitch estimators;
   `_voiced_frames += 1`; first-10-voiced median pitch > 180 Hz → shift F1/F2 priors +12%
   once; distribution-shift check (§5.4).
5. **Map to parameters** (§5.3), **condition** (§5.4), **detect events** (§5.5).
6. Return `LipsyncFrameResult(keyframes=[...], events=[...])` — only the keyframes that
   survived the dead-band, with offsets.

`flush(context)`: process remaining full hops from `_pending`, drop the sub-hop tail,
finalize any pending event candidates (a closure candidate without confirmed trailing
speech resolves to `silence`-or-drop, see §5.5), return final result, clear per-utterance
state. `reset()`: clear per-utterance state only — adaptive estimators survive (spec §6.3:
voice unchanged).

### 5.3 Feature → parameter mapping

With effective ranges (§5.4) `[f1_lo, f1_hi]`, `[f2_lo, f2_hi]`:

```
openness  = clamp01((F1 − f1_lo) / (f1_hi − f1_lo))
width     = clamp01((F2 − f2_lo) / (f2_hi − f2_lo))
rounding  = clamp01((f2_mid − F2) / (f2_mid − f2_lo)) · window(openness)
            where f2_mid = (f2_lo + f2_hi)/2 and
            window(o) = smoothstep band gate ≈ 1 for o ∈ [0.15, 0.75], → 0 outside
            (rounded vowels /u,o/ are mid-open with low F2; spec: "coarse is acceptable")
energy    = log1p(9 · rms / (_energy_max + EPSILON)) / log1p(9)      # 0..1, log-compressed
pitch     = clamp01((f0 − p5) / (p95 − p5)) if voiced else 0.0
confidence = c_lpc · c_conv · c_snr                                   # §5.6
```

Unvoiced frames: hold previous openness/width/rounding decayed toward neutral (multiply
distance-to-0.35 by 0.8 per hop) — fricatives keep the mouth loosely open rather than
snapping shut; energy still tracks.

### 5.4 Adaptive normalization (spec §4.3)

- Priors: F1 ∈ [250, 900] Hz, F2 ∈ [800, 2500] Hz (+12% both edges if high-pitch voice).
- Learned: P5/P95 estimators over voiced-frame F1/F2.
- Effective range: `lerp(prior, learned, w)` with `w = min(1, _voiced_frames / 150)`
  (≈3 s of speech to full convergence).
- Distribution shift: track a slow EMA of voiced F1; if a fresh 0.5 s window's median
  deviates > 25% from the long EMA, set `_prior_decay = 1.0` and decay it → 0 over 2 s;
  effective range during decay: `lerp(effective, prior, _prior_decay)`; also reset
  `_voiced_frames` to 30 (not 0 — partial trust). Simplest correct implementation: keep a
  100-frame ring of voiced F1, compare median of last 25 vs full ring.
- Session-keyed only (spec: per-processor-instance cache is acceptable v1; `voice_id`
  keying is spec §13 — explicitly deferred, noted in the class docstring).

### 5.5 Discrete events (spec §4.2) — state machines

All thresholds are module-level tunable constants with short rationale comments.

**Closure (M/B/P):**
- Candidate: energy < `max(3·_noise_floor, 0.15·recent_p95_energy)` for ≥ 2 consecutive
  frames (40 ms hysteresis) while the *preceding* 150 ms contained speech (check
  `_recent_energy` ring).
- Confirmation requires speech within the *following* 150 ms — an inherent ~150 ms decision
  latency. This is fine: the batcher (§6.5) doesn't finalize a batch until the analysis
  cursor is ≥ 150 ms past the batch window's end (TTS outruns real time, so this rarely
  delays emission past its pts deadline).
- Bounded: duration ≤ 250 ms, else the candidate resolves to `silence` handling.
- Emit `LipsyncEvent(offset=candidate_start, kind=CLOSURE, duration=measured,
  confidence=depth_margin)` where `depth_margin = clamp01(1 − dip_energy/threshold)`.

**Nasal (spec: the sole guard against open-mouth "hmm"):**
- Condition per frame: `voiced AND centroid < 1000 Hz AND low_band_ratio > 0.6 AND F2
  damped` (no valid F2 root in [800, 2500] or its bandwidth > 400 Hz).
- Enter after 2 consecutive matching frames; exit after 2 consecutive non-matching.
- While active: **override** the vowel trajectory — force emitted openness toward 0.05
  (closed lips despite voicing) and emit `LipsyncEvent(kind=NASAL, offset=enter,
  duration=0)` at entry, then a final event with measured duration at exit.
- Confidence: `min(low_band_ratio, clarity)`.

**Silence:**
- Energy below threshold > 300 ms → emit `LipsyncEvent(kind=SILENCE, offset=start,
  duration=0, confidence=0.9)` once (not repeatedly); client returns to neutral. Reset when
  speech resumes.

**Ordering rule:** events may be *confirmed* after keyframes at later offsets exist, but
within an emitted batch, `keyframes` and `events` lists are each sorted by offset, and every
offset lies within `[window_start, window_end)` — the deferred-finalize rule above makes
this invariant cheap to keep.

### 5.6 Confidence components (spec §4.4)

```
c_lpc  = 1.0 when: plausible roots AND all kept bandwidths < 500 Hz
              AND |F1−prev_F1| < 300 AND |F2−prev_F2| < 300 (per frame)
         0.3 when roots merged/missing (formants held from previous frame)
         linear 0.3→1.0 for the delta criterion (soft, not a cliff)
c_conv = min(1, _voiced_frames / 150)
c_snr  = clamp01(20·log10((rms+EPSILON)/(_noise_floor·3+EPSILON)) / 20)   # 0 dB→0, 20 dB→1
```

### 5.7 Conditioning (spec §4.5) — order matters

Per continuous param, per hop: **median-3** (adds one hop = 20 ms latency, inside the
scheduling lead) → **slew clamp** (max |Δ| 0.25/hop) → **dead-band gate**: emit a keyframe
only if any of openness/width/rounding moved > 0.04 since the last *emitted* keyframe, or
> 240 ms elapsed (heartbeat), or an event fired at this hop (keyframe accompanies event so
clients have a fresh anchor). `emit_energy=False` / `emit_pitch=False` zero those fields
(they don't participate in the dead-band test). Expected rates (spec): 15–25 kf/s speaking,
~4/s sustained vowel.

### 5.8 Debug feature tap (benchmark seam)

`FormantLipsyncAnalyzer(collect_debug: bool = False)`: when set, append one
`LipsyncDebugFrame` (dataclass in the same module: offset, raw F1/F2/F3 Hz, pitch, voiced,
rms, centroid, low-band ratio, confidence components, effective normalization ranges) per
hop to `self.debug_features`. Cost when off: one `if` per hop. Never enabled by the
processor — consumed by [benchmark-harness-accuracy.md](benchmark-harness-accuracy.md)
(raw-Hz scoring vs Praat) and available to tests.

**Definition of done (M2):** analyzer-level tests green (§8.2): synthetic vowels map to
correct relative openness/width ordering; nasal synthetic (low-passed voiced buzz) triggers
NASAL and openness override; silence produces exactly one SILENCE event; chunk-size
invariance at the analyzer level; conditioning caps keyframe rate.

---

## 6. M3 — `LipsyncProcessor` implementation spec

### 6.1 Frame dispatch (fill the TODO table in `process_frame`)

Every branch forwards the frame; consumption is a side effect. Order: handle-then-push for
state that must exist before downstream reacts; push-then-handle is never needed here
because nothing downstream depends on our side effects.

| Frame | Action (before forwarding) |
|---|---|
| `StartFrame` | `_start()`: create analysis task via `self.create_task(...)`; prime analyzer `await analyzer.start(frame.audio_out_sample_rate or 24000)` (re-primed lazily if first audio's rate differs) |
| `TTSStartedFrame` | open context: `_contexts[frame.context_id] = _Context(...)` (evict-oldest at 8, log once) |
| `TTSAudioRawFrame` | `_ingest(frame)` — §6.3. O(memcpy) only |
| `TTSStoppedFrame` | mark context `closing=True`, set `_wake` event (task flushes + closes it) |
| `InterruptionFrame` | §6.6 |
| `LipsyncUpdateSettingsFrame` | apply `frame.settings` onto `_params` (only known fields; warn on unknown); `enabled=False` → clear contexts, task parks; `enabled=True` → resume |
| `EndFrame` | set `_stopping`, wake task, let it drain current work, then `await self.cancel_task(self._task)` after push |
| `CancelFrame` | `await self.cancel_task(self._task)` immediately |

Notes: `TTSStartedFrame`/`TTSStoppedFrame` are ControlFrames and arrive in order with the
audio. `InterruptionFrame` is a SystemFrame and can overtake — all interruption handling
must be idempotent and safe against frames from the aborted context still sitting in our
ring buffers (they get cleared) or arriving late (context id no longer registered → ignore
silently, spec §6.3 "stale context → drop residue silently").

### 6.2 Concurrency model

- **One analysis task per processor** (`self.create_task`, participates in lifecycle).
- Frame path → task communication: per-context `bytearray` ring buffer + a single shared
  `asyncio.Event` (`_wake`). Frame path appends bytes and sets the event; the task drains
  every context with pending work, clears the event when nothing is pending, and awaits it.
- The task calls `await self.push_frame(lipsync_frame)` directly — pushes are
  task-agnostic but must go through `push_frame` for ordering (spec §6.2).
- A `_generation` int guards interruptions: the frame path increments it on
  `InterruptionFrame`; the task snapshots it before analyzing and discards results if it
  changed (abandon in-flight batch without locks).
- The task loop is the **only** consumer of ring buffers; the frame path is the only
  producer. Single-threaded asyncio → no locking needed beyond the generation counter.

### 6.3 Ingest path (`_ingest`, on the frame path — keep it tiny)

```python
ctx = self._contexts.get(frame.context_id)        # None → stale/unknown: ignore
if ctx is None or not self._params.enabled or self._failed: return
if ctx.t0 == 0:                                    # first audio of context
    now = self.get_clock().get_time()
    ctx.t0 = max(now, self._last_emitted_pts + 1)  # §2.2 continuity rule
    ctx.sample_rate = frame.sample_rate
    ctx.transport_destination = frame.transport_destination
if len(ctx.buffer) + len(frame.audio) > ctx.capacity:   # 2 s cap → backpressure
    overflow: del oldest bytes; ctx.dropped_silence_pending = True   # task emits the event
ctx.buffer += frame.audio                          # int16 bytes, source rate
ctx.samples_seen += frame.num_frames
self._wake.set()
```

The **task-side** drain per context: swap out `ctx.buffer` (`buf, ctx.buffer = ctx.buffer,
bytearray()`), resample bytes to 16 kHz via the processor-owned stream resampler
(`create_stream_resampler()`, one per context — soxr streams are stateful), convert int16 →
float32 (`np.frombuffer(...).astype(np.float32) / 32768.0`, into a preallocated scratch when
possible), then `result = await analyzer.analyze(pcm16k, ctx.analysis_context)`.

### 6.4 Timing model (spec §2.3)

- `ctx.t0` (ns) — set at first audio, continuity-clamped (§6.3).
- Offsets come from the analyzer (16 kHz hop domain — sample-accurate by construction;
  source-domain `samples_seen` is kept for diagnostics and the §8 determinism test).
- Batch pts: for a batch covering `[w_start, w_end)` seconds:

```python
pts = ctx.t0 + seconds_to_nanoseconds(w_start) - seconds_to_nanoseconds(self._params.scheduling_lead_ms / 1000)
pts = max(pts, self.get_clock().get_time())        # clamp: never in the past
frame.pts = pts
self._last_emitted_pts = pts
```

- Burst protection is inherent: early batches clamp to `now` and the priority queue orders
  them; later batches release at their computed lead.

### 6.5 Batching & emission (spec §6.4)

- Accumulate analyzer results per context until covered audio-time ≥ `batch_window_ms`
  **and** the analysis cursor is ≥ `EVENT_FINALIZE_HORIZON_SEC` (0.4 s = 250 ms max closure
  duration + 150 ms speech-confirmation window) past `w_end`, or the context is closing
  (flush emits whatever remains immediately). TTS outruns real time, so the horizon costs
  wall-clock only at utterance start, where the pts clamp already floors delivery.
- Build `TTSLipsyncFrame(context_id=..., window_start=w_start, window_end=w_end,
  keyframes=[...], events=[...])`; set `frame.transport_destination =
  ctx.transport_destination`; assign pts; `await self.push_frame(frame)`.
- Skip empty batches (no keyframes, no events) — silence costs zero messages after the one
  SILENCE event.
- Wire budget check (spec §6.4): ≤ 6 keyframes typical per 200 ms batch ≈ ≤ 350 bytes JSON,
  ≤ 5 msg/s. Nothing to enforce in code beyond the dead-band; noted for the observer's
  quantization (§7).

### 6.6 Interruption (spec §6.3)

On `InterruptionFrame` (frame path, before forwarding):

1. `self._generation += 1` (in-flight results become discardable).
2. Clear all contexts (buffers, counters, pending batches); `self._last_emitted_pts` stays.
3. `await analyzer.reset()` — per-utterance state cleared, **adaptive state preserved**.
4. Do not touch the transport — the recreated clock task discards unplayed batches
   automatically (§2.1).

`TTSStoppedFrame` for a context already cleared by interruption: no-op (context lookup
fails silently).

### 6.7 Disabled / failed paths

- `enabled=False` (constructor or `LipsyncUpdateSettingsFrame`): ingest returns
  immediately; task parks on `_wake`; zero CPU (spec §9). Toggling on mid-utterance starts
  analysis from the *next* context (no retroactive analysis — document in the param
  docstring).
- `_failed=True` (analysis task crashed): identical to disabled, permanent for the session,
  after one `push_error(..., fatal=False)`.

### 6.8 Stats counters (benchmark seam)

Read-only `LipsyncProcessor.stats` property returning a dict of plain int counters —
`batches_emitted, keyframes_emitted, events_emitted, pts_clamped, bytes_dropped,
contexts_opened, contexts_evicted` — incremented inline where the work happens (zero
hot-path cost). Consumed by
[benchmark-harness-performance.md](benchmark-harness-performance.md) (clamp/backpressure
visibility) and useful in tests.

**Definition of done (M3):** `test_lipsync_processor.py` skipped placeholders implemented
and green (§8.3), including passthrough, chunk invariance, offset determinism, interruption.

---

## 7. M4 — RTVI observer wiring

`observer.py` already has `bot_lipsync_enabled: bool = False` and a TODO at the dispatch
site. Implement exactly per the `AggregatedTextProgressFrame` precedent:

```python
elif isinstance(frame, TTSLipsyncFrame) and self._params.bot_lipsync_enabled:
    if not isinstance(src, BaseOutputTransport):
        # Only handle the frame once it has gone through the transport (released
        # from the clock queue at pts) so message delivery is playout-timed.
        mark_as_seen = False
    else:
        await self._handle_lipsync(frame)
```

```python
async def _handle_lipsync(self, frame: TTSLipsyncFrame):
    """Send a bot-tts-lipsync message for an analyzed window of TTS audio."""
    q = lambda v: round(v, 2)          # wire compaction, spec §9: 2-decimal quantization
    data = RTVI.LipsyncMessageData(
        ctx=frame.context_id,
        kf=[(q(k.offset), q(k.openness), q(k.width), q(k.rounding),
             q(k.energy), q(k.pitch), q(k.confidence)) for k in frame.keyframes],
        ev=[(q(e.offset), str(e.kind), q(e.duration), q(e.confidence)) for e in frame.events],
    )
    await self.send_rtvi_message(RTVI.BotTTSLipsyncMessage(data=data))
```

Details: add `TTSLipsyncFrame` to the observer's frames import; `version`/`t0` keep their
model defaults (1 / 0.0); `offset` gets 2-decimal treatment too — at 20 ms hop resolution,
10 ms wire precision is sufficient and matches the spec's size budget (revisit only if a
client needs finer). `str(e.kind)` serializes the `StrEnum` to `"closure"|"nasal"|"silence"`.

**Definition of done (M4):** observer unit test green (§8.4); manual check of one real
message against the spec §7.1 JSON example.

---

## 8. Unit tests (Pipecat conventions)

Conventions (from repo tests + AGENTS.md): plain `unittest` classes, async cases in
`unittest.IsolatedAsyncioTestCase`, pipeline behavior via
`pipecat.tests.utils.run_test(...)`, delays via `SleepFrame`, run with `uv run pytest`.
License header, no module docstring (matches existing test files).

### 8.1 `tests/test_lipsync_dsp.py` (new file, M1)

Synthetic-signal helpers at the top of the file (test-only, pure numpy):

```python
def synth_vowel(f1, f2, f3=2900, f0=120, secs=0.5, fs=16000):
    """Impulse train through cascaded 2nd-order resonators (all-pole vowel model)."""
    n = int(secs * fs)
    x = np.zeros(n, dtype=np.float32)
    x[:: int(fs / f0)] = 1.0
    for fc, bw in ((f1, 60), (f2, 80), (f3, 120)):
        r = np.exp(-np.pi * bw / fs)
        c1, c2 = 2 * r * np.cos(2 * np.pi * fc / fs), -r * r
        y = np.zeros_like(x)
        for i in range(n):                      # small-n IIR; test-only, clarity > speed
            y[i] = x[i] + c1 * y[i - 1] + c2 * y[i - 2]
        x = y
    return x / (np.abs(x).max() + 1e-9)
```

| Test | Assertion |
|---|---|
| `test_levinson_reconstructs_ar_coefficients` | Generate AR(4) process with known `a`; recovered coefficients ≈ within 5% |
| `test_formants_recovered_from_synthetic_vowels` | /a/ (700, 1200), /i/ (300, 2300), /u/ (300, 800): F1/F2 within **±40 Hz** (spec §11.1) |
| `test_formants_implausible_on_silence_and_noise` | zeros → `plausible=False`, no NaN/inf; white noise → no crash |
| `test_pitch_on_synthetic_glottal_train` | f0=120 Hz vowel → voiced, frequency within ±5 Hz; f0=280 Hz within ±8 Hz |
| `test_unvoiced_noise_not_voiced` | white noise → `voiced=False` |
| `test_rms_energy_scales` | 2× amplitude → 2× RMS (±1e-3); zeros → ~0 without error |
| `test_spectral_low_band_ratio` | 200 Hz sine → ratio ≈ 1; 3 kHz sine → ratio ≈ 0; centroids ordered |
| `test_p2_estimator_matches_numpy_percentile` | 10k lognormal samples: P5/P95 within 3% of `np.percentile` |

### 8.2 Analyzer-level tests (same file, M2) — no pipeline, direct calls

| Test | Assertion |
|---|---|
| `test_vowel_openness_ordering` | after 1 s convergence feed: openness(/a/) > openness(/i/) and > openness(/u/); width(/i/) > width(/u/) |
| `test_nasal_override` | synthetic nasal (voiced buzz low-passed at 400 Hz): NASAL event emitted; concurrent keyframes have openness < 0.15 |
| `test_silence_event_once` | 1 s of near-zero samples → exactly one SILENCE event |
| `test_closure_detection` | vowel–40 ms gap–vowel → one CLOSURE with duration ≈ 0.04 ± hop; vowel–500 ms gap–vowel → no CLOSURE (exceeds 250 ms bound) |
| `test_conditioning_caps_keyframe_rate` | sustained identical vowel → ≤ 6 keyframes/s (heartbeat only) |
| `test_analyzer_chunk_invariance` | same PCM via 160-sample vs 5120-sample `analyze()` calls → identical keyframe offsets/values (atol 1e-5) |
| `test_reset_preserves_adaptation` | converge on /a/-/i/ corpus; `reset()`; `_voiced_frames` unchanged, per-utterance state cleared |

### 8.3 `tests/test_lipsync_processor.py` — implement the skipped placeholders (M3)

Shared helper: `make_tts_frames(pcm_f32, sample_rate, context_id, chunk_ms)` → 
`[TTSStartedFrame(ctx), *TTSAudioRawFrame chunks, TTSStoppedFrame(ctx)]`.

```python
async def test_passthrough_forwards_all_frames_unmodified(self):
    processor = LipsyncProcessor()
    frames = make_tts_frames(synth_vowel(700, 1200), 24000, "ctx-1", chunk_ms=10)
    received_down, _ = await run_test(
        processor,
        frames_to_send=frames,
        expected_down_frames=None,      # inspect manually: order + identity
    )
    sent_ids = [f.id for f in frames]
    got_ids = [f.id for f in received_down if f.id in set(sent_ids)]
    self.assertEqual(got_ids, sent_ids)          # same objects, same order
    lipsync = [f for f in received_down if isinstance(f, TTSLipsyncFrame)]
    self.assertTrue(lipsync)                     # and analysis produced output
    self.assertTrue(all(f.context_id == "ctx-1" for f in lipsync))
    self.assertTrue(all(f.pts for f in lipsync))
    self.assertEqual([f.pts for f in lipsync], sorted(f.pts for f in lipsync))
```

| Test | Approach |
|---|---|
| `test_output_invariant_to_ingest_chunking` | run twice (10 ms vs 320 ms chunks); compare flattened `(offset, openness, width, rounding)` tuples, atol 1e-4; batch boundaries may differ — compare keyframes, not batches |
| `test_offsets_derived_from_sample_counts_under_jitter` | interleave `SleepFrame(sleep=0.05)` between audio chunks in one run only; keyframe offsets identical to the no-sleep run (pts differ — that's wall clock; offsets must not) |
| `test_interruption_clears_buffers_and_preserves_adaptive_state` | send half an utterance, `InterruptionFrame()`, then a fresh context; assert: no `TTSLipsyncFrame` whose `window_start` exceeds the audio-time actually sent pre-interrupt appears after the interrupt; new context produces frames; `processor._analyzer._voiced_frames > 0` before new context starts (adaptation survived) |
| `test_update_settings_disables_and_reenables` | `LipsyncUpdateSettingsFrame(settings={"enabled": False})` → context produces no lipsync frames; re-enable → next context produces frames |
| `test_overlapping_contexts_isolated` | interleave two contexts' audio; every emitted frame's keyframe count > 0 and `context_id` ∈ expected; no frame mixes offsets from the other context (offsets ≤ that context's audio length) |
| `test_stale_context_audio_ignored` | `TTSAudioRawFrame` with unknown `context_id` (no TTSStartedFrame) → no crash, no output for it |
| `test_scheduling_lead_and_clamp` | with a 10 s `scheduling_lead_ms`, all pts clamp to ≈ now (monotonic, none in the past vs `SystemClock` reading taken before run) |
| `test_analyzer_failure_degrades_to_passthrough` | inject analyzer whose `analyze` raises; assert `ErrorFrame` upstream (run_test `expected_up_frames=[ErrorFrame]`), all input frames still forwarded, no lipsync output, no unhandled task error |

(Keep `TestLipsyncScaffolding` construction smoke tests; they stay green.)

### 8.4 Observer test (M4) — extend `tests/test_lipsync_processor.py` or follow `test_rtvi_observer_config.py` patterns

- Build `RTVIObserver` with `RTVIObserverParams(bot_lipsync_enabled=True)` over a
  transport-terminated `run_test` pipeline (see `enable_rtvi=True` in `run_test`), push a
  `TTSLipsyncFrame` with one keyframe + one event, assert exactly one sent message with
  `type == "bot-tts-lipsync"`, positional arrays match, floats quantized to 2 decimals, and
  `kind` serialized as `"closure"`.
- Negative: `bot_lipsync_enabled=False` (default) → no message.
- Gating: frame pushed from a non-transport src → no message (mirrors the
  `AggregatedTextProgressFrame` tests if present; otherwise assert via observer internals).

---

## 9. Using it in a bot — code examples

### 9.1 Quickstart bot.py (matches the repo's current `pipecat init quickstart` output)

The diff from a stock quickstart bot is **three lines**: import, one pipeline entry, one
observer param.

```python
from pipecat.processors.audio.lipsync_processor import LipsyncProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserverParams

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"), ...)
    llm = OpenAIResponsesLLMService(api_key=os.getenv("OPENAI_API_KEY"), ...)

    lipsync = LipsyncProcessor()                 # defaults: DSP tier, 200 ms batches

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            lipsync,                             # ← between TTS and transport.output()
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        rtvi_observer_params=RTVIObserverParams(bot_lipsync_enabled=True),  # ← RTVI gate
    )
    ...
```

Clients then receive, just ahead of the corresponding audio (spec §7.1):

```json
{
  "label": "rtvi-ai",
  "type": "bot-tts-lipsync",
  "data": {
    "version": 1,
    "ctx": "a1b2c3",
    "t0": 0.0,
    "kf": [[0.42, 0.61, 0.35, 0.1, 0.7, 0.44, 0.9], [0.46, 0.55, 0.4, 0.12, 0.68, 0.44, 0.9]],
    "ev": [[0.55, "closure", 0.12, 0.8]]
  }
}
```

with `kf` = `[offset, openness, width, rounding, energy, pitch, confidence]` and offsets in
seconds from the first audio of context `ctx`.

### 9.2 Tuning parameters

```python
from pipecat.processors.audio.lipsync_processor import LipsyncParams, LipsyncProcessor

lipsync = LipsyncProcessor(
    params=LipsyncParams(
        batch_window_ms=200,      # audio-time per emitted TTSLipsyncFrame
        scheduling_lead_ms=200,   # delivery runway ahead of playout
        dead_band=0.04,           # min param delta to emit a keyframe
        heartbeat_ms=240,         # max keyframe gap while speaking
        emit_pitch=False,         # drop fields you don't render
    )
)
```

### 9.3 Runtime toggle (e.g. only animate when an avatar is on screen)

```python
from pipecat.frames.frames import LipsyncUpdateSettingsFrame

# From any event handler with access to the worker:
await worker.queue_frames([LipsyncUpdateSettingsFrame(settings={"enabled": False})])
# ... later:
await worker.queue_frames([LipsyncUpdateSettingsFrame(settings={"enabled": True})])
```

Disabled costs zero CPU (the analysis task parks); the processor keeps forwarding frames.

### 9.4 Explicit RTVI setup (when not using the worker's built-in RTVI)

```python
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIObserverParams, RTVIProcessor

rtvi = RTVIProcessor()
pipeline = Pipeline([transport.input(), rtvi, stt, ..., tts, lipsync, transport.output(), ...])
worker = PipelineWorker(
    pipeline,
    observers=[RTVIObserver(rtvi, params=RTVIObserverParams(bot_lipsync_enabled=True))],
)
```

### 9.5 Custom analyzer (the §8 seam — documentation-only for v1)

```python
from pipecat.audio.lipsync.base_lipsync_analyzer import BaseLipsyncAnalyzer

class MyAnalyzer(BaseLipsyncAnalyzer):
    ...  # start / analyze / flush / reset

lipsync = LipsyncProcessor(analyzer=MyAnalyzer())
```

Tier selection is explicit (constructor), never auto-negotiated (spec §8).

---

## 10. PR checklist (M5)

- [ ] All unit tests green: `uv run pytest src/tests/test_lipsync_dsp.py src/tests/test_lipsync_processor.py` (from the harness root) and the full suite `uv run pytest` inside `src/` is no worse than `main`.
- [ ] `uvx ruff@0.15.14 check` and `format --check` clean on all touched files.
- [ ] Docstrings finalized (repo `docstring` skill can polish); no `TODO(lipsync)` markers remain in implemented code paths.
- [ ] Manual end-to-end pass with the harness `bot.py` (§9.1 diff applied): keyframes arrive in the browser console via the prebuilt UI, timed ahead of audio; barge-in stops messages immediately.
- [ ] Foundational example added upstream (`examples/foundational/`, follow existing numbering) mirroring §9.1 — small, runnable, commented.
- [ ] Changelog fragment via the repo `changelog` skill (towncrier, named by PR number, `.added.md`): one entry for `LipsyncProcessor` + `bot-tts-lipsync` (experimental).
- [ ] PR body: built from this plan's §1 table + spec link; mark the feature **experimental** (spec §12: wire format versioned `1`).
- [ ] `pyproject.toml` / `uv.lock` untouched (no new deps — verify with `git diff`).
- [ ] Spec §13 open questions carried into the PR description as explicit non-goals:
  `voice_id`-keyed adaptation (deferred), adaptive `scheduling_lead` (deferred), msgpack
  encoding (deferred until measured), multi-destination (resolved — destination copied from
  source audio frames, §2.1).

## 11. Decisions log (deviations from / refinements of the tech spec)

| Topic | Spec said | Implemented as | Why |
|---|---|---|---|
| `LipsyncParams` | `@dataclass` sketch | pydantic `BaseModel` | Repo convention (AGENTS.md: params are BaseModel; `VADParams` precedent) |
| Keyframe/event types | defined next to frames | `pipecat/audio/lipsync/types.py` | Mirrors `audio/dtmf/types.py` → `frames.py` import pattern |
| Event `kind` | `str` literal | `LipsyncEventKind(StrEnum)` | Type-safe, serializes to the same strings (`KeypadEntry` precedent) |
| Settings frame | "pattern: `TTSUpdateSettingsFrame`" | plain `ControlFrame` with `settings` mapping | Processor is not an `AIService`; the service settings machinery (delta objects, targeting) doesn't apply |
| Transport routing | "reuses clock queue" (assumed work needed) | **zero transport changes** | `_handle_frame`'s generic `elif frame.pts:` branch (verified §2.1) |
| Resampling home | analyzer front-end | processor ingest; analyzer contract is 16 kHz float32 | Processor owns bytes + per-context stream resamplers; keeps analyzer deterministic for tests |
| Batch finalize | at window close | window close **+ 0.4 s analysis-cursor horizon** (`EVENT_FINALIZE_HORIZON_SEC`) | Events are final only after max closure duration (250 ms) + confirmation window (150 ms); keeps event offsets inside window bounds |
| New deps / extra | optional extras table | none; no `lipsync` extra | Tier 0 needs only core numpy+soxr; decided 2026-07-14 |

Discovered during implementation (M1–M4, 2026-07-14):

| Decision | Why |
|---|---|
| Lag-zero regularization `1 + 1e-6`, not spec §5.3's `1.0001` | Measured: 1e-4 noise injection visibly shrinks pole radii on high-prediction-gain AR signals (Levinson matched a direct Toeplitz solve; the reg factor was the error source). 1e-6 keeps the anti-singularity property |
| `dsp.lpc_coefficients()` public; `lpc_formants(frame, lpc=None)` | Formants and residual pitch share one Levinson run per frame |
| Pitch clarity compensated by the Hamming window's autocorrelation (clamped at 0.1) + octave guard (smallest lag ≥ 0.85 × peak) | A perfectly periodic 120 Hz signal peaks at only ~0.5 of lag zero under the window, breaking the 0.35 voiced threshold; bare argmax can land an octave down |
| Nasal spectral features from the raw (un-emphasized) windowed spectrum | Pre-emphasis brightens exactly the low band the detector keys on |
| Nasal emits one entry event (`duration=0`); no exit event | Keyframes carry the exit (openness rises when the override ends); a second event would just re-trigger clients |
| Noise floor = min-statistics capped at 5% of the recent peak | Sustained speech with no pauses inside the 1 s window otherwise inflates the floor into the speech range, gating everything as silence |
| `LipsyncFrameResult.processed_up_to` field | The processor's batcher needs the analysis cursor; keyframe offsets alone are too sparse (dead-band) |
| Conditioning params (`dead_band`, `heartbeat_ms`) live on `FormantLipsyncAnalyzer`; `LipsyncParams` forwards them to the default analyzer | Conditioning is analyzer-internal; explicitly constructed analyzers own their settings |
| Contexts analyzed strictly FIFO (newer context waits until the active one flushes) | The analyzer holds one utterance's state; TTS contexts don't overlap in normal flow, and interruption clears everything anyway |
| Synthetic vowel fixtures include a glottal-tilt stage (one-pole 0.97 lowpass) | Analysis pre-emphasis assumes speech's natural spectral slope; a flat impulse train biases LPC formants up +40..80 Hz |

Accuracy-tuning pass (2026-07-14, benchmark-driven; composite 28.8 → 70.1 — see accuracy-improvements.md):

| Decision | Why |
|---|---|
| Formant slots assigned by band (`F1_BAND_HZ`..) via exhaustive filled-count-first assignment with prev-frame continuity targets | Ascending-rank slotting promoted F3 into a damped F2's slot (f2_mae 625 Hz) and let spurious low poles shift every slot |
| `FormantEstimate.f2_bandwidth` field; per-slot hold in the analyzer with a 3-hop cap decaying toward prior centers | Found slots stay usable when a sibling slot is empty; stale holds can't freeze an old mouth shape |
| Adaptation spike rejection (>400 Hz jumps skipped, ≤2 consecutive) instead of an Hz-median | The median added a hop of lag (MAE up, shape down); spike rejection protects the learned ranges with zero lag |
| Nasal features on the **pre-emphasized** spectrum; "damped F2" = absent, broad (>300 Hz bw), or >1200 Hz while low-band ratio >0.9 | Raw-spectrum low-band ratio saturates at ~1.0 for every voiced frame; murmurs fit narrow spurious mid-band poles |
| Nasal 1-hop fast path (ratio >0.9) + pre-latch openness soft cap (≤0.2) | Cuts override latency without loosening event hysteresis |
| `lpc_coefficients` returns `LpcResult` (coefficients + prediction gain); confidence = c_lpc × c_fit × c_snr × √c_conv | Gain separates well-modeled frames from noise (unit-tested >0.2 gap); sqrt keeps cold-start damping without flattening calibration (spec §4.4 deviation) |
| Heartbeat keyframes suppressed while a silence stretch is active; `dead_band` 0.04 → 0.05 | Silence heartbeats were 27% of pause-heavy clips' wire traffic; rate now 24/s (spec band 15–25) |
| `_CLOSURE_PEAK_FRACTION` 0.15 → 0.22 | Voiced stops keep a voice bar above the old threshold (bilabial corpus checks now 8/8; `closures_max` guard held) |

Pass-2 tuning (2026-07-15, oracle-calibrated corpus; see accuracy-improvements-2.md):

| Decision | Why |
|---|---|
| Learned range edges are P10/P90 with 60-frame convergence (was P5/P95, 150) | Range slack was ~all of the peak-openness compression (raw /a/ mapped 0.46 vs clip-normalized ideal 0.92); 30-frame and asymmetric hi-trust variants traded trajectory shape and were rejected |
| `FormantEstimate.f1_broad`: a broad F1-band root (bw 500–800 Hz) reported when the F1 slot is empty; feeds the openness mapping only — never slots, adaptation, confidence or the debug tap | F1 physically broadens with mouth opening; peak /ɑ/ frames otherwise have no F1 at all. Blanket bandwidth admission broke the f1_mae guard — evidence and measurement had to be decoupled |
| True silence rests openness at 0.15 (`_SILENCE_REST_OPENNESS`); unvoiced *speech* still decays to neutral 0.35 | Utterance-onset keyframes were broadcasting the silence-rest schwa (measured on hum onsets); a silent mouth shouldn't sit half-open |
