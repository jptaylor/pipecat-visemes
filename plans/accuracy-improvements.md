# Accuracy Improvements — the Five Baseline Findings

**Status:** COMPLETE (2026-07-14). Composite **28.8 → 70.1** (target ≥55, stretch ≥65 — both beaten). Re-baselined to 70.1 at exit; the original baseline is preserved as `results/baseline-original-28.8.json`, per-fix snapshots as `results/accuracy-after-fix-N.json`.

**Historical.** Superseded by the 2026-07-15 pass ([accuracy-improvements-2.md](accuracy-improvements-2.md), 70.9) and the 2026-09-18 DSP bundle ([deep-review-2026-09-results.md](deep-review-2026-09-results.md), 87.5 Cartesia / 90.7 Deepgram). File paths refer to the pre-`server/` tree, and the `results/…` snapshots named here are no longer tracked (only `baseline*.json` are). Current status: [README.md](README.md).

**Outcomes vs §6 targets:** `f2_mae` 625→**154** (beat stretch) · `f1_mae` 237→**72** (beat stretch) · `f1_r` 0.35→0.73 · `openness_r` 0.21→0.34 (short of 0.50) · `width_r` 0.15→0.35 (short of 0.45) · keyframe rate **24.1/s** ✓ · checks 12/22→**16/22** (short of 19) · conf gap 0.05 cold / 0.06 warm (short of 0.15 — see below).

**Deviations discovered while executing (the measure-everything discipline paid off):**
- Fix 2's Hz-median was **tried and rejected**: it lifted `f1_r` but added a hop of lag that raised MAE and hurt trajectory shape. Kept: bounded hold + adaptation spike-rejection only.
- Fix 5c's bandwidth criterion **doesn't discriminate** (murmur's spurious F2 poles are narrow, p50 138 Hz). The working criterion: F2 found above 1200 Hz while low-band ratio > 0.9 is spurious (no real vowel pairs a high F2 with a near-total low-band spectrum).
- The nasal spectral features had to move to the **pre-emphasized** spectrum: on the raw spectrum, glottal harmonics saturate the low-band ratio at ~1.0 for *every* voiced frame (even /a/), making it non-discriminative. The M2-era "don't pre-emphasize the nasal FFT" reasoning was empirically backwards.
- The vowel-oo/ee "false" nasals turned out to be **true positives** — those sentences are full of real /n, m/ segments.
- Confidence gap target missed and accepted: component decomposition showed the residual "inaccurate" frames are largely well-modeled frames that *disagree with Praat* (reference-level), so pushing per-frame evidence further would overfit the reference. The vowel-vs-noise unit test proves the factors discriminate signal quality (gap > 0.2); warm levels are 0.17 vs 0.11.

**Residual findings for the next pass (5a family, 6 failing checks):**
- The vowel-probe *sentence-mean* expectations were miscalibrated at corpus authoring time: voiced F1 median on vowel-aa is 258 Hz (most voiced frames are transitions/schwa/nasals, not peak /a/) — even a Praat-perfect tracker would fail `openness_mean_voiced ≥ 0.45`. Next corpus revision should use peak-reaching statistics (p90-style) with honest bars.
- Genuine dynamic-range compression remains: raw /a/ openness peaks at ~0.54 (should approach ~0.8) — prior span [250, 900] is wide for this voice and convergence blends slowly. Candidate lever: faster/loss-aware range convergence.
- `nasal-hum` t1 `openness_p90` 0.347 vs 0.30 — pre-latch onset hops; marginal.

---

**Status (original):** Ready for implementation
**Input:** First frozen accuracy baseline, composite **28.8** (`benchmarks/results/baseline.json`, 2026-07-14: cartesia quickstart voice, 12 sentences × 2 takes, cold analyzer).
**Goal:** Address the five issues the accuracy harness surfaced, one change-set at a time, measured by `uv run python -m benchmarks.accuracy --offline --compare` after every change.
**Companions:** [benchmark-harness-accuracy.md](benchmark-harness-accuracy.md) (how scores are computed), [pipecat-implementation.md](pipecat-implementation.md) (the code being tuned; update its §11 decisions log as fixes land).

---

## 0. Method and rules

**Work order matters.** Fix 1 (F2 slotting) is upstream of everything: slot errors corrupt F1 (Fix 2), poison the adaptive range estimators, defeat confidence calibration (Fix 3), inflate keyframe jitter (Fix 4), and directly cause the vowel-oo failures (Fix 5a). Land fixes in numbered order; re-measure between each.

1. One fix = one change-set, measured before the next starts.
2. **Composite thresholds and weights never move** (frozen 2026-07-14). The corpus and its expectations are also frozen until the final step — corpus edits change scores, so any expectation refinements are bundled at the end with an explicit re-baseline.
3. `baseline.json` stays the *original* 28.8 file throughout, so `--compare` always shows cumulative progress. Snapshot per-fix results as `results/accuracy-after-fix-N.json` (just rename the timestamped output).
4. Pipecat unit tests stay green after every fix (`uv run pytest src/tests/test_lipsync_dsp.py src/tests/test_lipsync_processor.py`); each fix section lists the tests it must update or add.
5. After all fixes: live bot sanity pass (run `bot.py`, confirm nasal override, keyframe cadence, message sizes) and a `--warm` benchmark run for the steady-state picture.

**Diagnosis workflow when a metric surprises you:** dump aligned tracks (ours vs Praat) for the worst clip — the pattern used to find these issues:

```python
kfs, evs, debug = await analyze_clip(clip)
ref = compute_reference(clip, offsets, ceiling)
# print (t, d.f1, ref.f1[i], d.f2, ref.f2[i], d.voiced, ref.voiced[i], d.rms) rows
```

---

## 1. F2 slotting — the dominant DSP error

**Evidence.** `f2_mae` 625 ± 298 Hz. Aligned dumps show the signature: ours F1 225 vs Praat 232 (correct) while ours F2 3398 vs Praat 1115 — a valid F1 with an F3-region root labeled F2.

**Root cause.** `dsp.lpc_formants` keeps every root in one global band [200, 3500] with bandwidth < 500, sorts ascending, and assigns F1/F2/F3 purely by rank. Two failure modes:

- **(a) Missing F2:** the true F2 pole is damped or merged (bandwidth ≥ 500, common in back vowels /u, o/ and nasalized segments) → drops out of the list → the actual F3 (~2400–3400 Hz) is promoted into the F2 slot. This is the 3398-vs-1115 signature and directly causes Fix 5a's failures.
- **(b) Spurious low root:** a nasal pole or strong harmonic below ~300 Hz becomes "F1", shifting the true F1 into the F2 slot and F2 into F3 — corrupting **both** F1 and F2 (feeds Fix 2).

**Fix: band-constrained slot assignment with a continuity hint.** In `dsp.py`:

1. New module constants (single source of truth — the nasal detector's hardcoded `800.0 <= f2 <= 2500.0` in the analyzer switches to these too):

```python
F1_BAND_HZ = (200.0, 1000.0)
F2_BAND_HZ = (650.0, 2600.0)   # lower edge below male /u,o/ F2 (~700 Hz)
F3_BAND_HZ = (1800.0, 3500.0)
```

2. `lpc_formants(frame, lpc=None, prev=None)` — `prev: FormantEstimate | None` is the previous frame's estimate, passed by the analyzer. Slotting algorithm:
   - Collect candidates as today (positive imag, 0 < bw < 500, freq ∈ [200, 3500]).
   - **F1**: candidates within `F1_BAND_HZ`; pick the one closest to `prev.f1` when `prev` has one, else the lowest. Remove it from the pool.
   - **F2**: candidates within `F2_BAND_HZ` **and** > F1 + 150 Hz (when F1 found); closest-to-prev else lowest. Remove.
   - **F3**: candidates within `F3_BAND_HZ` and > F2 + 200 Hz; lowest/closest.
   - An empty slot stays `0.0`. `plausible` becomes `f1 > 0 and f2 > 0` (same public meaning, stricter mechanics).
3. **Per-slot hold in the analyzer** (`_process_hop`): replace the all-or-nothing `if plausible: use all else: hold all` with per-slot — a found F1 is used even when F2 is missing; only the missing slot holds its previous value. Track per-slot `found` flags for Fix 3's confidence.

**Tests.**
- Existing `test_formants_recovered_from_synthetic_vowels` must stay green (±40 Hz) — band slotting can't break clean vowels.
- New: `test_damped_f2_not_mis_slotted` — synthetic vowel with the F2 resonator bandwidth widened to ~600 Hz (so its root fails the bw filter): assert F2 comes back 0.0/held and F3 (2900) is **not** reported as F2.
- New: `test_low_spurious_root_does_not_shift_slots` — add a 250 Hz narrow resonator to an /a/ (700/1200): F1 must stay ≈700-or-250-consistent per band rule (F1 band admits both; closest-to-prev disambiguates after the first frame — assert stability, not a specific pick).
- Update `test_nasal_override` only if the unified band constants change the f2_damped result (they shouldn't: 650–2600 vs 800–2500 — verify).

**Verify (benchmark):** `f2_mae` < 300 (stretch < 200); `f1_mae` should also drop (mode b); `width_r` up materially; vowel-oo checks likely start passing.

**Risks.** Band edges are new tunables: /i/ has F2 ~2200–2500 (inside), some female /i/ up to ~2800 — if `f2_mae` improves but /i/ clips (vowel-ee) regress, widen `F2_BAND_HZ` upper edge before touching anything else. Continuity (`prev`) can lock onto a wrong track after an error; the per-frame `found` reset plus Fix 2's hold cap bound the damage.

---

## 2. F1 transition errors

**Evidence.** `f1_mae` 237 ± 255 Hz — bimodal: steady vowels within ~20 Hz of Praat (234↔252, 709↔738), transitions/consonant clusters wildly off. Post-convergence MAE (269) is *worse* than full-clip (237): the error is segment-type-dependent, not warmup.

**Root causes.**
- (a) Slot corruption (Fix 1 mode b) — expected to remove a large share.
- (b) 25 ms windows straddling fast transitions merge two vocal-tract configurations; single-frame spikes result.
- (c) Low-energy voiced frames just above the silence gate produce noisy LPC roots that are scored (and fed to adaptation) at full weight.
- (d) Raw spiky F1/F2 values feed the P5/P95 estimators directly, corrupting the adaptive ranges that later normalize everything.

**Fix: median-filter the Hz tracks and bound the hold policy.** In `formant_lipsync_analyzer.py`:

1. **3-tap median on raw F1/F2 Hz** after slotting/hold, *before* both `_update_adaptation` and parameter mapping. (The existing median-3 on normalized params stays; total added lag = 1 hop = 20 ms, still well inside the 200 ms scheduling lead — note in the processor docstring's latency accounting.)
2. **Bounded hold:** currently a missing slot holds its previous value forever with only `c_lpc = 0.3` marking it. Cap consecutive holds at 3 frames per slot; beyond that, decay the held value toward the prior-band center (e.g. 10%/hop) so a long implausible stretch drifts to neutral instead of freezing a stale shape.
3. **Spike rejection into adaptation:** skip `_update_adaptation` for a frame whose post-median F1 or F2 jumped > 400 Hz from the previous accepted frame (the estimate is still *used* for mapping — conditioning smooths it — but doesn't pollute the learned ranges).

**Tests.** Existing analyzer tests stay green; add `test_adaptation_ignores_single_frame_spikes` — feed a steady /a/ with one corrupted frame injected (splice 25 ms of noise); assert learned P95(F1) stays within a few Hz of the clean run.

**Verify:** `f1_mae` < 120 mean and std well under 150; `openness_r` > 0.5; aligned dump on `harvard-02` (worst clip) shows spikes gone.

---

## 3. Confidence calibration

**Evidence.** `conf_on_accurate` 0.08 vs `conf_on_inaccurate` 0.06 — a 0.02 gap; confidence doesn't know when the DSP is wrong. `mean_confidence` 0.04 on cold clips.

**Root causes.**
- (a) Multiplicative `c_conv` (= voiced/150) crushes everything on 2–4 s cold clips (max ~0.3–0.6), flattening whatever discrimination `c_lpc`/`c_snr` carry.
- (b) `c_lpc` can't see slot errors: a consistently mis-slotted F2 is "plausible" with small deltas → high c_lpc while 2000 Hz wrong.
- (c) `c_snr` saturates: the noise floor is peak-capped, so during speech SNR is huge and `c_snr ≈ 1` everywhere — no signal.

**Fix: add per-slot and fit-quality evidence; soften convergence damping.**

1. **Per-slot plausibility → c_lpc** (from Fix 1): `c_slot = 1.0` both found, `0.5` one held, `0.3` both held. `c_lpc = c_slot × (0.3 + 0.7 × delta_term)` as today.
2. **LPC prediction gain**: Levinson already computes the final prediction error internally — expose it. Change `lpc_coefficients` to return a small NamedTuple `LpcResult(coefficients, prediction_gain)` where `prediction_gain = r[0] / err_final` (update its two callers + dsp tests in the same change). Well-modeled voiced speech has gain ≫ 100; transitions/noise are low. New factor: `c_fit = clamp01(log10(gain) / 3)`.
3. **Rebalance:** `confidence = c_lpc × c_fit × c_snr × sqrt(c_conv)`. The sqrt is a deliberate deviation from tech-spec §4.4's plain product: cold-start blending toward neutral is preserved (sqrt(0.2) ≈ 0.45 still damps hard) but no longer flattens calibration into unmeasurability. Record in the implementation plan's decisions log.
4. Leave `c_snr` as is once (1)–(3) land; only revisit if the gap target still misses.

**Tests.** Extend the debug-tap test: on a clip that is half synthetic vowel, half noise, mean confidence on the vowel half must exceed the noise half by ≥ 0.2. Update any test asserting exact confidence values.

**Verify:** `conf_on_accurate − conf_on_inaccurate` ≥ 0.15; `mean_confidence` ≥ 0.15 cold / ≥ 0.4 with `--warm`. (Calibration stays a *reported* metric — not scored — since thresholds are frozen.)

---

## 4. Keyframe rate

**Evidence (decomposed 2026-07-14):**

| clip | duration | speech | kfs | rate/duration | rate/speech | kfs in silence |
|---|---|---|---|---|---|---|
| vowel-aa | 2.79 s | 2.18 s | 78 | 27.9 | 35.8 | 11 |
| harvard-01 | 2.15 s | 1.58 s | 66 | 30.6 | 41.8 | 6 |
| nasal-hum | 1.91 s | 0.82 s | 45 | 23.5 | 54.9 | 12 |

Three separable causes: the harness denominator inflates the headline number (duration-based ≈ 24–31/s matches the live bot's ~26/s); **6–12 keyframes per clip are silence heartbeats** (27% of nasal-hum's total — pure wire waste, confidence ≈ 0, client already at neutral from the SILENCE event); and the genuine speaking-time rate ~28/s still exceeds the spec's 15–25 because param jitter (Fixes 1–2) defeats the 0.04 dead-band.

**Fix, in three parts.**

1. **Harness (do first, it's a measurement bug):** report both `keyframe_rate` (duration-based — comparable to live) and `keyframe_rate_speech` (speech-based). No composite impact (not a scored component).
2. **Suppress silence heartbeats** (`_maybe_emit_keyframe`): when the silence run is active (`_silence_run_hops ≥ _SILENCE_EVENT_HOPS`), emit keyframes only for `event_fired` or dead-band movement — never for heartbeat alone. The SILENCE event already parked the client at neutral; heartbeats there pin nothing. Expected saving: ~4/s during every pause. Update the tech-spec-derived docstring ("heartbeat while audio is being analyzed" → "while speech is active").
3. **Re-measure after Fixes 1–2** — cleaner tracks mean fewer dead-band crossings. Only if speaking-rate still > 25/s: raise `dead_band` 0.04 → 0.05 as the last resort (perceptual trade; note in decisions log).

**Tests.** `test_conditioning_caps_keyframe_rate` gets a companion: sustained vowel followed by 1 s silence → zero keyframes after the SILENCE event fires (except any event-fired ones).

**Verify:** duration-based rate 15–25 in benchmark **and** in a live bot session; `bytes_per_msg` stays ≤ ~350.

---

## 5. Failing L3 checks (three distinct problems)

### 5a. `vowel-oo`: `rounding_mean_voiced_min` + `width_mean_voiced_max` (4 failures)

**Root cause:** back vowels /u, o/ are exactly the damped-F2 case — Fix 1 mode (a) replaces their F2 (~800 Hz) with F3 (~2800 Hz), so width reads *spread* instead of *back* and the rounding term (keyed off low F2) never fires.

**Action:** expected to be substantially fixed by Fix 1 — **measure before changing anything else.** If residual failures remain, revisit the rounding formula with these options, in order: (i) compute rounding from F2 in Hz against an absolute band (`clamp01((1200 − f2_hz) / 400)`, prior-shifted for high voices) instead of the normalized-width midpoint — less dependent on adaptive range quality; (ii) relax the openness gate window (currently ≈[0.15, 0.75]) — /u/ openness may sit below 0.15 cold.

### 5b. `bilabial-bob`: `closures_min` (2 failures — fewer than 5 closures detected)

**Root cause (hypotheses, verify with the closure-threshold trace on the clip):** voiced stops /b, g/ keep a voice bar — energy dips but not below `0.15 × recent_peak`, so runs never qualify; and sentence-initial "Bob" has < 150 ms of preceding speech, failing the `speech_before` requirement.

**Fix (tune against the corpus, offline loop):**
1. Raise `_CLOSURE_PEAK_FRACTION` 0.15 → ~0.22 (catches partial dips). The over-fire brake is already in the corpus: `closures_max: 14` on both bilabial sentences must keep passing, and vowel/harvard clips must not sprout event storms (watch `closures` counts across all clips in the JSON).
2. If initial-"Bob" specifically is the miss: reduce the required pre-speech evidence to ≥ 1 speech hop within the window rather than the current window `any` over a possibly-empty slice at utterance start (verify the slice isn't degenerate at t < 150 ms first — that would be a plain bug).

### 5c. `nasal-hum`: `openness_p90_max` (2 failures — p90 of voiced-hop openness > 0.30)

**Root cause:** ≥ 10% of voiced hum hops are *not* overridden. The 2-hop enter hysteresis costs ~40 ms per "Hmm" (×3 per clip), breathy onsets delay the voiced flag, and mid-hum frames whose `low_band_ratio` dips below 0.6 drop out of `nasal_now`, releasing the override.

**Fix — graded response instead of binary latch:**
1. **Fast path:** enter after 1 hop (not 2) when the evidence is strong (`low_band_ratio > 0.75`).
2. **Soft cap pre-latch:** any voiced frame with `f2_damped and centroid < 800 and low_band_ratio > 0.5` caps the openness *target* at 0.2 even when the state machine hasn't latched — the event stays hysteresis-gated (no event spam), only the continuous signal reacts faster.
3. Only if still failing: lower `_NASAL_LOW_RATIO_MIN` 0.6 → 0.55, watching for false nasals on /u, o/ clips (vowel-oo's `nasals` count in the JSON is the guard — it should stay ~0; add `nasals_max: 2` to vowel-oo in the final corpus-refinement step).

**Tests:** 5b/5c threshold changes will move `test_closure_detection` and `test_nasal_override` boundaries — update the synthetic gaps/assertions alongside, keeping the *intent* (bounded duration, surrounded-by-speech, override active) not the exact constants.

---

## 6. Success criteria

| Metric | Baseline | Target | Stretch |
|---|---|---|---|
| `f2_mae_hz` | 625 | < 300 | < 200 |
| `f1_mae_hz` | 237 | < 120 | < 90 |
| `openness_r` | 0.21 | > 0.50 | > 0.65 |
| `width_r` | 0.15 | > 0.45 | > 0.60 |
| conf gap (accurate − inaccurate) | +0.02 | ≥ 0.15 | ≥ 0.25 |
| keyframe rate (duration-based) | 24–31/s | 15–25/s | — |
| L3 checks passing | 12/22 | ≥ 19/22 | 22/22 |
| **Composite** | **28.8** | **≥ 55** | **≥ 65** |

**Exit checklist:** all unit tests green · `--offline --compare` shows the cumulative gain · `--warm` run recorded · live bot sanity pass (nasal override, cadence, ~≤350 B messages) · corpus-refinement step (if any) applied with explicit re-baseline · implementation-plan §11 decisions log updated (band constants, sqrt(c_conv), silence heartbeats, any threshold moves) · memory updated.

## 7. Out of scope (noted, not attempted)

- LPC order / hop-size experiments (order 14, 10 ms hop) — only if band slotting + medians leave `f1_mae` > 120.
- Praat-side reference changes — already litigated; the reference is frozen with the thresholds.
- Making confidence calibration a *scored* composite component — revisit at re-baseline time.
- Voice-id-keyed adaptation, provider timestamp tiers — tech-spec §13/§8 territory, not accuracy tuning.
