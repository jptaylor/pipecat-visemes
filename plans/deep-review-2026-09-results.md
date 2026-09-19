# Deep review §2/§6 — validation on real fixtures (2026-09-18)

**Status:** done, two passes. DSP bundle landed behind constants; then the two leads it surfaced (conditioning lag, nasal fast path) — see the second pass at the end. Baselines re-saved and committed after each pass.
**Input:** `plans/deep-review-2026-09.md` §2 (DSP) and §6.2 (coverage), branch `claude/dazzling-fermi-cxmn8r` @ `1659e98`.
**Corpus:** 12 sentences × 2 takes × 2 voices — Cartesia `71a7ad14…` (Praat ceiling 5000, the cached fixtures behind the 70.9) and Deepgram `aura-2-helena-en` (ceiling 5500, synthesized for this run). The second voice was not in the brief; it overturned one conclusion and qualified two others, so every row below is reported for both.

Reproduce (cached fixtures, no keys): `cd server && uv run python ../plans/experiments/ab_table.py [--provider deepgram] [--ceiling N]`. Every variant states all its overrides, so the table does not depend on which defaults are landed.

## Results

Composite is the unchanged v1 composite. Coverage = L1-scored hops / Praat-voiced hops (new, reported, not scored). Each numbered row adds one change to the row it names.

**Cartesia (ceiling 5000)**

| variant | composite | f1_mae | f2_mae | f1_cov | f2_cov | openness_r | width_r | voicing | kf/s | checks |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 baseline (order 12, residual pitch) | 70.9 | 76 | 158 | 0.31 | 0.24 | 0.31 | 0.34 | 0.67 | 25.4 | 18/20 |
| 1a order 14 | 70.5 | 79 | 143 | 0.35 | 0.29 | 0.33 | 0.39 | 0.67 | 26.8 | 17/20 |
| 1b order 16 | 82.1 | 23 | 91 | 0.35 | 0.33 | 0.43 | 0.45 | 0.65 | 27.7 | 17/20 |
| 1c order 18 | 73.6 | 78 | 125 | 0.34 | 0.34 | 0.37 | 0.42 | 0.65 | 27.5 | 17/20 |
| 2a 1b + slot prior on F1,F2,F3 (λ 1) | 82.8 | 27 | 83 | 0.35 | 0.32 | 0.44 | 0.44 | 0.65 | 27.6 | 17/20 |
| 2b 1b + slot prior on F2,F3 only | 82.8 | 23 | 83 | 0.35 | 0.32 | 0.43 | 0.45 | 0.65 | 27.7 | 17/20 |
| 3 2b + LPC window 40 ms (residual pitch) | 74.2 | 16 | 109 | 0.20 | 0.18 | 0.33 | 0.34 | 0.56 | 19.9 | 16/20 |
| 4a 2b + NCC voicing, 25 ms pitch frame | 84.1 | 23 | 113 | 0.69 | 0.70 | 0.48 | 0.49 | 0.92 | 30.9 | 20/20 |
| **4b 2b + NCC voicing, 40 ms pitch frame (landed)** | **84.3** | 24 | 112 | 0.70 | 0.71 | 0.49 | 0.49 | 0.95 | 30.5 | 20/20 |
| 4c 4b + LPC window 30 ms | 84.7 | 24 | 109 | 0.70 | 0.71 | 0.48 | 0.50 | 0.95 | 30.5 | 20/20 |
| 4d 4b + LPC window 40 ms | 84.0 | 23 | 113 | 0.70 | 0.72 | 0.48 | 0.48 | 0.95 | 30.6 | 20/20 |
| 5 4b + nasal: missing F2 not "damped" | 84.1 | 24 | 112 | 0.70 | 0.71 | 0.48 | 0.48 | 0.95 | 30.7 | 20/20 |

**Deepgram (ceiling 5500)**

| variant | composite | f1_mae | f2_mae | f1_cov | f2_cov | openness_r | width_r | voicing | kf/s | checks |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 baseline | 85.6 | 33 | 87 | 0.70 | 0.55 | 0.59 | 0.62 | 0.80 | 29.7 | 17/20 |
| 1a order 14 | 86.5 | 12 | 53 | 0.70 | 0.59 | 0.56 | 0.66 | 0.80 | 30.1 | 17/20 |
| 1b order 16 | 86.5 | 21 | 64 | 0.69 | 0.62 | 0.55 | 0.67 | 0.79 | 30.4 | 18/20 |
| 1c order 18 | 86.4 | 28 | 78 | 0.68 | 0.62 | 0.54 | 0.67 | 0.79 | 30.7 | 18/20 |
| 2b 1b + slot prior on F2,F3 | 86.3 | 21 | 65 | 0.69 | 0.62 | 0.55 | 0.66 | 0.79 | 30.4 | 18/20 |
| 3 2b + LPC window 40 ms (residual pitch) | 85.8 | 17 | 59 | 0.54 | 0.48 | 0.52 | 0.66 | 0.72 | 27.8 | 18/20 |
| 4a 2b + NCC, 25 ms pitch frame | 84.2 | 23 | 75 | 0.91 | 0.81 | 0.52 | 0.71 | 0.84 | 33.6 | 17/20 |
| **4b 2b + NCC, 40 ms pitch frame (landed)** | **86.3** | 24 | 75 | 0.91 | 0.81 | 0.50 | 0.70 | 0.85 | 33.7 | 18/20 |
| 4d 4b + LPC window 40 ms | 86.6 | 27 | 76 | 0.91 | 0.82 | 0.51 | 0.72 | 0.85 | 33.7 | 18/20 |
| 5 4b + nasal: missing F2 not "damped" | 81.3 | 24 | 75 | 0.91 | 0.81 | 0.51 | 0.70 | 0.85 | 33.0 | 16/20 |

The two Deepgram failures in every row are `pause-probe:silences_min` (that voice's pause is shorter than the 300 ms SILENCE threshold; unrelated to this work). Reference sensitivity: with the ceilings swapped the bundle still reads Cartesia 69.8 → 80.7 (at 5500) and Deepgram 72.8 → 73.5 (at 5000).

CPU: 168 → 201 µs per 20 ms hop (RTF 0.0084 → 0.0101, +20 %); order 16 is +26 µs, NCC on the 40 ms frame +7 µs.

## What the review got right, and what it got wrong

1. **§6.2 coverage blind spot — confirmed, and larger than stated.** At baseline L1 scored 31 % (F1) / 24 % (F2) of Praat-voiced hops on Cartesia. About half of the loss was the *voicing detector* disagreeing with Praat, not empty slots (slot coverage 0.60 / 0.49). The predicted trap appears in rows 4a/4b: f2_mae "rises" 83 → 112 although the formant tracks are bit-identical (only the voicing mask changed). On matched hops (`plans/experiments/ab_matched.py`) residual vs NCC error is identical; the ~600 newly admitted hops per slot have median error 10 / 21 Hz with a heavy F2 tail. Baseline vs bundle on matched hops: F1 68 → 19 Hz, F2 138 → 72 Hz, with 2.2× / 2.9× the committed hops. **The v1 composite understates the bundle** because it charges f2_mae for the new hops.
2. **[DSP-1] order 16 — keep, but the headline number is partly by construction.** The best order by median error follows the Praat ceiling's pole density on *both* voices: ceiling 4500 → order 18, 5000 → 16, 5500 → 14 (`order_vs_ceiling.py`). So "f1_mae 76 → 23" is order 16 meeting a 2-poles-per-kHz reference, and this benchmark cannot separate 14 from 16. What is robust: order 16 beats order 12 in median F1 and F2 error at every ceiling on both voices and is never the worst, F2 commits rise 666 → 1134 (Cartesia) and 1163 → 1515 (Deepgram), and on exact-truth synthetic vowels with resonances above F3 order 12 is starved (F1 found on 33 % of frames, 158 Hz off) while 14/16/18 read 13 Hz at 100 % (now a unit test). Pole starvation is voice-dependent: severe on the Cartesia voice, mild on Deepgram (baseline f1_cov already 0.70).
3. **[DSP-3] slot prior — small, and harmful on F1.** Real-fixture gain is −8 Hz f2_mae on Cartesia (proxy: −40), nothing on Deepgram. With λ on all three slots it breaks F1 where the true F1 is low (harvard-05/t2: 14 → 78 Hz). Landed on F2/F3 only, where it fixes genuine lock-in on four clips (harvard-01/t1 226 → 131 Hz) at no F1 cost.
4. **LPC window 40 ms — not a lever; the proxy result does not transfer.** Row 3's regression is an artefact of the residual pitch detector (its clarity threshold is tuned to the 25 ms window; voicing 0.65 → 0.56). With voicing held fixed by NCC, 25/30/35/40 ms score 84.3 / 84.7 / 84.5 / 84.0 and openness_r is flat. The decoupled framing works as designed (closure counts identical at every length, chunk-invariant) and is kept because the NCC pitch frame uses it.
5. **[DSP-2] NCC voicing — the largest real-fixture win, better than predicted.** Voicing agreement 0.65 → 0.95 (Cartesia), 0.79 → 0.85 (Deepgram); coverage doubles on Cartesia; all 20 Cartesia checks pass, including the two "accepted residual" vowel-probe failures and nasal-hum with a much wider margin (openness_p90 0.29/0.19 → 0.13/0.10). The gate matters as stated (+0.03–0.04 agreement). Two refinements over the proposal: the threshold is flat from 0.55 to 0.70 (landed 0.6, gate 0.05 of recent peak; agreement alone hides the precision/recall trade, see `voicing_sweep.py`), and a **40 ms pitch frame** is worth +0.03 agreement over 25 ms on Cartesia and +2 composite on Deepgram (reference segment 374 vs 134 samples).
6. **Step 5 numbers differ from the review's.** Prediction gain rises only ~0.05–0.11 in log10 from order 12 to 16 (median gain 19 → 24 and 27 → 31; the review inferred a real-speech median of 2.5–3.1 and a 3.0 → 3.4 shift), so `_C_FIT_LOG10_FULL` is 3.2. Strict F1 coverage on vowel-aa at order 16 is 0.78 / 0.85 — under the 0.9 bar — so `f1_broad` stays.
7. **Nasal re-tune — the single-voice answer was wrong.** On Cartesia at order 16 every hum hop has a narrow F2 root near 1.9 kHz, so dropping "missing F2 counts as damped" looked free and cut false nasal closures on a nasal-free sentence from 28 % to 9 % of voiced hops. On Deepgram the murmur has no F2 root and the same change loses the hums (row 5: −5.0, 16/20). `_NASAL_MISSING_F2_DAMPED` stays `"always"`; thresholds unchanged; nasal-hum is 4/4 on both voices.

## Found along the way (not fixed here)

- **False nasal closures are common and invisible to the corpus.** The nasal override closes the mouth on 26–29 % of our voiced hops in harvard-03, a sentence with no nasal, and fires 3–10 NASAL events on sentences with 0–2 true nasals. The rate per voiced hop is pre-existing (harvard-03 at baseline: 45 % / 19 %), but NCC voices twice as many hops on Cartesia, so nasal-closed time over Praat-voiced hops rose 0.27 → 0.41 there (Deepgram 0.25 → 0.28). Mostly it lands on close vowels, which is why L2 barely reacts: it *adds* 0.07 openness_r on Cartesia and costs Deepgram 0.05. No check bounds it — add `nasals_max` or a nasal-active-fraction guard before touching the detector (`plans/experiments/nasal_tap.py` prints both).
- **The conditioning stage is now the biggest openness loss.** Stage-by-stage r against Praat (`openness_stages.py`): the bundle improves the *mapped* openness on both voices (Cartesia 0.35 → 0.61, Deepgram 0.73 → 0.75), but median-3 + slew + keyframing then lose 0.19–0.20 r, more than before because the target is livelier. That is why Deepgram's emitted openness_r falls 0.59 → 0.50 while its DSP improves. Review [COND-1] is the next lever, and zero-lag Pearson r ([BENCH-2]) is what hides it.
- **[ADAPT-1] does not fire on this corpus**: 0 of 48 clips, results identical with the detector disabled. The clips are too short for its ring to fill; it needs the long-paragraph fixture of §6.8 to be measured at all.
- Keyframe rate rose 25 → 31–34/s (more voiced hops move the mouth); above the ~25/s guard noted in `accuracy-improvements-2.md`.
- Harness: `DeepgramTTSService` on pipecat 1.10 never drains the `EndFrame`, which hung fixture synthesis; teardown is now best-effort and detached.

## What I would keep

| change | constant | verdict |
|---|---|---|
| LPC order 16 | `dsp.LPC_ORDER = 16` | keep — minimax across ceilings and voices; do not quote the 23 Hz figure without the ceiling caveat |
| slot prior, F2/F3 only | `dsp.SLOT_PRIOR_WEIGHTS = (0, 1, 1)` | keep — small, free, never on F1 |
| NCC voicing, 40 ms frame, gated | `dsp.PITCH_METHOD = "ncc"`, `PITCH_FRAME_SIZE = 640`, `NCC_VOICED_THRESHOLD = 0.6`, `_VOICED_MIN_PEAK_FRACTION = 0.05` | keep — the main win; residual path stays as a fallback |
| LPC window 40 ms | `dsp.LPC_FRAME_SIZE = 400` | drop the change, keep the decoupling |
| c_fit for order 16 | `_C_FIT_LOG10_FULL = 3.2` | keep |
| nasal missing-F2 rule | `_NASAL_MISSING_F2_DAMPED = "always"` | unchanged — re-tune refuted by the second voice |
| `f1_broad` path | — | keep (vowel-aa strict F1 coverage 0.78 / 0.85 < 0.9) |
| coverage metrics, `--set`, `--tag`, `--ceiling`, per-clip table | harness | keep; next, score coverage and median error as §6.2 proposes |

§6.7: `results/baseline.json` (Cartesia) and `baseline-deepgram.json` are committed and re-saved after each pass; the pre-bundle 70.9 run is not kept as a file (its numbers are the row 0 tables above). Fixtures are not committed: provider redistribution terms were not checked.

---

## Second pass (2026-09-18): the two leads

Follow-up on the two leads above — the conditioning loss and the false nasal closures — on the landed DSP bundle, both voices. Reproduce: `uv run python ../plans/experiments/ab_table.py --ladder conditioning [--provider deepgram]`; stage attribution: `openness_stages.py`.

### Harness first

- `openness_lag_ms` / `width_lag_ms` (review §6.3): best cross-correlation shift of our interpolated track against the reference on a 5 ms grid over ±120 ms, positive = ours late, reported only when it improves r by ≥ 0.05 and lies inside the window (else NaN); `*_r_best` is r at that lag. `openness_jitter`: mean |second difference| of the interpolated openness over speech hops (§6.4).
- `nasal_fraction`: nasal-override duty cycle over Praat-voiced hops, with `nasal_fraction_max` guards on the low-nasal sentences (harvard-03 ≤ 0.20, harvard-02/-04 and bilabial-bob ≤ 0.30). Event counts cannot see a detector that latches on close vowels; the duty cycle can. Six new checks enter the `checks_nasal` bucket, so composites below are not comparable with the first table (the r columns are).

### Lead 1 — conditioning: the loss was almost pure lag

Stage-by-stage openness r against Praat, and the best lag of each stage (both voices, before the change):

| stage | Cartesia r | lag | Deepgram r | lag |
|---|---|---|---|---|
| mapped (F1 → openness) | 0.61 | +1 ms | 0.75 | +4 ms |
| after nasal override | 0.68 | +5 | 0.70 | +6 |
| after trailing median-3 + slew 0.25 | 0.48 | **+32** | 0.49 | **+38** |
| emitted keyframes | 0.49 | +33 | 0.50 | +37 |
| emitted, at its best lag | 0.71 | — | 0.76 | — |

The conditioning stage added 27–32 ms and cost 0.20 r; at the best lag it kept 0.70/0.76 of the target's 0.73/0.78 — timing, not smoothing. Fix ([COND-1]): the median is now zero-phase — hop t is conditioned and its keyframe emitted when hop t+1 arrives (one hop of analysis latency, inside the processor's 0.4 s horizon; events keep their own stamps; chunk-invariant) — and the slew is 0.4/hop.

| variant | Cartesia open_r | width_r | lag | jitter | kf/s | Deepgram open_r | width_r | lag | jitter | kf/s |
|---|---|---|---|---|---|---|---|---|---|---|
| before (trailing median, slew 0.25) | 0.49 | 0.49 | +33 | 0.06 | 30.5 | 0.50 | 0.70 | +37 | 0.06 | 33.7 |
| zero-phase median, slew 0.25 | 0.62 | 0.55 | +19 | 0.05 | 29.7 | 0.63 | 0.76 | +20 | 0.06 | 32.9 |
| + anchor keyframes | 0.62 | 0.55 | +19 | 0.05 | 37.4 | 0.63 | 0.76 | +21 | 0.06 | 40.6 |
| **zero-phase, slew 0.4 (landed)** | **0.65** | **0.56** | +4* | 0.06 | 28.5 | **0.69** | **0.77** | +10* | 0.08 | 31.4 |
| zero-phase, slew 0.6 | 0.66 | 0.57 | — | 0.07 | 27.5 | 0.71 | 0.77 | — | 0.08 | 30.5 |
| zero-phase, no slew | 0.67 | 0.57 | — | 0.07 | 26.8 | 0.71 | 0.77 | — | 0.09 | 29.7 |

\* per-stage lag from `openness_stages.py`; the harness's `openness_lag_ms` aggregate is noisy once most clips fall under the significance guard. After the change the conditioning stage costs 0.02–0.03 r. Slew above 0.4 buys +0.02 r for visibly more jitter; the anchor keyframe of the review ([COND-1]) changed r by 0.00–0.01 and added 7–8 keyframes/s, so it is landed off (`_ANCHOR_KEYFRAMES`). Keyframe rate fell 2/s.

### Lead 2 — false nasal closures: mostly dark voiced consonants, and a 1-hop latch

What the override was latching on (harvard-03 has no nasal): its dark voiced consonants — "the" ×2 (ð), the voice bar of /d/ in "depth", /v/, /w/ and /l/ in "well"/"tell" — F1 ≈ 260 Hz, no F2 root, > 90 % of the pre-emphasized energy below 500 Hz. Spectrally these *are* murmurs minus the nasal formant, and the mouth is nearly closed during them, which is why the override helps openness_r on Cartesia. So the defect is mostly the NASAL label (and /i u/ frames), not the aperture. Cues measured on both voices against the hum takes: F1 ≤ 400 Hz, a 500–1500 Hz mid-band ratio (antiformant test; now in the debug tap), and run length all fail to separate murmurs from these consonants (each moves the false duty cycle by ~0.01; false runs last 60–120 ms, like a real /m/). What did work: removing the 1-hop fast path (`_NASAL_STRONG_RATIO`; a single hop with low-band ratio > 0.9 latched). Entry now needs 2 consecutive hops; the pre-latch soft cap still closes the mouth within one hop.

| | Cartesia guard duty | harvard-03 | hum duty | Deepgram guard | harvard-03 | hum |
|---|---|---|---|---|---|---|
| 1-hop fast path (before) | 0.31 | 0.27 | 0.85 | 0.19 | 0.13 | 0.70 |
| 2-hop entry (landed) | 0.21 | 0.18 | 0.76 | 0.12 | 0.08 | 0.58 |
| 3-hop entry | 0.15 | 0.12 | 0.69 | 0.07 | 0.04 | 0.47 |

Hum duty on Deepgram drops because that voice's murmur carries a narrow ~1.55 kHz pole, so its evidence flickers on the 300 Hz bandwidth threshold — but the mouth is at openness 0.00 on those hops anyway (F1 250 sits at the learned floor), and nasal-hum stays 4/4 on both voices. Longer exit holds (3–4 hops) bring the false latches back; a wider F2 band (to 2800–3000 Hz, to stop /i/ reading as "no F2") costs 25 Hz of f2_mae on Cartesia for a small nasal gain and was not kept.

### Result

| | composite | f1_mae | f2_mae | openness_r | width_r | voicing | kf/s | nasal duty | checks |
|---|---|---|---|---|---|---|---|---|---|
| Cartesia, DSP bundle (first pass) | 84.3 | 24 | 112 | 0.49 | 0.49 | 0.95 | 30.5 | 0.41 | 20/20 |
| **Cartesia, + both leads** | **87.5**† | 24 | 112 | **0.65** | **0.56** | 0.95 | 28.5 | 0.32 | 27/28 |
| Deepgram, DSP bundle | 86.3 | 24 | 75 | 0.50 | 0.70 | 0.85 | 33.7 | 0.28 | 18/20 |
| **Deepgram, + both leads** | **90.7**† | 24 | 75 | **0.69** | **0.77** | 0.85 | 31.4 | 0.21 | 26/28 |

† with the six new nasal guards in the composite. CPU unchanged (185 µs/hop). Baselines re-saved. Residuals: harvard-02/t2 on Cartesia reads nasal duty 0.41 against the 0.30 bar (a dark sentence: "Glue the sheet to the dark blue background"); Deepgram's nasal-hum openness_p90 is 0.29 against 0.30 — the open hops are the breathy "H" onset, NCC-voiced but with no F1, decaying toward the 0.35 neutral; pause-probe silences on Deepgram (pre-existing).

Next levers, in order: the NASAL event needs a place cue or a rename ("dark voiced closure") before clients style it; keyframe economy (28–31/s vs the 25/s guard — the dead band is a constructor default the harness cannot sweep yet); [CONS-1] energy-gated closures, now that the conditioning no longer hides timing. Tracked, together with what the eval recorder found since (utterance-start latency), in [README.md](README.md).

---

## Third pass (2026-09-19): the /u/ vowels read as murmurs

Found from the Eval tab, not the benchmark: "Soon the new moon grew blue, and the room felt cool"
rendered with the mouth shut through its /u/ words — 48 % of the clip's keyframes at the nasal
override level, nasal duty 0.62 / 0.53 on the two Cartesia takes (0.28 / 0.29 on Deepgram). The
benchmark could not see it: the sentence's only check was the rounding peak, which "cool" meets,
and the nasal-duty guards were on the harvard and bilabial sentences.

A per-hop dump against Praat (take 1, "Soon the new moon") showed this voice's /u/ in two forms.
In "Soon" and "new" the F2 root is there at 650–1000 Hz (Praat: 700–1150) but 380–960 Hz wide, so
it was either dropped by the 500 Hz formant cap and read as "missing" (which counts as damped) or
kept and read as "broad" — murmur evidence either way. In "moon" the vowel between /m/ and /n/ is
nasalized: F1 330–440 Hz, a narrow F2 at 1.8–1.9 kHz and a low-band ratio above 0.9, which is
exactly the spurious-F2 rule's definition of a murmur. The true murmurs (the /m/ of "moon", the
/n/ of "Soon") have no root at all between 500 and 1200 Hz, and on the hum takes F2 is always
found, narrow, at 1850–1990 Hz (`nasal_tap.py --hist`). So the cue is "any root in the back-vowel
F2 range above F1, however broad".

Ladder on both voices (nasal duty per take; hum p90 is the hum probe's openness bar, ≤ 0.30):

| variant | Cartesia oo t1/t2 | ee t1/t2 | harvard-03 | hum p90 | Deepgram oo | ee | hum p90 |
|---|---|---|---|---|---|---|---|
| before | 0.62 / 0.53 | 0.41 / 0.39 | 0.17 | 0.19 | 0.28 / 0.29 | 0.10 / 0.23 | 0.29 |
| F2 slot band edge 500 or 550 Hz | 0.62 / 0.53 | 0.41 / 0.39 | 0.17 | 0.19 | 0.28 / 0.29 | 0.10 / 0.23 | 0.29 |
| missing F2 counts only when dark | 0.58 / 0.48 | 0.25 / 0.24 | 0.15 | 0.19 | 0.24 / 0.28 | 0.07 / 0.08 | 0.29 |
| F1 cap 350 Hz | 0.53 / 0.49 | 0.18 / 0.26 | 0.09 | 0.22 | 0.18 / 0.12 | 0.08 / 0.15 | 0.29 |
| mid-band ratio ≤ 0.15 | 0.62 / 0.53 | 0.41 / 0.38 | 0.15 | 0.19 | 0.25 / 0.29 | 0.10 / 0.23 | 0.29 |
| low-root veto (`f2_low` ≤ 1200 Hz) | 0.49 / 0.48 | 0.36 / 0.33 | 0.13 | 0.19 | 0.20 / 0.17 | 0.10 / 0.23 | 0.29 |
| **veto + F1 cap 350 (landed)** | **0.41 / 0.45** | **0.12 / 0.20** | **0.09** | **0.22** | **0.10 / 0.03** | **0.08 / 0.15** | **0.29** |
| veto + F1 cap 320 | 0.33 / 0.41 | 0.07 / 0.11 | 0.09 | 0.26 | 0.10 / 0.00 | 0.07 / 0.11 | 0.29 |

Lowering the F2 slot band's edge does nothing (the root is not out of band, it is too broad for
the cap); the mid-band ratio does nothing; the veto is the lever for the back vowels and the F1
cap for the nasalized ones. 320 Hz buys a little more but halves the hum margin on one voice.

Landed:

- `dsp.lpc_formants` reports `f2_low` (the lowest root above F1 from 500 Hz up, any bandwidth up
  to 1000 Hz, whichever root won the F2 slot) and `f2_broad` (a broad F2-band root when the slot
  is empty — rounding evidence for the mapping, like `f1_broad`, never a measurement).
- `_NASAL_VOWEL_F2_MAX_HZ = 1200`: a low root vetoes the murmur reading. `_NASAL_F1_MAX_HZ = 350`
  (was off). The override now caps openness at `_NASAL_OPENNESS_MAX = 0.15` instead of forcing
  0.05: a murmur's F1 sits at the learned floor and maps near zero anyway, while a nasalized vowel
  keeps a small opening and its rounding.
- Rounding's low-side openness gate is gone (`_ROUNDING_OPEN_GATE` keeps the wide-open side): it
  zeroed the rounding of /u/, whose opening is small by nature. Vetoed back vowels keep an
  openness of at least 0.1 (`_ROUNDED_VOWEL_MIN_OPENNESS`; their F1, 210–300 Hz on these voices,
  maps to a shut mouth otherwise).
- Guards: vowel-oo `nasal_fraction_max: 0.45` (six nasals and the nasalized vowels between them
  come to roughly 0.45 of voiced time; the bar catches the /u/ vowels reading as murmurs),
  vowel-ee 0.30.

| | Cartesia before | after | Deepgram before | after |
|---|---|---|---|---|
| composite | 87.5 (85.2 with the two new guards) | **87.5** | 90.7 | **92.0** |
| checks | 27/28 | 31/32 | 26/28 | 30/32 |
| openness_r / width_r | 0.65 / 0.56 | 0.62 / 0.57 | 0.69 / 0.77 | 0.74 / 0.79 |
| vowel-oo rounding p90 | 0.80 | 1.00 | 0.84 | 0.91 |
| harvard-02/t2 nasal duty (was a residual) | 0.41 | 0.23 | — | — |
| f1 / f2 mae | 24 / 112 | 24 / 113 | 24 / 75 | 24 / 75 |

On the eval recording (the bot's live Cartesia voice), the oo clip's keyframes at the override
level fell from 48 % to 22 %: "grew" 80 → 0 %, "blue" 47 → 0 %, "Soon" 50 → 18 %, "room" 78 → 60 %,
"new" 100 → 67 %, "moon" 100 → 57 % — what remains is the /n/ and /m/ themselves — and the /u/
words carry rounding again (new / moon / room 0.00 / 0.13 / 0.11 → 0.18 / 0.23 / 0.39). The ee
clip went from 21 % to 12 %. Deepgram's rise comes partly from the freed rounding motion emitting
keyframes where openness was under-sampled (openness_r 0.70 → 0.74 with the gate change alone).

Residuals: vowel-oo/t2 reads 0.454 against the 0.45 bar; the hum probe's p90 rose 0.19 → 0.22 on
Cartesia (the F1 cap releases hum hops with F1 above 350 Hz); rounding is F2-only, so any low-F2
vowel (/ɑ ɔ/ in "calm", "walked") reads as rounded (vowel-aa rounding p90 0.73) — [ROUND-1]; the
NASAL label still latches on dark voiced consonants (harvard-03 duty 0.09–0.15).

---

## Fourth pass (2026-09-19): seven Cartesia voices

Every pass so far had tuned on one Cartesia voice and checked on one Deepgram voice. Six
library voices were added to the corpus (three masculine, three feminine; ids and pitches in
`corpus.yaml`), fixtures synthesized (144 clips), and the Praat ceiling set by pitch: ~100 Hz
voices get 4500 (5000 read their F2 ~30 Hz worse and openness_r 0.03–0.09 lower), ~160 Hz
voices 5500. The harness's `--voices` now keeps the corpus ceilings (it hard-coded 5500).

| voice | f0 | first run | landed | what still fails |
|---|---|---|---|---|
| quickstart | 172 Hz | 87.5 | 87.4 | vowel-oo/t2 nasal duty (0.454 vs 0.45) |
| Gemma (f, British) | 163 | 88.7 | 93.6 | hum: no NASAL event, t1 openness |
| Archie (m, British) | 99 | 83.2 | 88.7 | hum: no NASAL event, t2 openness |
| Jolene (f, Southern) | 154 | 83.9 | 86.1 | hum t1 openness; pauses |
| Skylar (f) | 170 | 81.8 | 84.3 | one pause |
| Daniel (m) | 110 | 78.0 | 82.3 | hum: no NASAL event; mama/t1 one closure; one pause |
| Ronald (m, deep) | 111 | 77.3 | 79.8 | hum: no NASAL event; pauses; vowel-aa/t2 openness peak |
| **pooled** | | 82.9 | **86.0** | 204/224 checks |

**What generalized:** the vowel tracking. F1 error 14–49 Hz, coverage 0.70–0.88 (F1) and
0.69–0.84 (F2), voicing agreement 0.90–0.95, openness_r 0.59–0.70, width_r 0.56–0.76, 27–31
keyframes/s — the same band as the two tuning voices. The back-vowel veto of the third pass
changes nothing on the new voices (identical with it off).

**What did not, and what was done:**

- **Silence.** The SILENCE event needed 300 ms; mid-sentence pauses run 235–430 ms by intensity
  and less under the silence gate, so the pause probe passed on one voice in seven. Ladder
  (hops of 20 ms): 15 → 2/14 takes, 12 → 3/14, 10 → 8/14, 8 → 10/14 with spurious events
  appearing on the balanced sentences at 8. Landed `_SILENCE_EVENT_HOPS = 10` (200 ms):
  pooled 82.9 → 85.1, keyframe rate and openness_r unchanged. Keyframes already rest the mouth
  in a pause; the event is informational.
- **Closures on "Mama made more mashed potatoes".** Deep voices dip little at /m/ (Ronald,
  Daniel, Jolene: 1–4 closures against a bar of 4). Raising `_CLOSURE_PEAK_FRACTION` makes it
  worse (0.30: fewer closures, because longer low runs exceed the 250 ms cap and fewer hops
  count as confirming speech); lowering it re-opens the voice-bar misses of 2026-07. The bar is
  now 2 (no dips at all is the gross failure); closing the mouth on /m/ is the nasal detector's
  job, not an energy dip's.
- **Nasal murmurs.** The hum probe got no NASAL event on Ronald, Archie, Daniel and Gemma and the
  mouth opened (p90 0.38–0.90). Their hums are not dark: on the pre-emphasized spectrum the
  centroid is 1.1–3 kHz and the low-band ratio 0.10–0.35 (gates: < 1 kHz, > 0.6), and relative
  to each voice's own vowels the hum sits at the median brightness, so no darkness cue, absolute
  or adaptive, can work. A feature study over all seven voices (pre-emphasized and raw centroid
  and band ratios, mid-band hollowness, F1, F1 presence, F2 band and bandwidth, NCC clarity,
  prediction gain, level) found no single-frame cue that separates murmur hops from vowel hops
  across voices; "voiced with no F1 at all" is the best and holds on four voices only (hum hops
  0.54–0.62 vs vowels 0.07–0.19; Jolene, Gemma and Skylar find an F1 of 300–400 Hz in their
  hums). Praat itself reads F1 above 1 kHz on 20–50 % of the Ronald and Gemma hum hops: those
  voices render "Hmm" partly as an open voiced sound, which the openness bar cannot admit.
  What did land is the failure mode's fix: a stale F1 used to drift toward the prior's center
  (`_bound_hold`), which opened the mouth half-way during a hum with no findable F1; it now
  drifts toward the low edge (`_F1_HOLD_DRIFT = 0.0`) — a voiced frame with no F1 is a murmur
  or a close vowel far more often than a mid vowel. Hum openness p90: Ronald 0.38/0.24 →
  0.05/0.04, Daniel 0.44/0.22 → 0.10/0.10, Archie t1 0.47 → 0.29, at no cost elsewhere
  (openness_r 0.65 → 0.64 pooled, F1 error unchanged).

**Residuals:** NASAL events on bright/breathy murmurs (four voices; the label, not the
aperture, on three of them now); hum takes whose audio has a vowel-like F1 (Archie t2, Jolene
t1, Gemma t1); pauses under 200 ms on four takes; Daniel's mama take with one dip; Ronald's
vowel-aa/t2 openness peak 0.70 against 0.72. Deepgram re-baselined at 91.9 (−0.1 from the
silence and hold changes).

