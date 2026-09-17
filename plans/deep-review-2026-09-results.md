# Deep review §2/§6 — validation on real fixtures (2026-09-18)

**Status:** done. DSP bundle landed behind constants; baselines re-saved and committed.
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

§6.7: `results/baseline.json` (Cartesia, 84.3), `baseline-deepgram.json` (86.3) and the pre-bundle `baseline-order12-2026-09-17.json` (70.9) are committed. Fixtures are not: provider redistribution terms were not checked.
