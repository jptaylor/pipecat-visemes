# Plans

The plan of record for pipecat-visemes (written 2026-09-19): what the project is, what the
recent work found, what is open, and what is deliberately not planned. It supersedes the
roadmap in [deep-review-2026-09.md](deep-review-2026-09.md) §8 and the tier list the top-level
README used to carry. The other documents in this directory are the design and the record of how
the numbers were reached; their status is in the table at the end.

## What the project is

Server-side lipsync for Pipecat voice bots that is

- **fast** — keyframes are playout-timed and reach the client ahead of the audio; analysis adds
  one 20 ms hop of latency to the signal;
- **light on CPU** — numpy-only DSP, ~185 µs per 20 ms hop, about 1 % of one core per speaking
  bot, no model, no new runtime dependency;
- **drop-in for any provider** — it reads the PCM stream and nothing else (no provider
  timestamps, viseme events or side channels), on released `pipecat-ai` through stock extension
  points (a frame processor and an RTVI `server-message`), so any `TTSService` works.

The formant DSP analyzer is the design of record and is judged close enough. The next step in
fidelity would be a trained model (the deep review's "Tier 0.5", the spec's Tier 3); it is not
planned, because the remaining error is consonant identity that a continuous mouth does not need,
and a model would cost the training pipeline and some of the three properties above. The
provider-specific tiers (Azure/Polly visemes, timestamp + grapheme-to-phoneme) are not planned
either: they contradict "any provider", and pipecat 1.10 exposes no viseme data anyway. The
project stays a standalone example app; it is not being upstreamed into pipecat (decided
2026-09-17, [pipecat-1.10-update.md](pipecat-1.10-update.md)).

## Where it stands

Baselines committed 2026-09-19 (`server/benchmarks/results/baseline*.json`), 12 sentences × 2
takes per voice, scored against Praat and the corpus expectations. The Cartesia column pools
seven voices (three masculine, three feminine library voices plus the quickstart voice; per voice
79.8–93.6), the Deepgram column is one voice:

| | Cartesia (`71a7ad14…`) | Deepgram (`aura-2-helena-en`) |
|---|---|---|
| composite (v1 + nasal guards) | **86.0** (7 voices) | **91.9** |
| f1_mae / f2_mae (Hz, committed hops) | 31 / 113 | 24 / 75 |
| F1 / F2 coverage of Praat-voiced hops | 0.81 / 0.78 | 0.91 / 0.81 |
| openness_r / width_r vs Praat | 0.64 / 0.63 | 0.73 / 0.79 |
| voicing agreement with Praat | 0.93 | 0.85 |
| conditioning lag (per-stage, quickstart voice) | +4 ms | +10 ms |
| keyframes / s | 29.5 | 31.1 |
| corpus checks | 204/224 | 30/32 |

CPU: ~185 µs per hop (RTF ≈ 0.009). Tests: 55 (`server/tests/`, including the eval recorder's replay, the delivery schedule and a synthetic back vowel that must not read as a murmur). Before the September work the same corpus read
70.9 / 85.6.

What got it there, all in [deep-review-2026-09-results.md](deep-review-2026-09-results.md):
LPC order 16, an F2/F3 slot prior, normalized-cross-correlation voicing on a separate 40 ms
frame (the largest single win: coverage of Praat-voiced hops doubled on Cartesia), zero-phase
conditioning with a 0.4/hop slew (the trailing median had been ~35 ms late and cost 0.20 of
openness correlation), and a 2-hop nasal entry. The harness gained coverage, lag, jitter and
nasal-duty metrics and `--set`/`--tag`/`--ceiling` for A/B runs. A third pass on 2026-09-19 (the
Eval tab showed the mouth shut through every /u/ of "Soon the new moon grew blue…") added a
back-vowel veto to the nasal detector — a root at 500–1200 Hz above F1, however broad, is a
vowel's F2, not a murmur's — an F1 cap of 350 Hz for nasalized vowels, an openness cap of 0.15
in place of the forced-shut override, rounding no longer gated off for small openings, and
nasal-duty guards on the vowel sentences; Deepgram 90.7 → 92.0, Cartesia unchanged at 87.5 with
four more checks passing.

A fourth pass the same day widened the Cartesia corpus to seven voices (results note, fourth
pass): the vowel tracking generalizes (F1 error 14–49 Hz, openness_r 0.59–0.70 per voice), the
event detectors did not. Landed: the SILENCE event at 200 ms (was 300; pauses on six of seven
voices are shorter), a stale F1 drifting toward closed rather than mid-open (which had opened the
mouth during hums with no findable F1), Praat ceilings by pitch, and a mama closure bar that only
guards against no dips at all.

Accepted residuals: the NASAL event never fires on four voices whose hums are bright or breathy
(no single-frame cue separates them from vowels; see the results note), three hum takes whose
audio has a vowel-like F1, pauses under 200 ms on four takes, one deep-voice mama take with a
single dip, Ronald's vowel-aa/t2 openness peak, and the quickstart vowel-oo/t2 nasal duty 0.454
against 0.45; on Deepgram the nasal-hum openness p90 0.29 against 0.30 and the pause probe.

Tooling since the baselines (2026-09-19): the **eval recorder** (`server/benchmarks/record.py`,
`eval_corpus.yaml`, 19 lines; `tests/test_eval_record.py`) speaks the corpus through the real output path — `LipsyncProcessor`,
a headless `BaseOutputTransport` with a real-time simulated audio device, the relay — and records
every lipsync message with its release time and its pts; `--reanalyze --set …` adds runs over the
retained audio without TTS calls. The client's **Eval tab** (`#eval`; `client/src/eval/`,
`EvalView`) replays a recording's clips with their audio through the same mouth and meters as the
Live tab, "as delivered" (batches at their recorded release times, anchored as the live client
anchors them) or "ideal" (every batch on time, analysis only), with a run picker and per-clip
delivery stats (start lag, settled anchor, late batches, longest clock-queue hold, clamped
batches). Recordings live under `client/public/eval/`, gitignored.

**Delivery timing (fixed 2026-09-19).** The first recording had shown the live client starting the
mouth +309 ms late (median; up to +631) and settling 90 ms early, for three reasons outside the DSP:
the processor held every batch until analysis was 0.4 s past its window (the closure-confirmation
horizon), a batch whose pts had passed then waited in pipecat's clock queue until the next
word-timestamp frame was due (52 of 52 held batches left at a word boundary; up to 840 ms during
"Photosynthesis"), and the client inferred each batch's anchor from its first keyframe under an
assumed 200 ms lead. All four fixes are in, all in our code:

- wire version 2 carries the window start (`ws`) and the lead remaining at send time (`lead`);
  the client anchors on `now + lead − ws`, exact but for transit, and keys restarts on `ws`;
- keyframes (and NASAL) leave one hop after their window; CLOSURE/SILENCE ride late in the next
  batch, keyed by offset; the first window of an utterance is 100 ms;
- the processor schedules delivery itself on the pipeline clock and pushes each batch (now a
  system frame, which the transport forwards at once) at playout minus the lead; unreleased
  batches are dropped on interruption; already-due batches leave immediately;
- keyframes that are final when audio stops arriving mid-turn are flushed after 100 ms; the client
  cuts the utterance on bot-stopped-speaking with a 150 ms grace and ignores in-flight batches.

Same recording, same keyframes, replayed through both paths (`live` vs the `delivery-v2` run):

| | before | after |
|---|---|---|
| first batch released, median (worst) | +119 ms (+441) after playout start | −0 ms (−0) |
| client anchor error at start, median (worst) | +309 ms (+631) | −3 ms (−1) |
| settled anchor, median (range) | −90 ms (−190 … +430) | −3 ms (−19 … −2) |
| batches released after their window began | 42 of 306 | 2 of 357 |
| longest late release | 840 ms | 21 ms |
| lead of batches from the third on, median | 212 ms | 201 ms |
| messages per second | 4.4 | 5.2 |

Accuracy benchmark unchanged (+0.0 on both voices); 54 tests. What is left is structural: the
first batch needs 0.14 s of audio analyzed, so at a TTS slower than ~3× real time the first
100 ms of motion is still missed; a client's own network transit adds to every lead equally.

Reading the numbers — caveats that apply to every future comparison:

- The best LPC order follows the Praat ceiling's pole density (4500 → 18, 5000 → 16,
  5500 → 14); order 16 is the minimax pick across ceilings and voices, and the 23 Hz f1_mae is
  partly by construction. Do not quote it without the caveat.
- `f1_mae`/`f2_mae` only score committed hops: a voicing change can raise MAE with bit-identical
  formant tracks. Use `experiments/ab_matched.py` (matched hops) before believing an MAE delta.
- Single-voice tuning is unsafe; the nasal "missing F2 = damped" rule looked free on Cartesia and
  lost the hums on Deepgram, and every event detector tuned on the quickstart voice failed on
  four of six library voices. Run all seven Cartesia voices and Deepgram; a fresh checkout
  synthesizes the Cartesia fixtures in about three minutes.
- Real prediction gain is 19–31 (median), so `_C_FIT_LOG10_FULL` is 3.2; confidence is a
  diagnostic, not a pose or opacity scale.

## Steps forward

Ordered by what matters for a fast, light, drop-in lipsync. Tags refer to the deep review's
sections, where each item is worked out in detail. Every runtime change is gated by
`uv run python -m benchmarks.accuracy --offline --compare` on **both** providers; timing changes
are measured with the eval recorder (`--reanalyze` on the same audio) before and after.

1. **The nasal detector across voices.** Its murmur evidence is "dark spectrum + damped F2",
   which holds on three of seven Cartesia voices and the Deepgram voice; on the other four the
   hum is as bright as their vowels and NASAL never fires (the mouth still closes on most of them
   since a stale F1 now drifts toward closed). The feature study in the results note found no
   single-frame cue; a murmur cue that works across voices probably needs temporal structure
   (stationarity, level relative to the surrounding vowels) or a place cue. Also still true: on
   the voices where it does fire it latches on dark voiced consonants (ð, /w l/, voice bars) with
   the aperture right and the label wrong, and rounding is F2-only, so /ɑ ɔ/ read as rounded
   ([ROUND-1]).
2. **Keyframe economy.** 28–31/s against the 25/s guard. The dead band is a constructor default
   the harness cannot sweep; expose it to `--set`, then decide.
3. **[CONS-1] Energy-gated closures.** The mouth should close with the energy dip instead of only
   badging the event (openness sits at ~0.44 during a /p b m/ today). Now that conditioning no
   longer hides timing, this is measurable with the existing checks.
4. **Closures on deep voices.** The energy-dip detector under-counts /m/ dips where the murmur
   is nearly as loud as the vowels (Ronald, Daniel, Jolene); thresholds do not fix it in either
   direction. The review's [CONS-1] energy-gated aperture and text-gated events are the routes.
5. **Processor hygiene** (review §7, unchanged since): a ≥ 2 s burst on the frame path silently
   drops audio and misplaces the SILENCE ([PROC-1], relevant to HTTP-burst providers); the stream
   resampler is never flushed at context close, so the last 40–60 ms of every context is not
   analyzed ([PROC-2]); evicting an unflushed context leaks its hops into the next (B13);
   `dead_band`/`heartbeat_ms` runtime updates are silent no-ops (B14).
6. **Client.** Events are shown as a badge but never shape the pose ([CLIENT-2]); a new
   context's first batch wipes the previous context's tail (B17).
The remaining review §7 items not listed here (F3 hold, rounding gate, closures pending at flush,
harness `--warm`/`--voices` bugs) are small and unaddressed; take them when touching the code
nearby. Also noted: keyframe counts differ by ±1 on a few clips between runs over identical audio
(how much audio each analysis call resamples follows task scheduling, and the stream resampler is
not bit-exact across chunkings); benign, unmeasured.

## Parked — evidence exists, no plan

- **Text-informed events** (review §4.8, §8 item 9a/12). The text is already in the frame stream
  ahead of the audio and 12 providers push playout-timed word frames; CMUdict-style lookup plus
  the DSP lifted closure precision/recall from 0.16/0.38 to 0.87/0.85 on the proxy corpus,
  numpy-only. Parked because it makes the signal depend on text frames and a lexicon, and no client
  needs bilabial/labiodental precision. The cheapest route if one ever does.
- **Provider hints as an optional side-channel** (review §8 item 13): Azure visemes, Cartesia
  phoneme timestamps, via app-level service subclasses feeding the same fusion. Not a tier; not
  planned.
- **Packaging** as a pip package plus an npm client with rig mappers (review §4.7). The current
  drop-in story is the app layout: copy `server/lipsync/`, place two processors around
  `transport.output()`, demux one `server-message` type on the client.
- **Performance harness** as designed in
  [benchmark-harness-performance.md](benchmark-harness-performance.md): superseded — CPU per hop
  comes from the accuracy harness, delivery timing from the eval recorder. RSS per session and a
  leak soak remain unmeasured; the differential-CPU design is the way to do them if ever needed.

## Not planned

- A trained model or learned classifier in the signal path (Tier 0.5 / Tier 3), and the forced
  alignment tooling it would need.
- Provider-viseme (Tier 1) or timestamp + G2P (Tier 2) analyzers.
- The L4 phoneme-target layer and "composite v2" benchmark (review §6.6): the accuracy harness
  stays Praat + designed corpus, with the guards it has.
- Upstreaming into pipecat; moving the client into pipecat-examples.

## Documents

| File | Status | Use it for |
|---|---|---|
| `README.md` (this file) | plan of record | status, findings, open / parked / not planned |
| [technical-specification.md](technical-specification.md) | design of record, as built (delta table at the top) | the design and its rationale |
| [benchmark-harness-accuracy.md](benchmark-harness-accuracy.md) | built (2026-07); as-built notes at the top | how the accuracy score is made and read |
| [deep-review-2026-09-results.md](deep-review-2026-09-results.md) | done (2026-09-18) | the current numbers, what each DSP change bought, how to read the benchmark |
| [deep-review-2026-09.md](deep-review-2026-09.md) | review complete (2026-09-17); roadmap superseded here | the findings catalogue (the [TAG]s above), bug list §7, external context §9 |
| [pipecat-1.10-update.md](pipecat-1.10-update.md) | done (2026-09-17) | the timing model on per-turn TTS contexts; the not-upstreaming decision |
| [experiments/](experiments/README.md) | repro scripts, not maintained with the code | re-running the A/B ladders and stage attribution |
| [benchmark-harness-performance.md](benchmark-harness-performance.md) | not built; superseded | design record only |
| [pipecat-implementation.md](pipecat-implementation.md) | historical (2026-07; in-tree fork layout) | the §11 decisions logs |
| [accuracy-improvements.md](accuracy-improvements.md), [accuracy-improvements-2.md](accuracy-improvements-2.md) | historical (2026-07) | how the first two tuning passes were done; slotting and nasal facts |
