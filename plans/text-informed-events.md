# Text-informed events (2026-09-19)

**Status:** planned. Reopens the "Text-informed events" item parked in [README.md](README.md) on the
evidence of the seven-voice Cartesia run; supersedes that Parked entry and amends the "Not planned"
entries it reopens or clarifies (§10).
**Input:** the seven-voice run of 2026-09-19; [deep-review-2026-09.md](deep-review-2026-09.md) §4.8
and §8 items 9a, 11, 12; [experiments/l4_proto.py](experiments/l4_proto.py).
**Constraints (unchanged, and they are the point):** numpy-only at runtime, no torch and no cloud
call in the hop loop, analysis lookahead ≤ 3 hops (60 ms; 1 hop is spent today on the zero-phase
median, `formant_lipsync_analyzer.py:155`), and **text is never required** — every provider still
works from PCM alone.

---

## 1. Why now

The formant analyzer was tuned on two voices. The seven-voice run split its output cleanly in two:

| | across 7 voices | reading |
|---|---|---|
| f1_mae | 14–49 Hz | holds |
| openness_r | 0.57–0.70 | holds |
| width_r | 0.57–0.76 | holds |
| nasal-hum probe | fails 5 of 6 new voices | breaks |
| bilabial closures | under-count on 4 of 6 | breaks |
| pause-probe silence | fails 6 of 6 | breaks |
| F2 on male voices | degrades | breaks |

**The continuous mapping generalizes across voices; the categorical detectors do not.** That is not
a coincidence of tuning. Openness/width/rounding are monotone functions of F1/F2 after per-voice
adaptive normalization (`_update_adaptation`, `_effective_range`), so a new voice moves the range
and the normalization absorbs it. The categorical detectors are fixed thresholds on absolute
spectral quantities — `_NASAL_CENTROID_MAX_HZ = 1000`, `_NASAL_LOW_RATIO_MIN = 0.6`,
`_NASAL_F1_MAX_HZ = 350`, `_SILENCE_EVENT_HOPS = 15`, `_CLOSURE_PEAK_FRACTION = 0.22` — with no
adaptation path at all. A seventh voice is a seventh spectrum, and the constants were fitted to the
first.

The direction follows from that split: **text decides the categorical questions, DSP keeps the
continuous ones.** Phone identity is exactly what a formant tracker cannot see and what the text
states for free, and it is voice-independent by construction — a /p/ is a /p/ on all seven voices.

### 1a. What text will *not* fix

Two of the four failures above are misfiled if they are read as a text problem, and the plan is
weaker if that is not said up front:

| failure | cause | fix |
|---|---|---|
| bilabial closures under-count | no phone identity; energy dip alone | **text** (§5) — the strong case |
| nasal-hum probe | nasal gates fitted to one voice's murmur spectrum | **text** gates the detector (§5); a per-voice adaptive nasal gate is the DSP half, and is worth doing anyway |
| pause-probe silence | `_SILENCE_EVENT_HOPS` is a constant 300 ms; real pauses are 235–430 ms | **not text.** A relative/adaptive threshold needs no lexicon and no frames. Text (punctuation) only tells you where a pause is *expected*, which is a second-order refinement |
| F2 on male voices | LPC ceiling/order follows pitch; `corpus.yaml` already carries `formant_ceiling` per voice | **not text.** DSP, per the ceiling/pole-density finding already in [deep-review-2026-09-results.md](deep-review-2026-09-results.md) (finding 2, and `experiments/order_vs_ceiling.py`) |

Land the two non-text fixes independently of this plan — they are small, they need no new machinery,
and leaving them inside the text tier would let the text tier take credit for them at measurement
time.

---

## 2. What the frame stream actually delivers

Verified against the installed `pipecat-ai==1.10.0` source in this session, not only against the
review. Three regimes, and the fallback is the normal case, not the exception:

**(a) Word-timestamp providers** (12–13 services, Cartesia among them).
`AggregatedTextFrame(text, will_be_spoken=True, context_id)` is pushed through the serialization
queue immediately before the context's `TTSStartedFrame` (`tts_service.py:1227–1243`), then audio,
then per-word `TTSTextFrame`s with `pts` (`:1471–1483`).

**(b) Non-streaming without word timestamps** (`_push_text_frames`): the same `will_be_spoken`
anchor before synthesis, then one sentence-level `TTSTextFrame` **after** generation completes
(`:1325–1344`). Text arrives in time; word timing does not.

**(c) TOKEN streaming mode:** the sequencer promotes a sentence anchor, possibly *after* some audio
has already flowed; non-streaming in TOKEN mode pushes **no text frame at all**. "No text for this
context" must be an ordinary, untested-for-badness path.

Four facts that shape the implementation, each of which the review either left open or got
pessimistically wrong:

1. **`TTSTextFrame` is a subclass of `AggregatedTextFrame`** (`frames.py:417`). A bare
   `isinstance(frame, AggregatedTextFrame)` in `process_frame` catches every per-word frame as well
   as the sentence anchor. Discriminate on `aggregated_by` (`AggregationType.SENTENCE` vs `WORD`)
   or on `not isinstance(frame, TTSTextFrame)`. This is the first bug anyone writing this will
   ship.
2. **The clock problem is smaller than §4.8 states.** The review says the TTS word baseline
   (`_initial_word_timestamp`) and our `t0` are "different clocks-of-record". They are the same
   pipeline clock sampled at two instants, both under a continuity rule
   (`max(_word_last_pts, get_clock().get_time())` at `:1385–1390` vs
   `max(now, self._last_playout_end)` at `lipsync_processor.py:360–364`). Better still: on the
   word-timestamp path the **anchor frame itself is stamped with that same baseline**
   (`:962–967`). So the offset between the word clock and our `t0` is *directly observable* from
   the anchor's `pts`, not something to estimate. Use it, and keep the `samples_seen`-at-anchor
   fallback for queued contexts.
3. **The anchor's text is the pre-transformation text.** TTS-specific transformations (spelling
   tags, emotion tags, `@` → "at") are applied after the anchor is pushed (`:1245–1250`), so the
   audio may be of a different string than the one we look up. G2P on the anchor text is a prior,
   never an assertion — which is what the alignment-cost gate in §6 is for.
4. **The anchor arrives before the context exists.** The processor opens a `_Context` on
   `TTSStartedFrame` (`lipsync_processor.py:344`); the anchor precedes it. Text must be parked by
   `context_id` and attached when the context opens, and dropped if it never does.

---

## 3. Where it plugs in

The existing separation holds and should not be disturbed: **the processor owns buffering, timing
and batching; analyzers only measure** (`base_lipsync_analyzer.py:9–13`). Text is an input to
measurement, so:

- `process_frame` taps the sentence anchor and the word frames, converts word `pts` to context
  seconds against the observed baseline, and parks a small immutable record per context.
- `LipsyncAnalysisContext` gains one optional field (a `TextPrior | None`) — it is already the
  per-context carrier the analyzer receives on every `analyze()` call, so no interface method is
  added and a text-unaware analyzer keeps working unchanged.
- The formant analyzer consumes it where it already decides: `_update_events` (closure, nasal,
  silence) and the rounding/openness mapping in `_process_hop`.
- The wire format does not change. Clients see the same keyframes and the same three event kinds.
  (Whether a `labiodental` or `stop` event kind is worth adding is a separate question, deferred to
  after stage 2 — no client needs it today, which is exactly why this was parked in the first
  place.)

**Invariants, to be asserted by tests, not by intent:**

- No text → byte-identical output to today. This is the regression gate on every stage.
- Text present but wrong (mispronounced, transformed, misaligned) → the alignment-cost gate falls
  back to DSP for that sentence; never worse than DSP-only.
- Nothing on the frame path blocks or allocates per word beyond O(words in sentence).
- Lookahead stays ≤ 3 hops total, including the 1 hop already spent on conditioning.

---

## 4. Stage 0 — make the seven-voice evidence reproducible (blocking)

**This branch cannot reproduce any number in §1.** `server/benchmarks/corpus.yaml` lists one
Cartesia voice and one Deepgram voice, not seven; `server/benchmarks/fixtures/` does not exist and
is gitignored; `server/benchmarks/results/` holds only `baseline.json` and `baseline-deepgram.json`
— `.gitignore:9–10` un-ignores `baseline*.json` only, so `accuracy-v-*.json` was never committable.
The run happened somewhere else and left nothing behind that this plan can be checked against.

That is the first work item, and everything below is gated on it:

1. Land the seven voices in `corpus.yaml` with each voice's `formant_ceiling` and a one-line note on
   pitch/sex, so the ceiling choice is auditable.
2. Commit a **per-voice summary** — one small JSON or a table in a results note, not the 48 KB
   per-clip dumps — and widen the `.gitignore` exception to cover it. Without a committed
   seven-voice baseline there is no "before" for any stage below.
3. Decide fixtures explicitly: they stay gitignored (they are provider audio), so record the exact
   `--refresh` invocation and the fixture `meta.json` fields that make a run repeatable, and state
   that a seven-voice re-run needs a Cartesia key.
4. Re-state the `checks` bars per voice. Bars calibrated on one voice (`--calibrate-expectations`,
   `bar = 0.8 × min(Praat oracle over takes)`) are not automatically right for seven. A check that
   fails on 6 of 6 new voices may be a wrong bar rather than a broken detector — the pause probe is
   the live example.

Expect the composite to move for measurement reasons when five voices join the mean. Re-baseline and
say so, as in the DSP pass.

---

## 5. Stage 1 — CMUdict prior

Vendored BSD CMUdict plus a small letter-to-sound fallback for OOV. Words → ARPAbet → phone
classes; no timing model beyond word-uniform placement.

On vendoring rather than depending: `nltk` is in fact an unconditional `pipecat-ai` requirement
(`nltk<4,>=3.10.0`), so the review's "already a core dep" is right and `g2p_en` would add only
itself. The argument for vendoring is therefore *not* weight — it is the drop-in story. The
integration instruction is "copy `server/lipsync/`"; a lexicon inside that directory survives the
copy, a pip dependency and its downloaded model data do not. `g2p_en` also carries a neural OOV
fallback we would not run in the hop loop. A ~1 MB packed CMUdict plus a few hundred lines of
letter-to-sound rules keeps the package self-contained and the runtime numpy-only.

What it changes, in decreasing order of confidence:

- **Closure gating and injection.** Emit `CLOSURE` only where /p b m/ is expected within the word
  window (±60 ms), and inject one where the expectation exists but the energy dip did not clear
  `_CLOSURE_PEAK_FRACTION`, letting the dip refine the onset inside the window. This is the item the
  four-of-six under-count points at.
- **Mouth shut through nasals.** Gate the nasal override on expected /m n ŋ/, and let "Hmm" be
  recognised from text rather than from a murmur spectrum. This is the one that fixes 5 of 6 voices,
  and it is also a *veto*: expected-nasal-absent suppresses the false latches on /w l ð/ and voice
  bars that item 2 of the plan of record is still chasing.
- **Rounding pins for /u o w/** and **labiodental /f v/**: pin rounding from phone identity instead
  of inferring it from low F2, which today marks any low-F2 vowel (/ɑ ɔ/) as rounded ([ROUND-1]).

Review proxy numbers for the closure half: **P/R 0.16/0.38 → 0.87/0.85** at ±60 ms, from 68 proxy
clips with word-uniform placement. Read them as a ceiling for stage 1, not a forecast: the proxy is
espeak formant synthesis with exact phone boundaries, and word-uniform placement is charitable on
short function words.

Do stage 1 on the seven-voice fixtures first, and only then on Deepgram, so cross-voice
generalization is the headline and not an afterthought.

---

## 6. Stage 2 — fixed-lag streaming alignment

Viterbi over phone classes (not phones) with a left-to-right topology plus skip/insert arcs; word
`pts` pin word boundaries as anchors; emissions from the existing DSP features. Fixed lag within the
≤ 3-hop budget. A per-sentence alignment-cost gate falls back to DSP when the text is late (TOKEN
mode), renormalized, or mispronounced — this is what makes §2's fact 3 safe.

Prototype evidence (independent verifier, DTW over 34 espeak clips): **0.81/0.80 with provider word
anchors, 53 ms median onset error**, against 0.68/0.68 text-only and 0.148/0.406 DSP-only.

Stage 2 only earns its complexity if stage 1's word-uniform placement leaves a measured onset error
that matters. Measure that before building it: if stage 1's onsets already land inside ±60 ms on the
real fixtures, the Viterbi buys precision nobody has asked for. Treat this as a decision point, not
a commitment.

---

## 7. Stage 3 — tiny learned emission scorer (conditional; reverses a decision)

~46 features already computed by the DSP/FFT path, 20–50 k parameters as `.npz`, numpy inference at
~5–10 µs/hop, trained offline on synthesized multi-voice data and evaluated on **held-out voices**.
Optional openness/width/rounding heads. Fusion stays `lerp(dsp_target, model_target, confidence)`
with DSP as fallback. (`onnxruntime~=1.24.3` is also an unconditional `pipecat-ai` requirement, so
an ONNX path would add no dependency either — but at 20–50 k parameters a numpy forward pass is
fewer moving parts than a session, and keeps the copy-the-directory property.)

This contradicts the plan of record's "Not planned: a trained model or learned classifier in the
signal path". Reopening it needs to be explicit and conditional, so:

- **Gate:** only if stages 1–2 leave a *measured* residual on the seven-voice corpus that matters to
  a client, and specifically for the text-less providers (regime (c) in §2) where stages 1–2 do
  nothing at all. That is the honest case for it: a learned scorer is the only thing in this
  document that helps a provider that sends no text.
- **Scope:** emission scoring for the categorical decisions. Vowels stay on the DSP path.
- **Cost to be stated before starting:** a training pipeline, a held-out-voice split, and a model
  artefact in the repo. That cost is what got it ruled out; the seven-voice result changes the
  benefit side, not the cost side.

---

## 8. Measurement, and the circularity trap

**The trap:** if closures are gated on CMUdict /p b m/ windows and then scored against CMUdict
/p b m/ windows, precision is 1.0 by construction and the benchmark has measured nothing. Every
proxy number quoted above comes from espeak, where the phone boundaries are *exact and independent*
of the detector. On the real fixtures no such ground truth exists — `accuracy.py` measures event
**counts** against corpus bars (`closures`, `nasals`, `silences`, `nasal_fraction`), never
precision/recall.

Honest gates, in order of preference:

1. **Cross-voice count agreement (free, non-circular, and the thing that actually broke).** Closure
   counts inside the corpus bars on all 7 voices, nasal-hum passing on all 7, `nasal_fraction` under
   its bars. This is the seven-voice failure restated as a pass/fail, and text cannot fake it: a
   wrong lexicon injects wrong counts.
2. **Negative controls.** Sentences with no /p b m/ must produce no closures; a nasal-free sentence
   must not latch the override. Cheap to add to the corpus, impossible to satisfy by construction.
3. **Openness inside expected bilabial spans.** A shape measure on the DSP output, not a
   detector-scoring-itself measure — the proxy read 0.42 → 0.28 when aligned spans force closure.
4. **Hand-labelled onsets on a small held-out subset** (one or two sentences × 7 voices) for the one
   number that is genuinely wanted: onset error in ms. This is the only non-circular source of
   closure P/R on real audio, and it is a couple of hours of work, not a tooling project.
5. **Timing, separately.** `benchmarks.record` already captures word `pts` and — verified —
   **replays them** (`record.py:440–443` reconstructs `TTSTextFrame`s with `pts` on the replay
   clock), so `--reanalyze` is already a text-tier rig over retained audio with no TTS calls. Two
   gaps to close: the replay emits no `AggregatedTextFrame` sentence anchor, and `accuracy.py`
   drives the analyzer through the ingest mirror rather than the processor, so it has no text path
   at all — though it holds `Sentence.text` already.

Every stage is gated by `uv run python -m benchmarks.accuracy --offline --compare` on **both**
providers and **all 7 voices**, plus `benchmarks.record --reanalyze` before/after for anything that
touches timing. The single-voice-tuning caveat from the plan of record now reads: a seven-voice
corpus is the minimum, and a result that holds on Cartesia only is not a result.

---

## 9. Risks and stop conditions

- **The lexicon is wrong about the audio.** Normalization, TTS transformations (§2 fact 3), proper
  nouns, numbers. Mitigated by the alignment-cost gate; measured by how often the gate fires. If it
  fires on more than a small fraction of ordinary sentences, stage 1 is not viable as shipped.
- **Text raises the floor and lowers the ceiling.** Gating closures on expectation removes true
  positives the DSP found in words the lexicon mis-covers. Watch recall, not only precision — the
  proxy's veto-only step read 0.81/**0.37**, i.e. precision bought with half the recall. Injection
  is what recovers it, and injection is the riskier half.
- **Scope creep into a phoneme pipeline.** The project's three properties (fast, light, drop-in) are
  the product. A stage that adds a dependency, a model file, or more than 3 hops of lookahead is a
  different project; say so and stop.
- **Stop condition.** If stage 1 on seven voices does not move the cross-voice count agreement in
  §8.1, stop. The whole case for reopening this is cross-voice robustness; if that does not
  materialise, the original parking reason ("no client needs bilabial precision") stands.

---

## 10. What this changes in the plan of record

[README.md](README.md) is edited as follows:

- **Parked → Steps forward.** "Text-informed events" moves out of Parked, pointing here.
- **"Not planned: a trained model or learned classifier in the signal path"** — amended, not
  deleted. It stays not planned as the primary path; §7 reopens it *conditionally*, scoped to a
  classifier-only emission scorer for text-less providers, gated on a measured residual.
- **"Not planned: the L4 phoneme-target layer and composite v2"** — amended. Composite v2 stays out.
  The minimum ground-truth needed to measure §8 (held-out hand labels, and expected-phone windows
  from an offline aligner used dev-side only) comes in as a measurement tool, not as a scored layer.
- **"Not planned: provider-viseme (Tier 1) or timestamp + G2P (Tier 2) analyzers"** — clarified,
  not reopened. This plan uses Tier 2's inputs but not its shape: an optional prior over the DSP,
  not a separate analyzer. What stays out is any code path whose output *depends* on text arriving.
  The same distinction is added to the "drop-in for any provider" bullet at the top, so the
  property is visibly held rather than quietly spent.
- Items 2 ([NASAL] semantics) and 4 ([CONS-1] energy-gated closures) in Steps forward overlap this
  work and are cross-noted rather than duplicated — they are the DSP halves of the same two
  decisions.

## 11. Open questions

1. Are the seven voices' `formant_ceiling` values set per pitch, or inherited? The male-voice F2
   result is unreadable until that is pinned down (§1a).
2. Does the pause probe fail because the detector is wrong or because the bar is (§4, item 4)? Answer
   before attributing it to anything.
3. Is a `labiodental`/`stop` event kind worth a wire change, given that no client renders events
   today ([CLIENT-2])? Defer until after stage 2.
4. Which of the seven voices are held out? Stage 3 needs a voice split decided before stage 1 tunes
   anything, or the held-out set is already contaminated.
