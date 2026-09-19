# Text-informed events: stage-1 review — 2026-09-19

Branch: `codex/text-informed-events`. Control: clean `main` at `8ea78f0`.
The [input-only checkpoint](text-informed-events-results.md) remains historical.

**Decision: keep this opt-in.** Stage 1 improves hum detection and removes false
events on the held-out negative controls, with a small CPU cost. It is not an
across-the-board accuracy improvement: a bilabial count check regresses, timed
shape hints reduce correlation slightly, and independent phone-onset accuracy
has not been established. No default was changed and no baseline was replaced.

## What was implemented

- A pinned, licensed CMUdict packed into 918,877 bytes on disk, 2,456,586 bytes
  unpacked. One shared byte table and a bounded 2,048-entry lookup cache, not
  a Python dictionary per bot. No added runtime dependency or provider calls.
- Opt-in `LipsyncParams(text_events_enabled=True)` (or the analyzer constructor
  of the same name). `text_prior_enabled` remains observation-only.
- Untimed English text provides conservative phone-inventory vetoes and
  explicit hum recognition. Usable word intervals add uniform phone placement,
  closure injection/refinement, /m/ closure, rounded-vowel and /f,v/ shape hints.
  /n, ng/ permit acoustic nasal evidence, rather than forcing closed lips.
- Existing DSP formants, adaptation, silence decisions, batching and wire format
  remain unchanged. No extra audio lookahead is introduced; the centered median
  still costs one 20 ms hop.
- Missing/OOV/unsupported/transformed/late text falls back. Multiple anchors in
  a shared context also fall back because sentence alignment is not yet solved.
  Limits: 256 words / 2,048 phones / 64 characters per word; unsuitable priors do not grow the
  runtime state indefinitely. A small OOV pronunciation approximation exists,
  but is deliberately not trusted for categorical overrides.

The clock origin uses first audio receipt and the previous context's last word
PTS, not the pre-synthesis sentence PTS or queued playout time. Equal word PTS
are grouped; a final word's duration is not invented before audio ends. Timing
learned more than 60 ms after a pending closure's offset cannot retrospectively
veto it. Already-delivered keyframes are never revised.

These are conservative checks, **not a speech recognizer or alignment-cost
model**. Plausible incorrect English text can still worsen output. The stronger
"wrong text is never worse" guarantee in the original proposal was corrected.

## Accuracy: same audio within every pair

No composite weights or original expectation bars were changed. The original
192 takes are preserved. A separate 192-take set captures actual sentence/word
availability, with two takes of all 12 sentences across seven Cartesia voices
and one Deepgram voice. It is deliberately **not** compared across different
syntheses: each row below compares DSP and text on identical PCM.

| Corpus / input | DSP composite | Text composite | Passing checks, DSP → text |
| --- | ---: | ---: | ---: |
| Fresh Cartesia, captured timing, 168 clips | 84.2541 | 84.8780 | 200/224 → 211/224 |
| Fresh Deepgram, captured untimed anchors, 24 clips | 91.3400 | 91.2360 | 29/32 → 29/32 |
| Original Cartesia, assumed untimed text, 168 clips | 86.0425 | 86.9104 | 204/224 → 214/224 |
| Original Deepgram, assumed untimed text, 24 clips | 91.9202 | 91.8700 | 30/32 → 30/32 |

Fresh Cartesia details:

- Hum NASAL-presence checks: **5/14 → 14/14**. Hum closed-mouth checks:
  **10/14 → 13/14**. A bright voice still misses one openness bar.
- Bilabial count checks: **55/56 → 54/56**. One formerly passing mama take
  (`5ee9feff`, take 2) now fails the minimum; another mama take already failed.
  This is a real regression, not hidden by the higher composite.
- Openness correlation: **0.6320 → 0.6286**. Width correlation:
  **0.6214 → 0.6095**. Raw F1/F2 and silence detection are unchanged.
- Only **12.7%** of analyzed hops had a usable timed phone interval (2,895 of
  22,853). Others used the inventory/hum prior. Deepgram supplied no word timing.
  This is the main limitation of word-timed stage 1 in actual streaming.

| Cartesia voice ID prefix | Checks / 32, DSP → text |
| --- | ---: |
| `71a7ad14` | 31 → 31 |
| `5ee9feff` | 28 → 29 |
| `ef191366` | 29 → 31 |
| `47c38ca4` | 28 → 30 |
| `d1d9c946` | 27 → 30 |
| `62ae83ad` | 27 → 30 |
| `db6b0ed5` | 30 → 30 |

### Held-out negative controls

Two new sentences, two takes per voice, synthesized only after the first event
implementation: “See the stars as they rise.” and “We will see you early.”
Neither contains a bilabial or nasal in ordinary English pronunciation. These
are separate from the saved corpus and were not used to tune thresholds.

| Provider | Clips | False closures | False NASAL events | Passing checks |
| --- | ---: | ---: | ---: | ---: |
| Cartesia, seven voices | 28 | 42 → 0 | 8 → 0 | 28/56 → 56/56 |
| Deepgram | 4 | 9 → 0 | 4 → 0 | 0/8 → 8/8 |

These 63 removed false events are useful evidence for inventory vetoes. They
are **not** onset precision/recall: an inventory veto can satisfy a negative
control by construction. Counts and word-uniform phone windows must not be
presented as independent phonetic ground truth. Hand-labelled onsets and visual
A/B review are still needed before deciding whether to replace DSP defaults.

## Performance and latency

### CPU and memory

macOS arm64, Python 3.11.8, Pipecat 1.10.0. Each trial processes the same 192
fresh clips / 532.801 seconds of audio, after a warmup pass. No Praat, debug
feature collection, network, fixture I/O or resampling is inside the timed
section. Context construction, causal text snapshots, lookup, analysis and
flush **are** included. Cold dictionary startup is measured separately.

Six trials per mode in separate processes, in main/text/text/main order:

| Measurement | Clean main | Text events |
| --- | ---: | ---: |
| Median CPU per 20 ms hop | 194.15 µs | 198.34 µs |
| Whole-trial range | 192.21–200.16 µs | 197.71–202.89 µs |
| Median CPU/audio ratio | 0.00951 | 0.00971 |
| Approximate fraction of one core, one real-time stream | 0.951% | 0.971% |
| p95 analysis-call CPU | 0.885 ms | 0.907 ms |
| Cold analyzer startup | 0.044–0.047 ms | 5.02–5.03 ms |

This run gives **+2.2%** steady-state CPU. An earlier forward/reverse sweep
gave +3.3%; do not interpret this as a precise production capacity estimate.
Both remain approximately 1% of one core per continuously speaking bot on this
machine. Calls may contain several hops because the resampler emits bursts;
the p95 call number is not a per-hop percentile. Constructor/start cost is
separate from steady-state CPU, and real bot/network/client cost is excluded.

The earlier sweep also checked the unchanged default: clean main 198.63 µs/hop,
branch DSP 198.59 (noise-level difference). Enabling the feature with **no text**
was 203.79 µs/hop, approximately +2.6%; it still pays the opt-in model checks.

Traced allocations during construction (not whole-process RSS):

- First enabled analyzer: 2,598,379 retained bytes versus 139,992 for DSP.
  Increment: **2.35 MiB**, overwhelmingly the shared lexicon.
- Each additional idle analyzer, measured over 16 instances: 141,395 bytes
  versus 139,917.5, approximately **1.44 KiB extra per bot** before active text
  and phoneme windows. The dictionary is not copied per bot.
- Decoded-word cache and active context allocations are additional and bounded;
  the idle figure is not a claim about peak memory during long utterances.
- Vendored compressed asset: **0.88 MiB**. The default mode never loads it.

These are local microbenchmarks with ordinary OS scheduling variation, not
confidence intervals. Raw trials: `performance-stage1-{main,text}-final-{a,b}.json`;
the earlier four-mode sweep is `performance-stage1-{main,dsp,no-text,text}-{a,b}.json`.
After the final input-length and clock-lifecycle safeguards, a further three-trial
paired check measured **196.62 → 199.77 µs/hop (+1.6%)**, consistent with the same
small overhead (`performance-stage1-{main,text}-confirm.json`). The enabled
runtime fingerprint for that check and the final accuracy rerun is
`1e8772c3f3887f8308d5eea8e56513aef6a7c293000c1e57c11ba7bd5c661003`.

### Delivery replay

Four sequential real-time replays (main/text/text/main), 19 examples each, reuse
`cartesia-71a7ad14-20260919-123250` and its exact TTS audio/arrival schedule. The
main runs use a clean detached checkout and the same Python environment. The
legacy recording lacks anchors, so **all 19 text-run anchors are explicitly
assumed early**, labelled in run metadata. Provider word PTS and arrival timing
are retained, not synthesized from total duration.

| Eval run | Median anchor error | Median first batch vs audio start | p95 release lateness | Max release lateness |
| --- | ---: | ---: | ---: | ---: |
| `stage1-main-a` | −5.7 ms | −0.6 ms | 6.6 ms | 18.5 ms |
| `stage1-events-a` | −5.5 ms | −0.5 ms | 6.4 ms | 19.6 ms |
| `stage1-events-b` | −4.7 ms | −0.4 ms | 5.4 ms | 30.7 ms |
| `stage1-main-b` | −3.6 ms | −0.4 ms | 5.5 ms | 36.1 ms |

Release lateness is message arrival minus its scheduled release; anchor error
uses the wire's actual lead/window start relative to recorded audio playout.
The p95 here is the nearest observed percentile. **No systematic added delivery
delay is evident**, and the algorithm adds zero lookahead. These small differences
are ordinary scheduling variation, not a claimed latency improvement. Network
and a real browser/device are not included in the headless replay.

No audio was dropped. Main emitted 355–356 batches and 2,027–2,034 keyframes;
text emitted 356 batches and 2,019–2,024 keyframes. Main had 142 closures in
both runs and 82–83 NASAL events; text had **105–112 closures and 65–69 NASAL
events**. The variation is materially greater for text events: word-interval
availability depends on when the analysis task drains its buffer. This remaining
streaming sensitivity is **not** hidden by deterministic offline accuracy scores
and is another reason to retain the baseline. Fewer events are not necessarily
better events.

Open the client's existing **Eval** tab, choose recording
`cartesia-71a7ad14-20260919-123250`, and compare `stage1-main-a/b` with
`stage1-events-a/b`, in both **as delivered** and **ideal** modes. Start with
`nasal-hum`, `bilabial-mama`, `bilabial-bob`, and the longer `explanation`.

## Validation and boundaries

- `uv run --offline pytest -q`: **89 passed**, 45 subtests. Ruff checks and
  formatting pass. One existing `audioop` deprecation warning remains.
- Original DSP outputs still match the pre-change hashes on all **192/192**
  original clips. Saved baseline files are untouched.
- Feature enabled, but text explicitly absent: **416/416** full-precision output
  hashes match DSP, across original takes, fresh takes and negative controls.
- The unit suite covers lexical lookup, OOV/normalization rejection, missing and
  late text, mismatched word prefixes, grouped timestamps, no invented final
  duration, causal word availability, late-event veto protection, silence,
  context reset/interruption, queued word-clock origin, and runtime disabling.
- A pre-existing real-time recorder test assumed the OS never inserted a gap.
  It now checks exact source recovery after removing **recorded** gaps plus
  bounded transport padding; the delivery timing assertions remain intact.
- No independent real-audio onset labels, forced aligner, Viterbi, learned
  scorer, or human visual-quality sign-off is claimed. Those are next-step
  decisions, not silently satisfied promotion gates.

## Next decision

Keep DSP as the default. Review the saved visual comparisons and label a small
held-out set of real bilabial onsets. An ablation retaining inventory/hum benefits
without the uniform timed pins is worth testing before committing to a more
complex aligner. Stage 2 should address the measured late-word and streaming
sensitivity, not assume that better count bars establish phonetic timing quality.

## Reproduction

From `server/` (fresh synthesis requires configured provider credentials):

```bash
# Separate fixtures preserve the original takes. Repeat with --provider deepgram.
uv run python -m benchmarks.accuracy --fixtures benchmarks/fixtures/text-events --tag stage1-dsp
uv run python -m benchmarks.accuracy --offline --fixtures benchmarks/fixtures/text-events --text-events --tag stage1-events --compare benchmarks/results/accuracy-stage1-dsp.json

# Independent controls, not part of the saved baseline.
uv run python -m benchmarks.accuracy --fixtures benchmarks/fixtures/text-events --negative-controls --tag controls-dsp
uv run python -m benchmarks.accuracy --offline --fixtures benchmarks/fixtures/text-events --negative-controls --text-events --tag controls-text --compare benchmarks/results/accuracy-controls-dsp.json

# CPU: run modes sequentially, never alongside other benchmark processes.
uv run python benchmarks/performance.py --fixtures benchmarks/fixtures/text-events --mode dsp --out benchmarks/results/perf-dsp.json
uv run python benchmarks/performance.py --fixtures benchmarks/fixtures/text-events --mode text --out benchmarks/results/perf-text.json
# --source-path /path/to/clean-main/server runs the same harness against old code.

# Retained audio in the existing client Eval tab. The legacy anchor assumption is explicit.
uv run python -m benchmarks.record --reanalyze cartesia-71a7ad14-20260919-123250 --text-events --assume-early-text --tag stage1-events
```

Local detailed accuracy results are gitignored under `server/benchmarks/results/`:
`accuracy-stage1-final-{dsp,events}-{cartesia,deepgram}.json`, plus the
`original-` and `control-` pairs. They retain exact output and PCM hashes,
runtime source/asset fingerprints, configuration and text coverage counters.
Comparisons reject different corpus settings or different hashed PCM.
