"""L4 prototype: phoneme-aligned viseme ground truth from espeak-ng, scored against the analyzer.

Synthetic (formant-synth) speech: numbers are NOT representative of neural TTS, but the
ground truth is exact, so this measures detector logic (closures, nasals, consonant shapes,
lag) in a way the Praat harness cannot.
"""

import asyncio, sys, json, argparse
import numpy as np
import pathlib as _pl

_HERE = _pl.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "server"))
sys.path.insert(0, str(_HERE))
import soxr
from espeak_synth import synth, init
from lipsync.formant_lipsync_analyzer import FormantLipsyncAnalyzer
from lipsync.base_lipsync_analyzer import LipsyncAnalysisContext
from lipsync.types import LipsyncEventKind

# IPA phoneme -> (class, openness, width, rounding)
T = {}


def _add(syms, cls, o, w, r):
    for s in syms.split():
        T[s] = (cls, o, w, r)


_add("p b m", "BILABIAL", 0.0, 0.4, 0.1)
_add("f v", "LABIODENTAL", 0.1, 0.5, 0.0)
_add("w", "ROUNDED", 0.15, 0.1, 0.9)
_add("uː u ʊ", "ROUNDED", 0.2, 0.15, 0.85)
_add("oʊ oː ɔː ɒ oːɹ ɔːɹ ɔ o", "ROUNDED", 0.45, 0.2, 0.7)
_add("ɔɪ", "ROUNDED", 0.45, 0.35, 0.5)
_add("iː i", "SPREAD", 0.15, 0.95, 0.0)
_add("ɪ ɪ‍ə ɪə", "SPREAD", 0.25, 0.8, 0.0)
_add("eɪ e", "SPREAD", 0.35, 0.8, 0.0)
_add("ɛ ɛɹ eə", "SPREAD", 0.45, 0.7, 0.0)
_add("æ a", "OPEN", 0.75, 0.65, 0.0)
_add("ɑː ɑ ʌ ɐ ɑːɹ", "OPEN", 0.85, 0.5, 0.0)
_add("aɪ aɪə", "OPEN", 0.7, 0.6, 0.0)
_add("aʊ", "OPEN", 0.7, 0.35, 0.4)
_add("ə ɚ ɜː ɜ əl ən əm ʊɹ ʊə", "NEUTRAL", 0.35, 0.45, 0.1)
_add("ʃ ʒ tʃ dʒ", "PROTRUDED", 0.15, 0.3, 0.6)
_add("s z", "SPREAD_CLOSED", 0.15, 0.7, 0.0)
_add("ɹ r", "NEUTRAL_C", 0.2, 0.35, 0.4)
_add("t d ɾ l θ ð k ɡ g h j ʔ", "NEUTRAL_C", 0.25, 0.5, 0.1)
_add("n ŋ", "NASAL_OPEN", 0.1, 0.45, 0.1)
SIL = ("SIL", 0.15, 0.35, 0.1)
BILABIAL_CLOSURE = {"p", "b", "m"}
NASALS = {"m", "n", "ŋ"}
OTHER_STOPS = {"t", "d", "k", "ɡ", "g", "ɾ"}

SENTENCES = {
    "vowel-aa": "Father was calm as he walked past the palm trees.",
    "vowel-ee": "See the green trees? Please believe me, these seeds are free.",
    "vowel-oo": "Soon the new moon grew blue, and the room felt cool.",
    "bilabial-mama": "Mama made more mashed potatoes for my mother.",
    "bilabial-bob": "Bob put a big pepper in the paper bag.",
    "nasal-hum": "Hmm. Hmm, hmm.",
    "pause-probe": "Wait for it. Now continue talking normally.",
    "harvard-01": "The birch canoe slid on the smooth planks.",
    "harvard-02": "Glue the sheet to the dark blue background.",
    "harvard-03": "It's easy to tell the depth of a well.",
    "harvard-04": "These days a chicken leg is a rare dish.",
    "harvard-05": "Rice is often served in round bowls.",
    "fricative-ff": "Five fine fellows fed the puffy fish very fast.",
    "alveolar-tt": "Tiny toads took turns tasting tea today.",
    "velar-kk": "Kicking cans, the cook kept counting cakes.",
    "minimal-pairs": "Pat, tat, cat. Bat, dad, gag. Mat, gnat, hang.",
    "shush": "She showed the shy chef the cheap shoes.",
}
VOICES = [("en-us", 50), ("en-us+f3", 50), ("en-us+m3", 40), ("en-gb+f4", 70)]


def segments(events, total):
    """(start, end, ipa) phoneme segments from espeak events."""
    ph = [(t, s) for t, s, _, _ in events if s not in ("<word>", "<sent>", "<end>")]
    segs = []
    for i, (t, s) in enumerate(ph):
        end = ph[i + 1][0] if i + 1 < len(ph) else total
        segs.append((t, end, s if s else "_"))
    return segs


def lookup(sym):
    if sym in T:
        return T[sym]
    if sym in ("", "_", " "):
        return SIL
    # strip length/stress marks and retry
    base = sym.replace("ː", "").replace("ˈ", "").replace("ˌ", "")
    if base in T:
        return T[base]
    for k in T:
        if sym.startswith(k):
            return T[k]
    return None


async def analyze(pcm16k):
    a = FormantLipsyncAnalyzer(collect_debug=True)
    await a.start(16000)
    ctx = LipsyncAnalysisContext("x", 16000)
    kfs, evs = [], []
    for i in range(0, len(pcm16k), 320):
        r = await a.analyze(pcm16k[i : i + 320], ctx)
        kfs += r.keyframes
        evs += r.events
    r = await a.flush(ctx)
    kfs += r.keyframes
    evs += r.events
    return kfs, evs, a.debug_features


def score(kfs, evs, debug, segs):
    offs = np.array([d.offset for d in debug])
    n = len(offs)
    cls = [None] * n
    tgt = np.zeros((n, 3))
    unknown = set()
    for i, t in enumerate(offs):
        sym = "_"
        for s, e, p in segs:
            if s <= t < e:
                sym = p
                break
        row = lookup(sym)
        if row is None:
            unknown.add(sym)
            row = ("NEUTRAL_C", 0.25, 0.5, 0.1)
        cls[i] = row[0]
        tgt[i] = row[1:]
    if len(kfs) < 2:
        return None, unknown
    ko = np.array([k.offset for k in kfs])
    ours = np.stack(
        [
            np.interp(offs, ko, [getattr(k, a) for k in kfs])
            for a in ("openness", "width", "rounding")
        ],
        1,
    )
    speech = np.array([c != "SIL" for c in cls])
    m = {}
    for j, name in enumerate(("openness", "width", "rounding")):
        m[f"{name}_mae"] = float(np.mean(np.abs(ours[speech, j] - tgt[speech, j])))
        a, b = ours[speech, j], tgt[speech, j]
        m[f"{name}_r"] = (
            float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
        )
    # lag via cross-correlation of openness (positive = ours late), hops
    a = ours[:, 0] - ours[:, 0].mean()
    b = tgt[:, 0] - tgt[:, 0].mean()
    best, bestc = 0, -1e9
    for lag in range(-8, 9):
        if lag >= 0:
            c = float(np.dot(a[lag:], b[: n - lag]))
        else:
            c = float(np.dot(a[: n + lag], b[-lag:]))
        if c > bestc:
            bestc, best = c, lag
    m["lag_hops"] = best
    # closures
    gt_cl = [(s, e) for s, e, p in segs if p in BILABIAL_CLOSURE]
    gt_other = [(s, e) for s, e, p in segs if p in OTHER_STOPS]
    our_cl = [e.offset for e in evs if e.kind == LipsyncEventKind.CLOSURE]
    tol = 0.06
    hit = sum(any(s - tol <= o <= e + tol for s, e in gt_cl) for o in our_cl)
    wrong_place = sum(any(s - tol <= o <= e + tol for s, e in gt_other) for o in our_cl)
    recall_hits = sum(any(s - tol <= o <= e + tol for o in our_cl) for s, e in gt_cl)
    m["closure_gt"] = len(gt_cl)
    m["closure_ours"] = len(our_cl)
    m["closure_precision"] = hit / len(our_cl) if our_cl else float("nan")
    m["closure_recall"] = recall_hits / len(gt_cl) if gt_cl else float("nan")
    m["closure_on_other_stops"] = wrong_place
    # bilabial openness: how closed are we during p/b/m (hops)?
    bil = np.array([c == "BILABIAL" for c in cls])
    m["openness_mean_on_bilabial"] = float(ours[bil, 0].mean()) if bil.any() else float("nan")
    # nasal coverage
    gt_n = [(s, e) for s, e, p in segs if p in NASALS and e - s >= 0.05]
    cov = []
    for s, e in gt_n:
        idx = (offs >= s) & (offs < e)
        if idx.any():
            cov.append(float((ours[idx, 0] <= 0.2).mean()))
    m["nasal_segments"] = len(gt_n)
    m["nasal_closed_fraction"] = float(np.mean(cov)) if cov else float("nan")
    m["nasal_events"] = sum(e.kind == LipsyncEventKind.NASAL for e in evs)
    # per-class means of our output
    per = {}
    for c in sorted(set(cls)):
        idx = np.array([x == c for x in cls])
        per[c] = (int(idx.sum()),) + tuple(round(float(v), 2) for v in ours[idx].mean(0))
    m["per_class"] = per
    m["keyframes"] = len(kfs)
    m["dur"] = float(offs[-1])
    return m, unknown


async def main(args):
    rate = init()
    rows = []
    unknown_all = set()
    per_class_acc = {}
    for voice, pitch in VOICES:
        if args.voices and voice not in args.voices.split(","):
            continue
        for sid, text in SENTENCES.items():
            if args.sentences and sid not in args.sentences.split(","):
                continue
            audio, events = synth(text, voice=voice, pitch=pitch)
            pcm = soxr.resample(audio, rate, 16000).astype(np.float32) / 32768.0
            segs = segments(events, len(audio) / rate)
            kfs, evs, debug = await analyze(pcm)
            m, unknown = score(kfs, evs, debug, segs)
            unknown_all |= unknown
            if m is None:
                continue
            for c, (cnt, o, w, r) in m["per_class"].items():
                acc = per_class_acc.setdefault(c, [0, 0.0, 0.0, 0.0])
                acc[0] += cnt
                acc[1] += o * cnt
                acc[2] += w * cnt
                acc[3] += r * cnt
            rows.append((voice, sid, m))
            if args.verbose:
                print(
                    f"{voice:10s} {sid:14s} oMAE {m['openness_mae']:.2f} r {m['openness_r']:.2f} | wMAE {m['width_mae']:.2f} r {m['width_r']:.2f} | rndMAE {m['rounding_mae']:.2f} r {m['rounding_r']:.2f} | lag {m['lag_hops']:+d} | clos gt/ours {m['closure_gt']}/{m['closure_ours']} P {m['closure_precision']:.2f} R {m['closure_recall']:.2f} other {m['closure_on_other_stops']} | bilab open {m['openness_mean_on_bilabial']:.2f} | nasal closed {m['nasal_closed_fraction']:.2f} ({m['nasal_segments']} seg, {m['nasal_events']} ev) | kf/s {m['keyframes'] / m['dur']:.1f}"
                )

    def agg(k):
        v = np.array([r[2][k] for r in rows], dtype=float)
        return np.nanmean(v)

    print("\nAGGREGATE over", len(rows), "clips")
    for k in (
        "openness_mae",
        "openness_r",
        "width_mae",
        "width_r",
        "rounding_mae",
        "rounding_r",
        "lag_hops",
        "closure_precision",
        "closure_recall",
        "openness_mean_on_bilabial",
        "nasal_closed_fraction",
    ):
        print(f"  {k:28s} {agg(k):.3f}")
    print(
        f"  closures: gt {sum(r[2]['closure_gt'] for r in rows)} ours {sum(r[2]['closure_ours'] for r in rows)} on-other-stops {sum(r[2]['closure_on_other_stops'] for r in rows)}"
    )
    print("\nPER-CLASS mean of OUR (openness, width, rounding)  [target in brackets]")
    tgt_by_class = {}
    for sym, (c, o, w, r) in T.items():
        tgt_by_class.setdefault(c, (o, w, r))
    tgt_by_class["SIL"] = SIL[1:]
    for c, (cnt, o, w, r) in sorted(per_class_acc.items(), key=lambda x: -x[1][0]):
        t = tgt_by_class.get(c, ("?",) * 3)
        print(
            f"  {c:14s} n={cnt:5d}  open {o / cnt:.2f} [{t[0]}]  width {w / cnt:.2f} [{t[1]}]  round {r / cnt:.2f} [{t[2]}]"
        )
    if unknown_all:
        print("\nunknown IPA symbols:", sorted(unknown_all))
    if args.json:
        json.dump(
            [(v, s, {k: val for k, val in m.items() if k != "per_class"}) for v, s, m in rows],
            open(args.json, "w"),
            indent=1,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--voices")
    p.add_argument("--sentences")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json")
    asyncio.run(main(p.parse_args()))
