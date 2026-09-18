"""Run an A/B ladder of plans/deep-review-2026-09-results.md and print it.

Every variant is a set of harness ``--set`` overrides, stated in full (nothing
inherits the source defaults), so a table reproduces regardless of which
values are currently landed — except for structural changes: the ``dsp``
ladder's numbers in the results note were taken at commit 3dc30fd, before the
trailing median became zero-phase and the 1-hop nasal fast path was removed;
re-run today it scores every row on the current conditioning. Needs cached
fixtures only (``--offline``).

Run from server/:
  uv run python ../plans/experiments/ab_table.py [--ladder dsp|conditioning]
      [--provider deepgram] [--ceiling 5500] [--only 0,4b]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("benchmarks/results")

BASE = {
    "dsp.LPC_ORDER": 12,
    "dsp.SLOT_PRIOR_WEIGHTS": "(0,0,0)",
    "dsp.LPC_FRAME_SIZE": 400,
    "dsp.PITCH_METHOD": "residual",
    "dsp.PITCH_FRAME_SIZE": 640,
    "dsp.NCC_VOICED_THRESHOLD": 0.6,
    "analyzer._VOICED_MIN_PEAK_FRACTION": 0.05,
    "analyzer._NASAL_MISSING_F2_DAMPED": "always",
    "analyzer._C_FIT_LOG10_FULL": 3.0,
    "analyzer._SLEW_MAX_PER_HOP": 0.4,
    "analyzer._ANCHOR_KEYFRAMES": False,
    "analyzer._NASAL_HYSTERESIS_HOPS": 2,
    "analyzer._NASAL_F1_MAX_HZ": "inf",
    "analyzer._NASAL_MID_RATIO_MAX": "inf",
}
ORDER16 = {"dsp.LPC_ORDER": 16}
PRIOR = {**ORDER16, "dsp.SLOT_PRIOR_WEIGHTS": "(0,1,1)"}
NCC = {**PRIOR, "dsp.PITCH_METHOD": "ncc"}

DSP_VARIANTS = [
    ("0 baseline (order 12, residual pitch)", {}),
    ("1a order 14", {"dsp.LPC_ORDER": 14}),
    ("1b order 16", ORDER16),
    ("1c order 18", {"dsp.LPC_ORDER": 18}),
    ("2a 1b + slot prior (1,1,1)", {**ORDER16, "dsp.SLOT_PRIOR_WEIGHTS": "(1,1,1)"}),
    ("2b 1b + slot prior (0,1,1)", PRIOR),
    ("3 2b + LPC window 40 ms (residual pitch)", {**PRIOR, "dsp.LPC_FRAME_SIZE": 640}),
    ("4a 2b + NCC, 25 ms pitch frame", {**NCC, "dsp.PITCH_FRAME_SIZE": 400}),
    ("4b 2b + NCC, 40 ms pitch frame", NCC),
    ("4c 4b + LPC window 30 ms", {**NCC, "dsp.LPC_FRAME_SIZE": 480}),
    ("4d 4b + LPC window 40 ms", {**NCC, "dsp.LPC_FRAME_SIZE": 640}),
    ("5 4b + nasal: missing F2 not damped", {**NCC, "analyzer._NASAL_MISSING_F2_DAMPED": "never"}),
]

# Second ladder (2026-09-18, on the landed DSP bundle): conditioning and the
# nasal state machine. The zero-phase median and the removal of the nasal
# fast path are structural, so their "before" rows cannot be expressed as
# overrides; the results note quotes them from tags 11-cond-* / 12-nasal-*.
COND_VARIANTS = [
    ("6a zero-phase median, slew 0.25", {**NCC, "analyzer._SLEW_MAX_PER_HOP": 0.25}),
    (
        "6b 6a + anchor keyframes",
        {**NCC, "analyzer._SLEW_MAX_PER_HOP": 0.25, "analyzer._ANCHOR_KEYFRAMES": True},
    ),
    ("6c slew 0.4 (landed)", NCC),
    ("6d slew 0.6", {**NCC, "analyzer._SLEW_MAX_PER_HOP": 0.6}),
    ("6e no slew", {**NCC, "analyzer._SLEW_MAX_PER_HOP": 1.0}),
    ("7a 6c + nasal F1 <= 400", {**NCC, "analyzer._NASAL_F1_MAX_HZ": 400}),
    ("7b 6c + nasal mid-band <= 0.10", {**NCC, "analyzer._NASAL_MID_RATIO_MAX": 0.10}),
    ("7c 6c + nasal exit 3 hops", {**NCC, "analyzer._NASAL_HYSTERESIS_HOPS": 3}),
]
LADDERS = {"dsp": DSP_VARIANTS, "conditioning": COND_VARIANTS}

COLUMNS = [
    ("composite", "composite", 1),
    ("f1_mae", "f1_mae_hz", 0),
    ("f2_mae", "f2_mae_hz", 0),
    ("f1_cov", "f1_coverage", 2),
    ("f2_cov", "f2_coverage", 2),
    ("open_r", "openness_r", 2),
    ("width_r", "width_r", 2),
    ("voicing", "voicing_agreement", 2),
    ("lag_ms", "openness_lag_ms", 0),
    ("jitter", "openness_jitter", 2),
    ("nasal_frac", "nasal_fraction", 2),
    ("kf/s", "keyframe_rate", 1),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", default="dsp", choices=sorted(LADDERS))
    parser.add_argument("--provider", default="cartesia")
    parser.add_argument("--ceiling", type=float)
    parser.add_argument("--only", help="comma-separated variant number prefixes, e.g. 0,4b,5")
    args = parser.parse_args()

    suffix = f"-c{args.ceiling:.0f}" if args.ceiling else ""
    print(
        f"provider {args.provider}{f', Praat ceiling {args.ceiling:.0f}' if args.ceiling else ''}\n"
    )
    print("| variant | " + " | ".join(c[0] for c in COLUMNS) + " | checks |")
    print("|---|" + "---|" * (len(COLUMNS) + 1))
    for name, overrides in LADDERS[args.ladder]:
        key = name.split()[0]
        if args.only and key not in args.only.split(","):
            continue
        tag = f"ab-{args.ladder}-{args.provider}{suffix}-{key}"
        cmd = [sys.executable, "-m", "benchmarks.accuracy", "--offline", "--tag", tag]
        cmd += ["--provider", args.provider]
        if args.ceiling:
            cmd += ["--ceiling", str(args.ceiling)]
        for const, value in {**BASE, **overrides}.items():
            cmd += ["--set", f"{const}={value}"]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(run.stderr[-2000:])
        payload = json.loads((RESULTS / f"accuracy-{tag}.json").read_text())
        checks = [ok for clip in payload["clips"] for ok in clip["checks"].values()]
        cells = []
        for _, metric, digits in COLUMNS:
            value = payload["composite"] if metric == "composite" else payload["aggregate"][metric]
            cells.append("—" if value is None else f"{value:.{digits}f}")
        print(f"| {name} | " + " | ".join(cells) + f" | {sum(checks)}/{len(checks)} |", flush=True)


if __name__ == "__main__":
    main()
