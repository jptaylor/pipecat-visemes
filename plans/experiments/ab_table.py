"""Run the DSP A/B ladder of plans/deep-review-2026-09-results.md and print it.

Every variant is a set of harness ``--set`` overrides, stated in full (nothing
inherits the source defaults), so the table reproduces regardless of which
values are currently landed. Needs cached fixtures only (``--offline``).

Run from server/:
  uv run python ../plans/experiments/ab_table.py [--provider deepgram] [--ceiling 5500]
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
}
ORDER16 = {"dsp.LPC_ORDER": 16}
PRIOR = {**ORDER16, "dsp.SLOT_PRIOR_WEIGHTS": "(0,1,1)"}
NCC = {**PRIOR, "dsp.PITCH_METHOD": "ncc"}

VARIANTS = [
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

COLUMNS = [
    ("composite", "composite", 1),
    ("f1_mae", "f1_mae_hz", 0),
    ("f2_mae", "f2_mae_hz", 0),
    ("f1_cov", "f1_coverage", 2),
    ("f2_cov", "f2_coverage", 2),
    ("open_r", "openness_r", 2),
    ("width_r", "width_r", 2),
    ("voicing", "voicing_agreement", 2),
    ("kf/s", "keyframe_rate", 1),
]


def main():
    parser = argparse.ArgumentParser()
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
    for name, overrides in VARIANTS:
        key = name.split()[0]
        if args.only and key not in args.only.split(","):
            continue
        tag = f"ab-{args.provider}{suffix}-{key}"
        cmd = [sys.executable, "-m", "benchmarks.accuracy", "--offline", "--tag", tag]
        cmd += ["--provider", args.provider]
        if args.ceiling:
            cmd += ["--ceiling", str(args.ceiling)]
        for const, value in {**BASE, **overrides}.items():
            cmd += ["--set", f"{const}={value}"]
        subprocess.run(cmd, check=True, capture_output=True)
        payload = json.loads((RESULTS / f"accuracy-{tag}.json").read_text())
        checks = [ok for clip in payload["clips"] for ok in clip["checks"].values()]
        cells = []
        for _, metric, digits in COLUMNS:
            value = payload["composite"] if metric == "composite" else payload["aggregate"][metric]
            cells.append("—" if value is None else f"{value:.{digits}f}")
        print(f"| {name} | " + " | ".join(cells) + f" | {sum(checks)}/{len(checks)} |", flush=True)


if __name__ == "__main__":
    main()
