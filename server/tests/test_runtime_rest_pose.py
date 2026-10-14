"""The client runtimes rest the mouth exactly where the analyzer does.

Silence decays toward the analyzer's _REST_POSE and every utterance closes on it,
while the runtimes ease to their own rest pose once keyframes run out. If the two
differ, the mouth drifts after speech to a pose the server never sends, and a client
that measures articulation as distance from rest never reads the mouth as closed
(jptaylor/pipecat-visemes#2).
"""

import re
import unittest
from pathlib import Path

from lipsync.formant_lipsync_analyzer import _REST_POSE

RUNTIMES = Path(__file__).resolve().parents[2] / "runtimes"


class TestRuntimeRestPose(unittest.TestCase):
    def test_typescript_rest_pose(self):
        source = (RUNTIMES / "typescript/src/feed.ts").read_text(encoding="utf-8")
        block = re.search(r"export const REST_POSE = \{(.*?)\};", source, re.S).group(1)
        pose = {key: float(value) for key, value in re.findall(r"(\w+): ([\d.]+)", block)}
        self.assertEqual((pose["openness"], pose["width"], pose["rounding"]), _REST_POSE)

    def test_swift_rest_pose(self):
        source = (RUNTIMES / "swift/Sources/PipecatVisemes/LipsyncTimeline.swift").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"static let rest = Pose\(openness: ([\d.]+), width: ([\d.]+), rounding: ([\d.]+)\)",
            source,
        )
        self.assertEqual(tuple(float(value) for value in match.groups()), _REST_POSE)
