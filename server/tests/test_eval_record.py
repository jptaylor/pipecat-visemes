#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The eval recorder's capture path (benchmarks.record), driven offline."""

import unittest

import numpy as np
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from benchmarks.record import Example, _Recorder, _run_examples, _Source, _strip_gaps
from lipsync.frames import TTSLipsyncFrame
from tests.synth import synth_vowel

SAMPLE_RATE = 24000
CHUNK_BYTES = SAMPLE_RATE // 25 * 2  # 40 ms of int16


def retained_take(secs: float) -> tuple[dict, bytes]:
    """A take as the recorder retains it: 40 ms chunks arriving at 4x real time."""
    samples = synth_vowel(700, 1200, secs=secs, fs=SAMPLE_RATE) * 0.5
    pcm = (samples * 32767).astype(np.int16).tobytes()
    chunks = [
        [round(n * 0.01, 4), len(pcm[offset : offset + CHUNK_BYTES])]
        for n, offset in enumerate(range(0, len(pcm), CHUNK_BYTES))
    ]
    arrival = {
        "ctx": "ctx-1",
        "sample_rate": SAMPLE_RATE,
        "chunks": chunks,
        "words": [],
        "stopped": chunks[-1][0] + 0.01,
        "gaps": [],
    }
    return arrival, pcm


class TestEvalRecorder(unittest.IsolatedAsyncioTestCase):
    async def test_replay_records_audio_as_played_and_message_timing(self):
        arrival, pcm = retained_take(1.2)
        example = Example(id="vowel", text="", tags=[], look_for="")
        source = _Source(arrivals={"vowel": arrival}, pcm={"vowel": pcm})
        takes, clean = await _run_examples([example], source, SAMPLE_RATE)

        self.assertTrue(clean)
        (take,) = takes
        # Audio as played: the TTS audio, padded out to whole transport writes.
        self.assertEqual(take.pcm[: len(pcm)], pcm)
        self.assertLess(len(take.pcm) - len(pcm), CHUNK_BYTES)
        self.assertEqual(take.arrival["gaps"], [])

        self.assertTrue(take.messages)
        for message in take.messages:
            self.assertEqual(message["data"]["type"], "bot-tts-lipsync")
            self.assertEqual(message["data"]["ctx"], "ctx-1")
            # Never released before its scheduled time, and all released while playing.
            self.assertGreaterEqual(message["at"], message["due"] - 0.005)
            self.assertLess(message["at"], take.duration)
        # Once analysis is ahead of playout, batches lead their audio.
        last = take.messages[-1]
        first_offset = min(row[0] for row in last["data"]["kf"] + last["data"]["ev"])
        self.assertLess(last["at"], last["data"]["t0"] + first_offset - 0.1)

    def test_pairs_each_message_with_its_batch_even_when_it_overtakes(self):
        # RTVIServerMessageFrame is a SystemFrame, so the relay's message can
        # reach the tap before the batch it was made from.
        recorder = _Recorder()
        capture = recorder.begin(Example(id="x", text="", tags=[], look_for=""), ctx="ctx-1")

        def message(n: int) -> RTVIServerMessageFrame:
            return RTVIServerMessageFrame(data={"type": "bot-tts-lipsync", "ctx": "ctx-1", "n": n})

        def batch(pts: int) -> TTSLipsyncFrame:
            frame = TTSLipsyncFrame(context_id="ctx-1")
            frame.release_ns = pts
            return frame

        recorder.on_output(message(1), 10)  # overtook its batch
        recorder.on_output(batch(5), 11)
        recorder.on_output(batch(15), 20)  # in order this time
        recorder.on_output(message(2), 21)

        self.assertEqual(
            [(at, due, data["n"]) for at, due, data in capture.messages],
            [(10, 5, 1), (21, 15, 2)],
        )

    def test_strip_gaps_recovers_the_tts_audio(self):
        # 4 bytes of stall silence inside, 2 bytes of write padding at the end.
        played = b"ab" + bytes(4) + b"cdef" + bytes(2)
        self.assertEqual(_strip_gaps(played, [[2, 4]], 6), b"abcdef")


if __name__ == "__main__":
    unittest.main()
