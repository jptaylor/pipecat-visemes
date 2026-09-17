"""How early does each lipsync batch get emitted relative to its audio playout, under different TTS delivery pacing?
lead = (t0 + window_start) - t_emit   (t0 = clock time the first audio chunk passed the processor).
Positive lead = batch exists before its audio plays; the client needs >= ~0 (scheduling_lead 200 ms is the design)."""
import asyncio, sys
import numpy as np
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "server"))
sys.path.insert(0, str(_HERE))
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame, run_test
from lipsync.frames import TTSLipsyncFrame
from lipsync.lipsync_processor import LipsyncProcessor, LipsyncParams
from tests.test_lipsync_processor import make_tts_frames
from tests.synth import synth_vowel

class Tap(FrameProcessor):
    def __init__(self):
        super().__init__(); self.t0 = None; self.rows = []
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = self.get_clock().get_time()
        if isinstance(frame, TTSAudioRawFrame) and self.t0 is None: self.t0 = now
        if isinstance(frame, TTSLipsyncFrame): self.rows.append((frame.window_start, frame.window_end, frame.pts, now))
        await self.push_frame(frame, direction)

async def run(pacing_sleep, label, secs=3.0, params=None):
    pcm = np.concatenate([synth_vowel(700, 1200, secs=secs/2), synth_vowel(300, 2300, secs=secs/2)])
    frames = []
    for f in make_tts_frames(pcm, "ctx-1", chunk_ms=20):
        frames.append(f)
        if pacing_sleep and isinstance(f, TTSAudioRawFrame): frames.append(SleepFrame(sleep=pacing_sleep))
    proc = LipsyncProcessor(params=params or LipsyncParams()); tap = Tap()
    await run_test(Pipeline([proc, tap]), frames_to_send=frames, start_timeout=5.0)
    rows = tap.rows; t0 = tap.t0
    leads = [((t0 + ws * 1e9) - t_emit) / 1e6 for ws, we, pts, t_emit in rows]
    print(f"{label:34s} batches={len(rows):2d} clamped={proc.stats['pts_clamped']:2d} | lead ms per batch (audio-time - emit-time): " + " ".join(f"{l:+.0f}" for l in leads[:8]) + (" ..." if len(leads) > 8 else ""))

async def main():
    print("chunk = 20 ms of audio per TTSAudioRawFrame")
    await run(0.0, "burst (all audio at once)")
    await run(0.002, "10x real time (2 ms per 20 ms chunk)")
    await run(0.010, "2x real time")
    await run(0.020, "1x real time (paced like playout)")
    await run(0.020, "1x real time, lead=400ms", params=LipsyncParams(scheduling_lead_ms=400))
if __name__ == "__main__": asyncio.run(main())
