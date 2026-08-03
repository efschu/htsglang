# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The client's downsample-and-frame loop, mirrored so it can be executed.

**What this is, honestly:** a Python transliteration of the JavaScript in
``client/index.html``. It is not the shipped code and cannot prove the shipped
code runs; there is no JS runtime in this venv. It exists because the original
loop was wrong in a way no server-side test could see, and three phone tests
were spent finding it:

The downsampled output was packed into frames and then DROPPED, while only the
input remainder was carried across calls. An AudioWorklet quantum is 128
samples; at the 48 kHz Android picks that is ~42 samples at the 16 kHz
pipeline rate, and a 20 ms frame needs 320. So the packing condition was false
on every call and not one frame was ever produced -- with the microphone open,
the context running and the worklet delivering. The phone reported 400 worklet
calls and 537 chunks against 0 frames.

The front-door harness cannot catch this class at all: it pushes PCM straight
into the WebSocket and never executes a line of the client. Neither can the
server, which sees a release with no audio and cannot distinguish it from a
user who said nothing.

``test_the_old_shape_produces_nothing`` is the can-fail proof: it implements
the previous logic and asserts the failure, so the mirror is shown to
discriminate rather than merely agreeing with whatever it was given.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_client_framing.py -v
"""

import math
import re
import unittest
from pathlib import Path

import numpy as np

PIPELINE_RATE = 16000
FRAME_MS = 20
FRAME = round(PIPELINE_RATE * FRAME_MS / 1000)
QUANTUM = 128  # what an AudioWorklet hands over per call

CLIENT = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/translator/client/index.html"
)


def decimate(merged: np.ndarray, ratio: float):
    """Box-average decimation, exactly as the client does it."""
    out_length = math.floor(len(merged) / ratio)
    out = np.zeros(out_length, dtype=np.float32)
    for i in range(out_length):
        start = math.floor(i * ratio)
        end = min(len(merged), math.floor((i + 1) * ratio))
        out[i] = float(np.mean(merged[start:end])) if end > start else 0.0
    return out, merged[math.floor(out_length * ratio):]


def run_capture(quanta, context_rate, accumulate: bool):
    """Feed ``quanta`` worklet callbacks; return the frames handed to the socket."""
    ratio = context_rate / PIPELINE_RATE
    carry = np.zeros(0, dtype=np.float32)
    pending = np.zeros(0, dtype=np.float32)
    frames = []
    for chunk in quanta:
        merged = np.concatenate([carry, chunk])
        out, carry = decimate(merged, ratio)
        if accumulate:
            pending = np.concatenate([pending, out])
        else:
            # The old shape: the downsampled output lived only for this call.
            pending = out
        start = 0
        while start + FRAME <= len(pending):
            frames.append(pending[start:start + FRAME].copy())
            start += FRAME
        if accumulate:
            pending = pending[start:]
    return frames


def tone(seconds, rate, hz=220.0):
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def quanta_of(signal):
    return [
        signal[i:i + QUANTUM]
        for i in range(0, len(signal) - QUANTUM + 1, QUANTUM)
    ]


class TestFramingMirror(unittest.TestCase):
    def test_the_old_shape_produces_nothing(self):
        """The can-fail proof, and the bug itself.

        One second of speech at the rate Android actually picks, and not a
        single frame reaches the socket.
        """
        for rate in (48000, 44100):
            with self.subTest(context_rate=rate):
                frames = run_capture(
                    quanta_of(tone(1.0, rate)), rate, accumulate=False
                )
                self.assertEqual(
                    frames, [],
                    "the old loop is expected to produce nothing -- if this "
                    "ever passes, the mirror no longer reflects the defect",
                )

    def test_accumulating_produces_a_continuous_stream(self):
        for rate in (48000, 44100, 16000):
            with self.subTest(context_rate=rate):
                frames = run_capture(
                    quanta_of(tone(1.0, rate)), rate, accumulate=True
                )
                # One second at 16 kHz in 20 ms frames is ~50, less whatever
                # the quantum boundary leaves pending.
                self.assertGreaterEqual(len(frames), 45)
                self.assertLessEqual(len(frames), 50)
                for frame in frames:
                    self.assertEqual(len(frame), FRAME)

    def test_no_samples_are_invented_or_lost_beyond_the_tail(self):
        rate = 48000
        signal = tone(1.0, rate)
        frames = run_capture(quanta_of(signal), rate, accumulate=True)
        produced = sum(len(f) for f in frames)
        expected = len(quanta_of(signal)) * QUANTUM / (rate / PIPELINE_RATE)
        # Everything except the last partial frame must come through.
        self.assertLessEqual(expected - produced, FRAME)

    def test_the_signal_survives_the_decimation(self):
        """A mirror that returns silence would satisfy every count above."""
        rate = 48000
        frames = run_capture(quanta_of(tone(1.0, rate)), rate, accumulate=True)
        peak = max(float(np.abs(f).max()) for f in frames)
        self.assertGreater(peak, 0.2, "decimation must not flatten the signal")


class TestShippedClientMatchesTheMirror(unittest.TestCase):
    """Weak but not nothing: the shipped file must carry the fixed shape."""

    def setUp(self):
        self.html = CLIENT.read_text(encoding="utf-8")

    def test_the_client_accumulates_across_calls(self):
        self.assertIn("this.pending = pending.slice(start)", self.html,
                      "the downsampled output must survive the call")
        self.assertIsNotNone(
            re.search(r"pending\.set\(held, 0\)", self.html),
            "the previous remainder must be prepended, not replaced",
        )

    def test_both_buffers_are_reset_together(self):
        # Leaving `pending` behind across turns would splice the end of one
        # utterance onto the start of the next.
        opener = self.html.split("setOpen(open)")[1][:400]
        self.assertIn("this.pending = new Float32Array(0)", opener)


if __name__ == "__main__":
    unittest.main()
