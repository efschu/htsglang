# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Where does the click at the start of speech come from? (user field report)

*"das rauschen/knacken gerade am anfang von gesprochenem ist immernoch
vorhanden"* -- an artefact at the START of synthesized playback, reported
more than once.

Three candidates, and the point of this probe is to READ them rather than
fade blind:

  (a) the talker's own waveform -- an onset that starts at a non-zero
      amplitude is a step from silence, which is a click no matter what the
      transport does;
  (b) the codec / decode seam;
  (c) the client's scheduling -- every 0.4 s slice becomes its own
      ``AudioBufferSourceNode`` started at an absolute cursor, with no fade
      anywhere, and `schedule()` re-anchors the cursor
      (`if (this.cursor < now + 0.02) this.cursor = now + 0.05`) whenever it
      has fallen behind the clock.

WHAT THIS PROBE CAN AND CANNOT SEE, stated because it decides who owns the
fix. The deployed client negotiates ``pcm16`` (index.html), and
``Pcm16Codec.encode`` only resamples -- so the float32 the client schedules
is a faithful copy of the talker's waveform apart from 24k->16k resampling.
That makes ONE capture answer (a) and (b) together: a clean onset here
exonerates the source and leaves (c); a hot onset here points upstream, at
which point the talker module belongs to the #488 profile mandate and this
script's job is to hand over numbers, not to edit it.

It deliberately does NOT load a second talker: the tenant holds that lock,
and a second copy would need VRAM that is not there.

    /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_audio_onset.py --json-out /tmp/onset.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client_gate import (  # noqa: E402  - sibling harness, reused deliberately
    DEFAULT_CLIP,
    DEFAULT_URL,
    prepare_fake_input,
)

#: Samples kept per captured buffer: 200 ms at 16 kHz is far more than an
#: onset needs and small enough to hand back through the CDP bridge.
KEEP_SAMPLES = 3200

#: Patch ``playback.schedule`` so every call records BOTH the samples it was
#: given and the clock it was given them against. The second half is what
#: answers the re-anchor question: a first buffer whose cursor already sits
#: behind ``currentTime`` is one the 0.05 s nudge has silently moved, and the
#: gap it opens is exactly where a listener hears a click.
CAPTURE_JS = """
(keep) => {
  window.__onset = {buffers: [], sched: []};
  const original = playback.schedule.bind(playback);
  playback.schedule = function (float32, rate) {
    const ctx = this.ctx;
    window.__onset.sched.push({
      at: performance.now(),
      rate: rate,
      length: float32.length,
      cursor: this.cursor,
      now: ctx ? ctx.currentTime : null,
      state: ctx ? ctx.state : 'none',
      // The condition the client itself applies, evaluated here so the
      // report says whether it FIRED rather than whether it could have.
      reanchored: ctx ? (this.cursor < ctx.currentTime + 0.02) : null,
    });
    if (window.__onset.buffers.length < 3) {
      window.__onset.buffers.push(
        Array.from(float32.slice(0, keep)).map((v) => Math.round(v * 32768))
      );
    }
    return original(float32, rate);
  };
  return true;
}
"""


def describe_onset(samples: np.ndarray, rate: int) -> dict:
    """The numbers that separate a step from a fade-in."""
    if samples.size == 0:
        return {"empty": True}

    def window_peak(ms: float) -> float:
        n = max(1, int(rate * ms / 1000.0))
        return float(np.max(np.abs(samples[:n])))

    peak = float(np.max(np.abs(samples))) or 1.0
    # The step a listener hears: the jump from the silence BEFORE the buffer
    # (which is what the output clock holds) into sample 0, and the largest
    # sample-to-sample jump inside the onset itself.
    first = float(samples[0])
    deltas = np.abs(np.diff(samples[: max(2, int(rate * 0.02))]))
    return {
        "first_sample": first,
        "first_sample_rel_peak": abs(first) / peak,
        "peak_full_buffer": peak,
        "peak_1ms": window_peak(1.0),
        "peak_5ms": window_peak(5.0),
        "peak_10ms": window_peak(10.0),
        "peak_20ms": window_peak(20.0),
        "max_step_in_first_20ms": float(np.max(deltas)) if deltas.size else 0.0,
        "dc_offset_first_20ms": float(
            np.mean(samples[: max(1, int(rate * 0.02))])
        ),
        # Samples until the envelope first exceeds 1% of the buffer peak. A
        # natural speech onset takes milliseconds; a step takes zero.
        "samples_to_1pct_of_peak": int(
            np.argmax(np.abs(samples) > 0.01 * peak)
        ),
    }


async def run(args) -> int:
    from playwright.async_api import async_playwright

    fake = Path("/tmp/onset_fake_input.wav")
    speak_s = prepare_fake_input(args.clip, fake)
    print(f"[onset] fake microphone: {fake} ({speak_s:.2f} s)")

    report: dict = {"url": args.url}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fake}",
            ],
        )
        context = await browser.new_context(permissions=["microphone"])
        page = await context.new_page()
        await page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        report["build"] = await page.evaluate(
            "typeof CLIENT_BUILD === 'string' ? CLIENT_BUILD : null"
        )
        print(f"[onset] build {report['build']}")
        await page.wait_for_function(
            "() => connection && connection.ws && connection.ws.readyState === 1",
            timeout=30000,
        )
        await page.evaluate(CAPTURE_JS, KEEP_SAMPLES)

        await page.click("#talk")
        await asyncio.sleep(min(speak_s, 4.0))
        await page.click("#talk")

        deadline = args.budget_s
        waited = 0.0
        while waited < deadline:
            if await page.evaluate("window.__onset.buffers.length") > 0:
                # Let a couple more land, so buffer boundaries can be read too.
                await asyncio.sleep(2.0)
                break
            await asyncio.sleep(0.25)
            waited += 0.25
        captured = await page.evaluate("window.__onset")
        await browser.close()

    report["scheduled"] = captured["sched"][:6]
    report["buffers"] = []
    for index, raw in enumerate(captured["buffers"]):
        samples = np.asarray(raw, dtype=np.float32) / 32768.0
        rate = captured["sched"][index]["rate"] if index < len(
            captured["sched"]) else 16000
        described = describe_onset(samples, int(rate or 16000))
        described["index"] = index
        described["rate"] = rate
        report["buffers"].append(described)

    if not report["buffers"]:
        print("[onset] NO AUDIO CAPTURED -- nothing to read")
        return 1

    print(f"[onset] captured {len(report['buffers'])} buffers")
    for entry in report["buffers"]:
        print(
            f"[onset] buffer {entry['index']} @{entry['rate']}Hz: "
            f"first {entry['first_sample']:+.5f} "
            f"({entry['first_sample_rel_peak'] * 100:.1f}% of peak) "
            f"peak1ms {entry['peak_1ms']:.4f} peak5ms {entry['peak_5ms']:.4f} "
            f"peak20ms {entry['peak_20ms']:.4f} "
            f"maxstep20ms {entry['max_step_in_first_20ms']:.5f} "
            f"dc {entry['dc_offset_first_20ms']:+.5f} "
            f"rise {entry['samples_to_1pct_of_peak']} samples"
        )
    for entry in report["scheduled"][:4]:
        print(
            f"[onset] schedule: cursor {entry['cursor']:.3f} "
            f"now {entry['now']:.3f} state {entry['state']} "
            f"reanchored {entry['reanchored']} len {entry['length']}"
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[onset] wrote {args.json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--budget-s", type=float, default=90.0)
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
