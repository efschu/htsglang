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

#: Buffers kept, WHOLE. The wire frame is 20 ms (320 samples at 16 kHz), not
#: the talker's 0.4 s chunk, so keeping "the first few buffers" keeps only the
#: leading silence -- the first attempt at this probe measured 60 ms of
#: nothing and proved nothing. 200 buffers is 4 s, which reaches well past the
#: onset of speech and lets the boundaries be examined as a population.
KEEP_BUFFERS = 200

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
    if (window.__onset.buffers.length < keep) {
      window.__onset.buffers.push(
        Array.from(float32).map((v) => Math.round(v * 32768))
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
        await page.evaluate(CAPTURE_JS, KEEP_BUFFERS)

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

    # THE DISCRIMINATOR. Concatenate the buffers in the order they were
    # scheduled and compare the sample-to-sample step ACROSS a buffer seam
    # against the steps INSIDE the buffers. The client starts one
    # AudioBufferSourceNode per 20 ms frame, so if the seams are where the
    # energy jumps, the click is framing/scheduling; if seams look like any
    # other sample pair, the stream is continuous and the artefact is not a
    # discontinuity at all.
    buffers = [np.asarray(b, dtype=np.float32) / 32768.0
               for b in captured["buffers"]]
    if buffers:
        stream = np.concatenate(buffers)
        steps = np.abs(np.diff(stream))
        seam_positions = np.cumsum([len(b) for b in buffers])[:-1] - 1
        seam_positions = seam_positions[seam_positions < len(steps)]
        seam_steps = steps[seam_positions]
        inner = np.delete(steps, seam_positions)
        peak = float(np.max(np.abs(stream))) or 1.0
        speech_at = int(np.argmax(np.abs(stream) > 0.02 * peak))
        report["stream"] = {
            "buffers": len(buffers),
            "samples": int(stream.size),
            "peak": peak,
            "speech_starts_at_sample": speech_at,
            "speech_starts_in_buffer": speech_at // len(buffers[0]),
            "seam_step_max": float(np.max(seam_steps)) if seam_steps.size else 0.0,
            "seam_step_mean": float(np.mean(seam_steps)) if seam_steps.size else 0.0,
            "inner_step_max": float(np.max(inner)) if inner.size else 0.0,
            "inner_step_p99": float(np.percentile(inner, 99)) if inner.size else 0.0,
        }
        st = report["stream"]
        print(f"[onset] stream: {st['buffers']} buffers, {st['samples']} samples, "
              f"peak {st['peak']:.4f}")
        print(f"[onset] speech starts at sample {st['speech_starts_at_sample']} "
              f"(buffer {st['speech_starts_in_buffer']})")
        print(f"[onset] SEAM steps : max {st['seam_step_max']:.5f} "
              f"mean {st['seam_step_mean']:.5f}")
        print(f"[onset] INNER steps: max {st['inner_step_max']:.5f} "
              f"p99 {st['inner_step_p99']:.5f}")
        verdict = ("SEAMS ARE THE OUTLIER -- framing/scheduling"
                   if st["seam_step_max"] > st["inner_step_max"]
                   else "seams look like ordinary sample pairs -- the stream "
                        "is continuous, the click is NOT a seam discontinuity")
        print(f"[onset] verdict: {verdict}")
        # The onset of SPEECH, which is what the user hears -- not the onset
        # of the buffer, which is leading silence.
        # Windowed from BEFORE the crossing. Slicing at the crossing and then
        # asking how long the rise took answers a question about the slice,
        # not the signal -- the first attempt did exactly that and reported a
        # rise of 0 samples, which was true by construction and meaningless.
        pre = 320
        start = max(0, speech_at - pre)
        window = stream[start:speech_at + 1600]
        report["speech_onset"] = describe_onset(window, 16000)
        env = np.abs(window)
        local_peak = float(np.max(env)) or 1.0
        crossings = {
            f"samples_to_{int(f * 100)}pct": int(np.argmax(env > f * local_peak))
            for f in (0.01, 0.10, 0.50)
        }
        report["speech_onset"]["rise_from_silence"] = crossings
        report["speech_onset"]["silence_before"] = float(np.max(env[:pre]))
        print(f"[onset] onset window ({pre} samples of lead-in): "
              f"silence before {report['speech_onset']['silence_before']:.5f}, "
              f"reaches 1%/10%/50% of local peak at samples "
              f"{crossings['samples_to_1pct']}/{crossings['samples_to_10pct']}/"
              f"{crossings['samples_to_50pct']} "
              f"(lead-in ends at {pre})")

    # A buffer whose cursor has already passed `currentTime` is started in
    # the PAST; Web Audio clamps that to "now", so it overlaps its
    # predecessor instead of following it. That is a crackle mechanism that
    # leaves the samples themselves perfectly continuous, which is why the
    # seam analysis above cannot see it.
    sched = captured["sched"]
    late = [e for e in sched
            if e["now"] is not None and e["cursor"] <= e["now"]]
    report["scheduled"] = sched
    report["scheduling"] = {
        "calls": len(sched),
        "reanchored": sum(1 for e in sched if e.get("reanchored")),
        "started_in_the_past": len(late),
        "min_lead_s": min(
            (e["cursor"] - e["now"] for e in sched if e["now"] is not None),
            default=None),
    }
    sc = report["scheduling"]
    print(f"[onset] scheduling: {sc['calls']} calls, "
          f"{sc['reanchored']} re-anchored, "
          f"{sc['started_in_the_past']} started in the past, "
          f"min lead {sc['min_lead_s']:.3f}s")
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
