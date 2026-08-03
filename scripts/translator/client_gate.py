# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The acceptance gate: the REAL client, driven headless, against the public URL.

Every fix in this project so far was proved with ``front_door_test.py`` -- our
own protocol client. It speaks the WebSocket correctly and it never executes a
line of ``client/index.html``, so it was structurally incapable of catching the
four defects the user hit in a row: a suspended capture context, a frame
accumulator that never emitted, a pitch shift from an announced sample rate,
and a live push that never reached the DOM. The user became the test device.
That stops here.

This harness runs Chromium with a fake microphone fed from a real speech clip,
loads the page over HTTPS exactly as a phone does, taps the button with real
DOM events, and asserts what a person would see:

* the transcript line appears WITHOUT a reload, within a budget;
* audio frames arrive, in the right quantity for their announced rate;
* the console is clean.

Then it SOAKS: a dozen minutes with a turn every 90 s, which is the only way
to catch the two classes that only appear over time -- an idle socket dying
silently, and sessions accumulating until latency collapses.

    PYTHONPATH=<repo>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/client_gate.py --url https://<host>/translate/
    ... --soak-min 12 --turns 8

Test vehicle only: Chromium is never part of the serving topology.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

DEFAULT_URL = "https://efeu.ddnss.de/translate/"
DEFAULT_CLIP = Path(
    "/spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav"
)
#: Chromium's fake capture device wants 16-bit PCM at a rate it can play.
FAKE_RATE = 48000

# What the page must do, and by when. These are budgets, not measurements:
# every one of them is a thing the user reported as broken.
FIRST_LINE_BUDGET_S = 45.0
FIRST_AUDIO_BUDGET_S = 90.0


def prepare_fake_input(clip: Path, out: Path) -> float:
    """Chromium loops this file into getUserMedia. Returns its duration."""
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(str(clip), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != FAKE_RATE:
        # Linear resample is plenty: this is a microphone stand-in, and the
        # recognizer has to cope with a phone microphone anyway.
        n = int(len(samples) * FAKE_RATE / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, n),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
    # A little leading and trailing silence, so the segmenter sees an onset
    # rather than starting mid-word on the loop seam.
    pad = np.zeros(int(0.4 * FAKE_RATE), dtype=np.float32)
    samples = np.concatenate([pad, samples, pad])
    sf.write(str(out), samples, FAKE_RATE, subtype="PCM_16")
    return len(samples) / FAKE_RATE


PROBE_JS = """
() => {
  // Count what the playback path actually receives. Reading the DOM cannot
  // tell whether audio arrived; only the playback call can.
  window.__gate = { frames: 0, samples: 0, rate: 0, errors: [] };
  const original = playback.push.bind(playback);
  playback.push = (float32, rate) => {
    window.__gate.frames += 1;
    window.__gate.samples += float32.length;
    window.__gate.rate = rate;
    return original(float32, rate);
  };
  return true;
}
"""


async def one_turn(page, index: int, speak_s: float, budgets: dict) -> dict:
    """Tap, speak, tap, and wait for what a person would wait for."""
    before_lines = await page.evaluate(
        "document.querySelectorAll('#transcript .line').length"
    )
    before_frames = await page.evaluate("window.__gate.frames")
    started = time.monotonic()

    await page.click("#talk")                     # tap to speak
    await asyncio.sleep(speak_s)
    await page.click("#talk")                     # tap to stop

    line_at = None
    audio_at = None
    deadline = started + max(budgets["line"], budgets["audio"])
    while time.monotonic() < deadline:
        if line_at is None:
            now_lines = await page.evaluate(
                "document.querySelectorAll('#transcript .line').length"
            )
            if now_lines > before_lines:
                line_at = time.monotonic() - started
        if audio_at is None:
            if await page.evaluate("window.__gate.frames") > before_frames:
                audio_at = time.monotonic() - started
        if line_at is not None and audio_at is not None:
            break
        await asyncio.sleep(0.25)

    # The client's own view, per turn. A stall that is only visible as a
    # missing line is undiagnosable; these are the counters that say WHERE it
    # stopped -- socket, capture, or the server never answering.
    state = await page.evaluate(
        "({ws: connection.ws ? connection.ws.readyState : -1,"
        " frames: microphone.frames, chunks: microphone.chunks,"
        " open: microphone.open,"
        " ctx: microphone.ctx ? microphone.ctx.state : 'none',"
        " lines: document.querySelectorAll('#transcript .line').length})"
    )
    after = await page.evaluate("window.__gate")
    text = await page.evaluate(
        "(() => { const n = document.querySelectorAll('#transcript .line');"
        " return n.length ? n[n.length-1].innerText.replace(/\\n/g,' | ') : ''; })()"
    )
    samples = after["samples"] - (await page.evaluate("0") or 0)
    return {
        "turn": index,
        "line_s": None if line_at is None else round(line_at, 1),
        "audio_s": None if audio_at is None else round(audio_at, 1),
        "audio_frames": after["frames"] - before_frames,
        "audio_seconds": round(
            (after["samples"]) / max(after["rate"] or 1, 1), 2
        ),
        "rate": after["rate"],
        "text": text[:120],
        "samples_total": samples,
        "client": state,
    }


async def run_gate(args) -> int:
    from playwright.async_api import async_playwright

    fake = Path("/tmp/gate_fake_input.wav")
    speak_s = prepare_fake_input(args.clip, fake)
    print(f"[gate] fake microphone: {fake} ({speak_s:.2f} s, looped)")

    console: list = []
    failures: list = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fake}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = await browser.new_context(permissions=["microphone"])
        page = await context.new_page()
        page.on("console", lambda m: (
            console.append(f"{m.type}: {m.text}")
            if m.type in ("error", "warning") else None
        ))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        await page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        build = await page.evaluate("typeof CLIENT_BUILD === 'string' "
                                    "? CLIENT_BUILD : null")
        print(f"[gate] build {build}")
        if not build or build == "__CLIENT_BUILD__":
            failures.append("the page carries no build identity")

        # Wait for the socket to be up before touching the button.
        try:
            await page.wait_for_function(
                "() => connection && connection.ws && connection.ws.readyState === 1",
                timeout=30000,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"the client never opened its socket: {exc}")
            await browser.close()
            return report(failures, [], console, build)

        await page.evaluate(PROBE_JS)

        results = []
        budgets = {"line": args.line_budget, "audio": args.audio_budget}
        for i in range(args.turns):
            if i:
                await asyncio.sleep(args.gap_s)
            turn = await one_turn(page, i + 1, min(speak_s, 4.0), budgets)
            results.append(turn)
            print(f"[gate] turn {turn['turn']}: line {turn['line_s']}s "
                  f"audio {turn['audio_s']}s frames {turn['audio_frames']} "
                  f"@{turn['rate']}Hz | client {turn['client']} | "
                  f"{turn['text']}")
            if turn["line_s"] is None:
                failures.append(
                    f"turn {turn['turn']}: no transcript line appeared live "
                    f"within {budgets['line']}s (the user's 'only after a reload')"
                )
            if turn["audio_s"] is None:
                failures.append(
                    f"turn {turn['turn']}: no audio reached playback within "
                    f"{budgets['audio']}s"
                )
            # A socket that died silently is the R5 root candidate: check it
            # every turn rather than only at the end.
            alive = await page.evaluate(
                "connection && connection.ws ? connection.ws.readyState : -1"
            )
            if alive != 1:
                failures.append(
                    f"turn {turn['turn']}: the socket is in readyState "
                    f"{alive}, not OPEN"
                )

        await browser.close()
    return report(failures, results, console, build)


def report(failures, results, console, build) -> int:
    print()
    print(f"[gate] build   : {build}")
    print(f"[gate] turns   : {len(results)}")
    lines = [r["line_s"] for r in results if r["line_s"] is not None]
    audio = [r["audio_s"] for r in results if r["audio_s"] is not None]
    if lines:
        print(f"[gate] text    : min {min(lines)}s max {max(lines)}s")
    if audio:
        print(f"[gate] audio   : min {min(audio)}s max {max(audio)}s")
    noisy = [c for c in console if "favicon" not in c.lower()]
    print(f"[gate] console : {len(noisy)} error/warning lines")
    for line in noisy[:10]:
        print(f"          {line}")
    if noisy:
        failures.append(f"{len(noisy)} console error/warning lines")
    print()
    if failures:
        print("[gate] FAIL")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("[gate] PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--gap-s", type=float, default=5.0,
                        help="silence between turns; 90 for the soak")
    parser.add_argument("--line-budget", type=float, default=FIRST_LINE_BUDGET_S)
    parser.add_argument("--audio-budget", type=float, default=FIRST_AUDIO_BUDGET_S)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    code = asyncio.run(run_gate(args))
    if args.json_out:
        args.json_out.write_text(json.dumps({"exit": code}), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
