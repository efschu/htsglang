# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Losing the microphone must hand the controls back, not freeze the UI.

THE LIVE FAILURE OF 2026-08-04. The user pressed record and nothing happened,
while the server showed his session `attached: 1` with a decision log that had
simply stopped. The mechanism: continuous mode DISABLES the speak button on
purpose (`talk.disabled = true`), and when Android takes the microphone away --
it MUTES the track for a call or another app, or ENDS it -- the page answered
with one line of text and changed no state. Every other indicator stayed
healthy, so the user was left with a dead microphone and a record button that
does nothing.

The arm drives the loss the way the platform delivers it: a real `mute` /
`ended` event dispatched on the actual `MediaStreamTrack` behind the fake
capture device, which is what invokes `track.onmute` / `track.onended`. What it
cannot do is make Chromium's fake device spontaneously lose itself; the event
is dispatched rather than provoked, and the handler under test is the same one
either way.

  precondition   continuous mode is on and the speak button IS disabled
                 (without this the recovery would be untestable -- there
                 would be nothing to recover)
  claim          after the loss: the button is enabled, the mode is back to
                 push-to-talk, the header says so, and the user is told
  control        `--sabotage-recovery` unhooks `onLost`, which is the pre-fix
                 client; the run must then FAIL

    PYTHONPATH=<worktree>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_mic_recovery.py
    ... --kind ended
    ... --sabotage-recovery     # must FAIL
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import shutil
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

DEFAULT_CLIENT = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/translator/client/index.html"
)
VIEWPORT = {"width": 390, "height": 720}

STUB_JS = """
(() => {
  class DeadSocket {
    constructor() { this.readyState = 0; }
    send() {} close() { this.readyState = 3; } addEventListener() {}
  }
  window.WebSocket = DeadSocket;
  window.fetch = () => Promise.resolve(new Response("{}", {
    status: 200, headers: {"content-type": "application/json"},
  }));
})();
"""

#: The pre-fix client: the capture still goes away, and nothing reacts.
SABOTAGE_JS = "() => { microphone.onLost = () => {}; }"

STATE_JS = """
() => ({
  disabled: talk.disabled,
  label: talk.querySelector(".lbl").textContent,
  mode: (() => {
    const on = document.querySelector("#segmode button.on");
    return on ? on.dataset.value : null;
  })(),
  state: document.body.dataset.state,
  status: document.getElementById("status").textContent,
  toasts: Array.from(document.querySelectorAll("#toasts .text"))
            .map((n) => n.textContent),
  open: microphone.open,
})
"""

#: Deliver the loss on the REAL track object, which is what the client listens
#: to. `mute` and `ended` are the two shapes Android uses.
LOSE_JS = """
(kind) => {
  const track = microphone.stream && microphone.stream.getAudioTracks()[0];
  if (!track) return {ok: false, why: "no track -- capture never started"};
  track.dispatchEvent(new Event(kind));
  return {ok: true, readyState: track.readyState};
}
"""


def serve_dir(directory: Path):
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    handler.log_message = lambda *a, **k: None  # type: ignore[assignment]
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


async def run(page, failures, verbose, kind, sabotage):
    # Continuous mode, which is the state the trap needs.
    await page.evaluate(
        '() => document.querySelector(\'#segmode button[data-value="vad"]\').click()'
    )
    for _ in range(60):
        if await page.evaluate("() => microphone.open === true"):
            break
        await page.wait_for_timeout(100)
    before = await page.evaluate(STATE_JS)
    if verbose:
        print(f"[probe]   listening   : {before}")
    if not before["open"]:
        return "INSTRUMENT"
    if not before["disabled"]:
        # Without the disabled button there is no trap to escape, and a PASS
        # afterwards would mean nothing.
        return "INSTRUMENT"

    if sabotage:
        await page.evaluate(SABOTAGE_JS)
        print("[probe] SABOTAGE: onLost unhooked (the pre-fix client)")

    lost = await page.evaluate(LOSE_JS, kind)
    if verbose:
        print(f"[probe]   lost ({kind}): {lost}")
    if not lost["ok"]:
        return "INSTRUMENT"
    await page.wait_for_timeout(400)

    after = await page.evaluate(STATE_JS)
    if verbose:
        print(f"[probe]   after       : {after}")
    if after["disabled"]:
        failures.append(
            "the speak button is STILL disabled after the microphone went "
            "away -- this is the live failure, pressing record does nothing"
        )
    if after["mode"] != "ptt":
        failures.append(
            f"the mode control still shows {after['mode']!r}; continuous mode "
            "cannot continue without a microphone"
        )
    if after["open"]:
        failures.append("the capture is still marked open after the loss")
    if after["state"] != "nomic":
        failures.append(
            f"the header state is {after['state']!r}, not 'nomic'; nothing "
            "tells the user why the app went quiet"
        )
    if not any("microphone" in t for t in after["toasts"]):
        failures.append(f"no message named the microphone ({after['toasts']})")
    return None


async def main_async(args) -> int:
    from playwright.async_api import async_playwright

    client = Path(args.client).resolve()
    staging = Path(tempfile.mkdtemp(prefix="mic-recovery-"))
    shutil.copy(client, staging / "index.html")
    httpd, port = serve_dir(staging)
    failures: list = []
    console: list = []
    verdict = None
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ])
            context = await browser.new_context(
                viewport=VIEWPORT, device_scale_factor=2,
                has_touch=True, is_mobile=True, permissions=["microphone"],
            )
            page = await context.new_page()
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            await page.add_init_script(STUB_JS)
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_selector("#speakers")
            print(f"[probe] client {client}")
            verdict = await run(page, failures, args.verbose, args.kind,
                                args.sabotage_recovery)
            if args.shot:
                await page.screenshot(path=args.shot)
                print(f"[probe] screenshot {args.shot}")
            hard = [c for c in console if c.startswith(("error", "pageerror"))]
            for line in hard:
                print(f"[probe] console {line}")
            if hard:
                failures.append(f"{len(hard)} console error(s)")
            await browser.close()
    finally:
        httpd.shutdown()
        shutil.rmtree(staging, ignore_errors=True)

    if verdict == "INSTRUMENT":
        print("[probe] INSTRUMENT FAILURE: the trap state was never reached "
              "(no capture, or the button was not disabled) -- nothing tested")
        return 2
    if failures:
        print("[probe] FAIL")
        for line in failures:
            print(f"[probe]   - {line}")
        return 1
    print(f"[probe] PASS (the controls come back after a {args.kind} track)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default=str(DEFAULT_CLIENT))
    parser.add_argument("--kind", choices=("mute", "ended"), default="mute")
    parser.add_argument("--sabotage-recovery", action="store_true",
                        help="unhook onLost; this run must FAIL")
    parser.add_argument("--shot", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
