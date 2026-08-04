# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The hidden-page audio gate, with a DRIVEN visibility change.

`69779293e3` shipped the gate and said, correctly, that it had no arm: "a
headless context reports `document.hidden` false unless it is driven". This is
that arm, and the driving is the hard part -- a page that is never actually
backgrounded reports `hidden === false` forever, and every assertion built on
top of it passes for the wrong reason.

WHAT THIS ARM PROVES, AND WHAT IT DOES NOT -- read this before trusting a PASS.
Four ways to make a real browser background the page were tried on this host
and all four failed: `Emulation.setPageVisibilityOverride` does not exist in
Chromium 151, `Page.setWebLifecycleState("frozen")` leaves
`visibilityState: "visible"`, activating a second tab does nothing in headless,
and `Browser.setWindowBounds({windowState: "minimized"})` likewise. Headed
Chromium under Xvfb would have the real semantics and cannot run here -- Xvfb
crashes in `libEGL_nvidia` on this box. Measured, not assumed; the transcript
of those four attempts is in the commit.

So the visibility is driven at the DOCUMENT contract instead: `document.hidden`
and `visibilityState` are overridden and a real `visibilitychange` event is
dispatched. That exercises every line this cut owns -- the `schedule()`
predicate, the flush in the listener, the held counter, the return notice --
with the real event. It does NOT prove that Android fires `visibilitychange`
when a PWA is backgrounded while holding a wake lock. That is platform
behaviour, it is what the whole feature is premised on, and the only thing that
can confirm it is the field: `held_while_hidden` arriving in a debug upload
from the phone. Until that number is seen, the premise is reasoned, not
measured, and this file is not allowed to imply otherwise.

That the override took effect is asserted before anything else is measured --
an instrument has to be shown to discriminate before its verdict counts, and
an undriven gate would otherwise report a clean PASS while testing nothing.
That failure is its own exit state here (INSTRUMENT), never a pass and never a
silent skip.

Three claims, each with the control that makes it evidence:

  1. hidden REFUSES.   Audio pushed while hidden is not scheduled and is
                       counted as held.
  2. hidden FLUSHES.   Audio already booked is cut when the page goes hidden.
  3. visible PLAYS.    The same push on the same page schedules normally, so
                       claim 1 is the gate and not a broken output path.

`--sabotage-gate` dispatches the SAME event while leaving `document.hidden`
false -- which is precisely the pre-fix client, whose only guard was
`ctx.state !== "running"` and whose context keeps running in the background.
Claims 1 and 2 must then FAIL, or this arm is agreeing with whatever it is
handed.

    PYTHONPATH=<worktree>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_hidden_gate.py
    ... --sabotage-gate     # must FAIL
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

#: No socket and no server: this arm is about the output path only.
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

#: Drive the document's visibility contract and fire the real event. With
#: `honest` false this is the SABOTAGE: the event still arrives, but the page
#: cannot see that it is hidden -- the pre-fix client exactly, since its only
#: guard was the context state and a backgrounded context keeps running.
DRIVE_JS = """
([hidden, honest]) => {
  if (honest) {
    Object.defineProperty(document, "hidden", {
      configurable: true, get: () => hidden,
    });
    Object.defineProperty(document, "visibilityState", {
      configurable: true, get: () => (hidden ? "hidden" : "visible"),
    });
  }
  document.dispatchEvent(new Event("visibilitychange"));
  return {hidden: document.hidden, state: document.visibilityState};
}
"""

#: A second of speech-like signal. Content is irrelevant -- what is measured is
#: whether it reaches the output clock at all.
PUSH_JS = """
(seconds) => {
  const rate = 16000;
  const n = Math.round(rate * seconds);
  const buf = new Float32Array(n);
  for (let i = 0; i < n; i++) buf[i] = Math.sin(i * 0.05) * 0.25;
  const before = {
    scheduled: playback.scheduled,
    held: playback.heldWhileHidden,
    live: (playback.live || []).length,
  };
  playback.push(buf, rate);
  return {
    before: before,
    after: {
      scheduled: playback.scheduled,
      held: playback.heldWhileHidden,
      live: (playback.live || []).length,
      blocked: playback.blocked,
    },
  };
}
"""

WATCH_JS = """
() => {
  window.__vis = [];
  document.addEventListener("visibilitychange", () => {
    window.__vis.push(document.visibilityState);
  });
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


async def settle(page):
    await page.evaluate(
        "() => new Promise(r => requestAnimationFrame("
        "() => requestAnimationFrame(r)))"
    )


async def unlock(page):
    """Start the output clock, as the first tap does on a phone."""
    await page.evaluate("() => { playback.unlock(); playback.ensure(); }")
    for _ in range(40):
        state = await page.evaluate("() => playback.ctx && playback.ctx.state")
        if state == "running":
            return True
        await page.wait_for_timeout(50)
    return False


async def run(page, failures, verbose, sabotage):
    if not await unlock(page):
        failures.append(
            "the output context never reached 'running'; nothing below could "
            "have been measured"
        )
        return
    await page.evaluate(WATCH_JS)

    # 3 first, while the page is indisputably visible: it is the control that
    # makes a refusal downstream mean the gate rather than a dead output.
    visible = await page.evaluate(PUSH_JS, 0.4)
    if verbose:
        print(f"[probe]   visible push : {visible}")
    if visible["after"]["scheduled"] != visible["before"]["scheduled"] + 1:
        failures.append(
            f"a VISIBLE page did not schedule the audio ({visible}); the "
            "refusals below would prove nothing"
        )
    booked = visible["after"]["live"]

    # Drive the visibility.
    seen = await page.evaluate(DRIVE_JS, [True, not sabotage])
    await page.wait_for_timeout(200)
    events = await page.evaluate("() => window.__vis")
    if verbose:
        print(f"[probe]   backgrounded: {seen} events {events}")
    if not events:
        # The event did not even reach a listener: nothing below is a
        # statement about the client.
        return "INSTRUMENT"
    if not sabotage and seen["state"] != "hidden":
        return "INSTRUMENT"

    # 2. The flush. What was already booked must be gone.
    after_hide = await page.evaluate(
        "() => ({live: (playback.live || []).length, "
        "held: playback.heldWhileHidden, blocked: playback.blocked})"
    )
    if verbose:
        print(f"[probe]   after hide  : {after_hide} (was {booked} booked)")
    if booked and after_hide["live"] != 0:
        failures.append(
            f"going hidden left {after_hide['live']} source(s) on the output "
            f"clock; {booked} were booked before"
        )

    # 1. The refusal.
    hidden_push = await page.evaluate(PUSH_JS, 0.4)
    if verbose:
        print(f"[probe]   hidden push : {hidden_push}")
    if hidden_push["after"]["scheduled"] != hidden_push["before"]["scheduled"]:
        failures.append(
            f"a HIDDEN page scheduled audio anyway ({hidden_push})"
        )
    if hidden_push["after"]["held"] != hidden_push["before"]["held"] + 1:
        failures.append(
            f"the refusal was not counted as held ({hidden_push}); the "
            "return-to-foreground notice is built from that counter"
        )
    # The regression this cut fixes: withholding on purpose must not raise the
    # autoplay alarm, which only a tap on the unblock banner can clear.
    if hidden_push["after"]["blocked"]:
        failures.append(
            "the hidden gate set `blocked`, which is the autoplay-refusal "
            "flag: the page raises the red 'sound blocked' banner and holds "
            "it until the user taps unblock"
        )

    # Back to the foreground: the page must say what it withheld, once.
    await page.evaluate(DRIVE_JS, [False, not sabotage])
    await page.wait_for_timeout(200)
    await settle(page)
    notice = await page.evaluate(
        "() => Array.from(document.querySelectorAll('#toasts .text'))"
        ".map(n => n.textContent)"
    )
    if verbose:
        print(f"[probe]   on return   : {notice}")
    if not any("screen was off" in text for text in notice):
        failures.append(
            f"returning to the page said nothing about the withheld audio "
            f"({notice})"
        )
    return None


async def main_async(args) -> int:
    from playwright.async_api import async_playwright

    client = Path(args.client).resolve()
    staging = Path(tempfile.mkdtemp(prefix="hidden-gate-"))
    shutil.copy(client, staging / "index.html")
    httpd, port = serve_dir(staging)
    failures: list = []
    console: list = []
    verdict = None
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"],
            )
            context = await browser.new_context(
                viewport=VIEWPORT, device_scale_factor=2,
                has_touch=True, is_mobile=True,
            )
            page = await context.new_page()
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            await page.add_init_script(STUB_JS)
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_selector("#speakers")
            print(f"[probe] client {client}")
            if args.sabotage_gate:
                print("[probe] SABOTAGE: the event fires, the page cannot see hidden")
            verdict = await run(page, failures, args.verbose, args.sabotage_gate)
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
        # Not a pass and not a failure of the client: the arm could not do the
        # one thing it exists to do. Reported as its own state so it can never
        # be read as a green run.
        print("[probe] INSTRUMENT FAILURE: the visibility change was not "
              "driven -- the page never went hidden, so nothing was tested")
        return 2
    if failures:
        print("[probe] FAIL")
        for line in failures:
            print(f"[probe]   - {line}")
        return 1
    print("[probe] PASS (hidden refuses and flushes, visible plays)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default=str(DEFAULT_CLIENT))
    parser.add_argument("--sabotage-gate", action="store_true",
                        help="hide `document.hidden` from the page, "
                             "reproducing the pre-fix client; must FAIL")
    parser.add_argument("--shot", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
