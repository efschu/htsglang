# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The merge drag with a REAL FINGER, via CDP ``Input.dispatchTouchEvent``.

``probe_roster.py`` drives the same gesture with `page.mouse` and passes on a
client the user could not drag at all. That is not a flake, it is a blind
spot with a mechanism: a MOUSE pointer is never taken away from the page,
while a TOUCH pointer is. With `touch-action: manipulation` (which the global
button rule sets) the browser may claim a touch as a pan after a few pixels of
travel, fire `pointercancel`, and the 600 ms hold timer is cleared before it
ever runs -- so the drag cannot start on a phone and every mouse-driven arm
still says PASS.

This arm therefore dispatches genuine touch points and, crucially, MOVES the
finger slightly during the hold, which is what a thumb does and what triggers
the pan claim. `--sabotage-touch-action` restores `touch-action: manipulation`
on the roster buttons at runtime, reproducing the pre-fix client: that run
must go RED, or this arm is not testing what it claims.

    PYTHONPATH=<worktree>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_roster_touch.py
    ... --sabotage-touch-action     # must FAIL
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
#: A phone, and it must be a TOUCH context or the page gets mouse semantics.
VIEWPORT = {"width": 390, "height": 720}

STUB_JS = """
(() => {
  window.__sent = [];
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

STATE_JS = """
(supports) => {
  connection.sendControl = (m) => { window.__sent.push(m); };
  applyState({
    session_id: "probe",
    speakers: [
      {speaker_id: "speaker-1", label: "Matthias", observations: 3,
       reference_seconds: 4.2, clonable: true},
      {speaker_id: "speaker-2", label: null, observations: 1,
       reference_seconds: 1.4, clonable: false},
    ],
    armed_speaker: null, supports: supports,
  });
  window.__sent.length = 0;
}
"""

#: Puts the pre-fix CSS back, so the arm can be shown to discriminate.
SABOTAGE_JS = """
() => {
  const style = document.createElement("style");
  style.textContent =
    "#speakers .sp > button.who { touch-action: manipulation !important; }";
  document.head.appendChild(style);
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


async def centre(page, selector):
    box = await page.locator(selector).bounding_box()
    if box is None:
        raise AssertionError(f"{selector} not on the page")
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


async def touch(cdp, kind, x=None, y=None):
    points = [] if kind == "touchEnd" else [{"x": x, "y": y}]
    await cdp.send("Input.dispatchTouchEvent",
                   {"type": kind, "touchPoints": points})


async def run(page, cdp, failures, verbose):
    await page.evaluate(STATE_JS, {"speaker_merge": True})
    await settle(page)
    sx, sy = await centre(page, '#speakers .sp[data-speaker="speaker-2"] button.who')
    tx, ty = await centre(page, '#speakers .sp[data-speaker="speaker-1"]')

    await touch(cdp, "touchStart", sx, sy)
    # A thumb is not a pin. These few pixels during the hold are exactly what
    # makes the browser consider the gesture a pan -- and what the mouse arm
    # can never reproduce.
    for dx in (1, 2, 3):
        await page.wait_for_timeout(120)
        await touch(cdp, "touchMove", sx + dx, sy + dx)
    await page.wait_for_timeout(400)          # past the 600 ms hold
    await touch(cdp, "touchMove", sx + 6, sy)
    await settle(page)

    lifted = await page.evaluate(
        "() => ({dragging: !!document.querySelector('#speakers .sp.dragging'),"
        " hint: !!document.querySelector('#speakers .mergehint')})"
    )
    if verbose:
        print(f"[probe]   after hold : {lifted}")
    if not lifted["dragging"]:
        failures.append(
            "the long press did not lift the entry under a real touch -- the "
            "browser took the gesture (see dragLog for the cancel)"
        )

    await touch(cdp, "touchMove", tx, ty)
    await settle(page)
    marked = await page.evaluate(
        "() => { const n = document.querySelector('#speakers .sp.droptarget');"
        " return n ? n.dataset.speaker : null; }"
    )
    if verbose:
        print(f"[probe]   drop target: {marked}")
    if marked != "speaker-1":
        failures.append(f"no drop target highlighted under the finger ({marked})")

    await touch(cdp, "touchEnd")
    await settle(page)
    try:
        await page.wait_for_selector("#modal.on", timeout=2500)
        await page.locator("#modal .actions button:last-child").click()
        await settle(page)
    except Exception:
        failures.append("no merge confirmation appeared after the drop")
    sent = await page.evaluate("window.__sent")
    if verbose:
        print(f"[probe]   sent       : {sent}")
        print(f"[probe]   dragLog    : {await page.evaluate('dragLog')}")
    expected = {"kind": "speaker.merge", "target_id": "speaker-1",
                "source_id": "speaker-2"}
    if sent != [expected]:
        failures.append(f"wrong or missing control frame: {sent}")


async def main_async(args) -> int:
    from playwright.async_api import async_playwright

    client = Path(args.client).resolve()
    staging = Path(tempfile.mkdtemp(prefix="roster-touch-"))
    shutil.copy(client, staging / "index.html")
    httpd, port = serve_dir(staging)
    failures: list = []
    console: list = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            context = await browser.new_context(
                viewport=VIEWPORT, device_scale_factor=2,
                has_touch=True, is_mobile=True,
            )
            page = await context.new_page()
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            await page.add_init_script(STUB_JS)
            cdp = await context.new_cdp_session(page)
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_selector("#speakers")
            print(f"[probe] client {client}")
            if args.sabotage_touch_action:
                await page.evaluate(SABOTAGE_JS)
                print("[probe] SABOTAGE: touch-action restored to manipulation")
            await run(page, cdp, failures, args.verbose)
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

    if failures:
        print("[probe] FAIL")
        for line in failures:
            print(f"[probe]   - {line}")
        return 1
    print("[probe] PASS (merge drag under real touch events)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default=str(DEFAULT_CLIENT))
    parser.add_argument("--sabotage-touch-action", action="store_true",
                        help="restore the pre-fix CSS; this run must FAIL")
    parser.add_argument("--shot", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
