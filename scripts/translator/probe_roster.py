# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Roster management, driven at the DOM without a server: can the user delete
a speaker and merge two of them from the main page?

User order, 2026-08-03, translated from the German original: speakers must be
deletable directly on the main page, individually, and mergeable by dragging
one onto another. Both halves were reachable only through a 600 ms long press
into a settings sheet, or not at all.

This is the client half. It executes the SHIPPED ``client/index.html`` in
Chromium, paints a roster through the real ``applyState``, and performs the
gestures with real pointer events -- a synthesised ``click()`` would not touch
the long-press timer, the pointer capture or the drop-target hit testing,
which is where a touch gesture actually goes wrong. What it asserts is what
the server is asked to do: the control frames the page sends.

The server half is proven separately and hermetically in
``test/registered/translator/test_speaker_buttons.py`` (``TestSpeakerMerge``)
and ``test_audio_and_http.py``; this probe deliberately does not reimplement a
session.

    PYTHONPATH=<worktree>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_roster.py --shot /tmp/roster.png
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
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

#: The same stubs the auto-scroll probe uses: neither the socket nor the REST
#: calls are under test here, and a probe that needs the tenant is the gate.
STUB_JS = """
(() => {
  window.__sent = [];
  class DeadSocket {
    constructor() { this.readyState = 0; }
    send() {}
    close() { this.readyState = 3; }
    addEventListener() {}
  }
  window.WebSocket = DeadSocket;
  window.fetch = () => Promise.resolve(new Response("{}", {
    status: 200, headers: {"content-type": "application/json"},
  }));
})();
"""

#: Two speakers and the capability flag the server sends once it can carry a
#: merge out. Painted through `applyState`, so the probe exercises the real
#: path rather than calling `renderSpeakers` behind its back.
STATE_JS = """
(supports) => {
  connection.sendControl = (message) => { window.__sent.push(message); };
  applyState({
    session_id: "probe",
    speakers: [
      {speaker_id: "speaker-1", label: "Matthias", observations: 3,
       reference_seconds: 4.2, clonable: true},
      {speaker_id: "speaker-2", label: null, observations: 1,
       reference_seconds: 1.4, clonable: false},
    ],
    armed_speaker: null,
    supports: supports,
  });
  window.__sent.length = 0;
}
"""


def serve_dir(directory: Path) -> tuple:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    handler.log_message = lambda *a, **k: None  # type: ignore[assignment]
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


async def settle(page) -> None:
    await page.evaluate(
        "() => new Promise(r => requestAnimationFrame("
        "() => requestAnimationFrame(r)))"
    )


async def centre(page, selector: str) -> tuple:
    box = await page.locator(selector).bounding_box()
    if box is None:
        raise AssertionError(f"{selector} is not on the page")
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


async def confirm_modal(page, accept: bool) -> str:
    """Answer the confirmation sheet and return the question it asked."""
    await page.wait_for_selector("#modal.on", timeout=3000)
    text = await page.locator("#modal .box").inner_text()
    button = ".actions button:last-child" if accept else ".actions button:first-child"
    await page.locator("#modal " + button).click()
    await settle(page)
    return text.replace("\n", " ")


async def sent(page) -> list:
    return await page.evaluate("window.__sent")


async def arm_delete(page, failures: list, verbose: bool) -> None:
    """A speaker is removable from the MAIN PAGE, in one visible control."""
    await page.evaluate(STATE_JS, {"speaker_merge": True})
    await settle(page)
    count = await page.locator("#speakers .sp > button.x").count()
    if count != 2:
        failures.append(
            f"delete: {count} delete controls on a roster of 2 -- the user "
            f"asked for one per entry, on the main page"
        )
        return
    # The cancel path first: a destructive control that fires on a mis-tap is
    # worse than one nobody finds.
    await page.locator('#speakers .sp[data-speaker="speaker-1"] button.x').click()
    question = await confirm_modal(page, accept=False)
    after_cancel = await sent(page)
    if after_cancel:
        failures.append(f"delete: cancelling still sent {after_cancel}")
    if "transcript" not in question.lower():
        failures.append(
            f"delete: the confirmation does not say what happens to the "
            f"transcript: {question!r}"
        )
    await page.locator('#speakers .sp[data-speaker="speaker-1"] button.x').click()
    await confirm_modal(page, accept=True)
    frames = await sent(page)
    if verbose:
        print(f"[probe]   delete    : asked {question!r}")
        print(f"[probe]   delete    : sent {frames}")
    if frames != [{"kind": "speaker.delete", "speaker_id": "speaker-1"}]:
        failures.append(f"delete: wrong control frame {frames}")


async def arm_delete_is_not_the_talk_button(page, failures: list) -> None:
    """The x sits on top of the speak button; a tap on it must not record."""
    await page.evaluate(STATE_JS, {"speaker_merge": True})
    await settle(page)
    await page.locator('#speakers .sp[data-speaker="speaker-2"] button.x').click()
    await confirm_modal(page, accept=False)
    frames = await sent(page)
    if any(f.get("kind") == "speaker.arm" for f in frames):
        failures.append(
            f"delete: the press reached the speak button underneath: {frames}"
        )


async def arm_merge_drag(page, failures: list, verbose: bool,
                         shot_path: str | None) -> None:
    """Long-press one entry, drag it onto another, confirm: one speaker."""
    await page.evaluate(STATE_JS, {"speaker_merge": True})
    await settle(page)
    source = '#speakers .sp[data-speaker="speaker-2"] button.who'
    sx, sy = await centre(page, source)
    tx, ty = await centre(page, '#speakers .sp[data-speaker="speaker-1"]')

    await page.mouse.move(sx, sy)
    await page.mouse.down()
    # Past the 600 ms hold. A shorter press is a turn, not a drag, and that
    # boundary is the reason this probe uses pointer events at all.
    await page.wait_for_timeout(750)
    await page.mouse.move(sx + 8, sy)
    await settle(page)
    lifted = await page.evaluate(
        "() => ({dragging: !!document.querySelector('#speakers .sp.dragging'),"
        " zones: document.querySelectorAll('#speakers .sp.dropzone').length,"
        " hint: !!document.querySelector('#speakers .mergehint')})"
    )
    if verbose:
        print(f"[probe]   merge     : lifted {lifted}")
    if not lifted["dragging"]:
        failures.append("merge: the long press did not lift the entry")
    if lifted["zones"] < 1:
        failures.append("merge: no drop zone was shown")
    if not lifted["hint"]:
        failures.append("merge: nothing said what the drag is for")

    await page.mouse.move(tx, ty, steps=6)
    await settle(page)
    marked = await page.evaluate(
        "() => { const n = document.querySelector('#speakers .sp.droptarget');"
        " return n ? n.dataset.speaker : null; }"
    )
    if verbose:
        print(f"[probe]   merge     : drop target {marked}")
    if marked != "speaker-1":
        failures.append(
            f"merge: the entry under the finger is not highlighted ({marked})"
        )
    if shot_path:
        await page.screenshot(path=shot_path, full_page=False)
        print(f"[probe] screenshot {shot_path}")
    await page.mouse.up()
    question = await confirm_modal(page, accept=True)
    frames = await sent(page)
    if verbose:
        print(f"[probe]   merge     : asked {question!r}")
        print(f"[probe]   merge     : sent {frames}")
    if "undone" not in question.lower():
        failures.append(
            f"merge: a destructive action did not say it is irreversible: "
            f"{question!r}"
        )
    expected = {"kind": "speaker.merge", "target_id": "speaker-1",
                "source_id": "speaker-2"}
    if frames != [expected]:
        failures.append(f"merge: wrong control frame {frames}")
    left = await page.evaluate(
        "() => document.querySelectorAll('#speakers .sp.dragging,"
        " #speakers .sp.dropzone, #speakers .mergehint').length"
    )
    if left:
        failures.append(f"merge: {left} drag decoration(s) left on the row")


async def arm_merge_cancelled(page, failures: list) -> None:
    """Dropping on nobody opens the sheet instead -- where renaming lives."""
    await page.evaluate(STATE_JS, {"speaker_merge": True})
    await settle(page)
    sx, sy = await centre(page, '#speakers .sp[data-speaker="speaker-2"] button.who')
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    await page.wait_for_timeout(750)
    await page.mouse.move(sx, sy - 200, steps=6)
    await page.mouse.up()
    await settle(page)
    opened = await page.evaluate(
        "() => !!document.querySelector('#sheet.on')"
    )
    frames = await sent(page)
    if any(f.get("kind") == "speaker.merge" for f in frames):
        failures.append(f"merge: a drop on nobody still merged: {frames}")
    if not opened:
        failures.append("merge: releasing over nothing lost the speaker sheet")


async def arm_merge_needs_the_server(page, failures: list, verbose: bool) -> None:
    """THE CAN-FAIL ARM for the capability gate.

    The page deploys by being merged into the worktree; its server half needs
    a tenant restart. In that window the gesture must not be offered, or the
    user drags two speakers together and nothing happens. Same page, state
    frame without the flag: the long press falls back to the sheet.
    """
    await page.evaluate(STATE_JS, {})
    await settle(page)
    sx, sy = await centre(page, '#speakers .sp[data-speaker="speaker-2"] button.who')
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    await page.wait_for_timeout(750)
    await page.mouse.move(sx + 8, sy)
    await settle(page)
    lifted = await page.evaluate(
        "() => !!document.querySelector('#speakers .sp.dragging')"
    )
    opened = await page.evaluate(
        "() => !!document.querySelector('#sheet.on')"
    )
    await page.mouse.up()
    await settle(page)
    if verbose:
        print(f"[probe]   gate      : lifted {lifted}, sheet {opened}")
    if lifted:
        failures.append(
            "gate: the merge drag was offered by a server that cannot do it"
        )
    if not opened:
        failures.append("gate: the old long-press behaviour was lost")
    # And the delete control stays available: it needs no new server half.
    if await page.locator("#speakers .sp > button.x").count() != 2:
        failures.append("gate: delete disappeared with the merge capability")


async def main_async(args) -> int:
    from playwright.async_api import async_playwright

    client = Path(args.client).resolve()
    staging = Path(tempfile.mkdtemp(prefix="roster-"))
    shutil.copy(client, staging / "index.html")
    httpd, port = serve_dir(staging)
    failures: list = []
    console: list = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await (await browser.new_context(
                viewport=VIEWPORT, device_scale_factor=2
            )).new_page()
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            await page.add_init_script(STUB_JS)
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_selector("#speakers")
            print(f"[probe] client {client}")

            await arm_delete(page, failures, args.verbose)
            await arm_delete_is_not_the_talk_button(page, failures)
            await arm_merge_drag(page, failures, args.verbose, args.shot)
            await arm_merge_cancelled(page, failures)
            await arm_merge_needs_the_server(page, failures, args.verbose)

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
    print("[probe] PASS (delete, merge drag, cancelled drag, capability gate)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default=str(DEFAULT_CLIENT))
    parser.add_argument("--shot", default=None,
                        help="screenshot taken mid-drag, over the drop target")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    code = asyncio.run(main_async(args))
    if args.json:
        Path(args.json).write_text(json.dumps({"exit": code}))
    return code


if __name__ == "__main__":
    sys.exit(main())
