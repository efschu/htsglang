# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Auto-scroll, driven at the DOM without a server: does the newest line stay
on screen while a bubble GROWS?

The acceptance gate already carries a geometry arm (``NEWEST_LINE_VISIBLE_JS``
in ``client_gate.py``) and it went red on its first run -- newest line 95 px
below the fold once the transcript overflowed. That arm needs the whole
pipeline: a tenant, a GPU, ASR, MT and a talker, ~40 s per turn, and it samples
ONCE per turn. This probe isolates the same question so it can be answered in
seconds and sampled after every single mutation, which is what actually
localises the defect: the bubble is appended and followed correctly, then it
keeps growing (queued notice -> clause partials -> final text) and nothing
scrolls again.

It executes the SHIPPED ``client/index.html`` in Chromium and drives the real
wire handler ``onEvent`` with the real event sequence. No server, no audio, no
GPU. What it cannot see is anything upstream of ``onEvent`` -- that is the
gate's job and this does not replace it.

Can-fail proof (run it before believing a green):

    git -C <worktree> show 8011c6fb05:python/sglang/srt/translator/client/index.html \\
        > /tmp/client_before.html
    ... probe_autoscroll.py --client /tmp/client_before.html   # must be RED

    PYTHONPATH=<worktree>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_autoscroll.py
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

#: A phone, roughly. The defect only exists once the content overflows, so the
#: viewport has to be small enough for four bubbles to do that.
VIEWPORT = {"width": 390, "height": 720}

#: Long enough that one bubble is several lines tall on that viewport -- a
#: one-line bubble hides the defect, which is exactly what the first gate run
#: showed (green at 1 and 2 lines, red at 3).
SOURCE_TEXT = (
    "Guten Abend, wir haetten gern einen Tisch fuer vier Personen, "
    "moeglichst draussen, und die Karte auf Spanisch wenn das geht."
)
#: The translation, in the clauses the server streams it in. Spanish, because
#: this is the DE->ES pair the deadline is for. Two of its words look like
#: English typos to the spell checker; they are silenced per line rather than
#: added to the repo-wide dictionary, where they would mask real typos.
CLAUSES = [
    "Buenas tardes, querriamos una mesa para cuatro personas,",
    "a ser posible en la terraza,",  # codespell:ignore
    "y la carta en espanol si es posible.",  # codespell:ignore
]

#: Geometry, not scrollTop arithmetic -- the same question the gate asks, so a
#: green here means the same thing there. Kept in sync with the copy in
#: ``client_gate.py`` on purpose.
#:
#: The first version of this predicate demanded the WHOLE newest line fit
#: inside the box, and it stayed red against a client that was demonstrably
#: pinned to the bottom (``overflow_below -4px, at bottom True``). A bubble
#: carrying a long turn is TALLER than the box on a phone, and then "fully
#: inside" is unsatisfiable by any scroll position -- the arm was asserting
#: something no correct client can do. What the user asked for is that the
#: NEWEST text is on screen, i.e. the line's bottom edge is inside the view;
#: the top edge only has to be inside when the line is short enough for it.
NEWEST_LINE_VISIBLE_JS = """
() => {
  const box = document.getElementById('transcript');
  const lines = box ? box.querySelectorAll('.line') : [];
  if (!box || !lines.length) return null;
  const last = lines[lines.length - 1].getBoundingClientRect();
  const view = box.getBoundingClientRect();
  const pill = document.getElementById('unread');
  const taller = last.height > view.height;
  return {
    lines: lines.length,
    taller_than_view: taller,
    visible: last.bottom <= view.bottom + 1
             && (taller || last.top >= view.top - 1),
    overflow_below: Math.round(last.bottom - view.bottom),
    overflows: box.scrollHeight > box.clientHeight + 1,
    scrolled_to_bottom:
      box.scrollHeight - box.scrollTop - box.clientHeight < 60,
    unread_shown: !!(pill && !pill.hidden),
  };
}
"""

#: The page talks to a server the moment it loads. Neither call is part of what
#: is under test, and both are stubbed rather than served: a probe that needs
#: the tenant is the gate again.
STUB_JS = """
(() => {
  class DeadSocket {
    constructor() { this.readyState = 0; }
    send() {}
    close() { this.readyState = 3; }
    addEventListener() {}
  }
  window.WebSocket = DeadSocket;
  const empty = () => Promise.resolve(new Response("{}", {
    status: 200, headers: {"content-type": "application/json"},
  }));
  window.fetch = empty;
})();
"""


def serve_dir(directory: Path) -> tuple:
    """A throwaway static server; file:// changes enough rules to matter."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    handler.log_message = lambda *a, **k: None  # type: ignore[assignment]
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


async def settle(page) -> None:
    """Two frames: one for the mutation callbacks, one for the layout they
    caused. Deliberately short -- a probe that waits a second would let a
    client that never re-scrolls at all look healthy if anything else on the
    page happened to touch the scroll position."""
    await page.evaluate(
        "() => new Promise(r => requestAnimationFrame("
        "() => requestAnimationFrame(r)))"
    )


async def send(page, event: dict) -> None:
    await page.evaluate("(e) => onEvent(e)", event)


async def measure(page, label: str, failures: list, verbose: bool) -> dict:
    await settle(page)
    shot = await page.evaluate(NEWEST_LINE_VISIBLE_JS)
    if shot is None:
        return {}
    if verbose:
        print(f"[probe]   {label:<26} lines {shot['lines']}, visible "
              f"{shot['visible']}, overflow below {shot['overflow_below']}px, "
              f"overflows {shot['overflows']}, at bottom "
              f"{shot['scrolled_to_bottom']}"
              + (", taller than view" if shot["taller_than_view"] else ""))
    if not shot["visible"]:
        failures.append(
            f"{label}: newest line is {shot['overflow_below']}px below the "
            f"visible area ({shot['lines']} lines)"
        )
    return shot


async def run_turn(page, index: int, failures: list, verbose: bool) -> dict:
    """One turn, in the order the server actually emits it."""
    turn = f"t{index}"
    await send(page, {"kind": "turn.opened", "turn_id": turn})
    await send(page, {
        "kind": "turn.transcript", "turn_id": turn,
        "text": f"{index}. {SOURCE_TEXT}", "language": "de",
    })
    shot = await measure(page, f"turn {index} transcript", failures, verbose)
    # The queued notice: the FIRST growth after the append, and the one the
    # handover note named as the likely mechanism.
    await send(page, {"kind": "turn.queued", "turn_id": turn})
    shot = await measure(page, f"turn {index} queued", failures, verbose)
    grown = []
    for order, clause in enumerate(CLAUSES):
        grown.append(clause)
        await send(page, {
            "kind": "turn.translation", "turn_id": turn, "target": "es",
            "text": clause, "partial": True,
        })
        shot = await measure(
            page, f"turn {index} clause {order + 1}", failures, verbose
        )
    await send(page, {
        "kind": "turn.translation", "turn_id": turn, "target": "es",
        "text": " ".join(grown), "partial": False,
    })
    shot = await measure(page, f"turn {index} final", failures, verbose)
    await send(page, {"kind": "turn.done", "turn_id": turn})
    return shot


async def run_scrolled_up_arm(page, failures: list, verbose: bool,
                              shot_path: str | None = None) -> dict:
    """The exception that makes the rule usable: a reader who scrolled up on
    purpose must NOT be yanked back down -- and must be told that something
    arrived, or the app looks frozen to exactly the person who is reading."""
    await page.evaluate("() => { document.getElementById('transcript').scrollTop = 0; }")
    await settle(page)
    top = await page.evaluate(
        "() => document.getElementById('transcript').scrollTop"
    )
    await send(page, {"kind": "turn.opened", "turn_id": "up"})
    await send(page, {
        "kind": "turn.transcript", "turn_id": "up",
        "text": "Und noch ein Satz, waehrend jemand oben liest.",
        "language": "de",
    })
    await send(page, {
        "kind": "turn.translation", "turn_id": "up", "target": "es",
        "text": "Y otra frase mientras alguien lee mas arriba.",
        "partial": False,
    })
    await settle(page)
    after = await page.evaluate(
        "() => document.getElementById('transcript').scrollTop"
    )
    shot = await page.evaluate(NEWEST_LINE_VISIBLE_JS)
    if verbose:
        print(f"[probe]   scrolled-up: scrollTop {top} -> {after}, "
              f"unread pill {shot['unread_shown']}")
    if after != top:
        failures.append(
            f"scrolled-up: the view moved from {top} to {after} while the "
            f"reader was scrolled up"
        )
    if not shot["unread_shown"]:
        failures.append(
            "scrolled-up: nothing told the reader that a new line arrived"
        )
    if shot_path:
        await page.screenshot(path=shot_path, full_page=False)
        print(f"[probe] screenshot {shot_path}")
    # And the way back: tapping the pill returns to the live end and resumes.
    await page.evaluate("() => document.getElementById('unread').click()")
    await settle(page)
    back = await page.evaluate(NEWEST_LINE_VISIBLE_JS)
    if verbose:
        print(f"[probe]   after tap : visible {back['visible']}, "
              f"unread pill {back['unread_shown']}")
    if not back["visible"]:
        failures.append("scrolled-up: the pill did not return to the live end")
    if back["unread_shown"]:
        failures.append("scrolled-up: the pill stayed up after being tapped")
    return back


async def main_async(args) -> int:
    from playwright.async_api import async_playwright

    client = Path(args.client).resolve()
    staging = Path(tempfile.mkdtemp(prefix="autoscroll-"))
    shutil.copy(client, staging / "index.html")
    httpd, port = serve_dir(staging)
    failures: list = []
    console: list = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
            page = await (await browser.new_context(
                viewport=VIEWPORT, device_scale_factor=2
            )).new_page()
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            await page.add_init_script(STUB_JS)
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_selector("#transcript")

            print(f"[probe] client {client}")
            saw_overflow = False
            for index in range(1, args.turns + 1):
                shot = await run_turn(page, index, failures, args.verbose)
                saw_overflow = saw_overflow or bool(shot.get("overflows"))
            # An arm that never made the content overflow proved nothing: the
            # defect only exists past that point, and the first gate run was
            # green for exactly the two turns that fit.
            if not saw_overflow:
                failures.append(
                    "the transcript never overflowed its box -- this run "
                    "could not have seen the defect at all"
                )
            if args.shot:
                await page.screenshot(path=args.shot, full_page=False)
                print(f"[probe] screenshot {args.shot}")
            if args.scrolled_up_arm:
                unread_shot = None
                if args.shot:
                    stem = Path(args.shot)
                    unread_shot = str(stem.with_name(
                        stem.stem + "_unread" + stem.suffix))
                await run_scrolled_up_arm(
                    page, failures, args.verbose, unread_shot
                )
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
    print(f"[probe] PASS ({args.turns} turns"
          + (", scrolled-up arm" if args.scrolled_up_arm else "") + ")")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default=str(DEFAULT_CLIENT),
                        help="the index.html under test")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--shot", default=None, help="screenshot path")
    parser.add_argument("--no-scrolled-up-arm", dest="scrolled_up_arm",
                        action="store_false", default=True,
                        help="skip the reader-scrolled-up exception (the old "
                             "client has no pill and fails it by absence)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    code = asyncio.run(main_async(args))
    if args.json:
        Path(args.json).write_text(json.dumps({"exit": code}))
    return code


if __name__ == "__main__":
    sys.exit(main())
