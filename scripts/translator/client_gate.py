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
* the OUTPUT context is running and its clock advances -- because "frames
  reached the playback path" was the assertion this harness made while a real
  phone sat there in silence;
* the console is clean.

**Known blind spot, tested rather than assumed.** Headless Chromium has no
audio device and starts its ``AudioContext`` whether or not a user gesture
preceded it. ``--prove-can-fail`` disables the page's own unlock, and the run
still passes -- so this gate CANNOT reproduce a mobile autoplay block, and a
green run here is not evidence that a phone makes a sound. The device's own
``playback`` telemetry is the instrument for that class (DESIGN §17.8.3).

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
#: How long to wait for the ``turn.done`` summary after the audio arrived.
TIMINGS_WAIT_S = 25.0


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
  window.__gate = { frames: 0, samples: 0, rate: 0, errors: [], timings: [] };
  const original = playback.push.bind(playback);
  playback.push = (float32, rate) => {
    window.__gate.frames += 1;
    window.__gate.samples += float32.length;
    window.__gate.rate = rate;
    return original(float32, rate);
  };
  // The server already measures every stage (``Stopwatch``) and ships the
  // numbers on ``turn.done``. Read them off a SECOND listener on the same
  // socket rather than by patching the client's dispatch: the page under test
  // must behave exactly as it does for a phone, and an extra listener changes
  // nothing about how the first one runs. This is what turns the gate from a
  // pass/fail into the per-stage instrument DESIGN §18.4 asks for.
  window.__gateAttach = () => {
    const ws = (typeof connection !== 'undefined' && connection.ws)
      ? connection.ws : null;
    if (!ws || ws.__gateHooked) return false;
    ws.__gateHooked = true;
    ws.addEventListener('message', (ev) => {
      if (typeof ev.data !== 'string') return;   // binary = audio, not events
      let event;
      try { event = JSON.parse(ev.data); } catch (err) { return; }
      if (event && event.kind === 'turn.done' && event.timings) {
        window.__gate.timings.push(event.timings);
      }
    });
    return true;
  };
  return window.__gateAttach();
}
"""

#: Stage keys carried by ``turn.done``, in pipeline order, with the two that
#: are cumulative-from-segment-close marked. ``first_audio_ms`` is THE number:
#: what a listener waits after the speaker stops.
STAGE_KEYS = (
    "asr_ms",
    "embed_ms",
    "mt_first_token_ms",
    "mt_total_ms",
    "tts_first_audio_ms",
    "tts_wait_ms",
    "tts_total_ms",
    "first_audio_ms",
    "total_ms",
)


#: Re-creates the defect the output assertions exist to catch: no unlock on
#: the gesture, and a context that is never resumed. Injected only under
#: ``--prove-can-fail``. A gate assertion nobody has ever seen fail is a
#: decoration, and this one replaced an assertion that WAS a decoration.
SABOTAGE_JS = """
() => {
  playback.unlock = function () { return this.ctx; };
  playback.ensure = function () {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return this.ctx;            // deliberately never resumed
  };
  return true;
}
"""


async def one_turn(page, index: int, speak_s: float, budgets: dict) -> dict:
    """Tap, speak, tap, and wait for what a person would wait for."""
    # Re-attach after a reconnect: ``connection.ws`` is replaced, and the
    # listener goes with the old socket. Idempotent via a marker on the socket.
    await page.evaluate("() => window.__gateAttach && window.__gateAttach()")
    before_lines = await page.evaluate(
        "document.querySelectorAll('#transcript .line').length"
    )
    before_frames = await page.evaluate("window.__gate.frames")
    before_timings = await page.evaluate("window.__gate.timings.length")
    started = time.monotonic()

    await page.click("#talk")                     # tap to speak
    await asyncio.sleep(speak_s)
    await page.click("#talk")                     # tap to stop

    line_at = None
    audio_at = None
    # Did the page ever SAY it was waiting (§17.8.2)? Polled rather than
    # checked at the end, because the notice is deliberately transient: the
    # translation that clears it is the same event that ends the wait.
    saw_waiting = False
    deadline = started + max(budgets["line"], budgets["audio"])
    while time.monotonic() < deadline:
        if not saw_waiting:
            saw_waiting = await page.evaluate(
                "document.querySelectorAll('.turn .dst.waiting').length > 0"
            )
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

    # ``turn.done`` lands AFTER the last audio frame, so the loop above has
    # already broken out by the time the stage timings exist. Wait for it
    # separately and bounded: the numbers are diagnostic, and a turn that
    # produced line and audio must not be failed for a late summary frame.
    stage_deadline = time.monotonic() + budgets["timings"]
    while time.monotonic() < stage_deadline:
        if await page.evaluate("window.__gate.timings.length") > before_timings:
            break
        await asyncio.sleep(0.25)
    timings = await page.evaluate(
        f"window.__gate.timings.slice({before_timings})"
    )
    # What the OUTPUT did, not what was handed to it. Counting playback.push
    # calls proves the frames arrived and says nothing about whether a sound
    # was made -- the distinction a real device paid for.
    out = await page.evaluate("playback.diagnostics()")

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
        "saw_waiting": saw_waiting,
        "out": out,
        # One entry per target language; a DE turn fans out to one target
        # today, but the list keeps a fan-out honest rather than averaging it.
        "stages": timings,
    }


async def run_gate(args) -> tuple:
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
                # NO autoplay override. It used to be here, and it is exactly
                # what made this harness unable to see the defect a real phone
                # hit: with the policy waived, an output context that a phone
                # would refuse to start starts anyway, every frame is
                # scheduled, and the gate reports audio while the device is
                # silent. Playwright's clicks are trusted events, so the page
                # gets real user activation here just as it does on a phone --
                # which is the only way this gate can speak for one.
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
            return report(failures, [], console, build), []

        await page.evaluate(PROBE_JS)
        if args.prove_can_fail:
            await page.evaluate(SABOTAGE_JS)
            print("[gate] SABOTAGE: the output unlock is disabled; a PASS "
                  "here would mean the output assertions cannot see it")

        results = []
        budgets = {
            "line": args.line_budget,
            "audio": args.audio_budget,
            "timings": args.timings_wait_s,
        }
        for i in range(args.turns):
            if i:
                await asyncio.sleep(args.gap_s)
            if args.reload_after and i == args.reload_after:
                # THE REPLAY-THEN-LIVE ARM. Every earlier run loaded the page
                # once and spoke into it, so the gate only ever exercised a
                # client with no history -- and a defect that needs a RELOAD to
                # appear was structurally invisible to it. That is exactly the
                # shape that reached the user: after a reload his text was
                # there, speaking again updated nothing and made no sound, and
                # reloading again showed the translation. The cursor his page
                # carried across the reload was the whole fault.
                #
                # The reload is a real navigation to the same URL, so
                # sessionStorage survives it just as it does on the phone.
                # Anything the client persists and reuses is therefore under
                # test here, which is the point.
                # The collector is emulated rather than waited for. A plain
                # reload alone does NOT reproduce the defect: the session is
                # still alive, so the stored cursor is still valid and live
                # delivery keeps working. What the user hit is the reload of a
                # page whose session had already been COLLECTED -- the server
                # then mints a fresh session under the same id with a journal
                # starting at zero, while the page still carries the old
                # conversation's high-water cursor. Dropping the session here
                # is that state exactly, and it takes a second instead of the
                # idle timeout.
                dropped = await page.evaluate(
                    """async () => {
                      const id = connection.sessionId;
                      if (!id) return null;
                      // Stop the client from racing us. Deleting the session
                      // closes the socket, the page reconnects on a backoff,
                      // and that reconnect RE-CREATES the session and starts
                      // its journal growing again -- if enough events land
                      // before the reload, the new journal overtakes the
                      // stale cursor and the defect hides. That race is why
                      // this arm passed twice against a build with the fix
                      // deliberately disabled before it was pinned down.
                      connection.closedByUser = true;
                      const r = await fetch(
                        BASE + "api/translator/sessions/" + id,
                        {method: "DELETE"});
                      return {id: id, status: r.status};
                    }"""
                )
                # The cursor is REPORTED, not assumed. This arm was written
                # once on the assumption that a reload alone re-sends a high
                # cursor; it passed with the fix disabled, because after a
                # single turn the stale cursor is small enough that the new
                # journal overtakes it within the next turn. The number is the
                # difference between a gate and a decoration, so it is printed.
                carried = await page.evaluate(
                    "() => ({cursor: connection.cursor, "
                    "stored: sessionStorage.getItem('translator.cursor')})"
                )
                print(f"[gate] the page carries cursor {carried} into the "
                      "reload; live delivery is only swallowed while the new "
                      "journal sits below it")
                print(f"[gate] dropped the session server-side: {dropped}")
                if not dropped or dropped.get("status") not in (200, 204):
                    failures.append(
                        f"the reload arm could not drop the session: {dropped}"
                    )
                print(f"[gate] reloading the page after turn {i} -- the "
                      "next turn must arrive LIVE, with history on board")
                await page.goto(
                    args.url, wait_until="domcontentloaded", timeout=60000
                )
                try:
                    await page.wait_for_function(
                        "() => connection && connection.ws "
                        "&& connection.ws.readyState === 1",
                        timeout=30000,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"the client never reopened its socket after the "
                        f"reload: {exc}"
                    )
                    break
                await page.evaluate(PROBE_JS)
                if args.prove_can_fail:
                    await page.evaluate(SABOTAGE_JS)
            turn = await one_turn(page, i + 1, min(speak_s, 4.0), budgets)
            results.append(turn)
            print(f"[gate] turn {turn['turn']}: line {turn['line_s']}s "
                  f"audio {turn['audio_s']}s frames {turn['audio_frames']} "
                  f"@{turn['rate']}Hz | client {turn['client']} | "
                  f"{turn['text']}")
            if turn["saw_waiting"]:
                print("[gate]   the page announced the queue while it waited")
            for stage in turn["stages"]:
                print(f"[gate]   stages: {stage_line(stage)}")
            if args.max_first_audio_s is not None:
                for stage in turn["stages"]:
                    server_s = stage.get("first_audio_ms", 0.0) / 1000.0
                    if server_s > args.max_first_audio_s:
                        failures.append(
                            f"turn {turn['turn']}: first audio took "
                            f"{server_s:.1f}s server-side, over the "
                            f"{args.max_first_audio_s:.1f}s budget"
                        )
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
            # The output half. A turn that pushed frames into a context that
            # is not running produced no sound, however healthy every other
            # counter looks -- which is precisely what a phone reported while
            # this gate was passing.
            out = turn["out"]
            print(f"[gate]   output : {out['state']} @{out['rate']} "
                  f"t={out['current_time']} pushed {out['pushed']}/"
                  f"{out['scheduled']} resume {out['last_resume']}")
            if out["pushed"] and out["state"] != "running":
                failures.append(
                    f"turn {turn['turn']}: {out['pushed']} frames were pushed "
                    f"into an output context in state {out['state']!r} "
                    f"(resume {out['last_resume']}) -- nothing was heard"
                )
            if out["pushed"] and out["current_time"] <= 0:
                failures.append(
                    f"turn {turn['turn']}: the output clock never advanced "
                    f"(currentTime {out['current_time']}), so no scheduled "
                    f"buffer can have played"
                )
            if out["scheduled"] < out["pushed"]:
                failures.append(
                    f"turn {turn['turn']}: {out['pushed'] - out['scheduled']} "
                    f"pushed frames were never scheduled: {out['errors']}"
                )
            if out["blocked"]:
                failures.append(
                    f"turn {turn['turn']}: the page flagged its own output as "
                    f"blocked"
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
    return report(failures, results, console, build), results


def stage_line(stage: dict) -> str:
    """One turn's server-side decomposition, in pipeline order."""
    return " ".join(
        f"{key[:-3]} {stage.get(key, 0.0) / 1000.0:.2f}s" for key in STAGE_KEYS
    )


def median(values: list) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def stage_summary(results: list) -> None:
    """Median per stage over every turn -- the §18.4 decomposition.

    Median rather than mean: one cold first turn (model warm-up, an empty
    reference buffer) would drag a mean and hide what the steady state costs.
    """
    stages = [s for r in results for s in r.get("stages", [])]
    if not stages:
        print("[gate] stages  : none reported "
              "(no turn.done carried timings)")
        return
    print(f"[gate] stages  : medians over {len(stages)} translated turns")
    for key in STAGE_KEYS:
        values = [s.get(key, 0.0) / 1000.0 for s in stages]
        print(f"          {key[:-3]:<16} med {median(values):6.2f}s  "
              f"min {min(values):6.2f}s  max {max(values):6.2f}s")


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
    stage_summary(results)
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
    parser.add_argument(
        "--timings-wait-s", type=float, default=TIMINGS_WAIT_S,
        help="how long to wait for the turn.done summary frame after audio; "
             "diagnostic only, a late summary never fails a turn",
    )
    parser.add_argument(
        "--max-first-audio-s", type=float, default=None,
        help="fail a turn whose SERVER-side first_audio_ms exceeds this. "
             "Off by default on purpose: a budget picked before the floor is "
             "measured (DESIGN §18.4) would be a number nobody can defend",
    )
    parser.add_argument(
        "--reload-after", type=int, default=0, metavar="N",
        help="reload the page after turn N and keep speaking. The remaining "
             "turns then run on a client that carries history and a stored "
             "cursor, which is the state a phone is actually in. A gate that "
             "only ever loads once cannot see a defect that needs a reload -- "
             "and one reached the user: after a reload the old text was "
             "there, speaking again updated nothing and produced no sound",
    )
    parser.add_argument(
        "--prove-can-fail", action="store_true",
        help="disable the page's output unlock, so a run that still PASSES "
             "proves the output assertions are blind",
    )
    parser.add_argument(
        "--label", default="gate",
        help="tag written into --json-out, so two runs can be compared",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    code, results = asyncio.run(run_gate(args))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {"label": args.label, "exit": code, "url": args.url,
                 "turns": results},
                indent=2,
            ),
            encoding="utf-8",
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
