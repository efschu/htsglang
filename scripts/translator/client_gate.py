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
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://efeu.ddnss.de/translate/"
#: The tenant's own metrics endpoint, read directly rather than through the
#: public front door: ``/translate/`` is a path mount and does not carry it.
DEFAULT_METRICS_URL = "http://192.168.0.101:30800/metrics"
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
  window.__gate = { frames: 0, samples: 0, rate: 0, errors: [], timings: [],
                    // §19.13 ordering: text and audio have to be timestamped
                    // on ONE clock to be comparable at all, so both are taken
                    // from `performance.now()` in the page rather than from
                    // the harness, which sees only what it polls for.
                    text: [], firstPushAt: null,
                    // §19.12: the stop's receipt, read where it arrives.
                    acks: [] };
  const original = playback.push.bind(playback);
  playback.push = (float32, rate) => {
    window.__gate.frames += 1;
    window.__gate.samples += float32.length;
    window.__gate.rate = rate;
    if (window.__gate.firstPushAt === null) {
      window.__gate.firstPushAt = performance.now();
    }
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
      // Every clause of translated TEXT, stamped on arrival. The defect this
      // catches is not "no text" -- it is text that arrives at the speed of
      // the SPEAKER, which only an ordering against the audio can show.
      if (event && event.kind === 'turn.translation') {
        window.__gate.text.push({
          at: performance.now(),
          turn_id: event.turn_id || null,
          partial: !!event.partial,
          chars: (event.text || '').length,
        });
      }
      if (event && event.kind === 'playback.stop.ack') {
        window.__gate.acks.push({at: performance.now(), ack: event});
      }
    });
    return true;
  };
  return window.__gateAttach();
}
"""

#: Translation text a PERSON can see: rendered, non-empty, and laid out with
#: a non-zero box. Read off the live DOM, never off the socket.
#:
#: Adopted from the UI agent's harness (`live_gate_cut3.log`), which is where
#: it caught the field defect: `case turn.translation` opened with
#: `if (event.partial) break;`, so every streamed clause was discarded and the
#: screen showed a placeholder until the whole-translation frame arrived at
#: speaking time. Every wire-level assertion in this gate passed throughout --
#: the text WAS early, 2.6-7.1 s before the first audio frame, and invisible.
#:
#: That is the THIRD instance of the §17.8 class (the arm reads the channel
#: the code writes to instead of the surface the user reads), after the
#: whitelist that bound the label and not the decode, and the cursor that
#: outlived its session. It belongs in the standard repertoire rather than in
#: one agent's harness, which is why it lives here now.
VISIBLE_TEXT_JS = """
() => {
  const rows = document.querySelectorAll('#transcript .line .dst .tr .txt');
  let best = '';
  for (const row of rows) {
    const box = row.getBoundingClientRect();
    const shown = box.width > 0 && box.height > 0;
    if (shown && row.textContent.trim().length > best.length) {
      best = row.textContent.trim();
    }
  }
  return best;
}
"""

#: Is the NEWEST transcript line actually inside the scroll container's
#: viewport? The user's report was "das neue gesprochene muss im text immer
#: automatisch hochgescrollt werden", and the client already answers it --
#: `atBottom()` with a 60 px threshold plus `follow()` at every append, which
#: is the standard "only auto-scroll if the user has not scrolled up" shape.
#: That behaviour had no arm, so nothing would have noticed it regressing.
#:
#: Geometry, not scrollTop arithmetic: a line can be "scrolled to" and still
#: sit behind a sticky control, and what the user means is that they can SEE
#: the newest line.
#:
#: CORRECTED after the fix: the first version demanded the whole newest line
#: fit inside the box, and it stayed red against a client provably pinned to
#: the bottom (`overflow_below -4px, at bottom True`). One long turn makes a
#: bubble TALLER than the box on a phone, and "fully inside" is then
#: unsatisfiable at any scroll position -- the arm was asserting something no
#: correct client can do. The bottom edge is the assertion; the top edge is
#: only required when the line is short enough to have one on screen.
#: `scripts/translator/probe_autoscroll.py` carries the same predicate and
#: exercises it per mutation without a server.
NEWEST_LINE_VISIBLE_JS = """
() => {
  const box = document.getElementById('transcript');
  const lines = box ? box.querySelectorAll('.line') : [];
  if (!box || !lines.length) return null;
  const last = lines[lines.length - 1].getBoundingClientRect();
  const view = box.getBoundingClientRect();
  const taller = last.height > view.height;
  return {
    lines: lines.length,
    taller_than_view: taller,
    // Inside, with a pixel of tolerance for sub-pixel layout.
    visible: last.bottom <= view.bottom + 1
             && (taller || last.top >= view.top - 1),
    overflow_below: Math.round(last.bottom - view.bottom),
    scrolled_to_bottom:
      box.scrollHeight - box.scrollTop - box.clientHeight < 60,
  };
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


#: How long a stop may still be followed by audio before the button is a lie.
#: Generous on purpose: frames already on the wire when the stop is written
#: still have to arrive, and failing those would be failing physics.
STOP_QUIESCE_S = 8.0
#: But the tail has to go QUIET, not merely thin out. A turn that keeps
#: trickling frames for the whole window has not been stopped, whatever the
#: rate. This is the assertion; the window above only bounds the wait.
STOP_QUIET_S = 2.5
#: Under the ``line`` trigger the arm fires BEFORE any audio exists, so the
#: question is no longer "did the tail stop" but "did the audio ever come".
#: That cannot be answered by an early break on silence -- silence is the
#: state the arm starts in. The window is therefore watched to the end, and
#: sized to outlast a healthy turn's synthesis: measured first audio is
#: 4-15 s from the turn's start against a transcript line at ~4 s, so 20 s
#: after the line covers it with margin. The control that proves the window
#: is long enough is the sabotage run, which must see frames inside it.
STOP_WATCH_S = 20.0


def read_talker_gauges(metrics_url: str) -> dict:
    """The two gauges that say WHERE a turn is: in synthesis, or behind it.

    Diagnostic only -- never an assertion. §17.8.11 added
    ``translator_talker_busy`` and ``translator_session_queue_depth_max``
    precisely so a stop arm can record whether the pipeline was synthesizing
    (busy 1, depth 0) or queued (depth > 0) at the instant the stop was
    written, instead of leaving that to be argued afterwards.

    Both gauges are process-wide rather than per session, so with more than
    one live conversation they answer "was this SERVER synthesizing", not
    "was this session". That is why they are printed and not asserted.
    """
    wanted = ("translator_talker_busy", "translator_session_queue_depth_max")
    out: dict = {}
    try:
        with urllib.request.urlopen(metrics_url, timeout=3) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    for raw in body.splitlines():
        if raw.startswith("#"):
            continue
        name, _, value = raw.partition(" ")
        if name in wanted:
            try:
                out[name.replace("translator_", "")] = float(value)
            except ValueError:
                continue
    return out


async def stop_arm(
    page, sabotage: bool = False, metrics_url: str = "",
    trigger: str = "line",
) -> dict:
    """Send ``playback.stop`` while the turn is live and watch the server.

    Two things are measured and they answer different questions. The ACK says
    the server understood and names what it abandoned -- that is the protocol
    half, and it is what the client needs in order to drop audio per turn id.
    The frame count going QUIET is the half that matters to a person: it is
    the difference between a stop and a receipt for a stop.

    ``sabotage`` runs the whole arm and sends nothing. It is the can-fail
    proof (§17.8.7's lesson: an arm whose failure nobody has seen is a
    decoration) -- with the frame never sent, no ack arrives and the arm must
    fail.

    WHEN THIS ARM FIRES IS THE WHOLE DESIGN, and it was wrong until run
    `gate_bundle_b.log` failed on it. It used to fire one poll after the first
    audio frame reached playback, which reads as "mid-playback" but is not:
    the talker is BATCH, so the client's first frame of a unit arrives only
    once that unit is fully synthesized. For a single-unit turn that instant
    is also the instant the turn ENDS and ``_active_turn_id`` is cleared
    (`session.py:1073`), so whether the ack could still name an aborted turn
    was decided by a race between a 250 ms poll and the server's own teardown.
    Run 1 of the bundle won that race, run 2 lost it and reported
    `aborted_turn_id None` -- correctly failed by the gate as an arm that
    tested nothing.

    The trigger is therefore the TRANSCRIPT LINE, not the first audio frame.
    A line on the page means the server is inside ``_run_turn_locked`` past
    recognition (`session.py:1157-1171`) with MT and synthesis still ahead of
    it, which is a guarantee rather than a race -- on the measured floor it
    buys ~6-10 s of margin instead of ~0. It also makes the QUIESCENCE half
    bite for the first time: fired before any audio exists, a working stop
    yields no frames at all, while the sabotage run gets the turn's entire
    output and fails. That half was a decoration under the old trigger
    because all the audio had already landed before the arm looked.

    ``--stop-trigger audio`` keeps the old behaviour for comparison. It tests
    the client half (drop what is already buffered), which the UI agent's own
    arm covers, and it is racy for the server half by construction.
    """
    gauges = read_talker_gauges(metrics_url) if metrics_url else {}
    frames_at_stop = await page.evaluate("window.__gate.frames")
    acks_before = await page.evaluate("window.__gate.acks.length")
    sent = False
    if not sabotage:
        sent = bool(await page.evaluate(
            """() => {
              if (!connection.ws || connection.ws.readyState !== 1) return false;
              connection.ws.send(JSON.stringify(
                {kind: 'playback.stop', reason: 'gate'}));
              return true;
            }"""
        ))
    # Watched to the end under ``line`` -- see ``STOP_WATCH_S``. Breaking
    # early on silence there would be reporting the arm's own starting state
    # as its result.
    watch_s = STOP_WATCH_S if trigger == "line" else STOP_QUIESCE_S
    t0 = time.monotonic()
    last_change = t0
    last_count = frames_at_stop
    # When audio FIRST appeared after the arm fired. Without this the turn's
    # ``audio_s`` would be stamped at the end of the watch window, reporting a
    # 20 s wait for audio that arrived in 5 -- a number that reads as a
    # measurement and is an artefact of how long the arm chose to look.
    first_frame_at = None
    while time.monotonic() - t0 < watch_s:
        await asyncio.sleep(0.25)
        now = await page.evaluate("window.__gate.frames")
        if now != last_count:
            if first_frame_at is None:
                first_frame_at = time.monotonic()
            last_count = now
            last_change = time.monotonic()
        elif (trigger == "audio"
                and time.monotonic() - last_change >= STOP_QUIET_S):
            break
    quiet_for = time.monotonic() - last_change
    acks = await page.evaluate(f"window.__gate.acks.slice({acks_before})")
    ws_state = await page.evaluate(
        "connection.ws ? connection.ws.readyState : -1"
    )
    return {
        "sent": sent,
        "sabotage": sabotage,
        "trigger": trigger,
        "first_frame_after_stop_at": first_frame_at,
        "frames_at_stop": frames_at_stop,
        "frames_after_stop": last_count - frames_at_stop,
        "quiet_for_s": round(quiet_for, 2),
        "quiesced": quiet_for >= STOP_QUIET_S,
        "acks": [entry["ack"] for entry in acks],
        "ws": ws_state,
        # What the server was doing when the stop was written. Printed, never
        # asserted -- see ``read_talker_gauges``.
        "gauges": gauges,
    }


async def one_turn(
    page, index: int, speak_s: float, budgets: dict, stop_mode: str = "",
    stop_trigger: str = "line", metrics_url: str = "",
) -> dict:
    """Tap, speak, tap, and wait for what a person would wait for."""
    # Re-attach after a reconnect: ``connection.ws`` is replaced, and the
    # listener goes with the old socket. Idempotent via a marker on the socket.
    await page.evaluate("() => window.__gateAttach && window.__gateAttach()")
    before_lines = await page.evaluate(
        "document.querySelectorAll('#transcript .line').length"
    )
    before_frames = await page.evaluate("window.__gate.frames")
    before_timings = await page.evaluate("window.__gate.timings.length")
    before_text = await page.evaluate("window.__gate.text.length")
    # The first push is a PER-TURN marker, so it is cleared per turn. Left
    # cumulative it would answer "did this page ever play anything", which is
    # not the question §19.13 asks.
    await page.evaluate("() => { window.__gate.firstPushAt = null; }")
    started = time.monotonic()

    await page.click("#talk")                     # tap to speak
    await asyncio.sleep(speak_s)
    await page.click("#talk")                     # tap to stop

    line_at = None
    audio_at = None
    text_visible_at = None
    # Did the page ever SAY it was waiting (§17.8.2)? Polled rather than
    # checked at the end, because the notice is deliberately transient: the
    # translation that clears it is the same event that ends the wait.
    saw_waiting = False
    # THE STOP ARM (§19.12). Fired from inside the wait loop the moment its
    # trigger is met -- see ``stop_arm`` for why that trigger is the
    # transcript line and not the first audio frame.
    stop_report = None
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
        # The surface a person reads, polled beside the wire so the two can be
        # compared on one clock. See ``VISIBLE_TEXT_JS``.
        if text_visible_at is None:
            if await page.evaluate(VISIBLE_TEXT_JS):
                text_visible_at = time.monotonic() - started
        if audio_at is None:
            if await page.evaluate("window.__gate.frames") > before_frames:
                audio_at = time.monotonic() - started
        armed = line_at if stop_trigger == "line" else audio_at
        if stop_mode and stop_report is None and armed is not None:
            stop_report = await stop_arm(
                page,
                sabotage=(stop_mode == "sabotage"),
                metrics_url=metrics_url,
                trigger=stop_trigger,
            )
            # The arm has already spent its own quiescence window watching the
            # frame counter, so everything this turn will ever deliver is in
            # its report. A stopped turn produces no audio BY DESIGN, and
            # waiting out the audio budget for it would only be waiting for
            # something the test just cancelled.
            if audio_at is None and stop_report["first_frame_after_stop_at"]:
                audio_at = (
                    stop_report["first_frame_after_stop_at"] - started
                )
            break
        # The exit condition must include EVERY property the turn is judged
        # on, or the arm races itself. It used to leave as soon as the line
        # and the audio were present, so on a fast turn -- first audio 1.74 s,
        # line and audio landing in the SAME poll -- the loop left before the
        # translation could render and reported "text was never visible" for a
        # turn whose text arrived a moment later. That is the §17.8 lesson
        # again, one level up: an arm that stops looking before the thing it
        # measures can appear is measuring its own timing.
        #
        # The stop turn is exempt: it is cancelled before MT finishes and
        # therefore owes no visible translation.
        done_text = text_visible_at is not None or bool(stop_mode)
        if line_at is not None and audio_at is not None and done_text:
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
    # Read AFTER the summary wait, so a clause that lands late is still in the
    # ordering rather than silently outside the window that judges it.
    text_events = await page.evaluate(
        f"window.__gate.text.slice({before_text})"
    )
    first_push_at = await page.evaluate("window.__gate.firstPushAt")
    # What the OUTPUT did, not what was handed to it. Counting playback.push
    # calls proves the frames arrived and says nothing about whether a sound
    # was made -- the distinction a real device paid for.
    scroll = await page.evaluate(NEWEST_LINE_VISIBLE_JS)
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
        "text_visible_s": (
            None if text_visible_at is None else round(text_visible_at, 1)
        ),
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
        # §19.13, on the page's own clock: every translated clause with its
        # arrival time, and when the first audio frame was pushed.
        "text_events": text_events,
        "first_push_at": first_push_at,
        "stop": stop_report,
        # Auto-scroll (user order): the newest line must be on screen.
        "scroll": scroll,
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
            stop_mode = ""
            if args.stop_during_playback and i + 1 == args.stop_during_playback:
                stop_mode = "sabotage" if args.stop_sabotage else "send"
            turn = await one_turn(
                page, i + 1, min(speak_s, 4.0), budgets, stop_mode=stop_mode,
                stop_trigger=args.stop_trigger,
                metrics_url=args.metrics_url,
            )
            results.append(turn)
            print(f"[gate] turn {turn['turn']}: line {turn['line_s']}s "
                  f"text-visible {turn['text_visible_s']}s "
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
            # A turn the stop arm ABORTED is supposed to produce no audio, so
            # the audio assertion would fail it for doing exactly what was
            # asked. The sabotage variant sends nothing and therefore still
            # owes its audio -- without it the can-fail proof is vacuous, and
            # that vacuity is asserted a few lines below.
            stopped_on_purpose = (
                turn["stop"] is not None and turn["stop"]["sent"]
            )
            if turn["audio_s"] is None and not stopped_on_purpose:
                failures.append(
                    f"turn {turn['turn']}: no audio reached playback within "
                    f"{budgets['audio']}s"
                )
            # §19.13 ORDERING. The user's report was not "no text" -- it was
            # text that arrived at the speed of the speaker, because `speak()`
            # was awaited inside the MT loop. The property that fix delivers is
            # an ORDER, so an order is what is asserted: every translated
            # clause of the turn is on the page before its first audio frame.
            #
            # This holds while MT is faster than one clause of synthesis, which
            # on the measured floor it is by more than an order of magnitude
            # (§18.4: mt_total ~0.4 s against tts_first_audio ~5.9 s). The
            # MARGIN is printed rather than only the verdict, so a future
            # narrowing is visible long before it becomes a failure instead of
            # arriving as one.
            # THE ARM THE FIELD DEFECT WALKED THROUGH (§17.8, third instance).
            # Skipped on the stop turn only because that turn is cancelled
            # before MT can finish, so it owes no translation at all.
            if not stop_mode:
                if turn["text_visible_s"] is None:
                    failures.append(
                        f"turn {turn['turn']}: no translated text was ever "
                        f"VISIBLE in the DOM within {budgets['line']}s -- "
                        f"every wire assertion can still pass while the "
                        f"screen shows a placeholder"
                    )
                elif (turn["audio_s"] is not None
                        and turn["text_visible_s"] > turn["audio_s"] + 0.5):
                    failures.append(
                        f"turn {turn['turn']}: the translation became "
                        f"readable at {turn['text_visible_s']}s, AFTER the "
                        f"audio started at {turn['audio_s']}s -- the text was "
                        f"early on the wire and the screen threw that away"
                    )
            if not stop_mode:
                order = text_order(turn)
                print(f"[gate]   text/audio: {order['events']} streamed "
                      f"clauses, last one {order['margin_s']}s before the "
                      f"first audio frame")
                if order["late"]:
                    failures.append(
                        f"turn {turn['turn']}: {len(order['late'])} of "
                        f"{order['events']} streamed clauses arrived AFTER "
                        f"the first audio frame (§19.13: text must not wait "
                        f"for audio); margins {order['late']}"
                    )
                if turn["audio_s"] is not None and not order["events"]:
                    # Silence here would be the decoration case: no clause to
                    # compare reads as a pass, on precisely the turn where the
                    # streaming path did not stream.
                    failures.append(
                        f"turn {turn['turn']}: audio arrived but no streamed "
                        f"translation clause did, so the ordering was never "
                        f"actually tested"
                    )
            scroll = turn.get("scroll")
            if scroll:
                print(f"[gate]   scroll : {scroll['lines']} lines, newest "
                      f"visible {scroll['visible']}, overflow below "
                      f"{scroll['overflow_below']}px, at bottom "
                      f"{scroll['scrolled_to_bottom']}"
                      + (", taller than view"
                         if scroll.get("taller_than_view") else ""))
                if not scroll["visible"]:
                    failures.append(
                        f"turn {turn['turn']}: the newest transcript line is "
                        f"{scroll['overflow_below']}px below the visible area "
                        f"-- new speech must scroll itself into view"
                    )
            if turn["stop"] is not None:
                stop = turn["stop"]
                print(f"[gate]   stop   : sent {stop['sent']} "
                      f"sabotage {stop['sabotage']} "
                      f"trigger {args.stop_trigger} "
                      f"frames before stop {stop['frames_at_stop']} "
                      f"frames after stop {stop['frames_after_stop']} "
                      f"quiet {stop['quiet_for_s']}s ws {stop['ws']} "
                      f"gauges {stop['gauges']} "
                      f"acks {stop['acks']}")
                # The can-fail proof has to be able to fail. Sent nothing and
                # got no audio either: the quiescence half was true for want
                # of a workload, not because a stop worked (§17.8.10).
                if stop["sabotage"] and stop["frames_after_stop"] == 0:
                    failures.append(
                        f"turn {turn['turn']}: the sabotage arm saw no audio "
                        f"at all, so 'it went quiet' was trivially true and "
                        f"the quiescence half proved nothing"
                    )
                # Fired before the talker produced anything, a stop that works
                # means the turn is never spoken AT ALL -- not merely that it
                # tails off. The sabotage run is the control that this window
                # is long enough to have seen audio.
                if (stop["trigger"] == "line" and stop["sent"]
                        and stop["frames_after_stop"] > 0):
                    failures.append(
                        f"turn {turn['turn']}: {stop['frames_after_stop']} "
                        f"audio frames arrived after a stop sent BEFORE any "
                        f"audio existed -- the turn was not abandoned, it was "
                        f"spoken anyway"
                    )
                if not stop["acks"]:
                    failures.append(
                        f"turn {turn['turn']}: playback.stop was never "
                        f"acknowledged -- the client cannot know which turn's "
                        f"audio to drop"
                    )
                else:
                    ack = stop["acks"][0]
                    missing = [
                        k for k in
                        ("aborted_turn_id", "dropped_queued", "stop_epoch")
                        if k not in ack
                    ]
                    if missing:
                        failures.append(
                            f"turn {turn['turn']}: the stop ack is missing "
                            f"{missing}"
                        )
                    elif ack.get("aborted_turn_id") is None:
                        failures.append(
                            f"turn {turn['turn']}: the stop ack names no "
                            f"aborted turn, so nothing was in flight when the "
                            f"arm fired -- the arm tested nothing"
                        )
                if not stop["quiesced"]:
                    failures.append(
                        f"turn {turn['turn']}: audio kept arriving for "
                        f"{STOP_QUIESCE_S}s after the stop "
                        f"({stop['frames_after_stop']} frames, longest quiet "
                        f"stretch {stop['quiet_for_s']}s) -- it did not stop"
                    )
                if stop["ws"] != 1:
                    failures.append(
                        f"turn {turn['turn']}: the socket is in state "
                        f"{stop['ws']} after the stop; a stop must free the "
                        f"talker, not the connection"
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


def text_order(turn: dict) -> dict:
    """Where each STREAMED clause sits relative to the first audio frame.

    Only the partial events are judged, which is the shipped contract
    (`test_text_before_audio.py` filters on exactly this flag): the streamed
    clauses are the text a reader sees arrive, while the final non-partial
    event is the whole-translation record, journalled once the turn is done
    and therefore legitimately after the audio it summarises. Asserting over
    both fails every healthy turn by ~0.02 s -- measured, and it is what this
    arm did before the contract was read.

    Returns the clause count, the margin of the LAST clause (the tight one --
    the first is trivially early), and any clause that was late. A turn with
    no partial clause at all answers "nothing to compare" rather than passing
    by default, which is how an ordering assertion becomes a decoration on
    exactly the turns that would fail it.
    """
    events = [e for e in (turn.get("text_events") or []) if e.get("partial")]
    first_push = turn.get("first_push_at")
    if not events or first_push is None:
        return {"events": len(events), "margin_s": None, "late": []}
    late = [
        round((e["at"] - first_push) / 1000.0, 2)
        for e in events if e["at"] > first_push
    ]
    margin = round((first_push - max(e["at"] for e in events)) / 1000.0, 2)
    return {"events": len(events), "margin_s": margin, "late": late}


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
        "--stop-during-playback", type=int, default=0, metavar="N",
        help="on turn N, send playback.stop once the first audio frame has "
             "reached the playback path, and require the server to say what "
             "it abandoned AND to go quiet. Put a turn after it (--turns N+1) "
             "to prove the stop freed the talker rather than wedging it",
    )
    parser.add_argument(
        "--stop-trigger", choices=("line", "audio"), default="line",
        help="WHEN the stop arm fires. 'line' (default) fires once the "
             "transcript line is on the page, which guarantees the server is "
             "inside the turn with synthesis still ahead of it. 'audio' is "
             "the old trigger and is racy against a batch talker: the first "
             "frame of a single-unit turn arrives as that turn ends, so the "
             "ack could name no aborted turn through no fault of the server "
             "(gate_bundle_b.log)",
    )
    parser.add_argument(
        "--metrics-url", default=DEFAULT_METRICS_URL,
        help="read translator_talker_busy / _session_queue_depth_max at the "
             "instant of the stop and print them. Diagnostic only, never "
             "asserted -- the gauges are process-wide, not per session. "
             "Empty string disables the read",
    )
    parser.add_argument(
        "--stop-sabotage", action="store_true",
        help="run the stop arm and send NOTHING. The can-fail proof: the arm "
             "must then FAIL on both halves (no ack, audio keeps arriving)",
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
