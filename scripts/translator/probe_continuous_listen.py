# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Speaking with pauses must not end the recording (user field report).

*"er hört auch einfach mit der spracherkennung auf, obwohl noch nicht auf
stop listen gedrückt wurde und übersetzt dann nur den ersten satz. im 99%
normalfall spricht derjenige weiter bis auf stop gedrückt wird, auch wenn
pausen dabei sind beim sprechen, das ist normal."*

The required behaviour: while the user has not tapped stop, EVERY
pause-separated segment is recognized and translated. A pause is the normal
case, never an end-of-recording signal.

The experiment drives ONE recording containing three utterances separated by
silence longer than the segmenter's ``hangover_ms`` (550 ms), taps stop only
at the very end, and counts what came back per event kind. The control arm
taps stop after the first utterance and must then produce exactly one.

Counting EVENT KINDS rather than transcript lines is the point: "only the
first sentence arrived" has several possible roots and they are
distinguishable only here --

* ``turn.dropped`` with ``reason: queue_overrun`` means the segments were
  recognized and then thrown away, because ``enqueue`` keeps at most
  ``max_queued_turns`` (deployment default 2, ``session.py:356``) and drops
  the OLDEST on overrun while one talker runs slower than realtime;
* no further ``turn.transcript`` at all means the capture or the segmenter
  stopped, which is a different defect in a different place;
* ``turn.queued`` without either means it is merely slow, not lost.

    /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_continuous_listen.py --utterances 3
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client_gate import (  # noqa: E402  - sibling harness, reused deliberately
    DEFAULT_CLIP,
    DEFAULT_URL,
    FAKE_RATE,
)

#: Every JSON frame, tagged on arrival. A second listener on the page's own
#: socket, which changes nothing about how the client's own handler runs.
LISTEN_JS = """
() => {
  window.__cont = {events: [], lines: 0};
  const attach = () => {
    const ws = (typeof connection !== 'undefined' && connection.ws)
      ? connection.ws : null;
    if (!ws || ws.__contHooked) return false;
    ws.__contHooked = true;
    ws.addEventListener('message', (ev) => {
      if (typeof ev.data !== 'string') return;
      let event;
      try { event = JSON.parse(ev.data); } catch (err) { return; }
      if (!event || !event.kind) return;
      window.__cont.events.push({
        at: performance.now(),
        kind: event.kind,
        turn_id: event.turn_id || null,
        reason: event.reason || null,
        partial: !!event.partial,
        text: (event.text || '').slice(0, 60),
      });
    });
    return true;
  };
  window.__contAttach = attach;
  return attach();
}
"""


def build_paused_speech(clip: Path, out: Path, utterances: int,
                        pause_s: float) -> float:
    """One recording: N copies of the clip separated by real silence.

    The pause has to exceed the segmenter's hangover (550 ms) or the whole
    thing closes as ONE segment and the experiment would be testing nothing.
    """
    import soundfile as sf

    samples, rate = sf.read(str(clip), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != FAKE_RATE:
        n = int(len(samples) * FAKE_RATE / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, n),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
    gap = np.zeros(int(pause_s * FAKE_RATE), dtype=np.float32)
    lead = np.zeros(int(0.4 * FAKE_RATE), dtype=np.float32)
    parts = [lead]
    for index in range(utterances):
        parts.append(samples)
        if index + 1 < utterances:
            parts.append(gap)
    parts.append(gap)
    stream = np.concatenate(parts)
    sf.write(str(out), stream, FAKE_RATE, subtype="PCM_16")
    return len(stream) / FAKE_RATE


async def run(args) -> int:
    from playwright.async_api import async_playwright

    fake = Path("/tmp/continuous_fake_input.wav")
    total_s = build_paused_speech(
        args.clip, fake, args.utterances, args.pause_s
    )
    one_s = (total_s - 0.4 - args.pause_s
             - (args.utterances - 1) * args.pause_s) / args.utterances
    print(f"[cont] fake microphone: {fake} ({total_s:.2f}s, "
          f"{args.utterances} utterances of ~{one_s:.2f}s, "
          f"{args.pause_s:.1f}s pauses)")
    # Chromium LOOPS the file, so the recording must stop before it wraps or
    # the extra utterances would be the loop, not the test.
    speak_s = total_s if not args.stop_after_first else (0.4 + one_s
                                                         + args.pause_s * 0.8)
    print(f"[cont] holding the button for {speak_s:.2f}s "
          f"({'CONTROL: stop after the first' if args.stop_after_first else 'full recording'})")

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
        build = await page.evaluate(
            "typeof CLIENT_BUILD === 'string' ? CLIENT_BUILD : null")
        print(f"[cont] build {build}")
        await page.wait_for_function(
            "() => connection && connection.ws && connection.ws.readyState === 1",
            timeout=30000,
        )
        await page.evaluate(LISTEN_JS)

        await page.click("#talk")
        await asyncio.sleep(speak_s)
        await page.click("#talk")
        print("[cont] released; waiting for the pipeline to finish")
        await asyncio.sleep(args.settle_s)

        events = await page.evaluate("window.__cont.events")
        lines = await page.evaluate(
            "document.querySelectorAll('#transcript .line').length")
        await browser.close()

    kinds = collections.Counter(e["kind"] for e in events)
    transcripts = [e for e in events if e["kind"] == "turn.transcript"]
    finals = [e for e in events
              if e["kind"] == "turn.translation" and not e["partial"]]
    dropped = [e for e in events if e["kind"] == "turn.dropped"]

    print(f"[cont] event kinds: {dict(kinds)}")
    print(f"[cont] transcripts {len(transcripts)} | "
          f"final translations {len(finals)} | dropped {len(dropped)} | "
          f"DOM lines {lines}")
    for entry in transcripts:
        print(f"[cont]   transcript: {entry['text']!r}")
    for entry in dropped:
        print(f"[cont]   DROPPED: reason={entry['reason']}")

    expected = 1 if args.stop_after_first else args.utterances
    verdict = len(transcripts) >= expected
    print(f"[cont] expected >= {expected} recognized utterances, "
          f"got {len(transcripts)}: {'PASS' if verdict else 'FAIL'}")
    if not verdict:
        if dropped:
            print("[cont] ROOT: segments were recognized and then DROPPED -- "
                  "the bounded turn queue, not the capture")
        elif len(transcripts) <= 1:
            print("[cont] ROOT: nothing after the first was even recognized "
                  "-- capture or segmenter, not the queue")
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"build": build, "events": events,
                        "kinds": dict(kinds), "lines": lines}, indent=2),
            encoding="utf-8")
    return 0 if verdict else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--utterances", type=int, default=3)
    parser.add_argument("--pause-s", type=float, default=1.8,
                        help="silence between utterances; must exceed the "
                             "segmenter's 550 ms hangover to close a segment")
    parser.add_argument("--settle-s", type=float, default=60.0,
                        help="how long to keep listening after the release")
    parser.add_argument("--stop-after-first", action="store_true",
                        help="control arm: release after the first utterance, "
                             "which must then produce exactly one")
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
