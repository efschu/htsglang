#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631 J.3 SURVIVAL ORACLE: does a request decoding across a flip live?

THE QUESTION THIS ANSWERS, and why it needs its own harness. Every flip in
this feature's history committed with an EMPTY pipeline, so "the flip
works" was only ever a statement about the flip, never about the requests.
The oracle puts one determined-answer request in flight, flips underneath
it, and asks two things in order:

  1. DID IT SURVIVE -- did the request continue and finish at all, or did
     the cutover drop it (the J.3 failure: resident_reqs 1 -> 1 -> 0 across
     the census bracket, then a stranded KV page and a stranded mamba lock
     and SIGQUIT).
  2. IS THE ANSWER STILL RIGHT -- surviving with a corrupted context is the
     WORSE failure, because nothing raises. The probe is a counting
     continuation: every token is checkable against the one before it, so a
     KV row moved wrongly, a row left behind, or a truncated GDN state
     shows up as a jump, a repeat or a restart rather than as silence.

DETERMINED, NOT BYTE-EXACT. The PP and TP layouts reduce in different
orders, so bit-identical output across a flip is not a property this
feature has or claims. The counting task has a single CORRECT answer
independent of layout, which is the property that makes it a usable
oracle; a byte-diff against a no-flip reference is reported as DATA, never
as the verdict.

Usage (server already up on the flip build):

    python3 scripts/route_a_631_survival_oracle.py --port 30030 \
        --direction pp_to_tp --flip-after 24 --limit 300

    # both legs, one command: flip out and back under the same request
    python3 scripts/route_a_631_survival_oracle.py --port 30030 --round-trip

Exit code 0 only when the request survived AND its answer is correct.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

START = 10
END = 400
# CHAT, THINKING OFF, and both halves are load-bearing.
#
# The first probe here was a raw completion ("continue this sequence"),
# and it produced a MEASURED FALSE NEGATIVE: the model kept the sequence
# perfectly across the flip and then editorialised about how long to
# count. The no-flip control drifted EARLIER (32 numbers) than the flip
# run (43), which is what proved the drift was the model rather than the
# cutover -- but a probe whose verdict depends on the model's appetite for
# commentary cannot be a standing test. Chat framing with thinking
# disabled holds the sequence for hundreds of tokens (measured: a clean
# 11..120), so a break in it means something real.
PROMPT = (
    f"Count from {START + 1} to {END}. Output ONLY the integers separated "
    f"by single spaces, nothing else."
)


def _post(url: str, payload: dict, timeout: float) -> Tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - the verdict must survive it
        return -1, repr(exc)


class _Stream:
    """A streaming completion, with a per-chunk stall watchdog.

    The stall clock is the load-bearing part: a dropped request does not
    return an error, it simply never produces another token. Without a
    bound the oracle would hang exactly where the defect is.
    """

    def __init__(self, base: str, limit: int, stall_s: float):
        self.base = base
        self.limit = limit
        self.stall_s = stall_s
        self.text = ""
        self.chunks = 0
        self.last_chunk_at = 0.0
        self.error: Optional[str] = None
        self.done = False

    def run(self) -> None:
        payload = {
            "model": "Qwen3.6-27B",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": self.limit,
            "temperature": 0.0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = f"{self.base}/v1/chat/completions"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.stall_s) as resp:
                self.last_chunk_at = time.time()
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choice = obj.get("choices", [{}])[0]
                    piece = choice.get("delta", {}).get("content") or choice.get(
                        "text", ""
                    )
                    if piece:
                        self.text += piece
                        self.chunks += 1
                        self.last_chunk_at = time.time()
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
        finally:
            self.done = True


def correct_prefix(text: str) -> Tuple[int, str]:
    """How many consecutive integers from START+1 the answer gets right.

    A COUNT, not a boolean, and that is the correction the first metal run
    forced. What this oracle must decide is whether the request kept a
    COHERENT CONTEXT ACROSS THE CUTOVER -- not whether the model is
    willing to count to 400. So the caller compares this prefix against
    the point where the flip landed: numbers produced correctly AFTER the
    flip are the evidence, and anything the model does later is its own
    business.
    """
    expect = START + 1
    n = 0
    pieces = text.replace(",", " ").split()
    # A max_tokens cut lands MID-NUMBER ("...157 1"), and counting that
    # stub as a break reads like corruption in the report. Drop a trailing
    # piece only when the text does not end on whitespace, i.e. only when
    # the generator was actually cut off mid-token.
    if pieces and not text.endswith((" ", "\n")):
        pieces = pieces[:-1]
    for piece in pieces:
        try:
            got = int(piece)
        except ValueError:
            return n, f"non-integer token {piece!r} after {n} numbers"
        if got != expect:
            return n, f"expected {expect}, got {got} at position {n}"
        expect += 1
        n += 1
    return n, "no break: every token was the next integer"


def flip(base: str, direction: str, timeout: float) -> Tuple[int, str]:
    return _post(
        f"{base}/phase_flip", {"direction": direction}, timeout=timeout
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--direction", default="pp_to_tp")
    ap.add_argument("--round-trip", action="store_true",
                    help="flip out after --flip-after chunks and back after "
                         "twice that, under ONE request")
    ap.add_argument("--flip-after", type=int, default=24,
                    help="stream chunks to observe before arming the flip")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--stall-s", type=float, default=120.0)
    ap.add_argument("--verify-log", default="/spinning/serving-30030.boot.log",
                    help="serving log to read the COMMIT evidence from. An "
                         "HTTP 200 from /phase_flip only means ARMED: a leg "
                         "that parks and abandons also returns 200, and "
                         "counting that as a pass is exactly how a refused "
                         "leg once read as a green round trip.")
    ap.add_argument("--margin", type=int, default=10,
                    help="numbers that must still be CORRECT after each "
                         "flip; the bar is post-cutover coherence, not the "
                         "model's willingness to keep counting")
    ap.add_argument("--reference", action="store_true",
                    help="run WITHOUT flipping, to record the determined "
                         "answer this build produces when nothing moves")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    # Watermark the log so only THIS run's lines are read back.
    log_at_start = 0
    try:
        with open(args.verify_log, "rb") as fh:
            fh.seek(0, 2)
            log_at_start = fh.tell()
    except OSError:
        pass

    stream = _Stream(base, args.limit, args.stall_s)
    t = threading.Thread(target=stream.run, daemon=True)
    t0 = time.time()
    t.start()

    flips: List[Tuple[str, int, str, float]] = []
    watermarks: List[Tuple[str, int]] = []
    if not args.reference:
        legs = [(args.direction, args.flip_after)]
        if args.round_trip:
            back = "tp_to_pp" if args.direction == "pp_to_tp" else "pp_to_tp"
            legs.append((back, args.flip_after * 2))
        for direction, after in legs:
            # Wait for real decode progress before flipping: a flip issued
            # before the first token would once again prove nothing.
            while not stream.done and stream.chunks < after:
                time.sleep(0.05)
            if stream.done:
                print(
                    f"[oracle] stream ended before the {direction} leg could "
                    f"be armed ({stream.chunks} chunks)"
                )
                break
            # The numbers already emitted when the flip is armed: the
            # correctness bar is what comes AFTER this watermark.
            at_numbers, _ = correct_prefix(stream.text)
            watermarks.append((direction, at_numbers))
            code, body = flip(base, direction, timeout=args.stall_s)
            flips.append((direction, code, body[:120], time.time() - t0))
            print(
                f"[oracle] armed {direction} at {stream.chunks} chunks / "
                f"{at_numbers} numbers -> HTTP {code}"
            )

    # Stall watchdog: a dropped request goes silent rather than erroring.
    while not stream.done:
        if stream.last_chunk_at and (
            time.time() - stream.last_chunk_at > args.stall_s
        ):
            print(
                f"[oracle] STALLED: no token for {args.stall_s:.0f}s after "
                f"{stream.chunks} chunks"
            )
            break
        time.sleep(0.1)
    t.join(timeout=5.0)
    elapsed = time.time() - t0

    # COMMIT EVIDENCE, from the log rather than from the arm response.
    tail = ""
    try:
        with open(args.verify_log, "rb") as fh:
            fh.seek(log_at_start)
            tail = fh.read().decode("utf-8", "replace")
    except OSError as exc:
        tail = f"<unreadable: {exc}>"
    committed = {
        direction: tail.count(f"PHASE-FLIP DONE {direction}")
        for direction, _ in watermarks
    }
    carried_lines = tail.count("PHASE-FLIP-CARRY carried")
    abandoned = tail.count("FLIP ABANDONED") + tail.count("abandon")

    n_correct, why = correct_prefix(stream.text)
    survived = stream.error is None and stream.chunks > 0 and stream.done
    # SURVIVED THE FLIP: the correct sequence has to extend past every
    # watermark by a margin, i.e. the request went on producing the RIGHT
    # continuation after the cutover moved its KV, its GDN state and its
    # scheduler bookkeeping to another layout.
    margin = args.margin
    survived_each = [
        (direction, mark, n_correct >= mark + margin)
        for direction, mark in watermarks
    ]
    print("=" * 68)
    print(f"[oracle] elapsed          {elapsed:.1f}s")
    print(f"[oracle] chunks           {stream.chunks}")
    print(f"[oracle] correct prefix   {n_correct} numbers ({why})")
    print(f"[oracle] flips            {flips if flips else 'none (reference)'}")
    print(f"[oracle] post-flip margin {survived_each or 'n/a (reference)'}")
    print(f"[oracle] committed legs  {committed or 'n/a (reference)'}")
    print(f"[oracle] carry log lines {carried_lines}")
    print(f"[oracle] abandon mentions {abandoned}")
    print(f"[oracle] stream error     {stream.error}")
    print(f"[oracle] survived         {survived}")
    print(f"[oracle] tail             ...{stream.text[-90:]!r}")
    if args.reference:
        verdict = survived and n_correct > 0
    else:
        verdict = (
            survived
            and len(flips) > 0
            # Every armed leg must have COMMITTED on this rank's log, and
            # the answer must stay right past each one. Either half alone
            # is a false pass: a leg that never commits proves nothing,
            # and a leg that commits while the answer breaks is the
            # silent-corruption case this oracle exists for.
            and all(committed.get(d, 0) > 0 for d, _ in watermarks)
            and all(ok for _, _, ok in survived_each)
        )
    print(f"[oracle] VERDICT          {'PASS' if verdict else 'FAIL'}")
    print("=" * 68)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
