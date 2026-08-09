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
PROMPT = (
    "Continue this sequence of integers, separated by single spaces, "
    "counting upward by one. Write only the numbers.\n"
    + " ".join(str(i) for i in range(1, START + 1))
    + " "
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
            "prompt": PROMPT,
            "max_tokens": self.limit,
            "temperature": 0.0,
            "stream": True,
        }
        url = f"{self.base}/v1/completions"
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
                    piece = obj.get("choices", [{}])[0].get("text", "")
                    if piece:
                        self.text += piece
                        self.chunks += 1
                        self.last_chunk_at = time.time()
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
        finally:
            self.done = True


def check_sequence(text: str) -> Tuple[bool, str]:
    """The determined answer: consecutive integers from START+1 upward."""
    toks: List[int] = []
    for piece in text.replace(",", " ").split():
        try:
            toks.append(int(piece))
        except ValueError:
            return False, f"non-integer token {piece!r} after {len(toks)} numbers"
    if not toks:
        return False, "no numbers produced at all"
    expect = START + 1
    for i, got in enumerate(toks):
        if got != expect:
            return (
                False,
                f"sequence broke at position {i}: expected {expect}, got "
                f"{got} (context corruption, not a slow answer)",
            )
        expect += 1
    return True, f"{len(toks)} consecutive integers, {toks[0]}..{toks[-1]}"


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
    ap.add_argument("--reference", action="store_true",
                    help="run WITHOUT flipping, to record the determined "
                         "answer this build produces when nothing moves")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    stream = _Stream(base, args.limit, args.stall_s)
    t = threading.Thread(target=stream.run, daemon=True)
    t0 = time.time()
    t.start()

    flips: List[Tuple[str, int, str, float]] = []
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
            at = stream.chunks
            code, body = flip(base, direction, timeout=args.stall_s)
            flips.append((direction, code, body[:200], time.time() - t0))
            print(
                f"[oracle] armed {direction} after {at} chunks -> HTTP {code}"
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

    ok_seq, why = check_sequence(stream.text)
    survived = stream.error is None and stream.chunks > 0 and stream.done
    print("=" * 68)
    print(f"[oracle] elapsed        {elapsed:.1f}s")
    print(f"[oracle] chunks         {stream.chunks}")
    print(f"[oracle] flips          {flips if flips else 'none (reference)'}")
    print(f"[oracle] stream error   {stream.error}")
    print(f"[oracle] survived       {survived}")
    print(f"[oracle] answer correct {ok_seq}: {why}")
    print(f"[oracle] tail           ...{stream.text[-120:]!r}")
    verdict = survived and ok_seq and (args.reference or len(flips) > 0)
    print(f"[oracle] VERDICT        {'PASS' if verdict else 'FAIL'}")
    print("=" * 68)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
