#!/usr/bin/env python3
"""#363 card gates 1+2 -- the workload that exercises the named regimes.

The classifier names four shapes (DESIGN_363 section 3.2). A trace that only
ever shows one of them proves nothing about the other three, so this driver
walks them deliberately, in the order a real server meets them:

    prefill burst  -> a batch of long prompts admitted at once. The queue
                      carries mass before a single prefill has run, which is
                      the predictive trigger of section 5.
    decode drain   -> those requests generating, queue empty. The shape the
                      DECODE_HEAVY entry threshold is written for.
    idle           -> nothing running, nothing queued. The window where every
                      actuator's group-idle boundary is reachable, and where
                      an idle window must classify as MIXED rather than as
                      100 % decode (an idle window is no measurement, not a
                      zero -- pinned hermetically, checked here on the rig).
    mixed          -> short prompts arriving while long ones generate.

Vanilla OpenAI-compatible client: nothing here knows it is talking to this
fork, so the trace is of the server's own behaviour and not of a bespoke
harness.

Card-less smoke: ``--dry-run`` prints the phase plan and the request shapes
without opening a socket, which is the execution rule for anything that will
later be pointed at a live server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List

#: Long enough that one prompt's prefill dominates a decode round by orders on
#: this rig, which is what makes the burst a burst. DESIGN_363 section 5.2
#: proposes 8192 as the single-prompt pre-stage trigger; the burst has to sit
#: above it or the phase does not test what it claims to.
BURST_PROMPT_TOKENS = 12000
#: Short enough to be a decode-shaped request rather than a second burst.
SHORT_PROMPT_TOKENS = 64

PHASES = ("prefill_burst", "decode_drain", "idle", "mixed")


def _prompt(tokens: int, tag: str) -> str:
    """A prompt of roughly ``tokens`` tokens.

    Deliberately repetitive filler plus a unique tag: the content does not
    matter to a regime classifier that reads only queue shape and round mix,
    and the tag keeps the prefix cache from serving one request out of
    another's prefill -- which would make a burst stop being one.
    """
    filler = ("the quick brown fox jumps over the lazy dog. " * ((tokens // 9) + 1))[
        : tokens * 4
    ]
    return f"[{tag}] {filler}\n\nSummarise the above in one sentence."


class Client:
    def __init__(self, base: str, model: str, timeout: float = 600.0):
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def completion(self, prompt: str, max_tokens: int) -> Dict:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=10) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False


def plan(args) -> List[Dict]:
    """The phase plan. Pure, so ``--dry-run`` prints exactly what will run."""
    out: List[Dict] = []
    out.append(
        {
            "phase": "prefill_burst",
            "why": "queue carries mass before a prefill has run -> PREFILL_HEAVY",
            "concurrent": args.burst,
            "prompt_tokens": BURST_PROMPT_TOKENS,
            # Long generations on long prompts is what actually HOLDS KV: a
            # burst that finishes in 16 tokens spikes the queue and empties
            # the pool again before occupancy can rise. The first gates window
            # peaked at 6.2 % and exercised nothing on the admissibility axis.
            "max_tokens": args.burst_tokens,
        }
    )
    out.append(
        {
            "phase": "decode_drain",
            "why": "long generations, empty queue -> DECODE_HEAVY",
            "concurrent": args.drain,
            "prompt_tokens": SHORT_PROMPT_TOKENS,
            "max_tokens": args.drain_tokens,
        }
    )
    out.append(
        {
            "phase": "idle",
            "why": (
                "nothing running, nothing queued -> MIXED (an idle window is "
                "no measurement, not 0 % prefill)"
            ),
            "concurrent": 0,
            "prompt_tokens": 0,
            "max_tokens": 0,
            "hold_s": args.idle_s,
        }
    )
    out.append(
        {
            "phase": "mixed",
            "why": "short arrivals during long generations -> MIXED",
            "concurrent": args.mixed,
            "prompt_tokens": SHORT_PROMPT_TOKENS,
            "max_tokens": args.drain_tokens // 2,
        }
    )
    return out


def run_phase(client: Client, spec: Dict, repeats: int) -> Dict:
    from concurrent.futures import ThreadPoolExecutor

    if spec["concurrent"] == 0:
        time.sleep(spec.get("hold_s", 30))
        return {"phase": spec["phase"], "sent": 0, "errors": 0}

    sent = errors = 0
    for cycle in range(repeats):
        prompts = [
            _prompt(spec["prompt_tokens"], f"{spec['phase']}-{cycle}-{i}")
            for i in range(spec["concurrent"])
        ]
        with ThreadPoolExecutor(max_workers=spec["concurrent"]) as pool:
            futures = [
                pool.submit(client.completion, p, spec["max_tokens"]) for p in prompts
            ]
            for fut in futures:
                try:
                    fut.result()
                    sent += 1
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    print(f"  request failed: {exc!r}", file=sys.stderr)
    return {"phase": spec["phase"], "sent": sent, "errors": errors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--burst", type=int, default=4, help="concurrent long prompts")
    ap.add_argument("--drain", type=int, default=4, help="concurrent generations")
    ap.add_argument("--drain-tokens", type=int, default=512)
    ap.add_argument(
        "--burst-tokens",
        type=int,
        default=16,
        help=(
            "generation length of the burst arm. Raise it with --burst to "
            "drive HELD TOKENS up: occupancy is what the admissibility "
            "interlock reads, and short generations never move it."
        ),
    )
    ap.add_argument("--mixed", type=int, default=6)
    ap.add_argument("--idle-s", type=float, default=45.0)
    ap.add_argument(
        "--repeats",
        type=int,
        default=2,
        help=(
            "how many times to walk the phase list. Two is the minimum that "
            "produces a regime RETURN, which is what the hysteresis and dwell "
            "interlocks are judged on."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    phases = plan(args)
    print(f"#363 gate workload: {args.repeats} cycle(s) over {len(phases)} phases")
    for spec in phases:
        print(
            f"  {spec['phase']:<14} conc={spec['concurrent']:<3} "
            f"prompt~{spec['prompt_tokens']:<6} max_tokens={spec['max_tokens']:<5} "
            f"-- {spec['why']}"
        )
    if args.dry_run:
        print("\ndry run: no socket opened, no request sent.")
        # One prompt is materialised so a length bug is caught at the desk
        # rather than after the window opens.
        sample = _prompt(BURST_PROMPT_TOKENS, "smoke")
        print(f"sample burst prompt: {len(sample)} chars")
        return 0

    client = Client(args.base, args.model)
    if not client.alive():
        print(f"no server at {args.base} (GET /health failed)", file=sys.stderr)
        return 2

    results = []
    for cycle in range(args.repeats):
        print(f"\n--- cycle {cycle + 1}/{args.repeats} ---")
        for spec in phases:
            t0 = time.time()
            res = run_phase(client, spec, repeats=1)
            res["cycle"] = cycle
            res["seconds"] = round(time.time() - t0, 1)
            results.append(res)
            print(
                f"  {res['phase']:<14} sent={res['sent']:<3} "
                f"errors={res['errors']:<3} {res['seconds']}s"
            )
    failed = sum(r["errors"] for r in results)
    print(f"\ndone: {sum(r['sent'] for r in results)} sent, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
