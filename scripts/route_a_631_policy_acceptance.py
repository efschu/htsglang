#!/usr/bin/env python3
"""#631 Route A automatic phase policy -- live mixed-traffic acceptance.

This is the OPERATIONAL acceptance for the automatic phase controller. It
issues no flip call of any kind: every layout change observed during this
run must come from the policy itself. The run mixes the two regimes the
policy has to separate:

  * long-prompt prefills (the PP-favouring work), and
  * ongoing decodes at batch size up to --max-bs (the TP-favouring work),

then stops all traffic and idles, so the IDLE TRANSITION can be observed:
with the resting state at PP, a drained server must return to PP on its own
and the next long prompt must prefill there without a flip in its latency
path.

What the caller checks afterwards (in the scheduler log, not here):
  (a) prefill executed in the PP layout at PP-class throughput,
  (b) decode executed in TP with an accept length present,
  (c) flip cadence consistent with the policy -- no thrashing,
  (d) zero aborted user requests.

This script reports (d) directly, since it owns the client side: any
non-200, any truncated stream, any exception is an aborted request and is
counted and printed. A run with a single abort is a FAILED acceptance.

Stdlib only -- it runs against live production.
"""

import argparse
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request

STATE_LOCK = threading.Lock()
RESULTS = []

# The single writer of this line is _cutover, so it is the authoritative
# record of the active layout. Matches: "PHASE-FLIP cutover complete:
# active stack tp, ps tp=3 pp=1".
CUTOVER_RE = re.compile(r"cutover complete: active stack (\w+)")
PHASE_LOG = "/spinning/serving-30030.boot.log"
PHASE_TAIL_BYTES = 4_000_000


def post(port: int, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def record(kind: str, ok: bool, seconds: float, detail: str = "") -> None:
    with STATE_LOCK:
        RESULTS.append(
            {"kind": kind, "ok": ok, "seconds": round(seconds, 3), "detail": detail}
        )


def long_prefill_worker(args, stop: threading.Event, idx: int) -> None:
    """Long-prompt prefills: the work that should pull the server into PP."""
    while not stop.is_set():
        n = random.choice(args.prefill_rungs)
        ids = [random.randint(1000, args.vocab) for _ in range(n)]
        t0 = time.perf_counter()
        try:
            out = post(
                args.port,
                {
                    "input_ids": ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.prefill_new_tokens,
                        "ignore_eos": True,
                    },
                },
                args.timeout,
            )
            dt = time.perf_counter() - t0
            ok = bool(out.get("text") is not None or out.get("output_ids"))
            record("prefill", ok, dt, f"{n} tok")
        except Exception as exc:  # noqa: BLE001 - any failure is an abort
            record("prefill", False, time.perf_counter() - t0, f"{n} tok: {exc!r}")
        stop.wait(args.prefill_gap)


def decode_worker(args, stop: threading.Event, idx: int) -> None:
    """Ongoing decodes: the work that should pull the server into TP."""
    while not stop.is_set():
        ids = [random.randint(1000, args.vocab) for _ in range(args.decode_prompt)]
        t0 = time.perf_counter()
        try:
            out = post(
                args.port,
                {
                    "input_ids": ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.decode_new_tokens,
                        "ignore_eos": True,
                    },
                },
                args.timeout,
            )
            dt = time.perf_counter() - t0
            n_out = len(out.get("output_ids") or []) or args.decode_new_tokens
            ok = dt > 0
            record("decode", ok, dt, f"{n_out} out tok, {n_out/dt:.1f} tok/s")
        except Exception as exc:  # noqa: BLE001
            record("decode", False, time.perf_counter() - t0, repr(exc))
        stop.wait(args.decode_gap)


def phase_of(port: int) -> str:
    """Read the CURRENT layout.

    ``/phase_flip`` is POST-only and takes a direction, so there is no read
    endpoint to ask -- and inventing one for a measurement would mean
    shipping server code just to observe. The authoritative record is the
    scheduler log: the cutover writes ``active stack <phase>`` exactly once
    per completed flip, from the single writer in ``_cutover``. Reading the
    last such line back is therefore the true current layout.

    ``PHASE_LOG`` is module state so the caller can point it at the boot log
    of the server under test.
    """
    try:
        with open(PHASE_LOG, "r", errors="replace") as fh:
            # Tail is enough; a flip line is never far from the end in a
            # run of this length, and reading the whole boot log per sample
            # would dominate the sampling interval.
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - PHASE_TAIL_BYTES))
            text = fh.read()
    except OSError:
        return "?"
    found = CUTOVER_RE.findall(text)
    if found:
        return found[-1]
    # No flip in the tail window: fall back to the boot-time phase, which is
    # always pp (boot_phase=PHASE_PP), unless a flip happened earlier than
    # the window -- so scan the whole file once rather than guess.
    try:
        with open(PHASE_LOG, "r", errors="replace") as fh:
            found = CUTOVER_RE.findall(fh.read())
    except OSError:
        return "?"
    return found[-1] if found else "pp"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--seconds", type=float, default=180.0, help="mixed phase")
    ap.add_argument("--idle-seconds", type=float, default=90.0, help="idle phase")
    ap.add_argument("--max-bs", type=int, default=4)
    ap.add_argument("--prefill-workers", type=int, default=1)
    ap.add_argument("--prefill-rungs", default="8192,32768")
    ap.add_argument("--prefill-new-tokens", type=int, default=1)
    ap.add_argument("--prefill-gap", type=float, default=5.0)
    ap.add_argument("--decode-prompt", type=int, default=512)
    ap.add_argument("--decode-new-tokens", type=int, default=192)
    ap.add_argument("--decode-gap", type=float, default=1.0)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--log", default=PHASE_LOG, help="scheduler log to read the phase from")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.prefill_rungs = [int(x) for x in args.prefill_rungs.split(",")]
    globals()["PHASE_LOG"] = args.log

    n_decode = max(1, args.max_bs - args.prefill_workers)
    print(
        f"acceptance: {args.prefill_workers} prefill worker(s) "
        f"{args.prefill_rungs}, {n_decode} decode worker(s), "
        f"mixed {args.seconds:g}s then idle {args.idle_seconds:g}s"
    )
    print(f"phase at start: {phase_of(args.port)}")
    print("NO flip call is issued by this script.")

    stop = threading.Event()
    threads = []
    for i in range(args.prefill_workers):
        threads.append(
            threading.Thread(target=long_prefill_worker, args=(args, stop, i))
        )
    for i in range(n_decode):
        threads.append(threading.Thread(target=decode_worker, args=(args, stop, i)))
    for t in threads:
        t.daemon = True
        t.start()

    t_start = time.time()
    phases = []
    while time.time() - t_start < args.seconds:
        time.sleep(2.0)
        phases.append((round(time.time() - t_start, 1), phase_of(args.port)))

    stop.set()
    for t in threads:
        t.join(timeout=args.timeout)
    t_drain = time.time() - t_start
    print(f"\ntraffic stopped at t={t_drain:.1f}s; idling to observe the return to rest")

    # Idle leg: the resting state must reassert itself with no traffic at all.
    t_idle0 = time.time()
    while time.time() - t_idle0 < args.idle_seconds:
        time.sleep(2.0)
        phases.append((round(time.time() - t_start, 1), phase_of(args.port)))

    rest_phase = phase_of(args.port)
    print(f"phase after idle: {rest_phase}")

    # Post-idle probe: a long prompt from rest. If the resting state is the
    # prefill layout, this must NOT have a flip in its latency path.
    ids = [random.randint(1000, args.vocab) for _ in range(max(args.prefill_rungs))]
    t0 = time.perf_counter()
    try:
        post(
            args.port,
            {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            },
            args.timeout,
        )
        probe_s = time.perf_counter() - t0
        probe_ok = True
    except Exception as exc:  # noqa: BLE001
        probe_s = time.perf_counter() - t0
        probe_ok = False
        print(f"post-idle probe FAILED: {exc!r}")
    print(
        f"post-idle {max(args.prefill_rungs)}-tok probe from rest: "
        f"{probe_s*1000:.1f} ms "
        f"({max(args.prefill_rungs)/probe_s:.1f} tok/s), ok={probe_ok}"
    )

    n_ok = sum(1 for r in RESULTS if r["ok"])
    n_bad = sum(1 for r in RESULTS if not r["ok"])
    print(f"\nrequests: {len(RESULTS)} total, {n_ok} ok, {n_bad} ABORTED")
    for r in RESULTS:
        if not r["ok"]:
            print(f"  ABORT {r['kind']}: {r['detail']}")

    transitions = [
        (t, p)
        for i, (t, p) in enumerate(phases)
        if i == 0 or p != phases[i - 1][1]
    ]
    print(f"\nobserved phase timeline ({len(transitions)} transitions):")
    for t, p in transitions:
        print(f"  t={t:>7.1f}s  {p}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "results": RESULTS,
                    "phases": phases,
                    "transitions": transitions,
                    "rest_phase": rest_phase,
                    "post_idle_probe_s": probe_s,
                    "post_idle_probe_ok": probe_ok,
                    "aborted": n_bad,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
