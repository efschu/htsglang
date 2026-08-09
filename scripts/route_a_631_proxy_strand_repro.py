#!/usr/bin/env python3
"""#631: the on-demand reproducer for the PP proxy STRAND, and its verdict.

WHAT IT REPRODUCES. A flip abandon is RANK-LOCAL -- each rank times out on
its own clock, so the ranks disarm at different instants. The first rank to
disarm resumes launching and sends its proxy hidden states; its downstream
is still armed and withholding, so it has no ``cur_batch`` and never makes
the matching receive. The message strands in ``_pp_tensor_dict_inbox``.
Under the ORIGINAL purely-positional pairing, every later receive on that
rank was then off by one -- silently, for the rest of the loop's life --
until a width mismatch surfaced 30 layers deep in a GDN kernel as an
out-of-bounds write (specimen pp_proxy_mispair_20260809T0626Z).

HOW IT FORCES THE STRAND. Three ingredients, all required:

  1. a SHORT park deadline on the server (SGLANG_PHASE_FLIP_PARK_DEADLINE_S,
     5 s in the recorded runs) so an armed flip gives up quickly;
  2. sustained decode traffic, so a request is always resident and every
     ``pp_to_tp`` arm therefore PARKS and then ABANDONS rather than
     committing -- a commit re-enters ``event_loop_pp`` and
     ``init_pp_loop_state`` wipes every buffer, which is exactly why a
     committing flip never showed this defect;
  3. an arm every few seconds, which buys ~12 abandons a minute instead of
     the ~2 an unassisted policy produces.

THE VERDICT IT PRINTS. With the stamp in place a leftover matches nothing
and is dropped LOUDLY, so the expected signature is:

    abandons > 0                  the reproducer actually reproduced
    shape-check ValueErrors == 0  no mispair reached model compute
    server still healthy          no wedge, no fault
    aborts == 0                   no user request paid for it

``PROXY LEFTOVER DROPPED`` lines are reported but are NOT themselves a pass
condition in either direction: a drop is the fix WORKING, and zero drops in
a run with abandons means the strand did not occur on this schedule rather
than that the fix is inert. The pass condition is the absence of the
mispair, not the presence of drops.

RUN IT AS A CONTROL TOO (``--cycles 0``): same load, no arms, so no
abandons -- which must yield ZERO drops. That control is what separates
"drops only leftovers" from "drops things it should have kept".

Stdlib only; it runs against a live server.
"""

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

LOCK = threading.Lock()
ABORTS = []
DECODES = []
STOP = threading.Event()


def _post(port, path, payload, timeout):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def _decode_worker(args, wid):
    """Keeps a request RESIDENT. Residency is the whole point: it is what
    makes every arm park instead of commit."""
    n = 0
    while not STOP.is_set():
        n += 1
        prompt = f"[w{wid}#{n}] Explain, step by step and at length, how a pipeline-parallel stage boundary works."
        t0 = time.time()
        try:
            status, body = _post(
                args.port,
                "/generate",
                {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": args.decode_new_tokens,
                        "temperature": 0.7,
                    },
                },
                args.timeout,
            )
            dt = time.time() - t0
            if status != 200:
                with LOCK:
                    ABORTS.append(f"w{wid}#{n} HTTP {status}")
            else:
                obj = json.loads(body)
                got = obj.get("meta_info", {}).get("completion_tokens", 0)
                with LOCK:
                    DECODES.append((got, dt))
                if got == 0:
                    with LOCK:
                        ABORTS.append(f"w{wid}#{n} empty completion")
        except Exception as exc:  # noqa: BLE001 - any failure is an abort
            with LOCK:
                ABORTS.append(f"w{wid}#{n} {type(exc).__name__}: {exc}")
            time.sleep(1.0)
        time.sleep(args.decode_gap)


def _log_counts(log, since_byte):
    """Count the signatures in the bytes this run appended, not the whole
    boot -- a boot log carries the previous investigation's lines."""
    try:
        with open(log, "rb") as fh:
            fh.seek(since_byte)
            blob = fh.read().decode("utf-8", "replace")
    except OSError:
        return {}, ""
    return (
        {
            "abandons": len(re.findall(r"FLIP ABANDONED", blob)),
            "cutovers": len(re.findall(r"cutover complete: active stack", blob)),
            "drops": len(re.findall(r"PROXY LEFTOVER DROPPED", blob)),
            # The stamp REFUSES rather than drops since corpse R. With the
            # slot hold in place this must be ZERO: a refusal now means the
            # ranks came out of an armed window on different slots, which
            # is the defect the hold exists to remove.
            "refusals": len(re.findall(r"PROXY LEFTOVER REFUSED", blob)),
            # #631 DEFECT Q, THE DIRECT VERDICT. Every armed window ends
            # with one PASS-CLOCK line per rank reporting the group's
            # resume slots. AGREED is the invariant; DIVERGED is the metal
            # defect, and it is a FAIL rather than a note.
            "slot_agreed": len(re.findall(r"RESUME SLOTS .* -- AGREED", blob)),
            "slot_diverged": len(re.findall(r"RESUME SLOTS .* -- DIVERGED", blob)),
            "gaveup": len(re.findall(r"gave up draining proxy leftovers", blob)),
            "mispairs": len(re.findall(r"#631 PP proxy/batch mismatch", blob)),
            "ima": len(re.findall(r"illegal memory access", blob)),
            "conv1d_guard": len(re.findall(r"conv_state_indices", blob)),
        },
        blob,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30030)
    p.add_argument("--cycles", type=int, default=12, help="arm/abandon cycles; 0 = CONTROL run")
    p.add_argument("--arm-period", type=float, default=7.0)
    p.add_argument("--decode-workers", type=int, default=2)
    p.add_argument("--decode-new-tokens", type=int, default=160)
    p.add_argument("--decode-gap", type=float, default=0.3)
    p.add_argument("--warmup", type=float, default=20.0, help="seconds of load before the first arm")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--log", default="/spinning/serving-30030.boot.log")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    try:
        with open(args.log, "rb") as fh:
            fh.seek(0, 2)
            start_byte = fh.tell()
    except OSError:
        start_byte = 0

    print(f"[repro] log watermark {start_byte}, cycles={args.cycles}")
    workers = [
        threading.Thread(target=_decode_worker, args=(args, i), daemon=True)
        for i in range(args.decode_workers)
    ]
    for w in workers:
        w.start()

    print(f"[repro] warming load for {args.warmup:.0f}s so a request is resident")
    time.sleep(args.warmup)

    arms_ok = 0
    arms_refused = 0
    for c in range(args.cycles):
        # Direction alternates only if a cutover happened; a parked flip
        # leaves the phase unchanged, so pp_to_tp stays legal throughout.
        try:
            status, body = _post(
                args.port, "/phase_flip", {"direction": "pp_to_tp"}, 30.0
            )
            if status == 200:
                arms_ok += 1
            else:
                arms_refused += 1
            print(f"[repro] arm {c + 1}/{args.cycles} -> {status} {body[:120]}")
        except Exception as exc:  # noqa: BLE001
            arms_refused += 1
            print(f"[repro] arm {c + 1}/{args.cycles} FAILED {type(exc).__name__}: {exc}")
        time.sleep(args.arm_period)

    print("[repro] draining load")
    STOP.set()
    for w in workers:
        w.join(timeout=args.timeout + 10)
    time.sleep(3.0)

    counts, _ = _log_counts(args.log, start_byte)

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/health", timeout=10
        ) as r:
            health = r.status
    except Exception as exc:  # noqa: BLE001
        health = f"DOWN ({type(exc).__name__})"

    with LOCK:
        aborts = list(ABORTS)
        decodes = list(DECODES)

    tok = sum(d[0] for d in decodes)
    sec = sum(d[1] for d in decodes)
    report = {
        "cycles_requested": args.cycles,
        "arms_accepted": arms_ok,
        "arms_refused": arms_refused,
        "requests_completed": len(decodes),
        "decode_tokens": tok,
        "decode_tok_s_aggregate": round(tok / sec, 2) if sec else 0.0,
        "aborts": len(aborts),
        "abort_detail": aborts[:10],
        "health_after": health,
        **counts,
    }

    print("\n=== #631 PROXY STRAND REPRODUCER ===")
    for k, v in report.items():
        print(f"  {k:28s} {v}")

    verdict = []
    if args.cycles == 0:
        verdict.append(("CONTROL: no arms issued", counts.get("abandons", 0) == 0))
        verdict.append(("CONTROL: zero leftover drops", counts.get("drops", 0) == 0))
    else:
        verdict.append(("the reproducer abandoned at least once", counts.get("abandons", 0) > 0))
    verdict.append(("no proxy/batch mispair reached compute", counts.get("mispairs", 0) == 0))
    # #631 DEFECT Q. These two are the slot hold's own pass conditions and
    # they are the reason this run means something the earlier ones did
    # not. Before the hold, a run with abandons produced refusals and
    # DIVERGED lines by construction; the old verdict could not see either,
    # so it called a phase-offset instance a pass.
    verdict.append(
        ("the ranks resumed on the SAME slot", counts.get("slot_diverged", 0) == 0)
    )
    verdict.append(
        ("no proxy was refused as a leftover", counts.get("refusals", 0) == 0)
    )
    if args.cycles > 0:
        verdict.append(
            (
                "a slot verdict was actually reported",
                counts.get("slot_agreed", 0) + counts.get("slot_diverged", 0) > 0,
            )
        )
    verdict.append(("no drain gave up", counts.get("gaveup", 0) == 0))
    verdict.append(("no illegal memory access", counts.get("ima", 0) == 0))
    verdict.append(("server still healthy", health == 200))
    verdict.append(("zero aborted requests", len(aborts) == 0))

    print()
    ok = True
    for label, passed in verdict:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"report": report, "verdict": dict(verdict), "ok": ok}, fh, indent=2)
        print(f"[repro] wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
