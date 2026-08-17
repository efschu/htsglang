# ANALYSE 734 — the three `sendBytes` specimens

Desk-only, read-only on logs. Verdict and fix SHAPE; no fix built (the boot
wrapper is F4-r4's, and the operator's ack is required before touching it).

Successor to `e66bde7834` ("A dead peer is not a slow peer"), which corrected
how this failure is REPORTED. It did not claim to say why the peer dies. That
is this note.


## 0. Verdict up front

**The TCPStore-cycling hypothesis does not hold, and it is refused on
structure rather than on absence of evidence.** The store port cannot collide
across boots because it is drawn fresh from the free-port pool on every boot
and then explicitly waited on. Section 2 gives the file:line.

**There are at least TWO roots, not one.** The specimens split on the host
ledger: one dies inside a kernel OOM event to the second, another dies with
103 GiB available. A single-root story cannot cover both.

**`sendBytes` is a tombstone, not a cause.** In every specimen on record —
including the two that predate this task — the sequence is: a peer PROCESS
disappears, and the survivors then fail writing to it. Chasing the socket is
chasing the corpse's last word.

The one real defect found in the boot path is unrelated to the store port and
is stated in section 4: `wait_host_release.sh` already computes the number of
surviving schedulers and then throws the number away.


## 1. Prior art — this symptom has been seen twice, and both times it was secondary

Neither is a #734 specimen; both are the same signature, and they set the
prior.

`docs/dev/631/HANDOFF_663.md:696-698` — PP2 dies in `recv_object` -> `work.wait()`
with `RuntimeError: Connection closed by peer`, **then** `TCPStore sendBytes
failed ... Broken pipe`. The note's own reading:

> Two ranks blocked in a receive from a third. The third left no traceback,
> which is exactly what a SIGKILL looks like. **This is NOT the corridor and
> NOT VRAM** [...] **Every capacity discussion in these handoffs has been
> about VRAM; this run died of host RAM.**

`docs/dev/631/HANDOFF_658.md:256` — "Rank 0's TCPStore died; ranks 1/2 spun on
`sendBytes ... Broken pipe`", downstream of rank 0 being lost to a runaway
allocation.

`e66bde7834` measured the same shape on 2026-08-17 18:39:07 and named it: the
underlying frame was `gloo ... Connection closed by peer [127.0.1.1]`, i.e.
"the peer PROCESS HAD DIED". It fixed the misattribution to timeout. The death
itself was left open.

So in 3 of 3 recorded instances the socket error is the SECOND event. Any
verdict that makes the socket primary is fighting the prior.


## 2. The cycling hypothesis, refused on structure

The filed suspicion (OPERATOR-STATE 2026-08-17 ~19:4xZ) is that an old
TCPStore is still alive when the successor binds, so the successor either
fails to bind or attaches to the predecessor's store and is orphaned when it
finally dies.

That cannot happen on this boot path:

- The store port is **not pinned**. `scripts/route_a_631_prod_boot.sh` passes
  neither `--nccl-port` nor `--dist-init-addr` (checked; no match).
- Therefore `server_args.py:18787-18788` applies:
  `if server_args.nccl_port is None: nccl_port = get_free_port()`. Every boot
  draws a port that is free AT THAT MOMENT, so a predecessor still holding its
  own store port is simply not a candidate.
- Even a lost race is caught: `server_args.py:18873` calls
  `wait_port_available(nccl_port, "nccl_port")`, which polls for 30 s and,
  past 10 s, calls `find_process_using_port` and names the holder. A collision
  would therefore surface as a NAMED port-in-use refusal at startup, not as a
  mid-run `Connection closed by peer`.

There is also a semantic argument, independent of this codebase.
`Connection closed by peer` is a connection that was ESTABLISHED and then
broken. A stale predecessor holding a port produces `EADDRINUSE` at bind, or a
refused connect — not a mid-stream close. The observed error is the wrong shape
for the hypothesis.

**Answer to "does `wait_host_release` check the store port?": no — and it
should not need to.** Extending it to the store port would be a guard against a
failure mode this path structurally cannot have. Section 4 has the check that
IS missing.


## 3. The specimens split — evidence of at least two roots

`/var/log/syslog` is the only OOM source readable from this session; `dmesg`
returns `read kernel buffer failed: Operation not permitted` and
`journalctl -k` returns "No entries". **So kernel-level OOM detail is
UNAVAILABLE here, not absent** — every statement below rests on the systemd
unit lines in syslog, which report only that the OOM killer fired and which
unit lost a process.

### Specimen A — 18:23:43: inside a kernel OOM event, to the second

    /var/log/syslog:1789
    2026-08-17T18:23:43.842677 CT999 systemd[1]:
      claude.service: A process of this unit has been killed by the OOM killer.

The specimen timestamp is not near this event, it IS this second. The unit
named is `claude.service` (agent processes), and the serving tree is launched
via `setsid` outside systemd, so a kill inside it produces no unit line at all
— absence of an sglang line is therefore not evidence the serving tree
survived. What the line does establish is that the machine was in a kernel OOM
event at the instant of the crash.

The same signature repeats at `18:39:05` (syslog) against `e66bde7834`'s
measurement at `18:39:07` — F4-r4 described that boot as "killed by an external
process exit", which is what an OOM kill of a peer looks like from inside a
survivor. A third OOM event sits at `18:45:30`.

### Specimen B — 19:31:45 rank2: NOT memory

    [HOST-LEDGER 92552585cc/post] 19:32:24Z posts=0.0 avail=103 headroom=97.0

39 seconds after the specimen the host had **103 GiB available, 97 GiB of
headroom**. There is no OOM line anywhere in the 19:3x window. Whatever killed
this one, host-memory pressure did not.

The boot log for it is `boot_bundle.log.20260817T193224Z` (9.5 MB, mtime
19:31).

### Consequence

One root cannot produce both A and B. Specimen A belongs to the host-RAM
family the two 631 handoffs already describe. Specimen B needs its own
classification against `#580` (gloo mismatch) or `#673`/`#693`
(teardown-terminate) — see section 6 for what is still open.


## 4. The real hole in the boot path — a computed counter nobody gates on

`/spinning/evidence-665-f1/wait_host_release.sh` (#721):

    NEED=${1:-90}          # GiB available before a boot may start
    while :; do
      A=$(free -g | awk 'NR==2{print $7}')
      S=$(ps -eo rss,comm 2>/dev/null | grep -c "sglang::schedul")
      [ "$A" -ge "$NEED" ] && { echo "... schedulers=$S -- clear to boot"; exit 0; }

`S` — the number of surviving schedulers — is computed on every iteration,
printed in the clear-to-boot line, and **never appears in a condition**. The
gate is `A -ge NEED` and nothing else. A predecessor whose schedulers are still
alive but whose big allocations have already been unmapped satisfies the gate
and the boot proceeds into a live predecessor.

This is the counter-without-a-reachable-actuator shape (#679/#681/#684/#715):
the value that would answer the question is already in hand and is discarded
one line before the decision.

The single-instance guard does not cover it either.
`scripts/route_a_631_prod_boot.sh:250`:

    if pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null 2>&1; then

That matches the **launcher**. The store and the ranks live in the
`sglang::scheduler` children. An orphaned scheduler set whose launcher has
already exited passes this guard cleanly — which is precisely the state
`S` would report and the gate ignores.

So the boot path checks: RAM (yes), launcher liveness (yes), scheduler
liveness (computed, discarded).


## 5. Fix shape — NOT built, pending ack

Recommended, and deliberately smaller than the filed candidate:

**Gate on the counter that already exists.** In `wait_host_release.sh`, make
the clear-to-boot condition require `S -eq 0` as well as `A -ge NEED`:

    [ "$A" -ge "$NEED" ] && [ "$S" -eq 0 ] && { echo "... clear to boot"; exit 0; }

and report `S` in the timeout branch as the reason. This closes the
"predecessor still alive" window for every resource it holds at once — host
RAM, GPU memory, and its store socket — without adding a bespoke port probe
for a collision that section 2 shows cannot occur.

Two cautions on that one-liner, both of which the owner should decide:

1. `grep -c "sglang::schedul"` reads `ps -eo comm`, which truncates at 15
   characters; `sglang::schedul` is exactly 15, so the match is right today but
   is one rename away from silently counting zero. A miscounting gate that
   reports "clear" is worse than no gate. Pin it with a test, or match on
   `-f`/full cmdline instead.
2. Requiring `S -eq 0` turns a soft wait into a hard one. If a scheduler ever
   fails to exit, boots stop rather than race. That is the correct direction —
   but it must fail with the PIDs named, not by timing out anonymously.

**Not recommended:** extending the gate to the store port. It guards a failure
mode this path cannot have (section 2), and it would encode the refuted
hypothesis into the boot wrapper where the next reader would trust it.

**Specimen B is not addressed by any of this** and must not be closed by it.


## 6. What is still open

- **Specimen B's root.** Not memory. Needs the boot-phase classification from
  `boot_bundle.log.20260817T193224Z` and a decision against `#580` /
  `#673`/`#693`. Filed, not answered here.
- **The third specimen.** The task named "one earlier". The 18:39:05 and
  18:45:30 OOM events are candidates, and 18:39 is already tied to
  `e66bde7834`'s measurement — but tying a timestamp to a specimen is not the
  same as reading its log, and I have not done the second.
- **Kernel OOM detail** is unreadable from this session (`dmesg` denied,
  `journalctl -k` empty). Which PROCESS the OOM killer took at 18:23:43 is
  therefore not established — only that it fired. If that matters to the fix,
  it needs a session that can read the kernel buffer.
- **Whether specimen A's serving tree was the victim at all** follows from the
  same gap. The correlation is to the second and the prior art is consistent,
  but "the OOM killer fired at that instant" is not "it took this rank".
