# ANALYSE 734 — the three `sendBytes` specimens

Desk-only, read-only on logs. Verdict and fix SHAPE; no fix built (the boot
wrapper is F4-r4's, and the operator's ack is required before touching it).

Successor to `e66bde7834` ("A dead peer is not a slow peer"), which corrected
how this failure is REPORTED. It did not claim to say why the peer dies. That
is this note.


## 0. Verdict up front

**None of the three specimens is a boot-cycling failure.** All three are
phase (d) POST-READY SERVING crashes. The servers had been live and serving
for roughly 2 minutes, 1 minute (past a real `200 OK`), and 11 minutes
respectively before dying. A crash eleven minutes into serving cannot be a
rendezvous collision with a predecessor, so the premise of the task's framing
does not survive contact with the logs.

**The TCPStore-cycling hypothesis therefore fails twice over** — once on
structure (section 2: the store port is drawn fresh from the free pool every
boot) and once on evidence (section 3: no specimen died at rendezvous, and no
specimen shows a colliding predecessor).

**THREE specimens, TWO roots.**

| # | when | root | phase | uptime |
|---|---|---|---|---|
| 1 | 2026-08-17 18:23:43 | host-memory OOM event | (d) post-ready | ~2 min |
| 2 | 2026-08-17 19:31:45 rank2 | barlink-BAR1 CUDA illegal memory access | (d) post-ready | ~1 min |
| 3 | 2026-08-05 20:59:42 rank2 | barlink-BAR1 spin-kernel abort path | (d) post-ready | ~11 min |

Specimens 2 and 3 share a family, across **different models and different
parallelism** (3 is pure `tp_size=3, pp_size=1` on Qwen3.6; 2 is `pp_size=3` on
Qwen3.8). That makes barlink-BAR1 the recurring root, not a one-off.

**`sendBytes` is a tombstone, not a cause.** In all three it fires from
`ProcessGroupNCCL::HeartbeatMonitor::runLoop()` SECONDS AFTER another rank has
already died. Chasing the socket is chasing the corpse's last word.

**Correction to my own fix recommendation, made before the logs were read:**
the `wait_host_release` gap in section 4 is real, but it closes a LATENT hole
and fixes **zero of the three specimens**. It must not be shipped as "the #734
fix". Section 5 says so plainly.

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


## 3. The three specimens

Boot-phase classification and root traces from a full read of the three logs.
Every one is phase **(d) post-ready serving**.

A note on what the logs do NOT contain: a targeted search of all three files
for `Address already in use`, `EADDRINUSE`, `stale`, `orphan`, `still alive`,
`SIGKILL`/`SIGTERM` teardown language returned **zero real matches** (the single
`stale` hit in specimen 3 is a substring inside the `server_args` dump). There
is no colliding-predecessor evidence in any specimen.

### Specimen 1 — 2026-08-17 18:23:43 — host-memory OOM

`/spinning/evidence-665-f1/CRASH_restore_706450.log` (1538 lines; a
byte-identical copy is at `boot_bundle.log.20260817T182532Z`).

    L1433  [rank2] recvValue failed on SocketImpl(fd=72,
           addr=[localhost]:38710, remote=[::ffff:0.0.0.0]:38409):
           Connection reset by peer                       18:23:43.962302
    L1516  Exception raised from sendBytes at ...c10d/Utils.hpp:653
    tail   No live scheduler processes found; skipping py-spy

Proximate frame: a **gloo** `all_reduce` inside
`Scheduler._update_uniform_pool_budget` -> `work.wait()` raising
`Connection closed by peer` (gloo `pair.cc:547`). Rank 0 — the TCPStore host —
went silent first; ranks 1/2 then lost gloo, then lost the store.

**The timing is the finding.** `/var/log/syslog:1789` records

    2026-08-17T18:23:43.842677  claude.service:
      A process of this unit has been killed by the OOM killer.

and rank2's first store failure is at `18:23:43.962302`. **120 milliseconds
later.** The machine was inside a kernel OOM event at the instant rank 0 went
quiet.

Store port **38409**, PID **706450**, `pp_size=3`,
`nccl_port=None`, `dist_init_addr=None`. Healthy decode loop logged
continuously 18:21:22 -> 18:23:39, i.e. ~2 minutes of live serving first.

The same OOM signature repeats at `18:39:05` — two seconds before
`e66bde7834`'s own measurement at `18:39:07`, which F4-r4 described as "a boot
killed by an external process exit". That is what an OOM kill of a peer looks
like from inside a survivor. A third OOM event sits at `18:45:30`.

### Specimen 2 — 2026-08-17 19:31:45 rank2 — barlink-BAR1 CUDA fault

`/spinning/evidence-665-f1/boot_bundle.log.20260817T193224Z` (50874 lines).

    L50862 [rank2] sendBytes failed on SocketImpl(fd=71,
           addr=[localhost]:46108, remote=[::ffff:0.0.0.0]:35019):
           Broken pipe                                    19:31:45.560596

Root, on **rank 0**, during a live `tp_to_pp` PHASE FLIP: four repeated
`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
in `barlink_bar1.py poll_status_word` (via `barlink_abort_gate.py:351`),
immediately followed by the same fault on the flip payload path:

    phase_flip_runtime.py:6926 _execute
      -> _pack_outgoing (:6286)
        -> kv_reshard.py:359 _checksum
          -> weights_arena.py:127 uint8_checksum   (torch.stack(parts).sum())

Ranks 1/2 began cascading at 19:30:41 and spun for ~65 s; the requested
19:31:45 line is near the END of that cascade, not its start.

**This specimen is emphatically not memory.** `[HOST-LEDGER 92552585cc/post]
19:32:24Z posts=0.0 avail=103 headroom=97.0` — 103 GiB available 39 seconds
later, and no OOM line anywhere in the 19:3x window.

Store port **35019**, PID **910785**. `The server is fired up and ready to
roll!` at 19:29:24, a served `POST /v1/messages ... 200 OK` immediately before
the fault.

### Specimen 3 — 2026-08-05 20:59:42 rank2 — barlink-BAR1 abort path

`/spinning/CRASH_20260805_boot9_2059.log` (1220 lines) — top-level
`/spinning`, NOT in the evidence dir. Established as the earliest occurrence by
a date-descending survey of `Utils.hpp:653` across the evidence tree; every
earlier file checked returned zero.

    L826  [rank2] sendBytes failed on SocketImpl(fd=63,
          addr=[localhost]:36750, remote=[localhost]:36521): Broken pipe
    L827  Exception raised from sendBytes at ...c10d/Utils.hpp:653

Root: `Bar1CollectiveAborted: barlink-BAR1 rank 1/3 group tp:0: a spin kernel
took its abort path, observed at all_reduce`, from
`barlink_bar1.py:4561 check_aborted` via `barlink.py:919 _after_transport` ->
`all_reduce` -> `tensor_model_parallel_all_reduce` -> model forward
(`qwen2_moe.py:292 down_proj`). Ranks 0 and 1 raise `Bar1CollectiveAborted`
directly and exit before they can log a store failure; only rank 2 survives long
enough to produce the `sendBytes` line.

Different configuration entirely: **`tp_size=3, pp_size=1`** (pure TP) on
**Qwen3.6-27B-INT8-W8A8**, against specimens 1-2's `pp_size=3` Qwen3.8. Store
port **36521**, PID **48566**. Ready at 20:48:33, fault at 20:59:38 — **~11
minutes** of live serving.

That a different model on a different parallelism produces the same barlink-BAR1
family is what makes specimens 2 and 3 a recurring root rather than two
accidents.

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

**First, the correction.** Before reading the logs I recommended gating
`wait_host_release` on the surviving-scheduler count. Section 4's hole is real
and worth closing, but now that all three specimens are known to be post-ready
serving crashes, that change **fixes none of them**. It closes a latent
boot-race hole. Shipping it under the #734 banner would let the ticket close
while all three roots stayed live — which is the failure mode this note exists
to prevent. Two separate items:

### 5a. The #734 roots

**Specimens 2 and 3 (barlink-BAR1) are the priority**: two incidents twelve
days apart, on different models and different parallelism, in the same
subsystem. The concrete lead from specimen 2 is a CUDA illegal memory access
reached twice — first in `barlink_bar1.py poll_status_word` via
`barlink_abort_gate.py:351`, then on the phase-flip payload path through
`kv_reshard.py:359 _checksum` -> `weights_arena.py:127 uint8_checksum`. Whether
the abort gate's poll is a victim of an earlier corruption or its cause is not
established here, and the ordering in the log (gate first, payload second)
does not settle it: the gate polls continuously, so it would see any
pre-existing fault first regardless.

Specimen 3's `Bar1CollectiveAborted` ("a spin kernel took its abort path") is
the same subsystem reporting an abort rather than faulting outright. Whether
these are one defect or two is the first question for whoever takes it.

**Specimen 1 (host-memory OOM)** belongs to the RAM family already being worked
under #737/#738 and needs no separate fix here — the 120 ms correlation is
strong, but which process the killer took is not established (section 6), so
this note classifies it rather than closing it.

### 5b. The latent boot-race hole (separate ticket, not #734)

In `wait_host_release.sh`, make the clear-to-boot condition require `S -eq 0`
as well as `A -ge NEED`:

    [ "$A" -ge "$NEED" ] && [ "$S" -eq 0 ] && { echo "... clear to boot"; exit 0; }

reporting `S` in the timeout branch as the reason. This closes the
predecessor-still-alive window for host RAM, GPU memory and the store socket at
once, without a bespoke port probe. Two cautions for the owner:

1. `grep -c "sglang::schedul"` reads `ps -eo comm`, truncated at 15 characters.
   `sglang::schedul` is exactly 15, so the match is correct today and is one
   rename away from silently counting zero. A miscounting gate that reports
   "clear" is worse than no gate. Pin it with a test, or match the full cmdline.
2. Requiring zero turns a soft wait into a hard one. If a scheduler ever fails
   to exit, boots stop rather than race. That is the right direction, but it
   must fail with the PIDs named rather than timing out anonymously.

**Not recommended in either item:** extending any gate to the store port. It
guards a mode this path cannot have (section 2) and no specimen exhibits, and
it would encode the refuted hypothesis where the next reader would trust it.


## 6. What is still open

- **The barlink-BAR1 root** (specimens 2 and 3). Classified, not solved. One
  defect or two is unanswered.
- **Which process the OOM killer took at 18:23:43.** `dmesg` is
  permission-denied from this session and `journalctl -k` returns "No entries",
  so kernel OOM detail is **unavailable here, not absent**. The systemd line
  names `claude.service`; the serving tree runs under `setsid` outside systemd,
  so a kill inside it produces no unit line and the absence of an sglang line is
  not evidence it survived. The 120 ms gap is strong correlation, not proof of
  which victim.
- **Whether specimen 1's gloo disconnect has a cause other than the OOM.** The
  proximate frame is `_update_uniform_pool_budget`; I did not rule out that the
  collective itself is implicated rather than merely being where the survivors
  were standing.
- **The task's framing.** "Rapid boot cycling" does not describe any of the
  three. If there IS a boot-cycling specimen, it is not among these, and the
  search that found these three (a `Utils.hpp:653` survey across the evidence
  tree, earliest hit 2026-08-05) did not surface one.
