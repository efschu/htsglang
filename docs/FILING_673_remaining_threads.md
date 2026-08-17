# #673 teardown family — what remains after the desk work

**2026-08-17, Slot-3. Filing only: nothing below was changed.**

With the lockstep-sentinel fix, the desk work on this family is complete. Of
the six abort-shaped items (candidate 1 plus the five inventory rows in
`evidence-665-f1/DIAG_673_TEARDOWN_ABORT.md:206-212`), **four are addressed and
two remain**, and neither of the two is mine to take.

## Addressed

| item | fix | gating |
|---|---|---|
| process groups never destroyed (candidate 1) | `5436fcb99a` | **gated, default OFF** (touches barlink, #722's lane) |
| `kvso-dest-io` | `4334c642d4` | ungated |
| `dual-group-lane-{id}` | `91adb7c156` | ungated |
| `lockstep-sentinel` | this branch | ungated |

All three of mine share one shape, deliberately: bounded join, handle kept on
timeout, explicit WARNING naming the thread, a component-level stop next to the
thing that built it, and an AST pin that the call site exists — because an
orphaned stop method is the defect this family keeps producing.

---

## FILED 1 — `barlink-peer-watchdog` (highest-value remaining)

`distributed/device_communicators/barlink_liveness.py:592`, `stop()` at `:598`,
**called ONLY from `reset_for_test()`**. Body is a four-byte D2H poll on the
transport's private stream at roughly **10 ms cadence**.

**Ranking: this is the highest-value item left, above the one I just fixed.**
The inventory ranks it third overall and says why plainly — on cadence alone it
is the most likely of any of these to be mid-CUDA-call at any given instant.
A ~10 ms poll is two orders of magnitude more exposed than the lockstep
sentinel's 0.5 s, and unlike the sentinel it is not opt-in.

**Why it is still not taken here.** It is #722's machinery, and the earlier
decision was that it stays with that lane. Nothing in this branch touched a
barlink file.

**A note for whoever picks it up, because the ground has moved.** #722 *as
scoped* was retracted and barlink was declared innocent of the defect it was
opened for. That changes the risk calculus for this specific change: the reason
the process-group destroy ships gated is that `GroupCoordinator.destroy` closes
`barlink_comm`, unlinking a POSIX shm segment and the device-mapped abort word
that spinning kernels read. **A pure teardown-stop caller for the watchdog does
none of that** — it stops a polling thread, it does not close the transport or
unlink memory. So the gating argument that applies to candidate 1 does not
obviously apply here, and a bounded-join stop in the family shape is likely
safe. That is a reading of the constraint, not a clearance: the lane that owns
barlink should confirm it before wiring, and the ordering caution below applies.

**Ordering caution, same class as the sentinel's.** Anything that stops a
thread using a transport must complete *before* that transport is closed. If a
watchdog stop is ever added alongside the gated destroy, it belongs on the same
side of the ordering as `release_lockstep_sentinel` — stopped and **joined**
first. The sentinel fix enforces its ordering inside `release_distributed`
rather than by call order for exactly this reason, and the same seam is
available.

---

## FILED 2 — mooncake / mori / nixl transfer workers (no action)

`disaggregation/{mooncake,mori,nixl}/conn.py`, several threads (transfer,
bootstrap, decode workers). **No `.join(` anywhere in those files**, bodies are
vendored C++ transfer engines.

**Filed as-is, no action proposed.** Three reasons, in order:

1. They are vendored engines, so the abort surface is not ours and a stop path
   would have to be designed against the vendor's lifecycle rather than added
   to it.
2. They only exist under PD disaggregation, which is not the specimens'
   configuration.
3. Unlike the four addressed items, there is no orphaned stop method to wire —
   there is no stop mechanism at all, so this is a design task, not a caller
   fix.

If the abort survives everything above on a PD boot, this is where to look
next; on a non-PD boot it cannot be the cause.

---

## Also recorded, not abort-shaped

`MooncakeFailedSessionProbe`'s shutdown `Event` and
`embedding_cache_controller.io_thread`'s `stop_event` are both created and
never `.set()`. Neither has a C/C++ body, so neither is in the abort class;
they are ordinary leaks and are noted only so the inventory stays complete.

---

## Status

**Desk work on the #673 teardown family is complete, pending metal proof.**
None of the four fixes has been validated on a boot. The abort is intermittent
per process, so absence in any single boot is not proof; what a validating boot
shows is the specific "joined before exit" line for each fix and no
`terminate called without an active exception` after a clean drain.
