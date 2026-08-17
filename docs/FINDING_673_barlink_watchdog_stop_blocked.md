# #673 barlink-peer-watchdog — STOP CALLER NOT BUILT, and why

**2026-08-17, Slot-3. Analysis only. No behaviour was changed on this branch.**

I was asked to build the stop caller unless the poll thread turns out to be
load-bearing for abort DETECTION during teardown, in which case: stop and
report. **It is load-bearing.** The caller was not built.

My own earlier reading — "a pure teardown-stop caller closes nothing, so the
destroy-gating argument does not apply" — was correct about *closing* and
still wrong about *safety*. The hazard is not what a stop closes. It is what a
stop leaves unread.

---

## The finding

The watchdog is not only an observer. Since #517 phase 2 it is **the only
reader of every device transport's abort word**, and the transports have
permanently stopped reading it themselves.

The chain, at the source:

1. `barlink_device.py:1440` `_arm_status_poll` is **called once from
   bring-up**. If a watchdog is running it sets `self._abort_poll_active =
   True` (`:1478`).
2. That latch is **one-way**. The only other assignments are the class default
   (`:993`) and the constructor (`:1071`). **Nothing sets it back to False** —
   not on watchdog stop, not anywhere.
3. `poll_status_word` (`:1494`) is the sole writer of `_abort_code_seen`
   (`:1504`). Its only caller is `barlink_abort_gate.poll_status_words()`
   (`:347`), which runs **only on the watchdog thread**. The field's own
   comment says so: *"Written only by the watchdog poll (0 -> non-zero, never
   back), read by the hot path"* (`:1074-1076`).
4. `check_aborted` (`:1540`) branches on the latch:

   ```python
   if self._abort_poll_active:
       if not self._abort_code_seen:
           return          # no device read, no guard
   ```

So with the latch armed and the watchdog stopped, `_abort_code_seen` can never
change again and `check_aborted` returns "not aborted" forever. **A device
abort occurring after the stop is invisible.**

### The documented fallback does not cover this

`should_poll_status` (`barlink_abort_gate.py:358`) reads:

> A watchdog that is NOT running would leave the word unread by anyone, so the
> transport must keep the deferred hot-path read instead — the guard degrades
> to the #517-phase-1 behaviour rather than to blindness.

That is exactly the right rule, and the code honours it — **at bring-up only**.
It is evaluated once, inside `_arm_status_poll`. There is no runtime
re-evaluation, so a watchdog that stops *later* produces precisely the state
the comment says must not happen: the word unread by anyone. The fallback
protects "no watchdog at boot", not "watchdog stopped at teardown".

### Why this matters specifically during teardown

The whole #673 family exists because things abort *during shutdown*. Stopping
the watchdog on the way out would remove the guard exactly across the window
the family is about — turning a named `Bar1CollectiveAborted` into a silent
hang or an unrelated secondary failure. That is masking a real abort, which is
the condition I was told to stop on.

### `polling_paused` is not prior art for this

`polling_paused()` (`:322`) is a CUDA-graph-capture scope (`_capture_depth`,
`pause_polling` used at `parallel_state.py:2823`). It suspends polling where
the guard is skipped anyway (`graph_capture_running()`), and it likewise does
not re-arm the in-line read. There is no designed teardown pattern here.

---

## What a safe fix looks like — and why neither is "strictly the stop caller"

**Option A — stop only after the transports are gone.** `abort_gate.unregister`
is *"Called from `close()`, before the tensors go"* (`:399-400`;
`barlink_device.py:1603`, `barlink_bar1.py:5507`), and `poll_status_words`
returns 0 on an empty registry. So after every transport has closed there is
nothing left to guard and the stop is free. But that ordering hangs off
`GroupCoordinator.destroy`, which closes `barlink_comm` — and that path is
**gated default-off**. On a default boot the transports are never closed, so
this option leaves the watchdog unstopped in the common case, and placing the
stop there means touching the close ordering I was told not to touch.

**Option B — re-arm the in-line read on stop.** Flip `_abort_poll_active =
False` on every registered transport as part of stopping, so the guard
degrades to pre-#517 behaviour instead of to blindness — which is what
`should_poll_status`'s own docstring says the intended degradation is. This is
the design-consistent answer, and it is a *transport state* change, not a stop
caller.

Both belong to the lane that owns barlink. Handing over a stop caller without
one of them would install a silent loss of abort detection, which is a worse
defect than the leak it fixes.

---

## Secondary defect, filed not fixed

`PeerWatchdog.stop` (`barlink_liveness.py:598`) has the same silent-detach
shape I fixed in the two siblings, in its sharpest form:

```python
def stop(self) -> None:
    self._stop.set()
    thread, self._thread = self._thread, None   # handle cleared FIRST
    if thread is not None:
        thread.join(timeout=5.0)
```

The handle is dropped **before** the join is attempted, so a join that times
out leaves a live thread with no record of it — and a second `stop()` returns
immediately, reporting success for a thread it abandoned. Same class as
`dual_group_lane` (`91adb7c156`), and worse in ordering.

It is harmless *today* only because the sole caller is `reset_for_test()`. It
must be fixed **before** any real caller is wired, for the same reason the
dual-group-lane method was fixed before its caller.

---

## Status

`barlink-peer-watchdog` remains **open and unfixed**, and it stays the
highest-value item in the #673 inventory on cadence (~10 ms CUDA D2H poll, not
opt-in). It is no longer merely "parked with #722" — it now has a named
blocker: **the stop caller cannot be added safely until the abort-word read
has somewhere else to go.**

#673 family: 4 of 6 abort-shaped items desk-addressed, 2 filed, none validated
on metal.
