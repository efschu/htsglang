# #867 — the watchdog swallowed a fault the process cannot survive

## What is fixed here, and what is not

**FIXED: the attribution.** A CUDA illegal-memory-access poisons the context.
Every later CUDA call in the process raises the same error at whatever site runs
next. `poll_status_words` caught it, logged `barlink-BAR1 status poll failed`,
and continued — which did not keep serving alive, it only decided that the crash
would be reported somewhere innocent.

**NOT FIXED, AND NOT GUESSED AT: which pointer is freed underneath.** Nothing in
this commit assumes an answer. See "Still open" below.

## The specimen

W40, `boot_w40_857strict_0825_2113.log` (`CUDA_LAUNCH_BLOCKING=1` arm). First
CUDA fault in the whole log, 21:17:13, immediately after
`SEAM DRAIN tp_to_pp: device-tier streams quiesced in 0.0 ms at the no-return point`:

```
barlink_abort_gate.py:351 poll_status_words -> barlink_bar1.py:4937/4948
  self._abort_poll_dst.copy_(self._ctl_dev[0:1], non_blocking=True)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Downstream of it, in the same second, the scheduler died in `get_cpu_copy`
inside the seam **capture**. The boot before (`..._2107.log`, no CLB) died in
`load_cpu_copy` inside the seam **restore**. Three sites, one fault, two of them
innocent — and this shift chased two wrong roots because of it.

## The class

**An exception handler that assumes its failure is survivable.** The handler is
right for a poll that hiccups (a transient, an OOM) and wrong for a context
kill, and it could not tell them apart because it caught bare `Exception`.

The classifier reads the **message**, not the class, and that is deliberate:
torch raises `AcceleratorError` for a recoverable OOM and for an illegal access
alike, so `isinstance` would over-refuse. Controls pin both directions.

**This module already owned this class.** Its own docstring opens on #431: a run
that tripped the cap on every collective produced an `abort_*.txt` with ZERO
matching lines, so "nothing tripped" and "everything tripped" were
indistinguishable. The handler had made "a poll hiccuped" and "the CUDA context
is dead" indistinguishable in exactly the same way, inside the module written to
end that.

## The wider pattern this is the fourth instance of

*Where it surfaced is not where it originated.*

1. W40b: IMA attributed to `restore_seam_state` — actually the offload side.
2. The seam emitter placed **after** the dangerous call, so the one event it
   existed to explain was the one it could not witness (fixed in #783b).
3. W40 arm 1 without CLB: attributed to the trailing `synchronize()`.
4. This: the true origin logged as a warning while an innocent site crashed.

`CUDA_LAUNCH_BLOCKING=1` in one standing acceptance arm is what separates them;
without it every IMA is misattributed by construction.

## Prior art, checked before building

* `pause_polling()` (`barlink_abort_gate.py:309`) **exists** and excludes the
  watchdog's device reads for a scoped window. It is entered by exactly one
  caller, `parallel_state.graph_capture` (`parallel_state.py:2888`), and its
  docstring scopes it to CUDA-graph capture. **The phase flip does not take it.**
  Filed below rather than used here, because using it would silence the poll
  without establishing that the poll is the origin.
* `hicache_phase_binding`'s generation stamping (`BindingState.advance`,
  `host_pool_at`, `write_back_stamp_is_current:527`) is the reusable shape for a
  phase-generation stamp **if** the mapping question resolves that way. **Not
  built here**, per the rule that a stamp must not ship on a guess about which
  mapping moves.
* Poison-class handling: **does not exist** anywhere in
  `srt/distributed/device_communicators/` — the only matches for "illegal memory
  access" are four comments in `pynccl.py:137,191,238,263` about avoiding one.

## Still open — and it must stay open

1. **Which pointer is freed underneath.** `_ctl_dev` is allocated in `_build_up`
   (`barlink_bar1.py:2596`) and nulled in `__init__` (:1765) and `close()`
   (:5794), where `vmm_free` runs on `_own`/`_own_flag` *before* the null. But
   `poll_status_word` early-returns when `_ctl_dev is None`, so a completed
   `close()` cannot produce this fault — which means either a race inside
   `close()`, or a different mapping entirely. `--phase-flip-spill-depth arena`
   is the named mechanism that makes device memory move under the flip. **Not
   established. Do not stamp on it.**
2. **Whether the barlink poll is the origin or merely the first reporter.**
   Earliest-in-time is evidence for origin, not proof. The falsifiable
   prediction: if the poll is only reporting, then pausing polling across the
   cutover moves the crash rather than removing it. That is the next arm, and it
   is why `pause_polling` was filed instead of applied.
3. **W40a's nine successful restores versus W40b's first failing one.**
   Narrowed, not closed. With `guard_refusals=0` (#783b's range check never
   fired) and the earliest fault in barlink, the restore was probably never the
   origin in either boot — but "probably" is not a finding.

## Filed, not built

* **The flip does not take `pause_polling()` across the cutover's no-return
  point.** One-line shape, but see open question 2 before applying it.
* **A phase-generation stamp on whichever device mapping proves to move**, reusing
  `hicache_phase_binding`'s mechanism rather than adding a second one.
