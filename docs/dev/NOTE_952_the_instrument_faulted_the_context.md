# #952 — the instrument faulted the context it was measuring

Desk analysis, 2026-08-28, pin `2897161bdb`. No boot. The fix is harness-side
(`/spinning/gpu-arb/devtools/trace_overlay_949/sitecustomize.py`,
`/spinning/gpu-arb/boot_946rf.sh`); this note is the reasoning and the
boundary against #867.

## Verdict

**The #949 trace overlay caused the CUDA illegal access, by tracing.** Not the
signal's latency, not the dump's I/O, not a barlink transport defect. #952 has
its own root and that root is entirely instrument-side.

The chain, each link established rather than assumed:

1. The overlay arms VizTracer with `log_func_args=True` over `managers/` and
   `mem_cache/` (`sitecustomize.py`, `_install`).
2. VizTracer's C tracer calls `PyObject_Repr` on **every argument of every
   traced call** when that flag is set — symbol present in
   `snaptrace.cpython-312-x86_64-linux-gnu.so`, gated on `log_func_args`.
   Confirmed hermetically: with the flag on, a traced argument's `__repr__`
   runs; with it off, it never does.
3. `torch.Tensor.__repr__` (`torch/_tensor.py:568`) reaches
   `torch/_tensor_str.py:278-283`, which formats via `.tolist()` — a
   **device-to-host copy of the tensor's storage**.
4. During `pp_to_tp` / `tp_to_pp`, `PHASE-FLIP-SPILL` returns device segments
   to the driver and `PHASE-FLIP-BOOT REFILL` re-maps 8.5–16 GiB per rank
   (both visible in every specimen). A traced argument whose storage was just
   freed or moved is read **by the act of describing it** → illegal access.
5. VizTracer swallows the Python-level error from that repr (confirmed
   hermetically), so nothing is reported at the scene. The CUDA context is
   poisoned silently.
6. The fault surfaces at the next CUDA call in the process. That is the
   barlink BAR1 status poll — **and it is the first reporter by construction**,
   not by coincidence: the poll runs on a dedicated daemon thread
   (`barlink_liveness.py:602-605`, `barlink-peer-watchdog`) at a **10 ms
   cadence** (`barlink_abort_gate.py:271-280`), issuing a D2H copy at
   `barlink_bar1.py:4937` and synchronising at `:4948`. A 10 ms-cadence CUDA
   call on a thread that keeps running while the main thread is busy will win
   the race to report *any* poisoning, wherever it came from.

## Why the priming dump is the trigger, and why 20 boots were clean

The overlay documents that the tracer session started in `sitecustomize` is
silently lost during the sglang/torch import — "the FIRST ring is empty while
every later one is full". So during normal running **tracing is not live** and
the instrument costs nothing. `_dump_trace`'s `finally: tracer.start()` is what
makes it genuinely live — *after* the imports that would have killed it.

That is the whole story of the invisibility. Until #949b the dump signal was
SIGUSR1, which `barlink_launch_dump.install_sigusr1_handler`
(`barlink_launch_dump.py:262`) silently overwrites — an unconditional
`signal.signal` that never inspects or chains the previous handler. The
overlay's handler never ran, so `tracer.start()` never ran, so arg-logging was
never live. `dc4895e1dc` was the first pin where the dump path worked, and the
first flip after the first dump faulted.

## The evidence, and a correction to how it was framed

The 2x2 as recorded (watcher ON ⇒ 2/2 IMA, watcher OFF ⇒ 0/2) **does not
support its conclusion**, and the reason matters more than the conclusion:

* The watcher-OFF boots (`..._0828_000240.log`, `..._0828_001113.log`) contain
  **zero `no-return` lines**. They never reached a cutover. Their 0 IMA is a
  non-observation, not a control — the arm never executed the hazardous region.
* "Watcher OFF" did not move the variable it appears to move: all four boots
  log the identical six `viztracer armed on pid … (dump signal 34)` lines. The
  watcher is the signal *sender*; arming is unconditional.

The real discriminator is **within-boot**, and it is much stronger than the
2x2 was:

`boot_943bx_dc4895e1dc_0827_235242.log` completed **four** flips.

| # | flip | no-return | dump before IMA? | IMA |
|---|------|-----------|------------------|-----|
| 1 | pp_to_tp | 23:56:33 | no | **no** |
| 2 | tp_to_pp | 23:57:21 | no | **no** |
| 3 | pp_to_tp | 23:57:34 | no | **no** |
| 4 | tp_to_pp | 23:57:48 | **yes** (log 2201-2204) | **yes**, 23:57:59 |

Same process, same code, same phase-transition type as flip 2 — three clean
flips with tracing dead, one faulting flip with tracing live. Adding boot 1
(one flip, one dump, one IMA at 23:47:13): **5 flips, 2 carried a dump, 2
faulted, 3 did not.**

## Boundary against #867 — trigger or own root?

**Own root, and #867 stays open on its own terms.**

#867's open question 2 asks whether the barlink poll is the origin or merely
the first reporter. For *this* specimen family the answer is now: **merely the
first reporter**, and mechanically so (§Verdict 6).

Two facts rule out the transport reading freed memory here:

* `barlink_bar1.py` and `barlink_abort_gate.py` contain **zero** references to
  `pp_to_tp`, `tp_to_pp`, `phase_flip`, `cutover` or `rebind`. The flip does
  not touch this mapping.
* `_ctl_dev` is allocated once in `_build_up` (`barlink_bar1.py:2596`) and
  nulled only in `close()` (`:5794`), and `close()` is reached only from
  `GroupCoordinator.destroy()` (`parallel_state.py:2505`) — full teardown, not
  a mid-run cutover. There is no per-flip rebuild for a dump to race.

But #867's **own** specimen (`boot_w40_857strict_0825_2113.log`, tp_to_pp,
2026-08-25) predates the working dump path: tracing was never live there, so it
had a different origin. **Two populations, one reporter.** #952 explains the
`dc4895e1dc` family and explains nothing about W40. #867's open question 1
(which pointer is freed) remains unanswered and unstamped.

One residual TOCTOU is worth recording for #867 rather than for #952:
`poll_status_word` re-reads `self._ctl_dev` at `barlink_bar1.py:4930` and again
at `:4937` with no cached local and no lock shared with `close()`, while
`barlink_abort_gate.unregister` (`:501-505`) only removes the transport from a
list and never joins an in-flight poll.

## The fix, and why not the two obvious ones

**Root fix: `log_func_repr=_safe_repr`.** VizTracer honours a caller-supplied
describer instead of `PyObject_Repr` (verified). `_safe_repr` never invokes a
foreign `__repr__` at all: cheap scalars and small containers of them are
repr'd — those are the rids and token counts the instrument exists to capture —
and everything else is described structurally from `shape`/`dtype`/`device`,
which are Python-side metadata that do not touch device memory. A whitelist and
not a torch-shaped blocklist, because any object with a `__repr__` that reaches
for device state is the same hazard and a blocklist would have to name it in
advance.

**Rejected: pausing the barlink poll across the cutover.** That is #867's filed
arm and it is a *diagnostic*, not a fix — NOTE_867 says so explicitly, and
applying it would silence the reporter without touching the origin. Worth
recording, since the sibling question was raised: pausing would **not** blind
abort detection. `pause_polling` (`barlink_abort_gate.py:391`, a reentrant
depth counter checked at `:425-426`) suppresses only the BAR1 status-word read.
Dead-peer detection runs on the same thread through a different path that is
ungated by it — `PeerWatchdog.probe_once` → `any_dead_peers` (a host-only
`/proc` check) → `trip_all_abort_windows` → `AbortWindow.trip`, a plain host
store, `barlink_liveness.py:664-671`, `:570-574`, `:458-474`.

**Rejected as sufficient: deferring the dump out of the flip window.** This was
the cheap candidate, and on its own **it does not fix anything** — the hazard is
not the dump's timing but that tracing stays live *afterwards*. Deferring only
moves the first fault to the next flip. It is kept as the *second* half of the
fix for a different and real reason (below), never as the first.

## Class remedy: the handler was unbounded, and it is not alone

`_dump_trace` did `tracer.stop()`, a JSON `save()`, file I/O and
`tracer.start()` **inside a signal handler** — on the main thread, at whatever
bytecode boundary the signal landed on, which during a cutover is inside the
straight-line no-return region of `PhaseFlipRuntime._execute` (emitter
`phase_flip_runtime.py:5790`, seam flag set at `:11768`). It is now a single
integer increment, with the work moved to a `949-dump-worker` thread that holds
the dump while a seam is open and **delivers it afterwards**, counting and
logging both the deferral and the delivery. A deferral that silently ate a dump
would be this family's own "success value that is not evidence" in a new
colour, so the timeout path dumps anyway and says it did.

**Geschwister-Sweep — this class is tree-wide and unguarded.** A sweep for
`pthread_sigmask` / `sigprocmask` / `block_signals` / `defer_signal` found
**zero hits anywhere in `python/`**: nothing in this codebase masks or defers a
signal across any critical section. Six handlers installed *in the scheduler
process itself* do unbounded work with no phase awareness:

| file:line | signal | handler body | fires in cutover? |
|---|---|---|---|
| `storage_hf3fs.py:258-260` | INT/TERM/QUIT | `close()` → `executor.shutdown(wait=True)` — **blocking** | yes; same subsystem the seam drains |
| `multi_ended_allocator.py:80-94` | TERM/INT | walks a WeakSet, formats ~15 counters, logs | yes |
| `expert_stats.py:550-573` | USR2/TERM/INT | `dump()` → JSON file I/O | yes |
| `barlink_launch_dump.py:246-267` | USR1 | takes non-reentrant `_LOCK`; self-deadlocks if interrupted inside `register()` | yes |
| `base_connector.py:27-72` | INT/TERM | `shutil.rmtree`, then **chains** to the previous handler | yes; installed at weight load, never removed |
| `regime_runtime.py:1082-1128` | TERM | `close_trace()` file I/O, then chains | yes |

The sharpest is `storage_hf3fs.py`: it can call a **blocking**
`executor.shutdown(wait=True)` from the interrupted thread while
`hicache_seam_active` is `True`. Not filed as fixed here — filed as named, with
the note that a cheap lock-free flip probe already exists and nothing uses it
for this purpose: `hicache_phase_guard._FLIP_AUTHORITY()` →
`hicache_seam_active` (`hicache_phase_guard.py:104-118`). The overlay's worker
now reads exactly that, rather than adding a second source of truth.

## Zukunfts-Check

The opposite hazard of a deferred dump is a *lost* dump, which is strictly
worse than a badly-timed one: it converts a measurement into silence. Hence the
`_DEFER_MAX_S` ceiling that dumps anyway and logs that it did, the
`deferred`/`delivered` counters on the arm line, and a mutant test that goes
red if a defer ever swallows a dump.

The opposite hazard of `_safe_repr` is a blinded instrument. Checked: the
values the overlay was built to capture — "reconcile ran with THIS rid and THIS
told" — are ints and strings, which `_safe_repr` still renders exactly.

## Instrument hygiene

`boot_946rf.sh` exported the overlay on `PYTHONPATH` (`:59`) with
`SGLANG_949_TRACE=1` (`:63`) and then ran two bare `python3` helpers. Those are
the **system** interpreter, which has no viztracer, so each printed
`[#949 trace-overlay] viztracer NOT armed` — indistinguishable from a serving
rank failing to arm. Reproduced verbatim at the desk against `/bin/python3`.

Both halves fixed: helpers now run through a `helperpy` wrapper that strips the
overlay from the environment and uses the venv interpreter, and the not-armed
line now carries the role and the reason, so the two cases can never be
confused again:

```
NOT armed [role=helper(-c) venv=NO->helper, not a rank exe=/bin/python3 …]: ModuleNotFoundError: No module named 'viztracer'
NOT armed [role=helper(-c) venv=yes exe=…/.venv/bin/python …]: ValueError: invalid literal for int() …
```

A related hygiene defect is recorded but **not** fixed, because fixing it
blindly would suppress real dumps: the empty-ring guard trips only below 2000
bytes, and the observed near-empty priming dumps were 6.4–10.7 KB. They
reported `dumped`, not `RING WAS EMPTY`. The threshold is a magic number that
does not measure emptiness; the honest fix is to count events in the ring, and
that needs a boot to calibrate against a genuinely full dump.

## Falsifier for the enable probe

With the fix in and tracing live, if an IMA still appears at a flip, mechanism
(iii) is refuted and #867's transport question reopens with better evidence
than it has today. The discriminating marker is the argument descriptor: traced
args must read `#952-not-read`, never a rendered tensor.
