# #431 — FP8 × barlink-BAR1 × uneven-weighted-DCP prefill wedge

Desk audit on `e80efb4af3`. No GPU was touched. Every claim below is either a
file:line in this tree or an artifact in
`/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench/`. The part
that still needs hardware is labelled as such, not smoothed over.

## 1. What the #424 evidence actually establishes

`RESULTS.md` §5 states: "the ranks are in different collectives, so it is a
genuine mismatch and not slowness". **That inference does not follow from the
artifacts.** The artifacts are host-side `py-spy` dumps without `--native`,
and the frames they caught are:

| arm | rank 0 | ranks 1, 2 |
|---|---|---|
| decode | `cp_all_gather_heads_uneven` (`dcp/comm.py:192`) | `dcp_weighted_write_slots` (`dcp/owner.py:390`) |
| prefill | `cp_lse_ag_out_ar_mha_uneven` (`dcp/comm.py:236`) | `_dcp_write_gather` (`flashinfer_backend.py:2425`) |

Read the lines:

* `comm.py:192` is `counts = list(head_counts)` — a Python list copy.
* `comm.py:236` is inside the *docstring* of `cp_lse_ag_out_ar_mha_uneven`.
* `owner.py:390` is `block * cp_ratio + (off - cp_lo)` — tensor arithmetic,
  and `_dcp_write_scatter` is documented at `flashinfer_backend.py:2439` as
  "Local half of the masked KV write (**no collectives**)".
* `flashinfer_backend.py:2425` is the `torch.cat(...)` argument line.

None of these is a collective. They are the lines at which each host thread
happened to block — which, once the launch queue behind a non-completing
device kernel fills, is an arbitrary tensor op. The DCP collectives are
issued asynchronously, so host frames cannot order them across ranks at all.
The dumps are consistent with a genuine sequence divergence **and** with one
stuck device-side collective plus two ranks blocked on a full launch queue.
Settling that is what `scripts/repro_431_fp8_bar1_dcp.sh` exists for.

## 2. The DCP path is rank-uniform — ruled out, with evidence

The obvious hypothesis (an fp8-specific dispatch inserting or skipping a
collective) is false:

* The fp8 cast is strictly **downstream** of every collective. `_dcp_write_gather`
  (`flashinfer_backend.py:2409-2426`) gathers the raw bf16 projection output;
  the cast happens in `_dcp_write_scatter` → `set_kv_buffer` →
  `memory_pool.py:2400-2409`. `memory_pool.py` contains no `dist.*` at all —
  there is no amax reduction, no scale all-reduce, no scale broadcast.
* `cp_all_gather_heads_uneven` (`dcp/comm.py:199-215`) **pads every rank to
  `max(head_counts)` before the collective**, over the replicated head-count
  vectors built at `flashinfer_backend.py:855-873`. Per-rank byte counts are
  therefore equal even under uneven TP. The equal-split fast path at
  `comm.py:200` tests the same replicated list.
* The per-layer sequence is A(kv gather) → B(q gather) → C(LSE all-gather) →
  D(all-reduce), and every branch between them —
  `do_write` (`:5789`), `has_prefix` (`:5804`, `lockstep.py:106-143`),
  `_scatter_late` (`:5813`), `dcp_kv_replicated_heads` (`:2412`, from
  `attn_kv_replicated` at `:850`), `_sess_verify_active` (`:5839`) — derives
  from replicated batch metadata or boot config. The weighted owner-rule
  constants `cp_S/cp_lo/cp_hi/cp_ratio` come from `resolve_cp_token_ratios`
  over replicated CLI args and are consumed **after** the last collective.
* The one genuinely rank-local predicate on the path,
  `_dcp_extend_has_host` (`flashinfer_backend.py:6939-6942`, derived from
  `kv_indices[:n_owned]`), is weightless-lane-only and collective-free on both
  arms — not this configuration.
* Both #424 checkpoints ran `--kv-cache-dtype fp8_e4m3`. The discriminator is
  the **weight** quantization, not the KV dtype: on FP8 the 3080s run the MLP
  through Marlin (lane ratio ~9.7:1) and on INT8-W8A8 natively (~3.7:1).

So the wedge is not a divergence produced by the DCP schedule.

## 3. The transport hole that is real (latent, not this arm)

`BarlinkCommunicator._select` (`barlink.py:553-595`) decides bar1-vs-gloo from
a **rank-local** byte count:

```
barlink.py:595   chosen = t if (t is not None and t.handles(op, nbytes)) else None
barlink.py:770   nbytes = inp.numel() * inp.element_size()          # all_reduce
barlink.py:842   self._select("all_gather", input_.numel() * input_.element_size())
barlink.py:884   (reduce_scatter)   :1048 (all_to_all)   :1168 (broadcast)
```

while every predicate behind it asserts the opposite premise:

```
barlink_bar1.py:2570   "Two ranks must never answer differently here -- one would
                        run into the collective and the other would not, and the
                        result would be a hang instead of an error."
barlink_bar1.py:3351   # ... Rank-uniform, because nbytes is.
barlink_bar1.py:3352   if -(-nbytes // int(geo["a2a_slot"])) > self.ag_max_rounds:
```

Nothing enforces the premise, and `barlink_all_gather` compounds it:

```
barlink_bar1.py:3390   plan = ag_plan([shard] * self.world, int(self._geo["a2a_slot"]))
```

`shard` is this rank's own byte count (`:3384`), replicated `world` times —
the group vector faked from the local row, in a function (`ag_plan`,
`:929-971`) written to take a genuine per-rank vector and warning that a rank
counting differently "would not be an error but a hang". The `all_to_all`
side already has the discipline this side lacks (`supports_a2a` /
`a2a_rounds_for`, `:2007-2040`, maximise group-wide first).

For `all_reduce`/`all_gather`/`broadcast` unequal sizes would break NCCL too,
so this is **not** the #424 trigger — but it is one `all_gatherv`-shaped
caller away from the same hang, and it is the same bug family. The recorder
added in this branch is the standing instrument for it.

## 4. The bar1-internal mechanism that fits #424

Three facts, each from the tree:

1. **One shared round counter, equality wait.** All bar1 collectives on a
   group sequence on a single device counter; the wait is
   `while (readFlag(...) != round)` (`barlink_bar1_ext.py:783`, `:1064`).
   Desynchronisation is absorbing: once two ranks' counters differ, no later
   collective can ever match. There is no per-call tag; slot disambiguation is
   `par = round & 1` (`:869-893`).
2. **The cap-cycle abort is silent.** `barlink_liveness.py:18-24`, verbatim:
   the deadline's "expiry writes `ctlStatus` into rank-local VRAM that no
   production code path reads. A tripped kernel is therefore silent: the
   stream continues over a partially written output buffer." Verified:
   `raise_if_aborted` is called from exactly three bring-up proofs
   (`barlink_bar1.py:3642`, `:4066`, `barlink_bar1_pipe_ext.py:1765`) and
   nowhere on the hot path.
3. **The deadline is long and load-dependent.** `SGLANG_BARLINK_BAR1_CAP_CYCLES`
   defaults to 6e10 (~30 s at 2 GHz, `barlink_bar1.py:1438-1440`) and is
   "multiplied by up to 40x inside the JIT cold-build window"
   (`barlink_liveness.py:20-22`). A 100 %-SM spin of 30 s — let alone 20 min —
   is indistinguishable from a deadlock to an operator, and to `py-spy`.

The FP8-only, BAR1-only, load-only difference on this rig is the Marlin lane
on the 3080s: a ~9.7:1 rank skew instead of ~3.7:1, plus first-use kernel
compilation that happens under load rather than at boot or during the
transport gate. That is a **timing** asymmetry, and mechanism (2)+(3) is what
converts a timing asymmetry into something that looks like a permanent hang
while NCCL simply waits. This is the leading hypothesis and it is **not
proven** — section D of the repro script reads the abort word for exactly it.

## 5. What this branch changes

* `barlink_uniformity.py` — recorder + pure `first_divergence` comparator +
  on-disk per-rank dump for post-mortems. Off by default; when off the hot
  path costs one module-global boolean test.
* `barlink.py::_select` — one guarded call, out of line.
* `ModelRunner._refuse_unproven_bar1_dcp_combination` — the scoped, named
  refusal (BAR1 × uneven weighted DCP × fp8 checkpoint only), before any
  weight is loaded. Every arm #424 ran to completion still boots.
  *Superseded: since the re-check below this is
  `_check_bar1_fp8_uneven_dcp_combination` and it warns.*
* `test/registered/unit/distributed/test_barlink_collective_uniformity_431.py`
  — 21 hermetic tests, including the structural falsifier that drives the
  **real** `_select` and `_handles_all_gather` per rank and shows the
  bar1/gloo split-brain that unequal `nbytes` produces. *31 after the
  re-check rewrote the refusal tests into notice tests.*
* `scripts/repro_431_fp8_bar1_dcp.sh` in the #424 battery dir — the four #424
  arms verbatim, with recording on and the refusal overridden.

## 6. What the GPU window must answer

Run arm A (and B). Then read `divergence_<arm>.txt`:

* **sequences diverge** → dispatch split-brain after all; the line names the
  op and byte count, and the call site follows from it.
* **sequences identical** → not a dispatch divergence; go to
  `abort_<arm>.txt` and to `transport.status()`. If a kernel tripped, §4 is
  confirmed and the fix is to make the abort loud (a cheap host-word check
  per forward, not per collective) rather than to re-route anything.

Only after that verdict should the refusal in §5 be narrowed or removed.

---

# The fix slice (branch `fix/bar1-timeout-loud-abort-431`, desk-only)

Written against tip `82f414c62d`, with no GPU: the window above already
produced the readings, and what follows is the code the readings pointed at.
Everything here is falsifier-backed — 21 hermetic tests in
`test/registered/unit/distributed/test_barlink_bar1_abort_431.py`, all of
which fail on the unfixed tree.

## Fix 1 — the BAR1 deadline now goes through the resolver

`BarlinkBar1Transport._deadline_cycles()` is the single producer of the cycle
budget, and it is `resolve_timeout_cycles(int(self.cap_cycles))` — the same
call `barlink_device.py:53` makes. All three launch sites use it:
`bar1_all_reduce`, `bar1_mesh_pipe`, `bar1_all_to_all`.

Outside the cold-build window `resolve_timeout_cycles` is the identity, so
the steady state and the default path are unchanged. The pass that RECORDS a
CUDA graph runs outside the window by construction
(`full_cuda_graph_backend.capture_one`: `run_capture_warmups` owns the
window, the recorded `forward_fn()` below it does not), so a captured graph
still carries the unmultiplied constant and serving is untouched. What
changes is exactly the warmup forwards this window measured stalling in.

Falsifiers: `_deadline_cycles` is the bare constant outside the window and
`constant * mult` inside it; the recorded kernel ARGUMENT of a real
`_all_reduce_one_round` call is the extended one inside the window; and no
launch site passes `int(self.cap_cycles)` any more (three sites, pinned at
the source, because two of them need a mapped BAR1 window to drive).

## Fix 2 — a tripped kernel raises instead of continuing

`BarlinkBar1Transport.check_aborted(where)` reads the sticky status word and
raises `Bar1CollectiveAborted` with `rank`, `world`, `group`, `op`, `nbytes`,
`rounds` and `launches` as structured attributes, not only as text.

Where it runs, and why there:

* **After every transport collective on the host path.**
  `BarlinkCommunicator._after_transport` is called from `all_reduce`,
  `all_gather`, `reduce_scatter`, `all_to_all_single` and `broadcast`. This
  is the first host code after the collective. (There is no gap on the async
  path: `all_reduce_async` exists only on the UCX transport, which has no
  device deadline.)
* **Never inside a stream capture.** Reading the word is a D2H copy plus a
  stream synchronization; issued while the current stream is being captured
  it is illegal. `graph_capture_running()` — the single definition of that
  question, `barlink.py:414` — gates every check. A launch recorded under
  capture does not advance the unchecked counter either: a recorded kernel
  has not run.
* **At the CUDA-graph replay boundary**, because that is the only host point
  a captured decode has. `barlink_abort_gate.check_after_graph_replay()` is
  called from `FullCudaGraphBackend.replay` and
  `BreakableCudaGraphBackend.replay`. With no BAR1 transport in the process
  it costs one truth test on an empty list.

A background watchdog THREAD was considered for the captured case and
rejected on the reasoning already recorded at `barlink_bar1._wait_abort`: the
thread would read the same device word, that read queues behind whatever is
on the stream, and the thing on the stream is exactly the kernel it is meant
to report on. A thread does not change that ordering — the replay boundary
does, because it runs after the graph has completed.

Knobs, all documented in `barlink_abort_gate`:
`SGLANG_BARLINK_BAR1_ABORT_CHECK=0` restores the pre-#431 silence exactly;
`..._CHECK_EVERY=N` trades reporting latency for synchronizations;
`..._CHECK_REPLAY=0` turns off only the replay-boundary check, for an
overlap-scheduled decode that must not be forced to synchronize.

## Fix 3 — the cold-build window logs its close

`jit/coldwindow_fp8_bar1_decode.txt` read 6 OPEN / 0 CLOSE and looked like a
leaked window. It was not one. `cold_build_window` logged only the open
direction, and `attach_arm.sh:158` greps `'JIT cold-build window
(open|close)'` — so zero closes was the only number that reading could ever
have returned, leak or no leak. `cold_build_window` now logs a symmetric
close line with the reason and the elapsed time, in the `finally` branch, so
the count is falsifiable.

This also settles the reading itself: the second window ("full cuda-graph
capture warmup") was genuinely still open at the stall, because the warmup
forward never returned. That is correct behaviour, and it means fix 1's
extension really would have been in force for the whole stall.

## What the GPU re-check must show before the refusal comes off

The refusal in `barlink_uniformity.unproven_bar1_combination` STAYS. Its text
now records that fixes 1 and 2 are merged and why neither lifts it: both are
about how the failure is bounded and reported, not about why the flag
rendezvous is slow. Re-run arm A with `SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_
BAR1=1`. Three outcomes, all diagnostic:

1. **Capture completes** (slowly or not). The raw 30 s cap was the binding
   constraint and the crawl was cold-build-window collisions all along. The
   refusal can be narrowed to a documented slow-boot warning.
2. **A named `Bar1CollectiveAborted`** arrives with rank/op/rounds. That is
   the first identification of WHICH collective wedges, which is what the
   FP8/Marlin round-desync hypothesis needs and could not get from py-spy.
   The refusal stays until that root cause is fixed.
3. **Still a silent crawl.** Then the status word is not being written, i.e.
   the kernels are not tripping the cap at all and the ~30-40 s interval has
   another source. That would falsify §4 of this document and is worth
   knowing.

Also worth collecting in the same run: `grep 'JIT cold-build window'` should
now show matching open/close counts, and any mismatch is a real leak.

---

# Verdict: OUTCOME 1, and the refusal is now a warning (#438a)

The re-check ran on 2026-08-02 14:34-14:48Z, tree `/spinning/wt-431-recheck`
at `8a699e3eaf` (contains the fix-slice merge `10372e902e`). Artifacts:
`/spinning/gpu-battery-results/2026-08-02_431_recheck/`, verdict in its
`RESULTS.md`.

**Outcome 1.** Arm A did not merely complete — it completed at the stock-NCCL
twin's speed, and there was no `Bar1CollectiveAborted`, no `PeerLost` and no
`CollectiveTimeout`. So it was never a deadlock; it is a slow first boot.

| | #431 repro (`022fb3872b`) | re-check (`8a699e3eaf`) |
|---|---|---|
| target-verify capture | not past batch 0 of 12 after 22 min | 12/12 in 04:58 |
| boot -> READY | never | 06:05 (`driver_A.log:4`) |
| load window | never reached | `COMPLETED: ok=176 fail=0 waves=44` |
| decision sequences | 103/95/94 `tp-0`, still growing | 28762 on all three ranks, identical, equal depth |

The transport was genuinely BAR1 throughout — nine `ACHIEVED=bar1` lines,
three ranks x three groups, in `gate_fp8_bar1_decode.txt`.

## What the timing actually looks like

`raw/server_fp8_bar1_decode.log` records 111 `JIT cold-build window` opens and
111 closes, balanced, so fix 3's symmetric close line works and there is no
leaked window. The close-duration histogram is
`<1s: 93   1-10s: 15   10-60s: 0   >60s: 3`. The three long ones are **one per
rank and concurrent**, not sequential: all three open at 14:38:01-14:38:13 on
the first CUDA-graph capture batch and all three close together at 14:41:18.

```
[14:41:18 TP0] JIT cold-build window close (full cuda-graph capture warmup) after 197.1s
[14:41:18 TP2] JIT cold-build window close (full cuda-graph capture warmup) after 184.4s
[14:41:18 TP1] JIT cold-build window close (full cuda-graph capture warmup) after 184.4s
```

That single ~190 s stretch is the whole phenomenon: batch 1 of 12 took
289 s (`04:49`), batch 2 took 1 s, and the pass finished in 4 min 58 s. One
rank sits in a first-call kernel build while its peers wait inside a
deadline-bearing BAR1 collective. Under the raw ~30 s cap a peer's spin kernel
gives up roughly six times inside one such window, which is exactly the
"~30-40 s per collective" crawl §4 measured and read as a wedge. With fix 1
routing the launch sites through `resolve_timeout_cycles`, the deadline inside
the window is ~20 minutes and the peers wait it out.

§4 is therefore confirmed in its mechanism (the cap-cycle deadline was the
binding constraint) and corrected in its conclusion (it is a timing artefact
of the cold-build window, not a wedge). Outcome 2 was reachable and did not
fire: the stall was in `run_capture_warmups`, which runs outside stream
capture, so the per-collective `check_aborted` was live there and no kernel
took its abort path.

## Consequence, implemented

`barlink_uniformity.unproven_bar1_combination` is replaced by
`bar1_fp8_uneven_dcp_notice`, which returns a `Bar1Fp8DcpNotice` with a
`refuse` flag instead of a refusal string. `ModelRunner` calls it from the
same point in `initialize()` — before any weight is loaded, so the notice is
on screen before the operator starts waiting — and now logs a warning where it
used to raise. The message names the three axes, states that the first boot
can spend ~190 s per rank in the JIT cold-build window, that this is not a
hang, that warm boots are normal, and where the artifacts are.

Overrides, with the compatibility argument stated once in
`_forced_refusal_from_env`:

* `SGLANG_BARLINK_REFUSE_FP8_UNEVEN_DCP_BAR1=1` restores the hard refusal.
  This is the explicit force-off and the name that reads correctly now that
  refusing is the non-default direction.
* `SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_BAR1` is kept and is not a no-op. Its
  two values keep their literal meanings: `0` still means "do not admit this
  arm" and still produces the hard refusal, so a launch script that pinned
  `ALLOW=0` behaves identically before and after; `1` still means "admit this
  arm", is still honoured, and the warning says in so many words that it no
  longer changes anything in that direction.
* The explicit name wins when both are set.

## Second, independent proof: full serving load, both FP8 layouts

The re-check above is a CAPTURE proof — it shows the boot completes. A second
window on the same day carried the same combination through real serving load:
`/spinning/gpu-battery-results/2026-08-02_435_coupling_fp8bar1/` (#435's
coupling window, raw per-arm server logs in the same directory).

Both FP8 layouts ran over barlink BAR1 with an uneven weighted DCP token
vector and the `Qwen3.6-27B-FP8` checkpoint, launched with the (then still
required) `SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_BAR1=1`
(`scripts/run_rest.sh:25-32`):

* `fp8_prefill_bar1` — `--rank-mlp-ratio 10,1,1 --rank-kv-ratio 2,11,10`,
  reserve `4500,4200,4200`;
* `fp8_decode_bar1` — `--rank-perf-tune phase-decode`, auto split.

Read off those artifacts:

* nine `ACHIEVED=bar1` lines per arm (three ranks x three groups) in
  `gate_fp8_prefill_bar1.txt` and `gate_fp8_decode_bar1.txt` — the transport
  was BAR1 for the whole run, not a silent fallback;
* no `Bar1CollectiveAborted`, no `PeerLost` and no `CollectiveTimeout` in any
  `.log` of the directory;
* both arms completed their full probe sets (`bench_fp8_prefill_bar1.txt`,
  `bench_fp8_decode_bar1.txt`) with the per-arm A-vs-A floors measured in the
  same boots (`floor_fp8_*.log`).

This is the evidence `barlink_uniformity.py` was written to wait for: not only
does the arm boot, it serves. Capture proof plus load proof is what makes the
refusal a warning rather than a hedge.

## What is still unmeasured

The re-check ran with a warm `extcache_docker`: the BAR1 extension's own build
(`barlink_bar1_ext_cuda_86_120`) opened and closed in 0.1 s on all three
ranks, so the boot was carried entirely by the capture-warmup windows. A boot
from a genuinely empty kernel cache adds that build on top. This is why the
notice is a loud warning rather than nothing at all.

One reading caveat carried over from `RESULTS.md`: the driver's built-in
`JIT VERDICT` line in `jit_fp8_bar1_decode.txt` claims no cold build ran and
the cap was therefore not multiplied. The premise (no `.so` rewritten) is
true and the inference is false — `run_capture_warmups` opens the window
unconditionally, and the 111 logged windows settle it. Do not take that
printed verdict at face value.
