# ANALYSE 917 — the cutover IMA: what the desk can prove, and the instrument for what it cannot

Determination date **2026-08-26**, against base `583810b6c4`
(`fix/916-pp-to-tp-double-claim`). Read the date before believing the verdict:
"not established" is a statement about the specimens that exist today.

Ticket framing on arrival: *"the CUDA IMA arose asynchronously on the
`load_stream`; the HiCache device-tier load writes or reads with invalid
addresses at or after the pp_to_tp cutover."*

**That framing does not survive its own specimens.** What follows is what
replaced it.

---

## 1. Verdict

**The fault is a CONSUMER of #918's sizing root, not a root of its own.** §5
carries the confirmation on the row digit, from this ticket's own specimen.
The reporting site — `load_stream`, the seam copy, or the barlink watchdog —
varies by which consumer touched the over-exposed band first, which is why
the three 0826 specimens name three different sites for one defect.

**What remains not establishable at the desk is the faulting ADDRESS.** Three
sites reported one sticky fault; the log's ordering cannot discriminate
between them; and the boot carried no `CUDA_LAUNCH_BLOCKING` arm, which
NOTE_867 already recorded as the condition under which "every IMA is
misattributed by construction". So no fix is built here on a guess about
which consumer it was, and none is built on #918's terrain at all.

What IS established is three exonerations and one delivery gap. Each is
load-bearing, because each removes a candidate the next window would
otherwise spend itself on.

**Three items in the arriving suspect list do not exist as stated**, and are
recorded here so the next reader does not re-derive them:

* **"#911 binding_generation routing"** — no such ticket in the tree
  (`grep -rn "#911"` over `.py`/`.md`: zero hits). The real generation-stamp
  machinery is tagged `#719/#760/W35` in `hicache_phase_binding.py` and
  `cache_controller.py`, and its documented scope is RELEASE routing, not
  load start.
* **`docs/dev/TICKET_718_contiguous_destination_extent.md`** is not the
  "five index-taking accessors" material; it is an unstarted BAR1
  destination-extent proposal for `kv_reshard.py`. The #718 index-axis
  material is `test_stray_host_index_718_class.py` + `memory_pool_host.py`
  (commit `628d9705b1`).
* **"#916 found that `check_prefetch_progress` holds indices of the old
  world"** — that is #718-class. #916 covers a census false positive and the
  `check_cpu_copy_rows` ordering fix.

Separately, and confirmed against the code: the device-kernel load path
(`HostPoolGroup.load_to_device_per_layer`, via `pool_host/mha.py:268-355` and
`pool_host/mla.py:241-290`) is NOT one of the five #718-guarded accessors and
performs no index-range check. That is a real design gap. It is not evidence
that this boot exercised it — see §2.

---

## 2. The device-tier load path is EXONERATED — by a counter-specimen

The ticket's own evidence contains the falsifier.

`boot_accept0826r7fix_0826_1817.log`, 18:27:14Z, direction **tp_to_pp**:

```
L16435-16444  PP1/PP2/PP0  #760 device-tier I/O quiesced ... in 0.0 ms   (ALL THREE CLEAN)
L16445        PP2          #867 barlink-BAR1 status poll hit an UNSURVIVABLE CUDA fault
L16451+       PP2          Scheduler hit an exception -> get_cpu_copy
```

Counted over that entire log:

| | r7 (`..._1817.log`) | rerun (`..._2149.log`) |
|---|---|---|
| `device-tier I/O quiesced` (success) | **96** | 3 |
| `#760 quiesce ... failed` | **0** | 2 |
| `load_stream failed` | **0** | 2 |
| barlink `UNSURVIVABLE CUDA fault` | 1 | 3 |
| death in `get_cpu_copy` | yes | yes |

Ninety-six clean device-tier quiesces, zero stream failures — and the same
fault, at the same reporting sites, killing the same call.

A cause that is absent in one instance of the fault is not the cause of the
family. The `load_stream failed` line in the 2149 boot is a **victim**: on
that rank the barlink poll had already reported the poison eighteen lines
earlier (L2171 against L2189/L2195), and a poisoned context raises the same
error at whatever call synchronizes next.

Second, weaker exoneration in the same direction: the fault is
**direction-independent** (r7 is `tp_to_pp`, 2149 is `pp_to_tp`), which no
account resting on pp_to_tp-specific bindings, indices or generations can
absorb.

---

## 3. The barlink watchdog is a DETECTOR, not the origin

NOTE_867 left this open ("earliest-in-time is evidence for origin, not
proof") and filed `pause_polling()` rather than applying it. It can now be
closed from the source, without a boot:

* **The flip never reaches `Bar1Transport.close()`, and in this specimen it
  provably had not run.** The only chain to it is
  `GroupCoordinator.destroy()` (`parallel_state.py:2501-2505`) →
  `BarlinkComm.close()` (`barlink.py:1444`) →
  `BarlinkMatrixTransport.close()` (`barlink_matrix_transport.py:588`) →
  `Bar1Transport.close()` (`barlink_bar1.py:5711`). `destroy_model_parallel()`
  has exactly one caller, `scheduler_teardown.py:114`, inside
  `release_distributed()`, whose own single caller is `scheduler.py:12847` —
  the **`finally` block** of `run_scheduler_process`, which runs only after
  `run_event_loop` returns. The specimen's traceback is a **live frame inside
  that event loop** (`run_event_loop → … → _release_residents_for_cutover`),
  so the `finally` had not been entered. #741's close()-race is therefore
  both unreachable at the cutover and demonstrably un-run here.
  (`barlink_matrix_transport.py:477` is a construction-time self-teardown,
  unreachable once a transport is up.)
* **Neither tensor the poll reads can be freed by the cutover.**
  `_round_dev` (`barlink_bar1.py:2595`) and `_ctl_dev` (`:2596`) are plain
  caching-allocator tensors — `torch.zeros(..., device=self.device)` — held
  by live attributes for the transport's whole life. `empty_cache()`, which
  the spill ladder calls (`phase_flip_spill.py:410, 1627`), returns only
  free segments; it cannot touch a referenced block. The VMM window that
  `close()` frees (`_own` / `_own_flag`, `:2528-2531`) is a **different
  allocation** that the poll never reads. Nor can the spill's other two rungs
  reach it: rungs 2 and 3 (`phase_flip_spill.py:645-654`, `:964-983`) each
  build their **own** `KvVmmArena` over a private `cuMemAddressReserve` range
  (`kv_vmm_backing.py:458-568`), and `cuMemUnmap` is scoped to the caller's
  own reservation — a disjoint address space from the caching allocator's.
  NOTE_867's open question 1 — "which pointer is freed underneath" — has the
  answer **none of them, on this path**, and the mechanism it named by name
  (`--phase-flip-spill-depth arena`) is ruled out with it.

The poll runs on a timer and the quiesce runs on the scheduler thread, so the
poll wins the race to observe an already-poisoned context regardless of who
poisoned it. Its own message asserts "treat THIS as the origin"; that
assertion is the message #867 installed to fix *attribution of the crash
report*, and it is not evidence of causation. **Indicator law: an indicator is
a finding only once it is established that it measures what it claims.**

---

## 4. The delivery gap: #741's fix was filed and never built

`1b6b989e9b55` — subject `[#741] The barlink-BAR1 family is ONE defect:
close() frees what the watchdog polls` — changed **one file**:

```
 docs/dev/ANALYSE_741_barlink_bar1_fault_family.md | 179 ++++++++++++++++++++++
 1 file changed, 179 insertions(+)
```

Its own body says so: *"FIX SHAPE filed (5.1 … 5.6). Not built in this
commit."* The ticket number in the subject is not a delivery receipt.

| item | shape | state at `583810b6c4` | evidence |
|---|---|---|---|
| 5.1 | disarm-then-quiesce before free in `close()` | **not built** | `barlink_bar1.py:5788-5805` still frees, then nulls, then disarms |
| 5.2 | atomic guard against check-then-use | **not built** | `poll_status_word` re-reads `self._ctl_dev` at `:4937` after checking it at `:4930` |
| 5.3 | `record_stream` for the private-stream read | **not built** | `grep -c record_stream` over `srt/distributed/device_communicators/` = **0** |
| 5.4 | pause polling across the flip teardown | **not built** | `pause_polling` has one real caller, `parallel_state.py:2888` (graph capture) |
| 5.5 | narrow the gate's `except` | **BUILT**, as #867 | `barlink_abort_gate.py:344-372` |
| 5.6 | validate the abort code | not built | no known-code table in the module |

5.1/5.2 are moot for these specimens per §3 (the path is unreachable). 5.3
and 5.4 remain live for the S3 arm — a plain recycle under memory churn, no
flip involved — and are filed here, not built, for the same reason NOTE_867
gave: neither is on the path these specimens took, and shipping a fix onto a
path the evidence has just cleared is how the last three shifts were spent.

---

## 5. The one root that IS measured, and where it stands

R7's fault has a measured root, recorded in `memory_pool.py:645-661`:

> PP2 held live rows to a high-water id of 122898 while its backing had been
> dialled down to 114688; `release_residents_for_cutover` → `seam_copy_state`
> → `offload_kv_cache` → `get_cpu_copy` read them, and the rank died at the
> next `synchronize()`.

The mechanism is `runtime_set_backing_tokens` (`memory_pool.py:3598`), whose
docstring states a precondition — *"rows above `num_tokens` must be dead"* —
and whose consumer comment (`schedule_batch.py:2129-2131`) says in as many
words that the statement *"is false while a resident holds one"*.

**The shrink interlock is not where this fails, and the successor root is
#918's, not this ticket's.** `_max_live_row` (`kv_backing_relief.py:1899-1933`)
returns a genuine high-water id, `max(live.max(), pending_top)` — not a row
count — and `live_floor_rows` (`:3855`) publishes it into the group reduction
(#792, `phase_flip_spill.py:1350-1358`). And the 2149 shrink released nothing
at all (`claimed=0 bytes`, `released_bytes=0` on every rank), so no page was
unmapped by a shrink in that boot.

**The direction that DOES fit is the opposite one: an over-exposing GROW.**
#918 (desk-proved on a parallel strand, 2026-08-26) has the mechanism — two
derivations of the same size diverge, the dial bounded by
`_reserved_num_tokens` and the byte span bounded by the descriptor's own
shape. This specimen carries that divergence on the row digit, identically on
all three ranks (`boot_rerun0826_0826_2149.log`, L2146/2156/2181):

```
BACKING-DIAL call: request=473525 prev_size=439970
    uniform_backed_rows=485366 reserved_backing_rows=485366
    store_bound_rows=468982 page_size=1 delta=+33555 branch=grow backing=shrink
BACKING-DIAL grow done: prev_size=439970 -> size=473525
```

* `485366 − 468982 = 16384`; with the `+1` term, **16385** — the exact gap
  #918 names.
* `uniform_backed_rows` and `reserved_backing_rows` print the **same number**.
  That is the fingerprint of the repoint in `940a7bba31` (the #913 commit, an
  ancestor of this branch's base), which changed
  `uniform_backed_rows` from `int(owner.uniform_backed_tokens)` to
  `int(owner.reserved_rows)` — so the field that is supposed to say what is
  physically backed now says what is reserved.
* The grow granted `size=473525`, which is **4543 rows inside** the
  unaddressable band `[468982, 485366)`.

**Ordering, per rank, in the specimen: every rank's first IMA follows its OWN
over-exposing grow, and none precedes it.**

| rank | over-exposing grow | first IMA observed |
|---|---|---|
| PP2 | L2147 | L2162 |
| PP1 | L2157 | L2171 |
| PP0 | L2182 | L2675 (its quiesce at L2187 still succeeded) |

R7 is the same shape with a different consumer: #916's docstring measures it
as live high-water 122898 against a backing of 114688 — exposure above
backing — in a log with zero `load_stream` failures. Two consumers, one
defect. That is why §2's exoneration and this section agree rather than
compete: the load stream was a consumer of the over-exposure on 2149, and the
seam copy was the consumer on r7.

**Not built here, by instruction and by ownership**: nothing in this branch
touches `kv_vmm_backing.py` or `memory_pool.py` sizing. The sizing root is
#918's terrain. The `940a7bba31` repoint is reported, not repaired here.

**What this specimen still cannot supply**: the faulting ADDRESS. Boot #2 ran
without #916's host-only prologue, so `check_cpu_copy_rows` died on its own
`indices.min()` before printing an index, and the boot carried no
`CUDA_LAUNCH_BLOCKING` arm. The fit is temporal, structural and digit-exact on
the band — it is not an address-level placement of the access inside it. §7's
instrument is what would supply that placement on the next flip.

---

## 6. The class, the siblings, the ratchet

**CLASS — `except Exception` around a device call on the no-return path,
where a transient and a context kill become one row.** This is #867's class.
Its first instance was the barlink watchdog. Its second was never swept.

**SIBLING SWEEP.** The cutover's own instrument, `SeamCensus.mark`
(`phase_flip_seam_census.py:254`), stops at ~19 named boundaries of the
cutover and calls the **driver** at each one (`mem_get_info`) — inside a bare
`except Exception: sample = None`. On a poisoned context that call raises, and
the census was recording the earliest host-visible evidence of the poison as
`probe-failed`, at the exact boundaries that bracket the candidates. The one
instrument positioned to answer the attribution question was discarding the
answer.

**SIBLING SWEEP, SECOND MEMBER — and this one was corrupting the record the
whole time.** `HiCacheController.quiesce_device_io`
(`managers/cache_controller.py:1309`) caught each stream's `synchronize()`
failure, logged it, and then emitted its summary line **unconditionally**:

```
L2189  PP1  #760 quiesce (pp_to_tp): synchronizing write_stream failed (IMA)
L2195  PP1  #760 quiesce (pp_to_tp): synchronizing load_stream failed (IMA)
L2201  PP1  #760 device-tier I/O quiesced for phase flip pp_to_tp in 0.2 ms
            (write and load streams drained while their pools are still live).
```

`grep -c "device-tier I/O quiesced"` over that boot returns **3** — one per
rank — in a boot where exactly **one** rank drained cleanly. Every reading
ever taken off that line, including the "96 clean drains" in §2, was taken
off a line that cannot distinguish a drain from a blind one. (§2's comparison
survives, because r7 has zero failure lines at all; but the line is now only
emitted when it is true.)

Fixed here: the failed-stream case emits a distinct ERROR naming which streams
did not drain, and the success wording is withheld. A poison-class failure
additionally registers with `record_poison` under the drain's own name, which
makes this boundary a fence at the most contested seam of the walk — between
the device-tier writeback before it and the resident release after it.

Other members checked and cleared: `barlink_abort_gate.poll_status_words`
(fixed, #867), `dual_group_lane.py:2256` and `barlink_liveness.py:715`
(both already classify via `is_poison_error` before swallowing).

**FUTURE CHECK — two files, both directions pinned.**

* `test/registered/unit/managers/test_census_poison_bracket_917.py` — a
  poison-class error must be recorded and bracketed; an OOM and a transient
  must still be swallowed; and the no-return-path contract this module has
  held since #631 (a broken gate degrades to a missing log line, never to a
  raise).
* `test/registered/unit/mem_cache/test_quiesce_reports_what_happened_917.py`
  — the success line must be withheld when any stream failed, one failed
  stream is enough, both streams are still attempted, an earlier reporter
  keeps the poison record, and the runtime's second emitter must report the
  failure rather than repeat the drain claim. Plus one pin that looks like
  trivia and is not: the poison helper must be a MODULE function, because the
  drain's own harness calls `quiesce_device_io` unbound against a
  `SimpleNamespace`, where a `self.`-looked-up helper would raise
  `AttributeError` out of an exception handler on the no-return path.

The over-refusal direction is pinned deliberately in both: torch raises the
same exception class for a recoverable OOM, and the cutover is where memory
pressure lives. A classifier that read the class instead of the message would
turn every memory-pressed flip into a reported context kill.

Not docked on the #859 `cutover_participants` registry, and that is a filed
gap rather than an omission: the registry's `Participant` rows carry a `probe`,
and the drain's probe would have been the very log line this ticket just
proved unreadable. The honest order is to fix the line first — done here —
and dock a probe on it once a window has seen it fire.

---

## 7. What was built: an INSTRUMENT, declared as one

It fixes no fault. It answers one question — **which segment of the cutover
the context died in** — using the stage names the codebase already curates, so
no second taxonomy enters the tree. Three files carry it:
`phase_flip_seam_census.py` (the bracket), `cache_controller.py` (the drain
tells the truth and becomes a fence), `phase_flip_runtime.py` (the second,
more-read emitter stops repeating the drain's old claim).

`SeamCensus.mark` now classifies its probe failure. Two readings result:

* `probe_poison` — the first boundary whose driver call failed poison-class;
* `last_clean_label` — the last boundary whose driver call answered.

Together they bracket the interval, and the bracket is emitted on the census
line the corpus is already grepped for:

```
*** #917 CONTEXT POISONED between 'hicache_quiesce' and 'resident_release': ... ***
```

It registers with `barlink_abort_gate.record_poison`, which is **first-wins
across the process**, so a census boundary that catches the origin outranks
the watchdog's later report instead of competing with it.

**And it gives the watchdog a coordinate rather than silencing it.** In both
specimens the poll reported first. Its record carries no position in the walk;
the census now supplies one:

```
*** #917 CONTEXT POISONED, first reported by 'barlink poll_status_word';
    this rank's last clean stage boundary was 'hicache_quiesce' ***
```

That is NOTE_867's open question 2 made answerable **without** taking
`pause_polling()` across the no-return point — which would have answered it by
removing the fastest detector from the path that needs it most.

**What it will tell the next window.** The bracket names one segment. Each
segment holds one candidate: `flip_writeback → hicache_quiesce` is the
device-tier writeback and the quiesce itself; `hicache_quiesce →
resident_release` is the resident release and `get_cpu_copy`; `kv_pack →
kv_local_read → kv_write` are the three legs of the payload walk;
`backing_release` / `backing_restore` are the dial. A bracket whose lower end
is `<none>` means the context was already poisoned when the flip began — which
would clear the whole cutover at a stroke.

**What it will NOT tell it.** It resolves to a segment, not to a line. Ranks
whose contexts die between two marks that are microseconds apart will read as
one segment for all three. And it cannot see a fault born before the census
opens; for that, the standing `CUDA_LAUNCH_BLOCKING` arm remains the only
separator, and this instrument is a cheap complement to it, never a substitute.

---

## 8. What is left standing, named

The candidates §2-§4 clear are not replaced by a new accusation here. What
remains, in the order a window should spend itself on it:

1. **The sizing root itself — #918's, confirmed by §5 and not owned here.**
   The one VMM arena §3 does not clear is the KV cache's own, and §5 shows the
   over-exposure on the row digit in this specimen. The bracket in §7 is what
   places a fault relative to the dial's own census marks
   (`backing_release` / `backing_restore`) on the next flip, which is the
   address-level step §5 cannot take from this log.
2. **NOTE_867's falsifiable prediction, still unspent.** Taking
   `pause_polling()` across the no-return point for ONE arm: if the poll is
   only reporting, the crash MOVES to the next CUDA call rather than
   disappearing. Not applied by default here — the instrument answers the
   same question without removing a detector from the path that needs it.
3. **A tensor-identity counter on `_ctl_dev`** (`data_ptr()` compared between
   polls, with no `close()`/`_build_up()` between) would falsify allocator
   recycling per-tensor, which the census's aggregate `mem_get_info` cannot
   see. Filed; §3 makes it low-yield for these specimens.
4. **Provenance question, raised and not answered:** this boot's `server_args`
   dump (`boot_rerun0826_0826_2149.log:56`) carries
   `scheduler_distributed_teardown=True`. `scheduler_teardown.py:49-55` is the
   gate that arms the destroy path at all, and its own docstring names
   "#722 (barlink abort-poll x flip)" as the reason it normally ships off. It
   did not matter here (§3: the loop had not exited), but whether that flag
   was intended for this acceptance boot is worth one look before the next.
