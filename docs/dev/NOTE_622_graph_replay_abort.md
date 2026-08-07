# NOTE 622 — the GRAPH-REPLAY abort family: what the evidence supports

Window: 2026-08-07. Tree: `f7df966ebc`. Specimens: 2026-08-05 21:10 (crash 10,
`/spinning/CRASH_20260805_boot10_2110.log`) and 2026-08-07 03:25 (boot pgid
483646, **boot log lost** — see §1).

This note is deliberately split into *what is established*, *what is
falsified*, and *what remains open*. The instrument shipped in this window
exists because the third list is not empty.

---

## 1. The evidence gap that shaped this window

The 03:25 boot log was **never archived**. `handle_unhealthy` archived only
under `if [[ -n "$pgid" ]]`, on the assumption that a `DOWN` verdict had
already been archived by an earlier pass holding a live pgid. That assumption
fails for exactly this crash family: when all three ranks raise and exit
between two 10 s ticks, the watchdog never observes a live pgid at all. The
watchdog log shows the whole cycle:

```
03:23:06 ALARM status=DEGRADED_HTTP pgid=483646
03:25:36 WARN status=DOWN pgid=none streak=1/3
03:25:46 WARN status=DOWN pgid=none streak=2/3
03:25:56 ALARM verdict=DOWN pgid=none
03:25:56 RESTART executing /root/bin/start-serving-30030.sh    <-- no ARCHIVED line
```

Fixed in `/root/bin/serving-30030-watchdog.sh` (host infrastructure, not in
this repo): archiving now happens on **every** restart cycle, deduplicated by
the boot log's `inode:mtime:size` signature so a failing start script retried
in a loop still cannot flood `/spinning`.

Consequence for this note: the 03:25 specimen contributes only the abort text
captured live in a monitor, plus the wedge dump in §3. Everything quantitative
below comes from the 08-05 log and the wedge-catcher inventory.

---

## 2. What both specimens have in common

Both raise from the identical stack — the **decode** graph, not a draft graph:

```
verify                        (eagle_worker_v2.py:2629)
forward_batch_generation      (tp_worker.py:556)
_forward_raw                  (model_runner.py:4237)
execute                       (decode_cuda_graph_runner.py:2100)
replay                        (full_cuda_graph_backend.py:173)
check_after_graph_replay      (barlink_abort_gate.py:382)
check_aborted                 (barlink_bar1.py:4561)
-> Bar1CollectiveAborted
```

08-05: ranks 0 and 1 raise `Bar1CollectiveAborted` (21:10:04 and 21:10:14).
Rank 2 does **not** — it dies at 21:10:14 with `RuntimeError: Connection
closed by peer` inside `broadcast_pyobj` / `_broadcast_reqs_across_ranks`,
i.e. it was blocked in the scheduler's request-receive broadcast waiting on
rank 0, which had already exited. Rank 2's traceback is therefore a
**consequence** of rank 0's death, not an independent fault, and it must not
be read as "rank 2 was the divergent one".

Both messages name a tiny control-plane collective as the last launch (08-05:
`all_to_all (8 bytes, 0 rounds)`; 08-07: `broadcast (8 bytes, 0 rounds)`) and
both messages say, correctly, that it is from an already-closed window and is
not the culprit. The 21 s between each rank's last normal log line and its
abort is consistent with the 60e9-cycle deadline (`CAP_CYCLES=60000000000`).

---

## 3. The wedge dump: all three ranks on ONE line

`/spinning/wedge-catch-603b/wedge_20260807_032306_*` was captured at 03:23:06
— the same second as the `DEGRADED_HTTP` alarm, ~2.5 min before the process
group died. All three ranks are on the **identical** host stack:

```
build_dcp_weighted_kv_indices (dcp/owner.py:548)
call_begin_forward            (flashinfer_backend.py:7160)
update_single_wrapper         (flashinfer_backend.py:6721)
init_forward_metadata_out_graph (flashinfer_backend.py:1485)
init_forward_metadata_out_graph (hybrid_linear_attn_backend.py:914)
load_batch                    (decode_cuda_graph_runner.py:2029)
eagle_prepare_for_verify      (eagle_utils.py:789)
verify                        (eagle_worker_v2.py:2577)
```

with all three GPUs at **100 % SM** (`nvidia-smi` in the dump's context file).

`owner.py:548` is `total = int(full_indptr[bs].item())` — a blocking D2H.
This is the site NOTE_616c §11 already named, at its old line number 529, with
the same all-three-ranks signature.

### The wiring gap

NOTE_616c proposed an optional `total_tokens` parameter so a caller holding a
CPU mirror can skip the sync. It was added by `fec3b11de5` — commit subject:
*"add a sync-free channel for the owner.py:529 total (**inert until wired**)"*.

It is still almost entirely inert. Of the five call sites, **one** passes it:

| call site | path | `total_tokens` |
|---|---|---|
| `flashinfer_backend.py:6456` | | **no** |
| `flashinfer_backend.py:6963` | extend / prefill | yes (`_dcp_host_total_tokens`) |
| `flashinfer_backend.py:7063` | | **no** |
| `flashinfer_backend.py:7160` | **spec verify (the wedge stack)** | **no** |
| `triton_backend.py:1054` | | **no** |

The call site in the wedge stack is the unwired one.

### Why this is a pin site and not, on this evidence, the root cause

NOTE_616c §11 records the conclusion that matters here, and it is not
convenient:

> That spread is itself the finding: the host gets pinned wherever it next
> touches a device queue that is not draining. Fixing any ONE of these sites
> moves the wedge to the next one; it does not cure it.

Three distinct pin sites are already on record (`flashinfer_backend.py:7429`
`.cpu()`, `memory_pool.py:1525` enqueue backpressure, `owner.py:529`
`.item()`). The host blocking at `owner.py:548` is the host **noticing** that
the device queue is not draining. The queue is not draining because a BAR1
spin kernel inside a replayed graph is waiting on a peer flag — which is the
thing still not identified.

Wiring `total_tokens` at 7160 (and the other three sites) is worth doing on
its own merits — it removes a blocking D2H from the hot spec-verify path — but
it should be expected to relocate this wedge, not end it. It is **not** filed
here as the fix for the family.

---

## 4. Hypotheses weighed

### (a) Host-path divergence between replay and non-replay steps — NOT SUPPORTED

The proposal was that a kernel inside the decode graph waits on a peer flag
that never arrives because the peer is not in a replay. The 03:23:06 wedge
dump shows the opposite: all three ranks are at the **same** host stack, same
function, same line. At the one instant this family has ever been observed
from all three ranks simultaneously, the ranks were convergent.

Confound, stated plainly: the dump is ~2.5 min before the death, and it shows
the state *after* everything is already stuck. A divergence that occurred
earlier and then converged into a shared block is not excluded by it. What is
excluded is "the ranks were in different host phases at wedge time".

### (b) Static graph divergence (different kernel lists baked per rank) — NOT SUPPORTED

This is the hypothesis `barlink_capture_census` was built to test. The three
census files in `/spinning/wedge-catch-603b/` are **byte-identical apart from
the rank number in the header comment** (lines 2-867 identical; md5s differ
only for that reason). Twelve segments, all `full/ShapeKey(size=1..4)` with
`#2`/`#3` variants, identical collective sequences, sizes and variants across
ranks.

Confound, and it is a real one: those files are timestamped 03:29:57, i.e.
they belong to the boot that came **after** the crash (pgid 522432), not to
the boot that crashed. They establish that this tree and configuration
normally capture rank-identical graphs; they do not prove the crashed boot
did.

### (c) The 8-byte control-plane collective as the trigger — WEAK

Both messages name an 8-byte op, which is suggestive. Against it: the message
itself states the op is from a closed window and cannot be the culprit, and
`_last_op` is stored on captured launches too, so at a replay boundary in the
steady state it is simply whatever was recorded last at capture time.

Worth noting rather than dismissing: the capture census shows tiny in-graph
broadcasts really are baked into every decode graph —
`broadcast|64` and `broadcast|32` from `spec_utils.py:138<spec_utils.py:190`
appear in every segment. So a small control-plane collective inside the
replayed graph is a live candidate for the aborting kernel. Nothing available
today can confirm which one, because nothing named the in-graph kernels.

### (d) Relation to #603's desync-hoist — NOT ESTABLISHED

No evidence in this window's artefacts bears on it either way. Recorded as
untested, not as excluded.

---

## 5. Verdict

**No root cause is claimed.** The two leading structural hypotheses (a) and
(b) are each weakened by direct evidence. The one concrete, actionable finding
is the unwired `total_tokens` channel at the spec-verify call site, and
NOTE_616c's own recorded conclusion is that fixing a pin site relocates this
wedge rather than curing it.

What is established:

* the abort is at the **decode** full-graph replay boundary, both specimens;
* the ranks are **convergent**, not divergent, at the observed wedge instant;
* the captured graphs are **rank-identical** on this tree and configuration;
* the host pins at a blocking D2H whose sync-free channel exists but is wired
  at 1 of 5 call sites;
* every instrument that could name the aborting kernel is blind inside a
  replay **by construction** — the collective census counts host calls and a
  replay makes none.

That last point is why the deliverable of this window is an instrument rather
than a fix.

---

## 6. What the next specimen will say that this one could not

`barlink_abort_gate` now carries a rank-local replay tag written at every
replay launch, and `Bar1CollectiveAborted` prints it. A GRAPH-REPLAY abort now
ends with, e.g.:

```
REPLAY WINDOW (#622): full key=ShapeKey(size=2, stream_idx=None,
variant_label=None) (replay #418713 on this rank).
```

Diffing that line across the three ranks decides between the surviving
hypotheses directly:

* **different windows across ranks** — host-path divergence, hypothesis (a)
  revived, and the differing keys say where;
* **the same window on every rank** — divergence is inside one graph. The
  abort path now also dumps that rank's **capture census**, so the recorded
  kernel list for the named segment is in the same log, on the rank that died
  first. Previously the capture census was dumped only from the scheduler's
  periodic tick, which the first rank to die never reaches — the 08-05
  specimen has rank 0 raising 10 s ahead of the others for exactly that
  reason.

The tag is written host-side at launch time, outside any capture: no device
read, no synchronization, no collective, no `.item()`. Sites: the full and
breakable graph backends, `BreakableCUDAGraph.replay` per segment, and the
multi-layer draft runner per rung.

---

## 7. Follow-ups this window did not take

1. **Wire `total_tokens` at the four remaining call sites**, starting with
   `flashinfer_backend.py:7160`. Removes a blocking D2H from the spec-verify
   hot path. Expect the wedge to relocate, per NOTE_616c §11.
2. **`test_barlink_bar1_abort_deferred_517.py` is broken on `f7df966ebc`** —
   7 of its own tests fail on a pristine checkout. Its `_transport` builder
   constructs via `__new__` and has fallen behind `__init__`, omitting
   `_abort_code_seen`, which `_read_status_for_check` reads. A test helper
   that drifts from the object under test is how a guard stops guarding.
3. **The wedge catcher never fires on the crash itself.** `catch.log` has zero
   hits for `abort`, `Bar1`, `replay` or `census`; its 241 dumps are
   liveness-timeout wedges. It caught 03:23:06 but nothing at 03:25:30, and it
   has no coverage at all before 2026-08-06 — the 08-05 specimen has no dump.
