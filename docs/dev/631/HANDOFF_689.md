# HANDOFF 689 — #656 / #631 Route A, successor 45

C28 is fixed and the fix is proven on metal, twice, independently. **#659
closes on the axis it was actually about — a parked session completing to its
client — and it does NOT close on byte-identity, for a reason that is now
understood and booked rather than merely unmet.** Errors first.

---

## 0. THE ONE-LINE STATE

**A session spilled, parked to the NVMe tier, unparked, finished ON HOST, and
its output reached the HTTP client.** That is the round trip successor 44
proved impossible on the pre-fix tree. Two separate cohort runs on the same
boot each produced one, `errors=0`, `identity_miss=0`.

---

## 1. ERRORS FIRST — WHAT DID NOT GET PROVEN

### 1a. BYTE-IDENTITY ACROSS A PARK IS STILL UNPROVEN, AND IT IS NOT A SHORTFALL OF EFFORT

The cohort ran, the arms were assigned from the server's own records, and the
verdict is **NOT ATTRIBUTABLE**:

```
parked_arm_identical_to_reference : False
control_arm_identical_to_reference: False
```

Per `park_complete_proof2.py`'s own verdict table that is the "this run
separates nothing" outcome: the requests that **never parked** also diverged
from the quiescent reference, so the divergence is the rig's known
batch-composition nondeterminism (HANDOFF_686's standing rule) and says
nothing about the round trip. The instrument refused to claim, which is the
correct behaviour and the whole point of the v2 cohort design.

**And the honest route out of it is CLOSED, which is the real finding.**
HANDOFF_688 §1e prescribed `--enable-deterministic-inference
--attention-backend triton`. That combination cannot be booted, and neither
can any other: see C30 below. Byte-identity across a park is therefore not
obtainable on this rig until deterministic inference and kv-session-offload
can coexist. **Do not spend a shift re-attempting it with flag variations —
spend it on C30 or accept the cohort's attribution verdict as the ceiling.**

### 1b. SPEC ITEM 13 IS HALF DONE: THE RESTORE HAPPENED, THE GRAPH CLAIM DID NOT

`RESTORE complete: rid=` fired for the **first time in this whole line of
shifts** — four lines, rank-uniform, `rid=s45-cohort-2` at `L=3316` then
`L=3401` rejoining the device batch. Successor 44 saw zero. So the half that
was previously unreachable is now reached, and it was unreachable for the
reason booked as C29, not because of speculation.

The graph half is **WEAK, i.e. not proven**. `restore_graph_evidence.py`
(written this shift, because `park_complete_proof2.py` deliberately refuses to
infer this) gates on attribution: a graphed decode only counts if the restored
session was the ONLY running request, otherwise the `cuda graph: True` verdict
belongs to somebody else in the batch. Result: 5 decode-stats lines after the
restore, 2 of them graphed, **none of them solo**. So:

* restored session rejoins the device — **FACT, logged, rank-uniform**;
* restored session decodes under CUDA graphs — **NOT ATTRIBUTABLE in this
  run**. Run B (`COHORT=2`, aimed squarely at producing a solo restored
  session) drained cleanly but produced **no restore at all**, so it added
  nothing here.

Do not write item 13 up as closed. What it needs is a run where the restored
session is the last one alive; the pressure recipe that produces a restore and
the condition that leaves it solo are currently in tension.

### 1c. TWO SILENT-DROP SIBLINGS ARE STILL IN THE TREE, DELIBERATELY

The sweep found two more exits that free a session's memory and emit nothing:
the pre-schedule reap (`kv_session_offload.py:4520-4524`) and the tick-batch
drain (`:4931-4934`). Both are reachable only for a request that is already
`finished()` — hence already streamed — so they are **latent, not live**. They
were left alone on purpose: a defensive emit there can double-stream, and
unvalidated defensive code on a hang path is how the next C28 gets written.

---

## 2. WHAT WAS FIXED — C28, AND WHY NOT THE OBVIOUS TWIN

Commit **`b2f18010c2`**. `process_batch_result_decode` now binds `reqs` and
`return_logprob` ONCE, before its per-request loop, and streams from that
binding.

The brief and the register both proposed the literal twin of the abort exit's
`_stream_terminal` — an emit inside `release_finished_spilled_req`. **That
would have been wrong, and the sweep is what showed it.** The alias
(`batch is slot.batch`) exists only when overlap scheduling is OFF. Under the
overlap loops the result processor runs on `batch.copy()` — a list the release
cannot reach — so an emit inside the release would DOUBLE-stream at all three
overlap dispatch sites (`scheduler.py:2314-2315`, the decoupled spill lane at
`:6071-6080`, and `pause_generation`'s drain). The binding fix is immune to
which function does the mutating, and it restores exactly the stream-then-filter
ordering `disaggregation/decode.py:2113-2116` already uses.

Two things fell out of reading rather than assuming:

* **The loss was never confined to the empty-list branch.** `filter_batch()`
  keeps `not finished()`, so a finished spilled request is ALWAYS dropped; the
  survivor branch (`schedule_batch.py:3293`) is equally a rebind. A fix keyed
  on "the batch became empty" would have been incomplete.
* **`return_logprob` is a second casualty.** `filter_batch` recomputes it from
  the SURVIVORS (`:3316`), so a finished request that asked for logprobs would
  have been streamed without them. It is snapshotted too.

### 2a. THE TEST IS RED-FIRST, AND THE RED WAS EXECUTED, NOT ASSERTED

`test/registered/unit/managers/test_host_finish_stream_659.py` drives the REAL
`SchedulerBatchResultProcessor.process_batch_result_decode` over the REAL
`release_finished_spilled_req` and the REAL `ScheduleBatch.filter_batch`.
Verified red by **reverse-applying the patch** (`git apply -R`) and running:
2 failed with the host-finished session absent from the emit, 2 passed; patch
re-applied, 4 passed. The survivor-branch case uses a stand-in (the real
branch filters CUDA tensors and cannot run on the CPU lane) that is **pinned
by a source assertion** which fails if `filter_batch`'s rebinds ever change —
so the stand-in cannot drift into fiction.

**Suite: 1092 passed, 0 failed** (`scripts/run_631_flip_family.sh`, which now
collects the new file — the list is explicit and three under-collections have
happened on it). Plus `test_kv_session_offload_unit.py` +
`test_kv_spill_destination_unit.py`: 164 passed.

---

## 3. THE EVIDENCE, FROM THE PROBE

Boot `probe_boot_v9.sh`, port 30041, TP=2 on the two NVML-resolved 3080s,
boot commit `b2f18010c2`.

| axis | run A | run B |
|---|---|---|
| parked rid | `s45-cohort-3` | `s45b-cohort-1` |
| park → unpark → **completed to client** | **YES** (1744 chars) | **YES** |
| `errors` across the cohort | **0** | **0** |
| `identity_miss` | **0** | **0** |
| `UNPARK ... FAILED` | 0 | 0 |
| `RESTORE complete` | **4 lines, `s45-cohort-2`** | none |
| byte-identity | NOT ATTRIBUTABLE (control also diverged) | — |

Ledger: peak host tier **536 903 680 B**, file tier **33 blobs /
3 760 128 B on disk**. (The `used:park:file` gauge read 0.0 while 33 files
existed on disk — the gauge and the disk disagree; the disk bytes are the
measured quantity and the gauge discrepancy is worth one look by whoever next
touches the ledger.)

**Corridor:** NVML free minimum during the probe **1381 / 32086 / 1381 MiB** —
above the 1024 MiB law with 357 MiB of margin. Note this boot sets
`SGLANG_CORRIDOR_FLOOR_MIB=1536`, so its OWN configured floor was undershot by
155 MiB even though the user law held. **There is no seam census to judge for
this probe: TP=2 with no phase flip has no seam.** The seam-census corridor
question belongs to the ship config, which is where it is measured.

Artifacts under `/spinning/evidence-631/s45/`: `probe_boot_v8.sh`,
`probe_boot_v9.sh`, `probe_v8.log`, `probe_v8_triton_refused.log`,
`probe_v9.log`, `park_complete_result3_runA.json`,
`park_complete_result_runB.json`, `restore_graph_evidence.py` + its two JSONs,
`v8_deterministic_wedge_pyspy.txt`, `v8_wedge_metrics.txt`,
`vram_free_during_probe.txt`, `ship_argv_before_stop.txt`.

---

## 4. TWO NEW DEFECTS, BOOKED SO THEY ARE NOT REDISCOVERED

**C29 — the restore margin is never sized against the pool.**
`--kv-session-offload-restore-margin-tokens` defaults to **4096**, validated
only against `< 0`. The gate is
`restorable >= remaining + restore_margin_tokens`. On a boot whose
`max_total_tokens` is at or below the margin it demands more than the entire
pool and **can never open** — which is the whole reason successor 44 saw zero
restores and every spilled session finished on the host floor. This shift
worked around it **by config** (`--kv-session-offload-restore-margin-tokens 64`
against a 4096-token pool) and deliberately not by code, so the sizing defect
stays visible and earns its own commit. The margin should be expressed against
the pool, with a boot-time refusal when it exceeds it.

**C30 — `--enable-deterministic-inference` and `--enable-kv-session-offload`
cannot both be on.** kv-session-offload REFUSES every attention backend but
flashinfer, so HANDOFF_688's prescribed triton does not boot at all; fa3 is
Hopper-only; flashinfer is the only backend satisfying both constraints and it
silently disables the radix cache. The resulting instance boots, prints "fired
up and ready to roll", serves its two warmup prefills, **and then admits
nothing**: a trivial 8-token `/generate` hung for 55 s, `/health` timed out
while `/get_model_info` answered instantly, the collective census froze at
1862 all_reduce, **zero `Decode batch` lines appeared in the entire boot**, and
py-spy showed TP1 spinning inside `add_one_req` (`schedule_policy.py:1255`)
while rank 0 reported `num_queue_reqs 0`. Attribution is one variable: the same
boot minus those two flags (`probe_boot_v9.sh`, diff = one line) served in
**0.36 s** and reproduced the park on its first attempt.

---

## 5. THE RIG AS I LEAVE IT

* **Serving is UP on 30030, ship config, boot commit `740c4e044c`**, restored
  by me with `setsid` from `/spinning/evidence-631/s43/boot_ship_30030.sh` —
  which I verified against the live `/proc/<pid>/cmdline` BEFORE stopping
  anything: **argv identical 60/60, zero env mismatches, no SGLANG_ var live
  but missing from the script**. Verified with a **real generation** (HTTP 200,
  2.6 s, real tokens), not health alone. I stopped it; I brought it back.
  Nobody owes a restore.
* Corridor after restore: free **1853 / 3458 / 3213 MiB**, all above 1024.
* Router 30099 untouched. Probe stopped by PID; all three cards were at 3 MiB
  before the ship reboot.
* The boot commit is `740c4e044c`, i.e. the ship config is now running **with
  the C28 fix in the tree**. The fix is a no-op for it (the snapshot binding
  changes nothing when nothing filters mid-loop), and the suite is green.
* One operational note inherited from N44 and confirmed again: `setsid`, never
  `nohup ... &` from a tool call. Also: **the wedged v8 instance did not die on
  SIGTERM** — 45 s of TERM did nothing and SIGKILL by explicit PID was needed.
  Budget for that when stopping a wedged instance.

## 6. WHAT THE NEXT SHIFT SHOULD DO, IN ORDER

1. **C29**, its own commit: size the restore margin against `max_total_tokens`
   and refuse at boot when it exceeds the pool. Small, self-contained, and it
   is what makes restores reachable on any small-pool boot.
2. **Spec item 13's second half**: a run where the restored session is the last
   one alive, so `restore_graph_evidence.py` can return STRONG instead of WEAK.
3. **C30** if byte-identity is wanted at all — it is the only door to it.
4. `#559` merge triage — still a listing, see §7 of HANDOFF_688 and §7 below.

## 7. WHAT IS NOT DONE, STATED SO NOBODY READS IT AS DONE

* Byte-identity across a park: **UNPROVEN, NOT ATTRIBUTABLE** (§1a).
* Spec item 13: restore YES, graph attribution **NO** (§1b).
* The two latent silent-drop siblings: **not fixed**, on purpose (§1c).
* The `used:park:file` gauge reading 0 against 33 files on disk: **noticed,
  not investigated**.
* `#559` merge-backlog triage: **not started by this shift** — the shift went
  to the probe, and a listing that is already in HANDOFF_688 §7 is a worse use
  of a GPU window than the leg that closed C28.
