# HANDOFF 688 — #656 / #631 Route A, successor 44

The shift that ran the last #659 leg on metal and found that the claim is
false. **#659 DOES NOT CLOSE.** C26's fix held exactly as advertised — the
instance survived the park that used to kill it — and behind it stood a
different, sharper defect that no previous shift could have seen, because no
previous shift ever got a session through the round trip alive. Errors first.

---

## 0. THE ONE-LINE STATE

**A session spilled, parked to the NVMe tier, unparked cleanly, and then
finished ON HOST without its output ever reaching the client, which blocked
forever against an idle scheduler.** The tier is sound; the COMPLETION is
not. Booked as **C28**. C26 is proven on metal and closed.

---

## 1. ERRORS FIRST

### 1a. THE MEASURED LIFECYCLE, BOTH RANKS IDENTICAL

Probe boot: TP=2 on the two 3080s, port 30040, `probe_boot_v7.sh` =
`probe_boot_v5.sh` + `--chunked-prefill-size 256` (HANDOFF_687 §4's knob) +
a fresh park dir. Boot commit `cd761abbb6`. One request, `rid=s44-sat-3`:

```
SPILL(partial): rid=s44-sat-3 L=3075 boundary=3010 device_head=3010 host_tail=65
first spill tick for rid=s44-sat-3 (L=3075 spec=False)
destinations: park PENDING rid=s44-sat-3 (tick stops; transfer starts once settled)
destinations: PARK start  rid=s44-sat-3 tier=file L=3372 boundary=3010 rows(rank)=362 region=0
destinations: PARK commit rid=s44-sat-3 tier=file region 0 freed (parked=1)
destinations: UNPARK start  rid=s44-sat-3 tier=file rows(rank)=362 -> region 0
destinations: UNPARK commit rid=s44-sat-3 -> region 0 (host-resident again)
  ... 2 s later ...
spilled session rid=s44-sat-3 finished on host; released device head=0
  (boundary=3010 protected=3010) + tree lock + mamba + req slot + region 0
  (no radix insert); admission gate reset
```

**Everything the tier is responsible for went right.** 33 blobs / 5 939 200
bytes on the file tier at the peak, `identity_miss=0`, zero `UNPARK ...
FAILED`, zero `is a MISS`, both ranks logging the same records at the same
iteration. The park/unpark machinery N42 proved in isolation works inside a
live request.

**And the request never finished for the caller.** Verified, not inferred:

* the driver's own py-spy shows **exactly one** worker thread parked in
  `urlopen` and every other worker idle, while the scheduler logged
  `#running-req: 0, #queue-req: 0`;
* **40 further requests** driven through the server did not release it, so
  this is not an idle-scheduler / tick-starvation stall;
* `POST /abort_request {"rid":"s44-sat-3"}` returned **200 and did nothing**,
  which is consistent with step 8 having already released the req slot —
  there was no longer anything to abort, and the waiter was orphaned.

### 1b. THE SESSION NEVER CAME BACK TO THE DEVICE AT ALL

`RESTORE complete: rid=` appears **zero** times in the entire boot. The
session finished on the host floor. Two consequences, and the second is the
one a successor must not lose:

1. The unpark restored it to *host residency*, not to the device. The
   round trip the tier advertises ("wave-back/restore unchanged from here")
   never proceeded past its first half.
2. **Spec item 13 — a restored session back on CUDA graphs — was never
   reachable in this run.** It is not "unproven pending a spec-enabled
   boot"; the session never returned to the device under ANY execution mode,
   so the graph question was never posed. Do not carry item 13 forward as
   merely blocked on speculation.

### 1c. THE RESTORE GATE COULD NOT HAVE FIRED, AND THE NUMBERS ARE IN THE BOOT LINE

The instance armed with `restore_margin=4096` (`kv-session-offload (S4)
armed: ... restore_margin=4096 hysteresis=4 host_pool=49153 tokens/rank
max_spills=1 region=32770 tokens`) against `max_total_num_tokens=4096` —
**the restore margin equals the ENTIRE KV pool.** The gate is

```python
fits_now = restorable >= remaining + self.restore_margin_tokens
```

so on this boot `fits_now` demands the whole pool plus the tail. Whether that
alone explains the host finish is settled in §2; it is recorded here because
it is visible in the boot banner and a successor reading only the banner
would already have grounds for suspicion. **The margin is not sized against
the pool anywhere** — a default in tokens meets a pool whose size is a
different decision, and on a small probe pool the two collide.

### 1d. AN INSTRUMENT DEFECT I SHIPPED AND CAUGHT BEFORE IT COULD LIE

The first driver (`park_complete_proof.py`) chose a TARGET request in
advance and compared that request's output against a quiescent reference. On
metal the server parked a **different** request — the spill victim is the
server's choice, not the test author's. The driver computed the attribution
term (`target_named_in_park_records`) **and never gated on it**, so it would
have printed PROVEN on a run where the session it measured never parked and
the session that parked was never measured. Same shape as the `proof_driver2`
defect one level in: the instrument held its own disproof and did not consult
it. `park_complete_proof2.py` removes the prediction instead of improving it
— a HOMOGENEOUS cohort (every member the same prompt and sampling as the
reference) with the arms assigned AFTER the fact from the server's own
`PARK commit rid=` / `UNPARK commit rid=` records. Booked as **law 18**.

### 1e. THE BYTE-IDENTITY ARM WAS NOT REACHED, AND WOULD HAVE BEEN INVALID AS PLANNED

The brief asks for output byte-identical against a never-parked same-boot
reference. That claim could not be evaluated: the parked session produced no
client-visible output at all. It also must not be attempted the naive way.
HANDOFF_686 already booked the standing rule — two model generations differ
on this rig with **zero** parks, because batch composition is not invariant
here — and this shift confirmed the mechanism independently: GDN *decode* is
per-sequence and batch-invariant, but prefill chunking, the GEMM/MoE path and
spec verify are not. The honest route, for whoever runs this next:
`--enable-deterministic-inference --attention-backend triton` (the 3080s are
SM86, so the deterministic default `fa3` is Hopper-only and will not boot —
triton is on `DETERMINISTIC_ATTENTION_BACKEND_CHOICES` and is the one that
works here), plus the cohort design of `park_complete_proof2.py` so a
mismatch can be attributed rather than guessed.

---

## 2. ROOT CAUSE — C28, AND IT IS AN ALIAS

**The session deleted its own completion from the list that was about to
carry it.**

`release_finished_spilled_req` runs from INSIDE the per-request loop of
`process_batch_result_decode` (`batch_result_processor.py:932-938`, gated on
`req.kv_spill_state == "host"`), and it calls `slot.batch.filter_batch()`
(`kv_session_offload.py:5868`). For a spill tick, **`batch` IS `slot.batch`,
one object under two names** — `maybe_take_tick` returns the persistent batch
(`kv_session_offload.py:4930, 4974`), the scheduler runs it as
`ret = spill_tick_batch` (`scheduler.py:4865`), and under
`--disable-overlap-schedule` (which this boot sets, and which the ship config
sets too) `event_loop_normal` passes it through unchanged
(`scheduler.py:2232-2233`).

`filter_batch` keeps only unfinished reqs and, when none survive, rebinds
`self.reqs = []` (`schedule_batch.py:3273-3277`). The enclosing `for` loop
completes normally because it already holds the old list. Then, ~100 lines
later:

```python
self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
```

(`batch_result_processor.py:805`) reads the NEW, EMPTY list.
`_stream_output_generation([])` accumulates nothing, `payload is None`,
nothing is sent, the detokenizer never sees a finished chunk, and the
tokenizer manager's waiter loops forever. That is the hang, exactly.

**The fix already exists — on the other exit.** The ABORT exit has
`_stream_terminal` (`kv_session_spill_destination.py:1537-1546, 1562-1572`),
added by `bcc72dd569 [#552]` under the comment *"Without it the abort frees
the memory and the caller hangs."* The FINISH exit never got it;
`kv_session_offload.py` contains **zero** occurrences of `stream_output`,
`output_streamer`, `send_to_detokenizer` or `_stream_terminal`. The same hole
sits on the pre-schedule reap (`kv_session_offload.py:4520-4524`). Booked as
**law 19** (a state has more than one exit) and **law 20** (a cleanup inside
an iteration must not mutate what the iteration's output depends on).

**Why every spilled session on this boot landed there — contributing, not the
wedge.** Restore was unreachable twice over, so finishing on host was forced:
an early return before the gate whenever any fast-lane request waits
(`kv_session_offload.py:4707-4712`; `s44-target` waited throughout), and the
gate itself demanding `restorable >= remaining + restore_margin` = `362 +
4096` **against a whole pool of 4096** (§1c). A restored session would have
finished in the device batch and streamed normally — which is why
`RESTORE complete:` = 0 and the hang co-occur.

**The park is exonerated.** `_commit_unpark` re-inserts the same slot object
and restores every flag (`kv_session_spill_destination.py:1479-1482`);
`slot.batch` survives the round trip untouched. Had the session finished
WHILE parked, `_release_parked_req` would have streamed it and the client
would have been served.

### 2a. THE FIX IS NOT YET WRITTEN, AND THE FALSIFIER IS CPU-ONLY

Do not start with a GPU boot. What the metal run witnesses is that nothing
was streamed and that the path has no emit; it does NOT witness that the
finish was processed on the spill-tick batch rather than via
`release_kv_cache` (`mem_cache/common.py:1098` — equally emit-less, but a
DIFFERENT fix site). Settle that first, on CPU:

* drive `process_batch_result_decode` over a one-req spill-tick batch with
  `slot.batch is batch` whose req finishes, and assert the streamer receives
  a non-empty list. **It will receive `[]`.** That is the red test.
* Secondary prediction that discriminates the alias from a generic missing
  emit: the same boot **without** `--disable-overlap-schedule` should NOT
  hang, because `event_loop_overlap` snapshots `batch.copy()`
  (`scheduler.py:2315`, `schedule_batch.py:3474-3479`). If it hangs anyway,
  the alias is not the whole story and the emit is missing outright.

Both exits must end up covered, and the restore-margin sizing (§1c) is a
separate, real finding that should not be folded into the same commit.

---

## 3. THE EVIDENCE

| axis | result |
|---|---|
| park to file tier | 33 blobs, 5 939 200 B at peak, both ranks |
| `identity_miss` | **0** |
| `UNPARK ... FAILED` / `is a MISS` | **0 / 0** |
| C26 (the crash N43 fixed) | **did not recur** — no assert, no traceback, instance healthy afterwards |
| `RESTORE complete: rid=` | **0** — the session never rejoined the device |
| client outcome | **hang**, one worker in `urlopen`, scheduler `#running-req: 0` |
| unblocked by 40 further requests? | **no** |
| `abort_request` on the rid | 200, no-op (req slot already released) |

Artifacts, all under `/spinning/evidence-631/s44/`:
`WEDGE_AFTER_UNPARK.txt` (the whole lifecycle, quoted from the server log),
`probe_boot_v7.sh`, `park_complete_proof.py` (v1, defective — kept as the
record of §1d), `park_complete_proof2.py` (v2, cohort design),
`park_complete.log`, `v1_pyspy.txt`, `probe_v7.log`, `pyspy_before_stop.txt`
and `live_pids.txt` (the ship config as it stood before I stopped it).

---

## 4. THE RIG AS I LEAVE IT

* **Serving is UP on 30030, ship config, boot commit `cd761abbb6`**, rebooted
  by me from `/spinning/evidence-631/s43/boot_ship_30030.sh` after I verified
  that script byte-for-byte against the live `/proc/<pid>/cmdline` BEFORE
  stopping anything. I stopped it; I brought it back. Nobody owes a restore.
* Router 30099 untouched. Probe stopped by PID, all three cards released to
  3 MiB before the ship reboot.
* **BOOT THE SHIP CONFIG WITH `setsid`, NOT `nohup ... &` FROM A TOOL CALL.**
  I lost my first restore to this and it is worth one paragraph so nobody
  repeats it. The instance came up healthy, served a verified generate, sat
  idle for a minute — and then died at 03:36:38 with **no fault of its own**:
  PP1/PP2 raised `RuntimeError: Connection closed by peer` on the gloo PP
  chain and the ranks logged *"maybe TCPStore server has shut down too
  early"*. There is no exception on PP0 and no CUDA error anywhere; the
  parent simply vanished when the launching tool call's process group was
  torn down. `nohup` covers SIGHUP to the immediate child and does not cover
  a process-group kill. The second restore used
  `setsid <script> > log 2>&1 < /dev/null &`, which puts the instance in its
  own session, and it survived. **Read this shape correctly if you meet it:
  ranks dying on peer-closed connections with a clean PP0 is an EXTERNAL
  death, not a member of the #622/#649 crash family, and chasing it as a
  distributed bug would burn a shift.**
* Worktree `feat/route-a-631`, pushed to the fork.

## 5. WHAT THE NEXT SHIFT SHOULD DO, IN ORDER

1. **Fix C28** (§2). It is the whole of #659 now: the tier is proven, the
   completion is not.
2. Re-run the probe with `probe_boot_v7.sh` and `park_complete_proof2.py`.
   The pressure band reproduces the park reliably — this shift hit it on the
   FIRST attempt with `--chunked-prefill-size 256`, `SAT=4`, `FAST_SHOTS=3`,
   `HOSTGIB=1.5`, `MAXSPILLS=3`. That recipe is now known-good; do not spend
   a shift rediscovering it.
3. Only then the byte-identity arm, per §1e.
4. `#559` merge-backlog triage was NOT started (§6).

## 6. WHAT IS NOT DONE, STATED SO NOBODY READS IT AS DONE

* **#659 is NOT closed** and this shift did not fix C28 — it found and
  characterised it. Stating that plainly is the point of this section.
* **#559 merge-backlog triage was not begun.** The probe, its root cause and
  the ship restore consumed the shift. Nothing about #559 is in this
  document, and its absence should not be read as "nothing to merge".
* The restore-margin sizing question in §1c is raised, not settled beyond
  what §2 establishes.
