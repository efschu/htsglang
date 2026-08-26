# NOTE 888b — the drain did not starve on KV; it starved on a request seat

Desk work, 2026-08-26, base `0cd27d957d`, branch
`fix/888b-paused-decode-kv-release`. No boot, no card. Everything below is
measured off `boot_w38rerun_0826_1304.log` or read out of the tree at that
pin.

## 1. The ticket's discriminator was wrong about which resource binds

#888 was opened with the discriminator "mamba usage 0.65 (35% of slots FREE),
evictable=0, KV pool 0.97 → the binder is KV-token release from paused
decodes". Remedy (b) was chosen on that basis: make a paused decode give up
its KV.

The boot log refutes the resource, not the phenomenon. Inside the stall
(13:11:00 → 13:13:14, 156 s, ended by the 173.6 s decode-starvation cap):

```
#788 PP-ADMISSION verdict=DECLINE n_reqs=0 avail=12876 evictable=0
     queue=2 running=4 chunked=0
     reason=gate=batch_full_or_empty_queue(batch_is_full=1,queue=2)
```

* `avail` on that line is `token_to_kv_pool_allocator.available_size()`
  (`scheduler.py`, the `#788 PP-ADMISSION` emitter). **12876 free KV tokens
  against 33–46 pending.** The KV pool was not the binder and no KV-token
  relief could have admitted this prefill. 0.97 utilisation of a large pool
  is true and irrelevant.
* Every emitted decline inside the stall names `batch_full_or_empty_queue`
  — 60 of 60 — and **zero** name `no_allocatable_reqs`. The refusal never
  reached a memory test.

The remaining candidate is the one nobody counted: `max_num_reqs =
self.max_running_requests` (`model_runner_kv_cache_mixin.py:3393`), so
`req_to_token_pool` has exactly as many **request seats** as the concurrency
cap — eight on this recipe. Eight carriers were resident, so
`available_size()` was 0, and `get_num_allocatable_reqs` min()s against it.

## 2. Two defects, and the second hides the first

**D1 — the seat is held by a resident the phase may not run.** Under strict
purity PP holds while prefill is pending; the pending prefill needs a seat;
every seat is held by a carrier PP may not decode to completion; nothing
takes a seat back. The phase waits for an event its own occupancy forbids.

#677 phase 1 answered the same wedge with arithmetic — stop charging a parked
carrier to `max_running` (`parked_decode_set.py`). Correct and insufficient:
it removed the cap that was not binding and left the pool that was. A
discount cannot free a seat the discounted carrier is still sitting in.

**D2 — `batch_is_full` is a latch with no reachable clear site in PP.** Its
clear sites are `update_running_batch` and the finish paths, all on the
decode path, which strict purity forbids. One pass with a full seat table
latches it; every later pass returns at `scheduler.py`'s
`(batch_is_full or queue empty)` gate — above the seat test, above #677's
discount, above every relief. That is why the log reports "batch is full" for
156 seconds without re-deriving whether it still is, and why the real binder
was invisible.

The class is already registered: `cutover_participants.py` lists
`latched_batch_flags` with `batch_is_full` named ("the last latched True at
running=0 and refused admission for ever"). Its hook,
`phase_flip_draft_bootstrap.reset_stale_batch_flags`, runs **at the seam**.
This stall happens in the middle of a residency, where no hook runs.

## 3. What was built

`managers/parked_carrier_relief.py` — the DECISION only, pure arithmetic, no
torch, no scheduler reference: may one resident carrier yield its seat this
pass, and **which resource is actually binding**. The binder is named from
measurement (`name_the_binder`) precisely because this stall was
mis-attributed from a utilisation ratio; a verdict carrying
`binder="req_slot"` is falsifiable against a boot log, "memory pressure" is
not.

Wired into `scheduler.py` at two points inside `_get_new_batch_prefill_raw`:

* `_rederive_latched_batch_full` immediately above the flag gate — under a
  recorded phase prohibition the latch is cleared so the gate re-derives.
  Clearing admits nothing; it restores the question.
* `_maybe_yield_parked_carrier` inside the `get_num_allocatable_reqs() <= 0`
  branch, before the flag is latched. On a yield the actuator is #679's
  already-extracted `_retract_decode_and_requeue` (one implementation, three
  call sites, no drift), with `uniform_avail_floor` set first because
  `retract_decode` bounds its loop on it and #583 is exactly the case where
  the entry decision was uniform and the loop bound was not.

**One victim per pass.** A constant bound is rank-uniform by construction;
`retract_decode`'s `first_iter` makes exactly one victim when memory is
plentiful, which is this state. The gate then re-reads the same expression
and still decides — #679's self-clearing park shape.

`SGLANG_PARKED_CARRIER_RELIEF` is a **kill switch, not an opt-in**. #679's
ladder was off on the boot that wedged, which is why it could not have helped;
shipping a second default-off relief for the same wedge would repeat that.
The arming condition is structural instead — the verdict yields only when a
recorded verdict says this layout forbids decode — so a boot without strict
phase purity is byte-identical either way.

## 4. What was NOT done, and why

* **The `phase_purity.validate_purity_policy_pair` refusal stays.** This fix
  weakens its premise ("prefill cannot drain") but does not eliminate it: the
  yield still refuses on a single resident, on an unreconciled parked set, and
  on a binder this module does not measure, and a residency meeting one of
  those has no exit at all. Lifting the refusal needs a loaded window showing
  the drain completing unaided. Pinned by
  `TheBootRefusalIsNotLiftedByThisFix` so a later reader has to delete the
  reasons before deleting the guard.
* **`validate_tp_exit_pair`'s dead escape is filed, not fixed.**
  `phase_purity.py:300` reads `tp_window_s` off the policy config; the field
  exists nowhere in `PhasePolicyConfig` and `git grep tp_window` over
  `python/` hits only lines 300–305 of that file (one test stub sets it to
  0.0). So `or tp_window > 0` at `:301` can never fire and the refusal text at
  `:303-313` advises operators to set a knob that does not exist. It belongs
  to #858b's guard, not to the PP guard #888b addresses, and changing a boot
  refusal without a boot is the thing this note's own section 4 refuses.
* **Nothing is proven on metal.** Desk-built, hermetically tested, never run
  on a card.

## 5. The class, and the sweep

**Class: a resource held by a resident that the current layout forbids to
make progress, with no release path reachable from that layout.** Not
"counter vs actuator" — the actuator (`retract_decode`) exists and is
proven. What was missing is a DOOR to it from the path that needed it, plus
a stale flag sitting above that door.

Siblings, from a sweep of what a parked/spilled request holds:

| resource | released by | reachable while the phase forbids decode? |
|---|---|---|
| request seat (`req_to_token_pool`) | `release_req` → `req_to_token_pool.free` (`mem_cache/common.py:1824`) | **was NO — this fix** |
| device KV rows | same call | same |
| mamba/GDN slot | same call; and #773 §8's "release the anchor pin on write-through ack" | pin release **ABSENT** — NOTE_773 defers it to its own ticket |
| kvso host region | `release_finished_spilled_req` (`kv_session_offload.py:6042`, called from `batch_result_processor.py:987`) and `_release_parked_req` (`kv_session_spill_destination.py:1608`) | **NO — both are finish/abort paths only.** A parked session holds its host region for its whole life. Same shape, host tier, unfixed. |
| draft weights | `PhaseFlipSpillLadder.on_enter_pp/on_enter_tp` | yes, handled at the seam |

**Future-proofing.** `cutover_participants.py` already exists for exactly
this: a participant declares a hook and a reachability probe. What it does
not yet declare is the READ WINDOW's twin — *which paths a resource's release
site lives on, and whether every layout can reach one of them.* A participant
row whose hook is only reachable from the decode path is a defect in a phase
that forbids decode, and that is checkable at desk time the same way the
registry's existing obligations are.
