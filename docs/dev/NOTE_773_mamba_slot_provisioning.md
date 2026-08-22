# NOTE_773 -- mamba state-slot provisioning: acceptance criteria and determination

Status: determination. No code change proposed by this note for the standing
PP boot; the remaining axis is named in section 6 with its risk profile.

## 1. Why this note exists

`#773` had no canonical register text and no acceptance criteria anywhere in
the tree. `docs/dev/*773*` did not exist; `NOTE_755` / `NOTE_743` / `NOTE_773`
are referenced by commit messages but were never committed (`d60b11a258
--stat` shows zero file changes). The specification existed only as commit
messages. Anyone re-opening the ticket therefore had no way to tell finished
work from unfinished work, and the ticket was re-tasked at least once on the
premise that nothing had been built.

This note fixes the number to a definition, so the next reader can decide
"done" or "not done" from the tree rather than from recollection.

## 2. Acceptance criteria (definition, previously absent)

`#773` is "lower the mamba slot provisioning". It is satisfied when all of the
following hold on the configuration under test:

* **A1 -- no structural over-provisioning.** The boot-sized state pool exceeds
  `mamba_hard_floor(server_args, max_running_requests)` only by a budget that
  has a named consumer. A pool larger than floor + named-consumer budget is a
  defect.
* **A2 -- no admission regression.** The stated `--max-running-requests` stays
  fully servable. Concurrency degrading below the stated value (sessions
  queueing for a state slot) fails acceptance, whether or not it OOMs.
* **A3 -- the surplus is actually spent.** Bytes kept out of the state pool
  reach the KV pool through the existing sizing route, with no second ledger
  and no transfer step.
* **A4 -- the floor is never promised below what a path can hold.** Any term
  dropped from `mamba_slots_per_running_req()` must correspond to a mechanism
  that every path under that config actually takes. Dropping a term the code
  does not implement is the `#581` late `alloc_req_slots` assert, and is the
  failure mode this ticket must not reintroduce.
* **A5 -- observable at boot.** The decomposition (pool, floor, retention
  budget) is visible in the boot log, so A1 can be checked per boot rather
  than argued.

## 3. The configuration this determination covers

The standing Arm-2 boot, `integ/802-ab-boot` @ `dd6967c46f`, served from
`/spinning/wt-802-ab`:

    --pp-size 3 --tp-size 1 --max-running-requests 8
    --disable-overlap-schedule --enable-hierarchical-cache
    --hicache-write-policy write_through --mamba-slot-reorder
    (no --max-mamba-cache-size, no --gdn-resident-state-slots)

Measured cost, per rank, 20 slots including intermediate caches:
PP0 1.74 GB, PP1 1.01 GB, PP2 0.73 GB.

## 4. Where the 20 slots come from, and what each one is for

`_auto_mamba_demand_size` (`model_runner_kv_cache_mixin.py:1932-1952`):

    ceil(target * ratio * MAMBA_AUTO_SAFETY_MARGIN) = ceil(8 * 2 * 1.25) = 20

* `target = 8` is the stated `--max-running-requests`
  (`_auto_mamba_target_concurrency`, `:1908-1930`).
* `ratio = 2` is `_calculate_mamba_ratio()` (`:2465-2509`), whose raw `3 + 1`
  is capped at `mamba_slots_per_running_req()` by the `#773` fix in
  `2ec0e664d4`. Without that cap the demand path derived 30 where the running
  set holds 20.
* `MAMBA_AUTO_SAFETY_MARGIN = 1.25` (`:139`) is concurrency headroom, not a
  VRAM fraction.

Per-request is 2, and this is proven from code rather than inferred from the
arithmetic: `mamba_ping_pong_slots()` returns 0 because
`enable_mamba_extra_buffer()` is False -- `mamba_radix_cache_strategy` resolves
to `no_buffer` under `--disable-overlap-schedule`
(`arg_groups/overrides.py:1197-1239`) -- and `mamba_slot_reorder_active()` is
True, so only the shared donation/pin term is added. `1 + 0 + 1 = 2`.

The decomposition is therefore:

    20 = 16 floor (8 running requests x 2 slots) + 4 retention budget

The 16 are **structurally required** by the stated concurrency: each running
request persistently holds its active state slot plus the anchor pinned at
admission (`schedule_policy.py:1182`, `:1312`). The 4 are the write-through
pin budget (`mamba_retention_pin_budget`, `mamba_pool_floor.py:218-246`),
consumed by `_mamba_write_through_pin_admissible`
(`unified_radix_cache.py:3435-3469`), which funds host-tier backup of mamba
checkpoints.

Neither part is idle over-provisioning. **A1 is already satisfied.**

## 5. What `#773` already delivered (all present in the booted stand)

Work already in the tree, contrary to the premise that the ticket was untouched:

* the hand-pinned `--max-mamba-cache-size 24` was dropped in favour of
  demand-driven sizing, deriving 20 -- the reduction the ticket asked for;
* the sizing ratio was capped at the floor's own per-request count
  (`2ec0e664d4`), removing a 30-vs-20 disagreement between the two
  single-sources-of-truth;
* the `#755` lock reorder was ported to the unified lineage
  (`_mamba_anchor_early_release`, `unified_radix_cache.py:995`;
  `dec_mamba_lock_only`, `:973`; `anchor_release_admissible`,
  `mamba_component.py:523`), which is why
  `UNIFIED_LINEAGE_IMPLEMENTS_SLOT_REORDER` is True and per-request is 2
  rather than 3;
* the `#581` write-through pin budget, previously enforced only in the dead
  `HiMambaRadixCache`, was moved onto the live lineage
  (`write_backup`, `unified_radix_cache.py:1921`);
* the boot log reports `pool / floor / retention_budget`
  (`unified_radix_cache.py:497`), satisfying A5.

## 6. What is NOT available on a PP boot, and why

The `#364` idle-vacate ladder cannot contribute here, and this is a config-level
exclusion rather than a runtime gate. `--enable-kv-session-offload` is refused
at argument-parse time when `pp_size > 1` (`server_args.py:7958-7994`). Its
spilled set is the ladder's standing population of idle slot holders, so on a
`pp_size=3` boot `scheduler.kv_session_offload` is None and the ladder's
`idle_holders()` closure (`gdn_slot_runtime.py:278-289`) returns `[]`
unconditionally. The ladder arms and never fires. The shipped help text for
`--gdn-resident-state-slots` states this together with its measurement: four
concurrent sessions against a four-slot pool completed correctly but
concurrency degraded to three with one queued.

Consequently `--gdn-resident-state-slots` cannot help on this boot either.
`_validate_gdn_resident_state_slots` (`server_args.py:14651-14707`) refuses any
cap below the floor of 16 -- correctly, because A2 forbids exactly the
degradation the help text measured -- and `cap_is_binding`
(`gdn_slot_ladder.py:240`) ignores any cap at or above the profiled 20. The
legal range is 16..19, and at 16 the retention budget is 0, which permanently
skips mamba write-through backups (soft, throughput-only: device-resident hits
are unaffected). The tree is internally coherent here; there is no missing
guard.

The refusal names its own route out, and it is a sourcing problem rather than
a gate to lift: "a PP-safe idle-session SOURCE, not lifting this refusal ...
the stated reason for the refusal -- host pool rows sized from the boot vector
-- is real." `GdnSlotRuntime` already takes its inventory through an injected
`idle_holders_fn` (`gdn_slot_runtime.py:90`, `:272-292`), so a second source
would be additive. What is missing is not the seam but the population: see
section 6.1.

The same refusal records that this costs a named spec axis. Under `pp_size>1`
the `#656` flip-setup capacity spec's "bs2-4 reserves, INCLUDING unused mamba
states, are spilled during bs1 time" is structurally unreachable, and
`--phase-flip-spill-depth` is explicitly *not* its substitute: that ladder
spills the inactive layout's cold memory at the flip seam, not idle-session
mamba state. Its rung count must not be reported as satisfying that axis.

### 6.1 The crux: the cost is the allocation, not the occupancy

The reduction being asked for ("at idle the pool should hold only active and
soon-needed slots") cannot be delivered by vacating, on any parallelism
layout, for a reason that precedes the PP exclusion.

The VRAM is the state pool's **allocation**, made once at boot. It does not
scale with how many slots are occupied. At idle with no traffic all 20 slots
are already free, and the rank still holds 1.74 GB on PP0. Vacating an
occupied slot returns it to the pool's own free list -- "a vacated slot is
reused by another session, not returned to the KV pool"
(`model_runner_kv_cache_mixin.py:2329-2331`) -- so it frees nothing the
allocator can hand to KV.

The ladder's value is therefore not runtime reclamation. It is that a pool may
be **boot-sized below the floor** because overflow sessions survive by
vacating instead of queueing. On a PP boot that survival mechanism is absent,
so the pool cannot be booted below the floor, and the floor validator
correctly refuses to let it try.

Runtime return of freed bytes to the KV pool does not exist and is explicitly
out of scope for the cap flag ("boot-time sizing only"). Both pools are fixed
at boot, so a runtime-vacated slot has no consumer to give its bytes to.

## 7. Determination

**For the standing Arm-2 PP boot, `#773` is complete.** The pool is floor plus
a named-consumer budget; the floor is set by a stated concurrency that A2
requires to remain servable; and the mechanisms that could trade further
(`#364` ladder, resident-slot cap) are excluded by `pp_size > 1` or bounded by
the same floor.

Lowering the number further requires one of:

* lowering `--max-running-requests`, which A2 excludes; or
* runtime-resizable state and KV pools, which do not exist and whose freed
  bytes would currently have no consumer; or
* the axis in section 8.

## 8. The one remaining axis, and its risk

Per-request could drop from 2 to roughly 1 by releasing the admission-time
anchor pin (`schedule_policy.py:1182`) once the checkpoint's write-through ack
has landed, leaving a running request holding only its active slot between
checkpoints. The floor would then be `max_running_requests + k`, where `k`
bounds requests donating concurrently, and the pool would fall from 20 to
roughly 12 -- about 290 MiB per rank of main state on PP0.

This is a new mechanism, not wiring. Two named incidents sit directly on its
failure path:

* `#767`: releasing on `host_value` alone produced degenerate output in 9 of
  10 salted probes. `anchor_release_admissible` already encodes the stricter
  condition (host copy present AND ack landed), so the guard exists -- but the
  admission pin is a different pin from the anchor the guard was written for,
  and the argument does not transfer for free.
* `#581`: the floor may only drop where every path holds within the reduced
  budget (A4). Bounding `k` requires an admission gate on concurrent
  donations, which does not exist today.

It also needs a genuine multi-turn load validation on GPU, not a unit suite:
the pin being removed is what keeps a matched prefix anchor alive, so the cost
lands on cache reuse and on resume correctness, neither of which a hermetic
test observes.

Recommendation: treat section 8 as its own ticket with the `#767`/`#581`
risk profile attached, rather than as a continuation of `#773`.
