"""Hard floor for the mamba/GDN state pool (`--max-mamba-cache-size`).

Single source of truth for "how many mamba slots does a running request hold
at once". Both the boot-time validation (`--max-mamba-cache-size` refusal /
auto floor) and the hierarchical cache's write-through pin budget derive from
the numbers computed here, so the two can never drift apart.

The floor is a DEMAND model, not a heuristic: every term below is one concrete
allocation site that a single running request can hold simultaneously.

    per running request
      1   active state slot          memory_pool.py HybridReqToTokenPool.alloc
      P   ping-pong track buffer     memory_pool.py _alloc_ping_pong_buffer,
                                     P = mamba_ping_pong_track_buffer_size
                                       = 2 with overlap schedule, else 1;
                                     lazy mode holds 1 and takes the second
                                     transiently at a track boundary
      1   donation slot              mamba_radix_cache.py cache_unfinished_req
                                     allocates the replacement slot BEFORE
                                     donating the tracked one
      1   pinned radix checkpoint    mamba_radix_cache.py cache_unfinished_req
                                     inc_lock_ref(new_last_node): the node the
                                     request resumes from is not evictable
                                     while the request runs

    floor_per_req = 1 + P + 1 + 1
    hard_floor    = max_running_requests * floor_per_req

Anything above `hard_floor` is cache: evictable prefix checkpoints. Anything
below it is a configuration that cannot run to completion at the configured
concurrency, because the last request's own required slots do not exist.

Cross-check against the two constant families that already encode parts of
this model:

* `common.MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3` (active + 2 ping-pong) is the
  ADMISSION charge -- it deliberately omits the donation slot and the pinned
  checkpoint because `alloc_req_slots` evicts to make room for them.
* `model_runner_kv_cache_mixin._calculate_mamba_ratio()` = 3 + 2 = 5 with the
  extra-buffer strategy and overlap is the SIZING ratio, and equals
  `floor_per_req` for that configuration by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

#: Active state slot every running request owns (`req.mamba_pool_idx`).
MAMBA_FLOOR_ACTIVE_SLOTS = 1

#: Replacement slot allocated before a checkpoint donation, plus the radix
#: checkpoint the request is pinned to (`inc_lock_ref(req.last_node)`). Both
#: exist only when the radix cache is on.
MAMBA_FLOOR_DONATION_SLOTS = 1
#: THE GATE IS THE RADIX CACHE, NOT `--mamba-checkpoint-interval`. #743's
#: determination note proposed dropping this term when
#: `mamba_checkpoint_interval is None`, to recover max_running_requests slots.
#: REFUTED, and recorded here rather than only in the note, because the
#: proposal is plausible enough to be tried again by the next reader:
#:
#: * `mamba_ckpt_utils.is_on_interval` returns True unconditionally for
#:   `interval is None`, so the grid check never rejects with the flag unset;
#: * `mamba_radix_cache` therefore inserts a real mamba_value on the ORDINARY
#:   path and pins it via `inc_lock_ref(new_last_node)`, which increments
#:   `mamba_lock_ref` for any node carrying a value;
#: * the admission-side pin of `req.last_node` (`schedule_policy`, every
#:   prefill add) has no interval gate at all.
#:
#: The slot is genuinely held with the interval unset, so removing this term
#: would UNDER-FLOOR the pool and resurrect the #581 late assert -- the same
#: failure the #755 note below warns about for the donation term. The
#: interval gates WHERE resume points may fall, never WHETHER a state is
#: cached or pinned. Full derivation in docs/dev/NOTE_743_slot_observability.md
#: §3; the correct axis is already pinned by
#: test_mamba_pool_floor.py's `test_radix_disabled_only_charges_the_active_slot`.
MAMBA_FLOOR_PINNED_CHECKPOINT_SLOTS = 1


def mamba_ping_pong_slots(server_args: "ServerArgs") -> int:
    """Ping-pong track-buffer slots a running request holds.

    Mirrors `HybridReqToTokenPool.mamba_ping_pong_track_buffer_size`
    (`2 if enable_overlap_schedule else 1`) and the lazy strategy, which
    allocates only the first slot up front and takes the second transiently
    at a track boundary -- transient, but concurrent with everything else the
    request holds, so it counts toward the floor.
    """
    if not server_args.enable_mamba_extra_buffer():
        return 0
    return 1 if server_args.disable_overlap_schedule else 2


#: #755: the env that opts into the lock reorder. Default OFF, so the floor and
#: the runtime are byte-identical to before this existed unless asked for.
MAMBA_SLOT_REORDER_ENV = "SGLANG_MAMBA_SLOT_REORDER"


#: #773: does the UNIFIED lineage implement the #755 lock reorder yet?
#:
#: Flip this to True in the same commit that ports the reorder into
#: `unified_radix_cache` / `unified_cache_components.mamba_component`, and the
#: floor reduction returns on its own. It is a constant rather than a config
#: flag on purpose: it describes what the CODE can do, not what an operator
#: may ask for, and an operator must never be able to assert it.
UNIFIED_LINEAGE_IMPLEMENTS_SLOT_REORDER = True


def mamba_reorder_lineage_supported(server_args: "ServerArgs") -> bool:
    """Does the tree cache this config will actually BUILD do the reorder?

    #773. The #755 gate asks three questions about the CONFIG and none about
    the LINEAGE, and those turn out to select for opposite worlds:

    * the gate requires `enable_hierarchical_cache`, because only a
      write-through host tier can promise the released anchor still exists;
    * but `registry.py` routes a hybrid-SSM model WITH hierarchical cache to
      `UnifiedRadixCache`, and `MambaRadixCache` -- the only class carrying
      the reorder -- is reachable only on the branch below it, i.e. only when
      hierarchical cache is OFF.

    So the reduction was taken exactly where the mechanism is absent, and the
    mechanism sat available exactly where the reduction was not taken.
    `CacheInitParams.mamba_slot_reorder` is filled from the same predicate on
    every boot and read only by `MambaRadixCache`: always False where it is
    read, always ignored where it is True.

    The direction matters. A floor that is too HIGH costs VRAM; a floor that
    is too LOW is #581 -- the boot validates a pool the runtime then
    over-draws, and the shortfall surfaces as a late assert after minutes of
    serving. This module's own rule is that a term may be dropped only if
    EVERY path under the config stays inside the reduced budget; no path does
    when the code that would is not built.
    """
    if UNIFIED_LINEAGE_IMPLEMENTS_SLOT_REORDER:
        return True
    # Mirrors registry.py's routing for the hybrid-SSM case. The caller has
    # already required hierarchical cache, so this is the unified lineage.
    return not bool(getattr(server_args, "enable_hierarchical_cache", False))


def mamba_slot_reorder_active(server_args: "ServerArgs") -> bool:
    """True when the #755 lock reorder may drop the donation/pin double-count.

    THREE CONDITIONS, all necessary, and the reason each is necessary is the
    reason the floor may move:

    1. the radix cache is on -- without it there is no donation and no pin to
       share, and the floor is already 1 + ping-pong;
    2. hierarchical cache is on AND the write policy is write-through -- the
       reorder releases the OLD anchor before allocating the new slot, so a
       failed alloc or an eviction inside that window must degrade to
       ``load_back`` rather than to a dead anchor (NOTE_755 section 3). A
       device-only pool cannot offer that, and write-around/write-back cannot
       promise the backup EXISTS at release time;
    3. the operator opted in;
    4. #773: the lineage this config actually BUILDS implements the reorder.
       Conditions 2 and 4 pull against each other today -- see
       :func:`mamba_reorder_lineage_supported` -- and 4 is what keeps the
       floor from promising a reduction no built class delivers.

    The predicate is CONFIG-level and decides the floor. The per-node question
    -- is THIS anchor backed up right now -- is asked again at the site, and a
    node that is not backed takes the skip path rather than silently reverting
    to the 3-slot order the floor no longer reserves. That split is the whole
    safety argument: the floor may only drop if EVERY path under this config
    stays within the reduced budget.
    """
    import os

    if server_args.disable_radix_cache:
        return False
    if not getattr(server_args, "enable_hierarchical_cache", False):
        return False
    if getattr(server_args, "hicache_write_policy", None) != "write_through":
        return False
    if not mamba_reorder_lineage_supported(server_args):
        return False
    return os.getenv(MAMBA_SLOT_REORDER_ENV, "") not in ("", "0", "false", "False")


def mamba_slots_per_running_req(server_args: "ServerArgs") -> int:
    """Mamba slots one running request can hold simultaneously."""
    slots = MAMBA_FLOOR_ACTIVE_SLOTS
    if server_args.disable_radix_cache:
        # No radix cache: no donation and no pinned checkpoint, the request
        # only ever owns its active state slot.
        return slots
    slots += mamba_ping_pong_slots(server_args)
    if mamba_slot_reorder_active(server_args):
        # #755: the donated slot BECOMES the next pinned checkpoint. The
        # double-count existed only because the old pin was held across the
        # alloc; releasing it first makes the two terms one. NOT a formula-only
        # edit -- mamba_radix_cache's insert flow performs the matching reorder
        # and refuses (skips the insert) for any node it cannot release safely,
        # so no path under this config exceeds the reduced budget. Changing
        # this constant alone would under-floor the pool and resurrect the #581
        # late assert (NOTE_755 section 3).
        slots += MAMBA_FLOOR_DONATION_SLOTS
        return slots
    slots += MAMBA_FLOOR_DONATION_SLOTS
    slots += MAMBA_FLOOR_PINNED_CHECKPOINT_SLOTS
    return slots


def mamba_hard_floor(server_args: "ServerArgs", max_running_requests: int) -> int:
    """Smallest `--max-mamba-cache-size` that can serve `max_running_requests`.

    A pool below this cannot run the configured concurrency to completion: the
    required (non-evictable) allocations of the running set alone exceed it.
    """
    per_req = mamba_slots_per_running_req(server_args)
    return max(1, int(max_running_requests)) * per_req


def mamba_retention_pin_budget(
    server_args: "ServerArgs",
    max_running_requests: int,
    mamba_pool_size: int,
) -> int:
    """Slots cache RETENTION may pin at once: the pool above the hard floor.

    The floor says what the running set structurally requires. Everything
    above it is cache, and cache may pin every slot the running set does not
    require -- and not one more. That is what turns the floor from an
    arithmetic exercise into a guarantee: a REQUIRED allocation (the active
    state slot, the ping-pong buffer) always has somewhere to go no matter how
    far behind the write-through ack drain is, independently of any drain
    rate.

    Without this cap a `--hicache-write-policy write_through` boot pins EVERY
    inserted checkpoint the moment it is created (the write-through threshold
    is 1), the eviction walk skips pinned nodes, and the pinned set ratchets
    until it owns the pool: #581, where raising `--max-mamba-cache-size` only
    buys time because the ratchet scales with the pool.

    Single source of truth for BOTH cache lineages on purpose. The bound used
    to exist only inside `HiMambaRadixCache`, which `registry.py` documents as
    having no construction site anywhere -- so every hybrid-SSM boot under
    hierarchical cache was charged the floor while running unbounded. Deriving
    it here means a lineage cannot silently ship without it again.
    """
    floor = mamba_hard_floor(server_args, max_running_requests)
    return max(0, int(mamba_pool_size) - floor)


def describe_mamba_floor(server_args: "ServerArgs", max_running_requests: int) -> str:
    """Human-readable derivation, for error messages and boot logs."""
    per_req = mamba_slots_per_running_req(server_args)
    pp = mamba_ping_pong_slots(server_args)
    if server_args.disable_radix_cache:
        terms = f"{MAMBA_FLOOR_ACTIVE_SLOTS} active (radix cache disabled)"
    elif mamba_slot_reorder_active(server_args):
        terms = (
            f"{MAMBA_FLOOR_ACTIVE_SLOTS} active"
            f" + {pp} ping-pong"
            f" + {MAMBA_FLOOR_DONATION_SLOTS} donation/pinned checkpoint"
            f" (#755 reorder: the donated slot BECOMES the pin, so the two"
            f" share)"
        )
    else:
        terms = (
            f"{MAMBA_FLOOR_ACTIVE_SLOTS} active"
            f" + {pp} ping-pong"
            f" + {MAMBA_FLOOR_DONATION_SLOTS} donation"
            f" + {MAMBA_FLOOR_PINNED_CHECKPOINT_SLOTS} pinned checkpoint"
        )
    return (
        f"{max_running_requests} running requests x ({terms}) "
        f"= {max_running_requests} x {per_req} = "
        f"{mamba_hard_floor(server_args, max_running_requests)} slots"
    )
