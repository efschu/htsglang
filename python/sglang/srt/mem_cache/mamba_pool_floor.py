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


def mamba_anchor_ack_release_active(server_args: "ServerArgs") -> bool:
    """#811: may a running request's anchor pin be released at the ack?

    Builds strictly on top of the #755/#773 reorder: the reorder makes the
    donated slot BECOME the pinned checkpoint; this releases that pin the
    moment the checkpoint's write-through backup is ACKNOWLEDGED, so between
    checkpoints a running request holds only its active slot.

    The release itself is gated per node by
    ``MambaComponent.anchor_release_admissible`` -- host copy present AND the
    ack landed (``node.id not in ongoing_write_through``). #767 is the reason
    that per-node gate is non-negotiable: releasing on ``host_value`` alone,
    while the copy is still in flight, resumes requests from a copy that does
    not exist yet (degenerate output in 9/10 salted probes). This predicate
    only decides whether the mechanism is ARMED; it never makes a node
    admissible.

    The exclusions are the dec-site audit (#811): kv-session-offload
    (kv_session_offload.py release_finished_spilled_req), the spill
    destination's parked-request cleanup, streaming sessions
    (streaming_session.py release_session), and the PD-disaggregation decode
    side all call ``dec_lock_ref(req.last_node)`` without DecLockRefParams,
    so an early-released mamba ref would be decremented a second time there.
    Rather than teaching every one of those sites the release marker, the
    feature refuses to arm alongside them.

    Read directly from server_args by user order: no SGLANG_* environment
    form exists for this flag.
    """
    if not mamba_slot_reorder_active(server_args):
        return False
    if not bool(getattr(server_args, "mamba_anchor_ack_release", None)):
        return False
    if getattr(server_args, "enable_kv_session_offload", False):
        return False
    if getattr(server_args, "enable_streaming_session", False):
        return False
    if getattr(server_args, "enable_session_radix_cache", False):
        return False
    if getattr(server_args, "disaggregation_mode", "null") not in (None, "null"):
        return False
    return True


def mamba_slots_per_running_req(server_args: "ServerArgs") -> int:
    """Mamba slots one running request can hold simultaneously."""
    slots = MAMBA_FLOOR_ACTIVE_SLOTS
    if server_args.disable_radix_cache:
        # No radix cache: no donation and no pinned checkpoint, the request
        # only ever owns its active state slot.
        return slots
    slots += mamba_ping_pong_slots(server_args)
    if mamba_anchor_ack_release_active(server_args):
        # #811: the merged donation/pin term moves off the per-request floor
        # and into the retention pin budget. A pin now exists only while the
        # checkpoint's write-through backup is IN FLIGHT (taken only when the
        # backup was admitted by the pin budget, released at the ack), so the
        # number of simultaneously pinned checkpoints is bounded by
        # `mamba_retention_pin_budget`, not by the number of running
        # requests. Worst case protected = floor (actives) + budget = pool.
        # NOT a formula-only edit: unified_radix_cache gates the pin-take and
        # performs the ack-time release; dropping this term without those
        # mechanisms is the #581 late assert (see NOTE_755 section 3).
        return slots
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


class Weg2PMambaRetentionZero(RuntimeError):
    """H113: group P of a weg2 boot with a mamba retention pin budget of 0.

    On group P the host backups of the mamba checkpoints are not cache, they
    ARE the hand-off: D reads the KV pages and the recurrent anchors P
    published through the host tier (arena/store). A budget of 0 declines
    every mamba write-through pin (``_mamba_write_through_pin_admissible``
    compares ``pins_held < 0``), the first chunk-end node carrying a mamba
    value is refused (``#1421 BACKUP-REFUSED why=mamba_pin``) and every
    descendant after it is refused as ``parent_unbacked`` -- so P publishes
    only the anchor-less pieces before the first checkpoint and nothing else,
    on every request, for the whole life of the boot. Measured on
    fnFL2h91bb2 (e17bd548b5, 2026-09-26): ``MAMBA-FLOOR pool=32 floor=32
    retention_budget=0 (8 running requests x 4)``, the 97k needle reached the
    arena with 192 of 1528 pages (the three KV-only pieces of chunk 1), D's
    ``#1439 ARENA-PRESENT leading_complete=192``, ``#1028B FETCH CAP mamba
    (0,-1)``, ``#1035c ZERO-ANSWER cause=CAPPED by=mamba`` and ``#1471 SETTLE
    read still short``. That read can never complete -- the bytes were never
    written -- so the settle wait is not latency, it ends in a short release
    and a second prefill that is refused the same way. Boots xsn127-134 were
    the same class at budget 1.

    Refused at tree construction, where the budget is the EXACT number the
    admission check later compares against (``_mamba_pin_budget``): no
    second derivation of the per-request slot count exists anywhere.
    """


def weg2_p_mamba_retention_refusal(
    server_args: "ServerArgs",
    max_running_requests: int,
    mamba_pool_size: int,
    pin_budget: int,
    group: str,
) -> "str | None":
    """H113: the refusal text when group P would run with pin budget 0, else None.

    Only group P (``SGLANG_WEG2_GROUP=P``) with a hierarchical cache: there
    the budget is a LIVENESS term of the flip, not a cache-size preference.
    Group D and a non-weg2 boot keep the #773 posture (budget 0 is stated,
    not refused). A negative budget means "no mamba pool" and is never
    refused.

    WHAT THE 4 SLOTS PER SEAT ARE on today's P form
    (``--hicache-write-policy write_back --disable-overlap-schedule
    --mamba-slot-reorder``), each one a concrete allocation site:

    * 1 active state slot (``req.mamba_pool_idx``);
    * 1 ping-pong track buffer (extra buffer, overlap schedule off -> 1);
    * 1 donation slot (allocated before the tracked one is donated);
    * 1 pinned resume checkpoint (``inc_lock_ref(new_last_node)``).

    ``--mamba-slot-reorder`` does NOT merge the last two on P:
    :func:`mamba_slot_reorder_active` requires ``hicache_write_policy ==
    "write_through"`` (the reorder releases the old anchor before the alloc
    and needs the backup to EXIST at release time; write_back cannot promise
    that), and #811 ack-release builds on the reorder. So write_back costs the
    reorder's rebate (3 -> 4 per seat) and the ack-release's (2 -> 4).
    ``describe_mamba_floor`` prints the branch the boot actually took.
    """
    if str(group or "").strip().upper() != "P":
        return None
    if int(pin_budget) != 0:
        return None
    if not bool(getattr(server_args, "enable_hierarchical_cache", False)):
        return None
    mrr = max(1, int(max_running_requests))
    per_req = mamba_slots_per_running_req(server_args)
    floor = mamba_hard_floor(server_args, mrr)
    return (
        f"H113 WEG2 P MAMBA RETENTION ZERO: --max-mamba-cache-size "
        f"{int(mamba_pool_size)} leaves group P a mamba retention pin budget "
        f"of 0 -- pool {int(mamba_pool_size)} - floor {floor} "
        f"({describe_mamba_floor(server_args, mrr)}). Every mamba "
        f"write-through backup would be declined ('#1421 BACKUP-REFUSED "
        f"why=mamba_pin', then 'parent_unbacked' for the rest of the chain), "
        f"so P publishes no recurrent anchor and D's hand-off read can never "
        f"complete (fnFL2h91bb2: 192 of 1528 pages, '#1471 SETTLE read still "
        f"short'). Minimum: --max-mamba-cache-size {floor + 1} "
        f"(= {mrr} seats x {per_req} + budget 1); the reference form h91v1 "
        f"ran budget 8 (pool 24 at 4 seats), i.e. {floor + 8} here. Or lower "
        f"--max-running-requests / --p-bs to at most "
        f"{max(0, (int(mamba_pool_size) - 1) // max(1, per_req))} for this "
        f"pool."
    )


def mamba_phase_pin_budget(
    server_args: "ServerArgs",
    max_running_requests: int,
    mamba_pool_size: int,
    allocator=None,
) -> int:
    """:func:`mamba_retention_pin_budget` of the pool AS IT IS REACHABLE NOW.

    H95e. A D phase of n < --d-bs seats hands out only slots 1..L(n)
    (``MambaSlotAllocator.set_phase_limit``, H95c) and admits at most n
    running requests (``d_seat_vram.admission_cap``). The budget's argument
    -- floor (the running set) + budget = pool -- then holds for THAT pool and
    THAT running set: ``L(n) - n x slots_per_running_req``. Taken from the
    boot pool and the boot's --max-running-requests instead, the budget stays
    at the cap form's value in every phase: NF D (38 slots, --d-bs 6, 5 slots
    per request) keeps 38 - 30 = 8 while n=1 reaches 7 slots, so write-through
    pins could hold every reachable slot and the floor is no guarantee any
    more. Per phase: 7-5=2, 13-10=3, 19-15=4, 25-20=5, 32-25=7, 38-30=8.

    ``allocator`` is duck-typed (``phase_limit``, ``phase_seats``); without a
    limit this is exactly :func:`mamba_retention_pin_budget`. A limit without
    a seat count charges the boot's --max-running-requests -- the larger
    floor, so the smaller budget (fewer host backups, never a starved slot).
    Replicated: the limit and n come from the wake request, the floor from
    server_args -- every rank computes the same number, no collective.
    """
    limit = getattr(allocator, "phase_limit", None) if allocator is not None else None
    if limit is None or int(limit) >= int(mamba_pool_size):
        return mamba_retention_pin_budget(
            server_args, max_running_requests, mamba_pool_size
        )
    seats = getattr(allocator, "phase_seats", None)
    running = max(1, int(max_running_requests))
    if seats is not None:
        running = max(1, min(running, int(seats)))
    return mamba_retention_pin_budget(server_args, running, int(limit))


def describe_mamba_floor(server_args: "ServerArgs", max_running_requests: int) -> str:
    """Human-readable derivation, for error messages and boot logs."""
    per_req = mamba_slots_per_running_req(server_args)
    pp = mamba_ping_pong_slots(server_args)
    if server_args.disable_radix_cache:
        terms = f"{MAMBA_FLOOR_ACTIVE_SLOTS} active (radix cache disabled)"
    elif mamba_anchor_ack_release_active(server_args):
        terms = (
            f"{MAMBA_FLOOR_ACTIVE_SLOTS} active"
            f" + {pp} ping-pong"
            f" (#811 ack release: the pinned checkpoint is retention-budget"
            f" funded, released at the write-through ack)"
        )
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
