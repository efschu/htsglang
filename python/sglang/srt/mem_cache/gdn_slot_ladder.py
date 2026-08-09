# SPDX-License-Identifier: Apache-2.0
"""Resident-slot cap for GDN/Mamba state, with idle-vacate (#364).

The GDN/Mamba state pool is dimensioned at boot for a fixed number of
sessions (``max_mamba_cache_size``, #47 target_concurrency). Every slot is
resident for the whole life of the server, whether or not a session is using
it. This module makes "GDN stays resident" a DEFAULT POLICY instead of a hard
rule:

* ``--gdn-resident-state-slots N`` caps the number of RESIDENT state slots.
  The pool is boot-sized to the cap, so the bytes the surplus slots would
  have held are simply never taken from the hybrid memory budget and land in
  the KV pool through the existing route (``handle_max_mamba_cache`` spends
  from ``total_rest_memory``; what it does not spend, the KV sizing gets).
  There is no second ledger and nothing to reconcile.
* Sessions beyond the cap are not refused. An IDLE session's state leaves
  VRAM as an opaque blob (``MambaPool.export_state_blob``) and its slot goes
  back to the allocator; the session is restored into a free resident slot
  before its next decode tick.

Three constraints are settled and encoded here rather than re-decided at each
call site:

1. ONLY IDLE SESSIONS VACATE. An active session -- one with work in the
   current batch -- never loses its state, whatever the cap says. The
   ordering among idle candidates is NOT a new policy: it is #242's
   ``session_priority_key``, imported, so the spill order of a GDN state and
   the spill order of a KV session cannot drift apart.
2. THE BLOB IS OPAQUE AND NEVER TAKES THE RADIX ROUTE (#212). A recurrent
   state is positional and not prefix-shareable; a MambaRadixCache
   truncation of it is a silent full re-prefill. It travels as a session
   blob through the spill pager's selectable targets (#224).
3. VACATE/RESTORE ONLY BETWEEN TICKS. The pool tensors are addressed by
   captured graphs (#52/#53): mutating a slot while a graph may replay is
   the corruption class those tasks named. :func:`vacate_plan` is a pure
   planner -- it decides, it does not move -- so the caller can run it at the
   same between-tick point ``kv_pressure_runtime`` uses (#287) and execute
   the plan there.

Cap unset = today's behaviour, byte-identical: :func:`effective_state_slots`
is the identity and :func:`vacate_plan` returns an empty plan.

Pure python + integer arithmetic. No torch, no CUDA, no globals.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence

#: Slot 0 is the pool's dummy write target for padded tokens. It is not a
#: session slot and never vacates, so a cap of N means slots 1..N.
DUMMY_SLOT = 0


class VacatePlan(NamedTuple):
    """What the caller must move at ONE between-tick boundary.

    ``vacate`` -- session ids whose state must leave VRAM (idle, beyond the
    cap), youngest/least-protected first.
    ``restore`` -- session ids that need their state back before their next
    decode tick, in arrival order (a resumed session is about to run).
    ``skipped`` -- session id -> why it was NOT vacated, so a server that
    fails to free a slot says which sessions refused and for what reason
    instead of just reporting a shortfall.
    """

    vacate: List[str]
    restore: List[str]
    skipped: Dict[str, str]

    @property
    def is_empty(self) -> bool:
        return not self.vacate and not self.restore


def effective_state_slots(
    profiled_slots: int,
    resident_cap: Optional[int],
) -> int:
    """The number of state slots the pool is built with.

    ``profiled_slots`` is what the existing sizing arrived at
    (``max_mamba_cache_size`` after every #79/#307 clamp);
    ``resident_cap`` is ``--gdn-resident-state-slots``.

    The cap only ever LOWERS: asking for more resident slots than the memory
    budget can hold is not a request the sizing can honour, and quietly
    growing the pool past a profiled ceiling is how a boot turns into an OOM.
    A cap above the profiled count is therefore a no-op, not an error -- the
    same "the flag is a ceiling, not a demand" reading as
    ``--max-total-tokens``.
    """
    slots = int(profiled_slots)
    if resident_cap is None:
        return slots
    cap = int(resident_cap)
    if cap < 1:
        raise ValueError(
            f"--gdn-resident-state-slots must be >= 1 (one session must be "
            f"able to hold state), got {cap}"
        )
    return min(slots, cap)


def freed_state_bytes(
    profiled_slots: int,
    resident_cap: Optional[int],
    bytes_per_slot: int,
) -> int:
    """Bytes the cap keeps out of the state pool at boot.

    This is a REPORTING figure for the boot log and the tests. It is not a
    transfer: nobody moves these bytes to the KV pool, they are simply never
    spent, and the KV sizing that runs after ``handle_max_mamba_cache`` sees
    a larger remainder by exactly this much. Deriving a number here and
    ALSO booking it somewhere would be the parallel ledger this design does
    not have.
    """
    dropped = int(profiled_slots) - effective_state_slots(
        profiled_slots, resident_cap
    )
    return max(0, dropped) * max(0, int(bytes_per_slot))


def session_admission_slots(
    profiled_slots: int,
    resident_cap: Optional[int],
    *,
    vacate_available: bool,
) -> int:
    """State-slot count the CONCURRENCY ceiling is sized from under the cap.

    ``max_running_requests`` is derived as ``slots // mamba_ratio``. Slice 1
    caps the physical pool (``max_mamba_cache_size``) to ``resident_cap``, so
    feeding that capped number here divides the concurrency ceiling by the
    cap and craters it -- cap=4, ratio=5 -> ``4 // 5 = 0`` requests, cap=32 ->
    ``max_running_requests`` 16 drops to 6 (#364 slice 3). That defeats the
    ladder: sessions beyond the cap are meant to live with their state
    VACATED to the spill pager, not to be refused admission.

    The admission ceiling must therefore be sized from the SESSION budget,
    which is the PRE-CAP profiled slot count -- exactly the concurrency the
    un-capped pool would have admitted. Three properties, each a guard:

    * It is NEVER more than ``profiled_slots``. The cap frees state VRAM to
      the KV pool; it is not a licence to admit past what the memory that was
      profiled to fit could back. This is the over-admission interlock: the
      un-capped baseline was affordable, and we never exceed it.
    * It is only raised above the resident cap when ``vacate_available`` --
      the flag that caps the pool has also armed the between-tick vacate
      runtime that backs the overflow (same flag, same boot). Without that
      runtime the honest budget is the resident cap itself: fewer concurrent
      sessions, never an admit-into-OOM.
    * Off the cap (``resident_cap`` None or non-binding) it is the identity
      on ``profiled_slots`` -- today's behaviour, byte-identical.

    The KV side of "what can actually back the sessions" is not re-checked
    here: ``_resolve_max_num_reqs`` already clamps ``max_num_reqs`` by
    ``token_capacity // 2``, and that clamp is unchanged.
    """
    profiled = int(profiled_slots)
    if resident_cap is None or not cap_is_binding(resident_cap, profiled):
        return profiled
    if vacate_available:
        return profiled
    return effective_state_slots(profiled, resident_cap)


def vacate_plan(
    *,
    resident_slots: int,
    sessions: Sequence[object],
    active_ids: Sequence[str],
    resumed_ids: Sequence[str] = (),
    parked_ids: Sequence[str] = (),
    session_id_of=None,
) -> VacatePlan:
    """Plan one between-tick round of the slot ladder.

    ``sessions`` are the live session objects (anything carrying the #242
    protection fields); ``active_ids`` are the ones with work in the batch
    that is about to run or has just run -- they are untouchable. ``parked_ids``
    are the sessions whose state is currently a blob.

    The plan first guarantees CORRECTNESS (every session that will decode
    this tick has a resident slot) and only then takes savings: restores are
    unconditional, vacates fill whatever room the restores need plus the
    overshoot above the cap. That ordering is the same "raising is immediate,
    lowering is hysteretic" rule the #286 session ladder uses at admission
    boundaries; here the raise is per tick because a resumed session cannot
    wait for a plateau.

    Returns empty when the cap is not binding -- the default path.
    """
    from sglang.srt.managers.kv_session_offload import session_priority_key

    if session_id_of is None:

        def session_id_of(s):
            return getattr(s, "session_id", None) or getattr(s, "rid", None)

    cap = int(resident_slots)
    if cap < 1:
        raise ValueError(f"resident_slots must be >= 1, got {cap}")

    parked = set(parked_ids)
    active = set(active_ids)
    by_id = {}
    for s in sessions:
        sid = session_id_of(s)
        if sid is not None:
            by_id[sid] = s

    # Restores: every resumed session needs its state back BEFORE it decodes.
    restore = [sid for sid in resumed_ids if sid in parked]

    # Residents after the restores, minus the ones already parked.
    resident_ids = [
        sid for sid in by_id if sid not in parked or sid in restore
    ]
    overshoot = len(resident_ids) - cap
    skipped: Dict[str, str] = {}
    vacate: List[str] = []
    if overshoot > 0:
        candidates = []
        for sid in resident_ids:
            if sid in active:
                skipped[sid] = "active session (has work in this tick)"
                continue
            if sid in restore:
                skipped[sid] = "resuming this tick"
                continue
            candidates.append(sid)
        # #242 ordering, imported not reinvented: the LEAST protected session
        # (youngest, lowest spill class, non-fast-lane) vacates first.
        candidates.sort(key=lambda sid: session_priority_key(by_id[sid]))
        for sid in candidates:
            if len(vacate) >= overshoot:
                break
            vacate.append(sid)
    return VacatePlan(vacate=vacate, restore=restore, skipped=skipped)


def cap_is_binding(resident_cap: Optional[int], profiled_slots: int) -> bool:
    """True when the cap actually changes the pool geometry. Used to keep
    every downstream hook off the default path entirely."""
    return (
        resident_cap is not None
        and int(resident_cap) < int(profiled_slots)
    )


#: Where the PRE-CAP profiled slot count is remembered. Underscore-prefixed
#: on purpose: ``ServerArgs.__setattr__`` exempts private names from the
#: strict post-resolution mutation guard, and ``copy.deepcopy`` carries it.
PROFILED_SLOTS_ATTR = "_gdn_profiled_state_slots"


def remember_profiled_state_slots(server_args) -> int:
    """The PRE-CAP profiled slot count, recorded ONCE on ``server_args``.

    WHY THIS IS NOT JUST ``server_args.max_mamba_cache_size``, and why the
    memory has to live on the ARGS rather than on the model runner:

    Applying the cap OVERRIDES ``max_mamba_cache_size`` to the capped value.
    That is fine for one stack. A phase-flip instance, however, builds a
    SECOND stack from ``copy.deepcopy(server_args)`` taken AFTER the first
    stack was sized (``phase_flip_boot``), so the copy already carries the
    CAPPED number. A second, naive read there sees ``capped`` where the
    first read ``profiled``, concludes the cap is not binding, records no
    profiled count -- and the concurrency ceiling for that stack is then
    derived from the shrunken pool. The two stacks disagree on
    ``max_num_reqs``, and the flip's boot guard refuses the instance:

        req_to_token shapes diverge between stacks: PP (5, 393224)
        vs TP (4, 393224)

    (measured on metal, ``--gdn-resident-state-slots 10``
    ``--max-mamba-cache-size 20``, ratio 3: PP sized from 20, TP from 10.)

    Write-once is the whole contract: the first caller records the un-capped
    truth, every later caller -- including one holding a deepcopy made after
    the override -- reads that same number back.
    """
    remembered = getattr(server_args, PROFILED_SLOTS_ATTR, None)
    if remembered is None:
        remembered = int(server_args.max_mamba_cache_size)
        setattr(server_args, PROFILED_SLOTS_ATTR, remembered)
    return int(remembered)


def recall_profiled_state_slots(server_args) -> Optional[int]:
    """Read the remembered pre-cap count, or ``None``. NEVER writes.

    The read-only twin of :func:`remember_profiled_state_slots`, for call
    sites on the default path: with the cap unset nothing was ever recorded
    and nothing must be, so the caller falls back to
    ``max_mamba_cache_size`` and the un-capped behaviour stays
    byte-identical.
    """
    remembered = getattr(server_args, PROFILED_SLOTS_ATTR, None)
    return None if remembered is None else int(remembered)
