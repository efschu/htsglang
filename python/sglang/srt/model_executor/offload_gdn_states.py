# SPDX-License-Identifier: Apache-2.0
"""GDN-/Mamba state sets as an offload-register class (#286, DESIGN_201
Nachtrag-13 Ergaenzung 8) -- CPU phase.

The GDN/Mamba state pool is dimensioned for MAX sessions; with fewer running
sessions the surplus state sets are dead weight. This module makes them a
register class (``gdn_state_sets``) with a SESSION-SET LADDER:

* Item granularity is ONE state SET = one session slot across all GDN layers
  (never the whole pool). Slot 0 of the pool is the dummy write target for
  padded tokens and is never registered (it must stay resident).
* The ladder (``SessionSetLadder``) has configurable rungs
  (``--gdn-state-set-ladder``, e.g. ``4,2,1``); the active rung is the number
  of resident sets. Rung changes happen ONLY at admission boundaries
  (``OffloadRegister.on_admission_boundary``), with hysteresis -- the same
  contract as the K-ladder: no per-round flapping.
* LOWERING waits for ``--gdn-state-set-ladder-hysteresis`` consecutive
  admission cycles below the threshold; RAISING is immediate and happens
  BEFORE the admission that needs the next set (correctness before savings).
  The enforced invariant -- an arriving session NEVER meets a parked set --
  lives in ``on_admission_boundary`` and in the ladder's target arithmetic
  (the target never answers below ``running + incoming``, even between
  rungs).
* Default = ladder off = today's behavior: every set stays resident and the
  admission hook plans nothing. All of this is additionally gated behind
  ``SGLANG_OFFLOAD_REGISTER=1`` (the adapters are no-ops otherwise, the
  default path is byte-identical).

Set size model: the per-set bytes are derived from the REAL tensor shapes of
the pool's persistent per-slot state -- the conv states (one per conv shape),
the temporal (SSM) state and, when allocated, the GDN ReplaySSM ring buffers.
All of these are laid out ``[num_layers, num_slots, ...]``, so one set's
share of a tensor is ``nbytes / num_slots``. The speculative intermediates
(``intermediate_ssm``, ``intermediate_conv_window``) are EXCLUDED: they are
transient verify scratch keyed to spec slots, not session state (the same
exclusion as the pool's RDMA/transfer inventory). The resolver is injectable
(``set_bytes_fn``) and evaluated lazily via the register's ``size_source``
mechanism, so hermetic tests feed fake/meta tensors and the GPU phase sees
the true figure at the first ``refresh_sizes()`` without adapter changes.

What moves, and when, is the GPU phase: executing a plan's
``park_candidates`` / ``wave_in_candidates`` through the movement backend
(the sets are ``va_stable_required`` -- kernels and captured graphs address
them, so their route is the #93 VMM/tag path, a named GPU-phase item), and
wiring the scheduler's admission path to call ``on_admission_boundary``.
Nothing in this module touches torch.cuda.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

GDN_STATE_SET_CLASS = "gdn_state_sets"

# Fields of MambaPool.State that are transient speculative scratch (keyed to
# spec slots, not session sets) -- mirrors the exclusion list of the pool's
# get_contiguous_buf_infos / transfer inventory.
_TRANSIENT_SPEC_FIELDS = ("intermediate_ssm", "intermediate_conv_window")

# Lowering waits this many consecutive admission cycles below the threshold
# before the rung drops (raising is always immediate). Deliberately > 1 so a
# single idle admission window does not already unpark/park sets back and
# forth at session churn.
DEFAULT_ADMISSION_HYSTERESIS_CYCLES = 2


def state_set_item_id(slot: int) -> str:
    """Register item id of one session state set. Zero-padded so the
    lexicographic order the planner sorts by IS the numeric slot order."""
    return f"gdn_state_set/{slot:05d}"


def parse_gdn_state_set_ladder(
    spec: Optional[str], flag: str = "--gdn-state-set-ladder"
) -> Optional[Tuple[int, ...]]:
    """Parse the ladder rungs (e.g. ``"4,2,1"``) into a descending tuple.

    ``None``/empty = ladder off (today's behavior, all sets resident).
    Hard errors -- at argument time, so a typo fails the boot, not the first
    admission: non-integer or non-positive rungs, duplicates, and rungs not
    in strictly descending order (the ladder is descended set count by set
    count; an unordered spec is a sign of a mis-typed flag, not a wish).
    """
    if spec is None or not spec.strip():
        return None
    rungs: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(
                f"{flag} contains an empty rung entry in {spec!r}; expected "
                f"a comma-separated list of positive integers, e.g. '4,2,1'."
            )
        try:
            rung = int(part)
        except ValueError:
            raise ValueError(
                f"{flag} rung {part!r} is not an integer (spec {spec!r})."
            ) from None
        if rung < 1:
            raise ValueError(
                f"{flag} rung {rung} is not a positive set count "
                f"(spec {spec!r}); the ladder bottoms out at 1 resident set."
            )
        rungs.append(rung)
    for prev, cur in zip(rungs, rungs[1:]):
        if cur >= prev:
            raise ValueError(
                f"{flag} rungs must be strictly descending, got {spec!r} "
                f"({cur} after {prev})."
            )
    return tuple(rungs)


class SessionSetLadder:
    """The Erg.-8 session-set ladder: which rung (= resident set count) is
    active, moved only at admission boundaries, with lowering hysteresis.

    Contract (analogous to the K-ladder):

    * ``rungs`` are the configured plateaus, strictly descending, each within
      ``[1, max_sets]``. The ladder starts on the top rung (all its sets
      resident).
    * ``on_admission_cycle(needed)`` is called once per admission boundary
      with ``needed = running + incoming`` and returns the TARGET resident
      set count. Raising is immediate: the target is always >= ``needed``
      (capped at ``max_sets``), even between rungs -- correctness never
      waits for a plateau. Lowering is hysteretic: the rung drops to the
      smallest plateau covering ``needed`` only after ``hysteresis_cycles``
      CONSECUTIVE cycles wanting a lower rung; any cycle at or above the
      current rung resets the counter.
    * The ladder never answers below its bottom rung: the deepest configured
      plateau is the floor, an idle server keeps that many sets warm.
    """

    def __init__(
        self,
        rungs: Tuple[int, ...],
        max_sets: int,
        hysteresis_cycles: int = DEFAULT_ADMISSION_HYSTERESIS_CYCLES,
    ):
        if max_sets < 1:
            raise ValueError(f"max_sets must be >= 1, got {max_sets}")
        parsed = parse_gdn_state_set_ladder(
            ",".join(str(r) for r in rungs) if rungs else None
        )
        if parsed is None:
            raise ValueError(
                "SessionSetLadder needs at least one rung; a ladder-off "
                "configuration simply attaches no ladder."
            )
        if parsed[0] > max_sets:
            raise ValueError(
                f"ladder top rung {parsed[0]} exceeds the pool's "
                f"{max_sets} session state sets; rungs must fit the pool."
            )
        if hysteresis_cycles < 1:
            raise ValueError(
                f"hysteresis_cycles must be >= 1 (1 = lower on the first "
                f"below-threshold cycle), got {hysteresis_cycles}"
            )
        self.rungs = parsed
        self.max_sets = int(max_sets)
        self.hysteresis_cycles = int(hysteresis_cycles)
        # Start fully resident on the top rung (today's behavior until the
        # first hysteresis-complete lowering).
        self._current = parsed[0]
        self._below_cycles = 0

    @property
    def current_rung(self) -> int:
        return self._current

    def _plateau(self, needed: int) -> int:
        """Smallest configured rung covering ``needed``; above the top rung
        the answer is ``needed`` itself (capped at max_sets) -- the invariant
        outranks the plateaus. At or below the bottom rung the answer is the
        bottom rung (the floor)."""
        if needed > self.rungs[0]:
            return min(needed, self.max_sets)
        candidate = self.rungs[0]
        for rung in self.rungs:  # descending
            if rung >= needed:
                candidate = rung
            else:
                break
        return candidate

    def on_admission_cycle(self, needed: int) -> int:
        """One admission boundary. Returns the target resident set count."""
        if needed < 0:
            raise ValueError(f"needed must be >= 0, got {needed}")
        plateau = self._plateau(needed)
        if plateau > self._current:
            # Raise immediately, BEFORE the admission that needs the sets.
            self._current = plateau
            self._below_cycles = 0
        elif plateau < self._current:
            self._below_cycles += 1
            if self._below_cycles >= self.hysteresis_cycles:
                self._current = plateau
                self._below_cycles = 0
        else:
            self._below_cycles = 0
        # Between rungs (needed above the current plateau but a raise not yet
        # latched -- cannot happen by construction, but keep the invariant
        # arithmetic in one place): never answer below needed.
        return max(self._current, min(needed, self.max_sets))


def mamba_state_set_nbytes(mamba_cache, num_slots: int) -> int:
    """Per-set bytes of one session slot across all GDN layers, from the
    REAL tensor shapes of the pool's persistent state (conv states, temporal
    state, ReplaySSM rings when allocated). All these tensors are laid out
    ``[num_layers, num_slots, ...]``, so one set's share is
    ``nbytes / num_slots``. Speculative intermediates are excluded (transient
    verify scratch keyed to spec slots -- same exclusion as the transfer
    inventory). Works with fake/meta tensors: only ``numel``/``element_size``
    are consulted, never data or device."""
    import dataclasses

    if num_slots < 1:
        raise ValueError(f"num_slots must be >= 1, got {num_slots}")
    total = 0
    for f in dataclasses.fields(mamba_cache):
        if f.name in _TRANSIENT_SPEC_FIELDS:
            continue
        value = getattr(mamba_cache, f.name, None)
        if value is None:
            continue
        tensors = value if isinstance(value, (list, tuple)) else [value]
        for t in tensors:
            if t is None:
                continue
            numel = getattr(t, "numel", None)
            element_size = getattr(t, "element_size", None)
            if not (callable(numel) and callable(element_size)):
                continue
            total += (int(numel()) * int(element_size())) // num_slots
    return total


def _slot_hot(pool, slot: int) -> bool:
    """Hot criterion of one set: the slot belongs to an ACTIVE session.

    The activity probe is attached by the allocator owner
    (``attach_state_set_activity_probe``); until one is attached every set
    reports HOT -- the safe direction (nothing parks on an unknown
    allocator state), same placeholder pattern as the saturation sensor.
    """
    probe = getattr(pool, "_offload_slot_active_fn", None)
    if probe is None:
        return True
    return bool(probe(slot))


def attach_state_set_activity_probe(
    pool, allocator, enabled: Optional[bool] = None
) -> bool:
    """Wire the pool's set-hot criterion to the slot allocator: a slot is
    ACTIVE (hot, unparkable) iff the allocator does not list it as free.

    Called by the allocator's owner (HybridReqToTokenPool) right after
    construction; a no-op unless the offload register is enabled. The probe
    is only consulted at admission boundaries (and by ``park()``'s hot
    check), never per token.
    """
    if enabled is None:
        from sglang.srt.model_executor.offload_register import (
            offload_register_enabled,
        )

        enabled = offload_register_enabled()
    if not enabled:
        return False

    def probe(slot: int) -> bool:
        free = allocator.free_slots
        if hasattr(free, "numel"):
            return not bool((free == slot).any().item())
        return slot not in free

    pool._offload_slot_active_fn = probe
    return True


def register_mamba_state_sets(
    pool,
    set_bytes_fn: Optional[Callable[[], int]] = None,
) -> List[str]:
    """Book every session state set of a MambaPool as a ``gdn_state_sets``
    register item (Erg. 8) and, when ``--gdn-state-set-ladder`` is
    configured, attach the session ladder to the global register.

    One item per session slot 1..size (slot 0 is the pool's dummy padding
    write target and stays resident by construction). Each item:

    * live size source = the per-set share of the pool's persistent state
      tensors (see ``mamba_state_set_nbytes``); injectable via
      ``set_bytes_fn`` for tests;
    * hot = the slot belongs to an active session (probe attached by the
      allocator owner; unknown = hot, the safe direction);
    * ``va_stable_required`` -- state kernels and captured graphs address
      the pool tensors, so a parked set must come back at the same VA
      (#93 route, GPU phase);
    * turn-class time constant, moved only via ``on_admission_boundary``.

    No-op (empty return) unless ``SGLANG_OFFLOAD_REGISTER=1`` -- the default
    path stays byte-identical.
    """
    from sglang.srt.model_executor.offload_register import (
        maybe_register_item,
        offload_register_enabled,
    )

    if not offload_register_enabled():
        return []

    num_slots = pool.size + 1  # tensors are dimensioned [layers, size+1, ...]
    if set_bytes_fn is None:

        def set_bytes_fn() -> int:
            return mamba_state_set_nbytes(pool.mamba_cache, num_slots)

    item_ids: List[str] = []
    for slot in range(1, pool.size + 1):
        item_id = state_set_item_id(slot)
        maybe_register_item(
            item_id,
            GDN_STATE_SET_CLASS,
            0,
            hot=lambda s=slot: _slot_hot(pool, s),
            va_stable_required=True,
            time_constant_tier="turn",
            size_source=set_bytes_fn,
        )
        item_ids.append(item_id)

    _maybe_attach_ladder_from_server_args(pool.size)
    return item_ids


def _maybe_attach_ladder_from_server_args(max_sets: int) -> None:
    """Build the session ladder from the global server args, when configured.

    Defensive by design: pools are also constructed in unit tests without
    global server args -- absent args (or an absent flag) simply means
    'ladder off', never an error. A CONFIGURED but invalid ladder still
    fails hard (the flag was validated at argument time; re-raised here for
    direct ServerArgs construction)."""
    from sglang.srt.model_executor.offload_register import get_global_register

    reg = get_global_register()
    if reg is None:
        return
    try:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
    except Exception:
        return
    spec = getattr(args, "gdn_state_set_ladder", None)
    rungs = parse_gdn_state_set_ladder(spec)
    if rungs is None:
        return
    hysteresis = int(
        getattr(
            args,
            "gdn_state_set_ladder_hysteresis",
            DEFAULT_ADMISSION_HYSTERESIS_CYCLES,
        )
    )
    reg.set_session_ladder(
        SessionSetLadder(rungs, max_sets=max_sets, hysteresis_cycles=hysteresis)
    )
