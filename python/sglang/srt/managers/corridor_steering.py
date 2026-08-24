# SPDX-License-Identifier: Apache-2.0
"""#656 item 16, the REBALANCE tier: ALLOCATION STEERING.

VERDICT FIRST, so nobody has to read the mechanism to learn its status:
this ships OFF (``SGLANG_CORRIDOR_STEERING=1`` to arm it) and its measured
window is recorded in the handoff that shipped it.

WHAT IT IS. Item 16's first relief stage is "redistribute onto the card with
the most headroom". Five shifts read that as a MOVER and could not build one:
under the weighted DCP owner rule a token's row is a pure function of its
global slot id and the boot-constant token vector (``layers/dcp/owner.py``),
and the one actuator that may change that vector needs a fully idle instance
(``kv_reshard.py``). So the ABSORB half of the water fill had no actuator and
the tier stayed empty.

This is the half that was there all along: the allocator CHOOSES which free
slot id to hand out, every id below the pool's capacity is already a legal
placement on its own rank, and the id decides the card. Steering the choice
therefore places new KV bytes on the card with the most headroom WITHOUT
moving a byte, freeing a byte, or changing ``available_size()``.

WHY THAT MATTERS FOR THE EVIDENCE. The predecessor of this tier -- the
continuous cache-dump lender -- was falsified because its action was to free
memory and its metric was free memory (12 breaches, decode halved). The
measurement law that killed it does not reach a mechanism that frees nothing:
the quantity to judge here is PLACEMENT (which card's rows the new
allocations landed on), and the free column is a CONSEQUENCE to be watched,
not the proof of the action.

THE ONE HARD CONSTRAINT: THE DECISION MUST BE GROUP-UNIFORM. The free list is
replicated scheduler state -- every rank hands out the same slot id for the
same token -- so two ranks that ordered their free lists differently would
write one token's KV to two different rows. Every input to the decision is
therefore either a boot constant or a value reduced across the group:

* the ABSORB card comes from NVML, which is rank-local (three ranks read
  three slightly different columns), so the resulting rank index is REDUCED
  with MIN before it is used. A MIN over three sane proposals is still a sane
  proposal, and it is identical everywhere, which the raw reading is not.
* the rank -> NVML column permutation is resolved per rank BY UUID and
  exchanged over the same reduction. This rig is the reason: ``--rank-gpu-id``
  is read in CUDA order and ``CUDA_DEVICE_ORDER`` is FASTEST_FIRST here, so
  rank 0 is the 5090 at nvidia-smi index 1 (register law 9). A steer that
  joined a rank id to a card index by assuming they agree would push bytes
  ONTO the binding card.
* the reduction also carries a CHECKSUM of the free list itself. If the ranks
  ever disagree about it, the replication assumption this mechanism rests on
  is false, and steering DISARMS ITSELF PERMANENTLY rather than continue on a
  premise it just watched fail. That check is also the first time anything in
  this tree has verified that assumption on metal.

WHERE THE TWO HALVES RUN. The decision is taken at the flip SEAM, which is
the one place a collective is already entered by every rank unconditionally
(the median PP dwell on this rig is 17 s, so "per cutover" is a continuous
clock in every sense that matters). The APPLICATION is a stable partition of
the free list on the round clock -- pure, cheap, and re-applied because frees
return pages to the head of the list and wash the order out.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

# #856 F7: the group-agreement idiom has ONE owner. `tree_congruence` holds no
# steering state and imports nothing from here, so this direction cannot cycle.
from sglang.srt.managers.tree_congruence import agreement, digest_pair

logger = logging.getLogger(__name__)

LOG_PREFIX = "CORRIDOR-STEER"

_MIB = 1024 * 1024

#: OFF unless armed. The ship configuration is protected by default.
ENABLE_ENV = "SGLANG_CORRIDOR_STEERING"

#: How level is level enough. Below this the steer stands down: reordering a
#: free list to chase a spread the cards do not have is churn, and a bias that
#: flips class every seam would spread each sequence's tokens the same way an
#: unbiased list does, at the cost of the partition.
MIN_SPREAD_ENV = "SGLANG_CORRIDOR_STEERING_MIN_SPREAD_MIB"
DEFAULT_MIN_SPREAD_MIB = 256

#: Seconds between free-list re-partitions on the round clock.
REAPPLY_ENV = "SGLANG_CORRIDOR_STEERING_REAPPLY_S"
DEFAULT_REAPPLY_S = 1.0

#: Sentinel for "this rank has no proposal", chosen so a MIN reduction ignores
#: it. It must stay larger than any rank index or NVML index.
_NO_PROPOSAL = 1 << 20


def enabled(default: bool = False) -> bool:
    raw = os.environ.get(ENABLE_ENV)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return float(default)


def owner_class_of(ratios: Sequence[int], rank: int) -> Tuple[int, int, int]:
    """``(mod, lo, hi)`` -- the weighted owner rule's slot class for ``rank``.

    The same prefix-sum derivation as ``distributed.utils.cp_token_prefix``,
    written against an explicit vector so the policy can ask about a rank
    that is not this one (which is the entire point: this rank has to steer
    bytes TOWARDS a peer).
    """
    vec = [int(v) for v in ratios]
    if not vec or rank < 0 or rank >= len(vec):
        raise ValueError(f"rank {rank} is outside token vector {vec}")
    lo = sum(vec[:rank])
    return sum(vec), lo, lo + vec[rank]


def absorbing_card(column: Sequence[int], min_spread_bytes: int) -> Optional[int]:
    """The NVML index that should ABSORB new bytes, or None if level enough.

    "Most free" and not "most free relative to its size", deliberately: the
    corridor law is stated in absolute free MiB per card, and item 16 says
    equal FREE HEADROOM rather than equal fill -- the totals differ 32/20/20
    GiB and levelling them by fraction would leave the small cards binding.
    """
    col = [int(f) for f in column]
    if len(col) < 2:
        return None
    if max(col) - min(col) < int(min_spread_bytes):
        return None
    return max(range(len(col)), key=lambda i: col[i])


@dataclass
class SteeringState:
    """Everything a successor needs to read off one log line."""

    armed: bool = False
    disarmed_reason: str = ""
    rank_to_nvml: Tuple[int, ...] = ()
    bias_rank: Optional[int] = None
    decisions: int = 0
    changes: int = 0
    stand_downs: int = 0
    disagreements: int = 0
    applies: int = 0
    promoted_last: int = 0
    #: Per-rank count of decisions that named that rank as the absorber --
    #: the PLACEMENT intent, which is the axis this mechanism is judged on.
    chosen: List[int] = field(default_factory=list)


class AllocationSteering:
    """One per rank. Decides at the seam, applies on the round clock."""

    def __init__(
        self,
        scheduler,
        *,
        ratios: Sequence[int],
        rank: int,
        nvml_index: Optional[int],
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        import time

        self.scheduler = scheduler
        self.ratios = [int(v) for v in ratios]
        self.rank = int(rank)
        self.nvml_index = None if nvml_index is None else int(nvml_index)
        self._clock = clock or time.monotonic
        self._last_apply = 0.0
        self._reapply_s = _env_float(REAPPLY_ENV, DEFAULT_REAPPLY_S)
        self._min_spread_bytes = int(
            _env_float(MIN_SPREAD_ENV, DEFAULT_MIN_SPREAD_MIB) * _MIB
        )
        self.state = SteeringState(armed=True, chosen=[0] * len(self.ratios))

    # -- the decision, at the seam ---------------------------------------

    def decide(self, reduce_fn, column: Sequence[int]) -> Optional[int]:
        """Agree on the absorbing RANK. Returns it, or None to stand down.

        ``reduce_fn`` is the group's MIN channel (``default_collective_min``).
        Every rank must call this, unconditionally, at the same point -- a
        reduction behind a rank-local condition is the desync that wedged
        this instance before (register law 8).
        """
        if not self.state.armed:
            return None
        n = len(self.ratios)
        # Slot layout of the reduced vector:
        #   [0 .. n-1]  the rank -> NVML permutation, each rank filling its own
        #   [n]         this rank's proposal for the absorbing RANK
        #   [n+1]       its negation, so MIN on both tells us whether the
        #               ranks agreed at all (min(x) == -min(-x) iff equal)
        #   [n+2]       free-list checksum, [n+3] its negation
        payload = [_NO_PROPOSAL] * (n + 4)
        if self.nvml_index is not None:
            payload[self.rank] = self.nvml_index
        proposal = self._propose(column)
        # #856 F7: the (x, -x) MIN-pair idiom comes from the ONE authority
        # that owns it (`tree_congruence.digest_pair` / `agreement`), not from
        # a third hand-rolled copy. The arithmetic is identical -- this is a
        # reconciliation, not a behaviour change -- and the point is that a
        # defect in "did the ranks agree?" is now fixable in one place instead
        # of being re-derived at every collective that needs it.
        payload[n], payload[n + 1] = digest_pair(proposal)
        checksum = self._free_list_checksum()
        payload[n + 2], payload[n + 3] = digest_pair(checksum)

        reduced = reduce_fn(payload)
        self.state.decisions += 1

        if not agreement(reduced[n + 2], reduced[n + 3]):
            # THE PREMISE FAILED. The free lists are not identical, so a
            # partition this rank applies is not the partition its peers
            # apply, and steering could split one token's KV across two rows.
            # Disarm for the life of the process and say so once, loudly.
            self.disarm(
                f"the free list is NOT replicated across ranks (checksums "
                f"{reduced[n + 2]} vs {-reduced[n + 3]}); steering cannot be "
                f"group-uniform on a list the ranks disagree about"
            )
            return None

        if not agreement(reduced[n], reduced[n + 1]):
            # The ranks read the column at slightly different instants and
            # named different cards. MIN still yields one answer everywhere,
            # which is all correctness needs; the count is kept because a
            # steer that disagrees constantly is chasing noise.
            self.state.disagreements += 1

        if not self.state.rank_to_nvml:
            perm = tuple(int(v) for v in reduced[:n])
            if any(v >= _NO_PROPOSAL for v in perm) or len(set(perm)) != n:
                self.disarm(
                    f"the rank -> NVML permutation did not resolve ({perm}); "
                    "refusing to steer rather than guess a column (law 9)"
                )
                return None
            self.state.rank_to_nvml = perm
            logger.info(
                "%s rank -> NVML column permutation is %s, resolved by UUID "
                "and agreed over the group; every per-card number below is "
                "joined through it",
                LOG_PREFIX,
                list(perm),
            )

        agreed = int(reduced[n])
        if agreed >= _NO_PROPOSAL or agreed < 0:
            self.state.stand_downs += 1
            return None
        self.state.chosen[agreed] += 1
        return agreed

    def _propose(self, column: Sequence[int]) -> int:
        """This rank's candidate absorbing RANK, or the no-proposal sentinel."""
        card = absorbing_card(column, self._min_spread_bytes)
        if card is None:
            return _NO_PROPOSAL
        perm = self.state.rank_to_nvml
        if not perm:
            # First seam: the permutation is being resolved by this very
            # reduction, so no rank can map a column yet. Propose nothing and
            # steer from the next seam on -- 17 s later.
            return _NO_PROPOSAL
        for r, nvml in enumerate(perm):
            if nvml == card:
                return r
        # The fullest card is not one of ours (another process's GPU). Item
        # 16 is about levelling THIS instance's cards; nothing to do.
        return _NO_PROPOSAL

    def _allocator(self):
        return getattr(self.scheduler, "token_to_kv_pool_allocator", None)

    def _free_list_checksum(self) -> int:
        """A cheap, order-SENSITIVE fingerprint of the free list.

        Order-sensitive on purpose: two ranks holding the same SET of free
        slots in a different order is exactly the failure this must catch, so
        a sum over the set would be blind to it. The head is what the next
        allocations consume, so it is weighted by position and the tail is
        represented by its length and total.
        """
        alloc = self._allocator()
        pages = getattr(alloc, "free_pages", None)
        if pages is None:
            return 0
        try:
            import torch

            head = pages[:256]
            weights = torch.arange(
                1, head.numel() + 1, dtype=torch.int64, device=head.device
            )
            fingerprint = int((head * weights).sum())
            return (int(pages.numel()) * 1000003 + fingerprint) % (1 << 40)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("%s free-list checksum failed: %s", LOG_PREFIX, e)
            return 0

    # -- the application, on the round clock ------------------------------

    def set_bias(self, absorb_rank: Optional[int]) -> None:
        """Install (or clear) the steer. Idempotent and cheap when unchanged."""
        if not self.state.armed:
            return
        alloc = self._allocator()
        if alloc is None or not hasattr(alloc, "set_owner_bias"):
            self.disarm("the active allocator does not support an owner bias")
            return
        if absorb_rank == self.state.bias_rank:
            return
        bias = None if absorb_rank is None else owner_class_of(self.ratios, absorb_rank)
        try:
            promoted = alloc.set_owner_bias(bias)
        except ValueError as e:
            # page_size > 1 or a malformed class: a configuration this
            # mechanism does not apply to, not a runtime fault.
            self.disarm(str(e))
            return
        self.state.bias_rank = absorb_rank
        self.state.changes += 1
        self.state.promoted_last = int(promoted)
        self._last_apply = self._clock()
        logger.info(
            "%s steering NEW KV allocations toward rank %s (owner class %s of "
            "vector %s): %d free slots promoted to the head of the list. "
            "Nothing was moved and nothing was freed -- this places the next "
            "allocations, it does not relocate the current ones.",
            LOG_PREFIX,
            absorb_rank,
            bias,
            self.ratios,
            int(promoted),
        )

    def on_round(self) -> None:
        """Re-apply the standing bias. Rate-limited; never raises."""
        if not self.state.armed or self.state.bias_rank is None:
            return
        now = self._clock()
        if now - self._last_apply < self._reapply_s:
            return
        self._last_apply = now
        alloc = self._allocator()
        if alloc is None:
            return
        try:
            self.state.promoted_last = int(alloc._apply_owner_bias())
            self.state.applies += 1
        except Exception as e:
            self.disarm(f"re-applying the bias raised {e!r}")

    def disarm(self, reason: str) -> None:
        if not self.state.armed:
            return
        self.state.armed = False
        self.state.disarmed_reason = reason
        alloc = self._allocator()
        try:
            if alloc is not None and hasattr(alloc, "set_owner_bias"):
                alloc.set_owner_bias(None)
        except Exception:  # pragma: no cover - best effort on the way out
            pass
        self.state.bias_rank = None
        logger.warning("%s DISARMED: %s", LOG_PREFIX, reason)

    def report(self) -> str:
        s = self.state
        return (
            f"{LOG_PREFIX} armed={s.armed} bias_rank={s.bias_rank} "
            f"decisions={s.decisions} changes={s.changes} applies={s.applies} "
            f"stand_downs={s.stand_downs} disagreements={s.disagreements} "
            f"chosen_per_rank={s.chosen} perm={list(s.rank_to_nvml)}"
            + (f" disarmed_reason={s.disarmed_reason!r}" if s.disarmed_reason else "")
        )


STEERING_ATTR = "corridor_allocation_steering"


def build_allocation_steering(scheduler) -> Optional[AllocationSteering]:
    """The rank's steer, or None when it does not apply to this boot.

    Returns None -- rather than a disarmed object -- for the configurations
    the mechanism has nothing to say about, so a successor reading the log
    can tell "not applicable" from "armed and then stood down".
    """
    if not enabled():
        return None
    existing = getattr(scheduler, STEERING_ATTR, None)
    if existing is not None:
        return existing

    from sglang.srt.distributed.utils import get_cp_token_ratios

    ratios = get_cp_token_ratios()
    if not ratios or len(ratios) < 2:
        logger.info(
            "%s not applicable: no uneven-DCP token vector is installed, so "
            "slot ids do not name a card",
            LOG_PREFIX,
        )
        return None

    alloc = getattr(scheduler, "token_to_kv_pool_allocator", None)
    if alloc is None or not hasattr(alloc, "set_owner_bias"):
        logger.info(
            "%s not applicable: the active allocator has no owner bias "
            "(paged, page_size == 1 is required)",
            LOG_PREFIX,
        )
        return None

    from sglang.srt.managers.corridor_rebalance import resolve_nvml_index

    model_runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
    device_index = getattr(model_runner, "gpu_id", None)
    nvml_index = None if device_index is None else resolve_nvml_index(int(device_index))

    # THE WORLD RANK, NOT ``ps.tp_rank``, and the first boot is why. The
    # scheduler's topology snapshot describes the CURRENT phase, and this
    # instance boots in PP3 -- where ``tp_size == 1`` and every rank's
    # ``tp_rank`` is 0. All three ranks then wrote their NVML column into
    # slot 0 of the reduction, the permutation came back as
    # ``(0, 1048576, 1048576)``, and the steer disarmed itself rather than
    # guess (which is the guard working, but it is not a steer).
    #
    # The token vector is indexed by the same identity the cutover uses to
    # rebuild the topology (``get_world_group().rank_in_group``), so that is
    # the one identity that means the same thing in both phases.
    from sglang.srt.distributed import parallel_state as _ps

    try:
        rank = int(_ps.get_world_group().rank_in_group)
    except Exception as e:
        logger.info("%s not applicable: no world group yet (%s)", LOG_PREFIX, e)
        return None
    if rank >= len(ratios):
        logger.info(
            "%s not applicable: world rank %d is outside the token vector %s",
            LOG_PREFIX,
            rank,
            ratios,
        )
        return None
    steer = AllocationSteering(
        scheduler, ratios=ratios, rank=rank, nvml_index=nvml_index
    )
    setattr(scheduler, STEERING_ATTR, steer)
    logger.info(
        "%s armed on rank %d (NVML column %s), token vector %s. The DECISION "
        "is taken at the seam and reduced with MIN; the free list is "
        "re-partitioned every %.1fs.",
        LOG_PREFIX,
        rank,
        nvml_index,
        ratios,
        steer._reapply_s,
    )
    return steer


def steer_on_round(scheduler) -> None:
    """Round-clock hook. Never raises: a steer is an optimisation."""
    try:
        steer = getattr(scheduler, STEERING_ATTR, None)
        if steer is None:
            return
        steer.on_round()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("%s round hook failed: %s", LOG_PREFIX, e)


def steer_at_seam(scheduler, reduce_fn) -> None:
    """Seam hook: build if needed, decide, install. Never raises.

    Called on the same unconditional path as the seam's other reduction, so
    every rank enters it whether or not it is under pressure.
    """
    try:
        steer = build_allocation_steering(scheduler)
        if steer is None or reduce_fn is None:
            return
        from sglang.srt.managers.phase_flip_spill import get_corridor_guard

        guard = get_corridor_guard(scheduler)
        column = guard.fleet_free() if guard is not None else []
        absorb = steer.decide(reduce_fn, column)
        steer.set_bias(absorb)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("%s seam hook failed: %s", LOG_PREFIX, e)
