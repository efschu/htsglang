# SPDX-License-Identifier: Apache-2.0
"""Phase-flip KV mover for #631 Route A (PP=3 prefill <-> TP=3 decode).

Moves the full-attention paged KV between the PP layout (stage owns whole
layers, pool row = global slot id) and the TP layout (rank owns tokens
under the weighted DCP vector, pool row = compact row), on the #297
envelope, carried over LITERALLY from ``managers/kv_reshard.py``:

* CONSENSUS FIRST, BYTES SECOND: every ``consensus_interval``-th round --
  gated by the replicated round counter, never local state -- every rank
  enters ONE bounded MIN-reduction with
  ``(armed, ready, epoch, direction, config_fp, vector...)``. ``armed``
  and ``ready`` are MIN-semantics (skew is legal and uniformly resolves
  to "wait"); ``epoch``, ``direction`` (once armed), the layer-map/vector
  fingerprint (ALWAYS -- it is boot config, divergence is fatal armed or
  not) and the vector are equality-checked with the same loud
  :class:`KvReshardError` on every rank.
* PACK -> EXCHANGE -> CHECKSUM -> WRITE with the pool untouched through
  pack, exchange and checksum verification; only the write phase is the
  no-return region. Source and destination are DIFFERENT pools here (the
  PP pool and the TP pool coexist), so the #297 aliasing hazard cannot
  arise inside one buffer -- the write order (local first, then incoming,
  disjoint injective targets) is kept anyway.
* Pools are pre-sized at boot for BOTH layouts: no growth, no address
  change, no CUDA-graph recapture. Bounds are checked loudly before any
  byte moves.

Payload layout per (stage s, dcp rank r) pair, identical on both ends by
convention (a checksum trailer keeps it falsifiable at runtime): layer
ordinals ascending, slots ascending within a layer, K bytes then V; one
row list per pair, reused for every layer (token ownership is
layer-independent). The receiver derives the expected byte count from ITS
OWN pool's per-layer row width -- a sender whose row format diverges is a
loud size/checksum error, which is the runtime pin of the "PP and TP rows
are byte-compatible" claim.

Weights-arena refill and GDN state movement are separate steps of the
flip protocol (DESIGN_631 section 3.6); ``pre_cutover_fns`` is their
injection seam so the scheduler wiring can order them inside the same
no-return region.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.phase_flip_plan import (
    PP_TO_TP,
    TP_TO_PP,
    PhaseFlipTransition,
    build_phase_flip_transition,
    default_wave_count,
    layer_waves,
    ordered_layer_waves,
    restore_first_wave_count,
    validate_layer_map,
)
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.managers import phase_flip_seam_census as seam_census
from sglang.srt.model_executor.weights_arena import (
    checksum_is_representable,
    uint8_checksum,
)
from sglang.srt.utils.common import ceil_align
from sglang.srt.managers.kv_reshard import (
    _CHECKSUM_BYTES,
    KvPoolView,
    _checksum,
    _encode,
    _gather_block_rows,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP"


def chunk_blocks_quiescence(chunked_req) -> bool:
    """Does this chunked prefill prevent a rank from being quiescent?

    ONE definition with TWO callers, and they must never disagree:

      * ``ready_fn`` asks it to decide whether this rank may announce
        itself at the flip entry;
      * ``get_next_batch_to_run``'s armed park asks it to decide whether
        the scheduler may build the NEXT chunk while a flip is armed.

    They are the same question -- "is this request at a settled boundary?"
    -- and when they were written as two separate expressions they drifted
    apart within one session. Defect O relaxed the quiescence side to
    "mid-admission only" (between chunks is settled: committed KV, a fully
    accounted extend_range, exactly the state the carry moves) while the
    park side kept the old blanket ``chunked_req is None``. The result was
    an armed tp_to_pp that could commit but was never allowed to stop
    prefilling, so it prefilled the whole pending queue in the slow layout
    and committed with nothing left to do -- production, 2026-08-09
    20:31:38-48Z. Sharing the definition is the fix for the drift; the
    narrowing is the fix for the behaviour.

    True ONLY while the request is mid-admission -- it has been chosen but
    has no pool row yet, so its KV has no home the carry could move.
    """
    return (
        chunked_req is not None and getattr(chunked_req, "req_pool_idx", None) is None
    )


class PhaseFlipJoinTimeout(RuntimeError):
    """#631(c): the consensus round did not assemble within the bound.

    Caught inside ``on_round`` and turned into a loud abandonment; it never
    escapes to the event loop, because the flip is optional and the parked
    requests are not.
    """


PHASE_PP = "pp"
PHASE_TP = "tp"

_DIR_ID = {PP_TO_TP: 1, TP_TO_PP: 2}
_DIR_OF_PHASE = {PHASE_PP: PP_TO_TP, PHASE_TP: TP_TO_PP}

#: How long an ARMED flip may wait for a group-wide quiescent boundary
#: before it gives up -- seconds, wall clock, measured on whichever rank is
#: still unparked.
#:
#: An armed flip withholds new work so the in-flight state drains; that is
#: what makes the flip interposable BETWEEN a request's prefill and its
#: decode instead of only after every stream has finished. The cost is that
#: a rank which never reaches quiescence withholds work forever, and the
#: requests it is holding never resume. This deadline bounds that: when it
#: expires the FLIP is abandoned, loudly, and serving continues. The user's
#: requests are never aborted -- they are the thing being protected.
#:
#: 30 s is chosen against the legitimate worst case: a drain is a handful of
#: iterations plus, at most, the continuation of one already-half-written
#: chunked prefill (exempt from parking, because a chunk that stops mid-way
#: could never satisfy the quiescence predicate at all).
DEFAULT_PARK_DEADLINE_S = 30.0
# #631(c): how long a rank waits INSIDE the flip's consensus reduction for
# the rest of the group. Generous on purpose -- a peer draining a long
# prefill is normal and must not trip it. This is a wedge breaker, not a
# latency control: without it, a rank that enters and finds no peers waits
# for ever, because every other flip deadline is checked BEFORE entry.
DEFAULT_JOIN_DEADLINE_S = 45.0
# #631 option 2(b): how long an armed rank polls for the whole group to
# reach the flip entry before giving up. Generous: a peer finishing a long
# prefill chunk is normal. This bound is PRE-ENTRY and therefore legal --
# abandoning a poll costs nothing, whereas abandoning an ENTERED
# all_reduce aborts every rank (see the withdrawn (c)).
DEFAULT_PRESENCE_DEADLINE_S = 60.0
# #631: the VRAM the flip's staging buffers must leave alone. The flip is
# not a memory-free operation: the exchange packs this rank's outgoing rows
# and allocates a receive buffer per peer, all from the same device memory
# the KV pool is filling. MEASURED 2026-08-09 -- one session holding 0.995
# of the PP pool turned a routine policy flip into
#
#   kv_reshard._exchange -> torch.empty(584 MiB) -> torch.OutOfMemoryError
#   ("600.38 MiB is free") -> Fatal Python error: Aborted, instance down
#
# and the free VRAM at that moment was already under the 1024 MiB corridor
# floor. Defaulting to the same 1024 MiB makes the flip respect the corridor
# rather than spend it: a flip that cannot be staged without dipping below
# the reserve is abandoned, and serving continues in the current layout.
DEFAULT_STAGING_RESERVE_BYTES = 1024 * 1024 * 1024
# #631: how long the armed spin sleeps between flag reads. Small enough
# that assembly is prompt at idle, large enough not to burn a core while
# a peer finishes draining. The spin touches no channel, so this is a
# pacing knob only -- it cannot affect correctness, just latency.
DEFAULT_PRESENCE_POLL_INTERVAL_S = 0.005

#: Env override for the above. Non-positive disables the deadline, which
#: restores the old unbounded wait -- available deliberately for debugging a
#: slow drain, and named so a reader sees that "no deadline" is a choice.
ENV_PARK_DEADLINE = "SGLANG_PHASE_FLIP_PARK_DEADLINE_S"

# #656 REGISTER C20: THE SEAM MUST ENTER WITH HEADROOM, NOT MERELY BE LEGAL.
#
# The corridor's deepest troughs are made INSIDE the cutover. Measured on
# successor 34's own GREEN window (evidence-631/s37/C20_SIZING.txt, 450
# cutovers against a 100 ms sampler):
#
#   * from a LOW entry the cutover draws at most 456 MiB on the binding card
#     -- the draw is self-limiting, because this gate frees to
#     floor + delta + want before the seam stages, so a big draw only ever
#     follows a HIGH entry (up to 1026 MiB there),
#   * the deep entries are INHERITED: the deepest minima come in pairs of
#     cutovers ~2 s apart, the second entering at the first one's trough,
#   * s34 therefore held the 1024 MiB law by +19 MiB and successor 36's
#     identical trough missed it by -23 MiB. That margin was never designed.
#
# 512 MiB covers the measured 456 MiB draw-from-a-low-entry with room over,
# and it is deliberately NOT the 1026 MiB worst case: requiring that at every
# seam would arm on 77% of cutovers on the binding card, which is successor
# 36's falsified continuous cache-dumper wearing a different hat.
DEFAULT_SEAM_ENTRY_MARGIN_MIB = 512
ENV_SEAM_ENTRY_MARGIN = "SGLANG_SEAM_ENTRY_MARGIN_MIB"

# How many CONSECUTIVE flips in one direction may be delayed for the margin
# before the gate stands down to the corridor LAW.
#
# WHY THIS IS BOUNDED AT ALL. An unbounded margin refusal of pp->tp starves
# decode outright: under strict purity decode runs only in TP, so requests
# prefill and then wait forever, and nothing the PP phase holds can fund the
# seam. Measured 2026-08-10 with a raised arming floor: 411 abandons, 0
# requests completed in 6 minutes, /health 503 with every rank alive. So the
# budget is spent and then the LAW governs -- which is exactly s34's shipped
# behaviour, making the worst case of this term the behaviour it replaces.
# Two rounds is enough for the paired trough to return (the pairs sit ~2 s
# apart and the resting level recovers to a 1807 MiB median).
DEFAULT_SEAM_ENTRY_DELAY_BUDGET = 2
ENV_SEAM_ENTRY_DELAY_BUDGET = "SGLANG_SEAM_ENTRY_DELAY_BUDGET"

#: The marker a margin delay carries into ``too_small``. It exists so the
#: group's abandon log can name a DELAY as a delay: every acceptance harness
#: in this corpus greps "FLIP ABANDONED", and a healthy by-design wait
#: counted there is indistinguishable from the 411-abandon decode wedge.
SEAM_MARGIN_DELAY_TAG = "seam entry margin short"

#: Modulus for the wire-frame digest (register C22). A Mersenne prime just
#: under 2**31 so that a product of two residues fits an int64 with room for
#: the sum, and the digest stays POSITIVE -- it is reduced as a ``[x, -x]``
#: MIN pair, and a value that could reach INT64_MIN could not be negated.
_FRAME_DIGEST_MOD = (1 << 31) - 1

# How many CONSECUTIVE group abandons in one direction may be spent before the
# seam stands down for good and says why.
#
# WHY A CAP EXISTS AT ALL (#485, C34). A seam that is short by a fixed,
# configuration-determined amount is short on every retry: the staging ask is
# a property of the LAYER MAP and the live set, and an abandon moves neither.
# Measured 2026-08-12 on the #485 planner cut: rank0 wanted 4881 MiB of
# staging against 4314 MiB spendable and the group re-armed every
# SGLANG_PHASE_POLICY_MIN_DWELL_S (3 s on the ship config) for nine minutes --
# 185 group abandons, 555 log lines. Each retry is not free: it runs the whole
# spill ladder and a torch ``empty_cache``, and the armed window withholds
# admissions, so nothing drained to the detokenizer. Its heartbeat expired,
# the instance kept the "fired up and ready to roll" it had already printed,
# and /health stopped answering inside its timeout. Nothing was deadlocked --
# every stack sat IDLE in a normal wait. An unbounded retry of a refusal that
# cannot change is what turned an unfundable CONFIG into a dead INSTANCE.
#
# The cap converts that into a diagnosis: serving continues in whichever phase
# the instance is already in, the reason is logged once with the numbers, and
# the operator gets a configuration verdict instead of a silent corpse.
DEFAULT_SEAM_ABANDON_CAP = 8
ENV_SEAM_ABANDON_CAP = "SGLANG_SEAM_ABANDON_CAP"

# Upper bound on the arm attempts a single backoff step may skip, so the
# damping cannot grow without limit before the cap is reached.
DEFAULT_SEAM_ABANDON_BACKOFF_MAX = 16
ENV_SEAM_ABANDON_BACKOFF_MAX = "SGLANG_SEAM_ABANDON_BACKOFF_MAX"

#: #662-F4 / A0: may an arm SPILL for the arming floor before refusing for it?
#:
#: On by default, and a no-op whenever the card already holds the floor, which
#: is every arm on a correctly sized instance -- the sizer charges the floor at
#: boot. What this catches is the transient the sizer cannot: a co-tenant, a
#: capture peak, a rank that drifted. Set to 0 to restore the plain refusal.
ENV_PREARM_RELIEF = "SGLANG_PHASE_FLIP_PREARM_RELIEF"
#: How many consecutive relief attempts one direction may spend before the
#: shortfall is called persistent. BOUNDED BY A COUNT, not a clock: the ladder
#: is synchronous, so a deadline would be measured across a call that cannot be
#: interrupted anyway, and an unbounded loop spills the instance flat.
DEFAULT_PREARM_RELIEF_ATTEMPTS = 3
ENV_PREARM_RELIEF_ATTEMPTS = "SGLANG_PHASE_FLIP_PREARM_RELIEF_ATTEMPTS"


def _prearm_relief_enabled() -> bool:
    return os.environ.get(ENV_PREARM_RELIEF, "1") not in ("0", "false", "False")


def _prearm_relief_attempts() -> int:
    """A malformed or non-positive bound falls back to the default rather than
    to zero: a zero bound would refuse every dip without ever asking the
    ladder, which is the behaviour this term exists to replace."""
    raw = os.environ.get(ENV_PREARM_RELIEF_ATTEMPTS)
    if raw is None:
        return DEFAULT_PREARM_RELIEF_ATTEMPTS
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PREARM_RELIEF_ATTEMPTS
    return n if n > 0 else DEFAULT_PREARM_RELIEF_ATTEMPTS


#: The guard a capped seam installs. ``arm`` already refuses on
#: ``blocking_guards``, so appending here is what makes "stay in the current
#: phase" a STATE rather than a hope -- and it surfaces in the status string.
SEAM_ABANDON_CAP_GUARD = "seam unfundable"

# #485: THE VERDICT IS TERMINAL FOR A REASON, NOT FOREVER.
#
# ``_install_seam_cap_guard`` argues the cap is safe because "the staging ask
# is set by the layer map and the live set; an abandon moves neither". The
# layer map is static. THE LIVE SET IS NOT -- it is the resident request set,
# and it drains. The seam-entry gate one level down already treats the same
# shortage as transient: it DELAYS rather than refuses, "because the
# paired-trough measurement says the memory comes back". Denying that one
# level up costs a long-lived instance its other phase for the rest of the
# process because of one crowded minute.
#
# Retirement is therefore allowed, and it is fenced three ways:
#
#   EARNED     the ask must have reversed against affordable WITH margin, at
#              the current live set. Retiring at exactly the entry bar
#              re-abandons on the next arm, and that pair is the livelock
#              with extra steps -- hence a hysteresis strictly above the C20
#              entry margin rather than equal to it.
#   UNANIMOUS  booked through ``_collective_min`` like ``reduced_fit``. A rank
#              that cleared while a peer did not has learnt nothing about the
#              group, which is the runtime's own reason for resetting the
#              streak collectively rather than in the rank-local gate.
#   BOUNDED    a retire path with no limit re-opens the livelock through the
#              back door: install, retire, re-abandon, forever. After this
#              many retirements the guard is installed for good and names the
#              limit it hit.
#
# NOT KEYED TO THE CUTOVER EPOCH, deliberately. The epoch advances on
# cutovers, and no cutover happens while the guard is installed, so an
# epoch-keyed retire clock would be frozen by the very state it exists to
# leave. The retire round carries its own count over the same transport.
#
# Zero disables retirement and restores the shipped behaviour exactly -- a
# VALUE of the same term, not a second code path.
DEFAULT_SEAM_CAP_RETIRE_LIMIT = 2
ENV_SEAM_CAP_RETIRE_LIMIT = "SGLANG_SEAM_CAP_RETIRE_LIMIT"

#: How far ABOVE the entry requirement the ask must have reversed before the
#: verdict may be retired. Independent of ``seam_entry_margin_bytes`` on
#: purpose: disabling the C20 margin must not collapse the retire bar onto
#: the entry bar, which is precisely the flapping pair this fences off.
DEFAULT_SEAM_CAP_RETIRE_HYSTERESIS_MIB = 512
ENV_SEAM_CAP_RETIRE_HYSTERESIS = "SGLANG_SEAM_CAP_RETIRE_HYSTERESIS_MIB"


def seam_cap_retire_limit() -> int:
    """How many times a capped seam may be given back. 0 disables retirement."""
    try:
        return max(
            0,
            int(
                os.environ.get(ENV_SEAM_CAP_RETIRE_LIMIT, DEFAULT_SEAM_CAP_RETIRE_LIMIT)
            ),
        )
    except ValueError:
        # An unreadable knob must not decide a safety bound by accident.
        return DEFAULT_SEAM_CAP_RETIRE_LIMIT


def seam_cap_retire_hysteresis_bytes() -> int:
    """Margin, on top of the entry requirement, that retirement must clear."""
    try:
        mib = int(
            os.environ.get(
                ENV_SEAM_CAP_RETIRE_HYSTERESIS,
                DEFAULT_SEAM_CAP_RETIRE_HYSTERESIS_MIB,
            )
        )
    except ValueError:
        mib = DEFAULT_SEAM_CAP_RETIRE_HYSTERESIS_MIB
    return max(0, mib) * 1024 * 1024


def _seam_transient_floor_bytes(law_floor_bytes: int) -> int:
    """The floor a CUTOVER TRANSIENT is judged against: the band's lower edge.

    The corridor law is a band and its verdict is the continuous minimum
    against the floor, so a dip that lasts one wave walk is lawful down to the
    floor. Judging a transient against the CENTRE reserves the band's whole
    tolerance for nothing and delays flips the law permits.

    Falls back to the value it was given if the band cannot be read: delaying a
    legal flip costs throughput, entering an illegal one costs the law, so the
    fallback goes the delaying way.
    """
    try:
        from sglang.srt.managers.corridor_guard import CORRIDOR_BAND_FRACTION

        mib = 1 << 20
        floor_mib = int(
            round((int(law_floor_bytes) / mib) * (1.0 - CORRIDOR_BAND_FRACTION))
        )
        return max(0, floor_mib) * mib
    except Exception:  # pragma: no cover - the seam must not fail on this
        return int(law_floor_bytes)


def seam_entry_margin_bytes() -> int:
    """The designed headroom a seam must have ON TOP OF its staging ask.

    Zero disables the term and restores the single pre-C20 ask exactly. It is
    a VALUE of the same term rather than a second code path, so the off
    switch cannot drift from the on switch.
    """
    try:
        mib = int(os.environ.get(ENV_SEAM_ENTRY_MARGIN, DEFAULT_SEAM_ENTRY_MARGIN_MIB))
    except ValueError:
        mib = DEFAULT_SEAM_ENTRY_MARGIN_MIB
    return max(0, mib) * 1024 * 1024


def seam_entry_delay_budget() -> int:
    try:
        n = int(
            os.environ.get(ENV_SEAM_ENTRY_DELAY_BUDGET, DEFAULT_SEAM_ENTRY_DELAY_BUDGET)
        )
    except ValueError:
        n = DEFAULT_SEAM_ENTRY_DELAY_BUDGET
    return max(0, n)


def seam_abandon_cap() -> int:
    """Consecutive group abandons a direction may spend before standing down.

    Zero disables the cap and restores the pre-#485 unbounded retry exactly,
    as a VALUE of the same term rather than a second code path.
    """
    try:
        n = int(os.environ.get(ENV_SEAM_ABANDON_CAP, DEFAULT_SEAM_ABANDON_CAP))
    except ValueError:
        n = DEFAULT_SEAM_ABANDON_CAP
    return max(0, n)


def seam_abandon_backoff_max() -> int:
    try:
        n = int(
            os.environ.get(
                ENV_SEAM_ABANDON_BACKOFF_MAX, DEFAULT_SEAM_ABANDON_BACKOFF_MAX
            )
        )
    except ValueError:
        n = DEFAULT_SEAM_ABANDON_BACKOFF_MAX
    return max(0, n)


def seam_backoff_skips(consecutive_abandons: int, backoff_max: int) -> int:
    """Arm requests to decline cheaply after ``k`` consecutive group abandons.

    ``0, 1, 3, 7, 15, ...`` clamped at ``backoff_max``. The FIRST abandon
    costs nothing: a seam that was short because a request happened to be
    resident deserves an immediate second look, and that is the case the
    existing C20 delay budget is built around. Growth starts only once the
    refusal has repeated, which is the signature of a demand the retry cannot
    change.

    A pure function of a group-uniform input, so every rank computes the same
    number without a collective.
    """
    k = int(consecutive_abandons)
    if k <= 0:
        return 0
    if k >= 32:  # 2**31 is already far past any sane clamp; do not overflow it
        return max(0, int(backoff_max))
    return min((1 << (k - 1)) - 1, max(0, int(backoff_max)))


def _seam_staging_reserve_bytes(server_args) -> int:
    """The user reserve, minus the band tolerance the seam may transiently use.

    Returns BYTES. Falls back to the reserve itself if the band cannot be
    read, which is the previous behaviour and the safe direction: a seam that
    reserves too much refuses a flip, while one that reserves too little
    breaches the law it is supposed to respect.
    """
    reserve_mib = int(getattr(server_args, "rank_user_reserve_mib", None) or 1024)
    try:
        from sglang.srt.managers.corridor_guard import CORRIDOR_BAND_FRACTION

        floor_mib = int(round(reserve_mib * (1.0 - CORRIDOR_BAND_FRACTION)))
    except Exception:  # pragma: no cover - the band must never break a boot
        floor_mib = reserve_mib
    return max(0, floor_mib) * 1024 * 1024


def park_deadline_s() -> float:
    try:
        return float(os.environ.get(ENV_PARK_DEADLINE, DEFAULT_PARK_DEADLINE_S))
    except ValueError:
        return DEFAULT_PARK_DEADLINE_S


_PHASE_AFTER = {PP_TO_TP: PHASE_TP, TP_TO_PP: PHASE_PP}


def _config_fingerprint(
    layer_map: Tuple[Tuple[int, ...], ...], vector: Tuple[int, ...]
) -> int:
    """31-bit stable fingerprint of the replicated flip configuration.

    Folded into every consensus payload and equality-checked ALWAYS: a
    rank booted with a different layer map or vector must die loudly at
    the first consensus round, not at the first wrong byte."""
    acc = 0
    for s, layers in enumerate(layer_map):
        for f in layers:
            acc = (acc * 1_000_003 + (s + 1) * 8191 + f * 131) % (2**31 - 1)
    for v in vector:
        acc = (acc * 1_000_003 + v * 65_537) % (2**31 - 1)
    return acc


class AbortDeferralWindow:
    """Pin 4 (DESIGN_631 3.6a): client disconnects during a flip.

    A parked request whose client vanishes mid-flip must not mutate the
    live slot set between the plan derivation and the write phase -- an
    abort applied on one rank before its peers diverges the replicated
    live set, which the runtime can only answer with a LOUD size/desync
    error (clean abort of the attempt, but a lost flip). Deferral makes
    the window airtight instead: while a flip is pending or executing,
    abort work is QUEUED; it drains in the first round after cutover (or
    after disarm). The queue preserves order. Slots are never leaked --
    the deferred abort frees them under the NEW layout, which is
    equivalent by the global-slot-id property (metadata never rewrites).
    """

    def __init__(self):
        self._deferred: List[Callable[[], None]] = []
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def deferred_count(self) -> int:
        return len(self._deferred)

    def activate(self) -> None:
        self._active = True

    def submit(self, work: Callable[[], None]) -> bool:
        """Run ``work`` now (returns False) or defer it (returns True)."""
        if self._active:
            self._deferred.append(work)
            return True
        work()
        return False

    def deactivate_and_drain(self) -> int:
        """Close the window and run everything deferred, in order."""
        self._active = False
        drained = 0
        while self._deferred:
            work = self._deferred.pop(0)
            work()
            drained += 1
        return drained


def _flip_spec_algo(scheduler):
    """The algorithm the TP DECODE phase will run, or a NONE sentinel.

    ``scheduler.spec_algorithm`` is NONE in the PP phase by design -- the
    configured algorithm is parked in ``flip_spec_algorithm`` at boot and
    swapped in at the cutover -- so the PP-phase question "will the phase
    I am about to enter speculate?" has to read the parked one.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = getattr(scheduler, "flip_spec_algorithm", None)
    if algo is None:
        return SpeculativeAlgorithm.from_string(None)
    return algo


def _flip_can_bootstrap_draft(scheduler) -> bool:
    """Can the cutover give a carried request draft state at all?

    The bootstrap's one structural requirement is a draft KV pool to scrub
    -- the seed and the trivial-verify round need nothing else. The draft
    worker itself lives on the flip's TP stack from boot
    (``PhaseFlipStacks.draft_worker``), not on ``scheduler.draft_worker``,
    which is None throughout the PP phase by design; asking the live
    scheduler would answer the wrong question and hold every flip forever.
    """
    from sglang.srt.managers.phase_flip_draft_bootstrap import draft_kv_pool

    stacks = getattr(scheduler, "phase_flip_stacks", None)
    draft = getattr(stacks, "draft_worker", None) if stacks is not None else None
    return draft_kv_pool(draft) is not None


def _harvest(scheduler):
    from sglang.srt.managers.phase_flip_resident_carry import (
        harvest_resident_batches,
    )

    try:
        return harvest_resident_batches(scheduler)
    except Exception:  # noqa: BLE001 - a readiness probe never breaks a flip
        return []


def build_flip_quiescence_fn(scheduler) -> Callable[[], bool]:
    """The flip ready predicate (DESIGN_631 3.5) -- NOT #297 fully-idle.

    True when no forward is in flight and no chunk is half-written, with
    requests PARKED: no partial chunk, previous batch drained, overlap
    result queue empty, PP micro-batches drained. Deliberately does NOT
    require an empty waiting queue or an empty running batch -- the flip
    exists to run between a request's prefill and its decode."""

    def _why_not() -> Optional[str]:
        """The reason this rank is not quiescent, or None if it is.

        SEPARATED OUT so a rank can SAY why it is holding. Defect I was
        diagnosed from three py-spy stacks because "ready=0" is all the
        log ever carried, and the interesting question -- WHICH rank is
        holding WHAT, and is it the same thing every epoch -- was
        unanswerable from the log alone. It costs one string on a path
        that only runs while a flip is armed.
        """
        # #631 DEFECT O, and it is the SAME CATEGORY ERROR for the third
        # time in this function: a term that refuses because WORK EXISTS
        # rather than because work is IN FLIGHT.
        #
        # This used to be "chunked_req is not None -> not quiescent". A
        # long chunked prefill occupies that attribute for its ENTIRE
        # duration, so a flip armed BECAUSE of pending prefill could only
        # commit once the prefill it was meant to accelerate had already
        # finished. Measured 2026-08-09 04:23:35-04:23:54Z: armed
        # "pending prefill 26624 tok > N", NOT QUIESCENT "a chunked
        # prefill is half-written" on all three ranks for 19 s, cutover to
        # pp at 04:23:54 -- at which instant the policy immediately armed
        # pp_to_tp again because prefill was "down to 0 tok". The whole
        # 32768-token prefill ran in the TP layout at 1525 tok/s against
        # 4553 tok/s measured in PP, and the instance paid two cutovers
        # for nothing.
        #
        # BETWEEN CHUNKS IS A SETTLED BOUNDARY. get_next_batch_to_run
        # caches ("stashes") the computed prefix every round, so a chunked
        # request that is not mid-forward holds committed KV and a fully
        # accounted extend_range -- exactly the state the carry can move.
        # What must be quiet is the FORWARD, which ``mbs`` and
        # ``result_queue`` below already answer.
        #
        # This is only sound because _live_reqs now enumerates
        # scheduler.chunked_req: the request is in NO batch, so without
        # that the flip would move the layout out from under a request
        # whose KV stayed behind. The two changes are one change.
        chunked = getattr(scheduler, "chunked_req", None)
        if chunk_blocks_quiescence(chunked):
            return "a chunked prefill has no pool row yet (mid-admission)"
        # #631 DEFECT L, and it is the SAME CATEGORY ERROR as the
        # _pp_microbatches_drained one two paragraphs down -- found the
        # same way, by a leg that could never commit.
        #
        # This used to read "last_batch is not empty". Under
        # event_loop_normal (the TP decode phase) the result is processed
        # in the SAME iteration as the forward and ``last_batch = batch``
        # is set afterwards, so at the hook a non-empty last_batch means
        # "requests are resident", NOT "work is in flight". A decoding
        # request makes it non-empty on every iteration for ever, so
        # tp_to_pp could never reach a quiescent boundary: armed at
        # 03:11:22Z and 03:12:52Z on all three ranks, "NOT QUIESCENT:
        # last_batch is not empty (1 req(s) visible)", abandoned at the
        # park deadline both times, while pp_to_tp had just carried the
        # same request across the other way without trouble.
        #
        # The genuine evidence of pending work is already checked: the
        # overlap loop's result_queue below, and the PP loop's in-flight
        # ``mbs`` further down. What remains -- and it is narrower -- is
        # whether every live request is reachable through the handle the
        # CARRY harvests. Right after a prefill the new requests are still
        # only in last_batch and are merged into the running batch by the
        # next get_next_batch_to_run: a real reason to wait, self-clearing
        # in one iteration.
        from sglang.srt.managers.phase_flip_resident_carry import (
            orphan_resident_reqs,
        )

        orphans = orphan_resident_reqs(scheduler)
        if orphans:
            return (
                f"{len(orphans)} request(s) are still only in "
                f"last_batch/last_mbs ({orphans[:4]}) and not yet merged "
                f"into the resident set the carry harvests"
            )
        result_queue = getattr(scheduler, "result_queue", None)
        if result_queue is not None and len(result_queue) > 0:
            return f"result_queue holds {len(result_queue)} result(s)"
        # IN-FLIGHT MICROBATCHES ONLY -- deliberately NOT
        # Scheduler._pp_microbatches_drained, which this used to call.
        #
        # That helper is the FULLY-IDLE predicate (is_fully_idle, on_idle)
        # and it also requires every ``running_mbs`` slot to be empty.
        # ``running_mbs`` is the RESIDENT DECODE SET, not work in flight:
        # it holds the requests currently being decoded and empties only
        # when they FINISH. Borrowing it made this function contradict its
        # own contract two lines up ("does NOT require an empty running
        # batch") and, worse, contradict the policy that drives it: the
        # policy arms pp_to_tp precisely BECAUSE requests are decoding, so
        # the arming condition and the quiescence condition could never
        # hold at the same time and every automatic flip abandoned at the
        # park deadline.
        #
        # MEASURED, 2026-08-09 01:29:50Z, POLICY=auto with one request
        # decoding: "NOT QUIESCENT: PP microbatches not drained (live mb
        # slots [], running_mbs slots [0])" on ranks 0 and 1 -- nothing in
        # flight, the resident decode set alone holding the flip. The gate
        # assembled, all three ranks entered the reduction and agreed to
        # abandon on the park deadline, with ready=0 everywhere.
        #
        # Carrying a resident decode set across the flip is what the rest
        # of the design already assumes: build_flip_live_slots_fn exists
        # to move exactly those requests' KV rows ("the flip runs with
        # requests parked, whose KV rows live in req_to_token"). What must
        # be quiet is the PIPELINE -- no forward in flight, no half-written
        # chunk -- which is what ``mbs`` answers.
        mbs = getattr(scheduler, "mbs", None)
        if mbs is not None:
            live = [
                i for i, mb in enumerate(mbs) if mb is not None and not mb.is_empty()
            ]
            if live:
                return f"PP microbatches still in flight (mb slots {live})"
        # #631: SPECULATION AND THE CARRIED REQUEST.
        #
        # A request that prefills in the PP phase has NO DRAFT STATE: the
        # PP phase carries no draft worker by design, so nothing ever ran
        # the draft_extend that a spec instance gives a request after its
        # target extend. Carrying such a request into a SPECULATING TP
        # phase used to kill the instance one pass later -- measured
        # 03:32:14Z on all three ranks:
        #
        #   eagle_worker_v2.draft -> eagle_draft_cuda_graph_runner.execute
        #   -> foreach_copy: output with shape [1, 1] doesn't match the
        #      broadcast shape [0, 1]        -> SIGQUIT
        #
        # THAT IS NOW BUILT, and this predicate no longer holds the flip
        # for it: managers/phase_flip_draft_bootstrap.py scrubs the stale
        # draft KV of the carried slots and installs a seed at the cutover,
        # and the first post-flip round runs a 1-node verify whose hidden
        # states seed the real draft chain.
        #
        # WHAT REMAINS IS THE STRUCTURAL CASE, and it stays a wait rather
        # than becoming an assumption: an instance whose armed draft worker
        # exposes no KV pool cannot be bootstrapped, so a resident request
        # would still meet the draft graph runner with nothing behind it.
        # Waiting, not refusing at arm time -- a rank-local refusal inside
        # arm() would let one rank decline while its peers armed, and
        # diverging epochs is corpse H, fatal. Readiness runs through the
        # bounded park/abandon machinery, which is unanimous by
        # construction.
        runtime = getattr(scheduler, "phase_flip_runtime", None)
        pending = getattr(runtime, "pending", None) if runtime is not None else None
        if pending == PP_TO_TP and not _flip_spec_algo(scheduler).is_none():
            n_resident = sum(
                len(getattr(b, "reqs", []) or []) for b in _harvest(scheduler)
            )
            if n_resident and not _flip_can_bootstrap_draft(scheduler):
                return (
                    f"{n_resident} resident request(s) would enter a "
                    f"SPECULATING TP phase with no draft state, and the "
                    f"armed draft worker exposes no KV pool to bootstrap "
                    f"them into; waiting for them to finish rather than "
                    f"crashing the draft graph runner"
                )
        return None

    def _ready() -> bool:
        return _why_not() is None

    _ready.why_not = _why_not
    return _ready


def build_flip_live_slots_fn(scheduler) -> Callable[[], torch.Tensor]:
    """Live slots = radix tree values UNION parked requests' rows.

    #297 Stage A enumerates the tree only, correct at fully-idle. The
    flip runs with requests parked, whose KV rows live in req_to_token
    and are NOT all in the tree yet -- omitting them would silently drop
    the freshest prefix KV at the flip (DESIGN_631 3.5). Replicated: the
    tree and the batch state are rank-replicated between rounds."""

    def _live() -> torch.Tensor:
        parts: List[torch.Tensor] = []
        values = scheduler.tree_cache.all_values_flatten()
        if values is not None and values.numel():
            parts.append(values.detach().to("cpu", torch.int64))
        # ALL RESIDENT SLOTS, not scheduler.running_batch (#631 J). That
        # attribute is the CURRENT microbatch slot under event_loop_pp, and
        # the flip's hook fires at the end of an arbitrary slot iteration,
        # so reading it sampled an empty slot while a request sat resident
        # in another one -- and that request's rows were then never moved.
        # See _live_reqs for the measurement.
        #
        # The ROW EXTENT below is still req.seqlen, deliberately: the
        # allocator-owned extent is kv_allocated_len and the two differ
        # under #486's spec reserve, but that delta has not yet been
        # measured on a flip where a resident request was actually
        # enumerated (it could not be -- none ever was). _probe_allocated_
        # extent reports it every flip; change this only on that evidence.
        reqs = _live_reqs(scheduler)
        req_to_token = scheduler.req_to_token_pool.req_to_token
        # A REQUEST WITHOUT A POOL SLOT OWNS NO ROWS, and indexing as if it
        # did is not a no-op -- it silently changes the SHAPE. `req_pool_idx`
        # is Optional and starts as None (schedule_batch.py: "self.
        # req_pool_idx: Optional[int] = None"); a Req is visible in
        # last_batch / running_mbs / chunked_req from the moment it is
        # admitted, which is BEFORE its slot is allocated. Then
        #
        #     req_to_token[None, :n]
        #
        # is not "row None" -- None is numpy-style newaxis, so a (R, C)
        # table returns a (1, n, C) tensor, and the concatenation below
        # died on it:
        #
        #     RuntimeError: Tensors must have same number of dimensions:
        #     got 1 and 3        (metal, 2026-08-09, all three ranks)
        #
        # Skipping is CORRECT and not merely convenient: req_to_token is
        # indexed BY req_pool_idx, so a request without one cannot have a
        # row there and therefore holds no KV that a flip could leave
        # behind. Its slot is allocated later, in whatever layout is then
        # current. This is the one case where omitting rows does not risk
        # the silent-wrong-context class the docstring names -- but it is
        # counted and logged rather than assumed, because "this cannot
        # happen" is what made it a crash instead of a log line.
        skipped_no_slot: List[str] = []
        for req in reqs:
            n = int(req.seqlen)
            if n <= 0:
                continue
            if getattr(req, "req_pool_idx", None) is None:
                skipped_no_slot.append(str(getattr(req, "rid", "?")))
                continue
            rows = req_to_token[req.req_pool_idx, :n]
            parts.append(rows.detach().to("cpu", torch.int64))
        if skipped_no_slot:
            logger.info(
                "%s live-slot enumeration skipped %d admitted request(s) "
                "with no req_pool_idx yet (%s): they hold no rows in "
                "req_to_token, so there is nothing for the flip to move; "
                "their slots are allocated in the post-flip layout.",
                LOG_PREFIX,
                len(skipped_no_slot),
                ", ".join(skipped_no_slot[:8]),
            )
        _probe_allocated_extent(scheduler, reqs)
        # #657: SPLIT THE CEILING BY WHO PINS IT. The KV rung's floor is
        # max(live id) + reserve, so the highest live id decides how much
        # backing stays committed on every card -- and this union has two
        # sources with completely different prices. A row held by a RESIDENT
        # request cannot be given up at all. A row held only by the radix
        # tree is EVICTABLE by the cache's own policy, and in the agent
        # traffic this instance serves the cache returned `#cached-token: 0`
        # on 41952 batches while pinning a five-figure row id.
        #
        # Recorded as a side channel rather than a second enumeration: the
        # parts are already materialised here, and the enumeration itself is
        # the expensive half. Whoever prices the rung reads it; nothing acts
        # on it yet, which is exactly why it is worth measuring first.
        try:
            n_tree = 1 if (values is not None and values.numel()) else 0
            tree_parts = parts[:n_tree]
            req_parts = parts[n_tree:]
            split = {
                "tree_max": (
                    int(max(int(p.max()) for p in tree_parts)) if tree_parts else -1
                ),
                "tree_rows": int(sum(int(p.numel()) for p in tree_parts)),
                "req_max": (
                    int(max(int(p.max()) for p in req_parts)) if req_parts else -1
                ),
                "req_rows": int(sum(int(p.numel()) for p in req_parts)),
            }
            # On the FUNCTION, so the rung that calls it can read the split
            # without a second enumeration and without either module having
            # to know the other's plumbing.
            _live.last_split = split
            scheduler.flip_live_split = split
            # #744 kept a sticky "last enumeration that saw requests" here
            # for the KV rung to consult while a flip was armed. #746
            # replaced that channel: the rung now reads the exact extent the
            # controller snapshots at ARM (``PhaseFlipRuntime.parked_
            # extent``), which cannot be stale and exists even when no
            # enumeration ran before the flip armed. The split above remains
            # the snapshot's measurement source.
        except Exception as e:  # pragma: no cover - an instrument, never a gate
            logger.warning("%s live-split instrument failed: %s", LOG_PREFIX, e)
            _live.last_split = None
            scheduler.flip_live_split = None
        if not parts:
            return torch.empty(0, dtype=torch.int64)
        return torch.unique(torch.cat(parts))

    return _live


def _live_reqs(scheduler) -> List:
    """Every request RESIDENT on this rank, across ALL microbatch slots.

    SLOT SCOPE IS THE DEFECT THIS EXISTS FOR (#631 J, measured 2026-08-09
    02:21:03Z). Under ``event_loop_pp``, ``scheduler.running_batch`` and
    ``scheduler.last_batch`` are rebound to ``running_mbs[mb_id]`` and
    ``last_mbs[mb_id]`` at the TOP of every slot iteration. They therefore
    describe ONE microbatch slot -- whichever slot's iteration happens to
    be running -- and NOT the rank's resident set. The flip's round hook
    fires at the END of a slot iteration, so reading ``running_batch``
    there samples an arbitrary slot, and an empty one is indistinguishable
    from "no requests resident".

    Measured at a real cutover:

        at-arm       cur_slot_reqs=1 resident_reqs=1 resident_slots=[1]
        pre-cutover  cur_slot_reqs=0 resident_reqs=1 resident_slots=[1]

    The request was resident in slot 1 throughout; the hook simply ran for
    a different, empty slot. Enumerating from ``running_batch`` alone
    therefore missed its rows entirely.

    THIS IS NOT MERELY AN ACCOUNTING BUG. Rows that are not enumerated are
    not MOVED, so the resident request's freshest KV is left behind in the
    source pool and never written into the destination layout. The leak
    detector notices the arithmetic; the request's CONTEXT would simply be
    wrong, silently. That is the failure class this feature must never
    ship.

    ``running_mbs`` is the per-slot resident set and is the authority
    here; ``running_batch``/``last_batch`` are unioned in for the non-PP
    event loop, where ``running_mbs`` does not exist. Deduplicated by
    identity because the same Req object appears in several of these.
    """
    seen = set()
    out: List = []

    def _take(batch) -> None:
        for req in list(getattr(batch, "reqs", []) or []) if batch else []:
            if id(req) in seen:
                continue
            seen.add(id(req))
            out.append(req)

    for mb in getattr(scheduler, "running_mbs", []) or []:
        _take(mb)
    for name in ("running_batch", "last_batch"):
        _take(getattr(scheduler, name, None))
    # THE CHUNKED PREFILL IS RESIDENT AND IS IN NO BATCH (#631 defect O).
    # get_next_batch_to_run deliberately moves it out ("Move the chunked
    # request out of the batch so that we can merge only finished requests
    # to running_batch"), so every batch-based enumeration misses it --
    # while it holds committed KV for everything computed so far, and a
    # mamba slot. Enumerating it here is what lets the flip happen BETWEEN
    # chunks instead of waiting for the whole prefill to finish; without
    # it, relaxing the quiescence term would move the layout out from
    # under a request whose KV stayed behind, which is J.3 again.
    chunked = getattr(scheduler, "chunked_req", None)
    if chunked is not None and id(chunked) not in seen:
        seen.add(id(chunked))
        out.append(chunked)
    return out


def _probe_allocated_extent(scheduler, reqs) -> None:
    """#631 defect J: MEASURE the gap between what the allocator owns and
    what this enumeration covers. Does not change what is moved.

    VERIFY BEFORE FIXING. A wrong guess about which KV rows belong to a
    request does not fail loudly -- it silently moves the wrong bytes, or
    silently leaves the right ones behind, and the request's context is
    then quietly corrupt. So the delta is measured on a real flip before
    the enumeration is changed to close it.

    ``req.kv_allocated_len`` is the AUTHORITATIVE extent: the number of KV
    slots the allocator has handed this request, and precisely what the
    invariant checker charges to the pool (invariant_checker._check reads
    the same field, page-aligned). ``req.seqlen`` is
    ``len(origin_input_ids) + len(output_ids)`` -- a property of the
    SEQUENCE, not of the allocation, and under speculative decoding the
    two are structurally different: task #486 reserves W + L slots ahead
    of ``kv_committed_len`` every decode step, where W is the draft/verify
    write footprint (topk * num_steps, or num_draft_tokens) and L is the
    commit lag. On this rig's NEXTN config that reserve is several slots,
    NOT one -- so any fix built on "seqlen + 1" would be wrong in general
    even where it happens to balance the books on a quiet flip.
    """
    if not reqs:
        return
    try:
        page_size = int(getattr(scheduler.token_to_kv_pool_allocator, "page_size", 1))
        rows = []
        for req in reqs:
            alloc = int(getattr(req, "kv_allocated_len", -1))
            aligned = (
                ceil_align(alloc, page_size) if (page_size > 1 and alloc > 0) else alloc
            )
            rows.append(
                f"rid={getattr(req, 'rid', '?')} seqlen={int(req.seqlen)} "
                f"kv_allocated_len={alloc} aligned={aligned} "
                f"kv_committed_len={int(getattr(req, 'kv_committed_len', -1))} "
                f"cache_protected_len={int(getattr(req, 'cache_protected_len', -1))} "
                f"delta_vs_seqlen={aligned - int(req.seqlen)}"
            )
        logger.warning(
            "%s FLIP EXTENT PROBE (page_size=%d): %s. delta_vs_seqlen is the "
            "number of allocator-owned rows this enumeration does NOT move; "
            "nonzero means they are left owned by nobody in the destination "
            "stack, which is defect J.",
            LOG_PREFIX,
            page_size,
            " | ".join(rows),
        )
    except Exception as exc:  # noqa: BLE001 - a probe must never break a flip
        logger.warning("%s flip extent probe failed: %s", LOG_PREFIX, exc)


#: #721: how many consecutive host-RAM defers before the guard stands aside.
#: A permanent hold is WORSE than the hazard it avoids -- the flip is how this
#: instance alternates prefill and decode, so refusing forever converts a
#: POSSIBLE process kill into a CERTAIN half-service outage, and the kill is
#: recoverable by a restore while the outage is not self-clearing. So the guard
#: defers a bounded number of times, then escalates loudly and proceeds.
FLIP_HOST_RAM_MAX_DEFERS: int = 3

#: Margin above the transient's own projected demand, in bytes.
#:
#: JUSTIFIED FROM THIS BOX, not chosen round. Steady state measured 2026-08-17
#: 04:0x: 118G total, 38G available, with the flip's host weight images already
#: pinned at 20.5G. 4G is ~10% of that steady headroom -- enough to absorb the
#: lane RSS jitter that shares this container (pytest/git spikes run under 2G)
#: without deferring on ordinary noise. It is deliberately NOT the 10G
#: PINNED_HOST_RESERVE_BYTES: that reserve protects PERMANENT pins, and a
#: staging transient is by definition returned.
FLIP_HOST_RAM_FLOOR_BYTES: int = 4 * (1024**3)

#: The defer reason, spelled once. #696 accounting must see a host-RAM defer as
#: THIS, not absorb it into the generic unfunded bucket -- a flip that did not
#: arm because the HOST was tight is a different fact from one that could not
#: fund its VRAM seam, and merging them would hide exactly the signal #721 is
#: trying to collect.
DEFERRED_HOST_RAM = "DEFERRED-HOST-RAM"


def flip_host_headroom_verdict(
    available_bytes,
    projected_transient_bytes: int,
    defers_so_far: int,
    floor_bytes: int = FLIP_HOST_RAM_FLOOR_BYTES,
    max_defers: int = FLIP_HOST_RAM_MAX_DEFERS,
):
    """``(allow, escalated, detail)`` for one flip's host-RAM headroom (#721).

    THIS GUARD IS ALSO THE DISCRIMINATOR, which is why it ships default-on
    before its hazard is fully attributed. Two host-OOM-shaped kills on this box
    landed 7 s and 11 s after a completed flip; one is a ledger-confirmed kernel
    OOM kill. The attribution needs host journal access this container does not
    have. But every firing of this guard produces the terms we lack: if it fires
    and no kill follows, the flip-transient candidate gains; if a kill happens
    anyway while this reported ample headroom, that candidate dies and lane
    spikes gain. Either way #721 moves without the journal.

    ``available_bytes`` None means no honest number was available -- degrade to
    NO GUARD rather than guess, exactly as ``pinned_host_memory_bytes`` requires
    of its callers. Refusing a flip on a fabricated figure is worse than not
    checking, because the refusal is the thing with a service cost.
    """
    need = max(0, int(projected_transient_bytes)) + max(0, int(floor_bytes))
    if available_bytes is None:
        return True, False, "host RAM unreadable -- guard stood down (no honest number)"
    avail = int(available_bytes)
    if avail >= need:
        return True, False, (
            f"host headroom OK: {avail / 1e9:.2f} GB available >= "
            f"{need / 1e9:.2f} GB needed "
            f"(transient {int(projected_transient_bytes) / 1e9:.2f} + floor "
            f"{int(floor_bytes) / 1e9:.2f})"
        )
    if int(defers_so_far) >= int(max_defers):
        return True, True, (
            f"{DEFERRED_HOST_RAM} ESCALATED after {defers_so_far} defers: "
            f"{avail / 1e9:.2f} GB available < {need / 1e9:.2f} GB needed "
            f"(transient {int(projected_transient_bytes) / 1e9:.2f} + floor "
            f"{int(floor_bytes) / 1e9:.2f}). PROCEEDING WITH EYES OPEN -- a "
            f"permanent hold would stop the instance alternating prefill and "
            f"decode, which is a certain half-service outage, while the kill "
            f"this defends against is recoverable by a restore."
        )
    return False, False, (
        f"{DEFERRED_HOST_RAM}: {avail / 1e9:.2f} GB available < "
        f"{need / 1e9:.2f} GB needed (transient "
        f"{int(projected_transient_bytes) / 1e9:.2f} + floor "
        f"{int(floor_bytes) / 1e9:.2f}); defer {int(defers_so_far) + 1} of "
        f"{max_defers}, the flip is retried next round"
    )


_DISK_TIER_ARM_WARNED = False


def _warn_first_disk_tier_arm(server_args) -> None:
    """Say once, on the first flip arm carrying a disk tier, that this path has
    a history (#703 review gate).

    A refusal here would be the counter-vs-actuator pattern -- the defect it
    named is fixed and covered -- but the path DID wedge once, so silence is
    not right either. Warning, not blocker, per corridor canon: the line exists
    so that a future regression is attributed in one grep instead of a
    bisect.
    """
    global _DISK_TIER_ARM_WARNED
    if _DISK_TIER_ARM_WARNED:
        return
    if not getattr(server_args, "enable_hierarchical_cache", False):
        return
    backend = getattr(server_args, "hicache_storage_backend", None)
    if not backend:
        return
    _DISK_TIER_ARM_WARNED = True
    logger.warning(
        "PHASE FLIP arming with a HiCache DISK tier (backend=%r). This "
        "combination wedged at warmup once (#630: PP x disk HiCache). It is "
        "ALLOWED because that wedge's root fix -- 9da9dfd025, bounded "
        "collectives in mem_cache/hicache_collective.py -- is an ancestor of "
        "this build, and test/registered/unit/mem_cache/"
        "test_hicache_bounded_waits_630.py is the active protection, not a "
        "refusal in flip_blocking_guards. If a warmup hang reappears on this "
        "path, start from that suite and that commit.",
        backend,
    )


def flip_blocking_guards(scheduler) -> List[str]:
    """Features that refuse flip arming (DESIGN_631 3.7). Mirrors the
    #297 Stage-A guard shape, plus the #630 PP x disk-HiCache wedge."""
    guards: List[str] = []
    server_args = scheduler.server_args
    try:
        from sglang.srt.disaggregation.utils import DisaggregationMode

        if scheduler.disaggregation_mode != DisaggregationMode.NULL:
            guards.append("PD disaggregation")
    except ImportError:
        pass
    # HiCache is a TIER SHAPE, not a feature flag (#703, same lesson the kvso
    # clause below already learned). This used to refuse arming whenever
    # hierarchical cache was merely ENABLED, which made the phase flip and any
    # prefix retention mutually exclusive: the deployment answered by running
    # `enable_hierarchical_cache=False`, i.e. with no cache tier at all on the
    # serving line. The wedge that earned the guard was specifically the DISK
    # tier at warmup, and it was fixed -- 9da9dfd025 bounded the collectives
    # (mem_cache/hicache_collective.py) and test_hicache_bounded_waits_630.py
    # covers it. The guard now names the tier that actually wedged, so the
    # device+host-local configuration can carry a prefix cache across the flip.
    # #703 stage 2: the #630 clause is GONE, not narrowed further. Keeping the
    # disk tier refused was my own stage-1 conservatism ("pending its own
    # evidence"), but the evidence is the same evidence that cleared the host
    # tier: the wedge's root fix is 9da9dfd025 (bounded collectives,
    # mem_cache/hicache_collective.py), it is an ancestor of every deployed
    # commit since, and test_hicache_bounded_waits_630.py covers it. A guard
    # cannot be justified by a defect that a green suite says is fixed.
    #
    # The live protection is that suite, not this clause. What remains gated is
    # the KV key's pp suffix, which is a statement about BYTES and belongs with
    # the whole-page format (#706) -- refusing the backend never protected the
    # bytes, it only prevented anyone from reaching them.
    #
    # LIFTED 2026-08-17, AGAINST THE CRITERION THIS CLAUSE ITSELF SET.
    #
    # The guard was restored earlier today because #630's desync was bounded but
    # not fixed, and it named its own exit condition verbatim: "Remove this only
    # when a test proves two ranks RENDEZVOUS, not when one proves a wait
    # expires." That test now exists, and the defect it was written against is
    # rooted rather than deferred.
    #
    # ROOT CAUSE, found with a 3-process real-gloo harness driving the real
    # _pp_sync: bounded_wait POLLED work.is_completed() and only called
    # work.wait() once the poll had already succeeded. is_completed() REPORTS,
    # wait() DRIVES -- so two polling peers never advanced the exchange and each
    # sat until its own deadline. The bound written to stop a hang was itself
    # the livelock. Fixed by handing the deadline to the wait
    # (hicache_collective.bounded_wait), which progresses AND stays bounded.
    #
    # THE EVIDENCE IS OF THE RIGHT KIND THIS TIME, which is the whole point of
    # the earlier withdrawal: test_pp_sync_rendezvous_630.py runs THREE REAL
    # PROCESSES over a REAL gloo group and asserts the ring rendezvouses with
    # the bound ACTIVE, that downstream ranks actually receive rank 0's values,
    # and that a dead peer still raises the named bounded error. Mutation-proven
    # -- restoring the poll turns the first two red. A mock suite could not have
    # produced any of those three facts, and that is exactly why the previous
    # removal, resting on one, was wrong.
    #
    # If this configuration ever wedges again, restore the clause and do NOT
    # accept a green mock suite as grounds to lift it a third time.
    # WARNING-NOT-BLOCKER (corridor canon): a disk tier WITHOUT a pipeline is
    # still allowed, and the first such arm says so exactly once, so a
    # regression there has an attribution line.
    _warn_first_disk_tier_arm(server_args)
    # kv-session-offload is a STATE, not a feature (#656, kvso_flip_contract).
    # This used to refuse arming whenever kvso was merely CONFIGURED, which
    # made the host half of spec items 6/12/15c and the phase flip mutually
    # exclusive: enabling the spill destination turned the flip off. The guard
    # now asks what kvso is DOING -- parked images stamped with the outgoing
    # layout are safe to carry across, a copy in flight or an unplaceable
    # image is not -- and refuses only the latter, for one round.
    kvso = getattr(scheduler, "kv_session_offload", None)
    if kvso is not None:
        from sglang.srt.managers.kvso_flip_contract import (
            FLIP_SAFE_STATES,
            flip_safety_state,
        )

        live_phase = getattr(scheduler, "phase_flip_active_stack", None)
        state, detail = flip_safety_state(
            kvso,
            current_phase=live_phase,
            incoming_phase=_PHASE_AFTER.get(_DIR_OF_PHASE.get(live_phase)),
        )
        if state not in FLIP_SAFE_STATES:
            guards.append(f"kv-session-offload {state}: {detail}")
    if getattr(scheduler, "is_dual_group_lane", False) or getattr(
        server_args, "dual_group_lane", None
    ):
        guards.append("dual-group lane")
    if not hasattr(scheduler.tree_cache, "all_values_flatten"):
        guards.append(
            f"tree cache {type(scheduler.tree_cache).__name__} (no "
            f"all_values_flatten enumeration)"
        )
    return guards


class PhaseFlipLoopExit(Exception):
    """Control-flow signal: a flip COMMITTED this round; the current event
    loop must exit to the re-dispatching wrapper (dispatch_event_loop picks
    its loop ONCE from pp_size, so a changed topology needs a fresh
    dispatch). Raised by the scheduler's on_round hook AFTER
    PhaseFlipRuntime.on_round returned commit stats -- never from inside
    the runtime, whose epoch/phase bookkeeping must complete first. The
    quiescence predicate guarantees the loop holds no half-processed batch
    state when this propagates."""

    def __init__(self, direction: str):
        super().__init__(direction)
        self.direction = direction


def derive_pp_full_attn_layer_map(
    full_attention_layer_ids: Sequence[int],
    num_hidden_layers: int,
    pp_size: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Per-stage FULL-ATTENTION ORDINALS from the global layer geometry.

    A pure replicated function of (the model's global full-attention layer
    ids, the layer count, the PP stage split) -- every rank derives the
    same map, which the consensus fingerprint then pins at runtime. The
    stage split comes from get_pp_indices, the SAME function the PP model
    build used (env-uniform SGLANG_PP_LAYER_PARTITION included), so the
    map cannot drift from the actual stage windows.

    IMPORTANT SOURCE RULE: ``full_attention_layer_ids`` must be the
    UNMUTATED global list (e.g. from the TP stack's model_config, whose
    pp_size=1 adjust is the identity) -- the PP stack's model_config was
    rewritten in place to its stage-local slice
    (adjust_hybrid_swa_layers_for_pp)."""
    from sglang.srt.distributed.utils import get_pp_indices

    ids = [int(x) for x in full_attention_layer_ids]
    if ids != sorted(set(ids)):
        raise KvReshardError(
            f"full_attention_layer_ids must be strictly ascending, got {ids}"
        )
    if ids and not (0 <= ids[0] and ids[-1] < num_hidden_layers):
        raise KvReshardError(
            f"full_attention_layer_ids {ids} outside [0, {num_hidden_layers})"
        )
    bounds = [get_pp_indices(num_hidden_layers, r, pp_size) for r in range(pp_size)]
    flat = [b for pair in bounds for b in pair]
    if flat != sorted(flat) or bounds[0][0] != 0 or bounds[-1][1] != num_hidden_layers:
        raise KvReshardError(
            f"PP stage bounds {bounds} do not partition [0, {num_hidden_layers})"
        )
    layer_map = []
    for start, end in bounds:
        layer_map.append(tuple(i for i, gid in enumerate(ids) if start <= gid < end))
    covered = sorted(o for stage in layer_map for o in stage)
    if covered != list(range(len(ids))):
        raise KvReshardError(
            f"stage map {layer_map} does not cover every full-attention "
            f"ordinal exactly once (bounds {bounds}, ids {ids})"
        )
    return tuple(layer_map)


def build_gdn_flip_guard(scheduler) -> Callable[[str], None]:
    """5.3 PLACEHOLDER for the GDN state mover, honest by refusal.

    The full mover (layer-axis -> head-axis re-shard of conv/ssm state via
    MambaPool blobs, DESIGN_631 3.4) lands as slice 5.3b. Until then a
    flip with LIVE linear-attention state must refuse LOUDLY inside the
    no-return region's first step -- before any pool byte moved -- never
    proceed and silently truncate GDN state (the #212 Store-Route lesson).
    The 5.5 validation ladder's first rung (flip empty -> flip back) is
    exactly what this permits."""

    def _guard(direction: str) -> None:
        running = getattr(scheduler, "running_batch", None)
        reqs = list(getattr(running, "reqs", []) or []) if running else []
        if reqs:
            raise KvReshardError(
                f"{LOG_PREFIX} flip {direction} refused: {len(reqs)} live "
                f"request(s) hold GDN conv/ssm state and the GDN state "
                f"mover is not wired yet (slice 5.3b); flipping now would "
                f"silently truncate linear-attention state. Drain or wait."
            )

    return _guard


def build_production_flip_cutover(scheduler, reduce_fn=None) -> Callable[[str], None]:
    """The cutover leg (DESIGN_631 3.6 step 5): everything the scheduler
    snapshotted from the boot topology is rebuilt for the target phase.
    Runs inside PhaseFlipRuntime._execute after KV/GDN/arena moves; the
    loop exit is raised LATER by the on_round hook (the runtime's
    epoch/phase bookkeeping must finish first)."""
    import dataclasses as _dc

    # Boot-phase snapshot for the return trip, taken ONCE at build time
    # (the scheduler's ps still holds the boot topology then).
    boot_ps = scheduler.ps
    boot_model_worker = scheduler.tp_worker

    def _cutover(direction: str) -> None:
        from sglang.srt.distributed import parallel_state as _ps
        from sglang.srt.distributed.utils import set_cp_token_ratios
        from sglang.srt.layers.dcp.owner import refresh_all_owner_bounds
        from sglang.srt.runtime_context import get_server_args

        # #690 capture A: cutover SUB-STEP timing. The aggregate cutover cost
        # spans 43.5 to 1041.5 ms -- a 24x spread, which is the signature of a
        # WAIT or a serialization, not of a fixed cost. The DONE line reports
        # only the total, so the spread cannot be attributed from it; the 43.5
        # ms minimum is the honest target. These marks cost one perf_counter
        # call per step and change no control flow.
        _marks = [("enter", time.perf_counter())]

        def _mark(label: str) -> None:
            _marks.append((label, time.perf_counter()))

        stacks = scheduler.phase_flip_stacks
        tp_phase = direction == PP_TO_TP
        n = len(stacks.vector)
        world_rank = _ps.get_world_group().rank_in_group

        # The phase's speculation state, decided ONCE here because two
        # separate steps below need the same answer: the component rebuild
        # (4b) and the scheduler's own swap (7). Speculation belongs to the
        # TP DECODE phase (#631) -- the draft worker was built on the flip's
        # TP stack at boot and is armed with it; the PP phase carries none,
        # which is bit-for-bit the state of an instance without speculation.
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        want_draft = stacks.draft_worker if tp_phase else None
        want_spec_algo = (
            scheduler.flip_spec_algorithm
            if (tp_phase and want_draft is not None)
            else SpeculativeAlgorithm.from_string(None)
        )
        if tp_phase:
            # Mirrors the boot dispatch: with speculation the model worker
            # IS the draft worker, which drives the target through it.
            want_model_worker = (
                want_draft if want_draft is not None else stacks.tp_worker
            )
        else:
            want_model_worker = boot_model_worker

        # 1. Module-level group routing (forward collectives resolve
        # through the parallel_state getters; see phase_flip_boot).
        _ps.set_phase_flip_tp_active(tp_phase)

        # 2. Owner rule: the vector is boot-constant; refresh the bounds
        # consumers so the TP backends read the (re)installed vector.
        # This is a TOKEN-space quantity, so it must be the token vector,
        # not the weight shard vector. They are equal unless
        # SGLANG_UNEVEN_TOKEN_VECTOR overrides the token side; reinstalling
        # the weight vector here would leave the owner rule splitting rows
        # under a different vector than the pools were SIZED under, which
        # is an out-of-bounds slot id, not a slow path.
        set_cp_token_ratios(list(stacks.token_vector))
        refresh_all_owner_bounds()

        _mark("routing+owner")
        # 3. Scheduler topology snapshot (frozen dataclass -> new instance).
        if tp_phase:
            scheduler.ps = _dc.replace(
                boot_ps,
                tp_rank=world_rank,
                tp_size=n,
                pp_rank=0,
                pp_size=1,
                attn_tp_rank=world_rank,
                attn_tp_size=n,
            )
        else:
            scheduler.ps = boot_ps

        # 4. Cached group handles, re-derived through the ROUTED getters.
        scheduler.tp_group = _ps.get_tp_group()
        scheduler.tp_cpu_group = scheduler.tp_group.cpu_group
        scheduler.attn_tp_group = _ps.get_attn_tp_group()
        scheduler.attn_tp_cpu_group = scheduler.attn_tp_group.cpu_group
        scheduler.pp_group = _ps.get_pp_group()
        # dp-attention is a flip arming guard; the dp routing group is tp.
        scheduler.dp_tp_group = scheduler.tp_group

        _mark("topology+groups")
        # 4b. Scheduler COMPONENTS holding ps / group snapshots (found on
        # the first post-flip serving attempt, 2026-08-08): the request
        # receiver kept the boot ps and relayed requests PP-chain-style
        # while rank 0 ran TP semantics -- one rank in the pool-budget
        # all_reduce, another in the relay's point_to_point recv, wedge.
        # The output streamer's stale ps mis-gated the detokenizer send
        # (heartbeat loss). Both are plain dataclasses over ps + group
        # handles; rebuild them against the freshly-routed handles. The
        # completeness self-check (step 9) pins each one.
        import dataclasses as _dc2

        scheduler.request_receiver = _dc2.replace(
            scheduler.request_receiver,
            ps=scheduler.ps,
            tp_group=scheduler.tp_group,
            tp_cpu_group=scheduler.tp_cpu_group,
            attn_tp_group=scheduler.attn_tp_group,
            attn_tp_cpu_group=scheduler.attn_tp_cpu_group,
        )
        # The streamer also holds a VALUE COPY of spec_algorithm, and that
        # copy decides whether the per-request spec counters are shipped at
        # all (_GenerationStreamAccumulator gates every spec_verify_ct /
        # accept-token append on it). A phase-flip instance parks
        # speculation at boot because it rests in the PP layout, so the copy
        # taken by init_output_streamer is NONE -- and refreshing only ps
        # left it NONE for the whole life of the process. The counters were
        # accumulated on the Req and then dropped on the way out, so
        # meta_info carried no spec_accept_length, /generate reported null,
        # and the TP phase looked like it was running without speculation
        # while it was in fact verifying normally. Refresh it from the phase
        # being entered, exactly like the batch result processor below.
        scheduler.output_streamer = _dc2.replace(
            scheduler.output_streamer,
            ps=scheduler.ps,
            spec_algorithm=want_spec_algo,
        )
        if getattr(scheduler, "load_inquirer", None) is not None:
            scheduler.load_inquirer = _dc2.replace(
                scheduler.load_inquirer, ps=scheduler.ps
            )
        # The batch result processor caches the WORKERS, and it is on the
        # decode hot path: with speculation it calls back into the spec
        # worker (on_verify_complete_cpu) to resolve verified tokens. Built
        # at boot, when a phase-flip instance deliberately has no draft
        # worker, so without this rebuild the first post-flip decode died
        # with "'TpModelWorker' object has no attribute
        # 'on_verify_complete_cpu'" -- the boot-cached target being asked
        # to behave like the draft stack that had just been armed around
        # it (measured, boot 21, 2026-08-08).
        if getattr(scheduler, "batch_result_processor", None) is not None:
            scheduler.batch_result_processor = _dc2.replace(
                scheduler.batch_result_processor,
                draft_worker=want_draft,
                model_worker=want_model_worker,
            )
        _mark("components")
        # 4c. Census round realign: the detector's cadence counter drifted
        # per-rank under the pp loop; the cutover is group-aligned, so
        # re-zero here or the post-flip detector fires its gloo
        # all_gather_object at per-rank rounds and mispairs with the
        # request broadcasts on the same group FIFO (measured wedge,
        # window-2 boot 13). See CollectiveCensus.realign_round.
        from sglang.srt.distributed.collective_census import census as _census

        _census().realign_round()

        # 5. pp_max_micro_batch_size for the new pp_size (boot formula).
        get_server_args().override(
            "phase_flip.pp_max_micro_batch_size",
            pp_max_micro_batch_size=max(
                scheduler.max_running_requests // scheduler.ps.pp_size, 1
            ),
        )

        _mark("census+microbatch")
        # 6. PP loop arrays: re-initialized for the new topology (reads the
        # NEW ps.pp_size).
        #
        # #631 J.3: this step USED TO DESTROY THE RESIDENT DECODE SET, and
        # that is the whole reason a flip under load was impossible. The
        # carry now lives inside init_pp_loop_state (it has three callers,
        # and the TP->PP leg re-enters event_loop_pp, which calls it again);
        # here we only bracket it with the evidence.
        #
        # The orphan check runs BEFORE the swap on purpose: a request
        # reachable only through last_mbs would mean the quiescence
        # predicate admitted a boundary that is not quiescent, and that is
        # a predicate bug to be raised, not a carry to be widened.
        from sglang.srt.managers.phase_flip_resident_carry import (
            assert_no_orphan_resident_reqs,
            promote_slot_zero_to_running_batch,
            resident_req_identity,
        )

        assert_no_orphan_resident_reqs(scheduler)
        resident_before = resident_req_identity(scheduler)
        # WHAT init_pp_loop_state IS ABOUT TO DESTROY, on the record.
        # It clears pp_outputs, last_rank_comm_queue, send_output_work and
        # the tensor-dict inbox with no drain and no carry. The request
        # side has a carry and a membership pin; the OUTPUT side has
        # neither, and a discarded output is a token the client never sees
        # (#631). Quiescence is supposed to make all of these empty -- this
        # line is what says so out loud instead of assuming it.
        _inflight = (
            getattr(scheduler, "pp_outputs", None) is not None,
            len(getattr(scheduler, "last_rank_comm_queue", None) or ()),
            len(getattr(scheduler, "send_output_work", None) or ()),
            sum(
                len(q) for q in getattr(scheduler, "_pp_tensor_dict_inbox", {}).values()
            ),
        )
        if any(_inflight):
            logger.warning(
                "%s CUTOVER DISCARDS IN-FLIGHT OUTPUT: pp_outputs=%s "
                "last_rank_comm_queue=%d send_output_work=%d inbox=%d -- "
                "each is a sampled token that reaches no output_ids",
                LOG_PREFIX,
                *_inflight,
            )
        else:
            logger.info("%s output path empty at cutover", LOG_PREFIX)
        scheduler.init_pp_loop_state()
        # 6b. The TP loops read ``running_batch``, not the slot array, so
        # the TP leg moves the re-seeded set over (and empties the slots,
        # or the next flip's harvest would resurrect a stale view of it).
        if tp_phase:
            promote_slot_zero_to_running_batch(scheduler)
        # 6c. MEMBERSHIP PIN, before the deferred aborts of step 8 are
        # allowed to change the set legitimately. Identity is (rid,
        # req_pool_idx): the slot ARRANGEMENT changes at a flip by design,
        # the MEMBERSHIP may not. A dropped request must fail here, loudly,
        # not surface a pass later as a stranded page and a stranded mamba
        # lock with the evidence already stale.
        resident_after = resident_req_identity(scheduler)
        if resident_after != resident_before:
            lost = [r for r in resident_before if r not in resident_after]
            gained = [r for r in resident_after if r not in resident_before]
            raise KvReshardError(
                f"{LOG_PREFIX} CUTOVER DROPPED THE RESIDENT DECODE SET "
                f"({direction}): {len(resident_before)} request(s) before, "
                f"{len(resident_after)} after; lost {lost[:8]}, gained "
                f"{gained[:8]}. Every request resident at a cutover must "
                f"survive it -- a dropped one strands its KV rows and its "
                f"mamba slot lock and its answer is simply never finished."
            )

        _mark("pp_loop_arrays")
        # 7. Active stack swap: the forward path follows model_worker.
        #
        # Speculation belongs to the TP DECODE phase (#631). The draft
        # worker was built on the flip's TP stack at boot and is armed
        # HERE, with the stack it targets; the PP phase runs with
        # spec_algorithm NONE and draft_worker None, which is bit-for-bit
        # the state an instance without speculation has. Mirrors the boot
        # dispatch in Scheduler.init_model_worker: with speculation the
        # model worker IS the draft worker, which drives the target
        # through it.
        scheduler.spec_algorithm = want_spec_algo
        scheduler.draft_worker = want_draft
        scheduler.model_worker = want_model_worker
        scheduler.phase_flip_active_stack = PHASE_TP if tp_phase else PHASE_PP

        _mark("stack_swap")
        # 7b. DRAFT STATE for the requests step 6 just carried in.
        #
        # AFTER the swap, deliberately: the pool that gets scrubbed is the
        # newly armed draft worker's, and the seed is installed for the
        # algorithm this scheduler now runs. Before the swap there is no
        # draft worker to ask.
        #
        # The PP phase has no draft worker at all, so a carried request has
        # never had a draft_extend and its draft-pool rows -- addressed by
        # the TARGET's slot ids, which are shared -- still hold the bytes of
        # whatever request last owned those slots. Without this the first
        # post-flip decode read a 0-row idle draft input and the graph
        # runner died (03:32:14Z, foreach_copy [1,1] vs [0,1]).
        #
        # BOTH directions, since 2026-08-09 20:31:48Z. The sentence that
        # used to stand here -- "the TP->PP leg is flipping speculation
        # OFF, and a request carried into a phase with no drafter needs no
        # draft state, its spec_info is simply not read there" -- is
        # FALSIFIED and cost the instance. spec_info is read on the TP->PP
        # side by ScheduleBatch.merge_batch, which has no drafter in it at
        # all. See corpse I in phase_flip_draft_bootstrap.
        from sglang.srt.managers.phase_flip_draft_bootstrap import (
            arm_draft_bootstrap_all_reachable,
            clear_spec_info_for_unspeculated_phase,
            retune_carried_batches_for_phase,
        )

        # Both directions: a carried batch's OWN spec_algorithm field still
        # says which phase BUILT it, and prepare_for_decode branches on it.
        # Retune before the bootstrap, so the batch and the scheduler agree
        # about the phase before anything reads either.
        retuned = retune_carried_batches_for_phase(scheduler, want_spec_algo)
        if retuned:
            logger.info(
                "%s retuned %d carried batch(es) to spec_algorithm=%s",
                LOG_PREFIX,
                retuned,
                want_spec_algo,
            )

        # 7b-i. SPILL RUNG 2 (#656 spec item 6): the draft weights.
        #
        # WHY HERE AND NOT AT THE PRE-WAVE SITE. The design note proposed
        # spilling before the wave loop, so the seam's own staging could use
        # the freed bytes. This site is later and gives up that bonus on
        # purpose: by the time control reaches it the active-stack swap has
        # already run, so on the PP leg ``scheduler.draft_worker`` IS None
        # and the drafter is unreachable through the scheduler by
        # construction. The corridor gain is unaffected -- the corridor
        # minimum that binds is measured in the PP PHASE, not at the seam --
        # so the safety is bought for nothing. Move it earlier only with a
        # measurement showing seam staging, not the PP steady state, is what
        # binds.
        #
        # ORDERING LAW: both legs are past the abandon decision (which is
        # taken in _execute before any byte moves), so neither can strand a
        # flip that then returns to its origin phase with no drafter.
        from sglang.srt.managers.phase_flip_spill import get_spill_ladder

        _ladder = get_spill_ladder(scheduler)

        if tp_phase:
            # RESTORE BEFORE THE BOOTSTRAP. arm_draft_bootstrap_all_reachable
            # scrubs the drafter's pool and therefore needs a drafter whose
            # weights are physically present. The bytes this commits were
            # priced into the pp->tp affordability verdict before the flip
            # committed, so reaching here means a rank already agreed it can
            # afford them.
            if _ladder is not None:
                _ladder.on_enter_tp(stacks.draft_worker)
            arm_draft_bootstrap_all_reachable(scheduler, want_draft)

            # #662-F4: AND THE MIRROR OF THE RECOVERY BELOW. Since the rung
            # funds the seam from whichever layout is RESIDENT, the tp_to_pp
            # leg pays out of the TP layout's pool -- the source, whose rows
            # above the live set hold nothing. Entering TP makes that pool
            # active again, so the relief taken against it has to be handed
            # back here, exactly as the PP pool's is on the other leg.
            #
            # Without this the reduction is a RATCHET: the seam restores each
            # layout to its own ``size``, which the shrink lowered, so the TP
            # pool would come back smaller after every tp_to_pp and never
            # climb. The grow is corridor-bounded inside ``recover``, so it
            # cannot breach the law to do it.
            from sglang.srt.managers.phase_flip_spill import recover_kv_backing

            recover_kv_backing(scheduler, reduce_fn=reduce_fn)
        else:
            # ``stacks.draft_worker``, not ``want_draft``: want_draft is None
            # on this leg by design (that is the point of the leg), while the
            # carrier is parked on the worker object the stacks still hold.
            if _ladder is not None:
                _ladder.on_enter_pp(stacks.draft_worker)

            # The scheduler's KV pool is the PP layout's, so entering PP makes
            # it the ACTIVE pool again and any residency relief taken against
            # it during the TP phase must be handed back. A cap that is never
            # lifted is a permanently smaller pool, which the standing rule
            # forbids; recovering here bounds the reduction to one phase.
            from sglang.srt.managers.phase_flip_spill import recover_kv_backing

            # #656 C22-e: the grow is rank-local by necessity, the ID SPACE it
            # produces may not be. See recover_kv_backing.
            recover_kv_backing(scheduler, reduce_fn=reduce_fn)

            # THE SEAM MUST LEAVE NO TP DRAFT STATE REACHABLE. Runs before
            # the relay re-seed, so that nothing between the stack swap and
            # the next event-loop iteration can observe a half-scrubbed
            # set of batches.
            spec_cleared, spec_rids = clear_spec_info_for_unspeculated_phase(scheduler)
            if spec_cleared:
                logger.info(
                    "%s cleared TP spec_info from %d reachable batch(es) "
                    "entering the PP phase (requests %s); the PP phase has "
                    "no drafter, and merge_batch dereferences the other "
                    "side's spec_info unconditionally",
                    LOG_PREFIX,
                    spec_cleared,
                    ", ".join(spec_rids) or "-",
                )

            # THE TP->PP LEG'S OWN HANDOVER. The PP phase's first decode
            # gathers its input token out of the future-map relay, and the
            # speculative phase it is leaving never wrote that relay (the
            # non-overlap V2 path installs next_draft_input directly and
            # skips _relay_forward_payload). Re-derive it from the requests
            # themselves, here, where the truth is present -- see
            # reseed_decode_input_relay for the rule and the falsifier.
            from sglang.srt.managers.phase_flip_resident_carry import (
                reseed_decode_input_relay,
            )

            reseeded = reseed_decode_input_relay(scheduler)
            if reseeded:
                logger.info(
                    "%s re-seeded the decode-input relay for %d carried request(s)",
                    LOG_PREFIX,
                    reseeded,
                )

        # 8. Deferred aborts drain in the first post-flip round.
        window = getattr(scheduler, "phase_flip_abort_window", None)
        if window is not None and window.active:
            drained = window.deactivate_and_drain()
            if drained:
                logger.info(
                    "%s drained %d deferred abort(s) after cutover",
                    LOG_PREFIX,
                    drained,
                )

        _mark("draft_state")
        # 9. Completeness self-check: every snapshot the rebuild list names
        # is verified against the routed source of truth, HERE, before the
        # first post-flip round can touch a stale handle. A missed rebuild
        # is a loud KvReshardError, never later corruption.
        verify_flip_cutover(scheduler, tp_phase)
        # 10. Publish the active layout for the API process (#631): the
        # log line below is the authoritative RECORD, but it is not
        # QUERYABLE, and utilisation cannot substitute for it -- a
        # pipelined PP prefill saturates all three cards exactly as TP
        # does. Published after verify, so what is advertised is a
        # cutover that passed its completeness check.
        from sglang.srt.managers.phase_flip_presence import publish_active_phase

        publish_active_phase(world_rank, scheduler.phase_flip_active_stack)
        # #631: dump the pre-cutover output clocks and arm the post-cutover
        # countdown. Placed after verify so the trace's own state cannot
        # sit between a failed completeness check and the exception.
        from sglang.srt.managers.phase_flip_output_trace import trace_cutover

        # #719: move the HiCache pool bindings to the phase that is now active.
        # Runs AFTER the stack swap, because the incoming pools are what it
        # binds to -- the mirror of #703's writeback, which runs BEFORE
        # anything moves because it reads the outgoing ones.
        #
        # Refusal is logged, not raised, for the reason the seam always gives:
        # with requests parked a raise takes down an instance that was serving
        # fine. A refused rebind is SAFE by construction -- the binding does not
        # move, so #718 keeps device-tier I/O disarmed in this phase, which is
        # exactly the state that held before this feature existed.
        try:
            from sglang.srt.mem_cache.hicache_phase_binding import (
                rebind_for_cutover,
            )

            rebind_for_cutover(scheduler, "tp" if tp_phase else "pp")
        except Exception as e:
            logger.error(
                "%s #719 HiCache rebind refused (%s); device-tier I/O stays "
                "disarmed for this phase and the flip continues.",
                LOG_PREFIX,
                e,
            )
        trace_cutover(scheduler, direction)
        _mark("verify+publish+trace")
        # #690 capture A: one line, sorted by cost, so the 24x spread can be
        # ATTRIBUTED instead of guessed at. Emitted after the completeness
        # check so a cutover that failed verification never reports timings as
        # if it had succeeded.
        try:
            steps = [
                (_marks[i + 1][0], (_marks[i + 1][1] - _marks[i][1]) * 1000.0)
                for i in range(len(_marks) - 1)
            ]
            total_ms = (_marks[-1][1] - _marks[0][1]) * 1000.0
            worst = sorted(steps, key=lambda kv: kv[1], reverse=True)
            logger.warning(
                "%s CUTOVER SUB-STEPS %s total=%.1f ms | %s",
                LOG_PREFIX,
                direction,
                total_ms,
                ", ".join(f"{k}={v:.1f}" for k, v in worst),
            )
        except Exception:  # noqa: BLE001 - timing must never break a cutover
            pass
        logger.warning(
            "%s cutover complete: active stack %s, ps tp=%d pp=%d",
            LOG_PREFIX,
            scheduler.phase_flip_active_stack,
            scheduler.ps.tp_size,
            scheduler.ps.pp_size,
        )

    return _cutover


def verify_flip_cutover(scheduler, tp_phase: bool) -> None:
    """Post-cutover invariants (the coordinator's completeness pin): every
    scheduler snapshot on the 5.3 rebuild list must AGREE with the routed
    source of truth for the now-active phase. Any single stale reference
    -- a cached group handle still pointing at the other phase's group, a
    ps that kept the old topology, a model_worker from the wrong stack --
    fails HERE, loudly, before any round runs on it."""
    from sglang.srt.distributed import parallel_state as _ps

    stale = []
    if _ps.phase_flip_tp_routing_active() != tp_phase:
        stale.append(
            f"module routing active={_ps.phase_flip_tp_routing_active()} "
            f"but tp_phase={tp_phase}"
        )
    expect_tp = _ps.get_tp_group()
    expect_attn = _ps.get_attn_tp_group()
    expect_pp = _ps.get_pp_group()
    if scheduler.tp_group is not expect_tp:
        stale.append("tp_group")
    if scheduler.tp_cpu_group is not expect_tp.cpu_group:
        stale.append("tp_cpu_group")
    if scheduler.attn_tp_group is not expect_attn:
        stale.append("attn_tp_group")
    if scheduler.attn_tp_cpu_group is not expect_attn.cpu_group:
        stale.append("attn_tp_cpu_group")
    if scheduler.pp_group is not expect_pp:
        stale.append("pp_group")
    if scheduler.dp_tp_group is not scheduler.tp_group:
        stale.append("dp_tp_group")
    stacks = scheduler.phase_flip_stacks
    n = len(stacks.vector)
    want_tp_size = n if tp_phase else 1
    want_pp_size = 1 if tp_phase else n
    if scheduler.ps.tp_size != want_tp_size or scheduler.ps.pp_size != want_pp_size:
        stale.append(
            f"ps topology (tp={scheduler.ps.tp_size}, "
            f"pp={scheduler.ps.pp_size}; want tp={want_tp_size}, "
            f"pp={want_pp_size})"
        )
    if scheduler.ps.attn_tp_size != want_tp_size:
        stale.append(f"ps.attn_tp_size ({scheduler.ps.attn_tp_size})")
    # Speculation state, pinned per phase (#631). The TP phase must carry
    # the configured algorithm AND the draft worker built on the TP stack;
    # the PP phase must carry neither. A half-armed cutover -- the
    # algorithm swapped in without its draft worker, or a draft worker
    # left armed against the PP stack it was never built for -- is the
    # silent-corruption shape this check exists to refuse.
    want_draft = stacks.draft_worker if tp_phase else None
    if getattr(scheduler, "draft_worker", None) is not want_draft:
        stale.append("draft_worker (wrong phase)")
    want_algo_none = not tp_phase or stacks.draft_worker is None
    if scheduler.spec_algorithm.is_none() != want_algo_none:
        stale.append(
            f"spec_algorithm ({scheduler.spec_algorithm}; want "
            f"{'none' if want_algo_none else 'the configured algorithm'})"
        )
    if tp_phase:
        want_worker = (
            stacks.draft_worker if stacks.draft_worker is not None else stacks.tp_worker
        )
    else:
        want_worker = scheduler.tp_worker
    if scheduler.model_worker is not want_worker:
        stale.append("model_worker (wrong stack)")
    # Component ps/group snapshots (step 4b): each holder rebuilt at
    # cutover must reference the CURRENT ps object and routed groups --
    # a stale receiver relays requests in the other phase's topology
    # (measured wedge, first post-flip serving attempt 2026-08-08).
    receiver = getattr(scheduler, "request_receiver", None)
    if receiver is not None:
        if receiver.ps is not scheduler.ps:
            stale.append("request_receiver.ps")
        if receiver.attn_tp_group is not scheduler.attn_tp_group:
            stale.append("request_receiver.attn_tp_group")
        if receiver.tp_cpu_group is not scheduler.tp_cpu_group:
            stale.append("request_receiver.tp_cpu_group")
    streamer = getattr(scheduler, "output_streamer", None)
    if streamer is not None:
        if streamer.ps is not scheduler.ps:
            stale.append("output_streamer.ps")
        # Pins the spec-counter wire: a stale copy here is silent (answers
        # stay correct, only the acceptance evidence disappears), which is
        # exactly the kind of defect this self-check exists to make loud.
        if streamer.spec_algorithm is not scheduler.spec_algorithm:
            stale.append("output_streamer.spec_algorithm")
    brp = getattr(scheduler, "batch_result_processor", None)
    if brp is not None:
        # On the decode hot path, and it calls into the SPEC worker.
        if brp.model_worker is not scheduler.model_worker:
            stale.append("batch_result_processor.model_worker")
        if brp.draft_worker is not scheduler.draft_worker:
            stale.append("batch_result_processor.draft_worker")
    inquirer = getattr(scheduler, "load_inquirer", None)
    if inquirer is not None and inquirer.ps is not scheduler.ps:
        stale.append("load_inquirer.ps")
    window = getattr(scheduler, "phase_flip_abort_window", None)
    if window is not None and window.active:
        stale.append("abort window still active (drain missed)")
    # The resident decode set must live in the handle the ACTIVE phase's
    # event loop reads, and nowhere else (#631 J.3). The TP loops read
    # ``running_batch``; ``event_loop_pp`` reads the slot array and rebinds
    # ``running_batch`` per slot. A set left in the other phase's handle is
    # not merely untidy: it is invisible to the loop that is now running,
    # so those requests never decode again, and it is a second ageing view
    # that the next flip's harvest would resurrect.
    slots = list(getattr(scheduler, "running_mbs", []) or [])
    slot_resident = [
        i for i, mb in enumerate(slots) if len(getattr(mb, "reqs", []) or [])
    ]
    running = getattr(scheduler, "running_batch", None)
    running_n = len(getattr(running, "reqs", []) or [])
    if tp_phase and slot_resident:
        stale.append(
            f"resident requests left in the PP slot array {slot_resident} "
            f"while the TP loop reads running_batch"
        )
    if not tp_phase and running_n and not any(running is mb for mb in slots):
        stale.append(
            f"running_batch holds {running_n} resident request(s) that are "
            f"in no PP slot, so event_loop_pp will never see them"
        )
    if stale:
        raise KvReshardError(
            f"{LOG_PREFIX} CUTOVER INCOMPLETE ({'tp' if tp_phase else 'pp'} "
            f"phase): stale after rebuild: {', '.join(stale)}. A stale "
            f"snapshot surviving cutover is the silent-corruption class "
            f"this check exists to catch -- refusing to run a round on it."
        )


def build_phase_flip_runtime(scheduler) -> "PhaseFlipRuntime":
    """Factory mirroring build_kv_reshard_runtime (kv_reshard.py): wires
    the scheduler's real state into PhaseFlipRuntime. Called lazily from
    the first scheduler round (house pattern); by then the boot builder
    has installed scheduler.phase_flip_stacks."""
    from sglang.srt.distributed.parallel_state import (
        get_phase_flip_group,
        get_world_group,
    )
    from sglang.srt.managers.kv_pressure_runtime import default_collective_min
    from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence
    from sglang.srt.managers.kv_reshard import _dist_exchange

    stacks = scheduler.phase_flip_stacks
    if stacks is None:
        raise KvReshardError(
            "build_phase_flip_runtime before build_phase_flip_tp_stack "
            "(the boot builder owns pools, arena and images)"
        )
    server_args = scheduler.server_args
    flip_tp = get_phase_flip_group("tp")
    world = get_world_group()

    pp_pool = scheduler.tp_worker.model_runner.token_to_kv_pool
    tp_pool = stacks.tp_worker.model_runner.token_to_kv_pool
    for name, pool in (("PP", pp_pool), ("TP", tp_pool)):
        if not hasattr(pool, "full_kv_pool"):
            raise KvReshardError(
                f"the {name} stack's pool {type(pool).__name__} has no "
                f"full_kv_pool; the flip moves hybrid-model full-attention "
                f"KV only (DESIGN_631 scope)"
            )
    pp_full = pp_pool.full_kv_pool
    tp_full = tp_pool.full_kv_pool
    pp_view = KvPoolView(pp_full.k_buffer, pp_full.v_buffer)
    tp_view = KvPoolView(tp_full.k_buffer, tp_full.v_buffer)

    # Global full-attention geometry from the TP stack's config (pp=1 ->
    # unmutated; the PP stack's was rewritten to its stage-local slice).
    # full_attention_layer_ids is a property of the HYBRID HF text config
    # (Qwen3NextConfig etc.), not of sglang's ModelConfig wrapper -- the
    # attention registry reads it via runner.mambaish_config, mirror that
    # (first real-metal flip boot, 2026-08-08).
    tp_model_config = stacks.tp_worker.model_config
    full_ids = list(tp_model_config.hf_text_config.full_attention_layer_ids)
    layer_map = derive_pp_full_attn_layer_map(
        full_ids,
        int(tp_model_config.num_hidden_layers),
        int(server_args.pp_size),
    )

    # #656 C22-e: ONE channel, named once, so the post-cutover recovery
    # levelling and the seam's own ballots provably run over the SAME group.
    # Built here rather than inside the cutover closure because the closure is
    # constructed as an argument to the runtime that would otherwise own it.
    flip_collective_min = default_collective_min(
        flip_tp.cpu_group, label="phase_flip.consensus"
    )

    runtime = PhaseFlipRuntime(
        n_ranks=world.world_size,
        rank=world.rank_in_group,
        layer_map=layer_map,
        n_layers=len(full_ids),
        # build_phase_flip_transition documents this as "the weighted DCP
        # token vector of the TP layout" -- which rank OWNS which rows, a
        # token-space question. The weight shard vector answers a different
        # one.
        tp_vector=stacks.token_vector,
        boot_phase=PHASE_PP,
        consensus_interval=int(
            getattr(server_args, "kv_reshard_consensus_interval", 8)
        ),
        park_deadline_s=park_deadline_s(),
        # The flip must leave the user's reserve alone, so it takes the
        # number from the same flag the rest of the server does rather than
        # carrying a second opinion about how much VRAM is not ours.
        #
        # #662: BUT THE SEAM IS A TRANSIENT, AND THE RESERVE IS A BAND. The
        # operator's corridor is 1024 MiB +-20 %, and the verdict is the
        # continuous minimum against the band FLOOR -- a cutover that dips to
        # 819 MiB for the length of a wave walk is lawful, which is the whole
        # reason the band was granted. Holding the CENTRE as a hard reserve
        # during staging spends that tolerance on nothing and refuses flips
        # the law permits.
        #
        # Measured on this rig 2026-08-15, GATE C, device 0:
        #   staging 1059 MiB needed, only 1000 MiB spendable, refused by 59;
        #   eight consecutive refusals then latched the direction "unfundable"
        #   and the instance held in TP with 50k+ tokens pending at
        #   1000-1600 tok/s, where the PP layout does 4000-7000.
        # Against the band floor the same instant offers 2024 - 819 = 1205
        # MiB and the flip funds.
        #
        # SCOPED TO THE SEAM DELIBERATELY. This is the staging reserve, not
        # the corridor law: the guard still judges every ordinary allocation
        # against the centre, and only the cutover -- bounded, unanimous and
        # over in seconds -- may reach into the band's tolerance.
        staging_reserve_bytes=_seam_staging_reserve_bytes(server_args),
        # Label it as OURS: a shared helper reporting under its own
        # module's name sent a live wedge into the wrong subsystem.
        collective_min=flip_collective_min,
        # #631 option 2(b): the pollable entry gate, and the non-blocking
        # pump that delivers this rank's arm forward while it waits.
        presence=PhaseFlipPresence(
            n_ranks=world.world_size,
            rank=world.rank_in_group,
        ),
        # #631 G: ONE mechanism where there used to be two half-working
        # ones. pump_fn and drain_fn are gone from the wiring -- both were
        # built on is_completed(), which never fires on this transport in
        # either direction (corpse F), so neither ever moved a byte. The
        # service turn does their job with a predicate the transport can
        # honour: a counter published on /dev/shm strictly after each
        # isend is posted.
        #
        # It is wired on EVERY rank, unlike the pair it replaces, which
        # were gated on the chain receiver and were therefore off on rank
        # 0 -- the intake rank, the one whose starvation defined corpse G.
        service_fn=getattr(scheduler, "pp_flip_service", None),
        channels_empty_fn=getattr(scheduler, "pp_flip_channels_empty", None),
        # (i) withhold presence until this rank's own forward is flushed,
        # so the flag means "I owe no send" rather than merely "I am
        # armed". This is now a condition that can be REACHED: the service
        # turn reaps the handle once the downstream's counter proves the
        # message consumed, where the pump could only ever fail to.
        owes_send_fn=getattr(scheduler, "pp_owes_chain_send", None),
        exchange=_dist_exchange(flip_tp.device_group, pp_view.device),
        pp_pool_view=pp_view,
        tp_pool_view=tp_view,
        live_slots_fn=build_flip_live_slots_fn(scheduler),
        ready_fn=build_flip_quiescence_fn(scheduler),
        cutover_fn=build_production_flip_cutover(
            scheduler, reduce_fn=flip_collective_min
        ),
        # DESIGN_631 3.6 order inside the no-return region: GDN state move
        # (5.3b mover -- its preconditions re-validate on every flip and
        # refuse loudly, the reachable-refusal contract), then the arena
        # refill. The full-attn KV move ran before these by the runtime.
        pre_cutover_fns=_labelled_movers(
            (_build_gdn_leg(scheduler), "gdn_state"),
            (stacks.refill, "weights_refill"),
        ),
        pre_write_fns=(
            _build_kv_backing_swap(scheduler, stacks, layer_map[world.rank_in_group]),
        ),
        guards=flip_blocking_guards(scheduler),
    )
    # #631 J: read-only handle for the pool census straddling the cutover.
    runtime._census_scheduler = scheduler
    # #665-F1 item 7: price the seam at boot instead of discovering it at arm
    # time. The design point is the live set at the deployed ladder's arming
    # threshold -- the smallest backlog that will actually ask for a flip, so
    # the projection answers "can the seam this config will attempt be paid
    # for?" A projection must never fail a boot, hence the broad guard inside.
    try:
        policy_cfg = getattr(scheduler, "phase_policy_cfg", None)
        design_tokens = int(getattr(policy_cfg, "flip_tokens", 0) or 0)
        page = int(
            getattr(
                getattr(scheduler, "token_to_kv_pool_allocator", None),
                "page_size",
                1,
            )
            or 1
        )
        if design_tokens > 0:
            base = max(1, -(-design_tokens // page))
            # THE LIVE SET IS THE RESIDENT POOL, NOT THE PENDING BACKLOG.
            # Projecting at the arming threshold under-read a real flip by a
            # constant ~511 MiB, and multiplying the threshold barely moved it
            # (+93 MiB for 4x the slots). Solving the slope put the real live
            # set near 200k slots -- i.e. the OCCUPIED POOL, which is what the
            # seam carries across, not the backlog that triggered it.
            #
            # So the named points are pool occupancies, which is both the
            # honest assumption and the one an operator can match to intended
            # load. The threshold point is kept as the floor of the range.
            pool = int(getattr(scheduler, "max_total_num_tokens", 0) or 0)
            pool_slots = max(0, -(-pool // page))
            points = [("threshold", base)]
            for frac in (0.50, 0.75, 0.90):
                if pool_slots > 0:
                    points.append(
                        (f"pool @ {int(frac * 100)}%", int(pool_slots * frac))
                    )
            runtime.log_staging_projection(points=tuple(points))
    except Exception as exc:
        logger.warning("%s staging projection skipped: %r", LOG_PREFIX, exc)
    return runtime


def _labelled_movers(*pairs) -> Tuple[Callable[[str], None], ...]:
    """Tag each pre-cutover mover with the name the seam census reports.

    A closure per pair rather than ``setattr`` on the callables: one of
    them is a BOUND METHOD (``stacks.refill``), and setting an attribute
    on a bound method either fails or silently lands on the underlying
    function and is then shared by every instance. Wrapping keeps the
    label a property of THIS wiring, which is what it actually is.
    """
    movers = []
    for fn, label in pairs:

        def _mover(direction: str, _fn=fn) -> None:
            _fn(direction)

        _mover.census_label = label
        movers.append(_mover)
    return tuple(movers)


def _in_wave(layers: Sequence[int], wave_set) -> List[int]:
    """``layers`` restricted to one seam wave; a None wave means all of them."""
    if wave_set is None:
        return list(layers)
    return [f for f in layers if f in wave_set]


class WavedBackingSwap:
    """The runtime half of exclusive KV backing, driven ONE WAVE AT A TIME.

    The seam is the single instant where the source layout has been fully
    read and the destination not yet written, so it is the only place the
    physical pages may move between layouts. Doing the whole pool there
    forces every crossing byte to be staged at once -- the unbounded term
    that wedged a 270k-token request (HANDOFF_666).

    Per wave the accounting nets out by construction: rank ``r`` releases
    the wave's layers it owns in the PP layout (each spanning the full PP
    pool) and commits the wave's layers in the TP layout (each spanning
    its token share), and ``layer_waves`` sizes those to be equal. So the
    pool residency never rises above the resting layout while the staged
    bytes fall by the wave count.

    ``__call__`` keeps the whole-pool behaviour for a single-wave seam and
    for callers that do not know about waves, so wave count 1 is
    byte-identical to the pre-wave code.
    """

    def __init__(self, scheduler, stacks, my_layers: Sequence[int]):
        from sglang.srt.managers.phase_flip_spill import (
            release_allocator_cache,
            resolve_spill_depth,
        )

        self._pp_pool = scheduler.tp_worker.model_runner.token_to_kv_pool
        self._tp_pool = stacks.tp_worker.model_runner.token_to_kv_pool
        self._my_layers = tuple(int(f) for f in my_layers)
        self._release_allocator_cache = release_allocator_cache
        # Resolved ONCE, at wiring time, so a malformed depth is a
        # boot-time refusal and not an exception thrown inside a cutover
        # that has already released the source pool's pages.
        self._spill_depth = resolve_spill_depth(getattr(scheduler, "server_args", None))
        # The rank's OWN device. Every worker here has all three cards
        # visible, so a bare current-device read can name a card this rank
        # does not own.
        self._spill_device = getattr(scheduler.tp_worker.model_runner, "gpu_id", None)

    @property
    def is_swappable(self) -> bool:
        return hasattr(self._pp_pool, "release_backing") and hasattr(
            self._tp_pool, "release_backing"
        )

    def _pools(self, direction: str):
        return (
            (self._pp_pool, self._tp_pool)
            if direction == PP_TO_TP
            else (self._tp_pool, self._pp_pool)
        )

    def _pool_local(self, pool_is_pp: bool, ordinals: Sequence[int]):
        """Pool-local layer indices of global ordinals.

        The TP pool holds every ordinal, so the index IS the ordinal. The
        PP pool holds only this stage's block, so the index is the
        position within it -- and ordinals outside the block simply are
        not this pool's business.
        """
        if not pool_is_pp:
            return [int(f) for f in ordinals]
        return [
            self._my_layers.index(int(f)) for f in ordinals if int(f) in self._my_layers
        ]

    def release_wave(self, direction: str, wave: Sequence[int]) -> None:
        """Hand back the source layout's pages for this wave's layers.

        Safe because every row this wave owes has already been read into
        the payloads; nothing reads those layers again.
        """
        src, _dst = self._pools(direction)
        if not hasattr(src, "release_backing"):
            return
        layers = self._pool_local(direction == PP_TO_TP, wave)
        if not layers:
            return
        src.release_backing(layers)
        seam_census.mark("backing_release")

    # -- #631 section 2.1: ROW-BLOCK granularity -----------------------------
    #
    # The transient the wave order can only MOVE, blocking can SHRINK. Under
    # restore-first the seam holds, at its worst instant, everything
    # committed through step j against everything released through j-1 --
    # one commit unit. With a whole layer as that unit the term is one
    # layer span (1821 MiB at pool 600000 on this rig, against a 753 MiB
    # budget: HANDOFF_669 section 3). Nothing requires the unit to be a
    # layer; it is a layer only because ``restore_backing`` takes a layer
    # list. Split each layer's rows into B blocks and the term becomes one
    # BLOCK span, which is a knob rather than a geometry constant.
    #
    # These two are the span-granular twins of release_wave/restore_wave.
    # They deliberately do NOT mark residency: a span restore is one step
    # of a stream, and ``restore_backing(layers)`` is what completes it
    # (memory_pool.restore_backing_span says so). ``_execute`` calls that
    # finaliser once per wave after the last block, which is safe only
    # because ``commit_range`` now consults the extent list rather than the
    # contiguous watermark -- without that fix the finaliser would re-map
    # extents the stream had already mapped.

    def is_span_swappable(self, direction: str) -> bool:
        """Can BOTH sides do span-granular backing on this direction?

        THE METHOD BEING PRESENT IS NOT THE QUESTION, and answering from
        ``hasattr`` alone is how the streamed seam shipped unrunnable.
        ``commit_span``/``decommit_span`` RAISE unless the arena was built
        with a commit chunk (``KvVmmArena._require_chunk``): without one the
        arena maps a single monolithic extent per buffer, and ``cuMemUnmap``
        only takes whole mappings, so a range op could release everything or
        nothing. ``SGLANG_FLIP_SEAM_CHUNK_MIB`` defaults to 0, so on a
        default boot both methods exist and both throw -- and the throw
        would land inside the flip's no-return region, where the exception
        climbs out into the event loop and takes all three ranks with it.

        So ask the pool what its ARENA can do. Pools that predate the
        property are treated as incapable rather than assumed capable: the
        cost of a false no is a slower seam, the cost of a false yes is the
        instance.
        """
        src, dst = self._pools(direction)
        return bool(
            getattr(src, "supports_backing_spans", False)
            and getattr(dst, "supports_backing_spans", False)
        )

    def commit_chunk_bytes(self, direction: str) -> int:
        """The destination arena's commit granule, 0 if it has none.

        The seam's staging reservation needs it as a FLOOR: backing moves
        in whole chunks and ``commit_span`` rounds outward, so a row block
        can never commit less than one chunk per buffer however fine the
        blocking gets (``PhaseFlipRuntime._seam_chunk_floor``).
        """
        _src, dst = self._pools(direction)
        return int(getattr(dst, "backing_commit_chunk_bytes", 0) or 0)

    def release_wave_span(
        self, direction: str, wave: Sequence[int], lo_row: int, hi_row: int
    ) -> None:
        src, _dst = self._pools(direction)
        if not hasattr(src, "release_backing_span"):
            return
        layers = self._pool_local(direction == PP_TO_TP, wave)
        if not layers or hi_row <= lo_row:
            return
        src.release_backing_span(layers, int(lo_row), int(hi_row))
        seam_census.mark("backing_release_span")

    def restore_wave_span(
        self, direction: str, wave: Sequence[int], lo_row: int, hi_row: int
    ) -> None:
        _src, dst = self._pools(direction)
        if not hasattr(dst, "restore_backing_span"):
            return
        layers = self._pool_local(direction == TP_TO_PP, wave)
        if not layers or hi_row <= lo_row:
            return
        dst.restore_backing_span(layers, int(lo_row), int(hi_row))
        seam_census.mark("backing_restore_span")

    def finalize_wave(self, direction: str, wave: Sequence[int]) -> None:
        """Mark the wave's destination layers resident after a streamed restore.

        The span restores above leave the pool "not fully backed" on
        purpose, because a partially backed pool must answer no to
        ``backing_is_resident``. This is the call that closes the wave.
        """
        _src, dst = self._pools(direction)
        if not hasattr(dst, "restore_backing"):
            return
        layers = self._pool_local(direction == TP_TO_PP, wave)
        if not layers:
            return
        dst.restore_backing(layers)
        seam_census.mark("backing_restore")

    def reclaim_between(self, direction: str) -> None:
        """SPILL RUNG 1 (#656 spec item 6), once per flip.

        This is the only instant in the whole cycle where it is both safe
        and maximally productive: the outgoing layout's scratch is dead by
        construction, the source pool's pages have just gone back to the
        driver, and the restore below asks the driver for RAW pages whose
        documented failure mode is precisely torch sitting on blocks it is
        not using. Reclaiming BEFORE the ask turns a retry-after-OOM into a
        first-attempt success. Censused on its own so the corridor credit
        is attributable to this rung and not netted out against the
        re-commit.
        """
        released = self._release_allocator_cache(
            direction, depth=self._spill_depth, device_index=self._spill_device
        )
        if released:
            seam_census.mark("allocator_cache_release")

    def restore_wave(self, direction: str, wave: Sequence[int]) -> None:
        """Re-map the destination layout's pages for this wave's layers.

        The restore CAN fail for want of memory. Boot sizes the budget for
        max(PP, TP) and this wave's source pages were just handed back, so
        the span is covered UNLESS something else on the card took physical
        pages meanwhile -- torch's caching allocator does exactly that, and
        the arena needs RAW driver pages. Measured 2026-08-09 under the
        mixed acceptance load: cuMemCreate failed with OUT_OF_MEMORY inside
        restore_backing and SIGQUIT took the instance down. The
        reclaim-and-retry now lives at the allocation itself
        (``_mem_create_reclaiming``); if it still raises, the card is
        genuinely full and it raises loudly here rather than corrupting
        anything.

        The two halves are censused SEPARATELY on purpose: a single mark
        around the pair would net the release against the re-commit and
        report their difference, which is exactly the reading that hides a
        re-commit that cannot be served.
        """
        _src, dst = self._pools(direction)
        if not hasattr(dst, "restore_backing"):
            return
        layers = self._pool_local(direction == TP_TO_PP, wave)
        if not layers:
            return
        dst.restore_backing(layers)
        seam_census.mark("backing_restore")

    def __call__(self, direction: str) -> None:
        """Whole-pool swap: release the source, reclaim, restore the target.

        SOURCE FIRST, and the order is the whole point. Restoring the
        destination first would hold both layouts' pages for the width of
        the swap, which is precisely the residency being removed -- and the
        corridor floor is a CONTINUOUS minimum, so a peak that lasts only a
        few milliseconds still counts against it.
        """
        src, dst = self._pools(direction)
        if not hasattr(src, "release_backing"):
            return
        src.release_backing()
        seam_census.mark("backing_release")
        self.reclaim_between(direction)
        dst.restore_backing()
        seam_census.mark("backing_restore")


def _build_kv_backing_swap(scheduler, stacks, my_layers) -> "WavedBackingSwap":
    return WavedBackingSwap(scheduler, stacks, my_layers)


def _build_gdn_leg(scheduler) -> Callable[[str], None]:
    from sglang.srt.managers.gdn_flip_mover import build_gdn_flip_mover

    return build_gdn_flip_mover(scheduler)


class PhaseFlipRuntime:
    """Drives one group's PP<->TP KV layout flip at a quiescent boundary.

    Injectables mirror ``KvReshardRuntime`` so the hermetic tests drive
    REAL threads through mock channels: ``collective_min`` is the
    consensus channel, ``exchange`` the pairwise byte channel,
    ``pp_pool_view``/``tp_pool_view`` the two resident pools (PP view
    layers = this stage's ordinals ascending; TP view layers = ALL
    ordinals ascending), ``live_slots_fn`` the replicated live slot
    enumeration (tree values UNION parked requests' rows -- DESIGN_631
    section 3.5), ``ready_fn`` the flip quiescence predicate,
    ``cutover_fn(direction)`` the snapshot-cache installer,
    ``pre_write_fns`` run at the read/write seam (cross-phase KV backing
    swap); ``pre_cutover_fns`` the ordered extra movers (weights arena, GDN
    state) executed inside the no-return region before cutover.
    """

    def __init__(
        self,
        *,
        n_ranks: int,
        rank: int,
        layer_map: Sequence[Sequence[int]],
        n_layers: int,
        tp_vector: Sequence[int],
        boot_phase: str = PHASE_PP,
        consensus_interval: int = 8,
        park_deadline_s: float = DEFAULT_PARK_DEADLINE_S,
        # #631: VRAM the flip's staging buffers must NOT eat into. Mirrors
        # --rank-user-reserve-mib, because it is the same promise: that many
        # MiB stay free on the card for everything that is not this server.
        staging_reserve_bytes: int = DEFAULT_STAGING_RESERVE_BYTES,
        # () -> (driver_free_bytes, allocator_cached_free_bytes). Injected
        # in tests; None reads the live CUDA allocator.
        mem_probe: Optional[Callable[[], Tuple[int, int]]] = None,
        presence=None,
        pump_fn: Optional[Callable[[], None]] = None,
        # #631 clause (ii): consume whatever the upstream has already sent,
        # without blocking, so no peer can block on this armed rank.
        drain_fn: Optional[Callable[[], None]] = None,
        # #631 clause (i): True while this rank still owes a chain send.
        # Presence is withheld until it reads False, so the flag means
        # "my chain is flushed; I owe no send".
        owes_send_fn: Optional[Callable[[], bool]] = None,
        # #631 G: ONE TURN OF THE ARMED SERVICE LOOP. Consume every inbound
        # message the upstream's counter accounts for, then reap this
        # rank's own sends the downstream's counter proves consumed. It
        # subsumes pump_fn and drain_fn, which were the same intent built
        # on is_completed() -- a predicate this transport never satisfies
        # (corpse F), so they moved nothing.
        service_fn: Optional[Callable[[], None]] = None,
        # #631 G: returns None when every channel of this rank is empty,
        # else a human-readable reason. Flip-commit hygiene: a message in
        # flight across the re-formation misframes the post-flip stream.
        channels_empty_fn: Optional[Callable[[], Optional[str]]] = None,
        presence_deadline_s: float = DEFAULT_PRESENCE_DEADLINE_S,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        exchange: Optional[
            Callable[[Dict[int, torch.Tensor], Dict[int, int]], Dict[int, torch.Tensor]]
        ] = None,
        pp_pool_view: Optional[KvPoolView] = None,
        tp_pool_view: Optional[KvPoolView] = None,
        live_slots_fn: Optional[Callable[[], torch.Tensor]] = None,
        ready_fn: Optional[Callable[[], bool]] = None,
        cutover_fn: Optional[Callable[[str], None]] = None,
        pre_cutover_fns: Sequence[Callable[[str], None]] = (),
        pre_write_fns: Sequence[Callable[[str], None]] = (),
        guards: Sequence[str] = (),
        clock: Callable[[], float] = time.perf_counter,
        # #631: injected so the spin can be driven deterministically in
        # tests. The spin blocks on no channel; this only paces it.
        sleep: Callable[[float], None] = time.sleep,
        presence_poll_interval_s: float = DEFAULT_PRESENCE_POLL_INTERVAL_S,
    ):
        if n_ranks < 2:
            raise KvReshardError(
                f"a phase flip needs a multi-rank group, got n_ranks={n_ranks}"
            )
        if not (0 <= int(rank) < n_ranks):
            raise KvReshardError(f"rank {rank} out of range for {n_ranks} ranks")
        if consensus_interval < 1:
            raise ValueError(
                f"consensus_interval must be >= 1, got {consensus_interval}"
            )
        if collective_min is None or exchange is None:
            raise KvReshardError(
                "a phase flip needs both a consensus channel (collective_min) "
                "and a pairwise byte channel (exchange); running without them "
                "would turn the first honest divergence into a hang instead "
                "of a loud error."
            )
        missing = [
            name
            for fn, name in (
                (pp_pool_view, "pp_pool_view"),
                (tp_pool_view, "tp_pool_view"),
                (live_slots_fn, "live_slots_fn"),
                (ready_fn, "ready_fn"),
                (cutover_fn, "cutover_fn"),
            )
            if fn is None
        ]
        if missing:
            raise KvReshardError(f"PhaseFlipRuntime needs {', '.join(missing)}")
        if boot_phase not in (PHASE_PP, PHASE_TP):
            raise KvReshardError(f"unknown boot phase {boot_phase!r}")

        self._n = int(n_ranks)
        self._rank = int(rank)
        self._map = validate_layer_map(layer_map, n_layers)
        self._n_layers = int(n_layers)
        self._vec = tuple(int(x) for x in tp_vector)
        if len(self._map) != self._n or len(self._vec) != self._n:
            raise KvReshardError(
                f"layer map has {len(self._map)} stages and the vector "
                f"{self._vec} has {len(self._vec)} entries, but the group "
                f"has {self._n} ranks -- the flip reuses the SAME ranks"
            )
        my_layers = self._map[self._rank]
        if pp_pool_view.num_layers != len(my_layers):
            raise KvReshardError(
                f"PP pool view has {pp_pool_view.num_layers} layers but "
                f"stage {self._rank} owns {len(my_layers)} "
                f"({my_layers}); the view must cover exactly this stage's "
                f"ordinals, ascending"
            )
        if tp_pool_view.num_layers != self._n_layers:
            raise KvReshardError(
                f"TP pool view has {tp_pool_view.num_layers} layers but the "
                f"model has {self._n_layers} full-attention layers; the TP "
                f"layout holds every ordinal on every rank"
            )
        self._fp = _config_fingerprint(self._map, self._vec)
        self._phase = boot_phase
        self._interval = int(consensus_interval)
        self._collective_min = collective_min
        # #631 option 2(b): the pollable entry gate.
        self._presence = presence
        self._pump_fn = pump_fn
        self._drain_fn = drain_fn
        self._owes_send_fn = owes_send_fn
        self._service_fn = service_fn
        self._channels_empty_fn = channels_empty_fn
        #: Diagnostics: how often presence was withheld because a channel
        #: was not yet empty, and how often the entry check actually
        #: caught a non-empty channel at the gate (which should be never).
        self.presence_withheld_channels = 0
        self.entry_channel_violations = 0
        self._last_withhold_log = None
        self._last_not_ready_log = None
        #: #631 J: read-only handle for the pool census. Set by the
        #: builder; absent in unit stubs, where the census is a no-op.
        self._census_scheduler = None
        self._presence_deadline_s = float(presence_deadline_s)
        self._presence_wait_started = None
        self._gate_open_epoch = None
        #: #631 THE ROUND STAMP. Counts the consensus reductions this arm
        #: has COMPLETED, and is the second half of the presence marker's
        #: identity. The ranks agree on it without exchanging it: a
        #: completed reduction is a synchronisation point they all leave
        #: together, so they all enter the next one carrying the same
        #: count. Never a local loop counter -- those diverge under
        #: event_loop_pp, which is the very reason the gate exists.
        self._entry_round = 0
        self._sleep = sleep
        self._presence_poll_interval_s = float(presence_poll_interval_s)
        #: The (epoch, round) whose pre-entry wait is currently being
        #: timed. The bound is PER ROUND: a fresh round is a fresh
        #: question and gets its own budget.
        self._presence_wait_stamp = None
        self.presence_timeouts = 0
        #: Diagnostics: rounds in which presence was WITHHELD because this
        #: rank still owed a chain send (clause (i)). A non-zero count on a
        #: healthy boot is normal -- it is the flush being waited out.
        self.presence_withheld_rounds = 0
        self._join_deadline_s = DEFAULT_JOIN_DEADLINE_S
        self.join_deadline_aborts = 0
        self._exchange = exchange
        self._pp = pp_pool_view
        self._tp = tp_pool_view
        self._live_slots_fn = live_slots_fn
        self._ready_fn = ready_fn
        self._cutover_fn = cutover_fn
        self._pre_cutover_fns = tuple(pre_cutover_fns)
        # #631: run at the read/write seam, where the source pool is fully
        # drained and the destination not yet touched -- the only safe
        # instant to move physical backing between the two layouts.
        self._pre_write_fns = tuple(pre_write_fns)
        self.blocking_guards = tuple(guards)
        self._clock = clock

        self._round = 0
        self._epoch = 0
        self._pending: Optional[str] = None
        self._last_hold_reason: Optional[str] = None
        self.desync_checks = 0
        self.completed = 0
        self.last_stats: Optional[dict] = None
        #: Wall-clock bound on the parked wait; see DEFAULT_PARK_DEADLINE_S.
        self._park_deadline_s = float(park_deadline_s)
        #: Clock reading of the moment this rank armed, or None when idle.
        self._armed_at: Optional[float] = None
        #: #746: ``(req_rows, req_max)`` measured by ``arm()`` at the arm
        #: instant -- the exact extent this flip will pack -- or None when no
        #: flip is armed or the arm-time measurement failed. Cleared at EVERY
        #: exit (commit and all abandon paths): a snapshot that outlives its
        #: flip pins the rung permanently, the M5 failure mode #744's
        #: mutation matrix refuses. Read through ``parked_extent()``, never
        #: directly.
        self._parked_extent: Optional[Tuple[int, int]] = None
        #: Flips abandoned because the park deadline expired. A counter, so
        #: "this never happens in practice" stops being an assumption.
        self.park_deadline_aborts = 0
        #: Flips abandoned because the live set did not fit the target
        #: pool. Same reason for counting it.
        self.fit_aborts = 0
        #: Flips abandoned because the ranks did not agree on the WIRE FRAME
        #: (register C22). Counted separately from every other abandon: a
        #: frame divergence is a broken replication premise, not a capacity
        #: verdict, and the two want opposite responses from an operator.
        self.frame_aborts = 0
        #: Rounds in which the KV cap agreement had to move THIS rank. Counted
        #: apart from the aborts: a levelling is the fix working, an abort is
        #: the ballot catching what the levelling could not reach.
        self.corridor_cap_levelled = 0
        #: #656 C22-d. Rounds the rung's ballot found the ranks enumerating
        #: DIFFERENT live slot sets -- the failure that wedged boot_m3 after
        #: 194 cutovers. Split three ways on purpose, because "the mechanism
        #: fired" and "the mechanism worked" are different measurements and
        #: the corpus has shipped that confusion before: ``divergences`` is
        #: how often it was needed, ``agreements`` how often the union
        #: repaired it, ``refusals`` how often the union reached past the
        #: group's backing and the round had to abandon instead.
        self.slot_set_divergences = 0
        self.slot_set_agreements = 0
        self.slot_set_refusals = 0
        #: The set this rank last put on the ballot, and its digest. Since the
        #: agreement above, that is not necessarily what ``_live_slots_fn``
        #: returned, and the difference is the whole point.
        self.last_framed_slots = None
        self.last_framed_slots_digest = None
        self.last_framed_digest = None
        #: Flips abandoned by the corridor gate (#656 item 15a) because no
        #: provider could fund the staging without breaking the floor.
        #: Distinct from staging_aborts: that one says "there is not enough
        #: room", this one says "there is not enough room AND nothing left
        #: to spill", which is the end of the ladder and not a transient.
        self.corridor_aborts = 0
        #: Consecutive pp->tp refusals. Reset by any other direction. See
        #: _corridor_gate: this direction's refusal is a decode deadlock,
        #: not a transient, and must be named as one.
        self._corridor_pp_refusals = 0
        #: Seams the corridor gate FUNDED by spilling first. The number that
        #: proves the gate is doing work rather than merely passing: a run
        #: with zero reclaims has not exercised item 15a at all.
        self.corridor_reclaims = 0
        #: #656 item 12, device half: seams at which the group agreed a KV
        #: backing target and this rank returned bytes for it, and the total
        #: those returned. Booked separately from ``corridor_reclaims``
        #: because this relief happens BEFORE the gate rather than inside its
        #: ladder -- a run where the gate never armed can still show KV
        #: relief, and reading one number for the other would hide that.
        self.corridor_kv_relief_count = 0
        self.corridor_kv_relief_bytes = 0
        #: #656 register C20. Seams DELAYED because the rank could satisfy the
        #: corridor law but not the designed seam-entry margin, and seams
        #: entered on the law alone after the delay budget was spent. Both are
        #: booked apart from ``corridor_aborts``: a margin delay is the flip
        #: WAITING for headroom that the paired-trough measurement says comes
        #: back, while an abort is the ladder having nothing left. Reading one
        #: for the other would make a healthy wait look like the 411-abandon
        #: decode wedge. A run whose yields dominate its delays is a run whose
        #: margin is not fundable on this configuration -- say so, do not
        #: quietly widen the margin.
        self.seam_margin_delays = 0
        self.seam_margin_yields = 0
        #: Yields WITHHELD because this rank's measured draw predicted a
        #: sub-law trough. Counted apart from the yields it replaces: one is
        #: the gate giving up its margin, the other is the gate refusing to.
        self.seam_yields_withheld = 0
        #: #656: seam entries whose own arithmetic, priced on the MEASURED
        #: in-cutover draw rather than on the staging reservation, predicted a
        #: trough below the corridor law. A counter and not only a log line
        #: because "never fired" and "fired and was right" have to be tellable
        #: apart in a bench row.
        self.seam_draw_predicted_breaches = 0
        #: Consecutive ABANDONED ATTEMPTS per direction -- the currency the
        #: C20 delay budget is spent in. Booked by ``note_seam_verdict`` from
        #: the REDUCED fit verdict, which is identical on every rank, so the
        #: budget means the same thing group-wide without a collective of its
        #: own.
        #:
        #: It is not "this rank was short N times". That version could not
        #: bound anything: the group abandons if ANY rank objects, so three
        #: ranks taking turns being short refund each other's budgets and the
        #: flip is delayed forever -- the decode wedge, entered through the
        #: mechanism meant to prevent it.
        #:
        #: Per direction, because the legs are not symmetric: delaying tp->pp
        #: defers prefill and is safe, delaying pp->tp defers decode and is
        #: the wedge. One shared counter would let the safe leg spend the
        #: dangerous leg's budget.
        self._seam_abandons_in_a_row = {PP_TO_TP: 0, TP_TO_PP: 0}
        #: #485: monotonic count of ARM REQUESTS this rank has seen, and the
        #: arm number at which each direction may next spend a full seam
        #: entry. Both are the backoff's clock, and the clock is ARMS rather
        #: than seconds or rounds ON PURPOSE: an arm reaches every rank as one
        #: broadcast ``PhaseFlipReqInput``, so all three ranks count the same
        #: sequence, while a wall clock is rank-local and a round count is not
        #: uniform across ranks within one arm. Group-uniform inputs, group-
        #: uniform decision, no collective added.
        self._arm_seq = 0
        self._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
        #: How many arm requests this rank declined cheaply for the backoff,
        #: per direction. Reported so a damped retry is visible as damping
        #: rather than as silence.
        self.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
        #: #656: the largest driver-visible in-cutover draw this rank has
        #: actually MEASURED, per direction, taken from the seam census.
        #:
        #: The seam entry law check priced itself on ``staging_bytes`` for
        #: three shifts. The staging is what the seam RESERVES; the census
        #: says the cutover DRAWS materially more, because the backing
        #: restore walk asks the driver for RAW pages while torch sits on its
        #: cache. Measured 2026-08-12 on PP1: staged 1625 MiB, drew 2066 MiB.
        #: Pricing the law on the smaller of the two is what let a yield
        #: enter a cutover whose own arithmetic predicted a breach.
        self._seam_draw_max = {PP_TO_TP: 0, TP_TO_PP: 0}
        #: Set once the seam-entry margin has announced itself, so the
        #: liveness line is one per process rather than one per flip.
        self._seam_margin_announced = False
        #: Flips abandoned because the STAGING buffers would not fit in
        #: free VRAM above the reserve. Counted separately from fit_aborts:
        #: "the target pool has no row for this slot" and "there is no room
        #: to stage the bytes on the way there" are different conditions
        #: with different answers, and a single counter would hide which
        #: one a boot is hitting.
        self.staging_aborts = 0
        self._staging_reserve_bytes = int(staging_reserve_bytes)
        self._mem_probe = mem_probe
        #: #631 2.1b SEAM ORDER. Restore the destination wave's backing
        #: before releasing the source wave's (the default), or the other
        #: way round (the pre-2.1b behaviour). Set
        #: ``SGLANG_FLIP_SEAM_RESTORE_FIRST=0`` to get the old order back
        #: WITHOUT also changing the wave count, which is what makes the
        #: reorder a one-variable A/B and, if metal disagrees with the
        #: desk arithmetic, a rollback that needs no code change.
        #:
        #: Aliased pools ignore this and always release first -- there the
        #: order is a correctness bound, not a tuning knob (``_flip_waves``
        #: and the seam block in ``_execute`` both say why).
        self._seam_restore_first = os.environ.get(
            "SGLANG_FLIP_SEAM_RESTORE_FIRST", "1"
        ).strip() not in ("0", "false", "no")
        #: #631 2.1 STREAMED SEAM. Row blocks per wave: 1 = commit a whole
        #: layer at a time (the 2.1b behaviour), >1 = restore/write/release
        #: one row block at a time, which shrinks the backing transient
        #: toward the arena's chunk floor.
        #:
        #: DEFAULT 16, MEASURED. Same boot, same direction, pool 500000
        #: (bench 2j): the binding 3080s' staging reservation falls
        #: 488.7 -> 305.6 -> 276.5 MiB at B = 1, 4, 16, and B=32 returns
        #: EXACTLY the B=16 numbers because the 16 MiB commit chunk's floor
        #: binds there -- while costing 8% more flip latency for it. So 16
        #: is the last block count that buys anything and the largest that
        #: costs nothing. Flip latency at 16 is +4% against the floor arm,
        #: inside the spread between ranks.
        #:
        #: LIVE SINCE #688: the commit chunk now defaults to 8 MiB, so these
        #: blocks are reached in a default boot. Priced under load at
        #: 216-377 MiB per rank saved (memory_pool._alloc_post_capture_buffers
        #: carries the table). Formerly inert:
        #: ``_effective_row_blocks`` mirrors ``_execute``'s gate and returns
        #: 1 when the arena cannot do span ops, so a default boot is
        #: unchanged. Pair this with ``SGLANG_FLIP_SEAM_CHUNK_MIB=16``; that
        #: default should not move until a LOADED corridor run exists at
        #: this setting, because every number above was taken at 90 live
        #: slots and prices the seam's constant, not its behaviour under a
        #: full pool.
        _blocks_env = os.environ.get("SGLANG_FLIP_SEAM_ROW_BLOCKS", "").strip()
        self._seam_row_blocks = 16
        if _blocks_env:
            try:
                self._seam_row_blocks = max(1, int(_blocks_env))
            except ValueError:
                raise KvReshardError(
                    f"SGLANG_FLIP_SEAM_ROW_BLOCKS={_blocks_env!r} is not an "
                    f"integer; it is the number of row blocks each seam wave "
                    f"is streamed in (unset or 1 = whole-layer commits)"
                )
        #: #631 SEAM WAVES. How many layer waves the move is split into.
        #: None = auto: one layer per wave under restore-first
        #: (``restore_first_wave_count``), else the most the layer map
        #: supports with every rank paying in (``default_wave_count``).
        #: 1 reproduces the pre-wave move byte for byte, which is what
        #: makes the wave count a one-variable A/B.
        #: Env override so the A/B needs no reboot flag of its own.
        _waves_env = os.environ.get("SGLANG_FLIP_SEAM_WAVES", "").strip()
        self._n_waves: Optional[int] = None
        if _waves_env:
            try:
                self._n_waves = max(1, int(_waves_env))
            except ValueError:
                raise KvReshardError(
                    f"SGLANG_FLIP_SEAM_WAVES={_waves_env!r} is not an "
                    f"integer; it is the number of layer waves the flip's "
                    f"seam is split into (unset = auto)"
                )

        logger.info(
            "%s armed at boot: rank %d/%d, phase %s, layer map %s, vector "
            "%s, consensus every %d rounds%s",
            LOG_PREFIX,
            self._rank,
            self._n,
            self._phase,
            self._map,
            self._vec,
            self._interval,
            (
                "; guards BLOCKING arming: " + ", ".join(self.blocking_guards)
                if self.blocking_guards
                else ""
            ),
        )

    # -- state ---------------------------------------------------------------
    @property
    def phase(self) -> str:
        return self._phase

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def pending(self) -> Optional[str]:
        return self._pending

    # -- arming (replicated callers) -----------------------------------------
    def _prearm_floor_relief(self, direction: str) -> Tuple[bool, str]:
        """SPILL for the arming floor before arming. NEVER refuse for it.

        "IT CAN ALWAYS BE SPILLED" IS TRUE BEFORE ARMING, AND ONLY BEFORE.
        The sizer charges the floor at boot, so a correctly planned instance
        never reaches this. What reaches it is the transient a sizer cannot
        see: a co-tenant, a capture peak, a rank that drifted. Letting that
        stop a flip is the same mistake as sizing without the floor -- it turns
        a recoverable dip into a phase the instance cannot enter.

        IT DOES NOT REFUSE, AND THE FIRST VERSION OF THIS METHOD DID. That
        version returned False when the ladder could not reach the floor, on
        the argument that "a rank that refuses simply does not arm, which the
        consensus round already handles". METAL SAYS OTHERWISE, and it is the
        second time the same error was made in one day, one layer apart.

        Measured on boot_slo_proof_r3.log: PP0 sat at 3130 MiB against its 1728
        MiB floor and ARMED; PP1 and PP2 sat at 1514 and 1982 against 1772 and
        2414 and REFUSED. The group split, and the armed rank parked at the
        entry -- "WITHHOLDING presence (8854 rounds so far) -- still owes a
        chain send" -- spinning for ever, no decode, all three cards at 0%.

        The existing blocking guards get away with being rank-local because
        they are config facts, identical on every rank in practice. A verdict
        keyed to this rank's FREE VRAM never is. So this rung keeps the half
        that is safe and rank-local -- SPENDING the ladder, which frees memory
        and cannot desync anything -- and leaves the REFUSING to the seam gate,
        which reduces its verdict across the group and already prints the
        numbers. One unanimous decision, not two.

        WHY THE RELIEF STILL BELONGS HERE rather than at that gate. The gate
        runs after the arm is committed, at the last point before the no-return
        region; by then the group has agreed to a flip and the staged fund must
        stay hard-resident, so freeing underneath it is the evictable-seam-fund
        mistake that produced the served-nothing class in #656 boots E/G. Here
        nothing is armed and nothing is staged, so a spill is free.

        BOUNDED, because an unbounded relief loop spills the instance flat.
        Attempts are counted per direction and reset the moment the floor is
        clear, so a rig that dips once pays one ladder.

        The (bool, str) shape is kept so the caller reads like every other
        precondition, but the bool is now always True: this rung reports, it
        does not decide.
        """
        if not _prearm_relief_enabled():
            return True, ""
        scheduler = getattr(self, "_census_scheduler", None)
        if scheduler is None:
            return True, ""
        try:
            from sglang.srt.managers.phase_flip_spill import get_corridor_guard

            guard = get_corridor_guard(scheduler)
            if guard is None:
                return True, ""
            # THE GUARD'S OWN FLOOR, NOT A SECOND DERIVATION OF IT. The gate
            # arms against ``floor_bytes``; re-deriving the same number here
            # would be two computations that must agree, which is how a rank
            # ends up spilling for a level nothing enforces. The sizer targets
            # this floor PLUS a load margin so the card starts above it; at
            # runtime the floor itself is the thing to reach.
            floor = int(guard.floor_bytes)
            free = int(guard.free_bytes())
        except Exception as e:  # pragma: no cover - defensive
            # AN UNREADABLE INSTRUMENT MAY NOT BLOCK A FLIP. The floor is a
            # precaution; failing to measure it is not evidence that it is
            # short, and refusing here would turn a probe failure into a
            # permanently stuck phase.
            logger.warning(
                "%s pre-arm floor could not be read (%s); arming proceeds",
                LOG_PREFIX,
                e,
            )
            return True, ""
        if not hasattr(self, "_prearm_relief_attempts"):
            self._prearm_relief_attempts = {PP_TO_TP: 0, TP_TO_PP: 0}
        if free >= floor:
            self._prearm_relief_attempts[direction] = 0
            return True, ""
        spent = int(self._prearm_relief_attempts.get(direction, 0))
        bound = _prearm_relief_attempts()
        if spent >= bound:
            # THE LADDER IS SPENT, AND THE ARM STILL PROCEEDS. The bound stops
            # a hot loop from spilling the instance flat; it is not a verdict.
            # The seam gate decides, unanimously, a few rounds from here.
            logger.warning(
                "%s %s arming floor still short: free %d MiB below %d MiB by "
                "%d MiB after %d bounded relief attempts. The arm PROCEEDS and "
                "the seam gate refuses if it must -- a rank-local refusal here "
                "split the group once already (r3: PP0 armed, its peers did "
                "not, and the armed rank parked at the entry for ever).",
                LOG_PREFIX,
                direction,
                free >> 20,
                floor >> 20,
                (floor - free) >> 20,
                spent,
            )
            return True, ""
        self._prearm_relief_attempts[direction] = spent + 1
        # ``ensure_headroom`` guarantees ``free_after - want >= law``, so the
        # ask that lands the card ON the arming floor is the floor MINUS the
        # law the guard already defends. Asking for the whole floor would
        # spill a second corridor's worth for nothing.
        want = max(0, floor - int(guard.law_floor_bytes))
        try:
            res = guard.ensure_headroom(
                want, reason=f"pre-arm arming floor for {direction}"
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "%s pre-arm relief raised (%s); arming proceeds on the "
                "seam gate's own ladder",
                LOG_PREFIX,
                e,
            )
            return True, ""
        reclaimed = int(getattr(res, "reclaimed", 0) or 0)
        providers = tuple(getattr(res, "used_providers", ()) or ())
        if getattr(res, "ok", False):
            logger.info(
                "%s pre-arm relief funded the %s arming floor: free %d -> "
                "%d MiB against a %d MiB floor, %d MiB reclaimed from %s. "
                "Spilled BEFORE the arm, so the staged fund stays resident "
                "once the flip begins.",
                LOG_PREFIX,
                direction,
                free >> 20,
                int(getattr(res, "free_after", free)) >> 20,
                floor >> 20,
                reclaimed >> 20,
                list(providers) or ["nothing"],
            )
            self._prearm_relief_attempts[direction] = 0
            return True, ""
        # SHORT, AND SAID SO WITH THE NUMBERS -- but not a refusal. The
        # operator asked for "how much was short and what the ladder freed",
        # and that is exactly what this line carries; what it must not carry is
        # a rank-local decision about whether the group flips.
        logger.warning(
            "%s %s arming floor NOT funded: free %d MiB is below %d MiB by "
            "%d MiB; the ladder freed %d MiB from %s (attempt %d of %d). The "
            "arm proceeds; the seam gate reduces the verdict.",
            LOG_PREFIX,
            direction,
            free >> 20,
            floor >> 20,
            (floor - free) >> 20,
            reclaimed >> 20,
            list(providers) or ["nothing"],
            spent + 1,
            bound,
        )
        return True, ""

    def arm(self, direction: str, source: str) -> Tuple[bool, str]:
        """Arm a flip. Replicated call; the consensus round commits it once
        every rank is armed AND ready. Returns (ok, msg)."""
        # THE BACKOFF CLOCK TICKS ON EVERY ARM REQUEST, including the ones
        # this function goes on to refuse. It has to: the sequence is only
        # group-uniform if every rank advances it for the same events, and a
        # refusal is an event every rank sees.
        self._arm_seq = getattr(self, "_arm_seq", 0) + 1
        if self.blocking_guards:
            # #662: THE THIRD DAMPER ON THE SAME PATH, and the last one.
            #
            # The policy hold and the seam's own abandon counter both learned
            # to stand down while the arming condition persists; this one then
            # blocked every entry anyway, so the seam was never re-priced.
            # Measured: arms climbed 22, 23, 24, 25 against "seam unfundable:
            # tp_to_pp abandoned 8 times consecutively" with 92k tokens
            # pending. Three independent counters guarding one decision is how
            # a fix keeps looking wrong after it is right.
            #
            # The abandon-cap guard is the only one that stands down: it is a
            # statement about affordability, and affordability is exactly what
            # a full pool changes. Every other blocking guard is a statement
            # about SAFETY (a half-built stack, a missing carrier) and keeps
            # refusing, because no amount of pending work makes those safe.
            standing = [
                g
                for g in self.blocking_guards
                if not g.startswith(SEAM_ABANDON_CAP_GUARD)
            ]
            if standing or not self._arming_condition_persists():
                msg = (
                    "phase flip refused (guards): "
                    f"{', '.join(standing or self.blocking_guards)}"
                )
                logger.warning("%s %s", LOG_PREFIX, msg)
                return False, msg
            if not self._storm_limiter_allows(direction):
                msg = (
                    f"phase flip {direction}: abandon-cap guard would stand "
                    "down, but the arm RATE limiter is holding this attempt. "
                    "Re-pricing is paced, never blocked -- the next attempt "
                    "runs on schedule."
                )
                logger.debug("%s %s", LOG_PREFIX, msg)
                return False, msg
            logger.info(
                "%s abandon-cap guard STOOD DOWN at arm %d: work is still "
                "waiting, so the seam is re-priced instead of refused unheard. "
                "Affordability is what a full pool changes; safety guards are "
                "unaffected and still refuse.",
                LOG_PREFIX,
                self._arm_seq,
            )
        if direction not in _DIR_ID:
            return False, f"unknown flip direction {direction!r}"
        # #485: DAMP A REFUSAL THAT CANNOT CHANGE. Declining here rather than
        # in ``_execute`` is deliberate -- nothing is armed, so no rank enters
        # the seam, no collective is reached, and the ranks cannot disagree
        # about whether this attempt happened. The alternative (an early
        # return inside ``_execute``) would have to be taken identically by
        # every rank on a path that already carries collectives, which is a
        # divergence waiting to be written.
        if not hasattr(self, "_seam_retry_at_arm"):
            self._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
            self.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
        retry_at = self._seam_retry_at_arm.get(direction, 0)
        # #662: THE SECOND TIMER. The policy's own backoff was replaced with
        # dwell-paced retry while the arming condition persists, and this
        # counter then vetoed the retry anyway -- so the KV rung was still
        # never asked. Measured on this rig: PP0 printed "the load still wants
        # this layout, so the next attempt is in 0.2s and it will ask the KV
        # rung again", and the very next arm was declined by "5 consecutive
        # group abandons, next entry at arm 38 (this is arm 26)".
        #
        # Two independent dampers on one path is how a fix lands half-wired.
        # While work is still waiting, this one stands down for the same
        # reason the other did: waiting frees no memory, and the pool being
        # full is what pays for the flip once the rung is reached.
        #
        # The counter is kept and still counts -- it is the right damper for
        # its actual case, an arming condition that has GONE AWAY, where the
        # bypass below is false and the skip applies exactly as before.
        if self._arm_seq < retry_at and self._arming_condition_persists():
            logger.info(
                "%s %s abandon backoff STOOD DOWN at arm %d (would have "
                "waited until %d): work is still waiting, so the attempt "
                "proceeds and the KV rung gets asked. Waiting frees no "
                "memory; the full pool is what funds the seam.",
                LOG_PREFIX,
                direction,
                self._arm_seq,
                retry_at,
            )
            retry_at = 0
        if self._arm_seq < retry_at:
            self.seam_backoff_skips[direction] = (
                self.seam_backoff_skips.get(direction, 0) + 1
            )
            book = getattr(self, "_seam_abandons_in_a_row", None) or {}
            msg = (
                f"flip {direction} backing off: {book.get(direction, 0)} "
                f"consecutive group abandons, next entry at arm "
                f"{retry_at} (this is arm {self._arm_seq})"
            )
            # DEBUG, not WARNING. The whole purpose of this branch is to stop
            # a hot loop from producing work -- including log work. The state
            # it represents is announced once, by the abandon that set it.
            logger.debug("%s %s", LOG_PREFIX, msg)
            return False, msg
        want = _DIR_OF_PHASE[self._phase]
        if direction != want:
            return False, (
                f"flip {direction} refused: current phase is {self._phase}, "
                f"the only legal transition is {want}"
            )
        if self._pending is not None and self._pending != direction:
            logger.warning(
                "%s re-arming %s -> %s (source %s)",
                LOG_PREFIX,
                self._pending,
                direction,
                source,
            )
        # #662-F4 / A0: SPILL for the arming floor here, where relief is still
        # free -- nothing is armed, no rank has entered the seam, and the
        # staged fund does not exist yet to be pulled out from under.
        #
        # ITS VERDICT IS DELIBERATELY DISCARDED. This rung is rank-local, and a
        # rank-local refusal splits the arm: on r3 PP0 cleared its floor and
        # armed while its peers did not, and the armed rank parked at the entry
        # for ever. Refusing is the seam gate's job, because the gate reduces
        # its verdict across the group. Ignoring the return value here is what
        # makes that structural rather than a promise in a docstring.
        self._prearm_floor_relief(direction)
        self._pending = direction
        # A fresh arm starts a fresh round sequence. The epoch already
        # distinguishes this arm from any earlier one, so the round
        # simply restarts at 0 rather than having to be globally unique.
        self._entry_round = 0
        self._presence_wait_stamp = None
        # The park clock starts at ARMING, not at the first unparked round:
        # the deadline bounds how long the requests are held, and they are
        # held from the moment this rank starts withholding work.
        self._armed_at = self._clock()
        # #746: measure the parked extent NOW. The requests quiesce after
        # arming, so this is the last instant the resident set is both
        # enumerable and final -- "the rows this flip will pack" is fixed
        # here. The KV rung reads this snapshot instead of remembering the
        # last enumeration that happened to see requests (which answered
        # UNKNOWN for a flip that armed before any enumeration ran, and
        # answered stale for a resident set that changed since). Taken ONCE
        # per arm, deliberately not re-taken at the park-clock re-base: by
        # then the requests are quiescing and an enumeration reports the
        # req_rows=0 blindness #744 exists to defeat.
        self._parked_extent = self._snapshot_parked_extent()
        # #631 J: census AT ARM. The pre/post-cutover pair proved the move
        # and the cutover innocent (identical unaccounted set on both
        # sides), and a no-flip control boot stayed clean, so the page goes
        # missing somewhere in the ARMED window. This bracket closes it.
        self._pool_census("at-arm", direction)
        msg = (
            f"phase flip armed: {direction} (source {source}); commits at "
            f"the next consensus boundary where every rank is quiescent, or "
            f"is abandoned after {self._park_deadline_s:g}s parked"
        )
        logger.warning("%s %s", LOG_PREFIX, msg)
        return True, msg

    # -- the per-round hook ---------------------------------------------------
    def on_round(self, require_armed_and_parked: bool = False) -> Optional[dict]:
        """One scheduler round; see KvReshardRuntime.on_round. Returns move
        stats when a flip executed this round, else ``None``.

        ``require_armed_and_parked`` is the PP-phase entry gate (measured
        wedges 2026-08-08, boots 9 and 10): under event_loop_pp the local
        round counters of the ranks diverge in ABSOLUTE value (pipeline
        fill, conditional per-slot ops), so ANY blocking reduction entered
        at a local cadence can pair with a peer blocked in a pipeline recv
        whose satisfying send sits behind this rank's reduction -- moving
        the hook inside the iteration only moved the wedge. With the gate,
        an UNARMED rank performs NO collective at all (there is nothing to
        agree on; arming state arrives via the broadcast RPC on every
        rank), and an armed rank enters only once it is locally PARKED
        (ready_fn: drained microbatches, no partial chunk) -- a parked
        rank owes no pipeline send, so no recv/reduction cycle can close.
        Peers converge on their own arm+drain, MIN-skew is legal, and the
        liveness bound turns a lost peer into a loud error. A flip under
        continuous load needs the posted-async two-phase consensus -- a
        named follow-up, not this gate.

        The wait is BOUNDED (see DEFAULT_PARK_DEADLINE_S): a rank armed
        past the deadline without parking joins the reduction anyway
        carrying ``expired``, and every participating rank abandons the
        flip on the reduced maximum. Abandoning the flip is the whole
        point -- the parked requests are never abandoned."""
        armed = 1 if self._pending is not None else 0
        # #631 DEFECT Q, SECOND INSTANCE -- and this one is fatal in the TP
        # phase. ``_round`` is a RANK-LOCAL counter, and ``_round %
        # _interval`` below gates ENTRY TO A BLOCKING COLLECTIVE. That is
        # only safe while the ranks' counts stay congruent, and nothing
        # makes them congruent except the loop being paced in lockstep.
        #
        # AN ARMED WINDOW IS EXACTLY WHERE THAT PACING STOPS. An armed rank
        # admits nothing and launches nothing, so its pass loop free-runs at
        # about 8 kHz and calls this hook every iteration: measured
        # 2026-08-09, 37371 / 28677 / 32344 calls in ONE 5 s window. The
        # ranks come out incongruent mod _interval, their periodic entries
        # never coincide again, and the FIRST periodic consensus after the
        # cutover deadlocks -- rank 0 inside the reduction, its peers inside
        # the broadcast recv that rank 0 owes them. Measured 08:09:39Z:
        # "barlink collective 'phase_flip.consensus' made no progress for
        # 120s", PP0 in event_loop_normal -> get_next_batch_to_run ->
        # _phase_flip_on_round, peers never arriving, then SIGQUIT.
        #
        # SO THE CADENCE COUNTS ONLY THE ROUNDS THE CADENCE GATES. The
        # per-pass hook of event_loop_pp is exactly the caller that passes
        # require_armed_and_parked=True, and its entry is decided by the
        # parked predicate and the presence gate -- never by this counter.
        # Its increments therefore buy nothing and cost the congruence of
        # every periodic round after them. The periodic caller
        # (event_loop_normal, require_armed_and_parked=False) is the one
        # this counter exists for, and that loop IS paced: rank 0 broadcasts
        # the request list every iteration, so those rounds stay in step.
        #
        # NOT "count unarmed rounds only", which was the first cut and is
        # wrong: an ARMED rank on the periodic path must still reach the
        # reduction, or a flip armed in the TP phase can never commit.
        # Freezing the counter while armed left ranks whose residue was not
        # already zero unable to enter at all, and
        # TestMoveCorrectness caught it -- the move never ran and the
        # destination pool kept its old bytes.
        #
        # This is the same warning the ``_entry_round`` comment below already
        # makes ("incrementing anywhere else -- a local loop counter, a clock
        # -- would reintroduce the absolute divergence between ranks"),
        # applied to the variable it was actually true of.
        if not require_armed_and_parked:
            self._round += 1
        ready = 1 if (armed and self._ready_fn()) else 0
        expired = 1 if self._park_expired(armed, ready) else 0
        # #631 QUIESCENT-ANNOUNCE. An armed rank that is NOT yet quiescent
        # must go back around the pass loop -- that is how it drains -- and
        # must NOT announce on the way. Announcing before quiescence is
        # what made the flag mean "I was at the entry once": the rank
        # published presence, returned to the loop, and met its top-of-pass
        # commit before it could come back and ENTER. The last announcer
        # then entered the reduction and every earlier one blocked behind
        # it (measured 23:39Z, three stacks).
        #
        # An EXPIRED rank is exempt: it has been armed past the park
        # deadline without ever draining, and it must be allowed to reach
        # the reduction to carry that fact into a group-agreed
        # abandonment. It owes no fresh work by then, having withheld
        # admissions for the whole deadline.
        # Scoped to a wired presence channel ON PURPOSE. This early
        # return exists solely to stop a rank ANNOUNCING before it is
        # quiescent; with no presence channel there is no announce, so
        # applying it would change the readiness-skew behaviour of the
        # plain consensus path (which holds uniformly inside the
        # reduction) for no gain. Caught by
        # TestConsensusDiscipline::test_readiness_skew_holds_uniformly.
        if armed and self._presence is not None and not ready and not expired:
            # SAY WHAT IS HOLDING THIS RANK, periodically. Without it the
            # only evidence is "ready=0" in an abandonment 30 s later, and
            # defect I had to be read off three py-spy stacks instead.
            self._log_not_ready()
            return None
        # #631 THE ENTRY GATE, evaluated AFTER park expiry now that
        # announcing requires quiescence. It SPINS: once this rank has
        # announced it does not return to the pass loop at all, because
        # that interval is exactly what kills it. The spin blocks on no
        # channel -- it reads flags and sleeps -- and is bounded per round,
        # so a group that never assembles abandons loudly instead of
        # hanging.
        if armed:
            gate = self._spin_for_group_presence()
            if gate is not True:
                return gate  # a pre-entry abandonment, or None if disarmed
        # The PP-phase entry gate, widened by the deadline: an armed rank
        # enters once it is PARKED, or -- if it has been armed past the
        # deadline without ever parking -- to carry that fact into the
        # consensus. Entering unparked is what makes the abandonment
        # GROUP-AGREED: the peers are already blocked in this reduction
        # waiting for exactly this rank, so the flag reaches them, every
        # rank abandons the same flip in the same round, and nobody is left
        # armed against a disarmed peer. It is safe here because an armed
        # rank has been withholding new work for the whole deadline, so it
        # owes no fresh pipeline send.
        if require_armed_and_parked and not (armed and (ready or expired)):
            return None
        if not require_armed_and_parked and self._round % self._interval != 0:
            return None
        dir_id = _DIR_ID[self._pending] if self._pending is not None else 0
        payload = _encode(
            [armed, ready, expired, self._epoch, dir_id, self._fp, *self._vec]
        )
        self.desync_checks += 1
        # #631(c) WITHDRAWN -- measured fatal, kept as a warning.
        #
        # Bounding this join and abandoning from inside CANNOT work on a
        # gloo collective. Measured 2026-08-08: the 45 s bound fired
        # (CollectiveTimeoutError), and the moment this rank walked away
        # its peers saw "gloo/transport/tcp/pair.cc:547 Connection closed
        # by peer" and every rank died with "Fatal Python error: Aborted".
        # A rank that has ENTERED an all_reduce owes that all_reduce; the
        # group has no way to un-enter it. So a wedge here cannot be
        # broken from inside the collective -- any bound has to be
        # applied BEFORE entry (do not enter unless the peers are known
        # to be joining), or the reduction has to become a non-blocking
        # poll that a rank re-enters, which is a different design.
        reduced = self._collective_min(payload)
        # THE ROUND ADVANCES HERE, and only here. Reaching this line
        # means every participant completed the SAME reduction, so this
        # is the one instant at which the ranks provably agree -- which
        # is exactly what makes the count usable as a shared stamp
        # without ever being exchanged. Incrementing anywhere else (a
        # local loop counter, a clock) would reintroduce the absolute
        # divergence between ranks that the gate exists to tolerate.
        self._entry_round += 1
        if len(reduced) != len(payload):
            raise KvReshardError(
                f"consensus channel returned {len(reduced)} values for a "
                f"{len(payload)}-value payload; the channel contract is "
                f"element-wise MIN of the packed proposal."
            )
        fields = [
            "armed",
            "ready",
            "expired",
            "epoch",
            "direction",
            "config_fp",
        ] + [f"vector[{i}]" for i in range(self._n)]
        lo = {f: reduced[2 * i] for i, f in enumerate(fields)}
        hi = {f: -reduced[2 * i + 1] for i, f in enumerate(fields)}

        # Equality family: epoch + config fingerprint + vector ALWAYS
        # (boot config); direction once every rank is armed.
        eq_checked = ["epoch", "config_fp"] + [f"vector[{i}]" for i in range(self._n)]
        if lo["armed"] == 1:
            eq_checked.append("direction")
        mismatches = [
            f"{f}: min={lo[f]} max={hi[f]}" for f in eq_checked if lo[f] != hi[f]
        ]
        if mismatches:
            raise KvReshardError(
                f"{LOG_PREFIX} DESYNC at round {self._round}: the ranks "
                f"disagree on the flip state ({'; '.join(mismatches)}; this "
                f"rank: armed={armed} pending={self._pending} "
                f"epoch={self._epoch} phase={self._phase}). A flip that "
                f"disagrees across ranks must fail loudly HERE, before any "
                f"rank moves a byte under the wrong layout."
            )
        # Park deadline, decided on the MAX: one rank out of time is enough
        # to abandon the flip, and every rank in this reduction reads the
        # same max, so the abandonment is unanimous by construction.
        # Checked before the armed/ready holds -- those are the states the
        # deadline exists to stop waiting in.
        if hi["expired"] == 1:
            return self._abandon_parked_flip(ready)

        if lo["armed"] == 0:
            if hi["armed"] == 1:
                self._hold("waiting for every rank to arm (delivery skew)")
            return None
        if lo["ready"] == 0:
            self._hold(
                f"armed ({self._pending}), waiting for a group-wide "
                f"quiescent boundary (this rank ready={ready})"
            )
            return None
        self._last_hold_reason = None
        return self._execute()

    def _pool_census(self, when: str, direction: str) -> None:
        """#631 defect J: the allocator's own view, straddling the cutover.

        Reproduces the invariant checker's leak arithmetic
        (expected - free - cached) at a point where the flip can still be
        reasoned about, because by the time on_idle raises it the stacks
        have already been swapped and the evidence is one pass stale.

        Read-only and best effort: a census must never be able to affect a
        flip it is only watching.
        """
        try:
            scheduler = self._census_scheduler
            if scheduler is None:
                return
            alloc = getattr(scheduler, "token_to_kv_pool_allocator", None)
            tree = getattr(scheduler, "tree_cache", None)
            if alloc is None or tree is None:
                return
            free = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())
            cached = set(tree.all_values_flatten().tolist())
            size = int(alloc.size)
            leaked = set(range(1, size + 1)) - free - cached
            reqs = _live_reqs(scheduler)
            # SLOT SCOPE MATTERS AND IS EASY TO MISREAD. Under
            # event_loop_pp, scheduler.running_batch / last_batch are
            # rebound to running_mbs[mb_id] / last_mbs[mb_id] at the TOP of
            # every slot iteration, so they describe ONE microbatch slot --
            # the one whose iteration is running -- not the rank's resident
            # set. A census that reports only those can read 0 while
            # requests sit in other slots, which is exactly how "the
            # request finished" got inferred from a slot that was merely
            # empty. Report both scopes so the two can never be confused.
            resident = 0
            slots_with_reqs = []
            for i, mb in enumerate(getattr(scheduler, "running_mbs", []) or []):
                n = len(getattr(mb, "reqs", []) or [])
                if n:
                    resident += n
                    slots_with_reqs.append(i)
            logger.warning(
                "%s POOL CENSUS %s %s: size=%d free=%d cached=%d "
                "available=%s cur_slot_reqs=%d resident_reqs=%d "
                "resident_slots=%s unaccounted=%d %s",
                LOG_PREFIX,
                when,
                direction,
                size,
                len(free),
                len(cached),
                getattr(alloc, "available_size", lambda: "?")(),
                len(reqs),
                resident,
                slots_with_reqs,
                len(leaked),
                sorted(leaked)[:12],
            )
        except Exception as exc:  # noqa: BLE001 - a census never breaks a flip
            logger.warning("%s pool census (%s) failed: %s", LOG_PREFIX, when, exc)

    def _log_not_ready(self) -> None:
        """Report what is holding this rank out of quiescence.

        Throttled to a quarter of the park deadline: a rank that drains in
        a pass or two stays silent, and one that never drains is on the
        record BEFORE the abandonment that names it.
        """
        why = None
        probe = getattr(self._ready_fn, "why_not", None)
        if probe is not None:
            try:
                why = probe()
            except Exception as exc:  # noqa: BLE001
                why = f"(quiescence probe failed: {exc})"
        if not why:
            return
        now = self._clock()
        if self._last_not_ready_log is not None and (
            now - self._last_not_ready_log
        ) < max(self._park_deadline_s / 4.0, 1.0):
            return
        self._last_not_ready_log = now
        logger.warning(
            "%s armed (%s) but NOT QUIESCENT: %s. This rank is holding the "
            "flip; it has not announced and is not at the entry.",
            LOG_PREFIX,
            self._pending,
            why,
        )

    def _spin_for_group_presence(self):
        """#631: announce, then SPIN here until the group assembles.

        THE POINT, and the whole reason this is a loop rather than one poll
        per pass: a rank that announces and then returns to the pass loop
        meets its top-of-pass commit before it can come back and ENTER, and
        that commit blocks behind whichever rank has already entered. The
        announce-to-entry interval must contain NO blocking channel
        operation, so the rank simply does not leave.

        WHY LEAVING THE LOOP IS SAFE HERE, and only here: this is reached
        only once ready_fn holds (or the park deadline has expired), i.e.
        the rank is drained -- no in-flight microbatches, no admissions, no
        owed payload. It therefore owes its peers neither hidden states nor
        chain data, and the only per-pass message it stops producing is the
        empty keep-alive forward. Peers that need nothing are either
        quiescent and spinning here too, or not yet quiescent -- and that
        second case is a BOUNDED RETRY, not a wedge: a mid-drain rank that
        stalls on its recv is released when the spinners' per-round bound
        expires, they abandon loudly, return to the loop and resume
        forwarding, it drains, and a later epoch retries with everyone
        genuinely quiescent. At true idle every rank is quiescent at once,
        so the gate opens on live evidence and the flip commits in the
        first epoch.

        Delegates each iteration to _await_group_presence, which stays the
        single-shot primitive: same announce, same round-scoped read, same
        pre-entry bound, same abandonment. Nothing new is invented here --
        this only stops the rank from going away between iterations.
        """
        while True:
            gate = self._await_group_presence()
            if gate is not True:
                if gate is not None:
                    return gate  # pre-entry abandonment: loud, nothing entered
                if self._pending is None:
                    # Disarmed underneath us (abandonment elsewhere).
                    return None
                # Not assembled yet. Sleep briefly and ask again WITHOUT
                # touching any channel. The pre-entry deadline inside
                # _await_group_presence is what ends this loop if the group
                # never arrives.
                self._sleep(self._presence_poll_interval_s)
                continue
            return True

    def _snapshot_parked_extent(self) -> Optional[Tuple[int, int]]:
        """#746: ``(req_rows, req_max)`` of the resident set, measured NOW.

        Runs the flip's own live-slot enumeration and reads its request half
        from the split side channel it populates. Called by ``arm()`` at the
        arm instant; the result is the exact extent this flip will pack,
        because parking begins at arming and quiescence only removes requests
        from the enumerable structures, never adds them.

        ``None`` means the measurement FAILED, and the #744 axiom carries
        over unchanged: UNKNOWN IS NOT EMPTY. A None snapshot leaves the KV
        rung's probe answering ``(-1, -1)`` while this flip is armed, which
        #748's ``_parked_ceiling`` turns into its wholesale refusal -- the
        conservative reading, now confined to a failed measurement instead of
        covering every flip that armed before an enumeration ran.
        """
        fn = getattr(self, "_live_slots_fn", None)
        if fn is None:
            return None
        try:
            fn()
            split = getattr(fn, "last_split", None)
            if not split:
                return None
            return (int(split["req_rows"]), int(split["req_max"]))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "%s arm-time parked-extent snapshot failed (%s); the KV rung "
                "will treat this flip's extent as UNKNOWN and refuse to "
                "shrink for its duration",
                LOG_PREFIX,
                e,
            )
            return None

    def parked_extent(self) -> Optional[Tuple[int, int]]:
        """#746: the extent snapshot of the CURRENTLY armed flip, else None.

        Gated on ``_pending`` -- the one authority for arming, same as
        ``is_armed`` -- so a snapshot can never be read once its flip has
        exited, even if an exit path forgot to clear it. That is the second
        of two defences against the M5 failure mode (a stale value pinning
        the rung permanently); the first is that every exit path clears the
        attribute.
        """
        if self._pending is None:
            return None
        return self._parked_extent

    def is_armed(self) -> bool:
        """#631: is a flip armed on this rank right now?

        The scheduler's intake asks this every pass to decide whether it
        may admit new work and whether it may block on the chain. It is a
        read of ``_pending``, the one authority for arming -- deliberately
        not a mirrored flag, which would be a second state to keep in
        sync.
        """
        return self._pending is not None

    def _park_expired(self, armed: int, ready: int) -> bool:
        """Has this rank been armed-but-unparked past the deadline?

        Wall clock, not a round count: rounds are what the PP loop makes
        incomparable across ranks in the first place, and the quantity the
        operator cares about is how long a request may be held. The reading
        is rank-local and does NOT need to be replicated -- one rank
        raising the flag is enough, because the DECISION to abandon is
        taken from the reduced maximum in on_round, which every
        participating rank reads identically.
        """
        if not armed or ready or self._park_deadline_s <= 0:
            return False
        if self._armed_at is None:
            return False
        return (self._clock() - self._armed_at) >= self._park_deadline_s

    def _await_group_presence(self):
        """#631 option 2(b): the non-blocking armed wait, and the gate.

        Returns True when every rank is at the entry and the caller may
        safely enter the blocking reduction; None to keep polling on later
        rounds; or the result of an abandonment when the pre-entry bound
        expires.

        NOTHING IN THIS LOOP BLOCKS, which is the whole point -- it is the
        construction that satisfies the design law (no rank blocks on any
        channel while a peer may be in a different one). Concretely, per
        iteration this rank only:
          * PUMPS its outstanding arm-forward -- progresses it
            non-blockingly, never a blocking commit. A blocking commit
            here is corpse B' (boot 13): rank 0 blocked in
            _pp_commit_comm_work while its peers sat in the hidden-states
            exchange, because "the peer is waiting for the arm" is simply
            not true -- it may be in another channel entirely,
          * DRAINS its incoming chain non-blockingly (clause (ii)),
            buffering what arrives. An armed rank that stops consuming
            makes its UPSTREAM block on the ordinary top-of-pass commit,
            upstream of the gate, where no gate can help it (boot 18),
          * announces its own presence for this epoch (a file create) --
            but ONLY once it owes no send (clause (i)), so the flag means
            "my chain is flushed", not merely "I am armed",
          * polls peers' flags (file existence).

        Once all flags are up, entering is safe by CONSTRUCTION rather
        than by argument: every rank that will participate is in this
        same loop -- not blocked elsewhere -- flags are monotone so every
        rank observes the same all-ready fact, and each rank's own chain
        send was pumped to completion before it announced.

        THE THREE CLAUSES ARE ONE MECHANISM, not three precautions. (i)
        alone is unsatisfiable: a rank cannot flush a forward to a peer
        that has stopped reading. (ii) alone leaves the flag a lie, which
        is what let the peers enter on a rank that was still blocked.
        Together they close boot 18: every announced rank owes nothing,
        and no rank can be prevented from announcing.

        The bound is PRE-ENTRY and therefore legal, unlike the withdrawn
        (c): abandoning a poll costs nothing, because nothing has been
        entered and no peer is owed anything. Abandoning an ENTERED
        all_reduce aborts the whole group, which is why that bound was
        withdrawn and pinned.
        """
        if self._presence is None:
            # No presence channel wired (unit tests, or a builder that
            # predates the gate): fall through to the old behaviour rather
            # than silently never flipping.
            return True

        epoch = self._epoch
        entry_round = self._entry_round
        # PER-ROUND PRE-ENTRY BOUND. A new round is a new question, so it
        # gets its own budget; carrying the previous round's elapsed time
        # forward would abandon a perfectly healthy later round for time
        # spent waiting on an earlier one.
        stamp = (epoch, entry_round)
        if self._presence_wait_stamp != stamp:
            self._presence_wait_stamp = stamp
            self._presence_wait_started = self._clock()
        if self._presence_wait_started is None:
            self._presence_wait_started = self._clock()

        # #631 G: THE SERVICE TURN, and the reason this loop is no longer a
        # starvation source. A spinning rank used to stop issuing its
        # per-pass chain forward, and its downstream reached the hook ONLY
        # by returning from the blocking recv that forward satisfied -- so
        # the first rank to quiesce blocked every rank behind it, every
        # epoch, identically. The answer is not to keep sending (an armed
        # rank has nothing to forward) but to make the downstream not NEED
        # the send: it services its channels here and reaches the hook by
        # its own poll. Blocking inside this call is bounded by transfer
        # time, never by peer scheduling, because a counter proved the
        # message exists before the receive was made.
        if self._service_fn is not None:
            try:
                self._service_fn()
            except Exception as exc:  # noqa: BLE001 - servicing is best effort
                logger.warning("%s service turn failed: %s", LOG_PREFIX, exc)

        if self._pump_fn is not None:
            # Progress our own arm forward WITHOUT blocking on it. This is
            # what actually delivers the arm to the next stage while we
            # wait, and it is the difference between this design and
            # corpse B'.
            try:
                self._pump_fn()
            except Exception as exc:  # noqa: BLE001 - pumping is best effort
                logger.warning("%s pump failed: %s", LOG_PREFIX, exc)

        # #631 CLAUSE (ii), and the boot-18 fix. An armed rank must keep
        # servicing EVERY channel obligation it has, or a peer blocks on
        # it. Pumping alone covers only what this rank SENDS; the other
        # half of the obligation is what it RECEIVES. Boot 18: rank 2
        # armed and stopped consuming the chain, so rank 1's ordinary
        # top-of-pass commit of the previous pass's forward blocked in
        # work.wait() -- a blocking point that PRECEDES the gate, which is
        # why the gate could never cover it. Rank 1 therefore never
        # announced, the gate never assembled, and rank 0 waited in the
        # reduction. Draining here is what keeps the upstream free to
        # reach its own announce.
        if self._drain_fn is not None:
            try:
                self._drain_fn()
            except Exception as exc:  # noqa: BLE001 - draining is best effort
                logger.warning("%s drain failed: %s", LOG_PREFIX, exc)

        # #631 CLAUSE (i). ANNOUNCE ONLY ONCE THIS RANK OWES NO SEND, so a
        # raised flag means "my chain is flushed", not merely "I am armed".
        # Announcing while a forward is still outstanding is what made the
        # boot-18 flag a lie: the rank announced, went back around the
        # pass, and blocked on the top-of-pass commit of that very send
        # before it could reach the reduction its flag had promised. The
        # peers, seeing a full quorum, entered and waited for a rank that
        # was blocked elsewhere.
        #
        # Withholding is safe: presence is monotone, so a later announce is
        # simply a later fact, and the pre-entry deadline still bounds the
        # wait. A rank that can never flush abandons LOUDLY instead of
        # dragging the group into a reduction it cannot join.
        #
        # WITHHOLDING MUST FALL THROUGH TO THE DEADLINE, never return
        # early. Returning from here skips the pre-entry bound below and
        # turns "wait until flushed" into a NEW unbounded wait -- the same
        # shape as the wedge this clause exists to remove. Caught by
        # test_can_fail_a_rank_that_never_flushes_abandons_instead_of_wedging
        # while building it.
        owes = False
        if self._owes_send_fn is not None:
            try:
                owes = bool(self._owes_send_fn())
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s owes-send probe failed: %s", LOG_PREFIX, exc)
                owes = False

        # #631 G, FLIP-COMMIT HYGIENE. Quiescent AND fully serviced implies
        # every channel is empty; a rank that is not there yet withholds
        # presence exactly as a rank that owes a send does. Withholding
        # rather than abandoning is what keeps this CONVERGENT: a message
        # still in flight is normally reaped by the next service turn, and
        # the pre-entry deadline below still bounds the wait, so a rank
        # that can never empty its channels abandons loudly instead of
        # dragging the group into a reduction it cannot join.
        unclean = None
        if not owes and self._channels_empty_fn is not None:
            try:
                unclean = self._channels_empty_fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s channel probe failed: %s", LOG_PREFIX, exc)
                unclean = None
        if unclean:
            self.presence_withheld_channels += 1

        if owes or unclean:
            self.presence_withheld_rounds += 1
            # SAY WHY, PERIODICALLY. A withholding rank is invisible in the
            # log -- it simply does not announce -- and the only symptom is
            # an abandonment 60 s later naming it as "never reached the
            # entry", which points at the wrong place: it DID reach the
            # entry and chose not to announce. The first metal run of this
            # design cost a log-dig for exactly that reason. Throttled by
            # the presence deadline so a healthy withhold of a few rounds
            # stays silent and a stuck one is on the record before the
            # abandonment that follows it.
            now = self._clock()
            if (
                self._last_withhold_log is None
                or (now - self._last_withhold_log) >= self._presence_deadline_s / 4.0
            ):
                self._last_withhold_log = now
                logger.warning(
                    "%s epoch %d round %d: WITHHOLDING presence (%d rounds so "
                    "far) -- %s. This rank is AT the entry and declining to "
                    "announce; it is not blocked upstream of it.",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                    self.presence_withheld_rounds,
                    "still owes a chain send" if owes else unclean,
                )
        else:
            self._last_withhold_log = None
            self._presence.announce(
                epoch, note=f"pending={self._pending}", round_=entry_round
            )

        # A rank that is withholding cannot be part of a full quorum (its
        # own flag is down), so this is skipped rather than merely false.
        # #631 H: the predicate is now "everyone present AND nobody
        # withdrawn". A stale presence flag from a rank that has since
        # abandoned must not form a quorum -- that is corpse H.
        if (
            not owes
            and not unclean
            and self._presence.quorum(epoch, round_=entry_round)
        ):
            # #631 G, THE ASSERT. Re-checked HERE, at the instant of entry,
            # because the withholding check above proves nothing about the
            # moment a quorum forms: a peer's message can land in between.
            # This is the cheap catch for the nastiest silent failure this
            # change can introduce -- a half-consumed two-step
            # point_to_point_pyobj message, or an unreaped isend, crossing
            # the re-formation and misframing the post-flip stream long
            # after the flip is forgotten. It also catches a sender that
            # died between posting its message and publishing its counter.
            #
            # Loud, and pre-entry: nothing has been entered, so abandoning
            # costs nothing and no peer is owed a collective. Crossing the
            # re-formation with a live channel would cost everything.
            late = None
            if self._channels_empty_fn is not None:
                try:
                    late = self._channels_empty_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s entry channel probe failed: %s", LOG_PREFIX, exc)
            if late:
                self.entry_channel_violations += 1
                logger.error(
                    "%s CHANNELS NOT EMPTY AT ENTRY for epoch %d round %d: "
                    "%s. A quiescent, fully serviced rank owes nothing on "
                    "any channel, so this is a framing or quiescence bug, "
                    "not a slow peer. Abandoning BEFORE entry -- nothing "
                    "was entered and no request was touched.",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                    late,
                )
                waited = self._clock() - self._presence_wait_started
                self._presence_wait_started = None
                return self._abandon_no_quorum(epoch, [], waited)
            self._commit_to_entering(epoch, entry_round)
            waited = self._clock() - self._presence_wait_started
            self._presence_wait_started = None
            # RE-BASE THE PARK CLOCK ON GROUP ASSEMBLY. The park deadline
            # measures "armed but never reached quiescence" -- a question
            # that is only meaningful once the group is actually
            # assembled. Left measuring from the arm, it races this gate:
            # a rank whose peers took longer than park_deadline_s to
            # arrive would abandon on the park deadline while those peers
            # were still polling, and the ranks would then disagree
            # around a gloo collective -- which is fatal, not merely
            # wrong ("Connection closed by peer" -> every rank aborts).
            # Measured 2026-08-08, boot 14: all three ranks announced
            # presence correctly, then abandoned at exactly 30.0s and the
            # group died.
            #
            # Re-basing keeps the two bounds from overlapping at all: the
            # presence bound governs assembly, the park bound governs
            # quiescence, and they now run in sequence rather than
            # concurrently.
            # ONCE PER ARM, never per round. Re-basing on every gate
            # opening makes the park deadline unreachable: the gate opens
            # each round, the clock resets each round, and a flip that can
            # never reach quiescence holds FOR EVER with its requests
            # parked -- measured 2026-08-08 (boot 17): repeated "group
            # present after 0.00s" on every rank, cutovers=0,
            # abandoned=0, and the server answering nothing.
            if self._gate_open_epoch != epoch:
                self._gate_open_epoch = epoch
                self._armed_at = self._clock()
                logger.warning(
                    "%s group present for epoch %d after %.2fs; park clock "
                    "re-based once, entering the consensus round",
                    LOG_PREFIX,
                    epoch,
                    waited,
                )
            return True

        waited = self._clock() - self._presence_wait_started
        if waited >= self._presence_deadline_s:
            # #631 H, THE WITHDRAWAL SIDE. Leaving is only permitted while
            # no peer has committed on this rank's presence. If one has,
            # this rank is still at the hook and owes it the reduction --
            # so it follows through instead of stranding it. That is the
            # invariant: any commit converts a withdrawing rank into an
            # enterer, and there is no interleaving where one enters and
            # another stays out.
            if not self._presence.may_withdraw(epoch, round_=entry_round):
                logger.warning(
                    "%s pre-entry bound expired for epoch %d round %d, but a "
                    "peer is already ENTERING on this rank's presence -- "
                    "following through into the reduction rather than "
                    "stranding it",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                )
                self._commit_to_entering(epoch, entry_round)
                self._presence_wait_started = None
                return True
            self._presence.declare_withdrawn(epoch, round_=entry_round)
            # Re-check AFTER publishing: a peer may have committed in the
            # window between the check and the write.
            if not self._presence.may_withdraw(epoch, round_=entry_round):
                logger.warning(
                    "%s withdrawal raced a peer's entry for epoch %d round "
                    "%d -- following through into the reduction",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                )
                self._commit_to_entering(epoch, entry_round)
                self._presence_wait_started = None
                return True
            missing = self._presence.missing(epoch, round_=entry_round)
            self._presence_wait_started = None
            return self._abandon_no_quorum(epoch, missing, waited)
        return None

    def _commit_to_entering(self, epoch: int, entry_round: int) -> None:
        """#631 H phase one: publish the intent to enter, then settle.

        Written BEFORE this rank actually enters, so a peer at its own
        pre-entry bound can see that someone has committed on its presence
        and is therefore forbidden from withdrawing. If a withdrawal is
        already visible, waiting is SAFE and terminates by construction:
        this rank's ENTERING marker forces that withdrawer to follow
        through, at which point it stops counting as withdrawn.
        """
        self._presence.declare_entering(epoch, round_=entry_round)
        if not self._presence.withdrawn(epoch, round_=entry_round):
            return
        deadline = self._clock() + self._presence_deadline_s
        while self._presence.withdrawn(epoch, round_=entry_round):
            if self._clock() >= deadline:
                logger.error(
                    "%s epoch %d round %d: a peer stayed WITHDRAWN despite "
                    "this rank ENTERING. The tie-break should have forced "
                    "it in; entering anyway would strand this rank.",
                    LOG_PREFIX,
                    epoch,
                    entry_round,
                )
                return
            self._sleep(self._presence_poll_interval_s)

    def _abandon_no_quorum(self, epoch: int, missing, waited: float):
        """Pre-entry abandonment: loud, safe, and retryable.

        Safe precisely because nothing was entered -- no peer is waiting
        on a collective this rank owes. Disarms and returns to normal
        cycling; the policy may re-arm, which mints a NEW epoch, so the
        stale flags of this one are never consulted again.
        """
        direction = self._pending
        self._pending = None
        self._armed_at = None
        self._parked_extent = None  # #746: a snapshot never outlives its flip
        self._last_hold_reason = None
        self.presence_timeouts += 1
        logger.error(
            "%s FLIP ABANDONED (no quorum): %s waited %.1fs for epoch %d "
            "and rank(s) %s never reached the flip entry (deadline %gs). "
            "NOTHING was entered and no request was touched -- serving "
            "continues on the %s stack and the policy may re-arm, which "
            "mints a new epoch. A rank that never reaches the entry is "
            "blocked upstream of it: look there, not at the flip.",
            LOG_PREFIX,
            direction,
            waited,
            epoch,
            missing,
            self._presence_deadline_s,
            self._phase,
        )
        return None

    def _join_bounded(self, payload):
        """#631(c): the consensus reduction under a deadline.

        Raises ``PhaseFlipJoinTimeout`` when peers do not join in time,
        instead of blocking for ever. The bound is deliberately generous
        (``DEFAULT_JOIN_DEADLINE_S``): a slow peer draining a long
        prefill is normal and must NOT trip it -- this is a wedge
        breaker, not a latency control.
        """
        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            CollectiveTimeoutError,
            PeerLostError,
        )

        try:
            return self._collective_min(payload, timeout_s=self._join_deadline_s)
        except TypeError:
            # An injected channel from a test/older builder that does not
            # take a deadline. Do not silently drop the bound: the whole
            # point is that this wait cannot be unbounded.
            logger.warning(
                "%s consensus channel takes no timeout; joining unbounded "
                "(a wedge here cannot be broken from inside)",
                LOG_PREFIX,
            )
            return self._collective_min(payload)
        except (CollectiveTimeoutError, PeerLostError) as exc:
            # BOTH, because the channel raises CollectiveTimeoutError and
            # catching only PeerLostError let it escape as a bare
            # "Fatal Python error: Aborted" (measured 2026-08-08).
            raise PhaseFlipJoinTimeout(
                f"no group-wide join within {self._join_deadline_s:g}s ({exc})"
            ) from exc

    def _abandon_unjoined_flip(self, why: str) -> None:
        """Give up on a flip whose consensus round never assembled.

        Same contract as ``_abandon_parked_flip``: disarm, log loudly,
        return to serving, and NEVER raise -- the flip is optional, the
        requests are not. The policy may re-arm at its next evaluation,
        so a transient skew costs one logged retry.
        """
        direction = self._pending
        self._pending = None
        self._armed_at = None
        self._parked_extent = None  # #746: a snapshot never outlives its flip
        self._last_hold_reason = None
        self.join_deadline_aborts += 1
        logger.error(
            "%s FLIP ABANDONED (join): %s never assembled a consensus "
            "round -- %s. Serving continues on the %s stack and no request "
            "was touched; the arm is dropped and may be retried. A peer "
            "that never joins is a peer that never reached its round hook: "
            "look for a rank blocked upstream of it, not for a slow flip.",
            LOG_PREFIX,
            direction,
            why,
            self._phase,
        )
        return None

    def _abandon_parked_flip(self, ready: int) -> None:
        """Give up on an armed flip that never reached quiescence.

        Disarms and returns to serving. Deliberately NOT an exception: the
        flip is the optional thing here, the requests are not. A raise
        would climb into the event loop and take the instance down with it,
        which is precisely the outcome this deadline exists to prevent --
        the parked requests would die with it.
        """
        waited = (
            self._clock() - self._armed_at
            if self._armed_at is not None
            else float("nan")
        )
        direction = self._pending
        self._pending = None
        self._armed_at = None
        self._parked_extent = None  # #746: a snapshot never outlives its flip
        self._last_hold_reason = None
        self.park_deadline_aborts += 1
        logger.error(
            "%s FLIP ABANDONED: %s was armed for %.1fs without the group "
            "reaching a quiescent boundary (deadline %gs; this rank "
            "ready=%d). The requests are NOT affected -- they were parked, "
            "not aborted, and serving resumes on the %s stack now. A rank "
            "that cannot park is holding work that never drains: look for a "
            "microbatch or a chunked prefill that never completes. Re-arm "
            "to try again.",
            LOG_PREFIX,
            direction,
            waited,
            self._park_deadline_s,
            ready,
            self._phase,
        )
        return None

    def _hold(self, reason: str) -> None:
        if reason != self._last_hold_reason:
            logger.info("%s hold: %s", LOG_PREFIX, reason)
            self._last_hold_reason = reason

    # -- pool/layer adapters --------------------------------------------------
    def _src_dst(self, direction: str) -> Tuple[KvPoolView, KvPoolView]:
        return (self._pp, self._tp) if direction == PP_TO_TP else (self._tp, self._pp)

    def _src_layer_idx(self, direction: str, ordinal: int) -> int:
        """Pool-local layer index of a global ordinal in MY sending pool."""
        if direction == PP_TO_TP:
            return self._map[self._rank].index(ordinal)
        return ordinal

    def _dst_layer_idx(self, direction: str, ordinal: int) -> int:
        if direction == PP_TO_TP:
            return ordinal
        return self._map[self._rank].index(ordinal)

    # -- seam tuning, re-read per flip ----------------------------------------
    def _refresh_seam_tuning(self) -> None:
        """Re-read the seam knobs from a tune FILE, if one was named.

        WHY A FILE AND NOT JUST THE ENV. The env is fixed at exec, so
        sweeping the row-block count costs one BOOT per point -- and a
        curve measured across boots cannot separate the knob from
        boot-to-boot variance, which is exactly the comparison the
        same-boot-floor rule exists to protect. A file the flip re-reads
        turns the whole curve into one boot with a floor arm in it.

        RANK-LOCAL DIVERGENCE IS SAFE HERE, and that is not true of every
        knob in this file. Row blocking touches only local backing and
        local writes -- the exchange is deliberately not blocked
        (``_stream_wave``) -- so two ranks at different block counts still
        call the collective the same number of times. The affordability
        verdict is a ``_collective_min``, so a differing reservation
        changes the number, never the unanimity. ``SGLANG_FLIP_SEAM_WAVES``
        would NOT be safe to put here for exactly the reason ``_flip_waves``
        documents, which is why this reads one key and not a settings dict.

        Inert unless ``SGLANG_FLIP_SEAM_TUNE_FILE`` names a path. A missing
        or malformed file leaves the boot-time value in place rather than
        failing a flip: this is measurement scaffolding, and it must not be
        able to abort a production seam.
        """
        path = os.environ.get("SGLANG_FLIP_SEAM_TUNE_FILE", "").strip()
        if not path:
            return
        try:
            with open(path) as fh:
                blocks = int(json.load(fh)["row_blocks"])
        except Exception as exc:  # pragma: no cover - defensive by design
            logger.warning(
                "%s seam tune file %s unreadable (%s); keeping row_blocks=%d",
                LOG_PREFIX,
                path,
                exc,
                self._seam_row_blocks,
            )
            return
        blocks = max(1, blocks)
        if blocks != self._seam_row_blocks:
            logger.info(
                "%s seam tune: row_blocks %d -> %d (from %s)",
                LOG_PREFIX,
                self._seam_row_blocks,
                blocks,
                path,
            )
        self._seam_row_blocks = blocks

    # -- wire frame agreement (register C22) ----------------------------------
    def _frame_digest(
        self,
        slots: torch.Tensor,
        direction: str,
        waves: Sequence[Sequence[int]],
    ) -> int:
        """Fingerprint of everything that FRAMES the wire, so the ranks can
        check they agree on it BEFORE a byte moves.

        WHAT THIS IS FOR. The per-peer payload length is
        ``rows x row_bytes`` summed over a wave's ordinals, and every term
        of that product is derived RANK-LOCALLY: the rows come from
        ``_live_slots_fn`` (documented "replicated", never verified), the
        wave partition from ``_flip_waves`` (pure, but see its own two
        documented gaps -- ``_pools_alias`` and ``SGLANG_FLIP_SEAM_WAVES``
        are both rank-local and both change the wave COUNT). Nothing on the
        wire carries a length, and the receiver's size check compares the
        buffer it allocated itself against the size it computed itself, so
        it is vacuous by construction and cannot see a divergence.

        WHAT A DIVERGENCE LOOKED LIKE BEFORE THIS (2026-08-13 13:03:16Z,
        the #656 acceptance run, after 320 clean cutovers): NCCL matched a
        send of one length against a recv of another, delivered the shorter
        one and completed, and the tail of the receiver's ``torch.empty``
        buffer -- which is where the checksum trailer lives -- was never
        written. The guard then reported a CHECKSUM MISMATCH naming a
        "sender" value of 4626949667419791296 on one rank and
        -4450328002521349435 on another. Neither is a possible uint8 sum
        (the second is negative; the first would need an 18-petabyte
        payload), so the failure was never about the data at all -- but it
        raised, and raising at the seam takes the INSTANCE down. One in 320
        cutovers, i.e. an unattended auto-flip instance's mean time to
        failure was about an hour.

        So the premise gets a BALLOT, exactly as #639 gave one to the
        prefix-length vector after the same class of bug. Reduced with the
        ``[x, -x]`` MIN pair the fit verdict already uses, on the collective
        that round already runs -- no extra round trip, and the answer is
        identical on every rank, so the abandon is unanimous and no rank can
        half-flip.

        The digest is modular (Mersenne 2**31-1, so every product fits an
        int64 and nothing wraps) and POSITIVE, which is what makes it safe
        to negate for the max half of the pair.
        """
        return self._frame_digest_parts(slots, direction, waves)["frame"]

    #: The three things a frame is made of, reported apart so a divergence can
    #: be ATTRIBUTED. See _frame_digest_parts.
    FRAME_PARTS = ("slots", "waves", "geometry")

    @staticmethod
    def _slots_acc(slots: torch.Tensor):
        """The live set's position-weighted accumulator, and its length.

        Factored out of :meth:`_frame_digest_parts` WITHOUT changing a single
        arithmetic step, because the same value has to be available one rung
        earlier: the live-slot agreement (#656 C22-d) votes on it at the KV
        rung, before the frame is computed, and a second implementation of the
        same fold would be a divergence source of its own.

        A pure function of the SET, given the sorted, deduplicated input every
        caller feeds it -- which is what makes it usable as a membership
        ballot. Fed raw it is order-sensitive by design; see
        :meth:`_frame_digest` and the test that pins the distinction.
        """
        mod = _FRAME_DIGEST_MOD
        s = slots.detach().to("cpu", torch.int64)
        n = int(s.numel())
        if not n:
            return 0, 0
        # Position-weighted so a permutation is not a collision; both
        # ends sort, so any difference here is a real set difference.
        acc = int(
            (((s % mod) * torch.arange(1, n + 1, dtype=torch.int64)) % mod).sum().item()
            % mod
        )
        return acc, n

    def _slots_membership_digest(self, slots: torch.Tensor) -> int:
        """The ``slots`` frame part, computed one rung early.

        Bit-for-bit the value ``_frame_digest_parts(...)["slots"]`` produces,
        so the rung's ballot and the frame's attribution can never disagree
        about what "the live slot set" means.
        """
        acc, n = self._slots_acc(slots)
        return (acc * 1000003 + n) % _FRAME_DIGEST_MOD

    def _frame_digest_parts(
        self,
        slots: torch.Tensor,
        direction: str,
        waves: Sequence[Sequence[int]],
    ) -> dict:
        """The frame digest, and the three parts it is made of.

        WHY THE PARTS EXIST. The combined digest detects a divergence and
        cannot attribute one, so its message has to hedge -- "the live slot
        set, the wave partition or the vector" -- and then names the pool
        census as the instrument, which only helps when the POOL is what
        differs. On boot_v2, 2026-08-13 16:00:42Z, it wasn't: the KV cap
        agreement had just levelled the group, all three POOL CENSUS lines
        were identical in every field (``size=579870 free=278572
        cached=300034 unaccounted=1264``), and the frames diverged anyway
        (PP1 250257408 against 1658515222). Six rounds of it followed with
        nothing in the log to say which term carried it.

        So each part rides the reduction as its own ``[x, -x]`` MIN pair.
        Six more integers in a payload the round already reduces: no new
        collective, and the collective COUNT invariant is untouched.

        ``frame`` is bit-for-bit what the ballot voted on before this
        change -- the parts ATTRIBUTE, they do not decide.

        All four values are modular (Mersenne 2**31-1, so every product fits
        an int64 and nothing wraps) and POSITIVE, which is what makes them
        safe to negate for the max half of each pair.
        """
        mod = _FRAME_DIGEST_MOD
        acc, n = self._slots_acc(slots)
        slots_part = (acc * 1000003 + n) % mod

        wave_terms: List[int] = [len(waves)]
        for wave in waves:
            wave_terms.append(len(wave))
            wave_terms.extend(int(o) for o in wave)
        waves_part = 0
        for term in wave_terms:
            waves_part = (waves_part * 1000003 + int(term)) % mod

        geometry_terms: List[int] = [
            1 if direction == PP_TO_TP else 2,
            int(self._n_layers),
        ]
        geometry_terms.extend(int(v) for v in self._vec)
        geometry_part = 0
        for term in geometry_terms:
            geometry_part = (geometry_part * 1000003 + int(term)) % mod

        # THE COMBINED VALUE IS THE ORIGINAL ONE, TERM FOR TERM AND IN ORDER.
        # Recomputed here rather than folded from the parts, because a fold
        # would quietly change the number the ballot votes on and every
        # digest in the corpus's logs would stop being comparable.
        terms: List[int] = [n, 1 if direction == PP_TO_TP else 2, len(waves)]
        terms.append(int(self._n_layers))
        terms.extend(int(v) for v in self._vec)
        for wave in waves:
            terms.append(len(wave))
            terms.extend(int(o) for o in wave)
        frame = acc
        for term in terms:
            frame = (frame * 1000003 + int(term)) % mod
        return {
            "slots": slots_part,
            "waves": waves_part,
            "geometry": geometry_part,
            "frame": frame,
        }

    @staticmethod
    def _name_frame_divergence(mine: dict, group_lo: dict, group_hi: dict) -> str:
        """Which framing term the group disagrees on, in words.

        ``group_lo``/``group_hi`` are the MIN and MAX of each part across the
        group. A part whose two ends differ is a part the ranks disagree on.
        """
        named = []
        for part, label in (
            ("slots", "the live slot set"),
            ("waves", "the wave partition"),
            ("geometry", "the layer geometry (vector, layer count or direction)"),
        ):
            lo, hi = group_lo.get(part), group_hi.get(part)
            if lo is None or hi is None or lo == hi:
                continue
            named.append(f"{label} (this rank {mine.get(part)}, group [{lo}, {hi}])")
        if not named:
            # NEVER SILENT. An unattributable divergence has to READ as
            # unattributable, or a successor takes the absence of a named
            # term as evidence about the terms rather than about the
            # granularity of the parts.
            return (
                "no single term explains it: every part agreed across the "
                "group while the combined digest did not, which means the "
                "parts are not fine-grained enough to carry this one -- split "
                "them further rather than concluding anything about the terms"
            )
        return "; ".join(named)

    # -- seam waves -----------------------------------------------------------
    def _flip_waves(self, direction: str) -> Tuple[Tuple[int, ...], ...]:
        """This flip's layer-wave split, ORDERED for the given direction.

        A PURE FUNCTION OF THE REPLICATED LAYER MAP AND THE DIRECTION: both
        ends of every pair must cut the wire payload the same way, and a
        wave count that travelled in a consensus payload would be one more
        thing that can disagree. The map is already replicated and already
        equality-checked at boot, and the direction is agreed by consensus
        before the seam runs, so deriving the split from the two makes
        agreement structural.

        THE DIRECTION IS AN INPUT because under restore-first the two
        directions want OPPOSITE orders (see ``ordered_layer_waves``): the
        destination of ``tp_to_pp`` is the PP layout, where a rank commits
        only on its own layers, so those want to be late; ``pp_to_tp``
        mirrors it exactly, so they want to be early. One static order
        cannot serve both.

        TWO KNOWN GAPS, neither introduced here, both worth a successor's
        attention because they get sharper as the wave count rises:

        * ``_pools_alias`` is a RANK-LOCAL pointer check, so a boot where
          one rank's pools alias and its peers' do not gives that rank 1
          wave and the others many. Ranks then call ``_exchange`` a
          different number of times. It is bounded rather than silent (the
          liveness poll aborts the flip) but it is not checked.
        * ``SGLANG_FLIP_SEAM_WAVES`` is read per process and never compared
          across ranks, with the same consequence.
        """
        if self._pools_alias():
            # ALIASED LAYOUTS CANNOT BE WAVED, and this is a correctness
            # bound rather than a tuning one. Waving interleaves reads of
            # the source layout with writes of the destination; when the
            # two overlay the same bytes, a wave's writes can land on rows
            # a later wave has not read yet -- the #297
            # reads-before-writes hazard, reachable exactly here. One wave
            # restores the invariant that every source read precedes every
            # destination write.
            return (tuple(range(self._n_layers)),)
        if not self._seam_restore_first:
            # THE ROLLBACK IS THE WHOLE OLD DESIGN, not just its order.
            # Restore-first and the lifted wave count are one change: a
            # one-layer wave under RELEASE-first has no release of its own
            # to pay for its commit, which is precisely the netting rule
            # that set the old cap. Rolling back the order while keeping
            # W = n_layers would be the worst of both -- measured on the
            # rig geometry at 354,868 tokens against the release-first
            # W=4 ceiling of ~435,000.
            n = self._n_waves or default_wave_count(self._map)
            return layer_waves(self._map, n)
        n = self._n_waves
        if n is None:
            # #631 2.1b: restore-first removes the per-wave netting rule
            # that capped this at the smallest stage, so the cap becomes
            # one layer per wave.
            #
            # MEASURED CAVEAT, recorded because the design note does not
            # say it: the PAYLOAD leg (send + receive buffers) stops
            # shrinking at about W=8 -- the widest wave still carries a
            # layer of one's own plus a layer from each peer, and that
            # floor is W-independent. Past W=8 the only thing still
            # improving is the backing transient via the ORDER below, and
            # each extra wave costs one more exchange round trip. If a
            # measurement ever shows the round trips dominating, W=8 is
            # the place to stand, not W=1.
            n = restore_first_wave_count(self._map)
        return ordered_layer_waves(self._map, self._vec, n, direction)

    def _pools_alias(self) -> bool:
        """Do the two layouts' pools overlay the same bytes? Cached.

        Pointers do not move after boot -- the VA reservations are fixed
        precisely so captured graphs keep replaying -- so this is asked
        once and remembered.
        """
        cached = getattr(self, "_seam_aliased", None)
        if cached is None:
            try:
                cached = bool(self._pp.overlaps(self._tp))
            except Exception:  # pragma: no cover - stub views
                cached = False
            if cached:
                logger.warning(
                    "%s the PP and TP pools share storage, so the seam runs "
                    "as a SINGLE wave: every source row must be read before "
                    "any destination row is written. Staging then scales "
                    "with the resident live set again, which is the "
                    "condition that livelocked a 270k-token request "
                    "(HANDOFF_666) -- an aliased arena and a waved seam are "
                    "alternative capacity designs, not a combination.",
                    LOG_PREFIX,
                )
            self._seam_aliased = cached
        return cached

    def _seam_swap(self):
        """The waved backing-swap hook, or None when the seam is unwaved.

        Unit stubs pass plain ``pre_write_fns`` callables; those run once
        at the seam exactly as before.
        """
        for fn in getattr(self, "_pre_write_fns", ()):
            if hasattr(fn, "release_wave"):
                return fn
        return None

    def _seam_backing_is_swappable(self) -> bool:
        swap = self._seam_swap()
        return bool(swap is not None and swap.is_swappable)

    # -- the seam's terminal verdict ------------------------------------------
    def _install_seam_cap_guard(
        self,
        direction: str,
        spent: int,
        too_small: Sequence[str],
        ask_bytes: Optional[int] = None,
        live_slots: Optional[int] = None,
    ) -> None:
        """Stand the flip down for good and say why, once.

        WHY THIS IS A VERDICT AND NOT A LOUDER RETRY. The seam's staging ask
        is set by the layer map and the live set; an abandon moves neither, so
        a refusal that has repeated ``cap`` times is a statement about the
        CONFIGURATION and no amount of patience answers it. Before this, the
        group re-armed every ``min_dwell`` forever: measured 185 group
        abandons in nine minutes on the #485 planner cut, each one running the
        full spill ladder while the armed window withheld admissions, until
        the detokenizer heartbeat expired and /health stopped answering inside
        its timeout -- with every stack IDLE in a normal wait and the instance
        still claiming the readiness it printed at boot.

        The verdict is installed as a BLOCKING GUARD because that is the one
        piece of state ``arm`` already honours, so "stay in the current phase"
        becomes a fact the next arm reads rather than a hope. ``_phase`` is
        deliberately untouched -- every abandon path already leaves it alone,
        and serving continues on whichever stack the instance is on.

        The headline is NOT "FLIP ABANDONED": every acceptance harness in this
        corpus counts that string, and a terminal verdict must not be
        summed into the same total as the retries it replaces.
        """
        detail = "; ".join(too_small) if too_small else "fits (a peer did not)"
        guard = (
            f"{SEAM_ABANDON_CAP_GUARD}: {direction} abandoned {spent} times "
            f"consecutively ({detail})"
        )
        # Idempotent: the group reaches this branch on every rank, and a
        # re-entry must not stack duplicate guards.
        if any(g.startswith(SEAM_ABANDON_CAP_GUARD) for g in self.blocking_guards):
            return
        self.blocking_guards = tuple(self.blocking_guards) + (guard,)
        self.seam_cap_verdicts = getattr(self, "seam_cap_verdicts", 0) + 1
        # #485 THE WITNESS. Retirement has to be able to say what changed, and
        # a verdict a reader cannot audit is a verdict a reader will disable.
        # ``ask_bytes`` may be None on a caller that does not have it: an
        # unknown ask can never be shown to have reversed, so such a guard is
        # simply not retirable, which is the shipped behaviour.
        if not hasattr(self, "_seam_cap_witness"):
            self._seam_cap_witness = {}
        self._seam_cap_witness[direction] = {
            "spent": int(spent),
            "ask_bytes": None if ask_bytes is None else int(ask_bytes),
            "live_slots": None if live_slots is None else int(live_slots),
            "detail": detail,
        }
        logger.error(
            "%s SEAM UNFUNDABLE -- PHASE FLIP STOOD DOWN (%s). %d consecutive "
            "group abandons reached the cap; this rank: %s. The staging ask is "
            "a property of the layer map and the live set, so retrying cannot "
            "change it and retrying is what kills the instance: each attempt "
            "runs the spill ladder while the armed window withholds "
            "admissions, and an unbounded loop starves the detokenizer "
            "heartbeat until the server stops answering /health while every "
            "stack sits IDLE. THE INSTANCE STAYS IN THE %s PHASE AND KEEPS "
            "SERVING -- degraded, because the other phase is now unreachable, "
            "but alive and saying so. To make the seam fundable, lower "
            "--max-total-tokens, raise this rank's --rank-gpu-memory-mib, or "
            "choose a layer cut whose seam fits; see SGLANG_SEAM_ABANDON_CAP "
            "to change or disable this bound.",
            LOG_PREFIX,
            direction,
            spent,
            detail,
            self._phase,
        )

    def retire_seam_cap_guard(
        self,
        direction: str,
        ask_bytes: int,
        affordable_bytes: int,
    ) -> bool:
        """Give the flip back once the shortage that stood it down is gone.

        Returns True only when the guard was actually removed. Every refusal
        path is silent-but-countable rather than raising: this is called from
        a service turn, and an instrument that can kill the loop it runs in is
        worse than no instrument.

        ``ask_bytes`` / ``affordable_bytes`` are THIS RANK's reading at the
        CURRENT live set -- not the ones the verdict was installed with. The
        whole point is that the live set moved.

        THE THREE FENCES, and each one is a test in
        ``test_seam_cap_retire_485.py``:

        EARNED. ``affordable >= ask + entry_margin + hysteresis``. The entry
        margin is the C20 term the seam would face on its next attempt, so
        clearing only that is clearing to the bar it is about to be measured
        against -- which flaps. The hysteresis is a separate constant so that
        disabling C20 cannot collapse the retire bar onto the entry bar.

        UNANIMOUS. The local verdict is reduced through ``_collective_min``,
        the same channel ``reduced_fit`` uses, so one rank that has drained
        cannot hand the group a seam a peer still cannot fund. Note the
        reduction runs on EVERY rank that reaches here, which is what makes
        the answer group-uniform rather than a rank-local opinion.

        BOUNDED. ``seam_cap_retire_limit()`` retirements, then the next
        verdict is permanent. Without this the pair (install, retire) is the
        unbounded retry the cap was built to end, one level up.

        WHAT THIS DELIBERATELY DOES NOT DO. It does not arm, does not run the
        spill ladder, and does not withhold admissions -- those three are what
        turned the original retry loop into a dead instance, and a retire
        probe that did any of them would rebuild it.
        """
        witness = getattr(self, "_seam_cap_witness", {}).get(direction)
        installed = [
            g for g in self.blocking_guards if g.startswith(SEAM_ABANDON_CAP_GUARD)
        ]
        if witness is None or not installed:
            # Never capped in this direction, or already retired. Not an error:
            # the caller is a periodic probe and asking is how it finds out.
            return False
        if witness.get("ask_bytes") is None:
            # An unknown ask cannot be shown to have reversed. Refusing here
            # keeps a caller that omits the witness on the shipped behaviour
            # rather than retiring on an assumption.
            return False
        limit = seam_cap_retire_limit()
        spent = getattr(self, "seam_cap_retirements", 0)
        if limit <= 0 or spent >= limit:
            return False
        bar = (
            int(ask_bytes)
            + seam_entry_margin_bytes()
            + (seam_cap_retire_hysteresis_bytes())
        )
        vote = 1 if int(affordable_bytes) >= bar else 0
        try:
            reduced = self._collective_min([vote])
        except Exception as exc:  # noqa: BLE001 -- a probe never kills the loop
            logger.debug(
                "%s seam cap retire vote could not be reduced (%s); the verdict stands",
                LOG_PREFIX,
                exc,
            )
            return False
        if not reduced or int(reduced[0]) == 0:
            return False

        self.blocking_guards = tuple(
            g for g in self.blocking_guards if not g.startswith(SEAM_ABANDON_CAP_GUARD)
        )
        self._seam_cap_witness.pop(direction, None)
        self.seam_cap_retirements = spent + 1
        # The streak and the damping go with it. Leaving the counter at the
        # cap would re-install the verdict on the very next abandon, which
        # would make the retirement a log line rather than a state change.
        if not hasattr(self, "_seam_abandons_in_a_row"):
            self._seam_abandons_in_a_row = {}
        self._seam_abandons_in_a_row[direction] = 0
        if not hasattr(self, "_seam_retry_at_arm"):
            self._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
            self.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
        self._seam_retry_at_arm[direction] = 0
        logger.warning(
            "%s SEAM CAP RETIRED (%s): the verdict was installed at an ask of "
            "%.1f MiB with %s live slot(s) after %d consecutive abandons (%s); "
            "the group now unanimously reads an ask of %.1f MiB against %.1f "
            "MiB affordable, clearing the %.1f MiB retire bar. The flip may "
            "arm again. Retirement %d of %d -- after the last one the verdict "
            "is permanent, because an unbounded install/retire pair is the "
            "livelock this cap exists to end (SGLANG_SEAM_CAP_RETIRE_LIMIT).",
            LOG_PREFIX,
            direction,
            witness["ask_bytes"] / 1048576.0,
            witness.get("live_slots"),
            witness.get("spent", 0),
            witness.get("detail", ""),
            int(ask_bytes) / 1048576.0,
            int(affordable_bytes) / 1048576.0,
            bar / 1048576.0,
            self.seam_cap_retirements,
            limit,
        )
        return True

    # -- staging affordability ------------------------------------------------
    def project_staging_bytes(self, direction: str, n_slots: int) -> int:
        """What the seam WILL want, for a live set of ``n_slots`` rows.

        #665-F1 item 7. The cost of a configuration change used to be
        discovered at arm time, by staging, being refused and abandoning on a
        live instance -- 512 -> 2048 took the pp_to_tp want from ~907 MiB to
        2481 and wedged the return leg with nothing having predicted it.

        This is the same `_staging_bytes` the gate itself calls, evaluated
        ahead of time against a projected live set, so it is EXACT rather than
        fitted and inherits every future change to the formula for free. An
        earlier attempt fitted want = base + k*chunk^2 to three measured
        anchors; it was falsified on a held-out point, and the anchors turned
        out to be one per RANK, so the curve was tracking per-rank arena_tail
        offsets rather than chunk width. There was never anything to fit: the
        plan's only live-set input is the slot count, and everything else is
        static layout.

        Per rank by construction -- `arena_tail` and the layer map differ
        across ranks, which is exactly what the fit mistook for a trend.
        """
        src, dst = self._src_dst(direction)
        waves = self._flip_waves(direction)
        slots = torch.arange(int(max(0, n_slots)), dtype=torch.int64)
        tr = build_phase_flip_transition(
            slots, self._map, self._n_layers, self._vec, self._rank, direction
        )
        return self._staging_bytes(tr, direction, src, dst, waves)

    def log_staging_projection(
        self, points: Sequence[Tuple[str, int]], floor_mib: float = 819.0
    ) -> None:
        """Print the projected want per direction at NAMED live-set points.

        A boot-time projection can only ever be a stated-assumption number,
        never an oracle: the seam is priced against the live set it actually
        meets, and at boot nobody knows what that will be. So every line names
        the assumption it was evaluated under, and several are printed, because
        the first version of this projected only at the ladder's arming
        threshold and under-read the real want by a constant ~511 MiB -- the
        flip ARMS at the threshold but EXECUTES against whatever has
        accumulated plus the resident set. The number was right; the sentence
        it silently implied was wrong.

        Reading these: find the line whose assumption is closest to the load
        you intend to run, and compare its "needs N MiB free" against the
        corridor. A configuration whose seam cannot be paid for at the live set
        you expect is visible here, at boot, instead of as a run of abandons on
        a live instance.
        """
        for label, n_slots in points:
            for direction in (TP_TO_PP, PP_TO_TP):
                try:
                    want = self.project_staging_bytes(direction, n_slots) / (1 << 20)
                except Exception as exc:  # must never fail a boot
                    logger.warning(
                        "%s staging projection unavailable for %s @ %s: %r",
                        LOG_PREFIX,
                        direction,
                        label,
                        exc,
                    )
                    continue
                logger.warning(
                    "%s STAGING PROJECTION rank %s %s: want %.0f MiB "
                    "projected @ %s = %d slots, + floor %.0f = needs %.0f MiB "
                    "free. Evaluated from _staging_bytes against the plan, not "
                    "fitted; valid for THIS stated live set only.",
                    LOG_PREFIX,
                    self._rank,
                    direction,
                    want,
                    label,
                    n_slots,
                    floor_mib,
                    want + floor_mib,
                )

    def _staging_bytes(self, tr, direction: str, src, dst, waves=None) -> int:
        """Device bytes the move will hold at once, from the PLAN.

        ``waves`` is the seam's layer-wave split (#631). The move stages ONE
        WAVE at a time -- pack, exchange, read the retained leg, swap that
        wave's backing, write -- so the peak is the WIDEST WAVE's legs, not
        the whole plan's. With a single wave containing every layer this is
        byte-identical to the unwaved formula, which is what makes the
        wave count a one-variable A/B.

        WHY THE WAVE SPLIT IS WHAT BOUNDS THIS. Before it, the seam swapped
        the two layouts' physical backing exactly once, so every byte
        crossing the seam had to be resident at that instant and the three
        legs below were each summed over the WHOLE plan. That made staging
        proportional to the resident live set, and past some request length
        no flip could ever be afforded -- under strict purity the request
        then never decodes, stays resident, and the refusal repeats
        forever (the 270k one-request livelock, HANDOFF_666). Waving the
        seam divides all three legs by the wave count, and the wave count
        is a property of the LAYER MAP, so what remains scales with the
        pool geometry instead of with the prompt.

        THREE legs, and the peak is not their sum. ``_execute`` spends
        them in two overlapping windows:

        * PACK + EXCHANGE holds ``outgoing + incoming``: this rank's
          packed send buffers (one exact-size buffer per peer, filled in
          place by ``_pack_outgoing``) and the receive buffers, which
          ``_exchange`` pre-allocates in full before posting the irecvs.
        * WRITE holds ``incoming + local``: the send buffers are released
          the moment the exchange returns, and only then is the retained
          local leg read -- in full, because every source read must
          complete before the first destination write (the cross-phase
          backing swap makes that invariant physical, not stylistic).

        So the high-water is ``incoming + max(outgoing, local)``. Both
        legs use their own side's ``row_nbytes`` -- the same quantities
        the move itself uses -- so this cannot drift from what is
        allocated.

        WHAT THIS REPLACED, and why the shape of the error mattered more
        than its size (HANDOFF_664 section 13a). The old formula was
        ``2 x outgoing + incoming``: the doubling modelled a packing that
        concatenated per-layer reads and so held each payload twice, and
        the LOCAL LEG WAS MISSING ENTIRELY. On the rig that under-reserved
        by 231-714 MiB per rank whenever the retained leg exceeded twice
        the outgoing one -- and every affordability refusal in this
        feature, including the one that livelocked pool 500000, was
        decided from it.

        ORDER OF THE TWO FIXES IS LOAD-BEARING. Correcting this formula
        alone would have made things WORSE: the gate's only action is to
        refuse, a refusal does not drain the resident set it refused on,
        so a larger reservation simply reaches the livelock at a smaller
        request. The packing was streamed FIRST -- measured 39.4 -> 19.2
        MiB peak on the hermetic three-rank flip, exactly the plan's floor
        -- so this honest budget is smaller than the dishonest one it
        replaces rather than larger. Do not reintroduce the doubling
        "for safety": it does not buy margin, it buys an earlier wedge.
        """
        # None = one wave over every layer, expressed as a null filter so
        # the formula needs no layer count of its own.
        if waves is None:
            waves = (None,)
        local_rows = tr.local_pp_rows if direction == PP_TO_TP else tr.local_tp_rows
        n_local = int(local_rows.numel())

        outgoing = incoming = local = 0
        for wave in waves:
            wave_set = None if wave is None else set(int(f) for f in wave)
            out_w = 0
            for peer in tr.send_layers:
                n = int(tr.send_rows[peer].numel())
                layers_w = _in_wave(tr.send_layers[peer], wave_set)
                if not layers_w or not n:
                    continue
                out_w += (
                    sum(
                        src.row_nbytes(self._src_layer_idx(direction, f)) * n
                        for f in layers_w
                    )
                    + _CHECKSUM_BYTES
                )
            in_w = 0
            for peer in tr.recv_layers:
                n = int(tr.recv_rows[peer].numel())
                layers_w = _in_wave(tr.recv_layers[peer], wave_set)
                if not layers_w or not n:
                    continue
                in_w += (
                    sum(
                        dst.row_nbytes(self._dst_layer_idx(direction, f)) * n
                        for f in layers_w
                    )
                    + _CHECKSUM_BYTES
                )
            local_w = sum(
                src.row_nbytes(self._src_layer_idx(direction, f)) * n_local
                for f in _in_wave(tr.local_layers, wave_set)
            )
            # The peak is per wave, and the widest wave is the one the gate
            # has to be able to afford.
            if in_w + max(out_w, local_w) > incoming + max(outgoing, local):
                outgoing, incoming, local = out_w, in_w, local_w
        # THE ONE-LAYER WINDOW. Streaming did not make the copies vanish,
        # it made them BOUNDED: ``read_rows_into`` still gathers one
        # layer's K and V before placing them, and ``write_rows`` still
        # materialises one layer's two halves before scattering them.
        # That is a fixed 1/L of a leg which dies before the next layer's,
        # so it does not scale with the sequence -- but it is real device
        # memory at the peak and the gate owns it. Measured: without this
        # term the prediction was 1.7 MiB short of a 29.4 MiB live set on
        # a local-leg-dominant geometry.
        widest_rows = max(
            [n_local]
            + [int(r.numel()) for r in tr.send_rows.values()]
            + [int(r.numel()) for r in tr.recv_rows.values()]
        )
        # Both sides are None when there is nothing to stage at all (an
        # empty live set): the formula must not reach for a row width it
        # has no use for.
        widest_layer_nbytes = 0
        if widest_rows:
            widest_layer_nbytes = max(
                max((src.row_nbytes(i) for i in range(src.num_layers)), default=0),
                max((dst.row_nbytes(i) for i in range(dst.num_layers)), default=0),
            )
        # Bounded by the gather block: KvPoolView reads and writes rows in
        # blocks, so the per-layer transient is a fixed window rather than
        # one layer of the whole live set.
        one_layer_window = min(widest_rows, _gather_block_rows()) * widest_layer_nbytes
        # THE WAVE-BOUNDARY SLACK, computed from the actual wave plan.
        backing_slack = self._backing_slack_bytes(direction, src, dst, waves)
        wave_peak = incoming + max(outgoing, local) + one_layer_window + backing_slack
        # THE SPILLED DRAFTER'S RESTORE (#656 rung 2), priced HERE and not at
        # the site that performs it. See _draft_restore_bytes.
        #
        # THE ARENA TAIL IS ADDED; THE DRAFT RESTORE IS NOT. #656, MERGE-R9
        # 12.4. This was one flat max() over all three on the reasoning that
        # the peaks belong to different instants of the seam. That reasoning
        # holds for the drafter -- rung 2's restore runs inside ``_cutover``,
        # after the waves' buffers are dead and after the source pool's pages
        # have gone back -- and it is FALSE for the arena tail, because
        # ``stacks.refill`` is a PRE-cutover function (see the pre_cutover_fns
        # list above, census label ``weights_refill``) and therefore commits
        # while the wave state is still outstanding.
        #
        # THE STAGE WALK, one cutover, tp_to_pp rank 1
        # (/spinning/evidence-631/remediation-656/boot_m1.log):
        #
        #   transient 1452 MiB (baseline free 2464 MiB, trough 1012 MiB at
        #   'weights_refill') *** CORRIDOR LAW BROKEN: deepest 1012 MiB ***
        #   ... backing_restore free=1250 | gdn_state free=1250
        #   weights_refill free=1012 step-238 | cutover free=1290 step+278
        #
        # The card entered at 2464, the wave walk left 1214 MiB outstanding at
        # 1250, and the refill's 238 MiB commit landed on top of THAT, twelve
        # MiB under the law. ``max(1214, 238)`` predicts a 1250 MiB trough --
        # 226 MiB clear -- so the seam entered on a verdict that could not see
        # the breach it was about to make. The additive form predicts it.
        #
        # AND IT IS A COMMIT, NOT A TRANSIENT: the tail stays backed into the
        # destination phase, so at the cutover it is still held while the
        # drafter is restored. Hence tail + max(waves, restore) rather than
        # max(tail + waves, restore).
        #
        # WHY THIS DOES NOT REINTRODUCE THE LIVELOCK the docstring above
        # refuses a larger reservation for. That objection is about terms
        # scaling with the RESIDENT SET: reserving more of those reaches the
        # wedge at a smaller request, because a refusal does not drain what it
        # refused on. The arena tail is a static LAYOUT quantity -- the span
        # one phase's weights image holds above the other's -- so it shifts
        # the affordable pool by a constant the prompt cannot move.
        #
        # IT OVER-RESERVES, and the direction is deliberate. Bounding the
        # outstanding wave state by the wave PEAK charges the full walk even
        # where the walk has partly drained by refill time. Across the ten
        # most-repeated tp_to_pp cutovers of that boot the max() form came out
        # at measured/predicted 1.11x (UNDER, worst single event +246 MiB) and
        # this form at 0.80x (OVER). For a gate whose only action is to
        # refuse, over-reserving costs a delayed flip and under-reserving cost
        # the corridor breach above.
        return int(
            self._arena_tail_bytes(direction)
            + max(wave_peak, self._draft_restore_bytes(direction))
        )

    def _arena_tail_bytes(self, direction: str) -> int:
        """Device bytes the tp->pp leg must commit for the weights-arena tail.

        RUNG 3's mirror of ``_draft_restore_bytes``, and it is here for the
        same reason: the commit happens inside ``PhaseFlipStacks.refill``,
        which runs at the pre-cutover seam -- past the point of no return. A
        failure there cannot be unwound.

        PRICED ON BOTH LEGS, and the reason it used to be priced on only one
        is a corrected assumption rather than an oversight. This said "the
        arena tail is re-committed on tp->pp, because PP is the larger layout
        on every rank of this rig" -- true for --pp-stage-ratio 14,10,8, false
        for 15,9,8, where a middle rank's PP layout falls below its TP layout
        and the pp->tp refill is the one that has to grow the arena. That
        unpriced commit faulted inside the no-return region and took all three
        ranks down at the first flip (measured 2026-08-11). The leg that must
        grow is a property of the LAYOUT SIZES, so both legs are priced and
        the carrier's own high-water answers how much.

        max(), not sum(), at the call site: the two peaks belong to different
        legs and cannot coexist.
        """
        # getattr, because this is now reached on BOTH legs. The old version
        # returned 0 for pp->tp before touching anything, so a runtime built
        # without a census scheduler never got here; pricing both legs means
        # the attribute is read on every staging estimate, including the ones
        # in tests that construct a bare runtime.
        scheduler = getattr(self, "_census_scheduler", None)
        stacks = getattr(scheduler, "phase_flip_stacks", None) if scheduler else None
        carrier = getattr(stacks, "arena_carrier", None) if stacks else None
        if carrier is None:
            return 0
        try:
            return int(carrier.pending_tail_bytes(stacks.refill_high_water_bytes()))
        except Exception:
            # An unreadable carrier must not take the flip down here; the
            # commit itself will still be attempted and the gate simply had
            # nothing to add.
            return 0

    def _draft_restore_bytes(self, direction: str) -> int:
        """Device bytes the pp->tp leg must be able to commit for the drafter.

        WHY THE GATE OWNS THIS. Rung 2 releases the draft weights' physical
        pages for the PP phase and re-commits them inside ``_cutover`` -- which
        is past the point of no return. A ``cuMemCreate`` failure there is not
        recoverable: the flip cannot go back, and the instance dies. That exact
        shape (``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`` inside a seam
        restore) took all three ranks down on 2026-08-09. Pricing the commit
        into the affordability verdict converts that death into a unanimous,
        free abandon before a single byte moves.

        WHY max() AND NOT A SUM. The wave staging and the draft restore never
        coexist: the waves' buffers are dead before the cutover runs, and the
        restore happens after the source pool's pages have gone back. Summing
        them would model a peak that does not occur and would abandon flips
        that fit -- and an abandoned flip is a visible functional regression
        against a record of 0 abandons in 402. What both asks DO share is the
        same free-minus-reserve budget, so the binding constraint is whichever
        is larger.

        Zero on the tp->pp leg (nothing is being restored) and zero whenever no
        carrier is installed or it is already resident, so instances without
        speculation and boots below depth 2 are untouched.
        """
        if direction != PP_TO_TP:
            return 0
        # getattr, not attribute access: _staging_bytes is exercised directly
        # by unit stubs built with object.__new__, which carry none of the
        # runtime's fields. An AttributeError here would surface as a staging
        # regression in tests that have nothing to do with the drafter.
        scheduler = getattr(self, "_census_scheduler", None)
        if scheduler is None:  # unit stubs
            return 0
        try:
            from sglang.srt.managers.phase_flip_spill import pending_restore_bytes

            stacks = getattr(scheduler, "phase_flip_stacks", None)
            return int(pending_restore_bytes(getattr(stacks, "draft_worker", None)))
        except Exception as e:  # pragma: no cover - never block a flip on this
            # A gate that cannot price the restore must not also refuse it;
            # the pre-rung-2 behaviour is the safe fallback.
            logger.warning(
                "%s could not price the draft restore, gating on the wave "
                "peak alone: %s",
                LOG_PREFIX,
                e,
            )
            return 0

    def _backing_slack_bytes(self, direction: str, src, dst, waves) -> int:
        """Residency the WAVE BOUNDARIES carry on top of the resting layout.

        A wave releases the source layout's pages for its own layers and
        then commits the destination's, and ``layer_waves`` sizes those to
        cancel. They cancel EXACTLY only when the wave count divides every
        stage's layer count; integer floors otherwise leave a drift, and
        while that drift is bounded by roughly one layer it is real device
        memory at the peak and the gate owns it.

        Charging a flat "one layer of the bigger pool" was the first
        version and it over-reserved by 3x on the rig ([7, 5, 4] over 16
        full-attention layers: 778 MiB charged against a true worst
        boundary of 242 MiB). Over-reserving is not the safe direction
        here -- the gate's only action is to refuse, and a refusal does
        not drain the resident set it refused on, so invented headroom
        moves the livelock to a smaller request rather than preventing
        one. So walk the plan and take the worst boundary, which is both
        exact and cheap.
        """
        if len(waves) <= 1 or not self._seam_backing_is_swappable():
            return 0
        if src is None or dst is None:
            return 0
        src_span = (
            max((src.row_nbytes(i) for i in range(src.num_layers)), default=0)
            * src.num_rows
        )
        dst_span = (
            max((dst.row_nbytes(i) for i in range(dst.num_layers)), default=0)
            * dst.num_rows
        )
        mine = set(self._map[self._rank])
        # Mirrors the gate in ``_execute``: aliased pools keep release-first
        # whatever the knob says, so they must be PRICED release-first too.
        restore_first = self._seam_restore_first and not self._pools_alias()
        blocks = self._effective_row_blocks(direction) if restore_first else 1
        released = committed = worst = 0
        for wave in waves:
            layers = [] if wave is None else list(wave)
            if direction == PP_TO_TP:
                # I release the wave's layers I own in PP; I commit all of
                # them in TP, where every ordinal lives on every rank.
                n_rel, n_com = len([f for f in layers if f in mine]), len(layers)
            else:
                n_rel, n_com = len(layers), len([f for f in layers if f in mine])
            com_w, rel_w = n_com * dst_span, n_rel * src_span
            committed += n_com * dst_span
            # THE ACCOUNTING FOLLOWS THE ORDER, and must: charging the
            # wrong one is a false verdict in whichever direction it errs.
            #
            # Restore-first (#631 2.1b) lands this wave's commit BEFORE its
            # own release, so the worst instant of wave j weighs everything
            # committed through j against everything released through j-1.
            # Crediting the current wave's release -- right under
            # release-first -- would under-reserve by one wave's release
            # span and let through a flip that cannot be afforded.
            #
            # Charging the restore-first term while actually running
            # release-first is the mirror error and is NOT the safe side:
            # the gate's only action is to refuse, and a refusal does not
            # drain the resident set it refused on, so invented headroom
            # moves the wedge to a smaller request rather than preventing
            # one (the reasoning in this method's docstring).
            if restore_first:
                # ROW BLOCKING (#631 section 2.1) SPLITS THIS WAVE'S COMMIT.
                # ``_stream_wave`` commits block b of the destination, writes
                # it, then releases through block b of the source, so the
                # peak inside the wave is the largest partial prefix rather
                # than the whole commit:
                #
                #   max over b in [0, B) of ((b+1)*com_w - b*rel_w) / B
                #
                # linear in b, so only the endpoints can be maximal. At B=1
                # it collapses to ``com_w`` and this whole branch is
                # byte-identical to the pre-2.1 accounting -- which is what
                # keeps the block count a one-variable A/B.
                #
                # THE ACCOUNTING MUST TRAVEL WITH THE LOOP, exactly as the
                # order and the wave count had to (HANDOFF_669 section 2.2).
                # The gate's reservation is what refuses a flip; leaving it
                # at the whole-layer term while the loop commits in blocks
                # means the shrink is real and never CASHED -- the pool
                # stays capped where it was and the change looks inert. The
                # mirror error is worse: pricing blocks the loop is not
                # running under-reserves and lets through a flip that ends
                # in an OOM inside the no-return region. Hence
                # ``_effective_row_blocks`` mirrors ``_execute``'s gate
                # rather than reading the knob.
                inner = com_w
                if blocks > 1:
                    num = max(com_w, blocks * com_w - (blocks - 1) * rel_w)
                    # Round the reservation UP: a partial byte of headroom
                    # is not headroom.
                    inner = -(-num // blocks)
                    # THE PHYSICAL FLOOR. Backing moves in arena chunks and
                    # ``commit_span`` rounds OUTWARD, so a block can never
                    # commit less than one chunk per buffer it touches. Past
                    # that point more blocks buy nothing, and claiming they
                    # do would under-reserve.
                    inner = max(
                        inner, min(com_w, self._seam_chunk_floor(direction, n_com))
                    )
                worst = max(worst, committed - com_w + inner - released)
                released += rel_w
            else:
                released += n_rel * src_span
                worst = max(worst, committed - released)
        return max(0, worst)

    def _effective_row_blocks(self, direction: str) -> int:
        """Row blocks the seam will ACTUALLY stream in on this direction.

        Mirrors the branch conditions in ``_execute`` rather than reading
        ``SGLANG_FLIP_SEAM_ROW_BLOCKS`` directly, because the knob being set
        is not the same as the loop running: release-first and aliased pools
        take the whole-wave branch, and a pool whose arena has no commit
        chunk cannot do span ops at all. Pricing a shrink that does not
        happen is an under-reservation, which the gate cannot survive.
        """
        # getattr: the accounting is reachable from runtimes built with
        # __new__ in tests that predate the knob, and defaulting to
        # whole-wave there is both correct and the safe direction.
        if int(getattr(self, "_seam_row_blocks", 1) or 1) <= 1:
            return 1
        if not self._seam_restore_first or self._pools_alias():
            return 1
        swap = self._seam_swap()
        if swap is None or not getattr(swap, "is_span_swappable", None):
            return 1
        blocks = int(getattr(self, "_seam_row_blocks", 1) or 1)
        return blocks if swap.is_span_swappable(direction) else 1

    def _seam_chunk_floor(self, direction: str, n_layers: int) -> int:
        """Smallest commit a block can make: one arena chunk per buffer.

        Asks the real destination POOL, not the plan's view -- the chunk is
        a property of the arena the pages come from. Zero when nothing can
        report one, which makes the floor inert rather than inventing a
        number.
        """
        swap = self._seam_swap()
        getter = getattr(swap, "commit_chunk_bytes", None) if swap else None
        chunk = int(getter(direction) or 0) if callable(getter) else 0
        if chunk <= 0 or n_layers <= 0:
            return 0
        # Two buffers (K and V) per full-attention layer in this pool.
        return chunk * 2 * int(n_layers)

    # -- #631 section 2.1: streamed (row-blocked) seam -------------------------

    @staticmethod
    def _write_jobs(dst, jobs) -> None:
        for li, rows, data in jobs:
            dst.write_rows(li, rows, data)

    def _stream_wave(self, swap, direction, wave, src, dst, jobs, blocks) -> None:
        """Restore, write and release ONE ROW BLOCK at a time.

        The peak this bounds is the seam's backing transient. Whole-wave
        restore-first commits an entire layer span before releasing
        anything; here the commit unit is a block, so the term shrinks
        roughly as ``1 / blocks``. Section 2.1 of the #631 notes, and the
        only remaining route to the >=600000 floor -- the wave ORDER can
        move that term between cards but not shrink it (HANDOFF_669).

        WHY A CONTIGUOUS ROW RANGE CAN SELECT SCATTERED WRITES CHEAPLY, and
        this is the load-bearing trick. The backing span API takes a
        contiguous pool row range, while the rows this wave writes are the
        LIVE slots, scattered through the pool. The two reconcile because
        the plan enumerates slots ASCENDING on both ends, so the rows
        falling inside a contiguous range are a contiguous SLICE of the row
        tensor -- one searchsorted per job per block, no gather, and the
        payload slice comes along at the same offsets.

        That ascending invariant is asserted rather than assumed: if a
        future plan change ever emits unsorted rows, the slice would
        silently write the wrong payload to the wrong rows, and a silent
        KV corruption is the worst failure this file can produce.

        The exchange is deliberately NOT blocked here. Row-blocking the
        exchange needs a GLOBAL round count so every rank calls the
        collective the same number of times -- derived from the replicated
        plan, never from a rank-local row count, because a rank-local count
        deadlocks the group and looks exactly like a hang. That is a
        separate, riskier change; this one touches only local backing and
        local writes, so no rank can diverge from its peers.
        """
        for _li, rows, _data in jobs:
            if int(rows.numel()) > 1 and not bool(torch.all(rows[1:] >= rows[:-1])):
                raise KvReshardError(
                    f"{LOG_PREFIX} streamed seam requires ascending row "
                    f"enumeration; got an unsorted row tensor. Blocking a "
                    f"scattered write by row RANGE relies on rows inside a "
                    f"range being a contiguous slice, so an unsorted tensor "
                    f"would write the wrong payload to the wrong rows."
                )
        dst_rows, src_rows = int(dst.num_rows), int(src.num_rows)
        for b in range(blocks):
            dlo = (b * dst_rows) // blocks
            dhi = ((b + 1) * dst_rows) // blocks
            swap.restore_wave_span(direction, wave, dlo, dhi)
            for li, rows, data in jobs:
                if not int(rows.numel()):
                    continue
                bounds = torch.tensor([dlo, dhi], dtype=rows.dtype)
                i0, i1 = torch.searchsorted(rows, bounds).tolist()
                if i1 > i0:
                    dst.write_rows(li, rows[i0:i1], data[i0:i1])
            # RELEASE CUMULATIVELY FROM ROW 0, and this is not sloppiness.
            # ``decommit_span`` rounds INWARD -- a chunk only partly inside
            # the range still holds live rows, so unmapping it would be
            # silent KV corruption. For an interior boundary that means
            # block b's ``hi`` rounds BELOW it and block b+1's ``lo`` rounds
            # ABOVE it, so the chunk straddling the boundary is released by
            # NEITHER: (blocks - 1) chunks per buffer left mapped on the
            # resting layout, a residual that GROWS with the block count and
            # eats the 1/blocks gain this loop exists to win.
            #
            # Restating [0, hi) each block covers those chunks on the next
            # pass, and re-releasing already-unmapped rows is free (the
            # extent is gone; ``decommit_span`` walks extents). It is sound
            # at any width because NO source row is live here: ``_execute``
            # reads the whole retained leg and drains the exchange before
            # the seam opens, so the source pool is write-only-dead for the
            # duration of this loop.
            shi = ((b + 1) * src_rows) // blocks
            swap.release_wave_span(direction, wave, 0, shi)
        # The span restores left the destination marked non-resident on
        # purpose; this is what closes the wave. Safe to call after a
        # streamed restore only because commit_range consults the extent
        # list rather than the contiguous watermark (3bbf2f50bb) -- on the
        # old code it would have re-mapped live extents.
        swap.finalize_wave(direction, wave)

    def _reclaim_cached_blocks(self) -> None:
        """Hand the allocator's unused blocks back to the driver.

        Injectable so the unit tests can model a reclaim that succeeds, one
        that partly succeeds, and one that returns nothing -- the three
        cases whose verdicts must differ.
        """
        hook = getattr(self, "_mem_reclaim", None)
        if hook is not None:
            hook()
            return
        if torch.cuda.is_available():  # pragma: no cover - needs a device
            torch.cuda.empty_cache()

    def _record_seam_peak(
        self,
        direction: str,
        staging_bytes: int,
        driver_free: int,
        cached_free: int,
    ) -> None:
        """#677: attribute the seam draw AT ITS PEAK INSTANT, per component.

        WHY HERE. This is the moment the flip's demand is weighed against
        free VRAM -- the peak the arming floor is sized to survive. Anywhere
        earlier the buffers do not exist yet; anywhere later the decision has
        already been taken on a number nobody decomposed.

        WHAT IT IS FOR. The arming floor is one scalar per rank (909 / 1006 /
        1648 MiB on this rig, NOTE_677_floor_components.md) with no recorded
        composition, so no part of it can be traded against anything: a
        component that could live in host RAM cannot be identified, and one
        that genuinely must stay resident cannot be defended. Splitting the
        peak is the precondition for every per-component decision #702 and
        #677 want to make.

        ON THE #605 CHANNEL, not a new one. ``flight_recorder.mark`` already
        carries the torch view, the NVML view and the boot id, and it is
        append-only -- a rank that dies here still leaves its line. A second
        channel would have to re-derive all of that and could disagree with
        it.

        NULLS, NEVER ZEROS, for components this layer cannot see. A zero here
        would read as "this component costs nothing", which is the #606
        defaulted-measurement defect; ``None`` reads as "not measured from
        here", which is true and actionable. ``unattributed_bytes`` is the
        residual the named terms do not explain and is the number that says
        how far this instrument still has to go.

        Cannot break a flip: the recorder no-ops unless its directory env is
        set, and the whole body is guarded -- an instrument on the seam path
        may cost a missing line, never a cutover.
        """
        try:
            from sglang.srt.mem_ledger import flight_recorder

            arena_tail = None
            try:
                arena_tail = int(self._arena_tail_bytes(direction))
            except Exception:  # pragma: no cover - defensive
                arena_tail = None

            named = int(staging_bytes) + int(arena_tail or 0)
            extra = {
                "direction": str(direction),
                "epoch": int(getattr(self, "_epoch", 0)),
                # The staging peak: incoming + max(outgoing, local) over the
                # WIDEST wave, per _staging_bytes.
                "staging_bytes": int(staging_bytes),
                # The refill destination: arena tail this direction commits.
                "refill_destination_bytes": arena_tail,
                # Graph capture workspace is a BOOT-time term under the
                # restore-never-rebuild invariant (#677) and is not visible
                # from this layer. Explicitly unmeasured rather than 0.
                "graph_workspace_bytes": None,
                "staging_reserve_bytes": int(self._staging_reserve_bytes),
                "driver_free_bytes": int(driver_free),
                "allocator_cached_free_bytes": int(cached_free),
                "named_bytes": named,
                # What the named terms do not explain. Signed on purpose: a
                # NEGATIVE residual means the named terms over-count, which is
                # a different defect from an unattributed remainder and must
                # not be hidden by a max(0, ...).
                "unattributed_bytes": int(staging_bytes) - named,
            }
            flight_recorder.mark(
                "seam_peak",
                rank=int(getattr(self, "_world_rank", 0) or 0),
                extra=extra,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("%s seam-peak attribution skipped: %s", LOG_PREFIX, e)

    def _staging_affordable(
        self, staging_bytes: int, direction: str = ""
    ) -> Tuple[bool, str]:
        """Can this rank stage ``staging_bytes`` without eating the reserve?

        THE TWO POOLS OF MEMORY ARE NOT EQUIVALENT, and conflating them
        gives the wrong answer in both directions:

        * bytes the caching allocator already holds but has not handed out
          are reusable AND invisible to NVML, so spending them cannot move
          the free-VRAM number the corridor is measured on.
        * bytes that must come from the driver DO move it, so only the
          amount above the reserve may be spent.

        THE CACHE CREDIT IS ONLY REAL ONCE IT HAS BEEN MATERIALISED.  This
        check used to read ``usable = cached_free + max(0, driver_free -
        reserve)`` and stop there, which credits a fungibility the
        allocator does not provide: ``cached_free`` is the SUM of every
        free block, and a 457 MiB staging buffer cannot be cut out of 1166
        MiB scattered across hundreds of small ones.  When the cache cannot
        serve the request the allocator goes to the driver instead, and the
        corridor -- the very number this reserve exists to protect -- pays
        for a credit that was never collectable.  Measured on the live rig
        (2026-08-10): ``/flush_cache`` on an IDLE instance handed 1166 MiB
        back to the driver on the binding card, so hoard at that scale is
        the normal resting state, not an edge case.

        So when the decision has to lean on the cache, RETURN THE CACHE TO
        THE DRIVER FIRST and then judge against ``driver_free`` alone.  If
        the reclaim fully succeeds the verdict is arithmetically identical
        to the old formula, and the corridor is restored as a side effect;
        if it only partly succeeds the verdict is correctly smaller.  This
        can never be more permissive than what it replaced, only more
        honest, which is why it cannot introduce an OOM the old code
        refused.

        The same reclaim doubles as the corridor keeper: if driver-free has
        already fallen under the reserve while the allocator sits on unused
        blocks, hand them back whether or not this particular flip needs
        them.  The flip boundary is the natural place for it -- the
        outgoing layout's buffers have just been released.

        Off CUDA (unit stubs) there is nothing to measure and this term is
        not the one under test, so it abstains rather than inventing a
        number.
        """
        probe = self._mem_probe
        if probe is None:
            if not torch.cuda.is_available():  # pragma: no cover - stubs
                return True, ""

            def probe():
                free_dev, _total = torch.cuda.mem_get_info()
                cached_free = (
                    torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                )
                return int(free_dev), int(max(0, cached_free))

        reserve = self._staging_reserve_bytes
        # Remembered for the ABANDONED census, which runs later and otherwise
        # has no way to quote the figure the refusal was about.
        self._last_staging_bytes = int(staging_bytes)
        driver_free, cached_free = probe()
        self._record_seam_peak(direction, int(staging_bytes), driver_free, cached_free)
        from_driver = max(0, driver_free - reserve)
        if cached_free > 0 and (staging_bytes > from_driver or driver_free < reserve):
            before = driver_free
            self._reclaim_cached_blocks()
            driver_free, cached_free = probe()
            from_driver = max(0, driver_free - reserve)
            mib = 1024 * 1024
            logger.info(
                "%s staging reclaim: driver free %.0f -> %.0f MiB "
                "(+%.0f returned), %.0f MiB still cached, reserve %.0f MiB, "
                "staging needs %.0f MiB",
                LOG_PREFIX,
                before / mib,
                driver_free / mib,
                (driver_free - before) / mib,
                cached_free / mib,
                reserve / mib,
                staging_bytes / mib,
            )
        usable = from_driver
        if staging_bytes <= usable:
            return True, ""

        # #689 THE GUARD IS ASKED BEFORE THE FLIP IS ABANDONED.
        #
        # THE MEASUREMENT THAT PUT THIS HERE. At an ABANDONED instant the
        # staging budget census read, per rank:
        #     rank0 5090 : needs 1305, spendable 1599, arena    0  -> fits
        #     rank1 3080 : needs 1059, spendable  867, arena  815  -> SHORT 192
        #     rank2 3080 : needs 1763, spendable 2051, arena 1456  -> fits
        # The binding rank was short by 192 MiB while holding 815 MiB of the
        # INACTIVE layout's weights -- 4.2x the shortfall -- and its headroom
        # over the corridor floor was 150 MiB against 1606 and 1334 on the
        # peers. So the seam was not failing on the corridor floor and not on
        # the live set: one rank was holding the OTHER layout's arena while
        # trying to stage this one, and nothing ever asked it to give it back.
        #
        # ensure_headroom IS THE DESIGNED COMPOSITION POINT, not a new
        # mechanism -- its own docstring discusses the seam ("at the seam:
        # abandon the flip") and refusal_is_fatal for exactly this leg. The
        # draft-weights provider is already registered in the rebalance tier;
        # it simply was never asked from here.
        #
        # refusal_is_fatal ON pp_to_tp, for the reason that docstring gives:
        # strict purity forbids decode in PP, so a refused pp_to_tp starves
        # decode and NOTHING IN PP CAN FREE the memory that would end the
        # refusal. That leg has no survivable wait, so the host tier opens.
        shortfall = int(staging_bytes) - int(usable)
        guard = None
        try:
            from sglang.srt.managers.phase_flip_spill import get_corridor_guard

            sched = getattr(self, "_census_scheduler", None)
            guard = get_corridor_guard(sched) if sched is not None else None
        except Exception:  # noqa: BLE001 - never break the refusal path
            guard = None
        if guard is not None and shortfall > 0:
            try:
                res = guard.ensure_headroom(
                    shortfall,
                    reason=f"seam staging {direction}",
                    refusal_is_fatal=(direction == PP_TO_TP),
                    # #689: the seam has ALREADY accounted the free column --
                    # `usable` above is computed from it. It needs the ladder
                    # to RELEASE `shortfall` more, so the ask must be judged by
                    # the measured delta. Without this the guard answered "are
                    # 178 MiB free" (trivially yes at 1428 free), reported
                    # success three times having freed nothing, and the seam
                    # abandoned believing it was funded.
                    must_reclaim=True,
                )
                driver_free, cached_free = probe()
                usable = max(0, driver_free - reserve)
                logger.info(
                    "%s seam staging asked the corridor guard for %.0f MiB "
                    "(%s): ok=%s, spendable now %.0f MiB against a need of "
                    "%.0f MiB. ok=False here means the ladder RELEASED nothing "
                    "(must_reclaim judges the delta, not the free column), so "
                    "the shortfall is real and every registered provider is "
                    "dry.",
                    LOG_PREFIX,
                    shortfall / (1024 * 1024),
                    direction,
                    getattr(res, "ok", "?"),
                    usable / (1024 * 1024),
                    staging_bytes / (1024 * 1024),
                )
                if staging_bytes <= usable:
                    return True, ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s seam staging could not ask the corridor guard (%r); "
                    "falling through to the abandon path unchanged.",
                    LOG_PREFIX,
                    exc,
                )
        mib = 1024 * 1024
        return False, (
            f"staging {staging_bytes / mib:.0f} MiB needed but only "
            f"{usable / mib:.0f} MiB is spendable "
            f"(driver free {driver_free / mib:.0f} MiB, allocator cache "
            f"{cached_free / mib:.0f} MiB, reserve "
            f"{self._staging_reserve_bytes / mib:.0f} MiB kept free). The "
            f"KV pool is too full to carry its own contents across the "
            f"flip; serving continues in this layout and the flip is "
            f"retried when occupancy drops. "
            + (
                # #688: when the flip was armed because NOTHING can run, an
                # unfunded seam is an idle window, not a deferral. Say so, name
                # the shortfall, and name the event that can change it -- an
                # operator reading "retried when occupancy drops" has no way to
                # know that occupancy cannot drop, because the only thing that
                # would drop it is the flip being refused here.
                f"ARMED IDLE-LOCKED, so this is a STALL, not a deferral: "
                f"{(staging_bytes - usable) / mib:.0f} MiB short after the "
                f"relief rung was asked for the margin as mandatory. Nothing "
                f"in this layout can admit or decode, so occupancy will NOT "
                f"drop on its own; the next event that can change this is a "
                f"resident request completing or being aborted. "
                if bool(getattr(self, "armed_idle_locked", False))
                else ""
            )
            + self._kv_rung_verdict()
        )

    #: Minimum seconds between arm ATTEMPTS on one direction while a damper is
    #: standing down. Not a latch: it paces re-pricing, it never stops it, and
    #: an attempt that made PROGRESS resets it to zero so a funding run is
    #: never throttled.
    SEAM_ARM_MIN_INTERVAL_S = 2.0

    def _storm_limiter_allows(self, direction: str) -> bool:
        """Pace arm attempts without ever blocking a funded one.

        THE STORM IS REAL AND SO IS THE FIX THAT CAUSED IT. Removing three
        dampers made the seam re-priceable, which is what let it fund at high
        occupancy -- and also let arms run at ~20/min where boot E's pathology
        was 179 in nine minutes. F1's pin is right to call that a storm.

        A LATCH IS THE WRONG SHAPE, which is the whole lesson of this ticket:
        every latch here blocked re-pricing while the arming condition
        persisted, and each one cost the prefill layout. So this is a RATE
        limiter. It bounds attempts per direction in time and nothing else:

        * it never refuses because of a COUNT, only because of an interval;
        * an attempt that made progress -- a completed flip, or a KV release
          that moved the driver -- clears the interval immediately, so a run
          that is actually funding is never throttled;
        * it applies only where a damper is already standing down, so the
          ordinary path is untouched.

        The result is that a seam which CAN fund keeps funding at full speed,
        and a seam which cannot stops burning a core on it.
        """
        now = time.monotonic()
        if not hasattr(self, "_seam_last_arm_at"):
            self._seam_last_arm_at = {}
        last = self._seam_last_arm_at.get(direction)
        if last is not None and (now - last) < self.SEAM_ARM_MIN_INTERVAL_S:
            return False
        self._seam_last_arm_at[direction] = now
        return True

    def note_seam_progress(self, direction: str) -> None:
        """A flip completed or the rung released bytes: stop pacing this one.

        Called on the paths that represent real progress, so the limiter can
        never throttle a direction that is successfully funding itself.
        """
        try:
            if hasattr(self, "_seam_last_arm_at"):
                self._seam_last_arm_at.pop(direction, None)
        except Exception:  # pragma: no cover - pacing must not raise
            pass

    def _arming_condition_persists(self) -> bool:
        """Is there still work waiting that wants the other layout?

        Deliberately coarse: any queued request is enough. The POLICY has
        already decided which layout this load wants and only issues an arm
        when it wants one, so this is not a second opinion about the decision
        -- it only distinguishes "the load is still there" from "the load has
        gone away", which is the one distinction the abandon counter needs and
        never had.

        False on anything unreadable, which keeps the counter's old behaviour
        exactly: an unreadable queue must not be able to disable a damper.
        """
        try:
            scheduler = getattr(self, "_census_scheduler", None)
            if scheduler is None:
                return False
            # QUEUED **OR RUNNING**, and the difference cost a whole boot.
            #
            # The first version of this asked only about the waiting queue,
            # which is empty exactly when the work has been admitted -- so at
            # 90k tokens resident the log read "#running-req: 1,
            # #full token: 457724, #queue-req: 0" and the damper did NOT stand
            # down, because by its reading nothing was waiting. The load that
            # most wants the other layout is the load that is already in the
            # machine.
            for name in ("waiting_queue", "grammar_queue"):
                q = getattr(scheduler, name, None)
                if q and len(q) > 0:
                    return True
            running = getattr(scheduler, "running_batch", None)
            reqs = getattr(running, "reqs", None) if running is not None else None
            if reqs and len(reqs) > 0:
                return True
            cur = getattr(scheduler, "cur_batch", None)
            cur_reqs = getattr(cur, "reqs", None) if cur is not None else None
            return bool(cur_reqs and len(cur_reqs) > 0)
        except Exception:  # noqa: BLE001 - a damper must not raise
            return False

    def staging_budget_census(self, staging_bytes: int = 0) -> str:
        """WHO IS HOLDING THIS RANK'S SEAM STAGING BUDGET, in MiB.

        The ABANDONED receipt says the pool is too small and, on eight of the
        twelve abandons measured 2026-08-16 11:05, that "This rank: fits (a
        peer did not)" -- which names a BINDING RANK and nothing about what is
        binding it. Three remedies were then plausible (arena tail, inactive
        layout arena, draft weights) with no way to choose between them, so
        this prints the occupants instead of leaving the next reader to guess.

        A CENSUS, NEVER A GATE: every term is read defensively and a failure
        prints as "?" rather than raising, because a refusal path is the worst
        possible place to add a new exception.
        """
        mib = 1024 * 1024

        def q(fn, default="?"):
            try:
                v = fn()
                return f"{v / mib:.0f}" if isinstance(v, (int, float)) else str(v)
            except Exception:  # noqa: BLE001 - a census must not raise
                return default

        probe = self._mem_probe
        driver_free = cached = None
        try:
            if probe is not None:
                driver_free, cached = probe()
            elif torch.cuda.is_available():
                driver_free, _t = torch.cuda.mem_get_info()
                cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        except Exception:  # noqa: BLE001
            pass
        # The REAL accessors, not invented attribute names: the runtime keeps
        # the scheduler as ``_census_scheduler`` and both the guard and the
        # measured reserve are fetched through their own modules, exactly as
        # the prearm-relief path above does.
        sched = getattr(self, "_census_scheduler", None)
        res = guard = None
        try:
            from sglang.srt.managers.phase_flip_seam_reserve import read_seam_reserve
            from sglang.srt.managers.phase_flip_spill import get_corridor_guard

            if sched is not None:
                guard = get_corridor_guard(sched)
                res = read_seam_reserve(
                    sched.server_args, int(getattr(self, "_rank", 0) or 0)
                )
        except Exception:  # noqa: BLE001 - a census must not raise
            pass
        parts = [
            f"staging needs {staging_bytes / mib:.0f}",
            f"driver free {q(lambda: driver_free)}",
            f"allocator cache {q(lambda: cached)} (reclaimable)",
            f"staging reserve kept free {q(lambda: self._staging_reserve_bytes)}",
            f"seam fixed {q(lambda: res.fixed_bytes)}",
            # NOT "the inactive layout's weights held on this card". This is
            # reserve.arena_fixed_bytes: the tp_to_pp leg's FUTURE commit cost,
            # priced into staging, and ZERO on pp_to_tp. Measured per rank:
            # rank1 arena_fixed 815 with pp_to_tp tail 0 / tp_to_pp tail 815;
            # rank2 1456 with 0 / 1456; rank0 0 / 0 / 0. The old label read as
            # reclaimable memory and sent a design note down a dead end -- rung
            # 3 has already released the real tail by then (receipts: "rung 3
            # released 1410.0 MiB ... TP layout needs 8977.8 of 10434.0").
            f"tp_to_pp ARENA COMMIT DUE (0 on this leg if pp_to_tp) "
            f"{q(lambda: res.arena_fixed_bytes)}",
            f"reserve active={getattr(res, 'active', '?')}",
            f"corridor floor {q(lambda: guard.floor_bytes)}",
        ]
        # THE TWO DERIVED NUMBERS THAT ACTUALLY DECIDE IT. Absolutes make the
        # reader do the arithmetic; the question "could this rank have paid"
        # is driver_free minus what must stay free, and that is what a
        # binding-rank diagnosis turns on.
        try:
            spendable = int(driver_free) - int(self._staging_reserve_bytes)
            parts.append(f"=> SPENDABLE {spendable / mib:.0f}")
        except Exception:  # noqa: BLE001
            parts.append("=> SPENDABLE ?")
        try:
            parts.append(
                f"headroom over corridor floor "
                f"{(int(driver_free) - int(guard.floor_bytes)) / mib:.0f}"
            )
        except Exception:  # noqa: BLE001
            parts.append("headroom over corridor floor ?")
        return (
            "STAGING BUDGET CENSUS (MiB) -- " + ", ".join(parts) + ". The "
            "arena term is a SCHEDULED COMMIT for the tp_to_pp leg, not memory "
            "held idle now and not reclaimable here: rung 3 releases the real "
            "tail immediately after each refill, so at a pp_to_tp gate the "
            "arena is already committed exactly to the active PP layout."
        )

    def _kv_rung_verdict(self) -> str:
        """What the KV rung decided this round, for a REFUSAL message.

        A refusal is not an edge, so the rung's edge-triggered trace is
        typically silent at exactly the moment a reader needs it -- and a
        silent decline is indistinguishable from a rung that was never
        reached. Those have completely different fixes, and on 2026-08-15 the
        difference cost a wrong diagnosis: the seam was refused by 59 MiB, the
        rung had logged nothing for five minutes, and that was read as a
        missing call when the rung had in fact been consulted every gate and
        declined quietly.

        Never raises and never blocks the refusal it is decorating: an
        instrument that can break the message it rides on is worse than no
        instrument.
        """
        try:
            from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

            scheduler = getattr(self, "_census_scheduler", None)
            relief = (
                getattr(scheduler, KV_BACKING_RELIEF_ATTR, None) if scheduler else None
            )
            if relief is None:
                return "No KV rung is installed on this rank."
            return relief.last_proposal_summary()
        except Exception as e:  # noqa: BLE001 - decoration must not raise
            return f"(the KV rung's verdict could not be read: {e})"

    def _corridor_gate(
        self,
        staging_bytes: int,
        direction: str,
        slots_digest: int = 0,
        max_live_row: int = -1,
        slot_ballot_out: Optional[dict] = None,
    ) -> str:
        """#656 item 15a: spill BEFORE the seam allocates. Returns "" if clear.

        WHY HERE AND NOT AT ``commit_range``. The seam's commits happen
        inside the no-return region, and there is no try/except on that path
        by design -- ``_abandon_parked_flip`` states the law: a raise from
        inside a cutover climbs into the event loop and takes the instance
        down. So the gate is consulted at the last point where refusing is
        still free, and its refusal joins ``too_small``, which already rides
        the ``_collective_min`` that makes the abandon unanimous. A
        rank-local refusal would half-flip the group.

        The gate runs BEFORE ``_staging_affordable`` on purpose: the
        providers return pages to the DRIVER, so whatever the gate reclaims
        is money the affordability check then sees. Reversing the two would
        have the cheaper check refuse a flip the gate could have funded.

        A refusal here is not an error. It is the flip declining to start,
        with every request intact, which is exactly the outcome the 2026-08-09
        ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`` death should have had.

        #656 REGISTER C20, THE SEAM-ENTRY MARGIN. The gate asks for the
        staging PLUS ``seam_entry_margin_bytes()``, because clearing on the
        LAW alone is what let s34 enter a cutover with 19 MiB of margin and
        s36 enter the same one 23 MiB short. The margin is a TERM in the
        existing ask -- one ladder, one refusal path, no second mechanism --
        and it is graded, not absolute:

          margin met            -> the seam enters (the common case)
          margin short, law met -> the seam is DELAYED, up to a per-direction
                                   budget, because the paired-trough
                                   measurement says the memory comes back
          budget spent, law met -> the seam enters on the law, loudly. This
                                   is s34's shipped behaviour, so the worst
                                   case of the margin is the behaviour it
                                   replaces, never a wedge.
          law short             -> refused, as before, however exhausted the
                                   budget is. The law is never budgeted.

        SAY THE GUARANTEE EXACTLY. No path proceeds when the pre-allocation
        law check fails. That is not the same as "no breach is possible": the
        YIELD path deliberately enters at the law, which is precisely the
        state register C20 measured a 456 MiB in-cutover draw from. The yield
        is bounded to be no worse than the run it replaces, not to be safe --
        a run whose yields dominate its delays has a margin this
        configuration cannot fund, and that is a finding, not a pass.
        """
        scheduler = self._census_scheduler
        # Stub runtimes built with __new__ (the hermetic gate tests) do not
        # run __init__, so the C20 state is established defensively here
        # rather than assumed.
        if not hasattr(self, "_seam_abandons_in_a_row"):
            self._seam_abandons_in_a_row = {PP_TO_TP: 0, TP_TO_PP: 0}
        if not hasattr(self, "seam_margin_delays"):
            self.seam_margin_delays = 0
            self.seam_margin_yields = 0
            self.seam_yields_withheld = 0
        if not hasattr(self, "seam_draw_predicted_breaches"):
            self.seam_draw_predicted_breaches = 0
        if not hasattr(self, "_seam_draw_max"):
            self._seam_draw_max = {PP_TO_TP: 0, TP_TO_PP: 0}
        margin_bytes = seam_entry_margin_bytes()
        ask_bytes = int(staging_bytes) + margin_bytes
        # ANNOUNCE ONCE, FROM THE PATH THAT ALWAYS RUNS. The margin appears in
        # the guard's reason string, but the guard only logs when it ARMS, so
        # counting reasons in the log measures how expensive the term was and
        # not whether it is wired. Those are different questions and a run
        # that funds the margin from cache every time would answer the second
        # one "inert" -- the exact false negative this corpus has shipped
        # seven times, inverted.
        if not getattr(self, "_seam_margin_announced", False):
            self._seam_margin_announced = True
            logger.info(
                "%s [#656 SEAM-ENTRY] ARMED: C20 entry margin %d MiB on top of "
                "the seam staging, delay budget %d consecutive abandoned "
                "attempts per direction. The corridor LAW is unchanged; this "
                "term decides how much headroom a cutover must ENTER with, "
                "because the law alone let s34 enter with 19 MiB.",
                LOG_PREFIX,
                margin_bytes // (1024 * 1024),
                seam_entry_delay_budget(),
            )
        guard = None
        try:
            from sglang.srt.managers.phase_flip_spill import get_corridor_guard

            if scheduler is not None:
                guard = get_corridor_guard(scheduler)
        except Exception as e:
            logger.error(
                "%s corridor guard could not be built (%s); the flip proceeds "
                "on the staging affordability check alone",
                LOG_PREFIX,
                e,
            )
            guard = None

        # SPEC ITEM 12, DEVICE HALF, AND THE ONE PLACE IT MAY HAPPEN.
        #
        # The KV cap changes admission, so it must be the SAME row count on
        # every rank or the group desyncs (HANDOFF_675 §1a). That needs a
        # reduction, and a reduction needs a call site every rank reaches
        # UNCONDITIONALLY -- which the guard's providers are not, since they
        # run behind its rank-local arm condition. This line is on the same
        # unconditional path as the fit reduction that follows it.
        #
        # IT SITS OUTSIDE EVERY EARLY RETURN ABOVE, INCLUDING ``guard is
        # None``, and that placement is the whole safety argument: a rank that
        # skipped the reduction because its own guard failed to build would
        # leave its peers inside a collective forever. A rank with no guard
        # ABSTAINS instead, which makes the group decline -- a lost
        # optimisation, never a hang.
        #
        # BEFORE the guard's own verdict, so what it frees is money the guard
        # can see.
        kv_freed = 0
        try:
            from sglang.srt.managers.phase_flip_spill import (
                collective_kv_backing_relief,
            )

            # The rung is asked to fund the MARGIN as well as the staging.
            # Its deficit is floor + delta + want - free - cheap_relief, so a
            # ``want`` that excluded the margin would have the funder of last
            # resort decline exactly the gap the gate is about to delay for.
            #
            # BUT THE MARGIN IS DECLARED DISCRETIONARY, and that distinction is
            # register C20's residual 1. The rung spends ADMISSION CAPACITY to
            # pay, and an unbounded ask made it spend all of it on every seam:
            # at 8192 MiB the deficit could never be closed, the rung went to
            # its floor 42 times, and the instance died in the scheduler loop
            # with ``available_size() == 0``. The margin's shortfall has a
            # graded answer (delay, then yield) and the staging's does not, so
            # the margin is the half that may be bounded and the staging is not.
            # #688: AN IDLE-LOCKED FLIP HAS NO MARGIN TO PROTECT. The C20
            # entry margin is discretionary because the rung pays for it out
            # of ADMISSION CAPACITY, and an unbounded ask once drove the rung
            # to its floor 42 times. But this flip was armed precisely because
            # nothing can be ADMITTED in this layout, so the capacity that
            # bound protects cannot be spent by anyone -- and leaving the
            # margin unfunded is exactly what leaves the seam short
            # (09:43:11Z: 1706 MiB needed, 1635 spendable, 71 MiB short).
            # Mandatory in that one state, discretionary everywhere else.
            discretionary_margin = (
                0
                if bool(getattr(self, "armed_idle_locked", False))
                else int(margin_bytes)
            )
            kv_freed = collective_kv_backing_relief(
                scheduler,
                self._collective_min,
                want_bytes=int(ask_bytes),
                guard=guard,
                direction=direction,
                discretionary_bytes=int(discretionary_margin),
                # #656 C22-d rides here too. See _agree_live_slots.
                slots_digest=int(slots_digest),
                max_live_row=int(max_live_row),
                slot_ballot_out=slot_ballot_out,
            )
        except Exception as e:
            logger.error(
                "%s collective KV backing relief failed (%s); the gate "
                "continues without it",
                LOG_PREFIX,
                e,
            )
        # #656 C22 NOTE: the KV cap AGREEMENT rides the very same reduction as
        # the shrink above (its payload is 8 fields, not 4). ``recover`` on the
        # tp->pp leg is bounded by each rank's own distance from the corridor
        # law, so the rank nearest the law comes back from a phase with fewer
        # rows than its peers, its live-slot enumeration differs by exactly
        # that many, and the frame ballot below refuses every subsequent flip
        # -- measured as a 40404-row divergence on rank 1 that wedged decode's
        # leg. Closing it HERE, before ``_frame_digest`` runs in this same
        # round, is what stops the frame tripping over it. No second
        # collective: the count is diffed across ranks by the census.
        # #657 item 16, the REBALANCE tier: agree on where the NEXT phase's
        # new KV rows should be placed. It runs here and not on the round
        # clock because the decision needs a REDUCTION -- its input is NVML,
        # which is rank-local, and a free list ordered differently on two
        # ranks would split one token's KV across two rows. This is the one
        # point every rank reaches unconditionally with a bounded collective
        # already in hand.
        #
        # It adds NO collective when steering is off, which is the shipped
        # default: the builder returns None before any reduction is entered.
        from sglang.srt.managers.corridor_steering import steer_at_seam

        steer_at_seam(scheduler, self._collective_min)

        if kv_freed > 0:
            self.corridor_kv_relief_bytes += int(kv_freed)
            self.corridor_kv_relief_count += 1
            logger.info(
                "%s KV backing relief returned %.0f MiB before the gate (%s), "
                "on a row target every rank agreed to",
                LOG_PREFIX,
                kv_freed / (1024 * 1024),
                direction,
            )
        else:
            # A RUNG THAT RETURNS NOTHING MUST STILL SAY SO.
            #
            # This branch used to be absent, and the silence cost a whole
            # morning of misdiagnosis on 2026-08-16. At 06:47:48 the seam was
            # refused 76 times with no relief line anywhere in the log, so
            # "the rung returned 0" and "the rung never ran" looked identical
            # -- and the guard's own "reclaimed 0 MiB from [nothing]" was then
            # read as the rung being exhausted. It never said that: that
            # string is the GUARD LADDER's provider list, which contains only
            # allocator-cache and draft-weights. No KV provider is registered
            # with the guard at all, by design (the cap is a group decision
            # and the ladder is rank-local), so the rung's bytes arrive as
            # `kv_freed` BEFORE the probe and can never appear in that list.
            #
            # Logged at the same level as the success so a seam's funding
            # story is one grep, and carrying the rung's own view of why it
            # could not pay -- the three causes are not interchangeable:
            # the floor is a healthy limit, an empty evictable set means the
            # pool is genuinely live, and a disqualified rung is a defect.
            from sglang.srt.managers.phase_flip_spill import (
                KV_BACKING_RELIEF_ATTR as _RUNG_ATTR,
            )

            rung = getattr(scheduler, _RUNG_ATTR, None) if scheduler else None
            logger.info(
                "%s KV backing relief returned NOTHING before the gate (%s): "
                "rung=%s, evicted %s rows over %s shrinks so far. This is not "
                "the guard's '[nothing]' -- no KV provider is registered with "
                "the guard, so the rung never appears in its provider list. "
                "Check in order: the admission floor (healthy), an empty "
                "evictable set (the pool is genuinely live), a disqualified "
                "rung (a defect).",
                LOG_PREFIX,
                direction,
                "absent" if rung is None else "present",
                getattr(rung, "evicted_rows_total", "?"),
                getattr(rung, "evict_count", "?"),
            )
        if guard is None:
            return ""
        try:
            verdict = guard.ensure_headroom(
                int(ask_bytes),
                reason=(
                    f"seam staging {direction}"
                    + (
                        f" +{margin_bytes // (1024 * 1024)} MiB C20 entry margin"
                        if margin_bytes
                        else ""
                    )
                ),
                # THE TWO LEGS ARE NOT SYMMETRIC. Refusing tp->pp is
                # survivable: the instance stays in TP, decode keeps running,
                # prefill defers. Refusing pp->tp is not -- strict purity
                # forbids decode in PP, so it starves decode outright and
                # nothing in PP can free the memory that would end it. So the
                # pp->tp leg is allowed to spend host RAM ahead of levelling
                # (spec item 15c: the price is tempo, never a corridor
                # breach), and the guard counts every time it has to.
                refusal_is_fatal=(direction == "pp_to_tp"),
            )
        except Exception as e:
            # The gate is a safety net, not a dependency. A net that tears
            # must not take down the thing it was protecting, so a broken
            # gate degrades to the pre-gate behaviour -- which is the
            # affordability check on the next line -- and says so loudly.
            logger.error(
                "%s corridor gate failed to evaluate (%s); the flip proceeds "
                "on the staging affordability check alone",
                LOG_PREFIX,
                e,
            )
            return ""
        if verdict.ok:
            if verdict.reclaimed > 0:
                self.corridor_reclaims += 1
                logger.info(
                    "%s corridor gate funded the seam: %s",
                    LOG_PREFIX,
                    verdict.detail,
                )
            return ""

        # #656 C20. THE REFUSED ASK INCLUDED THE MARGIN, so it does not yet
        # say WHICH of the two events this is. Answer that from the verdict's
        # OWN numbers instead of asking the guard a second time. The guard's
        # contract is
        #
        #     ok = (free_after_the_ladder - want) >= law_floor_bytes
        #
        # (corridor_guard.py), so subtracting the STAGING from the free the
        # ladder actually reached asks exactly the question a second call
        # would answer.
        #
        # THE SECOND CALL WAS THE FIRST VERSION OF THIS AND IT WAS WRONG
        # TWICE. On the refusal path it re-armed the entire ladder -- a
        # second empty_cache, a second forced host spill, every counter
        # double-booked -- and an exception inside it discarded a verdict
        # that had ALREADY said the law would break, returning "" and walking
        # the seam into the breach this gate exists to prevent. Arithmetic on
        # a value already in hand cannot raise and cannot spend.
        law_floor = int(getattr(guard, "law_floor_bytes", 0))
        # PRICE THE LAW ON THE DRAW, NOT ON THE STAGING. #656, 2026-08-12.
        #
        # ``staging_bytes`` is what the seam RESERVES. It is not what the
        # cutover TAKES from the driver: the backing restore walk commits raw
        # driver pages while torch's caching allocator is still sitting on its
        # own reserve, so the driver-visible draw runs materially above the
        # staging figure. Both windows in the corpus yielded on a law check
        # that was sub-law by its own numbers once the real draw is used:
        #
        #     s38  free_after 2190 - staged 1184 = 1006  (18 MiB under the law)
        #     s42  free_after 3154 - staged 1625 = 1529  (looks clear)
        #          ...but the census measured a 2066 MiB draw, and the cutover
        #          entered at 3006 and troughed at 940.
        #
        # s38 survived only because its staging OVER-estimated that flip's
        # draw by ~388 MiB. That is luck, not a margin, and it is the same
        # shape as C20 one level up: a check that passes on an estimator's
        # conservatism rather than on the quantity it claims to bound.
        #
        # So the subtrahend becomes max(staged, WORST MEASURED DRAW for this
        # direction). Until this rank has completed one cutover in this
        # direction the measured term is 0 and the behaviour is exactly the
        # old one -- an unmeasured bucket is never a licence to invent a
        # number (the ``measured_capture_mib_per_token`` rule).
        #
        # GROUP SAFETY: this stays a per-rank OBJECTION, not a per-rank
        # ACTION. ``law_ok`` already rides the existing reduction -- the group
        # abandons if ANY rank objects -- so a rank whose measured draw is
        # larger makes the GROUP delay, which every rank then observes
        # identically. No rank acts on its own number (register laws 14, 15).
        measured_draw = 0
        try:
            book = getattr(self, "_seam_draw_max", None)
            if isinstance(book, dict):
                measured_draw = int(book.get(str(direction), 0) or 0)
        except Exception:  # noqa: BLE001 -- arithmetic here may not raise
            measured_draw = 0
        law_want = max(int(staging_bytes), measured_draw)
        predicted_trough = int(verdict.free_after) - law_want
        draw_short = measured_draw > 0 and predicted_trough < law_floor
        # #662: THE SEAM TRANSIENT IS JUDGED AGAINST THE BAND FLOOR, second
        # site of the same rule as the staging reserve (5658c9683f).
        #
        # The corridor is 1024 MiB +-20 % and its verdict is the continuous
        # minimum against the FLOOR, so a cutover that dips to 819 for the
        # length of a wave walk is lawful. Comparing against the CENTRE here
        # delayed flips the law permits: measured 2026-08-15, want 2251 MiB
        # against free 3206 leaves 955 -- which is 69 MiB under the centre and
        # 136 MiB CLEAR of the floor -- and 21,480 tokens waited ~35 s for a
        # flip that was legal the whole time.
        #
        # AND THE MARGIN WAS NEVER IN THIS COMPARISON. ``margin_bytes`` is an
        # enable flag here, nothing more, yet the delay message named it -- so
        # the log said "the 512 MiB C20 entry margin is not met" about an
        # arithmetic in which 512 appears nowhere. That is why this read as a
        # double-counted arming floor from outside; the arming floor is not in
        # it either. The message below now names the number that actually
        # bound.
        seam_floor = _seam_transient_floor_bytes(law_floor)
        law_ok = margin_bytes > 0 and (
            int(verdict.free_after) - int(staging_bytes) >= seam_floor
        )
        # WHY THE MEASURED DRAW DOES NOT MOVE ``law_ok``, WHICH IS THE
        # OBVIOUS THING TO DO AND IS WRONG HERE.
        #
        # A False ``law_ok`` routes into the abort path below, and for
        # pp->tp that path is the DECODE WEDGE: under strict purity decode
        # runs only in TP, so a persistently refused pp->tp starves decode
        # outright and nothing the PP phase holds can ever end the refusal.
        # It is measured, not feared -- 411 abandons, 0 requests completed in
        # 6 minutes, /health 503 with every rank alive (2026-08-10). Trading
        # a 1.5 s corridor dip for a total outage is not a fix, and register
        # law 13 says the same thing from the other side: relief that arrives
        # eventually cannot fund an allocation happening now.
        #
        # So the measured draw does what it can legitimately do: it PREDICTS
        # the trough and says so, and the walk itself is what keeps the law,
        # by spending torch's cache at the moment of crossing
        # (``kv_vmm_backing._corridor_preempt``). That resource was measurably
        # there -- 1054 MiB of slack at the 940 MiB trough -- and needed no
        # prediction at all to spend. Prediction that cannot act safely is a
        # log line; the actuator belongs where the bytes are taken.
        if draw_short:
            self.seam_draw_predicted_breaches += 1
            logger.warning(
                "%s seam entry PREDICTS A SUB-LAW TROUGH (%s): %d MiB free "
                "after the ladder minus this rank's worst MEASURED draw of "
                "%d MiB leaves %d MiB, below the %d MiB corridor law. The "
                "staged figure alone (%d MiB) puts it %d MiB CLEAR, which is "
                "the gap every previous yield entered on. The seam is NOT "
                "refused -- refusing pp->tp starves decode -- so the walk "
                "carries the law instead: the arena spends torch's cached "
                "blocks at the commit that would cross. If a breach is "
                "recorded anyway, the walk had no slack to spend and the "
                "budget is the finding.",
                LOG_PREFIX,
                direction,
                int(verdict.free_after) // (1024 * 1024),
                measured_draw // (1024 * 1024),
                predicted_trough // (1024 * 1024),
                law_floor // (1024 * 1024),
                int(staging_bytes) // (1024 * 1024),
                (int(verdict.free_after) - int(staging_bytes) - law_floor)
                // (1024 * 1024),
            )
        if law_ok:
            budget = seam_entry_delay_budget()
            # GROUP-UNIFORM CURRENCY. The budget is spent in consecutive
            # ABANDONED ATTEMPTS, booked by ``note_seam_verdict`` from the
            # already-reduced fit verdict, so all three ranks read the same
            # number. A per-rank counter reset by each rank's own clearance
            # was the first version and it could not bound anything: the
            # group abandons if ANY rank objects, so three ranks taking turns
            # being short never spend a budget between them and pp->tp delays
            # forever -- the 411-abandon decode wedge, reached through the
            # mechanism that exists to prevent it.
            spent = self._seam_abandons_in_a_row.get(direction, 0)
            if spent < budget:
                self.seam_margin_delays += 1
                logger.warning(
                    "%s seam entry DELAYED (%s): staging would leave less than "
                    "the %d MiB seam floor (the corridor band's lower edge, "
                    "which is what a cutover transient is judged against) "
                    "(%s). The deepest "
                    "troughs of this corpus are made INSIDE a cutover and the "
                    "level recovers between them -- consecutive abandoned "
                    "attempts %d of a budget of %d, after which the law "
                    "governs.",
                    LOG_PREFIX,
                    direction,
                    margin_bytes // (1024 * 1024),
                    verdict.detail,
                    spent,
                    budget,
                )
                return (
                    f"{SEAM_MARGIN_DELAY_TAG}: {verdict.detail} "
                    f"(attempt {spent + 1}, budget {budget}, {direction})"
                )
            # #656 AXIS 3: THE YIELD MAY NOT ENTER A TROUGH THIS RANK HAS
            # ALREADY MEASURED.
            #
            # The yield is the one path that deliberately enters at the law,
            # and it is where every corridor breach in this corpus was made --
            # the acceptance's five, and the remediation boot's one remaining
            # 12 MiB dip, which the 100 ms trace puts at 14:42:25 inside a
            # tp_to_pp cutover at stage 'weights_refill', three seconds after
            # this rank yielded. Not a prefill transient: the same mechanism.
            #
            # WHY THIS IS NOW SAFE, WHEN THE PREDICTION ABOVE STILL REFUSES TO
            # ACT. That comment's premise was "refusing pp->tp starves decode
            # outright", and it was true: under strict purity decode runs only
            # in TP. It is no longer true. The purity stand-down valve
            # (phase_purity._relaxed) lets decode run in the PP layout once
            # pp_to_tp has been abandoned a few rounds running, so a withheld
            # flip now costs THROUGHPUT rather than the instance. The corridor
            # law is a hard user limit; the decode layout is a performance
            # choice; this trade goes the other way from the one that comment
            # refused.
            #
            # AND IT STAYS A DELAY. The objection carries the margin-delay
            # tag, which is exempt from the seam-abandon cap, so the flip is
            # not stood down for good over a condition that clears itself:
            # once decode drains in the degraded layout the memory comes back
            # and the very next round enters with room.
            if draw_short:
                # USER DECISION 2026-08-16: WARN, DO NOT WITHHOLD.
                #
                # This branch used to return a margin-delay tag, which is
                # EXEMPT from the stand-down cap by design -- so when the
                # condition did not clear, nothing bounded it. Measured at
                # 07:02:15: PP1 withheld on a predicted 864 MiB trough while
                # PP2 yielded, the ranks disagreed, "consecutive delayed
                # attempts" climbed 15/16/17 with no exit, and the instance
                # sat at bs 0, GPU 0%, with 794179 tok pending.
                #
                # Its safety argument was also self-defeating: it justified
                # waiting by pointing at the purity valve, but the valve opens
                # on the stand-down cap that this very tag is exempt from.
                #
                # The law is a fill-quality target, not a gate. A predicted
                # dip is now SAID and stepped over. The prediction keeps its
                # job -- it sizes the warning, and it is what the pre-flip
                # spill rung aims at -- it just no longer stops the machine.
                self.seam_law_warned = getattr(self, "seam_law_warned", 0) + 1
                logger.warning(
                    "%s CANNOT FULLY HOLD THE CORRIDOR FLOOR through this "
                    "seam entry (%s): predicted trough %d MiB below the %d "
                    "MiB law -- this rank's worst MEASURED draw of %d MiB "
                    "against %d MiB free. PROCEEDING ANYWAY: the law is a "
                    "fill-quality target and only OOM is hard (user decision "
                    "2026-08-16). The seam is no longer delayed on a margin "
                    "prediction; an idle instance was never the cheaper side "
                    "of this trade.",
                    LOG_PREFIX,
                    direction,
                    predicted_trough // (1024 * 1024),
                    law_floor // (1024 * 1024),
                    measured_draw // (1024 * 1024),
                    int(verdict.free_after) // (1024 * 1024),
                )
            self.seam_margin_yields += 1
            logger.warning(
                "%s seam entry margin YIELDED (%s) after %d consecutive "
                "abandoned attempts: entering on the corridor law alone, "
                "which is the pre-C20 behaviour, so this is never worse than "
                "the run it replaces -- but it enters at the law and is "
                "therefore exposed to the in-cutover draw register C20 "
                "measures. %s. A run whose yields dominate its delays has a "
                "margin this configuration cannot fund: read that as "
                "evidence, not as a reason to widen it.",
                LOG_PREFIX,
                direction,
                spent,
                verdict.detail,
            )
            return ""

        self.corridor_aborts += 1
        # A REFUSED pp->tp IS NOT A TRANSIENT, and the two directions are not
        # symmetric. Refusing tp->pp is safe: the instance stays in TP, decode
        # keeps running, prefill defers. Refusing pp->tp under strict purity
        # starves DECODE COMPLETELY -- decode is forbidden in PP, so requests
        # prefill and then wait forever, and nothing the PP phase can do frees
        # the memory that would end the refusal. Measured 2026-08-10 with a
        # deliberately raised arming floor: 411 abandons, 0 requests completed
        # in 6 minutes, /health 503 while every rank was alive and logging.
        #
        # The gate must never be the silent cause of that. It cannot fix it
        # either -- the answer is a provider that can fund the seam (item 15c,
        # kvso over the host tier) -- so it says exactly what is happening,
        # once per escalation rather than once per flip.
        if direction == "pp_to_tp":
            self._corridor_pp_refusals += 1
            n = self._corridor_pp_refusals
            if n in (1, 10) or (n % 100 == 0):
                logger.error(
                    "%s corridor gate has refused pp->tp %d time(s) in a row. "
                    "Under strict purity decode runs ONLY in TP, so a "
                    "persistently refused pp->tp means DECODE IS STARVED and "
                    "no amount of waiting will clear it: the PP phase holds "
                    "nothing that would fund the TP seam. This needs a "
                    "provider that can (spec item 15c, kvso over the host "
                    "tier), a smaller pool, or a lower arming floor. Verdict: "
                    "%s",
                    LOG_PREFIX,
                    n,
                    verdict.detail,
                )
        else:
            self._corridor_pp_refusals = 0
        return f"corridor gate refused the seam staging: {verdict.detail}"

    def _seam_funding_verdict(self, staging_bytes: int, direction: str, **kw) -> str:
        """The gate's verdict, with #662-F4's per-direction injection on top.

        WHY THIS WRAPS THE GATE INSTEAD OF LIVING INSIDE IT. Everything
        ``_corridor_gate`` runs -- the KV rung's reduction, the guard's
        ladder, the C20 margin -- sits on a path every rank reaches
        unconditionally, which is the entire argument
        ``collective_kv_backing_relief`` makes for living where it lives. An
        injection that short-circuited the gate would skip a reduction the
        peers had already entered, the first time one rank read the variable
        differently, and this file's standing rule is that "safe because a
        value upstream is agreed" is precisely the reasoning that produces
        collective hangs. So the gate always runs in full, and only its
        VERDICT is overridden. The cost is one ladder that spent for nothing;
        what it buys is that no rank can skip a collective.

        The result joins ``too_small`` like any other objection, so the
        abandon it produces is the real one: the same unanimous MIN, the same
        FLIP ABANDONED log, the same per-direction hold and backoff. What the
        gate itself decided is kept in the message, because a proof that hides
        the state it overrode is not evidence.
        """
        from sglang.srt.managers.phase_flip_spill import seam_unfundable_objection

        detail = self._corridor_gate(staging_bytes, direction, **kw)
        injected = seam_unfundable_objection(direction)
        if not injected:
            return detail
        return f"{injected} [the gate itself said: {detail or 'the seam was fundable'}]"

    # -- the move -------------------------------------------------------------
    def _pack_outgoing(
        self,
        tr: PhaseFlipTransition,
        direction: str,
        src,
        peer: int,
        layers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """One peer's wire payload, streamed into a single exact-size buffer.

        ``layers`` restricts the payload to one seam WAVE's ordinals (#631);
        None is the whole pair, which is what a single-wave seam asks for.

        BYTE-FOR-BYTE what ``torch.cat([*per_layer_reads, checksum])``
        produced -- layers ascending, each layer's rows row-major, K bytes
        then V, the int64 checksum in the last 8 bytes. The receiver walks
        it back with ``offset += n * width`` in the same layer order, so
        the wire format is a contract between this function and that loop
        and it has not moved. ``test_streamed_pack_equals_the_
        concatenation_reference`` pins it against the old expression.

        What changed is the LIVE SET, and that is the entire reason this
        function exists. The concatenating form held, at its peak, three
        copies of one peer's payload: the per-layer ``parts`` are still
        referenced when ``flat = torch.cat(parts)`` exists, and ``flat`` is
        still referenced while the checksum-appended copy is built. That
        transient scales with the resident sequence -- the live set is
        every resident request's ``req_to_token[:seqlen]`` -- which is what
        made a chunked prefill cost memory proportional to the WHOLE
        prompt, breach the 1024 MiB corridor, and then starve this flip's
        own affordability gate into a livelock (HANDOFF_664 sections 12-13:
        the corridor bound and the livelock bound are the same allocation,
        which is why ``--max-prefill-tokens`` could never bound either).

        Here the payload is written once, in place. The only transient is
        one layer's gather inside ``read_rows_into`` -- a fixed fraction
        1/(2L) of the payload, dead before the next layer's -- so the
        staging window is bounded by the geometry instead of by the prompt.
        """
        layers = tr.send_layers[peer] if layers is None else tuple(layers)
        rows = tr.send_rows[peer]
        n = int(rows.numel())
        widths = [src.row_nbytes(self._src_layer_idx(direction, f)) for f in layers]
        payload_nbytes = sum(w * n for w in widths)
        buf = torch.empty(
            payload_nbytes + _CHECKSUM_BYTES, dtype=torch.uint8, device=src.device
        )
        offset = 0
        for f, width in zip(layers, widths):
            seg = buf[offset : offset + n * width].view(n, width)
            src.read_rows_into(self._src_layer_idx(direction, f), rows, seg)
            offset += n * width
        payload = buf[:payload_nbytes]
        buf[payload_nbytes:].copy_(_checksum(payload))
        return buf

    def _pack_local(
        self,
        tr: PhaseFlipTransition,
        direction: str,
        src,
        local_src: torch.Tensor,
        layers: Optional[Sequence[int]] = None,
    ) -> List[torch.Tensor]:
        """The retained leg, streamed into one buffer, as per-layer views.

        ``layers`` restricts it to one seam WAVE's ordinals (#631).

        The local rows must ALL be read before the first destination write
        -- the reads-before-writes invariant below, which the cross-phase
        backing swap makes load-bearing rather than theoretical -- so this
        leg is genuinely irreducible at one copy. It is still worth
        streaming: one allocation instead of L, and no per-layer
        concatenation of the K and V halves on top of the gather.
        """
        n = int(local_src.numel())
        local_layers = tr.local_layers if layers is None else tuple(layers)
        widths = [
            src.row_nbytes(self._src_layer_idx(direction, f)) for f in local_layers
        ]
        buf = torch.empty(
            sum(w * n for w in widths), dtype=torch.uint8, device=src.device
        )
        out: List[torch.Tensor] = []
        offset = 0
        for f, width in zip(local_layers, widths):
            seg = buf[offset : offset + n * width].view(n, width)
            src.read_rows_into(self._src_layer_idx(direction, f), local_src, seg)
            out.append(seg)
            offset += n * width
        return out

    def _row_bounds_detail(self, tr: "PhaseFlipTransition") -> List[str]:
        """Do BOTH pools cover the rows this plan touches? One list, no side
        effects, so it can be recomputed after the live-slot agreement has
        changed which rows the plan touches (#656 C22-d)."""
        detail: List[str] = []
        if tr.max_pp_row() >= self._pp.num_rows:
            detail.append(
                f"PP row {tr.max_pp_row()} vs pool {self._pp.num_rows} rows "
                f"(the PP pool must cover every live global slot id)"
            )
        if tr.max_tp_row() >= self._tp.num_rows:
            detail.append(
                f"TP row {tr.max_tp_row()} vs pool {self._tp.num_rows} rows "
                f"(the TP pool must cover the compact rows of vector "
                f"{self._vec})"
            )
        return detail

    def _agree_live_slots(self, slots: torch.Tensor, ballot: dict):
        """#656 C22-d: make the ranks frame the SAME live slot SET.

        WHAT #656 C22 CLOSED AND WHAT IT LEFT OPEN. The cap agreement
        (``phase_flip_spill.apply_cap_agreement``) levels how MANY rows every
        rank exposes, in the rung reduction that runs before ``_frame_digest``
        in the same round. Boot ``boot_m3``, 2026-08-13, showed that is not
        enough: the cap agreement fired (-15455 rows, then -77), the three
        POOL CENSUS lines matched, and PP1 still framed a different digest in
        four consecutive ``pp_to_tp`` episodes -- ``THE DIVERGING TERM IS: the
        live slot set``. After 194 clean cutovers the instance never flipped
        again. Levelling how many rows a rank exposes does not make the ranks
        hold the SAME rows.

        THE TRIGGER, NAMED. ``KvRowCap.release`` SORTS the free list;
        ``engage`` preserves whatever order it found. A corridor-bounded
        ``recover()`` therefore reorders the recovering rank's free list while
        its peers, which never shrank, keep theirs in eviction order. The only
        thing that re-normalises the group is ``reconcile_to``'s final sort --
        and ``collective_cap_target`` returns ``None``, skipping it, exactly
        when the exposed counts already AGREE. So a recovery whose row
        shortfall is levelled (leg C's single 23199-row episode: the next
        shrink caps every rank to one absolute target, by construction) opens
        no divergence, while a recovery that leaves the counts EQUAL and the
        ORDER different opens one that nothing closes. That asymmetry is why
        the trigger is narrower than "a rank came back short", and it is why
        the source-side repair (``normalize_free_lists``, run unconditionally
        at the rung) is only half the answer: it stops NEW divergences and
        cannot undo rows already handed out to live requests and cached in the
        radix tree, which is why draining the instance and ``/flush_cache``
        both failed to clear boot 1's wedge.

        SO THIS IS THE RECOVERY HALF. When the rung's ballot says the ranks
        enumerate different sets, every rank -- unanimously, because the
        verdict is read off a value the reduction produced identically
        everywhere -- adopts the group's UNION. The union is the only
        reconciliation that is safe in both directions: a rank-0-authoritative
        broadcast would drop a peer's live rows from the plan and lose that
        request's KV at the seam, and an intersection would do the same to
        everyone. A union never removes a row from the rank that holds it, so
        no request loses context, and it never asks a rank to give up backing,
        so it cannot breach the corridor law the recovery was bounded by.

        IT COSTS NOTHING WHEN THE RANKS AGREE, which is the shipped case: 1134
        consecutive cutovers on boot 2 with zero divergences would enter this
        function and return at the first branch. The detection is four extra
        integers on the rung's existing reduction (8 fields -> 12, the same
        move R2 made from 4 -> 8). Only the repair itself adds a reduction,
        and only on the round that needs one.

        WHY THE UNION DOES NOT DISTURB THE ROWS EVERY RANK ALREADY AGREED ON.
        ``reshard_plan.rows_of`` maps a slot to its compact TP row as
        ``(slot // s) * ratio + (slot % s - lo)`` -- a pure function of the
        SLOT ID and the vector, not of the slot's position in the enumeration.
        Adding rows to the framed set therefore moves no other row's
        destination; it only adds copies. The extra rows carry whatever the
        holder's pool holds and nothing reads them on a rank that has no
        request behind them.

        THE ONE BOUND THAT IS NOT NEGOTIABLE. A row id at or above a rank's
        BACKED row count is not mapped on that rank, and the mover reading it
        is ``cudaErrorIllegalAddress``, which kills every rank rather than
        raising. The union is therefore refused when it reaches the group's
        MINIMUM backing (carried in the same four fields), and the refusal
        joins ``too_small`` -- a free, unanimous, announced abandon, with the
        rung's levelling and the free-list normalisation still running every
        round underneath it, so the offending rows drain as their requests
        finish and the next round can agree.

        Returns ``(slots, detail)``: the set to frame, and "" or one
        ``too_small`` entry.
        """
        if not ballot or ballot.get("agree") is not False:
            return slots, ""
        self.slot_set_divergences = getattr(self, "slot_set_divergences", 0) + 1
        span = int(ballot.get("max_live_row", -1)) + 1
        if span <= 0:
            return slots, ""
        # THE UNION, ON THE MIN CHANNEL THE GROUP ALREADY HAS. A membership
        # vector of -1/0 reduced with MIN yields -1 wherever ANY rank holds
        # the row, which is the OR this channel cannot express directly. The
        # same [x, -x] inversion the fit verdict uses, one field per row id.
        presence = torch.zeros(span, dtype=torch.int64)
        local = slots[slots < span]
        if local.numel():
            presence[local] = -1
        reduced = self._collective_min(presence.tolist())
        union = (
            (torch.tensor(reduced, dtype=torch.int64) < 0)
            .nonzero()
            .flatten()
            .to(torch.int64)
        )
        min_backed = int(ballot.get("min_backed_rows", 0))
        highest = int(union[-1].item()) if union.numel() else -1
        if highest >= min_backed:
            self.slot_set_refusals = getattr(self, "slot_set_refusals", 0) + 1
            return slots, (
                f"live slot set divergence cannot be repaired this round: the "
                f"group's union reaches row {highest} and the poorest rank has "
                f"only {min_backed} rows BACKED, so framing the union would "
                f"have that rank read unmapped memory. This rank enumerates "
                f"{int(slots.numel())} rows, the union has {int(union.numel())}. "
                f"The rung's cap agreement and free-list normalisation run "
                f"every round underneath this, so the rows above the group's "
                f"backing drain as their requests finish"
            )
        self.slot_set_agreements = getattr(self, "slot_set_agreements", 0) + 1
        added = int(union.numel()) - int(slots.numel())
        logger.warning(
            "%s live slot SET agreed by union: this rank enumerated %d rows, "
            "the group holds %d (%+d), digests spanned [%s, %s]. The count "
            "agreement (#656 C22) had already levelled the group; agreeing "
            "WHICH rows is what stops the frame ballot refusing every "
            "subsequent flip. No rank gives up a row it holds, so no request "
            "loses context and no backing is released",
            LOG_PREFIX,
            int(slots.numel()),
            int(union.numel()),
            added,
            ballot.get("digest_lo"),
            ballot.get("digest_hi"),
        )
        return union, ""

    def _execute(self) -> Optional[dict]:
        direction = self._pending
        assert direction is not None
        t0 = self._clock()
        # #631 seam census: per-STAGE memory attribution across this cutover.
        # The aggregate cost was measured from outside the process; which
        # STAGE spends it was not, and the candidates have different fixes.
        seam_census.begin(direction, self._rank)
        # #703: push warm prefixes to the geometry-neutral store BEFORE
        # anything moves. Ordering is the whole content of the hook: the radix
        # cache is bound to the pool that BUILT it (the boot PP stack's), so
        # this is the last moment at which reading that pool reads live bytes.
        # After the cutover the same copy would read a pool the model is no
        # longer writing into.
        #
        # NEVER RAISES INTO THE FLIP. The configuration errors this hook cares
        # about -- no storage tier, or a store without the #706 canonical
        # format -- are refused at PARSE time by ServerArgs, where refusing is
        # free. Here, with requests already parked, a raise climbs into the
        # event loop and takes down an instance that was serving fine in its
        # current phase, which is the lesson the bounds check below learned the
        # expensive way. A prefix that misses the store costs a later cache
        # miss; that is the cheaper failure, and it is logged, not thrown.
        try:
            from sglang.srt.mem_cache.hicache_flip_writeback import (
                maybe_flip_writeback,
            )

            if maybe_flip_writeback(getattr(self, "_census_scheduler", None)):
                seam_census.mark("flip_writeback")
        except Exception as e:
            logger.error(
                "%s #703 flip-time writeback did not run (%s); the flip "
                "continues and prefixes written in this phase will miss in "
                "the next one.",
                LOG_PREFIX,
                e,
            )
        slots = self._live_slots_fn()
        slots = torch.unique(slots.detach().to("cpu", torch.int64))
        tr: PhaseFlipTransition = build_phase_flip_transition(
            slots, self._map, self._n_layers, self._vec, self._rank, direction
        )

        src, dst = self._src_dst(direction)
        # Bounds BEFORE any byte moves, and GROUP-AGREED before acting on
        # them. Both pools are pre-sized at boot, but whether the live set
        # FITS is a runtime quantity: it grows with the resident prefix
        # cache, and the TP pool shrank once speculation put a draft KV
        # allocation inside the same budget (#631 window 3, boot 19 --
        # "needs TP row 10896 but the TP pool holds 7719").
        #
        # Two things were wrong with raising here. First, the reading is
        # RANK-LOCAL (each rank has its own pool sizes and its own compact
        # rows), so a rank that raised while a peer proceeded would leave
        # the group half-flipped -- the same rank-local-state-feeds-
        # collective shape this file keeps having to fix. Second, raising
        # climbs into the event loop and takes the INSTANCE down; it killed
        # a healthy server that was serving fine in its current phase.
        #
        # Nothing has been mutated at this point -- the transition is a
        # plan, not a move -- so the safe answer is unanimous: reduce the
        # local verdict, and if ANY rank does not fit, every rank abandons
        # the flip and keeps serving. The flip is the optional thing here.
        # SECOND TERM OF THE SAME VERDICT: can the bytes be STAGED at all?
        # The rows fitting in the target pool says nothing about the memory
        # needed to carry them there. The exchange packs this rank's
        # outgoing rows and allocates one receive buffer per peer, and it
        # takes them from the same device memory the KV pool has been
        # filling. Measured (2026-08-09): a single 276214-token session at
        # 0.995 pool occupancy left 600 MiB free, the exchange asked for
        # 584 MiB, and the OutOfMemoryError climbed out of
        # kv_reshard._exchange into the event loop and killed all three
        # ranks -- the exact "raising here takes the INSTANCE down" failure
        # the paragraph above refuses for the row bounds.
        #
        # It belongs in THIS verdict, not in a check of its own: the sizes
        # are known from the plan before a single byte is allocated, and
        # the answer when they do not fit is the same unanimous abandon.
        # A rank-local abandon would half-flip the group.
        # BEFORE the price, not after: the reservation and the loop must be
        # computed from the SAME block count or the gate is pricing a seam
        # that is not the one about to run.
        self._refresh_seam_tuning()
        waves = self._flip_waves(direction)
        staging_bytes = self._staging_bytes(tr, direction, src, dst, waves)
        # #656 item 15a/16: spill before the allocation, and level the cards
        # before touching host RAM. Runs first so its reclaim is money the
        # affordability check below can see.
        #
        # #656 C22-d RIDES THIS SAME RUNG. Four more integers on the reduction
        # the gate already runs (12 fields, not 8) carry the live slot SET
        # ballot -- the membership digest, the group's row extent, and the
        # group's minimum BACKED rows -- so a set divergence is known here,
        # before ``_frame_digest`` is computed in this same round. See
        # _agree_live_slots for the trigger this closes and why the count
        # agreement above it was not enough.
        slot_ballot: dict = {}
        corridor_detail = self._seam_funding_verdict(
            staging_bytes,
            direction,
            slots_digest=self._slots_membership_digest(slots),
            max_live_row=int(slots[-1].item()) if slots.numel() else -1,
            slot_ballot_out=slot_ballot,
        )
        # A ``slot_detail`` is a divergence the union cannot safely repair; it
        # joins the unanimous abandon like every other objection, and the plan
        # is left exactly as it was because nothing may move on that round.
        slots, slot_detail = self._agree_live_slots(slots, slot_ballot)
        if not slot_detail and int(slots.numel()) != int(tr.total_slots):
            # The union is a SUPERSET, so an unchanged cardinality is an
            # unchanged set -- this comparison is exact, not a heuristic.
            # THE PLAN IS REBUILT ON THE AGREED SET, and everything priced from
            # it with it. ``rows_of`` is a pure function of the slot id, so no
            # row that both sets already contained changes destination -- but
            # the bounds and the staging price DO change with the extra rows,
            # and they are re-derived here rather than carried over from the
            # provisional plan. The rung's relief was asked for the provisional
            # price, which is the smaller one; a shortfall that leaves is
            # caught by ``_staging_affordable`` below and votes into the same
            # unanimous abandon, so the conservatism costs a flip and never a
            # rank.
            tr = build_phase_flip_transition(
                slots, self._map, self._n_layers, self._vec, self._rank, direction
            )
            staging_bytes = self._staging_bytes(tr, direction, src, dst, waves)
        # BOUNDS ON THE FINAL PLAN. Computed after the agreement, so the rows
        # the group actually intends to move are the rows the pools are
        # checked against.
        too_small = self._row_bounds_detail(tr)
        if slot_detail:
            too_small.append(slot_detail)
        if corridor_detail:
            too_small.append(corridor_detail)
        affordable, staging_detail = self._staging_affordable(
            staging_bytes, direction
        )
        if not affordable:
            too_small.append(staging_detail)

        # #721: HOST RAM, measured at the flip, every flip.
        #
        # THE MEASUREMENT IS THE POINT, and the gate is secondary -- I want that
        # the right way round in the record. Two host-OOM-shaped kills landed
        # 7 s and 11 s after a completed flip, but steady-state host headroom
        # was ~38 GB, so a floor small enough to be safe would never have fired
        # and a floor large enough to fire would defer constantly. I do not have
        # a measured projected transient for the flip's host demand, and I will
        # not invent one to make a gate look decisive.
        #
        # So this LOGS the headroom at every flip -- a number nobody currently
        # records -- and defers only under a floor that is deliberately
        # conservative. If the logs show availability collapsing at flips, the
        # transient candidate is confirmed and the floor can then be set from
        # data. If availability stays flat while kills continue, the flip is
        # exonerated and lane RSS gains. That is the discriminator working
        # whichever way it points.
        try:
            from sglang.srt.mem_cache.pinned_host_budget import (
                pinned_host_memory_bytes,
            )

            _host_total, _host_avail = pinned_host_memory_bytes()
            allow_host, escalated, host_detail = flip_host_headroom_verdict(
                _host_avail, 0, int(getattr(self, "_host_ram_defers", 0))
            )
            logger.info("%s HOST HEADROOM %s: %s", LOG_PREFIX, direction, host_detail)
            if not allow_host:
                self._host_ram_defers = int(getattr(self, "_host_ram_defers", 0)) + 1
                too_small.append(host_detail)
            else:
                if escalated:
                    logger.warning("%s %s", LOG_PREFIX, host_detail)
                self._host_ram_defers = 0
        except Exception as exc:  # noqa: BLE001 - a guard must not break a flip
            logger.warning(
                "%s host-headroom guard could not run (%r); the flip proceeds "
                "unguarded rather than being refused on an unknown.",
                LOG_PREFIX,
                exc,
            )

        fits = 0 if too_small else 1
        # #656 C20: is the group's objection NOTHING BUT the seam-entry
        # margin? It rides the SAME reduction rather than adding one. A rank
        # that does not object votes 1, a rank whose only objection is the
        # margin votes 1, any other objection votes 0; the MIN therefore
        # answers "this is a margin DELAY and nothing else". Without it the
        # group log calls a healthy by-design wait a FLIP ABANDONED, which is
        # the string every acceptance harness in this corpus counts and the
        # one the 411-abandon decode wedge was measured with.
        margin_only = 0 if any(SEAM_MARGIN_DELAY_TAG not in d for d in too_small) else 1
        # #656 C22: THE WIRE FRAME RIDES THE SAME BALLOT. See _frame_digest
        # for why the premise needs verifying and what it cost when it was
        # only asserted. The [x, -x] pair makes MIN answer "are they all
        # equal", which is the only question here.
        # THE PARTS RIDE THE SAME PAYLOAD (#656 R2). Six more integers, three
        # more [x, -x] pairs, so a divergence can be ATTRIBUTED instead of
        # only detected -- see _frame_digest_parts for the metal round that
        # made that necessary. No new collective.
        # ``frame`` still comes from ``_frame_digest`` and nowhere else: it is
        # the value the ballot VOTES on, and the can-fail arm that reproduces
        # the metal signature works by stubbing exactly that method. Routing
        # the vote through the parts instead would have quietly disarmed the
        # one test that proves this ballot can fail.
        parts = self._frame_digest_parts(slots, direction, waves)
        frame = self._frame_digest(slots, direction, waves)
        # WHAT THIS RANK ACTUALLY FRAMED, recorded before the vote. #656 C22-d
        # made the framed set differ from the enumerated one -- the union is
        # what goes on the wire -- so a test (or an operator) that re-derives
        # a digest from ``_live_slots_fn`` is no longer asking the same
        # question the ballot asked. This is the question the ballot asked.
        self.last_framed_slots = slots
        self.last_framed_slots_digest = parts["slots"]
        self.last_framed_digest = frame
        payload = [fits, -fits, margin_only, frame, -frame]
        for name in self.FRAME_PARTS:
            payload.extend((parts[name], -parts[name]))
        reduced_fit = self._collective_min(payload)
        frames_agree = len(reduced_fit) < 5 or reduced_fit[3] == -reduced_fit[4]
        part_lo, part_hi = {}, {}
        if len(reduced_fit) >= 5 + 2 * len(self.FRAME_PARTS):
            for i, name in enumerate(self.FRAME_PARTS):
                part_lo[name] = reduced_fit[5 + 2 * i]
                part_hi[name] = -reduced_fit[6 + 2 * i]
        if not frames_agree:
            # NOT a capacity verdict, so it does not join `too_small` before
            # the reduction -- it cannot be known before it. It joins after,
            # so the abandon below reports it, and it forces the abandon
            # regardless of how the fit voted.
            self.frame_aborts += 1
            too_small.append(
                f"wire frame divergence: this rank framed digest {frame}, the "
                f"group spans [{-reduced_fit[4]}, {reduced_fit[3]}]. THE "
                f"DIVERGING TERM IS: "
                f"{self._name_frame_divergence(parts, part_lo, part_hi)}. The "
                f"per-peer payload LENGTHS would therefore not "
                f"match. Sending them anyway is register C22: the peer's "
                f"receive buffer keeps an unwritten tail, the checksum "
                f"trailer read out of it is not a checksum, and the guard "
                f"kills the instance reporting a corruption that never "
                f"happened. NOTHING HAS BEEN MOVED. But read the next "
                f"sentence before calling this benign: a divergence that "
                f"PERSISTS starves the pp_to_tp leg, and that leg is the one "
                f"decode needs, so a repeated refusal here ends as a WEDGE "
                f"-- alive, every request's KV intact, /health 503, no "
                f"tokens. Measured on metal 2026-08-13 14:46:39Z. That is "
                f"still strictly better than the SIGQUIT this replaces "
                f"(recoverable, and diagnosed rather than mysterious), but "
                f"it is not 'serving continues'. Compare the ranks' POOL "
                f"CENSUS lines: the one whose 'unaccounted' count differs is "
                f"the rank whose allocator holds rows the others do not "
                f"enumerate, and that count IS the payload-length mismatch"
            )
        if reduced_fit[0] == 0 or not frames_agree:
            # THE BUDGET'S CURRENCY, BOOKED WHERE EVERY RANK AGREES. This is
            # the reduced verdict, so all three ranks increment together and
            # a delay budget means the same thing on each of them.
            book = getattr(self, "_seam_abandons_in_a_row", None)
            if book is None:
                book = self._seam_abandons_in_a_row = {}
            book[direction] = book.get(direction, 0) + 1
            # A frame divergence is never a by-design margin wait, however
            # the margin half of the ballot voted.
            delayed_for_margin = (
                frames_agree and len(reduced_fit) > 2 and bool(reduced_fit[2])
            )
            self._pending = None
            self._armed_at = None
            self._parked_extent = None  # #746: cleared on EVERY exit
            self._last_hold_reason = None
            # Which of the two conditions THIS rank hit, so a boot that is
            # short of staging room is not read as a pool-sizing problem.
            if not frames_agree:
                # Counted as frame_aborts above. Kept ahead of the capacity
                # arms so a broken replication premise is never booked as a
                # pool that is too small -- they want opposite responses.
                pass
            elif corridor_detail:
                # Already counted by the gate itself, and counted there
                # rather than here so that a PEER's corridor refusal (which
                # reaches this rank only as a reduced verdict) is not
                # miscounted as this rank's pool being too small.
                pass
            elif staging_detail:
                self.staging_aborts += 1
            else:
                self.fit_aborts += 1
            if delayed_for_margin:
                logger.warning(
                    "%s FLIP DELAYED (seam entry margin, %s): every rank's "
                    "only objection is the C20 entry margin, so no rank is "
                    "short of the corridor LAW and nothing is wrong with the "
                    "pool. This rank: %s. Consecutive delayed attempts in "
                    "this direction: %d; when they reach the budget the law "
                    "governs and the seam enters. Serving continues on the "
                    "%s stack with every request intact.",
                    LOG_PREFIX,
                    direction,
                    "; ".join(too_small) if too_small else "clear (a peer was not)",
                    self._seam_abandons_in_a_row.get(direction, 0),
                    self._phase,
                )
            elif not frames_agree:
                logger.error(
                    "%s FLIP ABANDONED (wire frame divergence, %s): %s. This "
                    "is NOT a capacity verdict -- every rank may have room. "
                    "The ranks disagree about what the payload IS, so the "
                    "per-peer lengths would not have matched, and the group "
                    "refuses before a byte moves. Serving continues on the %s "
                    "stack with every request intact. The replication premise "
                    "that broke is named in the digest above; the live slot "
                    "set is the term that varies at runtime, and #639 had to "
                    "give the prefix-length vector the same ballot for the "
                    "same reason.",
                    LOG_PREFIX,
                    direction,
                    "; ".join(too_small),
                    self._phase,
                )
            else:
                logger.error(
                    "%s FLIP ABANDONED (pool too small for the live set): %s. "
                    "This rank: %s. No bytes were moved -- the bound is checked "
                    "before the plan is executed -- and serving continues on the "
                    "%s stack with every request intact. The live set grows with "
                    "the resident prefix cache, so flushing the cache or sizing "
                    "the target pool up are both real answers; a smaller TP pool "
                    "is also what a draft-KV allocation inside the same budget "
                    "produces.",
                    LOG_PREFIX,
                    direction,
                    "; ".join(too_small) if too_small else "fits (a peer did not)",
                    self._phase,
                )
                # #689: NAME THE OCCUPANT, not just the shortfall. Which rank
                # binds is already in the line above; what is holding that
                # rank's budget was not, and without it the remedy is a guess.
                try:
                    logger.error(
                        "%s %s",
                        LOG_PREFIX,
                        self.staging_budget_census(
                            int(getattr(self, "_last_staging_bytes", 0) or 0)
                        ),
                    )
                except Exception:  # noqa: BLE001 - never break the refusal path
                    pass
            # #485: BOUND THE RETRY, AND END IT IF IT CANNOT WIN.
            #
            # Only real abandons are bounded. A margin DELAY is a by-design
            # wait whose own budget already ends it -- after
            # ``seam_entry_delay_budget()`` rounds the gate yields to the law
            # and the flip goes through -- so damping it here would slow a
            # path that is already self-terminating, and would change the
            # shipped C20 behaviour. This whole block is therefore inert on a
            # configuration whose only objection is the entry margin.
            if not delayed_for_margin:
                # #656: PUBLISH THE ABANDON so the phase policy can see it.
                #
                # The policy learns an arm's fate from ``arm``'s return, and
                # an abandon happens long after arm returned True -- so for
                # the first ``seam_abandon_cap()`` rounds the policy is told
                # the flip was accepted while every one of them dies at the
                # seam. That is the window in which boot E burnt its arms.
                # A sequence number rather than a flag: the reader is the
                # scheduler's round loop and it must be able to tell a NEW
                # abandon from the same one seen twice.
                self.seam_abandon_seq = getattr(self, "seam_abandon_seq", 0) + 1
                self.last_seam_abandon = (
                    direction,
                    "; ".join(too_small) if too_small else "a peer refused",
                )
                spent = self._seam_abandons_in_a_row.get(direction, 0)
                cap = seam_abandon_cap()
                if cap and spent >= cap:
                    self._install_seam_cap_guard(direction, spent, too_small)
                else:
                    skips = seam_backoff_skips(spent, seam_abandon_backoff_max())
                    if not hasattr(self, "_seam_retry_at_arm"):
                        self._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
                        self.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
                    self._seam_retry_at_arm[direction] = (
                        getattr(self, "_arm_seq", 0) + 1 + skips
                    )
                    if skips:
                        logger.warning(
                            "%s seam BACKING OFF (%s): %d consecutive group "
                            "abandons, so the next %d arm request(s) are "
                            "declined before the seam is priced again, and the "
                            "flip stands down for good at %d. Each entry runs "
                            "the spill ladder and an empty_cache while the "
                            "armed window withholds admissions, so retrying a "
                            "refusal that cannot change is what starves the "
                            "detokenizer -- the damping is the point, not a "
                            "delay tactic.",
                            LOG_PREFIX,
                            direction,
                            spent,
                            skips,
                            cap,
                        )
            # Abandoned before any byte moved: there is no seam to attribute.
            seam_census.reset()
            return None

        # The group is going through, so this direction's delay budget is
        # whole again. Reset here rather than in the gate: the gate is
        # rank-local and a rank that cleared while a peer did not has learnt
        # nothing about the group.
        if not hasattr(self, "_seam_abandons_in_a_row"):
            self._seam_abandons_in_a_row = {}
        self._seam_abandons_in_a_row[direction] = 0
        # #485: and so is the backoff. A seam that went through has proved the
        # demand fundable, so the next refusal starts its damping from zero
        # rather than inheriting a streak the group has already broken.
        if not hasattr(self, "_seam_retry_at_arm"):
            self._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
            self.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
        self._seam_retry_at_arm[direction] = 0

        # THE MOVE, ONE LAYER WAVE AT A TIME (#631).
        #
        # Pack, exchange, read the retained leg, swap THIS WAVE's backing,
        # write -- then the next wave. Every buffer allocated for a wave is
        # dead before the next wave allocates its own, so the staged bytes
        # are one wave's share of the crossing rather than all of it. That
        # is the difference between a flip whose cost tracks the resident
        # live set (and therefore becomes unaffordable at some request
        # length, then wedges under strict purity) and one whose cost is a
        # property of the layer map.
        #
        # A single wave containing every layer reproduces the previous
        # code exactly, which is what keeps the wave count a one-variable
        # A/B.
        seam_census.mark("plan")
        local_src = tr.local_pp_rows if direction == PP_TO_TP else tr.local_tp_rows
        local_dst = tr.local_tp_rows if direction == PP_TO_TP else tr.local_pp_rows
        swap = self._seam_swap()
        read_ms = xfer_ms = write_ms = 0.0
        sent_bytes = 0
        received_bytes = 0
        local_bytes = 0
        reclaimed = False
        # Asked ONCE per flip rather than per wave: ``_pools_alias`` is
        # cached, but the gate it drives decides an ordering that must not
        # be allowed to differ between two waves of the same seam.
        seam_restore_first = self._seam_restore_first and not self._pools_alias()

        for wave in waves:
            wave_set = None if wave is None else set(int(f) for f in wave)

            # PACK (reads only): per peer, layers ascending, one row list.
            t_read0 = self._clock()
            outgoing_payloads: Dict[int, torch.Tensor] = {}
            for peer in tr.send_layers:
                layers_w = _in_wave(tr.send_layers[peer], wave_set)
                if not layers_w or not int(tr.send_rows[peer].numel()):
                    continue
                outgoing_payloads[peer] = self._pack_outgoing(
                    tr, direction, src, peer, layers_w
                )
            read_ms += (self._clock() - t_read0) * 1000.0
            seam_census.mark("kv_pack")

            # Expected incoming sizes from MY OWN pool's row widths -- the
            # runtime pin of row byte-compatibility across layouts.
            incoming_nbytes: Dict[int, int] = {}
            wave_recv_layers: Dict[int, List[int]] = {}
            for peer in tr.recv_layers:
                n = int(tr.recv_rows[peer].numel())
                layers_w = _in_wave(tr.recv_layers[peer], wave_set)
                if not layers_w or not n:
                    continue
                wave_recv_layers[peer] = layers_w
                incoming_nbytes[peer] = (
                    sum(
                        dst.row_nbytes(self._dst_layer_idx(direction, f)) * n
                        for f in layers_w
                    )
                    + _CHECKSUM_BYTES
                )

            # EXCHANGE (pools still untouched): failure up to and including
            # checksum verification aborts with both pools byte-identical
            # FOR THIS WAVE. Earlier waves have already been written, so a
            # raise here is still inside the no-return region -- the same
            # contract the single-wave move had once its first byte landed.
            t_xfer0 = self._clock()
            received = self._exchange(outgoing_payloads, incoming_nbytes)
            xfer_ms += (self._clock() - t_xfer0) * 1000.0

            # THE SEND BUFFERS ARE DEAD HERE, and holding them any longer is
            # pure corridor. ``_exchange`` returns only after every work in
            # the batch has been polled to completion (``bounded_collective``,
            # then a device synchronize), so nothing reads these bytes again.
            sent_bytes += sum(int(t.numel()) for t in outgoing_payloads.values())
            outgoing_payloads.clear()

            incoming_data: Dict[int, torch.Tensor] = {}
            for peer, layers_w in wave_recv_layers.items():
                payload = received.get(peer)
                if payload is None or payload.numel() != incoming_nbytes[peer]:
                    got = 0 if payload is None else payload.numel()
                    raise KvReshardError(
                        f"{LOG_PREFIX} exchange returned {got} bytes from peer "
                        f"{peer}, expected {incoming_nbytes[peer]} -- size "
                        f"mismatch means the layouts' row formats or the "
                        f"payload convention diverged"
                    )
                data = payload[:-_CHECKSUM_BYTES]
                want = int(payload[-_CHECKSUM_BYTES:].clone().view(torch.int64).item())
                have = uint8_checksum(data)
                # #656 C22: SAY WHICH OF THE TWO FAILURES THIS IS. A uint8
                # sum over N bytes can only land in [0, 255N]. A field
                # outside that range is not a checksum the sender computed
                # and this rank disagrees with -- it is not a checksum at
                # all, so the payload was never framed the way this rank
                # expected and the DATA is not the thing that is wrong. The
                # acceptance run reported one of these as a corruption and
                # killed a healthy instance for it; the two diagnoses send
                # an operator to opposite ends of the system.
                if not checksum_is_representable(want, int(data.numel())):
                    raise KvReshardError(
                        f"{LOG_PREFIX} payload TRAILER from peer {peer} is "
                        f"NOT A CHECKSUM: the field reads {want}, outside "
                        f"[0, {255 * int(data.numel())}], the only values a "
                        f"uint8 sum over {int(data.numel())} bytes can take. "
                        f"This is a FRAMING failure, not payload corruption: "
                        f"the peer sent a different number of bytes than this "
                        f"rank allocated for it, so the tail of the receive "
                        f"buffer -- where the trailer lives -- was never "
                        f"written. The pre-move frame ballot agreed this "
                        f"round, so the divergence is in the TRANSPORT, not "
                        f"in the plan (register C22)."
                    )
                if want != have:
                    raise KvReshardError(
                        f"{LOG_PREFIX} payload checksum mismatch from peer "
                        f"{peer}: sender {want}, receiver {have} -- both are "
                        f"possible sums over these {int(data.numel())} bytes, "
                        f"so the frame held and the DATA differs. Refusing to "
                        f"scatter."
                    )
                incoming_data[peer] = data
            received.clear()
            received_bytes += sum(incoming_nbytes.values())

            # THE RETAINED LEG of this wave, read BEFORE the backing swap.
            #
            # INVARIANT, load-bearing: every SOURCE READ of a wave completes
            # before the first DESTINATION WRITE of that wave. The local leg
            # used to read and write per layer in one loop, which is safe
            # only while the two pools are disjoint allocations. That is the
            # #297 reads-before-writes hazard, and it becomes reachable the
            # moment the phases share backing (one arena / mutually
            # exclusive VMM backing sized max(PP, TP) instead of their sum)
            # -- then a destination write can land on a source row that has
            # not been read yet.
            t_write0 = self._clock()
            local_layers_w = _in_wave(tr.local_layers, wave_set)
            local_data = (
                self._pack_local(tr, direction, src, local_src, local_layers_w)
                if local_layers_w and int(local_src.numel())
                else []
            )
            local_bytes += sum(int(t.numel()) for t in local_data)
            seam_census.mark("kv_local_read")

            # #631 CROSS-PHASE BACKING SWAP, PER WAVE. Every byte this wave
            # owes has now been read, so the source layers of this wave may
            # hand their physical pages back -- and the destination layers of
            # this wave need theirs before the writes below.
            #
            # THE ORDER IS reclaim -> restore -> release (#631 section 2.1b),
            # and each of the three positions is load-bearing.
            #
            # RESTORE BEFORE RELEASE. Release-first makes a wave's releases
            # pay for its own commits, which is why the wave count had to be
            # capped at the SMALLEST stage -- a wave with no layer of mine to
            # release cannot afford the layers it must commit. That coupling
            # is the seam's staging SLOPE (~4.5 MiB per 1000 live slots at
            # W=4) and it is what caps the KV pool near 438k tokens under the
            # corridor law. Restore-first budgets the overlap explicitly
            # instead: the peak becomes the worst PREFIX imbalance rather
            # than a per-wave one, the wave cap disappears, and the slope
            # falls with W.
            #
            # RECLAIM AHEAD OF THE RESTORE, not between it and the release.
            # The restore is the allocation that can fail inside the
            # no-return region -- it raised cuMemCreate OUT_OF_MEMORY on
            # metal on 2026-08-09 and took the instance down. Under
            # release-first that restore happened at the memory TROUGH, with
            # the source already unmapped. Restore-first moves it to the
            # PEAK, with the source still fully mapped, so it is strictly
            # MORE exposed and the reclaim must lead it. Moving the reclaim
            # after the restore re-opens that crash; TestWavedSeamOrdering
            # fails if anyone does.
            #
            # ALIASED POOLS KEEP THE OLD ORDER, and this gate is a
            # correctness bound rather than a tuning one. When the two
            # layouts overlay the same bytes, the destination's pages ARE
            # the source's pages: restoring first and releasing second would
            # hand back the very mapping just committed and leave the
            # destination unbacked. ``_flip_waves`` already collapses the
            # aliased seam to a single wave; the order needs its own gate
            # because a single wave still runs this branch.
            # THE WAVE'S DESTINATION WRITES, built ONCE as a job list.
            #
            # Both the whole-wave path and the row-blocked path below
            # consume this same list in this same order, so they write
            # identical bytes by construction rather than by two code paths
            # agreeing. The row-blocked path only chooses WHEN each row is
            # written relative to the backing calls around it.
            jobs: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
            for f, data in zip(local_layers_w, local_data):
                jobs.append((self._dst_layer_idx(direction, f), local_dst, data))
            for peer, layers_w in wave_recv_layers.items():
                n = int(tr.recv_rows[peer].numel())
                rows = tr.recv_rows[peer]
                offset = 0
                for f in layers_w:
                    li = self._dst_layer_idx(direction, f)
                    width = dst.row_nbytes(li)
                    chunk = incoming_data[peer][offset : offset + n * width]
                    jobs.append((li, rows, chunk.view(n, width)))
                    offset += n * width

            if swap is None:
                for fn in self._pre_write_fns:
                    fn(direction)
                self._write_jobs(dst, jobs)
            elif not seam_restore_first:
                swap.release_wave(direction, wave)
                if not reclaimed:
                    swap.reclaim_between(direction)
                    reclaimed = True
                swap.restore_wave(direction, wave)
                self._write_jobs(dst, jobs)
            else:
                if not reclaimed:
                    swap.reclaim_between(direction)
                    reclaimed = True
                if self._seam_row_blocks > 1 and swap.is_span_swappable(direction):
                    self._stream_wave(
                        swap,
                        direction,
                        wave,
                        src,
                        dst,
                        jobs,
                        self._seam_row_blocks,
                    )
                else:
                    swap.restore_wave(direction, wave)
                    self._write_jobs(dst, jobs)
                    swap.release_wave(direction, wave)
            local_data = []
            incoming_data.clear()
            write_ms += (self._clock() - t_write0) * 1000.0
            seam_census.mark("kv_write")

        # EXTRA MOVERS (weights arena, GDN state) then CUTOVER.
        #
        # #631 defect J: census the pool on BOTH sides of the cutover. The
        # leak the checker raises one pass later cannot distinguish "the
        # enumeration missed a row" from "the cutover mis-registers the
        # destination allocator", and those have opposite fixes. Straddling
        # the cutover answers it directly: if the unaccounted page is
        # already there BEFORE, the enumeration is innocent.
        # #690: TIME THE TAIL, because it is most of the flip and nothing
        # measured it. read/exchange/write cover the wave loop only -- the
        # backing swap included, since t_write0 is taken before it -- so
        # everything below fell into the residual. Measured over 291
        # same-regime DONE lines that residual is 2.0-2.1 s and FLAT across a
        # 3600x range of live slots (123 -> 440095), i.e. 81 % of a
        # low-occupancy flip. A term that large cannot stay a subtraction:
        # #677 prices windows against it and #692 prices depth against it.
        #
        # Split into movers vs cutover because they have different fixes. The
        # movers are occupancy-INDEPENDENT by construction (the weights arena
        # refill is the same bytes whatever the KV live set is), which is the
        # leading explanation for the flatness; the cutover is the group step.
        t_movers0 = self._clock()
        self._pool_census("pre-cutover", direction)
        for fn in self._pre_cutover_fns:
            fn(direction)
            # Labelled per mover: the GDN state leg and the weights refill
            # have different sizes AND different fixes, so one combined
            # "pre_cutover" bar would be unattributable.
            seam_census.mark(getattr(fn, "census_label", "pre_cutover_fn"))
        movers_ms = (self._clock() - t_movers0) * 1000.0
        t_cutover0 = self._clock()
        self._cutover_fn(direction)
        seam_census.mark("cutover")
        self._pool_census("post-cutover", direction)
        cutover_ms = (self._clock() - t_cutover0) * 1000.0
        self._phase = _PHASE_AFTER[direction]
        self._pending = None
        self._armed_at = None
        # #746: the commit is an exit too -- the packed rows are moved, the
        # extent has no referent, and a surviving snapshot would pin the
        # rung into the next phase for ever (the M5 failure mode).
        self._parked_extent = None
        self._epoch += 1
        self.completed += 1
        total_ms = (self._clock() - t0) * 1000.0
        stats = {
            "direction": direction,
            "phase": self._phase,
            "epoch": self._epoch,
            "live_slots": tr.total_slots,
            "outgoing_cells": tr.outgoing_cells,
            "incoming_cells": tr.incoming_cells,
            "sent_bytes": sent_bytes,
            "received_bytes": received_bytes,
            "local_bytes": local_bytes,
            "staging_bytes": staging_bytes,
            "seam_waves": len(waves),
            "read_ms": read_ms,
            "exchange_ms": xfer_ms,
            "write_ms": write_ms,
            # #690: the tail, so the fixed cost is a MEASUREMENT and not a
            # residual anyone has to regress for.
            "movers_ms": movers_ms,
            "cutover_ms": cutover_ms,
            "total_ms": total_ms,
        }
        self.last_stats = stats
        logger.warning(
            "%s DONE %s (epoch %d) in %.1f ms over %d seam wave(s): %d live "
            "slots, sent %d cells / %.2f MiB, received %d cells / %.2f MiB, "
            "local %.2f MiB, staging reserved %.2f MiB (read %.1f ms, "
            "exchange %.1f ms, write %.1f ms, movers %.1f ms, "
            "cutover %.1f ms)",
            LOG_PREFIX,
            direction,
            self._epoch,
            total_ms,
            len(waves),
            tr.total_slots,
            tr.outgoing_cells,
            stats["sent_bytes"] / 1048576.0,
            tr.incoming_cells,
            stats["received_bytes"] / 1048576.0,
            stats["local_bytes"] / 1048576.0,
            stats["staging_bytes"] / 1048576.0,
            read_ms,
            xfer_ms,
            write_ms,
            movers_ms,
            cutover_ms,
        )
        census = seam_census.end()
        if census is not None:
            peak = census.peak_bytes()
            if peak is not None:
                stats["seam_transient_bytes"] = int(peak)
                low = census.trough()
                stats["seam_trough_stage"] = low[0] if low else ""
                # #485 T3: AND FEED THE #485 TRANSIENT CENSUS, which is a
                # DIFFERENT consumer from the gate below and had a hole the
                # gate did not.
                #
                # transient_census.note() is called from exactly one site --
                # Scheduler.process_batch_result -- and labels its sample
                # with batch.forward_mode.name. A cutover is not a batch, so
                # the census that the planner cut gate funds "the WORST load
                # state" from could not take a single sample inside the
                # largest transient in the system. Measured over 196 flips of
                # the two certification windows: 5800 MiB modal, 7055 MiB
                # worst on rank 0, against a census reporting 1989 MiB.
                #
                # The TROUGH's free level, not the peak DRAW: the census
                # stores per-state minima of the free column and computes the
                # draw itself against its own at-rest baseline. Handing it a
                # draw here would be measuring against a different reference
                # than every other state in the same table.
                if low is not None:
                    try:
                        from sglang.srt.planner import transient_census

                        transient_census.note_free(
                            transient_census.seam_load_state(direction), low[1]
                        )
                    except Exception:  # noqa: BLE001 -- never kill a cutover
                        pass
                # FEED THE MEASUREMENT BACK TO THE GATE THAT GUESSED IT.
                # This is the only place the driver-visible in-cutover draw is
                # ever known, and until now it was written to a stats dict and
                # read by nobody. A running MAX, not a mean: the law check is
                # a safety predicate and must be priced on the worst draw this
                # rank has seen, not on a typical one. It only ever ratchets
                # up, so an unusually cheap flip cannot re-open the gap.
                try:
                    book = getattr(self, "_seam_draw_max", None)
                    if isinstance(book, dict):
                        key = str(direction)
                        if int(peak) > int(book.get(key, 0)):
                            book[key] = int(peak)
                except Exception:  # noqa: BLE001 -- bookkeeping, never fatal
                    pass
        return stats
