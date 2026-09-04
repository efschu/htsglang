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
import re
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
from sglang.srt.managers import seam_coverage
from sglang.srt.managers import tree_congruence
from sglang.srt.managers.io_struct import PhaseFlipDecision
from sglang.srt.managers.kv_reshard import (
    _CHECKSUM_BYTES,
    KvPoolView,
    _checksum,
    # #969 §W3: `_encode` packed the per-rank consensus proposal
    # [armed, ready, expired, ...] for the reduce that has been deleted.
    _gather_block_rows,
)
# #969 §W3: the presence-disposition catalogue went with the presence gate
# (`_await_group_presence`, `_spin_for_group_presence`, `_abandon_no_quorum`).
# It was the taxonomy of "why did a rank withhold its announcement" -- a
# question that cannot be asked any more, because no rank announces anything:
# PP0 decides and the ranks below execute.
from sglang.srt.managers.warmup_latency import WarmupLatencyLedger
from sglang.srt.model_executor.weights_arena import (
    checksum_is_representable,
    uint8_checksum,
)
from sglang.srt.utils.common import ceil_align

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP"
# #1028: how many consecutive flips may be deferred for an incomplete writeback
# fence before the flip proceeds anyway and the recompute is accepted out loud.
# Small on purpose: this is a bound against a wedge, not a retry budget.
_WRITEBACK_DEFER_LIMIT = 3

#: #871: consecutive work-retracting cutovers whose fence persisted nothing
#: before the blind-fence alarm fires. FOUR, taken from #719's settled reading
#: at the stale-gate streak: two quiet cutovers are an ordinary stretch, so a
#: threshold of 2 would alarm on normal operation and a threshold that alarms
#: on normal operation gets ignored, which is how the condition stayed
#: invisible in the first place.
FENCE_BLIND_STREAK = 4


def advance_fence_blind_streak(
    previous, *, released: bool, persisted_nothing: bool
) -> int:
    """#871: one step of the blind-fence streak. Pure, so it can be falsified.

    THE GATE IS ``released``, MIRRORING #719's BUSY GATE, and it is the whole
    reason this is not a crying-wolf alarm. A fence over an empty tree is
    CORRECT to persist nothing -- there was nothing to persist -- so an idle
    instance must never accumulate a streak. The streak advances only when this
    cutover actually took work away AND nothing was written that could give it
    back. Any cutover that persists something, or that retracts nothing, resets
    it: this reports a SUSTAINED condition, never a historical one.

    Extracted from the seam rather than left inline so it is reachable without a
    scheduler. A guard whose logic can only be exercised by booting is a guard
    that gets shipped unexercised, which is the failure mode this whole ticket
    is about.
    """
    prior = int(previous or 0)
    if released and persisted_nothing:
        return prior + 1
    return 0


def chunk_blocks_quiescence(chunked_req) -> bool:
    """Does this chunked prefill prevent a rank from being quiescent?

    ONE definition with TWO callers, and they must never disagree:

      * ``ready_fn`` asks it to decide whether this rank may announce
        itself at the flip entry;
      * ``get_next_batch_to_run``'s armed park asks it to decide whether
        the scheduler may keep working while a flip is armed.

    True ONLY while the request is mid-admission -- it has been chosen but
    has no pool row yet, so its KV has no home and the state is at no
    settled boundary. That clears within a round. Between chunks IS a
    settled boundary: committed KV, a fully accounted extend_range.

    #1065 (2026-09-01): THE STRICT CLAUSE IS DELETED, together with its
    #858b/#887 runnability plumbing (``prefill_runnable_in_current_layout``).
    It held the flip on "the flip would discard an incomplete chunk" -- but
    under the cutover-full-reset design the flip discards nothing: the
    committed chunks sit in the tree, the save fence persists them, and
    re-admission serves them back by read-through. A flip behaves like a
    freshly started server with a cache hit (user design, 2026-09-01); an
    incomplete chunk is therefore never a reason to refuse the cutover.

    The clause was also a TWO-ORACLE defect: its runnability term answered
    "can this chunk make progress here" from the raw #887 chunk budget,
    while the builder's own gate (``phase_purity.prefill_blocked_here``)
    additionally requires ``tp_compute_fits_in_one_chunk`` and an open seam
    grant. Measured 2026-09-01 05:04:53-05:23Z (boot_855_tiprevert1033):
    33094 tok pending, budget>0 but fits=False -> the builder refused every
    continuation chunk while this predicate blocked quiescence on it -- 37
    arm/abandon cycles per rank over 1114 s, 11 queued / 0 running, and the
    abort of the hanging request deferred until after a cutover that never
    came. Same class as the #1033 half-grant hold (F3, bc934655a1): a second
    site answering the builder's question differently. Waiting for work the
    current layout's builder will not run is never drain-and-flip.
    """
    return (
        chunked_req is not None
        and getattr(chunked_req, "req_pool_idx", None) is None
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

#: #968 P2: slack added on top of the flip park deadline for the dynamic
#: recv-abort bound (see `pp0_flip_hold_recv_bound_s`).
DEFAULT_FLIP_HOLD_RECV_SLACK_S = 30.0
ENV_FLIP_HOLD_RECV_SLACK_S = "SGLANG_PP0_FLIP_HOLD_RECV_SLACK_S"


def pp0_flip_hold_recv_bound_s(runtime) -> float:
    """#968 P2: the abort bound PP0's CURRENT flip state justifies for a
    parked ``recv_object``, in seconds; 0 = no bound (no flip armed).

    trainA (2026-09-01 19:31:30): PP0 armed pp_to_tp with microbatches in
    flight, then parked inside ``recv_object[src=2]`` for 90 s. Its 30 s
    park deadline (``_park_deadline_s`` -- THE one timeout in the system,
    rank 0 only) could never fire, because the abandon decision runs in the
    scheduler loop, i.e. in the very thread that was parked (the #977 form:
    recovery actuator inside the wedged thread). The ring starved until
    #1071 killed rank 1 at 90 s.

    Bound = park deadline + slack: the deadline is the design's own budget
    for a healthy armed wait, so a recv outliving it by the slack means the
    one recovery the group has is being outlived. Registered by
    ``event_loop_pp`` on PP0 only; consulted by the transport layer on
    expired steps. Defensive throughout -- any surprise reads as "no bound".
    """
    try:
        if runtime is None:
            return 0.0
        if getattr(runtime, "_pending", None) is None:
            return 0.0
        if getattr(runtime, "_armed_at", None) is None:
            return 0.0
        deadline = float(getattr(runtime, "_park_deadline_s", 0.0) or 0.0)
        if deadline <= 0:
            return 0.0
        slack = float(
            os.environ.get(ENV_FLIP_HOLD_RECV_SLACK_S, DEFAULT_FLIP_HOLD_RECV_SLACK_S)
        )
        return deadline + max(0.0, slack)
    except Exception:  # noqa: BLE001 - a bound provider may never wedge the recv
        return 0.0

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




def build_launched_passes_fn(scheduler) -> Callable[[], Tuple[Sequence[int], int]]:
    """#1173: which microbatch slots hold a pass THIS rank launched and owes.

    Returns ``(sorted outstanding slot ids, forward_ct)``. The set is
    ``Scheduler._pp_launched_pending``, which the mixin adds to immediately
    before ``_pp_launch_batch`` and discards where the slot's result is
    processed -- i.e. exactly "launched and not yet returned". Reusing it
    rather than deriving a second notion is the point: the #1020 void guard
    already refuses to null a slot on this same authority, so the arm and
    the guard cannot disagree about what is outstanding.

    An empty set on a rank with no PP ring (no attribute) reads as "nothing
    outstanding", which is the truth there and reproduces the pre-#1173
    arming behaviour byte for byte.
    """

    def _launched() -> Tuple[Sequence[int], int]:
        pending = getattr(scheduler, "_pp_launched_pending", None) or ()
        try:
            fwd_ct = int(getattr(scheduler, "forward_ct", -1))
        except Exception:  # noqa: BLE001 - a probe never breaks the arm
            fwd_ct = -1
        return sorted(int(i) for i in pending), fwd_ct

    return _launched


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
        # #1065 execution proof: an incomplete chunk no longer holds the
        # flip (strict clause deleted -- see chunk_blocks_quiescence).
        # Logged once per boot, on the first flip a pre-#1065 tree would
        # have held.
        if chunked is not None and not getattr(
            scheduler, "_1065_chunk_unblock_announced", False
        ):
            scheduler._1065_chunk_unblock_announced = True
            logger.info(
                "%s #1065 CHUNK DOES NOT HOLD THE FLIP: chunked prefill %s "
                "is between chunks (settled boundary); the flip proceeds and "
                "re-admission serves the committed prefix by read-through "
                "(cutover-full-reset design). The strict hold that livelocked "
                "tp_to_pp on 2026-09-01 is deleted.",
                LOG_PREFIX,
                str(getattr(chunked, "rid", "?"))[:8],
            )
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
        # #858: THE ORPHAN GATE IS GONE, NOT NARROWED. It blocked on requests
        # "reachable ONLY through last_mbs/last_batch ... not yet merged into
        # the resident set THE CARRY HARVESTS" (#631 defect L). There is no
        # harvest: #856 retracts residents instead of carrying them, and
        # de4f541b41 made `_live_reqs` enumerate running_mbs, last_mbs,
        # running_batch AND last_batch -- the identical population. The
        # retraction already sees them, so this gate only delayed flips for a
        # mechanism that no longer exists. Narrowing it would have left a
        # third stale premise behind.
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
                # #1173: UNREACHABLE ON PP0 FOR A PASS PP0 ITSELF LAUNCHED.
                # `arm()` now DEFERS while `_pp_launched_pending` is non-empty
                # (see the ARM DEFERRED line), so on the request-origin rank
                # this hold can no longer be entered for the weg1b4 cause.
                # It is NOT deleted, and the two survivors are named because a
                # reader must be able to tell them apart: (a) a FOLLOWER, which
                # takes the arm as an order and may legitimately still hold an
                # in-flight slot, and (b) a pass launched between PP0's
                # deferral check and the order reaching this rank. Both drain
                # on their own; neither is the launch-and-arm race.
                return (
                    f"PP microbatches still in flight (mb slots {live}); "
                    f"#1173: on PP0 this is unreachable for a pass PP0 itself "
                    f"launched -- arm() defers instead -- so this is a "
                    f"follower state or a pass launched around the order"
                )
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
                len(getattr(b, "reqs", []) or []) for b in scheduler._resident_batches()
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


_EXTENT_FALLBACK_SAID: set = set()


def _warn_extent_fallback(req, n: int) -> None:
    """#922: say it once per rid when the authoritative extent is missing.

    Silence here would be the old behaviour wearing the new code's name --
    the exact shape this ticket closes.
    """
    rid = str(getattr(req, "rid", "?"))
    if rid in _EXTENT_FALLBACK_SAID:
        return
    _EXTENT_FALLBACK_SAID.add(rid)
    logger.warning(
        "%s #922 EXTENT FALLBACK rid=%s: kv_allocated_len is unavailable, so "
        "this enumeration uses seqlen=%d, which is known to over-count by the "
        "spec reserve. One stale row may be moved; none is dropped.",
        LOG_PREFIX,
        rid,
        n,
    )


def owned_row_extent(req, page_size: int = 1) -> int:
    """Rows the ALLOCATOR has handed this request. #922.

    ONE DEFINITION, TWO CALLERS, and that is the point rather than tidiness.
    `_resident_rows` says it in its own docstring -- "two enumerations of
    'which rows does this request hold' that can disagree is the shape #822
    exists to end" -- and the two enumerations then disagreed about the
    EXTENT instead of the membership, which is the same defect one level down.

    `req.seqlen` is `len(origin_input_ids) + len(output_ids)`: a property of
    the SEQUENCE. `req.kv_allocated_len` is what the allocator handed out and
    is precisely what the invariant checker charges to the pool, page-aligned.
    Both readers sliced with the former.

    THE SIGN, MEASURED TWICE, ON TWO BOOTS AND TWO CONFIGS:
      * 2026-08-09 (phase_flip_presence.py:513-516): `seqlen=82
        kv_allocated_len=81 delta_vs_seqlen=-1`;
      * 2026-08-27, boot 2f/2g: SIXTY-THREE of 63 `FLIP EXTENT PROBE`
        emissions, three ranks, every flip, `delta_vs_seqlen=-1` -- at
        `seqlen=9448/kv_allocated_len=9447` and at `seqlen=2/
        kv_allocated_len=1`.
    `seqlen` OVER-counts by one, so `req_to_token[idx, :seqlen]` reads one row
    BEYOND what the allocator owns: a stale cell left by that row's previous
    tenant.

    BOTH CALL SITES DEFERRED THE CHANGE PENDING EXACTLY THIS EVIDENCE.
    presence.py:521-524: "NOT yet changed: one measurement on one config is
    not enough to re-cut an enumeration whose errors are silent, and the
    change is a one-liner once a second flip confirms the sign."
    build_flip_live_slots_fn: "change this only on that evidence." The second
    flip has now confirmed the sign 63 times, so this is that one-liner.

    Returns -1 when the request cannot state its extent. The two callers
    degrade differently ON PURPOSE and the asymmetry is the safety argument:
    the census answers "no verdict" (declaring an owner that holds the wrong
    rows is worse than declaring none), while the mover falls back to the OLD
    extent and says so, because moving one stale row is what it does today
    while DROPPING a live one would lose a request's context.
    """
    alloc = int(getattr(req, "kv_allocated_len", -1) or -1)
    if alloc <= 0:
        return -1
    return ceil_align(alloc, page_size) if page_size > 1 else alloc


def _page_size_of(scheduler) -> int:
    try:
        return int(
            getattr(
                getattr(scheduler, "token_to_kv_pool_allocator", None), "page_size", 1
            )
            or 1
        )
    except Exception:  # noqa: BLE001
        return 1


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
            # #922: the allocator's extent, with a LOUD fallback. The
            # comment above asked for the probe's evidence before re-cutting
            # this; 63 of 63 emissions on boot 2f/2g report
            # delta_vs_seqlen=-1, so seqlen moves one row the allocator does
            # not own -- a stale cell "moved as if it were live KV"
            # (phase_flip_presence.py:516-519).
            #
            # FALLING BACK RATHER THAN SKIPPING, and the direction is the
            # whole safety argument: moving one stale row is what this does
            # today and is recoverable; DROPPING a live row loses a request's
            # context at the seam, silently. So a request that cannot state
            # its extent keeps the old one and says so.
            n = owned_row_extent(req, _page_size_of(scheduler))
            if n < 0:
                n = int(req.seqlen)
                _warn_extent_fallback(req, n)
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
            # controller snapshots at ARM (``PhaseFlipRuntime.parked_extent``),
            # which cannot be stale and exists even when no enumeration ran
            # before the flip armed. The split above remains the snapshot's
            # measurement source.
            #
            # #808: #802's layout-tagged sticky record is NOT kept as a
            # fallback. It was the authoritative-but-wrong-layout value that
            # priced the unfundable floor in the first place; re-admitting it
            # when the snapshot is unreadable would restore the defect to
            # guard a rarer one. See _flip_pending in kv_backing_relief.py.
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

    Measured at a real cutover (#1205: the labels in these quoted lines are the PRE-FIX ones. No
    shipped code emits ``cur_slot_reqs``/``resident_reqs`` any more; the
    census prints ``live_reqs`` (the wide, id-deduped count) and
    ``resident_slot_entries`` (undeduped list entries across slots).
    Kept verbatim because they are quotations from real boot logs.)::

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
    # W30: `last_mbs` IS A RESIDENCY ROUTE AND WAS THE ONE ROUTE MISSING HERE.
    #
    # `last_batch` above is the NON-PP handle. Under `event_loop_pp` the
    # per-slot equivalent is `last_mbs[mb_id]`, and a request that has just
    # finished a prefill iteration sits THERE and nowhere else until the next
    # `get_next_batch_to_run` merges it into the running batch. The cutover
    # guard `orphan_resident_reqs` checked `last_mbs`; this authority did not,
    # so the two disagreed about what "resident" means and the retraction
    # could not see a request the guard would then refuse to flip past.
    #
    # #1202: THAT GUARD NO LONGER EXISTS. It lived in
    # `phase_flip_resident_carry`, which #969 deleted whole (911 LOC; see
    # scheduler.py:6463), and nothing in this tree defines the name any more.
    # The paragraph is kept because it records WHY `last_mbs` is a route --
    # the reason is the container, not the guard -- but it is written in the
    # past tense, because a comment that cites a module a reader cannot open
    # costs that reader the same hour twice.
    #
    # W30 arm 2 measured it, all three ranks, 21 s into load:
    #   ResidentCarryError: PHASE-FLIP-CARRY 1 request(s) are reachable only
    #   through last_mbs/last_batch at the cutover: ['56fddcc3c0ef...']
    #
    # THE GUARD IS RIGHT AND STAYS AS STRICT AS IT IS. Its docstring names the
    # correct remedy and forbids the tempting one: "a bug to raise, not a
    # carry to widen. Silently widening the harvest here would hide a broken
    # predicate behind a carry that appears to work." So the route is added to
    # the AUTHORITY -- the same "fix it at the one authority, not at the
    # consumers" rule the W27 fix above was built on, which simply stopped one
    # route short.
    for mb in getattr(scheduler, "last_mbs", []) or []:
        _take(mb)
    # #1202: `mbs` IS THE THIRD PER-SLOT ARRAY AND IT WAS NOT A ROUTE.
    #
    # `scheduler_pp_mixin.py:7556-7561` rebuilds `mbs`, `last_mbs` and
    # `running_mbs` as THREE DISTINCT arrays, and `mbs` is written
    # unconditionally on all three planning paths (`:4670`, `:4689`, `:4701`)
    # -- it is the batch a slot is CURRENTLY planning, which under
    # `event_loop_pp` is a request that is resident on this rank and in none
    # of the two arrays above. `scheduler_pp_mixin.py:5122-5127` states the
    # gap in writing.
    #
    # Added HERE for the same reason `last_mbs` was: the route belongs to the
    # AUTHORITY, never to the consumers. Dedup is by `id()` above, so the
    # alias case (`scheduler.py:8354-8362`, where `mbs[i]` and
    # `running_mbs[i]` are the same object) costs exactly nothing.
    for mb in getattr(scheduler, "mbs", []) or []:
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


def note_armed_residents(snapshot: Dict[int, object], scheduler) -> Dict[int, object]:
    """Union THIS instant's resident set into the armed-window ledger (#1202).

    THE MEASURED DEFECT THIS EXISTS FOR. Boot 9
    (boot_855_weg1b9_1116175f6d_0904_164023.log) read, at the arm, on all
    three ranks::

        log:1883 PP0 at-arm pp_to_tp: cur_slot_reqs=1 ...
        log:1886 PP1 at-arm pp_to_tp: cur_slot_reqs=1 ...
        log:1888 PP2 at-arm pp_to_tp: cur_slot_reqs=1 ...

    ``cur_slot_reqs`` IS ``len(_live_reqs(scheduler))`` (see ``_pool_census``),
    so the flip's residency authority saw one request on every rank. That
    label was the #1205 defect and is now printed as ``live_reqs``; the lines
    above are quoted from boot 9 and keep the label that boot emitted. One
    second later the release ran its OWN ``_live_reqs`` and reported
    ``1 / 0 / 0`` retracted (log:2203/:2210/:2213), the rows stayed locked
    (log:2189/:2194, ``67 row(s) still locked after a drop that evicted 0``)
    and the cutover died at ``ReqPoolRebindRefused: 1 of 8 rows are still
    held`` (log:2305/:2351). The same authority, the same rank, two answers
    one second apart, and NOTHING RECONCILED THE TWO.

    THE LEDGER IS CUMULATIVE OVER THE WHOLE ARMED WINDOW, not a pair of
    readings at two instants. Neither endpoint alone is sufficient: the
    request that killed boot 9 was visible at the arm and gone at the
    release, and a request admitted after the arm is visible at the release
    and absent from the arm. A union over every armed round is the only
    reading that covers both, and it is cheap -- the walk is the one the
    census already performs, bounded by the park deadline.

    Deliberately holds STRONG references. A request that has left every
    container is exactly the one this ledger exists to retract, and a weak
    reference to it may be gone by the time the seam asks. The ledger is
    cleared at every exit from the armed state, so it never outlives its
    flip -- the same discipline ``_parked_extent`` is under (#746 M5).

    Never raises: an instrument at the seam may cost a missing entry, never a
    flip.
    """
    try:
        for req in _live_reqs(scheduler):
            snapshot.setdefault(id(req), req)
    except Exception as exc:  # noqa: BLE001 - a ledger may never break a flip
        logger.warning(
            "%s #1202 armed-resident ledger could not be updated (%s); the "
            "cutover falls back to the release-instant enumeration, which is "
            "the pre-#1202 behaviour",
            LOG_PREFIX,
            exc,
        )
    return snapshot


def cutover_resident_set(scheduler, armed_snapshot=None):
    """The set the cutover retracts: live NOW, reconciled with the arm (#1202).

    Returns ``(reqs, report)``. ``reqs`` is the release-instant enumeration
    followed by every armed-window resident that has since become invisible
    AND still holds a request-pool row. ``report`` carries the arithmetic so
    the seam's log line can state what it reconciled instead of asserting it.

    THE FILTER IS THE WHOLE DESIGN, AND IT IS ASYMMETRIC ON PURPOSE.

    Under-retraction and over-retraction are not two sides of one error.
    Retracting too little leaves a row in the outgoing pool and the cutover
    STOPS, loudly, at ``phase_req_pool_binding.rebind_req_pool_for_cutover``
    -- which is what boot 9 did, and a loud stop is recoverable. Retracting
    too much frees a row its current owner still holds; the pool then hands
    one row to two requests, they share a ``req_to_token`` row and a mamba
    mapping entry, and nothing raises. That is a WRONG ANSWER, and this
    module's standing rule is that a wrong answer is worse than a loud
    failure. So a snapshot member is carried only when all of the following
    are true, and is dropped -- counted, never guessed -- otherwise:

    * it names a row at all (``req_pool_idx is not None``);
    * that row is NOT in the pool's free list. The pool's own free list is
      the authority on whether the row came back, exactly as it is for
      ``census_outgoing_req_pool``; a request that finished cleanly between
      arm and release is already free and must not be freed twice
      (``ReqToTokenPool.free_slot`` calls that "the row was returned twice");
    * no request that IS live now names the same row. That is the
      reallocation case: the snapshot member is a stale object and the row
      has a new owner, who is enumerated on the live half anyway.

    UNKNOWN IS NOT HELD. When the pool cannot be read -- no pool bound, no
    readable free list -- nothing is carried from the snapshot and the
    abstention is counted. Carrying blind would be the over-retraction
    branch, taken on the strength of a measurement that failed.

    The census at ``_pool_census`` deliberately keeps reporting the RAW
    authority reading. It is the instrument that made this gap visible, and
    an instrument that reports the reconciled set can no longer show the
    divergence it exists to find.
    """
    live = list(_live_reqs(scheduler))
    report = {
        "live_now": len(live),
        "from_arm_ledger": 0 if not armed_snapshot else len(armed_snapshot),
        "carried_from_arm": 0,
        "carried_rids": [],
        "skipped_no_row": 0,
        "skipped_row_free": 0,
        "skipped_row_reallocated": 0,
        "skipped_pool_unreadable": 0,
    }
    if not armed_snapshot:
        return live, report

    seen = {id(r) for r in live}
    live_rows = set()
    for r in live:
        idx = getattr(r, "req_pool_idx", None)
        if idx is not None:
            live_rows.add(int(idx))

    pool = getattr(scheduler, "req_to_token_pool", None)
    free_slots = getattr(pool, "free_slots", None)
    try:
        free_rows = None if free_slots is None else {int(x) for x in free_slots}
    except (TypeError, ValueError):
        free_rows = None

    out = list(live)
    for req in armed_snapshot.values():
        if id(req) in seen:
            continue
        idx = getattr(req, "req_pool_idx", None)
        if idx is None:
            report["skipped_no_row"] += 1
            continue
        idx = int(idx)
        if free_rows is None:
            report["skipped_pool_unreadable"] += 1
            continue
        if idx in free_rows:
            report["skipped_row_free"] += 1
            continue
        if idx in live_rows:
            report["skipped_row_reallocated"] += 1
            continue
        out.append(req)
        report["carried_from_arm"] += 1
        report["carried_rids"].append(str(getattr(req, "rid", "?"))[:8])
    return out, report


def seam_probe_reading_age(
    current_seq: Optional[int], stamped_seq: Optional[int]
) -> Optional[int]:
    """How many probe passes ago a remembered seam figure was measured (#846).

    ``_staging_affordable`` remembers figures the census cannot re-measure, and
    nothing ever clears them: ``_last_cache_promised_bytes`` and
    ``_last_cache_delivered_bytes`` are written ONLY inside the reclaim branch
    (:7512-7513) and assigned ``None`` nowhere in this module. So after the
    first reclaim in a process they are never ``None`` again, and the census's
    own contract -- "None when no reclaim was attempted in this pass" -- became
    unreachable. This is the age it could not state.

    ``None`` means NEVER MEASURED and must never collapse into ``0``: "no probe
    has recorded this" and "recorded in this very pass" are the two readings a
    census most needs to separate, and a ``getattr(..., 0)`` default is exactly
    how the first silently reads as the second.

    A stamp AHEAD of the counter also returns ``None`` rather than a negative
    age. That state means the counter was reset or never incremented (a
    stand-in, a partially wired object); reporting it as "never measured" is
    true and readable, while a negative number would read as a corrupt census.
    """
    if current_seq is None or stamped_seq is None:
        return None
    age = int(current_seq) - int(stamped_seq)
    return age if age >= 0 else None


def seam_probe_age_phrase(age: Optional[int]) -> str:
    """The age as the census says it. Never empty -- an empty phrase would
    restore precisely the silence #846 exists to end."""
    if age is None:
        return "never measured"
    if age == 0:
        return "measured this pass"
    if age == 1:
        return "measured 1 pass ago"
    return f"measured {int(age)} passes ago"


def releasable_cache_bytes_from_stats(stats, alloc_conf: str = "") -> Optional[int]:
    """Bytes an ``empty_cache()`` draw CAN hand the driver, or None (#852).

    Pure arithmetic over a ``torch.cuda.memory_stats()`` mapping, kept out of
    the device call so both directions are testable without a GPU -- the figure
    AND every abstention.

    ``reserved - allocated`` counts every free block, including blocks
    fragmented inside segments that still carry live allocations.
    ``empty_cache()`` releases only whole free segments, and
    ``inactive_split_bytes`` is exactly the trapped remainder, so the
    difference is what a draw can deliver.

    IT ABSTAINS RATHER THAN GUESSES, in three cases:

    * ``expandable_segments:True`` -- ``reserved`` then describes a VIRTUAL
      extent, not physical bytes. This tree measured it at 36910 MiB on a
      32607 MiB card (``phase_flip_spill``, "it cannot be compared to a
      physical budget at all") and refuses another feature outright on the
      same env (``adaptive_graph_memory`` :354). Subtracting under that config
      UNDER-reports, and an under-report here suppresses a draw that would
      have paid -- which would make the flip stickier, the exact defect #852
      exists to remove.
    * no ``inactive_split_bytes`` counter (cudaMallocAsync) -- without the
      trapped figure the only available number is the phantom promise itself.
    * nothing reserved -- there is no cache to price.

    Every abstention returns the caller to its pre-#852 behaviour exactly.
    """
    try:
        if "expandable_segments:True" in (alloc_conf or ""):
            return None
        reserved = int(stats.get("reserved_bytes.all.current", 0))
        allocated = int(stats.get("allocated_bytes.all.current", 0))
        inactive_split = int(stats.get("inactive_split_bytes.all.current", -1))
        if reserved <= 0 or inactive_split < 0:
            return None
        # Counters sampled without a lock can disagree by a block; floor at
        # zero so a transient over-subtraction reads as "nothing to collect"
        # rather than as a corrupt census.
        return max(0, reserved - allocated - inactive_split)
    except Exception:  # noqa: BLE001 - a measurement may abstain, never break
        return None


class SeamOrderError(RuntimeError):
    """A seam step ran before the step it depends on (#856)."""


class SeamMoverError(RuntimeError):
    """A pre-cutover mover failed INSIDE the no-return region (#1204)."""


def run_pre_cutover_movers(fns, direction: str, mark, note_failure=None) -> None:
    """Run the extra movers (weights arena, GDN state) before the cutover.

    #1204. This loop sits inside the seam's no-return region: the outgoing
    layout's backing is already on its way out and the arena refill is already
    rewriting weight pages. It never swallowed -- a raise here climbs out of
    ``_execute`` through a bare ``finally`` and kills the rank, which is the
    right answer under ``raenge-nie-uneins-crash-stop``: a rank that cannot
    complete the movers must not go on to serve from a half-refilled arena,
    which is the wrong-answer failure ``weights_arena.verify_boot_anchor``
    exists to refuse.

    WHAT WAS MISSING WAS THE NAME. What climbed out was whatever the mover
    raised -- a ``RuntimeError`` from ``cuMemMap``, say -- with nothing saying
    which leg it was or that it happened after the region closed. The weights
    refill and the GDN state leg have different sizes and opposite fixes (the
    same reason the census labels them separately one line below), and from a
    boot log the two are indistinguishable without this.

    So: wrap, name the leg, record it where a post-mortem can read it BEFORE
    the raise, and re-raise. Never continue -- the movers behind a failed one
    would run over an arena that is already wrong -- and never mark a census
    label for a leg that did not finish.

    WHY THE GROUP IS NOT TOLD. The obvious wish is to reduce a failure flag so
    all three ranks refuse together, but there is no collective between this
    loop and ``self._cutover_fn`` to carry it: the seam's consensus MIN is
    upstream of the no-return point, ``_reconcile_trees_if_diverged`` only
    reads a verdict that reduction already produced, and the levelling's
    reduction runs on the tp->pp POST-cutover hook. A flag reduced there would
    announce the failure after the cutover it was supposed to prevent, on one
    leg of two. Opening a NEW collective inside the no-return region is the
    2026-08-08 boots 9/10 wedge shape and is not worth a message this rank can
    deliver by dying. So this stays a rank-local, loud, named death.
    """
    for fn in fns:
        label = str(getattr(fn, "census_label", "pre_cutover_fn"))
        try:
            fn(direction)
        except Exception as e:
            if note_failure is not None:
                try:
                    note_failure(label, e)
                except Exception:  # noqa: BLE001 -- a recorder, never a gate
                    logger.debug(
                        "%s #1204 mover-failure note failed", LOG_PREFIX, exc_info=True
                    )
            logger.error(
                "%s #1204 pre-cutover mover %r FAILED on %s, INSIDE the "
                "no-return region: the outgoing backing is already going and "
                "the arena refill is already rewriting weight pages, so this "
                "rank cannot serve and must not reach the cutover. The movers "
                "behind it are not run",
                LOG_PREFIX,
                label,
                direction,
                exc_info=True,
            )
            raise SeamMoverError(
                f"#1204 pre-cutover mover {label!r} failed on {direction} "
                f"inside the no-return region: {e}"
            ) from e
        # Labelled per mover: the GDN state leg and the weights refill have
        # different sizes AND different fixes, so one combined "pre_cutover"
        # bar would be unattributable. Marked only once the leg FINISHED.
        mark(label)


def release_residents_for_cutover(reqs, *, retract, reset_tree):
    """RETRACT STRICTLY BEFORE RESET. The order is the law (#856).

    Once the flip carries no KV, the new phase's device pool holds no valid
    rows while the radix tree still maps prefixes to row ids, so the tree must
    be dropped for a lookup to MISS and fall through to the host tier. That
    action has been tried and it took the instance down on all three ranks
    (2026-08-23, recorded at this module's #825 reconcile site):

        cache_finished_req -> dec_lock_ref
        -> full_component.py:239  `if cur.id in skip_lock_node_ids`
        AttributeError: 'NoneType' object has no attribute 'id'

    with the cause stated there: "PARKED IS NOT UNREFERENCED. The cutover
    carries RESIDENT requests across, and each holds a `last_node` with a lock
    ref. `reset()` rebuilds the root, orphaning those nodes, so the parent walk
    in `dec_lock_ref` no longer terminates at the live root and runs off the
    top into None."

    THE NO-CARRY RULE REMOVES THE PRECONDITION rather than working around it.
    Retraction releases every resident request's rows AND its tree lock ref
    (`release_req` -> `release_kv_cache` -> `cache_finished_req` ->
    `dec_lock_ref`, whose loop is `while node != self.root_node: node =
    node.parent`). Run it FIRST and every walk terminates at a root that is
    still live; run it after `reset()` and it walks off exactly as #825 did.

    So this function exists to make that ordering a named, testable property
    instead of a comment two callers have to remember. It is deliberately
    thin: the value is the order and the refusal, not the work.

    Refuses rather than repairs when either step is missing. A seam that
    silently skipped the retraction would leave locked nodes for the reset to
    orphan -- the crash -- and one that silently skipped the reset would leave
    the tree pointing at rows that no longer hold KV, which is a WRONG-ANSWER
    failure rather than a loud one and therefore worse.
    """
    if not callable(retract) or not callable(reset_tree):
        raise SeamOrderError(
            "the cutover needs BOTH a retraction and a tree reset: retracting "
            "without resetting leaves the tree naming rows that hold no KV, "
            "and resetting without retracting orphans locked nodes (#825)"
        )
    retracted = retract(reqs)
    reset_tree()
    return retracted


def tree_evictable_full_rows(tree) -> Optional[int]:
    """How many FULL-pool rows the prefix tree can still hand back, or ``None``.

    ONE CLOCK, AND IT IS THE BASE-CLASS CONTRACT. This read used to be
    ``getattr(tree, "full_evictable_size_", 0)`` -- a trailing-underscore
    attribute that is an implementation detail of exactly three of the caches
    in this tree (``MambaRadixCache``, ``SWARadixCache``,
    ``HiMambaRadixCache``) and does not exist on ``UnifiedRadixCache``, which
    keeps the same quantity in ``component_evictable_size_[BASE_COMPONENT_
    TYPE]``. So on the tree this rig actually runs, the read returned the
    number ZERO, the caller took that as licence to skip its eviction, and
    ``reset()`` orphaned every row the tree held.

    W29 MEASURED IT (SPECIMEN_w29_a1_pool_leak_1row.log, tree_type=
    UnifiedRadixCache in every census line). The seam's own #832 census named
    the orphan by id -- ``unaccounted=1 [1]``, flat from the first flip that
    crossed a non-empty tree onward -- and the scheduler's idle check killed
    all three ranks:

        pool memory leak detected! [full] total=469733, available=107041,
          evictable=1, protected=0, session_held=0, uncached=0,
          withheld=362690

    ONE row only because the tree held exactly one, a 1-token health check.
    The orphan is the SIZE OF THE TREE, not a constant, and it was read as a
    constant unit deficit -- and therefore as a reserved/boundary slot --
    because the only sample available held one row on both ranks.

    THE INDICATOR WAS THE BUG, which is why the fix is a reader and not a
    tolerance. ``full_evictable_size()`` is declared on ``BasePrefixCache``
    and implemented by every cache in this tree, and on the three
    attribute-keeping caches it returns that very attribute -- so the method
    is a strict superset of what the old read could see, never less. The
    suite's own double had the attribute and not the method, exactly backwards
    from production, which is how ten green tests survived the boot this
    killed.

    ``None`` means the tree could not answer, and that is NOT zero: zero is a
    licence to skip the eviction, and skipping it is the defect. The caller
    reports the abstention out loud instead of proceeding quietly.
    """
    reader = getattr(tree, "full_evictable_size", None)
    if not callable(reader):
        return None
    try:
        return max(0, int(reader()))
    except Exception:  # noqa: BLE001 - an unreadable count is not zero rows
        return None


#: #938: rows the drop left PROTECTED, summed over this process's cutovers.
#: Module-level because `drop_prefix_tree_returning_rows` is a free function
#: and the quantity only means anything as a running total: the specimen's
#: signature is a block that grows by one request's allocation per cutover,
#: which a per-drop number cannot show and a total can.
_PROTECTED_RESIDUE_ORPHANED_ROWS = 0
_PROTECTED_RESIDUE_ORPHANED_DROPS = 0


def tree_protected_full_rows(tree) -> Optional[int]:
    """How many FULL-pool rows the prefix tree holds LOCKED, or ``None``.

    THE OTHER HALF OF THE SAME BASE-CLASS CONTRACT. ``full_protected_size()``
    is declared on ``BasePrefixCache`` beside ``full_evictable_size()`` and
    implemented by every cache in this tree; on ``UnifiedRadixCache`` it
    returns ``component_protected_size_[BASE_COMPONENT_TYPE]``. Same reader
    shape, same abstention rule -- ``None`` means the tree could not answer
    and is NOT zero, because zero here reads as "nothing was orphaned" which
    is exactly the sentence this instrument exists to stop being assumed.
    """
    reader = getattr(tree, "full_protected_size", None)
    if not callable(reader):
        return None
    try:
        return max(0, int(reader()))
    except Exception:  # noqa: BLE001 - an unreadable count is not zero rows
        return None


def _protected_node_sample(tree, limit: int = 4) -> str:
    """Best-effort: name a few of the locked nodes, with their lock refs.

    Diagnostic garnish, never load-bearing -- the COUNT above is the finding.
    The root is skipped on purpose: ``_reset_full`` gives it ``lock_ref = 1``
    on every component by construction, so including it would report a
    permanent false positive on every drop.
    """
    try:
        nodes = tree._collect_all_nodes()
    except Exception:  # noqa: BLE001 - a sample may never break a seam
        return "unavailable"
    root = getattr(tree, "root_node", None)
    out = []
    try:
        for node in nodes:
            if node is root:
                continue
            locks = [int(cd.lock_ref) for cd in node.component_data]
            if any(lock > 0 for lock in locks):
                out.append(f"id={getattr(node, 'id', '?')} locks={locks}")
                if len(out) >= limit:
                    break
    except Exception:  # noqa: BLE001
        return "unavailable"
    return "; ".join(out) if out else "none"


def drop_prefix_tree_returning_rows(tree) -> int:
    """Empty the prefix tree AND return its rows, then reset it (#856).

    THE W27-RETRY DEFECT, derived from the tree code rather than guessed.
    `MambaRadixCache.reset` (mamba_radix_cache.py:555) installs a NEW
    `TreeNode()` as root and zeroes `full_evictable_size_` /
    `full_protected_size_`. It never frees a device row: the old tree is
    simply dereferenced, and the rows its nodes held are orphaned. It is a
    BOOKKEEPING reset, not a deallocation -- correct for a teardown where the
    pool is reset too, wrong for a seam that keeps serving.

    Measured on metal (boot_w27r_0824_1551.log, third retract+drop cycle):

        pool memory leak detected! [full] total=472864, available=126802,
          evictable=22, protected=0, session_held=0, uncached=0,
          withheld=345888

    126802 + 22 + 345888 = 472712 against 472864 -> 152 rows belonging to
    nobody, accumulating once per cycle. `evictable=22` is the NEW tree; the
    old tree's rows are gone from every owner's books, which is why the
    detector can see them only as a total mismatch.

    THE CALL THAT ACTUALLY RETURNS ROWS is `evict` -> `evict_full`, whose leaf
    path frees through `token_to_kv_pool_allocator.free`. So the drop is
    evict-then-reset: empty the tree by the route that pays the allocator
    back, and only then rebuild the root.

    NOT A FLUSH-BY-ANOTHER-NAME. Eviction here is legitimate precisely because
    the #703 fence has already persisted these prefixes to the canonical
    store; the new layout re-reads them. Without that fence this would be data
    loss, which is why the seam order is fence -> retract -> DROP and not any
    permutation of it.

    Returns the number of rows the tree reported evicting, for the seam's log
    line -- a drop that returns zero on a non-empty tree is exactly the defect
    this function exists to remove, and it has to be visible.

    Does not raise on a COUNT or EVICTION shortfall: it runs at the seam
    with requests already retracted, and an unreadable count or a refused
    evict is reported, never thrown. A failed ``tree.reset()`` is the one
    exception, and it PROPAGATES (#1068 slice 2 fix 4, A12.4 amendment
    (c)): since slice 2 the reset chain reaches
    ``UnifiedRadixCache._reset_full`` -> ``cache_controller.reset()`` ->
    ``_stop_storage_threads``, which raises RuntimeError on a torn storage
    pipeline AFTER ``_reset_full`` has already cleared ``enable_storage``
    and ``ongoing_prefetch``. Swallowing that left the rank serving with
    enable_storage False, the stop event set and no storage threads, so its
    next ``prefetch_from_storage`` returned before the #580 vote: a
    rank-divergent wedge on one rank instead of a crash-stop on all of
    them. raenge-nie-uneins (ranks-never-disagree): a torn rank crashes, it
    never compensates. The #856 line is still printed before the re-raise.
    """
    returned = 0
    evictable = tree_evictable_full_rows(tree)
    if evictable is None:
        # NOT DEFAULTED TO ZERO, and that default is the entire W29 defect.
        # A tree that cannot state its evictable size is a tree whose rows
        # this function is about to orphan, and it has to say so.
        logger.error(
            "%s #856: %s cannot answer full_evictable_size(); the drop below "
            "does not know what to evict and reset() will orphan whatever the "
            "tree still holds. Reading a count this code cannot get and "
            "calling the miss zero is the W29 defect verbatim -- give the "
            "tree the BasePrefixCache contract, do not default this",
            LOG_PREFIX,
            type(tree).__name__,
        )
    elif evictable > 0:
        try:
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            result = tree.evict(EvictParams(num_tokens=evictable))
            returned = int(getattr(result, "num_tokens_evicted", 0) or 0)
        except Exception:  # noqa: BLE001 - never abort a flip mid-drop
            logger.error(
                "%s #856: the prefix tree refused to evict %d row(s) before "
                "the drop; they will be orphaned and the pool census will "
                "report them missing",
                LOG_PREFIX,
                evictable,
            )
        else:
            # RE-READ, DO NOT TRUST THE ASK. `evict` walks leaves and can stop
            # short (a locked or un-backed node refuses, see #841), and the
            # difference between "asked for N" and "the tree is now empty" is
            # precisely the difference W29 could not see. Residue here is the
            # census's `unaccounted` one pass before it becomes fatal.
            residue = tree_evictable_full_rows(tree)
            if residue:
                logger.error(
                    "%s #856: the prefix tree still reports %d evictable "
                    "row(s) after a drop that asked for %d and was told %d "
                    "were evicted; reset() is about to orphan them and the "
                    "next idle check will read them as a pool leak",
                    LOG_PREFIX,
                    residue,
                    evictable,
                    returned,
                )
    # #938 INSTRUMENT ONLY -- IT MEASURES, IT NEVER FREES.
    #
    # THE RESIDUE GUARD ABOVE IS BLIND IN ONE DIRECTION. It re-reads
    # `tree_evictable_full_rows`, a metric that EXCLUDES locked rows by
    # definition, so protected residue reads back as 0 and the check designed
    # to catch "the tree still holds rows reset() is about to orphan" cannot
    # see the one kind of row that is guaranteed to still be held. `evict`
    # walks leaves and refuses locked nodes; `_reset_full` then sets
    # `component_protected_size_` to 0 and installs a fresh root without
    # freeing a single device row. So a protected row is dropped silently and
    # is visible to nobody until the pool census reads it as unaccounted.
    #
    # WHY THIS IS A COUNTER AND NOT A FIX. A node is still locked here mainly
    # because a write-through is IN FLIGHT against it -- a live reader that is
    # copying those exact device rows to the host. Releasing the lock and
    # evicting mid-flight would free the copy's source underneath it, which is
    # a use-after-free in the #913 IMA family: strictly worse than the leak it
    # would close. The row's release has to be hung off the write-through ACK
    # instead, and which mechanism does that is a decision for AFTER this line
    # has reported a number from metal, not before.
    #
    # WHAT THE NUMBER SETTLES. The 2j forensics (SPECIMEN-2026-08-27T1125Z-2j-
    # UNOWNED-FORENSIK.txt) measured an unowned block growing ~283 rows per
    # cutover -- one retracted request's full allocation, kv_allocated_len ==
    # kv_committed_len == cache_protected_len, no gap. If this line reports
    # that same quantity, the retract-donated anchor is the leak and the ACK
    # route is the fix. If it reports 0 while the block still grows, the loss
    # is somewhere else entirely and this hypothesis is dead. Logged
    # UNCONDITIONALLY, including the zero, because the negative reading is
    # what makes the comparison decisive rather than suggestive.
    global _PROTECTED_RESIDUE_ORPHANED_ROWS, _PROTECTED_RESIDUE_ORPHANED_DROPS
    protected = tree_protected_full_rows(tree)
    if protected is None:
        logger.error(
            "%s #938: %s cannot answer full_protected_size(); reset() is about "
            "to orphan any locked rows and this drop cannot say how many. An "
            "unreadable count is not zero rows -- give the tree the "
            "BasePrefixCache contract",
            LOG_PREFIX,
            type(tree).__name__,
        )
    elif protected > 0:
        _PROTECTED_RESIDUE_ORPHANED_ROWS += protected
        _PROTECTED_RESIDUE_ORPHANED_DROPS += 1
        # #1050: THE TEXT BELOW USED TO END "so they belong to nobody from here
        # on ... NOT freed here on purpose", and that sentence became FALSE the
        # moment the reclaim landed -- measured on boot 20 (2026-08-31), which
        # emitted 6 of these lines while every one of its 90 census readings
        # said `unaccounted=0`. Six "orphaned" events and zero orphaned rows.
        # Leaving the old wording would have handed the next reader six leaks
        # that did not happen, which is the instrument-text-lies class this
        # fork keeps paying for. This line now states what it MEASURES (residue
        # existed at the drop) and points at the line that says what HAPPENED
        # to it. The two must be read together, in that order.
        logger.error(
            "%s #938 PROTECTED RESIDUE AT DROP: %d row(s) still locked after "
            "a drop that evicted %d; `evict` refuses locked nodes and reset() "
            "zeroes the protected book without freeing them. Whether these "
            "rows are RETURNED or orphaned is decided by the `#1050 CUTOVER "
            "ROW RECLAIM` line that follows on this same drop -- read it "
            "before reading this number as a leak. Sample: %s. "
            "(%d row(s) over %d drop(s) this process.) This site still frees "
            "nothing itself: a lock here can mean an in-flight write-through "
            "reading these very rows, and that gate lives in the reclaim.",
            LOG_PREFIX,
            protected,
            returned,
            _protected_node_sample(tree),
            _PROTECTED_RESIDUE_ORPHANED_ROWS,
            _PROTECTED_RESIDUE_ORPHANED_DROPS,
        )
    else:
        logger.info(
            "%s #938 protected residue at drop: 0 row(s) (evicted %d). The "
            "negative reading is logged so a growing unowned block can be "
            "attributed away from the retract-donated anchor.",
            LOG_PREFIX,
            returned,
        )
    # #1050: THE DROP NOW KEEPS ITS OWN CONTRACT, under a CHECKED premise.
    #
    # Everything above measures; this returns. `evict` refuses locked nodes and
    # `reset()` then zeroes the protected book without freeing a row, so the
    # rows that were locked here were leaving with nobody owning them -- a
    # MONOTONE ratchet, one drop's worth per flip cycle, fatal at the first
    # genuine idle when `on_idle`'s pool invariant reads the total mismatch.
    #
    # MEASURED, five boots, no counter-instance (2026-08-31): zero load-backs
    # -> zero orphans -> `unaccounted=0` (boots 1043b, 1043c); load-backs > 0 ->
    # orphans -> ratchet (1046cut 3/18/54626, 1048fix 69/21/43803, 1049n9
    # 30/6/21608, as loadback-genuine / orphan-events / max-unaccounted). The
    # orphan sizes are LITERALLY the load-back prefix lengths (5834, 4618, 350),
    # rank-uniform on PP0/PP1/PP2, and the last census value before death is
    # identical to the size of the killer's `leaked_full_pages` set (21608 =
    # 21608, contiguous 1..21608). Not a plausibility -- an identity.
    #
    # WHY IT IS SAFE TO FREE HERE NOW, when #938 correctly refused. #938's
    # premise was that the lock is a LIVE READER (an in-flight write-through
    # copying these exact rows), and freeing under it is a use-after-free in
    # the #913 IMA family. That premise is no longer assumed, it is TESTED:
    # `reclaim_rows_for_drop` frees only when `ongoing_write_through` is empty.
    # On boot_855_1049n9 every one of the 13 `#792 post-retract writeback
    # fence` lines reported `outstanding=0`, including the two drops that
    # orphaned 5834 and 4618 rows -- the in-flight explanation held on no drop
    # of that boot. When a copy IS outstanding the reclaim refuses and says so
    # with a number, which is exactly today's behaviour, kept.
    #
    # THE DEFERRAL IS INSTRUMENTED UNCONDITIONALLY, including the zero and
    # including the refusals. A clearer that can silently never run is how the
    # same ratchet comes back one level up, in a new pocket; this line is what
    # makes that visible on the drop it happens, not on the boot it kills.
    reclaim = {}
    try:
        reclaimer = getattr(tree, "reclaim_rows_for_drop", None)
        if reclaimer is not None:
            reclaim = reclaimer() or {}
    except Exception:  # noqa: BLE001 - a reclaim may never abort a flip
        reclaim = {"reason": "reclaim raised"}
    logger.warning(
        "%s #1050 CUTOVER ROW RECLAIM: reclaimed=%s full_rows=%s mamba_slots=%s "
        "(tree still held full=%s mamba=%s, already_free=%s) reason=%r -- "
        "rows the drop returns that `evict` could not, because the node was "
        "locked. reclaimed=False with a nonzero held count is the DEFERRED "
        "state and is the number to watch: it is the ratchet, per drop.",
        LOG_PREFIX,
        reclaim.get("reclaimed", False),
        reclaim.get("full_rows", 0),
        reclaim.get("mamba_slots", 0),
        reclaim.get("full_held", 0),
        reclaim.get("mamba_held", 0),
        reclaim.get("already_free", 0),
        reclaim.get("reason", ""),
    )
    try:
        tree.reset()
    except Exception as exc:
        # #1068 (A12.4 amendment (c), slice 2 fix 4): print the #856
        # instrument line, then PROPAGATE. The reset has already torn the
        # rank (root replaced, enable_storage cleared, storage threads
        # stopped); returning here would be compensation on one rank, the
        # raenge-nie-uneins (ranks-never-disagree) form this fork
        # crash-stops instead.
        logger.error(
            "%s #856: prefix tree reset failed after eviction (%s: %s); "
            "propagating, the rank is torn and must not serve",
            LOG_PREFIX,
            type(exc).__name__,
            exc,
        )
        raise
    return returned


def consume_retracted_from_live_universe(scheduler, reqs) -> int:
    """A retracted request must LEAVE the live universe, not merely be freed.

    THE W27 DEFECT, and it is the root the crash exposed. `retract_all`
    releases a request's KV rows, its mamba slot and its tree lock ref -- but
    the scheduler's batch structures keep REFERENCING the `Req`. `_live_reqs`
    is the one authority for "who is resident", and this function must clear
    every container that authority walks -- today: every `running_mbs`,
    `last_mbs` and `mbs` slot, `running_batch`, `last_batch`, and the
    out-of-batch `chunked_req`. None of them are touched by retraction, so
    every seam consumer that asks "who is live" still gets a request whose
    resources are gone.

    W27 (boot_w27_0824_1510.log) died on all three ranks one second after the
    retraction, in the first consumer to look:

        resident_mamba_slots (gdn_flip_mover.py:620)
        KvReshardError: PHASE-FLIP-GDN live request ... has no mamba slot
          -- refusing to flip past unmoved linear state

    That guard was RIGHT. The request really had no mamba slot, and refusing
    to flip past unmoved linear state is exactly what it should do. What was
    wrong is that the request was still being offered to it at all.

    SAME SHAPE AS #731's FIX, deliberately: there the carry had to CONSUME the
    queue entry rather than leave a request counted in two places. Here the
    retraction has to consume the request out of the resident set rather than
    leave it enumerable with freed resources. Freeing a resource and retiring
    the reference to it are two different jobs, and doing only the first is
    what produces a live object that every reader must special-case.

    FIXED AT THE ONE AUTHORITY, not at the consumers. `resident_mamba_slots`
    is the first reader to hit this and is explicitly not expected to be the
    only one; teaching each reader to skip freed requests would be the same
    defect once per reader, and the next reader added would reintroduce it.

    Uses `filter_batch(keep_indices=...)` rather than mutating `.reqs`,
    because a batch carries per-request tensors alongside the list and a raw
    list edit desynchronises them.

    Returns how many references were retired, for the seam's own log line.
    Never raises: it runs at the seam with requests already parked, and an
    exception here would abort a flip that has already released its state.
    """
    targets = {id(r) for r in (reqs or ())}
    if not targets:
        return 0
    retired = 0

    def _consume(batch) -> None:
        nonlocal retired
        if batch is None:
            return
        current = list(getattr(batch, "reqs", []) or [])
        keep = [i for i, r in enumerate(current) if id(r) not in targets]
        if len(keep) == len(current):
            return
        retired += len(current) - len(keep)
        try:
            batch.filter_batch(keep_indices=keep)
        except Exception:  # noqa: BLE001 - never abort a flip mid-release
            logger.warning(
                "%s #856: filter_batch refused a retracted-request removal; "
                "the seam continues and the stale reference is reported",
                LOG_PREFIX,
            )

    for mb in getattr(scheduler, "running_mbs", []) or []:
        _consume(mb)
    # W30: and it must be CLEARED from `last_mbs` too, not merely enumerated
    # there. Retracting a request the guard can still reach through
    # `last_mbs[slot]` leaves exactly the orphan the cutover refuses on --
    # freeing the resource and retiring the reference are two jobs, and this
    # route needs both.
    for mb in getattr(scheduler, "last_mbs", []) or []:
        _consume(mb)
    # #1202 REPAIR: `mbs` IS A ROUTE OF THE AUTHORITY, SO IT IS A ROUTE HERE.
    #
    # #1202 added `mbs` to `_live_reqs` and stopped there. That is the exact
    # asymmetry W30 closed for `last_mbs` -- and W30 closed it by adding the
    # route to BOTH walks, because freeing a resource and retiring the
    # reference to it are two jobs. Widened here alone, the retraction frees a
    # request the authority then re-enumerates: `resident_mamba_slots`
    # (gdn_flip_mover.py:620) sees a live request with `mamba_pool_idx is
    # None` and raises `KvReshardError`, which is the W27 death this function
    # exists to prevent, reached from inside the no-return region where the
    # only outcome is a dead rank.
    #
    # The shape is ordinary, not exotic: under `event_loop_pp` a freshly
    # planned prefill batch lands in `mbs[mb_id]` while `running_mbs[mb_id]`
    # still holds the decode set, so an mbs-only resident is the common
    # `pp_to_tp` case rather than a corner. The `id()` dedup in `_live_reqs`
    # and the no-op `_consume` on an unchanged keep-list make the aliased slot
    # free.
    for mb in getattr(scheduler, "mbs", []) or []:
        _consume(mb)
    for name in ("running_batch", "last_batch"):
        _consume(getattr(scheduler, name, None))
    # The chunked prefill is resident and in NO batch (#631 defect O), which is
    # why `_live_reqs` enumerates it separately -- so it has to be cleared
    # separately too, or a retracted chunked request stays live by that route
    # alone.
    chunked = getattr(scheduler, "chunked_req", None)
    if chunked is not None and id(chunked) in targets:
        scheduler.chunked_req = None
        retired += 1
    return retired


def build_cutover_release(scheduler):
    """The two callables `release_residents_for_cutover` needs, or None (#856).

    Bound HERE rather than inline at the seam so the seam reads as the ordered
    sequence it is, and so both halves can be exercised without a scheduler.

    ``offload_kv=False`` on purpose: that flag exists for decode-disaggregation
    to copy retracted KV device->host so it can be restored without recompute.
    Under this design the FENCE has already persisted the prefixes to the
    canonical store, so a second device->host copy at the seam would pay twice
    for the same bytes at the one instant the instance is blocked.

    Returns ``None`` when the scheduler cannot supply a tree cache or a
    resettable one. The caller REFUSES on that -- a flip that cannot drop its
    tree would enter the next phase with prefixes naming rows that hold no KV,
    which is a wrong answer rather than a loud failure.
    """
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is None or not callable(getattr(tree_cache, "reset", None)):
        return None

    def _retract(reqs):
        if not reqs:
            return []
        from sglang.srt.managers.schedule_batch import retract_all
        from sglang.srt.mem_cache.base_prefix_cache import (
            FORCE_HOST_WRITE_THROUGH_ATTR,
        )

        # #792 THE BOUNDARY SNAPSHOT, at the one boundary this seam owns.
        #
        # Retraction runs `release_req` -> `release_kv_cache` ->
        # `cache_finished_req` (see `release_residents_for_cutover`), so each
        # resident's mamba state is already donated to the tree at exactly the
        # position the flip interrupted it. The tree is then RESET two lines
        # later, which drops every device-only anchor -- so the anchor has to
        # be in the canonical store before it is dropped, or the re-admission
        # this seam logs ("the new layout re-admits them and serves the prefix
        # by read-through") reaches a node with KV and no recurrent state, the
        # match refuses it, and the request recomputes in full. That is the
        # #858 livelock, and it is why W40/W41 read `#cached-token: 0` on
        # every re-admission.
        #
        # `requests_forced_host_write_through` is exactly the right existing
        # mechanism and its docstring already describes this situation: "the
        # donated prefix is not a caching opportunity but the session's only
        # surviving copy". The hit-count write-through heuristic must not get
        # a vote here for the same reason it must not for a session hand-off.
        #
        # This route is the CANONICAL carrier (#1068: the per-request seam
        # copy that used to sit beside it is deleted; the store is the single
        # carrier). It deposits the anchor in the shared store keyed by the
        # prefix, where a re-admission -- or any other request sharing that
        # prefix -- can read it back.
        for r in reqs or ():
            try:
                setattr(r, FORCE_HOST_WRITE_THROUGH_ATTR, True)
            except Exception:  # noqa: BLE001 - a stamp may never break a seam
                pass

        out = retract_all(
            reqs=list(reqs),
            server_args=scheduler.server_args,
            req_to_token_pool=scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            hisparse_coordinator=getattr(scheduler, "hisparse_coordinator", None),
            offload_kv=False,
            # #969D: the cutover RETAINS. This is the one caller whose
            # retraction is a park-and-re-read, not a discard -- see
            # schedule_batch.release_req.
            retain=True,
        )
        # W30 SEAM STAMP. `Req.reset_for_retract` sets `is_retracted` /
        # `retracted_stain`, but those are NOT seam-specific: ordinary
        # decode-OOM preemption (`Req.retract_decode`), the PD prefill path
        # and the PP void path all set the identical two booleans through the
        # same shared helper. An exemption keyed on `is_retracted` would
        # therefore also exempt every OOM-preempted request's re-prefill,
        # which is real workload and must NOT be exempt.
        #
        # So the seam stamps its own retractions, here and nowhere else. This
        # closure is reached only from `build_cutover_release`, i.e. only from
        # the #856 no-carry cutover, which is exactly the population whose
        # re-admission is flip TRANSPORT rather than work.
        #
        # RANK-UNIFORM BY CONSTRUCTION: the cutover is a group-unanimous
        # event and every rank retracts the same resident set, so the stamp
        # appears on the same rids on every rank in the same round. That is
        # the property the purity gate's branch requires (see the
        # rank-uniformity note at its call site in scheduler.py).
        epoch = getattr(getattr(scheduler, "phase_flip_runtime", None), "epoch", None)
        for r in reqs or ():
            try:
                r.seam_readmit_epoch = epoch
                # #906: a fresh retraction is a fresh grant. Without this a
                # request retracted by a SECOND cutover would arrive with the
                # first cutover's grant already spent and never transport.
                from sglang.srt.managers.phase_purity import reissue_seam_grant

                reissue_seam_grant(r)
            except Exception:  # noqa: BLE001 - a stamp may never break a seam
                pass
        return out

    # #856 W27-retry: the drop must RETURN the tree's rows, not orphan them.
    # A bare `tree_cache.reset` is a bookkeeping reset and leaked 152 rows per
    # cycle on metal; see `drop_prefix_tree_returning_rows`.
    def _drop_tree():
        return drop_prefix_tree_returning_rows(tree_cache)

    return _retract, _drop_tree


def _writeback_fence_ms(report) -> Optional[float]:
    """The HiCache fence's own elapsed time, in ms, or None (#856).

    ``None`` means NO FENCE RAN and must never collapse into ``0.0``. The two
    readings a seam census most needs to separate are "this cost nothing" and
    "this did not happen" -- the fence is skipped outright without a canonical
    store, and a defaulted zero there would report a flip as fully fenced when
    nothing was persisted at all.

    Reads the report defensively rather than by attribute access, because a
    stand-in or a partially wired object is the ordinary state in tests and an
    instrument may never be the thing that breaks a flip.
    """
    if report is None:
        return None
    try:
        return float(report.elapsed_s) * 1000.0
    except Exception:  # noqa: BLE001 - an instrument, never a gate
        # Deliberately broad. A narrow (AttributeError, TypeError, ValueError)
        # was the first version and its own can-fail test killed it: a report
        # whose `elapsed_s` is a property that raises anything else would have
        # climbed out of here into the cutover, with requests already parked.
        # This runs on the flip path, where the module's standing rule is that
        # an instrument may cost a missing line and never a flip.
        return None


def _segment_pool_id(seg) -> tuple:
    """The private-pool id of one snapshot segment, ``(0, 0)`` for the general
    pool. Torch names the field ``segment_pool_id`` in the snapshot dict and
    ``owner_private_pool_id`` in the C++ ``SegmentInfo``; both spellings are
    accepted so a torch version bump degrades to "general pool" rather than to
    a wrong subtraction."""
    raw = seg.get("segment_pool_id", seg.get("owner_private_pool_id", (0, 0)))
    try:
        return tuple(int(x) for x in raw)
    except (TypeError, ValueError):
        return (0, 0)


def graph_pool_free_bytes_from_segments(segments) -> Optional[int]:
    """THE FOURTH TERM (#852 R3): free bytes trapped in CUDA-graph pools.

    W25 printed, five times, stable to the MiB:

        staging reclaim: driver free 2896 -> 2896 MiB (+0 returned,
                         predicted releasable 88 MiB)

    A nonzero prediction against a zero delivery, which #852's own text says
    "falsifies it and indicts this estimator instead". The phantom was down
    3.6x from W24's ~309-324 MiB, but 88 MiB of it survived, and it was
    STABLE while the general cache drifted 310 -> 305 MiB across the same
    seven passes. A term that does not move while its neighbours do is not
    fragmentation; it is a fixed structure.

    IT IS THE GRAPH POOLS. ``adaptive_graph_memory`` captures decode graphs
    into PER-TAG PRIVATE pools (:841 ``torch.cuda.graph_pool_handle()``,
    :873 ``with torch.cuda.graph(cuda_graph, pool=pool, ...)``), and this
    boot captured them: decode ``backend='full'``, bs 1..24. Torch's
    ``release_cached_blocks`` frees whole blocks from the GENERAL
    ``large_blocks``/``small_blocks`` pools only; a private pool is released
    solely through ``graph_pools_freeable``, i.e. once nothing references the
    graph. So while the graphs live, those bytes can NEVER be handed back --
    and ``reserved``/``allocated``/``inactive_split`` are device-global
    ``.all`` counters that COUNT THEM. The three-term arithmetic therefore
    promises bytes the driver will never see, which is exactly the observed
    88 MiB.

    Returns ``None`` when the snapshot cannot be read as segments, never 0 --
    "no verdict" and "no trapped bytes" are the two readings this whole
    ticket family exists to keep apart.
    """
    try:
        total = 0
        seen = False
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seen = True
            if _segment_pool_id(seg) == (0, 0):
                continue
            size = int(seg.get("total_size", 0))
            used = int(seg.get("allocated_size", 0))
            total += max(0, size - used)
        return total if seen else None
    except (TypeError, ValueError, AttributeError):
        return None


def releasable_cache_bytes_from_segments(
    segments, alloc_conf: str = ""
) -> Optional[int]:
    """What ``empty_cache()`` can return, computed SEGMENT BY SEGMENT (#852 R3).

    The three-term arithmetic is a proxy; this is the thing itself. Torch's
    ``release_blocks`` frees a block only when it is unsplit -- i.e. when its
    segment carries no live allocation at all -- and only from the general
    pools. So the exact answer is: sum ``total_size`` over segments that are
    (a) entirely free, (b) not in a private/graph pool, and (c) not
    expandable.

    THE ABSTENTIONS ARE UNCHANGED, deliberately. The expandable-segments
    abstention still keys on the ENV, not on the per-segment
    ``is_expandable`` flag, because #852's reason for it is that ``reserved``
    describes a virtual extent under that allocator and the whole comparison
    is void -- an under-report there suppresses a draw that would have paid
    and makes the flip STICKIER, the precise defect #852 exists to remove.
    Narrowing that abstention is a separate decision with its own evidence,
    not a side effect of naming the fourth term.

    ``None`` when there is nothing to read, so the caller falls back to the
    stats arithmetic exactly as before rather than treating an unreadable
    snapshot as an empty cache.
    """
    try:
        if "expandable_segments:True" in (alloc_conf or ""):
            return None
        total = 0
        seen = False
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seen = True
            if seg.get("is_expandable"):
                continue
            if _segment_pool_id(seg) != (0, 0):
                continue
            if int(seg.get("allocated_size", 0)) != 0:
                continue
            total += int(seg.get("total_size", 0))
        return total if seen else None
    except (TypeError, ValueError, AttributeError):
        return None


def _resident_rows(scheduler) -> Optional[set]:
    """KV rows held by RESIDENT REQUESTS -- the census's missing owner (#822).

    THE DEFECT THIS CLOSES, stated as the specimen states it.
    ``_pool_census`` names three owners: the allocator's free lists, the radix
    tree, and the cap's withheld band. A row that is none of those is printed
    as ``unaccounted``. But there is a fourth holder, and it is the most
    ordinary one on the stack: a row handed to an in-flight request is out of
    the free lists and not yet in the tree. It is OWNED, by that request, and
    the census had no term for it.

    boot_window1_0823_1204 shows the shape with nothing else moving. The FIRST
    census of the boot, before any cutover, printed ``unaccounted=122 [1..12]``
    against ``resident_reqs=1`` (the pre-#1205 label; today's line spells that
    term ``resident_slot_entries``) -- one live request, 122 rows, and every one of
    them read as a leak. That is not accretion; it is the working set.

    AND IT REFUTES THE RATCHET READING. The same boot's censuses go
    122 -> 122 -> 122 -> 0 -> 0 -> 0 (12:09:02 onward, all three ranks) and
    then up again as load returns. An unaccounted population that returns to
    zero is not a leak that never comes back; it tracks what requests hold.
    Naming the owner is what tells the two apart, which is the whole of #822.

    Returns ``None`` when the enumeration cannot be trusted -- no request pool,
    or a read that raised. ``None`` means "no verdict", NOT "no rows": handing
    an empty set to the authority would declare an owner that holds nothing and
    turn the working set back into a leak, which is the defect with an extra
    step. The skip rules mirror ``build_flip_live_slots_fn`` exactly (a request
    with no ``req_pool_idx`` owns no row in ``req_to_token``), because two
    enumerations of "which rows does this request hold" that can disagree is
    the shape #822 exists to end.
    """
    try:
        pool = getattr(scheduler, "req_to_token_pool", None)
        req_to_token = getattr(pool, "req_to_token", None)
        if req_to_token is None:
            return None
        rows: set = set()
        page = _page_size_of(scheduler)
        for req in _live_reqs(scheduler):
            if getattr(req, "req_pool_idx", None) is None:
                continue
            # #922: the ALLOCATOR's extent, not the sequence's. Slicing to
            # seqlen read one row past the allocation -- a stale cell from the
            # row's previous tenant -- and handed it to the census as
            # resident-owned. Downstream that is a false EXCLUSIVITY_DOUBLED
            # whenever the stale id is also in the free list or the tree
            # (#912 publishes that count onto the allocator and on_idle
            # consumes it), and a false ACQUITTAL whenever the stale id is a
            # genuinely unaccounted row that now looks owned.
            n = owned_row_extent(req, page)
            if n < 0:
                # No verdict, per this function's own contract: a partial or
                # wrong owner set turns the working set back into a leak.
                return None
            if n == 0:
                continue
            rows.update(int(r) for r in req_to_token[req.req_pool_idx, :n].tolist())
        return rows
    except Exception:  # noqa: BLE001 -- an instrument, never a gate
        return None


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

def flip_defer_budget_after(*, objected: bool, escalated: bool, prior: int) -> int:
    """One arm-gate verdict's effect on ONE guard's own defer budget (#1203 A5).

    THE WHOLE RESET POLICY, in one pure function, because the three guards that
    share it were three copies and the copies disagreed.

    ``objected``  -- this guard voted the flip down this round.
    ``escalated`` -- this guard stood aside at its limit and let the flip
                     through; the budget has been SPENT and starts over.

    A GUARD THAT MERELY CLEARED KEEPS ITS COUNT. That is the whole of #1203's
    A5 premise and it is the line most likely to be "simplified" back: the
    abandon is GROUP-unanimous, so this rank clearing while a peer objects
    tells this rank nothing. Zeroing here is what let three ranks take turns
    objecting without any of them ever reaching the limit -- the direction
    then defers for ever, which is the 411-abandon decode wedge reached
    through the mechanism that exists to prevent it.

    THE BUDGET IS STILL THIS GUARD'S OWN, though, which is where A5 itself
    went wrong: it spent the budget in ``_seam_abandons_in_a_row``, the book of
    EVERY abandon whatever caused it. Two of the three consumers escalate at
    their limit -- ``flip_host_headroom_verdict`` returns "PROCEEDING WITH EYES
    OPEN" under a failed #721 host floor, ``flip_seam_budget_verdict`` past the
    #830 F4 ceiling, and the writeback arm proceeds with an incomplete #703
    fence -- so three abandons for unrelated reasons disarmed all three guards
    and the next GENUINE shortfall proceeded on its first firing, having
    deferred zero times. A bound may be widened to the group's currency; an
    escalation that permits the hazard may not.

    The one legitimate refund is a COMPLETED flip, and that is booked at the
    group's own reset point in ``_execute_body`` rather than here, beside the
    abandon book it belongs with.
    """
    if objected:
        return int(prior) + 1
    if escalated:
        return 0
    return int(prior)


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
        return (
            True,
            False,
            (
                f"host headroom OK: {avail / 1e9:.2f} GB available >= "
                f"{need / 1e9:.2f} GB needed "
                f"(transient {int(projected_transient_bytes) / 1e9:.2f} + floor "
                f"{int(floor_bytes) / 1e9:.2f})"
            ),
        )
    if int(defers_so_far) >= int(max_defers):
        return (
            True,
            True,
            (
                f"{DEFERRED_HOST_RAM} ESCALATED after {defers_so_far} defers: "
                f"{avail / 1e9:.2f} GB available < {need / 1e9:.2f} GB needed "
                f"(transient {int(projected_transient_bytes) / 1e9:.2f} + floor "
                f"{int(floor_bytes) / 1e9:.2f}). PROCEEDING WITH EYES OPEN -- a "
                f"permanent hold would stop the instance alternating prefill and "
                f"decode, which is a certain half-service outage, while the kill "
                f"this defends against is recoverable by a restore."
            ),
        )
    return (
        False,
        False,
        (
            f"{DEFERRED_HOST_RAM}: {avail / 1e9:.2f} GB available < "
            f"{need / 1e9:.2f} GB needed (transient "
            f"{int(projected_transient_bytes) / 1e9:.2f} + floor "
            f"{int(floor_bytes) / 1e9:.2f}); defer {int(defers_so_far) + 1} of "
            f"{max_defers}, the flip is retried next round"
        ),
    )


#: #830 F4: how many consecutive seam-budget refusals before the guard stands
#: aside. Same bound and same reasoning as #721's FLIP_HOST_RAM_MAX_DEFERS: the
#: flip is how this instance alternates prefill and decode, so a PERMANENT
#: refusal converts a latency hazard into a certain half-service outage. This
#: guard exists to make a long seam NAMED, not to make the flip optional.
FLIP_SEAM_BUDGET_MAX_DEFERS: int = 3

#: The refusal reason, spelled once, so a boot log greps for one token.
SEAM_BUDGET_REFUSED = "SEAM-BUDGET-REFUSED"


def flip_seam_drain_budget_ms() -> int:
    """#830 F4: the seam drain budget in ms, from the environment.

    Read through ``envs`` so the default and its derivation live in one place
    (see ``SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS`` in environ.py: the 1094 ms
    maximum cutover observed across 1014 fault-free pre-integration flips).
    """
    try:
        from sglang.srt.environ import envs

        return max(0, int(envs.SGLANG_FLIP_SEAM_DRAIN_BUDGET_MS.get()))
    except Exception:  # noqa: BLE001 - a budget lookup must not break a flip
        return 0


def flip_seam_budget_verdict(
    projected_drain_ms,
    defers_so_far: int,
    budget_ms: Optional[int] = None,
    max_defers: int = FLIP_SEAM_BUDGET_MAX_DEFERS,
):
    """``(allow, escalated, detail)`` for one flip's seam-window budget (#830 F4).

    THE POINT IS THE NAME, NOT THE REFUSAL. ANALYSE_830 F4: "Nothing in the
    tree caps how long the seam may hold the ring. A named ceiling converts a
    silent 24-second exposure into a refusal that names itself." The measured
    exposure this answers is real -- single flips reaching 24892 ms and single
    cutovers 13742 ms, against a HiCache-off corpus where 1014 flips never
    exceeded 4155 ms total or 1094 ms of cutover.

    THE PROJECTION IS THE LAST MEASURED DRAIN, deliberately. It is the only
    honest projector available before the seam: the drain's duration is a
    property of the backlog on the controller's private streams, which nothing
    at arm time can enumerate. Using last-measured means the FIRST slow drain
    is never refused -- it is what teaches the guard -- and a persistently slow
    one is. That is a real limitation and it is stated rather than dressed up:
    this guard cannot catch a one-off spike, only a standing condition.

    ``projected_drain_ms`` None, or a budget of 0, means NO GUARD -- allow and
    say so. This mirrors #721's rule that a flip must never be refused on a
    fabricated number: a refusal is the thing with a service cost, so an
    unknown must not produce one.
    """
    if budget_ms is None:
        budget_ms = flip_seam_drain_budget_ms()
    budget_ms = max(0, int(budget_ms))
    if budget_ms == 0:
        return True, False, "seam drain budget disabled (0) -- guard stood down"
    if projected_drain_ms is None:
        return (
            True,
            False,
            "seam drain unmeasured -- guard stood down (no honest projection)",
        )
    projected = float(projected_drain_ms)
    if projected <= budget_ms:
        return (
            True,
            False,
            (f"seam drain OK: projected {projected:.1f} ms <= budget {budget_ms} ms"),
        )
    if int(defers_so_far) >= int(max_defers):
        return (
            True,
            True,
            (
                f"{SEAM_BUDGET_REFUSED} ESCALATED after {defers_so_far} "
                f"refusals: projected drain {projected:.1f} ms > budget "
                f"{budget_ms} ms. PROCEEDING WITH EYES OPEN -- a permanent "
                f"refusal would stop the instance alternating prefill and "
                f"decode, which is a certain outage, while a long seam is a "
                f"latency hazard. The exposure is now NAMED, which is the "
                f"whole difference from before this guard existed."
            ),
        )
    return (
        False,
        False,
        (
            f"{SEAM_BUDGET_REFUSED}: projected drain {projected:.1f} ms > "
            f"budget {budget_ms} ms; refusal {int(defers_so_far) + 1} of "
            f"{max_defers}, the flip is retried next round. The seam would "
            f"hold the ring for the drain on top of its own work."
        ),
    )


#: #834 A: the arm-time refusal reason, spelled once so a boot greps one token.
#: Distinct from ``SEAM_BUDGET_REFUSED`` on purpose -- that one refuses INSIDE
#: ``_execute``, after the group has already agreed to go through, and it
#: projects from the PREVIOUS flip's drain. This one refuses at ``arm``, before
#: anything is pending, on a drain THIS arm just measured.
PREARM_DRAIN_REFUSED = "PREARM-DRAIN-REFUSED"

#: #834 B: the two names a boot log needs to follow a deferred grow. The debt
#: line fires when a grow is still outstanding after the guard's patience; the
#: refusal fires when a rank would expose an id the group has not levelled to.
GROW_DEBT_UNPAID = "GROW-DEBT-UNPAID"
UNLEVELLED_EXPOSURE_REFUSED = "UNLEVELLED-EXPOSURE-REFUSED"


def _seam_shrink_master() -> bool:
    """#834: the master gate. Off by default; see ``SGLANG_SEAM_SHRINK``."""
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_SEAM_SHRINK.get())
    except Exception:  # noqa: BLE001 - a gate lookup must never break a flip
        return False


def _seam_shrink_half(name: str) -> bool:
    """One half of the shrink, with its own -1/0/1 override.

    The overrides exist so a GPU window can attribute a result to ONE half.
    Moving both at once and then reporting "the shrink helped" is the shape
    this family has twice paid for (ANALYSE_830's F2 was aimed at the wrong
    call for exactly that reason), so the ability to cut them apart ships
    with the feature rather than being added after the first ambiguous boot.
    """
    try:
        from sglang.srt.environ import envs

        override = int(getattr(envs, name).get())
    except Exception:  # noqa: BLE001 - an override lookup must not break a flip
        override = -1
    if override == 0:
        return False
    if override >= 1:
        return True
    return _seam_shrink_master()


def seam_shrink_prearm_quiesce_enabled() -> bool:
    """#834 A: is the #760 drain pulled forward to arm time?"""
    return _seam_shrink_half("SGLANG_SEAM_SHRINK_PREARM_QUIESCE")


def seam_shrink_defer_grow_enabled() -> bool:
    """#834 B: is the rank-local KV grow pulled out of the no-return window?"""
    return _seam_shrink_half("SGLANG_SEAM_SHRINK_DEFER_GROW")


def release_prearm_quiesce(holder, why: str) -> None:
    """#834 A: drop a pre-arm hold on a holder that may not have one.

    THE W12 CONVENTION, AND FOR THE SAME REASON IT WAS INVENTED. #795's arm
    epoch is read through ``pp_flip_epoch_of`` so that "a holder with no
    accessor keeps pre-#795 behaviour"; this is that rule applied to the seam
    shrink, and it is not defensive decoration. The abandon and disarm paths
    are driven throughout ``unit/managers`` by minimal duck-typed stubs -- a
    bare object carrying the four or five attributes the path under test
    touches -- because that is the only way to test an abandon without a live
    three-rank group.

    A NEW HARD ATTRIBUTE ON AN ABANDON PATH IS THE WORST PLACE TO PUT ONE.
    That path exists to survive trouble; making it raise AttributeError turns
    every abandon into a crash the moment a caller is anything other than a
    fully constructed runtime. Measured, not imagined: the first cut of this
    change called the method directly and took out 11 tests across
    test_phase_policy.py and test_pp_presence_withholding_deadlock_800.py,
    every one of them a pin on "this rank abandons instead of wedging".
    """
    fn = getattr(holder, "_release_prearm_quiesce", None)
    if fn is None:
        return
    fn(why)


def prearm_quiesce_held(holder) -> bool:
    """#834 A: is a pre-arm hold up? False on a holder that cannot say.

    Same convention as ``release_prearm_quiesce``. False is the safe answer
    here in the precise sense that matters: it means the #760 insurance clear
    still runs, which is the shipped behaviour.
    """
    fn = getattr(holder, "_prearm_quiesce_held", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 - insurance must not depend on a predicate
        return False


def pay_deferred_grow(holder) -> None:
    """#834 B: run a booked grow on a holder that may not book any."""
    fn = getattr(holder, "_pay_deferred_grow", None)
    if fn is None:
        return
    fn()


def seam_shrink_grow_debt_rounds() -> int:
    """#834 B step 4: rounds of patience before an unpaid grow is shouted."""
    try:
        from sglang.srt.environ import envs

        return max(0, int(envs.SGLANG_SEAM_SHRINK_GROW_DEBT_ROUNDS.get()))
    except Exception:  # noqa: BLE001 - never break a flip on a patience lookup
        return 0


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
    #
    # ------------------------------------------------------------------
    # #830 F3 -- THE TRIGGER ABOVE HAS FIRED. THIS IS THE ANSWER TO IT.
    # 2026-08-23. See /spinning/evidence-665-f1/ANALYSE_830_flip_regression_
    # attribution.md, sections 6.1 and 8 (candidate F3).
    #
    # The author of d4e71e64cf (2026-08-17 13:56) wrote the sentence directly
    # above as this clause's own exit condition:
    #
    #     "If it wedges again: restore the clause, and do not accept a green
    #      mock suite as grounds to lift it a third time."
    #
    # It wedged again: #767, #771, #787, #796, #798, #801, #802, and W12. The
    # antecedent is not in dispute, and an author-set trigger that fires and
    # goes unanswered is worse than no trigger. So it is answered here, in the
    # tree, at the clause it governs.
    #
    # DECISION: THE REFUSAL IS NOT RESTORED. Recorded with its reasons so a
    # later reader can overturn it on evidence rather than re-litigate it.
    #
    # 1. THE CLAUSE IS NO LONGER AVAILABLE AS A REMEDY. Three standing user
    #    laws now require this exact combination to run: default serving always
    #    carries the full feature set; sglang never boots on anything but the
    #    newest tree; and HiCache must be phase-uniform, with a per-phase cache
    #    geometry named as a defect rather than an endpoint. Restoring the
    #    clause would not make the instance safe -- it would make it refuse to
    #    boot in its required configuration, converting a latency defect into a
    #    total outage. That is the same trade #721's own bounded defer already
    #    rejected: "a permanent hold is WORSE than the hazard".
    #
    # 2. THE MEASUREMENT SAYS THE CLAUSE WOULD NOT HAVE FIXED THE RIGHT THING.
    #    ANALYSE_830's verdict is exposure-window inflation, not a new hole:
    #    the timing-coupled cutover edges predate the integration, and with
    #    HiCache off, 1014 instrumented flips never exceeded 4155 ms total.
    #    What the integration changed is how long the seam holds the ring --
    #    32-50 ms of cutover became 2209-3773 ms (max 13742 ms). A gate that
    #    refuses the configuration hides that window; it does not shrink it.
    #
    # 3. WHAT IS BEING DONE INSTEAD -- the three postings that shrink the
    #    window the trigger was really complaining about:
    #      F2  take the KV-pool recovery out of the cutover. Measured, not
    #          assumed: the cutover term is `recover_kv_backing`, at 99% of it
    #          across 63 rank-flips in two independent boots. NOTE, because it
    #          corrects ANALYSE_830 section 8 itself: F2 was filed against
    #          #719's pool rebind, and that is WRONG -- the rebind's own step
    #          measures 0.25-0.7 ms mean (max 2.4 ms) on both sides of the
    #          flag, and `rebind_for_cutover` returns None by default anyway
    #          (hicache_phase_binding.py, `phase_flip_rebind_hicache`).
    #      F1  get the HiCache stream drain off the seam's critical path.
    #      F4  a NAMED seam budget: refuse to arm when the projected drain
    #          exceeds it, instead of silently exposing a 24-second window.
    #
    # 4. WHAT WOULD OVERTURN THIS. Not a green mock suite -- the author's
    #    second sentence stands and is honoured. Restore the clause if, after
    #    F1/F2/F4 are on metal, a boot still shows the seam holding the ring
    #    beyond the F4 budget, or the `PROXY LEFTOVER REFUSED` form recurs with
    #    the seam measured back inside its pre-integration band. Either would
    #    mean the window was not the mechanism, and then the configuration
    #    itself is what has to go.
    #
    # HONEST RESIDUE, so this is not read as a clean acquittal: `recover_kv_
    # backing` is not HiCache code, yet it costs 8.8 ms per cutover with
    # HiCache off and ~5000 ms with it on. HiCache is therefore implicated in
    # the inflation, but the MECHANISM by which its presence slows the pool
    # grow (cached-segment competition, pinned-host pressure, allocator
    # serialization) is NOT established here. That is an open question, not a
    # closed one, and it is the reason item 4 above keeps the clause a live
    # option rather than deleting it.
    # ------------------------------------------------------------------
    #
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


def _apply_phase_release(scheduler, direction: str) -> None:
    """#778 Posten 2: lend the prefill activation reserve to TP, take it back for PP.

    RANK-LOCAL AND COLLECTIVE-FREE, deliberately. Every rank books the same
    1024 MiB reserve at boot and reaches this point in the same cutover, so the
    loan is uniform by construction and needs no reduction to agree on. It also
    changes no EXPOSED id here -- it moves a target that the recovery converges
    to, and exposure is still raised only by the collective levelling that
    follows. So this cannot be the "one rank exposes an id a peer cannot map"
    fault, which is the failure this seam is most afraid of.

    Never fatal. A loan that cannot be priced or applied is simply not taken:
    the pool keeps exactly today's behaviour, which is the shipped path.
    """
    try:
        from sglang.srt.managers.kv_backing_relief import (
            PHASE_RELEASABLE_ACTIVATION_RESERVE_MIB,
        )
        from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

        rung = getattr(scheduler, KV_BACKING_RELIEF_ATTR, None)
        if rung is None or not hasattr(rung, "set_phase_release_rows"):
            return
        if direction == PP_TO_TP:
            rows = int(rung.default_phase_release_rows())
            if rows <= 0:
                return
            prev = rung.set_phase_release_rows(rows)
            if prev != rows:
                logger.info(
                    "%s phase-release LEND %s: %d rows (~%d MiB) of prefill "
                    "activation reserve offered to the KV pool for the TP "
                    "phase; the recovery below converges to it under the "
                    "corridor law and the arena ceiling. Returned on tp_to_pp.",
                    LOG_PREFIX,
                    direction,
                    rows,
                    int(PHASE_RELEASABLE_ACTIVATION_RESERVE_MIB),
                )
        else:
            prev = rung.set_phase_release_rows(0)
            # REPAY FIRST, then let the recovery run. See converge_phase_release:
            # recovery only grows, so the rows come back only because this call
            # shrinks them back -- lowering the number alone would silently
            # leave the prefill hazard unfunded.
            freed = rung.converge_phase_release()
            if prev or freed:
                logger.info(
                    "%s phase-release REPAY %s: loan %d -> 0 rows, %d MiB "
                    "returned to the prefill activation reserve before the "
                    "prefill leg resumes",
                    LOG_PREFIX,
                    direction,
                    int(prev),
                    int(freed) // (1024 * 1024),
                )
    except Exception as e:  # noqa: BLE001 - a loan is never worth a seam
        logger.warning(
            "%s phase-release step skipped (%s); the pool keeps its boot "
            "activation reserve, which is today's shipped behaviour",
            LOG_PREFIX,
            e,
        )


def seam_kv_recover(scheduler, reduce_fn, direction: str) -> None:
    """#834 B: the recovery, with the expensive half optionally out of the seam.

    ONE CALL SITE SHAPE FOR BOTH LEGS, so the two cannot drift apart. With the
    gate off this is exactly ``recover_kv_backing(scheduler,
    reduce_fn=reduce_fn)`` and nothing else -- the shipped path, byte for byte.

    WITH THE GATE ON, the halves are separated along the line the #830 F2
    design note draws, and the direction each half moves is NOT symmetric:

      * the COLLECTIVE levelling STAYS HERE, inside the no-return window. This
        is the load-bearing judgement and it is a refusal to be clever. The
        levelling is what stops one rank exposing an id a peer cannot map --
        which aborts all three inside ``store_kvcache``'s bounds assert -- and
        every rank reaches this point exactly once per cutover, unanimously,
        because the cutover itself is unanimous. Moving it to a rank-local
        cadence is the 2026-08-08 boots 9/10 PP wedge shape: a blocking
        reduction entered at a local cadence pairing with a peer blocked in a
        pipeline recv. It is also CHEAP -- one reduction over three integers
        and, at most, one non-allocating cap engage -- so moving it would buy
        almost nothing for that risk.
      * the RANK-LOCAL grow LEAVES. It is the cuMemCreate/cuMemMap work that
        #830 F2 measured at 99-101% of the cutover term, it touches no
        collective anywhere in kv_backing_relief.py, and it is therefore the
        one half whose timing is free.

    WHAT THE SEAM LEVELS TO WHEN THE GROW HAS LEFT. The pre-grow backing --
    which is the honest answer, not a degraded one. Every rank enters this
    cutover with the backing it actually has; levelling that is exactly the
    invariant the seam needs. The grow that follows adds BACKED rows which stay
    UNEXPOSED until a later collective raises the level, so at no instant does
    any rank offer an id the group has not agreed on.

    AND SKIPPING THE RECOVERY IS NOT AN OPTION, which is why the deferral is
    booked as a DEBT rather than dropped. ``recover_kv_backing_on_abandon``'s
    docstring records what that costs: the cap never lifts, the pool sits at
    26.8% of its id space for the life of the process, and a user is served
    overloaded_error against a pool sized 3.7x larger (#814). The debt is
    counted, has a deadline, and says so loudly when it is not paid.
    """
    from sglang.srt.managers.phase_flip_spill import (
        level_kv_backing_to_group,
        recover_kv_backing,
    )

    # #778 Posten 2: SET THE LOAN BEFORE THE RECOVERY, ON BOTH LEGS.
    #
    # The prefill activation reserve (1024 MiB/rank, model_runner_kv_cache_mixin
    # .py:2171-2176) covers a hazard that only exists in prefill -- the
    # DCP-extend prefix-gather scratch. Under strict purity there is no prefill
    # in the TP phase, so on pp->tp those rows are lent to the KV pool and on
    # tp->pp they are returned BEFORE the leg that re-enters prefill completes.
    #
    # ORDER IS THE SAFETY ARGUMENT, and it is why both calls sit here rather
    # than at the two direction arms: the lend must be visible to the recovery
    # that follows (which is the only thing that grows the backing), and the
    # repayment must happen BEFORE that same recovery, so the grow converges to
    # the repaid level instead of re-granting a loan that was just cancelled.
    # One function, both legs, one order -- the same reason this function
    # exists at all ("ONE CALL SITE SHAPE FOR BOTH LEGS").
    _apply_phase_release(scheduler, direction)

    runtime = getattr(scheduler, "phase_flip_runtime", None)
    if not seam_shrink_defer_grow_enabled() or runtime is None:
        # THE SHIPPED PATH, and the ``runtime is None`` half of that condition
        # is not defensive noise: without a runtime there is nobody to hold the
        # debt or to run the deferred grow, and a grow deferred to nobody is
        # #814's ratchet with extra steps. No runtime, no deferral.
        recover_kv_backing(scheduler, reduce_fn=reduce_fn)
        return
    level = level_kv_backing_to_group(scheduler, reduce_fn)
    runtime.book_deferred_grow(direction, level)


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
        #
        # #830 F5: the marks below used to end the whole 7b..8 span in ONE
        # bucket named ``draft_state``, and that bucket carried 99.98% of the
        # cutover term in every boot measured (PP0, mean ms):
        #
        #   boot_735_nohc2  HiCache off  cutover  50.2  draft_state  18.9
        #   boot_735_hc1    HiCache on   cutover 3448.9 draft_state 3417.3
        #   boot_735_acc767 HiCache on   cutover 3773.0 draft_state 3741.2
        #
        # while ``verify+publish+trace`` -- which contains #719's pool rebind,
        # the term ANALYSE_830 F2 named as the leading suspect -- measured
        # 0.25-0.7 ms mean, max 2.4 ms, on BOTH sides of the flag. So the
        # single-bucket shape was actively misleading: it hid the only step
        # that moves behind a label naming a step that does not. The span is
        # now split at its own call boundaries, so the next boot attributes
        # the term by measurement instead of by reconstruction.
        #
        # The label ``draft_state`` therefore no longer appears on its own;
        # it is the SUM of retune + spill_rung2 + rung4_cold_stack +
        # draft_bootstrap + kv_recover + spec_clear + relay_reseed +
        # abort_drain, which is how historical logs stay comparable.
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
        # #969: NO ORPHAN ASSERT, NO IDENTITY SNAPSHOT. Both policed the
        # CARRY -- "every request resident at a cutover must survive it as the
        # same object". Under the user design a resident does not survive the
        # cutover at all: it is retracted, everything is zeroed, and it comes
        # back through the ordinary queue with its prefix served by a HiCache
        # read. Asserting object survival across a re-entry asserts the very
        # thing the design removes.
        # WHAT init_pp_loop_state IS ABOUT TO DESTROY, on the record.
        # It clears pp_outputs, last_rank_comm_queue and send_output_work
        # with no drain and no carry. The request side has a carry and a
        # membership pin; the OUTPUT side has neither, and a discarded
        # output is a token the client never sees (#631). Quiescence is
        # supposed to make all of these empty -- this line is what says so
        # out loud instead of assuming it.
        #
        # #800 CORRECTION: this sentence used to include "and the tensor-dict
        # inbox", and that stopped being true at #753, which moved the inbox
        # off the scheduler onto the pp_group so the crossing wire could share
        # it. init_pp_loop_state explicitly does not touch it any more
        # (scheduler_pp_mixin.py, "NOT assigned here any more"), and nothing
        # else did either -- so this instrument was counting an inbox under the
        # heading CUTOVER DISCARDS while the cutover discarded nothing, and a
        # message parked there outlived the whole TP phase and was handed to
        # the next PP epoch's receive. pp_flip_retire_pp_loop_stash below makes
        # the discard real, and names what it discarded.
        #
        # AND THE COUNT IS SPLIT BY DISPOSITION, because one number here meant
        # three different fates. The heading says these are sampled tokens that
        # reach no output_ids; that is true of an OWED entry and false of a
        # PP-loop-only one, which is not a token at all and is retired below on
        # purpose. Reporting them as one figure made a routine retirement look
        # like data loss and real data loss look routine.
        from sglang.srt.managers.pp_stash_disposition import census_stash

        try:
            _stash = census_stash(getattr(scheduler, "_pp_tensor_dict_inbox", {}))
            _inbox_owed = _stash.blocking_total + _stash.undeclared_total
            _inbox_pp_loop = _stash.gate_blind_total
        except Exception:  # noqa: BLE001 - an instrument may never break a flip
            _inbox_owed = 0
            _inbox_pp_loop = 0
        _inflight = (
            getattr(scheduler, "pp_outputs", None) is not None,
            len(getattr(scheduler, "last_rank_comm_queue", None) or ()),
            len(getattr(scheduler, "send_output_work", None) or ()),
            _inbox_owed,
        )
        # #795: AND WHAT IT CANNOT SEE. Every probe above reads a structure
        # inside this process. The tensor-dict WIRE is not one of them, and an
        # undelivered proxy sitting on it survives this line, survives
        # init_pp_loop_state below, and is then handed to the rebuilt ring by
        # slot number -- the 2026-08-21 06:10:48 mispair. Boot instr15 logged
        # "output path empty at cutover" 72 times while exactly that was
        # happening. The CHAN_DICT sent/consumed counters are the one reading
        # of the wire this rank can take without touching the transport (the
        # corpse-F rule), so take it and report it: this is EVIDENCE, not a
        # gate. The correctness fix is that the proxy stamp now names the flip
        # epoch, so a message that does cross the cutover can no longer be
        # mistaken for one of the new ring's (scheduler_pp_mixin.py,
        # pp_proxy_stamp_names_pass).
        _wire_gap = 0
        _counters = getattr(scheduler, "pp_flip_counters", None)
        _upstream_fn = getattr(scheduler, "_pp_flip_upstream", None)
        if _counters is not None and _upstream_fn is not None:
            try:
                from sglang.srt.managers.phase_flip_counters import CHAN_DICT

                _wire_gap = max(
                    0,
                    int(_counters.sent(CHAN_DICT, _upstream_fn()))
                    - int(_counters.local_consumed(CHAN_DICT)),
                )
            except Exception:  # noqa: BLE001 - an instrument may never break a flip
                _wire_gap = 0
        if any(_inflight) or _wire_gap:
            logger.warning(
                "%s CUTOVER DISCARDS IN-FLIGHT OUTPUT: pp_outputs=%s "
                "last_rank_comm_queue=%d send_output_work=%d inbox_owed=%d "
                "unconsumed_on_wire=%d -- each is a sampled token that reaches "
                "no output_ids, and an unconsumed tensor dict outlives the "
                "slot ring this cutover is about to rebuild (#795). Separately, "
                "inbox_pp_loop=%d message(s) are retired by design below: their "
                "only consumer is the PP loop body, they carry no token, and "
                "the ring they name is about to be rebuilt (#800)",
                LOG_PREFIX,
                *_inflight,
                _wire_gap,
                _inbox_pp_loop,
            )
        else:
            logger.info(
                "%s output path empty at cutover, and no unconsumed tensor "
                "dict on the wire"
                + (
                    f" (inbox_pp_loop={_inbox_pp_loop} retired by design below)"
                    if _inbox_pp_loop
                    else ""
                ),
                LOG_PREFIX,
            )
        # #800: retire the PP-loop-only stash BEFORE the ring is rebuilt. Its
        # messages name a pass in the ring this call is about to destroy, so
        # after this point no consumer can ever take them; leaving them would
        # hand them to the NEXT PP epoch's receive. Getattr for the same reason
        # #787 uses it here: these paths are driven in tests by stand-ins that
        # carry only the fields the logic reads.
        _retire_fn = getattr(scheduler, "pp_flip_retire_pp_loop_stash", None)
        if _retire_fn is not None:
            try:
                _retire_fn()
            except Exception as exc:  # noqa: BLE001 - a sweep may never kill a flip
                logger.warning(
                    "%s #800 cutover stash retirement failed: %s", LOG_PREFIX, exc
                )
        scheduler.init_pp_loop_state()

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
        # Both directions: a carried batch's OWN spec_algorithm field still
        # says which phase BUILT it, and prepare_for_decode branches on it.
        # Retune before the bootstrap, so the batch and the scheduler agree
        # about the phase before anything reads either.
        # #861c: THE THIRD MEMBER OF THE SAME CLASS. `spec_algorithm` and
        # `spec_info` each got a bespoke handler; `batch_is_full` latched
        # unnoticed for a whole window because the class had never been named.
        # Reset here, beside its siblings, so the three are handled in one
        # place and a fourth member has an obvious home.
        from sglang.srt.managers.phase_flip_draft_bootstrap import (
            arm_draft_bootstrap_all_reachable,
            clear_spec_info_for_unspeculated_phase,
            reachable_batch_count,
            reset_stale_batch_flags,
            retune_carried_batches_for_phase,
        )

        # #861i: the step budget is PER PHASE, so the cutover resets it.
        # Without the reset the counter would run away and the anti-chop floor
        # would stop protecting after the first phase.
        scheduler._decode_steps_this_phase = 0

        # #962a THE RECEIPT IS UNCONDITIONAL, and the reach is part of it.
        # This line is the REACHABILITY PROBE `cutover_participants.py`
        # registers for `latched_batch_flags`, and until now it was emitted
        # only when something was cleared -- so "the hook ran and found
        # nothing" and "the hook never ran" were byte-identical, which is the
        # #719 shape the registry's own docstring forbids. Settling the
        # window-958 `batch_is_full=1 at running=0` question needed exactly
        # this distinction and had to borrow it from unrelated lines.
        # `reached` is reported because W37-C proved a bare zero is not
        # enough: reach 0 is the blind case, reach N with nothing cleared is a
        # genuine all-clear, and they must not read the same.
        _reached = reachable_batch_count(scheduler)
        _stale = reset_stale_batch_flags(scheduler)
        logger.info(
            "%s #861c latched batch flags inspected across the seam: "
            "reached=%d cleared=%s. Every clear site for these is a FINISH "
            "path and #856 RETRACTS instead of finishing, so a flag set "
            "before the cutover would otherwise refuse admission for ever "
            "(W37-C: batch_is_full=1 with running=0 and avail=468981). "
            "reached=0 means this probe saw nothing and proves nothing.",
            LOG_PREFIX,
            _reached,
            _stale,
        )

        retuned = retune_carried_batches_for_phase(scheduler, want_spec_algo)
        if retuned:
            logger.info(
                "%s retuned %d carried batch(es) to spec_algorithm=%s",
                LOG_PREFIX,
                retuned,
                want_spec_algo,
            )

        _mark("retune")

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

            _mark("spill_rung2")
            # 7b-ii. SPILL RUNG 4: the posts the BOOT deferred.
            #
            # BETWEEN the weight restore above and the bootstrap below, and
            # neither neighbour is arbitrary. The capture bakes draft
            # parameter addresses, so rung 2 must have put pages under them
            # first; the bootstrap arms graphs, so the graphs must exist.
            #
            # This is a ONE-TIME move, not a per-flip spill. The boot skipped
            # these posts because the PP phase it entered cannot execute a TP
            # or draft forward -- and that PP phase is what sizes the KV pool.
            # From the second pp->tp leg on this is a no-op.
            from sglang.srt.managers.phase_flip_boot import (
                restore_deferred_cold_stack,
            )

            if restore_deferred_cold_stack(scheduler, stacks):
                logger.info(
                    "%s rung 4: built the flip TP stack's deferred cold posts "
                    "(attention workspaces + decode graphs) at the first "
                    "pp->tp cutover; the KV budget was solved with this "
                    "credit already taken and the seam priced the restore",
                    LOG_PREFIX,
                )

            _mark("rung4_cold_stack")
            arm_draft_bootstrap_all_reachable(scheduler, want_draft)
            _mark("draft_bootstrap")

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
            # #834 B: routed through ``seam_kv_recover`` below, which is
            # ``recover_kv_backing`` unchanged with the gate off.

            seam_kv_recover(scheduler, reduce_fn, direction)
            _mark("kv_recover")
        else:
            # ``stacks.draft_worker``, not ``want_draft``: want_draft is None
            # on this leg by design (that is the point of the leg), while the
            # carrier is parked on the worker object the stacks still hold.
            if _ladder is not None:
                _ladder.on_enter_pp(stacks.draft_worker)

            _mark("spill_rung2")

            # The scheduler's KV pool is the PP layout's, so entering PP makes
            # it the ACTIVE pool again and any residency relief taken against
            # it during the TP phase must be handed back. A cap that is never
            # lifted is a permanently smaller pool, which the standing rule
            # forbids; recovering here bounds the reduction to one phase.
            # #834 B: routed through ``seam_kv_recover`` below, which is
            # ``recover_kv_backing`` unchanged with the gate off.

            # #656 C22-e: the grow is rank-local by necessity, the ID SPACE it
            # produces may not be. See recover_kv_backing.
            # ==============================================================
            # #830 F2 -- THIS CALL IS THE CUTOVER TERM. Measured, not guessed.
            #
            # ANALYSE_830 section 8 filed F2 against #719's pool rebind. That
            # was wrong, and the correction matters more than the original
            # claim: the rebind's own sub-step measures 0.25-0.7 ms mean (max
            # 2.4 ms) on both sides of the HiCache flag, and rebind_for_cutover
            # returns None by default anyway. The cost is HERE.
            #
            # THE MEASUREMENT (from the #690 marks, which were emitting in
            # every boot all along -- see the F5 note at the top of _cutover):
            #   cutover term, PP0 mean:  50.2 ms HiCache off, 3448.9 / 3773.0
            #   ms HiCache on. The whole of it lands in the 7b..8 span, and
            #   inside that span the gap between "rung 2 SPILLED the draft
            #   weights" and the next emitted line -- which contains ONLY this
            #   call -- tracks the cutover total at 99-101% across 63
            #   rank-flips in two independent boots (boot_735_acc767,
            #   boot_735_hc1), and falls to 0 s exactly when the cutover falls
            #   to ~9 ms. The indicator moves in BOTH directions.
            #
            # WHY IT IS STILL HERE, i.e. why this is a note and not a patch.
            # The obvious cut -- defer the recovery to the first post-flip
            # round -- was analysed against the code and NOT taken:
            #
            #   * recover_kv_backing is TWO halves. The expensive one,
            #     relief.recover() -> runtime_set_backing_rows -> cuMemCreate /
            #     cuMemMap (kv_backing_relief.py:2682, memory_pool.py:2972-3081,
            #     kv_vmm_backing.py:1297-1347), is 100% RANK-LOCAL: there is not
            #     one collective anywhere in kv_backing_relief.py. The cheap one
            #     -- the cap agreement at phase_flip_spill.py:1250-1320 -- is a
            #     collective through reduce_fn (production passes
            #     flip_collective_min, this file's cutover build site).
            #   * So the grow could move out safely. THE LEVELLING CANNOT MOVE
            #     WITH IT, and it cannot be dropped either: recover() ends by
            #     re-exposing ids against its OWN backing, and a corridor-bounded
            #     grow leaves ranks unequal (measured on this rig: 210944 /
            #     124928 / 131072 backed rows). An id one rank exposes and a peer
            #     cannot map aborts ALL THREE inside store_kvcache's bounds
            #     assert. The levelling is what prevents that, and it must
            #     therefore run before the pool is used again.
            #   * Running that collective at a post-cutover round cadence is
            #     precisely the shape on_round's own docstring was written
            #     against (the 2026-08-08 boots 9/10 PP wedge): a blocking
            #     reduction entered at a LOCAL cadence can pair with a peer
            #     blocked in a pipeline recv. "Just after a cutover" is
            #     approximately synchronized, and approximately is what that
            #     wedge punished.
            #   * And skipping a recovery has its own catastrophic mode, already
            #     paid for once: recover_kv_backing_on_abandon's docstring
            #     records the ratchet -- cap never lifted, pool at 26.8% of its
            #     id space for the life of the process, a user served an
            #     overloaded_error against a pool sized 3.7x larger.
            #
            # WHAT THE NEXT STRAND SHOULD DO, in order:
            #   1. Split recover_kv_backing into grow (rank-local) and level
            #      (collective) as separate callables. The seam then runs
            #      grow-then-level as today, with no behaviour change, and the
            #      split is provable by test before anything moves.
            #   2. Defer ONLY the grow, and keep the pool's exposure clamped to
            #      the pre-grow group level until the levelling runs, so no rank
            #      can expose an id a peer has not backed. That is the invariant
            #      to write the red arm against FIRST.
            #   3. Level on the seam's existing OFF-SEAM funding-verdict cadence
            #      (collective_kv_backing_relief -> apply_cap_agreement, already
            #      collective and already outside the no-return window), not on
            #      a fresh round hook. cap_proposal is documented as strictly
            #      non-allocating and self-healing in exactly this direction:
            #      "a rank that recovers raises its own proposal, and its peers
            #      follow by RELEASING a cap over pages they never gave up."
            #   4. Guard the ratchet explicitly: a deferred grow that has not
            #      drained after N rounds is a loud error naming #814's trap,
            #      the way the abort window's "drain missed" check does.
            #
            # This needs a GPU window to validate: the failure modes are a
            # three-rank abort and a silent permanent pool shrink, neither of
            # which a hermetic suite can observe. Shipping it on desk evidence
            # alone would be the "green mock suite" this file already refuses
            # to accept once, seventy lines up.
            # ==============================================================
            seam_kv_recover(scheduler, reduce_fn, direction)
            _mark("kv_recover")

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

            _mark("spec_clear")

            _mark("relay_reseed")

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

        _mark("abort_drain")
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

        # #1040: point the scheduler at the incoming phase's OWN request pool.
        #
        # Deliberately OUTSIDE the #719 try/except below, and deliberately NOT
        # gated on --phase-flip-rebind-hicache. The #719 argument -- "a refused
        # rebind is SAFE, the binding just does not move" -- is true of HiCache
        # because a stale HiCache binding has a disarmed state (#718 turns
        # device-tier I/O off and the phase behaves as it did before the
        # feature existed). A request pool has no such state: since the boot no
        # longer aliases the two stacks' `req_to_token`, a scheduler left on the
        # outgoing pool reads and writes a tensor the running phase never
        # touches, and every row id it hands out was minted by the wrong
        # allocator. So this runs on EVERY cutover and RAISES when it cannot.
        #
        # It also censuses the OUTGOING pool first (#919 on the request axis)
        # and clears the incoming one, so a row carried across the seam is
        # refused by the wrong-row guard rather than landing in range.
        from sglang.srt.managers.phase_req_pool_binding import (
            rebind_req_pool_for_cutover,
        )

        rebind_req_pool_for_cutover(scheduler, "tp" if tp_phase else "pp")
        # #1201 B3: rebuild the cross-iter relay for the phase being entered.
        #
        # HERE, and not at the stack swap 300 lines up, because the FutureMap is
        # stamped with BOTH of the things this seam replaces: the speculative
        # algorithm (swapped at `scheduler.spec_algorithm = want_spec_algo`) and
        # the request pool (rebound on the line above). Built between the two it
        # would carry the incoming algorithm and the OUTGOING pool -- half a fix
        # that still addresses the other phase's rows.
        #
        # The stamp that matters on the standing boot form is `spec_algo`. A
        # phase-flip instance parks speculation at NONE for the PP phase (#631),
        # `Scheduler.init_overlap` builds the map out of that NONE, and until now
        # nothing rebuilt it -- so `resolve_forward_inputs` read a NONE stamp on
        # every TP decode round and gathered last iteration's sampled token into
        # `batch.input_ids` on a batch whose input ids the spec worker owns. It
        # is on the LIVE path with `--disable-overlap-schedule`: the non-overlap
        # spec branch calls `resolve_forward_inputs(batch, self.future_map)`
        # outside the `if self.enable_overlap:` above it. Silent, because the
        # only assertion on that branch is gated on SGLANG_IS_IN_CI.
        from sglang.srt.managers.overlap_utils import build_future_map

        _old_map = getattr(scheduler, "future_map", None)
        scheduler.future_map = build_future_map(scheduler)
        logger.info(
            "%s #1201 future map rebuilt for the incoming phase: "
            "spec_algo %s -> %s, req_pool binding %s -> %s. The boot map is "
            "stamped with the PP phase's parked NONE and its consumer "
            "(resolve_forward_inputs) branches on that stamp.",
            LOG_PREFIX,
            getattr(_old_map, "spec_algo", None),
            scheduler.future_map.spec_algo,
            getattr(
                getattr(getattr(_old_map, "confidence_relay", None), "pool", None),
                "binding_tag",
                "<untagged>",
            ),
            getattr(scheduler.req_to_token_pool, "binding_tag", "<untagged>"),
        )
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
        # #1201: every holder of the request pool must now name ONE pool.
        # Placed AFTER the HiCache rebind so it sees the settled seam, and
        # OUTSIDE its try/except for the #1040 reason: a divergent request
        # handle has no disarmed state. The divergence it looks for cannot
        # fail on its own -- both phases' pools hold the same row count, so a
        # wrong-pool read lands in range and returns another phase's rows.
        from sglang.srt.managers.phase_req_pool_binding import (
            assert_req_pool_identity,
        )

        assert_req_pool_identity(scheduler)
        # #1201 B3: and the relay must name this phase too. Same argument as the
        # line above -- neither divergence fails on its own, so the check is the
        # only thing standing between a stale stamp and a phase that answers
        # wrongly. Placed here so a fifth holder is a loud stop at the seam
        # rather than a wrong decode input three rounds later.
        from sglang.srt.managers.overlap_utils import assert_future_map_identity

        # `_old_map` is what makes this probe able to fail here: the two stamp
        # arms are construction invariants immediately after a rebuild (see the
        # function's docstring), so the only seam-time evidence that the
        # rebuild ran is that the object is no longer the outgoing phase's.
        assert_future_map_identity(scheduler, previous=_old_map)
        # #1066: the #1025b re-issue that stood here is DELETED. Re-admission
        # itself now runs after this cutover (deferred by
        # `_release_residents_for_cutover`, executed by
        # `_post_cutover_readmit`), so its intake prefetch opens on the
        # current binding and there is nothing mis-stamped left to re-issue.
        trace_cutover(scheduler, direction)
        _mark("verify+publish+trace")
        # #690 capture A: one line, sorted by cost, so the 24x spread can be
        # ATTRIBUTED instead of guessed at. Emitted after the completeness
        # check so a cutover that failed verification never reports timings as
        # if it had succeeded.
        #
        # #830 F5: the literal token "#690" is part of the EMITTED text, not
        # only of this comment. ANALYSE_830 section 6.3 ran `grep -c '#690'`
        # over six boot logs, got 0, and concluded the marks "are not present
        # in these boots" -- so the cutover term was reported as UNMEASURABLE
        # (open item O4) and F2 was aimed at the rebind on that basis. The
        # marks were in fact emitting in every one of those logs, three per
        # flip; the grep was searching for a string that only ever existed in
        # the source. An instrument whose documented grep does not match its
        # own output is not an instrument. Keep the ticket token in the line.
        try:
            steps = [
                (_marks[i + 1][0], (_marks[i + 1][1] - _marks[i][1]) * 1000.0)
                for i in range(len(_marks) - 1)
            ]
            total_ms = (_marks[-1][1] - _marks[0][1]) * 1000.0
            worst = sorted(steps, key=lambda kv: kv[1], reverse=True)
            logger.warning(
                "%s [#690] CUTOVER SUB-STEPS %s total=%.1f ms | %s",
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


def build_phase_flip_runtime(scheduler) -> PhaseFlipRuntime:
    """Factory mirroring build_kv_reshard_runtime (kv_reshard.py): wires
    the scheduler's real state into PhaseFlipRuntime. Called lazily from
    the first scheduler round (house pattern); by then the boot builder
    has installed scheduler.phase_flip_stacks."""
    from sglang.srt.distributed.parallel_state import (
        get_phase_flip_group,
        get_world_group,
    )
    from sglang.srt.managers.kv_pressure_runtime import default_collective_min
    from sglang.srt.managers.kv_reshard import _dist_exchange
    from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence

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
        # #969 §W3: `pp_flip_service` is deleted. The armed window no longer
        # has a private service loop, because it no longer switches the
        # request chain off -- the chain keeps running and carries PP0's
        # decision, which is the only thing an armed rank ever needed from it.
        service_fn=None,
        channels_empty_fn=getattr(scheduler, "pp_flip_channels_empty", None),
        # (i) withhold presence until this rank's own forward is flushed,
        # so the flag means "I owe no send" rather than merely "I am
        # armed". This is now a condition that can be REACHED: the service
        # turn reaps the handle once the downstream's counter proves the
        # message consumed, where the pump could only ever fail to.
        owes_send_fn=getattr(scheduler, "pp_owes_chain_send", None),
        # #787 sender-side half: reap/count anything already posted on
        # CHAN_DICT one last time before an abandon clears local flip
        # state. See PhaseFlipRuntime.__init__'s flush_pending_sends_fn
        # docstring and _abandon_no_quorum / _abandon_unjoined_flip.
        flush_pending_sends_fn=getattr(
            scheduler, "pp_flip_flush_pending_dict_sends", None
        ),
        exchange=_dist_exchange(flip_tp.device_group, pp_view.device),
        pp_pool_view=pp_view,
        tp_pool_view=tp_view,
        live_slots_fn=build_flip_live_slots_fn(scheduler),
        ready_fn=build_flip_quiescence_fn(scheduler),
        # #1173: the arm PRECONDITION. Reads the scheduler's own launch
        # bookkeeping (`_pp_launched_pending`, written at the launch site in
        # scheduler_pp_mixin and discarded where the slot's result is
        # consumed) -- no second ledger, and the same set the #1020 void
        # guard already trusts.
        launched_passes_fn=build_launched_passes_fn(scheduler),
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
            # #771: STASH THE THRESHOLD FIGURE FOR THE POOL SOLVE. The
            # projection is exact but is only computable here, AFTER the pool
            # has been sized -- so this boot cannot spend it and records it for
            # the next one instead (write_seam_reserve carries it). The
            # THRESHOLD point is the basis on purpose: it is the floor of the
            # range and the same live set the gate's own arming threshold uses,
            # so it is the minimum a seam needs to ENTER. A rig that intends to
            # flip at high pool occupancy needs the larger figure, which the
            # lines just logged above give directly.
            try:
                runtime._staging_ask_at_threshold = int(
                    runtime.project_staging_bytes(PP_TO_TP, int(base))
                )
            except Exception as exc:  # a projection may never fail a boot
                logger.warning(
                    "%s staging ask not recorded for the pool solve: %r",
                    LOG_PREFIX,
                    exc,
                )
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


def _build_kv_backing_swap(scheduler, stacks, my_layers) -> WavedBackingSwap:
    return WavedBackingSwap(scheduler, stacks, my_layers)


def _build_gdn_leg(scheduler) -> Callable[[str], None]:
    from sglang.srt.managers.gdn_flip_mover import build_gdn_flip_mover

    return build_gdn_flip_mover(scheduler)


#: Consecutive ``checked=0`` cutovers that make the #719 stale-generation gate
#: an ALARM rather than an ordinary quiet stretch. Raised 2 -> 4 by #861e after
#: 24 fires on a healthy W37-D boot; kept at 4 by #1205, which repaired the
#: CONDITION that #861e described but never implemented.
STALE_GATE_ZERO_STREAK_ALARM = 4


def controller_device_queue_depth(cc) -> Optional[int]:
    """How many device-tier HiCache operations the controller is holding.

    #1205 -- THE PROBE THIS REPLACES COULD NOT RETURN FALSE. It read

        bool(cc.write_queue or cc.load_queue or cc.ack_backup_queue)

    ``write_queue`` and ``load_queue`` are plain lists
    (``cache_controller.py:716-717``) and are honestly falsy when empty, but
    ``ack_backup_queue`` is a ``queue.Queue`` (``cache_controller.py:801``) and
    ``Queue`` defines NEITHER ``__bool__`` NOR ``__len__`` -- so the object is
    truthy at every depth including zero. Whenever storage is enabled the
    ``or`` chain therefore ended on a constant ``True`` and the #861e gate
    ("count a zero-streak only while the controller reports device-tier work in
    flight") was never in the tree; the only thing that shipped was the
    threshold change.

    Counts, never tests: a ``Queue`` is read through ``qsize()``, a list
    through ``len()``. Returns ``None`` when NOTHING readable was found, which
    is not the same fact as a measured zero and must not be spelled the same
    way (#872's probe failure, one module over).
    """
    if cc is None:
        return None
    total = 0
    seen = False
    for name in ("write_queue", "load_queue", "ack_backup_queue"):
        q = getattr(cc, name, None)
        if q is None:
            continue
        try:
            if hasattr(q, "qsize"):
                total += int(q.qsize())
            else:
                total += len(q)
        except Exception:  # noqa: BLE001 - a probe never breaks a seam
            continue
        seen = True
    return total if seen else None


def parse_gate_heartbeat(report) -> Tuple[Optional[int], Optional[int]]:
    """``(checked, refused)`` out of ``gate_heartbeat``'s string, or ``(None, None)``.

    Parsed rather than substring-matched, because the streak arithmetic needs
    the VALUE -- a cutover that checked something is what proves the gate is
    reachable on this workload. An unreadable report yields ``None``, which the
    streak treats as "no evidence" rather than as a zero.
    """
    if report is None:
        return (None, None)
    m = re.search(r"checked=(\d+)\s+refused=(\d+)", str(report))
    if m is None:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


def stale_gate_zero_streak(
    prev_streak: int,
    *,
    checked: Optional[int],
    depth: Optional[int],
    ever_checked: bool,
) -> int:
    """The new consecutive-blind-cutover count for the #719 gate.

    #1205 -- WHY THE TRAFFIC TERM IS THE HEARTBEAT'S OWN HISTORY AND NOT THE
    QUEUE DEPTH. Reading the depth honestly is necessary but not sufficient:
    this probe runs inside ``_release_residents_for_cutover``, which the seam
    calls one statement after ``self._seam_drain_ms = self._quiesce_hicache(
    direction)``. New device-tier I/O is refused for the whole seam by
    ``hicache_seam_active`` and the old I/O has just been drained, so an honest
    depth reading there is ~always 0 -- and a depth-only gate would make this
    alarm permanently SILENT.

    That is the direction that costs a boot. A constant-true probe is noise (24
    fires on a healthy W37-D boot, #861e's complaint); a constant-false probe is
    a FALSE ALL-CLEAR, and a false all-clear is precisely how W37-C logged
    ``checked=0 refused=0`` on all eighteen flips with nobody woken. Both
    failures are wrong; only one of them lets the next W37-C through.

    So: a zero counts as evidence of blindness when EITHER the controller is
    demonstrably holding work right now, OR this process has already reported
    ``checked>0`` at some earlier cutover. The second term is what makes the
    alarm workload-aware without asking the workload: an instance that has
    reached the gate once has proved the gate reachable, so a run of zeroes
    after that is a disconnection and not an idle stretch. An instance that has
    never reached it may simply have nothing to check, which is #861e's
    legitimate case and stays silent.

    ``checked=None`` (unreadable heartbeat) neither counts nor clears -- an
    absent reading is not evidence in either direction.
    """
    if checked is None:
        return int(prev_streak)
    if checked > 0:
        return 0
    traffic = bool(depth) or bool(ever_checked)
    return int(prev_streak) + 1 if traffic else 0


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
        # #787 SENDER-SIDE HALF. Called synchronously by _abandon_no_quorum
        # and _abandon_unjoined_flip, strictly BEFORE local flip state is
        # cleared -- i.e. before this rank is free to resume launching new
        # admissions and race ahead of a downstream peer's own disarm-time
        # drain settle window (DRAIN_SETTLE_BUDGET_S in scheduler_pp_mixin.
        # py). It reaps/counts anything this rank has ALREADY posted on the
        # CHAN_DICT wire one final time, closing the gap between the most
        # recent ordinary service turn and the abandon decision. It does
        # NOT and cannot force an in-flight forward computation to finish
        # early -- that send lands whenever it actually completes, same as
        # before -- which is exactly why the receiver-side settle window
        # exists as the complementary half; neither half alone is sound.
        flush_pending_sends_fn: Optional[Callable[[], None]] = None,
        presence_deadline_s: float = DEFAULT_PRESENCE_DEADLINE_S,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        exchange: Optional[
            Callable[[Dict[int, torch.Tensor], Dict[int, int]], Dict[int, torch.Tensor]]
        ] = None,
        pp_pool_view: Optional[KvPoolView] = None,
        tp_pool_view: Optional[KvPoolView] = None,
        live_slots_fn: Optional[Callable[[], torch.Tensor]] = None,
        ready_fn: Optional[Callable[[], bool]] = None,
        # #1173: () -> (outstanding microbatch slots, this rank's forward
        # count). PP0's arm PRECONDITION, not a post-arm hold: see `arm`.
        launched_passes_fn: Optional[Callable[[], Tuple[Sequence[int], int]]] = None,
        # #1173 review: seconds of frozen ring (same slots, same forward count)
        # after which the deferral above escalates to a named group STOP.
        launched_pass_stall_s: float = 120.0,
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
        self._flush_pending_sends_fn = flush_pending_sends_fn
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
        # #856: served-request latency by rounds-since-cutover. The cutover
        # side is wired here; the request side is an explicit open integration
        # -- request latency is assembled in `tokenizer_manager`, a DIFFERENT
        # PROCESS, so feeding it is a cross-process change and not a line.
        # Recorded as open rather than guessed at: a wrongly-wired instrument
        # reports a number, which is worse than reporting none.
        self.warmup_ledger = WarmupLatencyLedger()
        # #825 tree-congruence state. Counters are attributes rather than a
        # latch: #823's whole lesson is that a divergence reported once can
        # say THAT the trees parted and never for how long, whether it got
        # worse, or whether it healed. Onsets and recoveries are the two
        # edges; `rounds` is the duration.
        self._tree_congruence = None
        self._tree_divergence_open = False
        self.tree_divergence_onsets = 0
        self.tree_divergence_rounds = 0
        self.tree_congruence_recoveries = 0
        self.tree_reconciles = 0
        self._tree_reconcile_suppressed_logged = False
        #: #760: True from the seam's no-return point until the cutover has
        #: installed the new phase. Read by the HiCache phase guard through
        #: the authority registration below: while True, device-tier HiCache
        #: I/O is refused outright -- pool bytes are in motion and the
        #: outgoing phase's backing is scheduled for release, so a copy
        #: enqueued now races the release whatever phase it names.
        self.hicache_seam_active = False
        #: #834 A: the direction whose ARM raised the guard above, or None.
        #: Non-None means the device tier is held down deliberately across the
        #: armed window rather than momentarily across the seam.
        self._prearm_hold_direction = None
        #: The drain that hold measured, in ms, and the consecutive count of
        #: arms refused because it was over budget.
        self._prearm_drain_ms = None
        self._prearm_drain_defers = 0
        #: #834 B: rows this rank grew locally that the group has NOT levelled
        #: to yet, and the round at which that debt was taken on. The rows are
        #: BACKED but must not be EXPOSED until a collective agrees -- see
        #: ``_deferred_grow_debt_check``.
        self._deferred_grow_rows = 0
        self._deferred_grow_round = None
        self._deferred_grow_level = None
        self._deferred_grow_pending = False
        # #760: hand the guard THIS object as its phase authority. The
        # routing global (parallel_state) toggles INSIDE the cutover, one
        # step among many, so it cannot express the seam; this runtime's
        # ``_phase`` is what the PHASE-FLIP DONE line reports, which the
        # 3s-lag crash correlation proves truthful. Registration is weak, so
        # a discarded runtime (hermetic tests build many) never gates I/O.
        # A failed registration is LOGGED, not swallowed silently -- an
        # inert guard that cannot be seen failing to arm is the #742 class.
        try:
            from sglang.srt.mem_cache.hicache_phase_guard import (
                register_flip_phase_authority,
            )

            register_flip_phase_authority(self)
        except Exception as e:  # noqa: BLE001 - never break runtime construction
            logger.error(
                "%s #760 flip phase authority registration FAILED (%s); the "
                "HiCache phase guard falls back to the routing global, which "
                "cannot see the seam. Device-tier I/O is NOT seam-protected "
                "in this process.",
                LOG_PREFIX,
                e,
            )
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
        #: #800: why THIS rank last withheld, or None if it announced. The
        #: abandonment log used to name exactly one of the two states that put
        #: a rank in the `missing` list -- "blocked upstream of the entry" --
        #: while the other one, "at the entry and declining to announce", is
        #: the one both 2026-08-22 wedges were in. A rank that holds the reason
        #: locally can say which it was instead of sending every reader
        #: upstream.
        self._last_presence_withhold_reason: Optional[str] = None
        #: #850: rounds withheld on a reason NO armed service turn can clear.
        #: Unlike `presence_withheld_rounds`, a non-zero count here is never
        #: healthy: it counts rounds spent waiting for a consumer this rank is
        #: itself excluding, which is the #800 shape one channel over.
        self.presence_futile_rounds = 0
        #: #850: distinct futile withholds DETECTED (one per epoch/round), which
        #: is a count of the defect and is raised even when the actuator below
        #: is switched off.
        self.presence_futile_detected = 0
        #: #850: flips actually abandoned early BY the shortened bound. Kept
        #: separate from `presence_futile_detected` on purpose: a detector that
        #: counts its own alarms as actions reports work it never did, and with
        #: SGLANG_PP_PRESENCE_FUTILE_S=0 this must stay 0 while detection goes
        #: on -- which is what makes the off-switch provable in both directions.
        self.presence_futile_abandons = 0
        #: #850: monotonic time this futile withhold began, or None.
        self._presence_futile_since: Optional[float] = None
        #: #850: the (epoch, round) `_presence_futile_since` belongs to, so a
        #: later arm never inherits an earlier one's age.
        self._presence_futile_key: Optional[Tuple[int, int]] = None
        #: #850: the shortened bound. Read once here so a test can override it
        #: on the instance without touching the process environment. Imported
        #: locally, the convention this module already uses for every other
        #: `envs` read; a failure to read it DISABLES the shortening rather
        #: than guessing a bound, which leaves the pre-#850 behaviour intact.
        try:
            from sglang.srt.environ import envs as _envs

            self._presence_futile_s = float(_envs.SGLANG_PP_PRESENCE_FUTILE_S.get())
        except Exception:  # noqa: BLE001 - a knob may never break construction
            self._presence_futile_s = 0.0
        #: #850: (epoch, round) already alarmed, so the DEFECT line is emitted
        #: once per occurrence rather than once per round.
        self._presence_futile_alarmed: Optional[Tuple[int, int]] = None
        self._join_deadline_s = DEFAULT_JOIN_DEADLINE_S
        self.join_deadline_aborts = 0
        self._exchange = exchange
        self._pp = pp_pool_view
        self._tp = tp_pool_view
        self._live_slots_fn = live_slots_fn
        self._ready_fn = ready_fn
        self._launched_passes_fn = launched_passes_fn
        #: #1173: DEFERRED arms, so "the arm never fired" and "the arm was
        #: deferred while the ring drained" can never read alike.
        self.arm_deferred_launched = 0
        #: #1173 review (blocker 4): the deferral STALL clock. Keyed on
        #: (outstanding slots, forward count) so that any forward progress in
        #: the ring restarts it -- only a frozen ring can reach the bound.
        self._launched_defer_key: Optional[Tuple[Tuple[int, ...], int]] = None
        self._launched_defer_since: Optional[float] = None
        self._launched_defer_streak = 0
        #: Seconds of NO ring progress after which a deferral escalates to the
        #: named group STOP. Generous on purpose: below it the deferral is the
        #: correct answer, and the false-positive direction is group-killing.
        #: 0 disables the escalation (deferral only), which is the pre-#1173
        #: hang and is therefore never the default.
        self._launched_pass_stall_s = float(launched_pass_stall_s)
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
        #: #969 §W3 follower cut. Rank 0 side: the decision it has taken and
        #: published, and whether the request stream has actually carried it
        #: yet (`take_flip_decision`). Rank k>0 side: the decision it was
        #: handed and must execute this round (`apply_flip_decision`). Two
        #: fields, one direction each -- there is no state here that both
        #: sides write, because that is what a second bookkeeping looks like.
        self._decision_out: Optional[PhaseFlipDecision] = None
        self._decision_taken: bool = False
        self._told: Optional[PhaseFlipDecision] = None
        #: #746: ``(req_rows, req_max)`` measured by ``arm()`` at the arm
        #: instant -- the exact extent this flip will pack -- or None when no
        #: flip is armed or the arm-time measurement failed. Cleared at EVERY
        #: exit (commit and all abandon paths): a snapshot that outlives its
        #: flip pins the rung permanently, the M5 failure mode #744's
        #: mutation matrix refuses. Read through ``parked_extent()``, never
        #: directly.
        self._parked_extent: Optional[Tuple[int, int]] = None
        #: #1202: every request this rank saw resident at ANY point of
        #: the armed window, keyed by ``id()``. The release reconciles
        #: its own enumeration against this ledger, because boot 9 proved
        #: the two are taken at different instants and disagree. Cleared
        #: at EVERY exit from the armed state, exactly like
        #: ``_parked_extent`` -- a ledger that outlives its flip would
        #: hand the NEXT cutover a stale object naming a reused row.
        self._armed_residents: Dict[int, object] = {}
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

    def _at_arm_census_due(self) -> bool:
        """#926: may the at-arm census run on THIS arm?

        THE INSTRUMENT BECAME THE DEFECT. `_pool_census("at-arm", ...)` walks
        the pool and, through `_census_ownership_audit`, the KV row-ownership
        map. #631 J installed it unconditionally, which was affordable when an
        arm was rare. It is not: the 0827 window measured 69 cutovers in five
        minutes, and one of four boots died in a CPU spin whose hot frame was
        this census. An arm is now a routine event and a full-pool walk per
        arm is a per-flip O(pool) tax on the scheduler thread.

        A CADENCE GATE, NOT A DELETION, and the difference is the whole point.
        The census is the only instrument that can see the #631 J page loss,
        so it must keep being ABLE to fire; what changes is how often. Two
        independent admissions, either of which opens the gate:

          * the first arm after `SGLANG_PP_ARM_CENSUS_MIN_INTERVAL_S` of wall
            clock (default 30s), so a slow-arming instance censuses every arm
            exactly as before;
          * every Nth arm (`SGLANG_PP_ARM_CENSUS_EVERY_N`, default 16), so a
            fast-arming instance still gets a bounded sample rather than none.

        Either env set to 0 disables that admission; setting BOTH to 0 restores
        the unconditional pre-#926 behaviour, which is the escape hatch an
        operator needs when chasing exactly the page loss this watches for.

        NEVER SILENTLY FALSE. A skipped census is counted and the count rides
        the next census that does run, so a reader can never mistake "sampled"
        for "clean" -- the #829/INDIKATOR-GESETZ rule that an instrument must
        be able to say what it did not measure.
        """
        from sglang.srt.environ import envs

        n = int(getattr(self, "_at_arm_census_arms", 0)) + 1
        self._at_arm_census_arms = n

        every_n = envs.SGLANG_PP_ARM_CENSUS_EVERY_N.get()
        min_interval = envs.SGLANG_PP_ARM_CENSUS_MIN_INTERVAL_S.get()

        if not every_n and not min_interval:
            return True  # both admissions disabled: pre-#926 behaviour

        now = time.time()
        last = getattr(self, "_at_arm_census_last_s", None)

        due = False
        if every_n and n % int(every_n) == 0:
            due = True
        if min_interval and (last is None or now - last >= float(min_interval)):
            due = True

        if not due:
            self._at_arm_census_skipped = (
                int(getattr(self, "_at_arm_census_skipped", 0)) + 1
            )
            return False

        skipped = int(getattr(self, "_at_arm_census_skipped", 0))
        if skipped:
            logger.info(
                "%s #926 at-arm census sampling: %d arm(s) since the last one "
                "were NOT censused (arm #%d, every_n=%s, min_interval_s=%s). "
                "The window this census covers is therefore a SAMPLE, not a "
                "continuous record.",
                LOG_PREFIX,
                skipped,
                n,
                every_n,
                min_interval,
            )
            self._at_arm_census_skipped = 0
        self._at_arm_census_last_s = now
        return True

    def arm(self, direction: str, source: str) -> Tuple[bool, str]:
        """Arm a flip. Replicated call; PP0's decision round commits it.
        Returns (ok, msg).

        #969 §W3: EVERY REFUSAL IN THIS FUNCTION IS RANK 0's ALONE. The arm
        originates at the request origin (request_receiver.py: the policy hook
        runs only there) and travels to the ranks below on the request stream;
        below rank 0 this call is an ORDER, not a proposal. A follower that
        consulted its own guards, its own storm limiter or its own pre-arm
        drain verdict could refuse an arm rank 0 accepted, and then rank 0
        would decide a flip for a rank that is not armed -- the exact
        disagreement `raenge-nie-uneins-crash-stop` forbids, manufactured by
        the defensive check that was meant to prevent it. So the follower
        takes the branch below and runs the side-effecting part only.
        """
        if self._rank != 0:
            return self._arm_as_follower(direction, source)
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
        # #1173: QUIESCENCE IS A PRECONDITION OF THE ARM, NOT A HOLD AFTER IT.
        #
        # MEASURED (boot_855_weg1b4_f58a71bde0_0903_080200.log). PP0 posted
        # forwarded pass fwd_ct=81 into slot 1 at 08:06:50 (`#631 PROXY-SEND
        # t6`, log 82144) and armed pp_to_tp on the NEXT line (82145). It then
        # detected its own non-quiescence and only LOGGED it -- "armed
        # (pp_to_tp) but NOT QUIESCENT: PP microbatches still in flight (mb
        # slots [0, 1])" (82182) -- and waited for a drain that only the
        # followers could produce, while the arm is exactly what stops them
        # producing it. PP0 then blocked 60 s in `recv_object[src=2]` and the
        # group died (#980 at 90592).
        #
        # A POST-ARM HOLD CANNOT CLOSE THIS. By the time the hold fires the
        # order has already travelled to the followers (`_arm_as_follower`),
        # so the state it is holding for is one the arm itself created. The
        # only place the launch can still be waited out for free is BEFORE
        # `_pending` is set: nothing is armed, no rank has entered the seam,
        # and the ranks cannot disagree about whether this attempt happened --
        # the #485 argument for declining here, applied unchanged.
        #
        # RANK 0 ONLY, by construction: followers returned at the top of this
        # method (#969 W3, an arm below rank 0 is an ORDER, not a proposal).
        # So this adds no rank-local verdict and no new synchronisation point.
        #
        # DEFERRED, NOT REFUSED FOR GOOD. The policy re-evaluates every round;
        # a launched pass returns in one or two passes, and the next arm goes
        # through. The counter beside the line is what makes "deferred" and
        # "never asked" distinguishable in a log.
        if self._launched_passes_fn is not None:
            try:
                _slots, _fwd_ct = self._launched_passes_fn()
                _outstanding = [int(i) for i in (_slots or ())]
            except Exception:  # noqa: BLE001 - an unreadable probe never arms
                _outstanding, _fwd_ct = [], -1
            if _outstanding:
                self.arm_deferred_launched += 1
                # #1173 review (blocker 4 / N2): AN UNBOUNDED DEFERRAL IS A
                # SILENT HANG, AND THE #1153 CONTRACT FORBIDS IT.
                #
                # The commit body's "a launched pass returns in one or two
                # passes" is an assumption, not a guarantee. If the ring truly
                # never returns the pass, this branch converts what used to be
                # a crash into a hang in PP that only this counter
                # distinguishes -- exactly the "no rank ever ends a
                # PP0-launched pass silently" clause. So the deferral is
                # bounded, and the bound escalates to the same named STOP the
                # follower raises.
                #
                # THE BOUND IS FORWARD PROGRESS, NOT DEFERRAL COUNT. This
                # method is driven from the receive poll (one call per poll, at
                # whatever rate traffic arrives), so N deferrals is not a
                # duration and a count-only bound would fire at a rate set by
                # the client. The state that must not persist is (outstanding
                # slots, forward count): while either moves the ring IS making
                # progress and the budget restarts. Only a ring frozen on the
                # SAME slots at the SAME fwd_ct for longer than
                # `launched_pass_stall_s` is a launched pass nobody will
                # execute.
                _key = (tuple(_outstanding), int(_fwd_ct))
                _now = time.monotonic()
                if _key != self._launched_defer_key:
                    self._launched_defer_key = _key
                    self._launched_defer_since = _now
                    self._launched_defer_streak = 0
                self._launched_defer_streak += 1
                _stalled_s = _now - (self._launched_defer_since or _now)
                if (
                    self._launched_pass_stall_s > 0
                    and _stalled_s > self._launched_pass_stall_s
                ):
                    self._launched_defer_key = None
                    self._launched_defer_since = None
                    raise RuntimeError(
                        "#1173 LAUNCHED PASS UNEXECUTED UNDER ARM STOP "
                        "rank=0 slots=%s fwd_ct=%d direction=%s "
                        "stalled_s=%.1f bound_s=%.1f deferrals=%d reason=%s"
                        % (
                            _outstanding,
                            int(_fwd_ct),
                            direction,
                            _stalled_s,
                            self._launched_pass_stall_s,
                            int(self.arm_deferred_launched),
                            (
                                "PP0 deferred the arm on the SAME outstanding "
                                "slots at the SAME forward count for longer "
                                "than the bound: the ring is not returning the "
                                "pass and no further deferral can change that. "
                                "Stopping through the group (#1153) instead of "
                                "hanging in PP for ever"
                            ),
                        )
                    )
                msg = (
                    f"#1173 ARM DEFERRED: launched passes outstanding "
                    f"slots={_outstanding} fwd_ct={_fwd_ct} "
                    f"direction={direction} deferrals={self.arm_deferred_launched} "
                    f"streak={self._launched_defer_streak} "
                    f"stalled_s={_stalled_s:.1f} bound_s={self._launched_pass_stall_s:.1f}. "
                    f"PP0 launched forwarded work the ring has not returned; "
                    f"arming now would stop the followers executing it and "
                    f"then wait for a drain only they can produce (weg1b4). "
                    f"The policy re-evaluates every round -- this is a "
                    f"deferral, not a refusal."
                )
                # #1173 review (N1/N10): RATE-LIMITED, the way the sibling
                # #1020 void guard is. This method is called from the phase
                # policy hook on the request-origin rank's RECEIVE POLL, so a
                # persistent deferral emits per poll -- against the #776
                # precedent (449 MB in 20 min from one armed emitter) that
                # would drown the very boot log this fix is judged on. The
                # suppressed occurrences are not lost: `deferrals=` and
                # `streak=` carry their own denominators, so a gap in the log
                # is readable rather than a zero (the DENOMINATOR LAW).
                if (
                    self.arm_deferred_launched <= 3
                    or self.arm_deferred_launched % 512 == 0
                ):
                    logger.warning("%s %s", LOG_PREFIX, msg)
                return False, msg
            # The ring is quiescent (or unreadable): no stall is in progress.
            self._launched_defer_key = None
            self._launched_defer_since = None
            self._launched_defer_streak = 0
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
        # #834 A: QUIESCE THE DEVICE TIER HERE, WHERE THE PIPELINE STILL RUNS.
        #
        # Deliberately the LAST thing before ``_pending`` is set, and after the
        # floor relief: everything above this line can still refuse cheaply,
        # and refusing after having disarmed the device tier means disarming it
        # for nothing. Below this line the arm is going through, so the hold is
        # the beginning of the flip rather than a speculative pause.
        #
        # UNLIKE ``_prearm_floor_relief`` DIRECTLY ABOVE, THIS VERDICT IS NOT
        # DISCARDED, and the asymmetry is the point. The floor relief's verdict
        # is thrown away because it is a RANK-LOCAL AFFORDABILITY judgement and
        # a rank-local refusal splits the arm -- one rank armed, its peers not,
        # the armed one parked at the entry for ever. This verdict is different
        # in kind: it refuses BEFORE anything is pending, so a rank that refuses
        # here is in exactly the state it was in before the call, which is the
        # same state a rank whose guards refused at the top of this method is
        # in. Nothing is armed, no collective is reached, and the ranks cannot
        # disagree about whether this attempt happened -- the #485 argument for
        # declining here rather than in ``_execute``, applied unchanged.
        prearm_ok, prearm_drain_ms, prearm_detail = self._prearm_quiesce(direction)
        if not prearm_ok:
            logger.warning("%s [#834] %s", LOG_PREFIX, prearm_detail)
            return False, prearm_detail
        if seam_shrink_prearm_quiesce_enabled():
            self._prearm_drain_defers = 0
            logger.warning(
                "%s [#834] PREARM DRAIN %s: device-tier streams quiesced in "
                "%.1f ms BEFORE arming, with the pipeline still serving. The "
                "seam's own quiesce still runs at the no-return point and is "
                "now a confirmation rather than a wait. %s",
                LOG_PREFIX,
                direction,
                prearm_drain_ms,
                prearm_detail,
            )
        self._enter_armed_state(direction)
        msg = (
            f"phase flip armed: {direction} (source {source}); PP0 decides "
            f"when it commits and tells every rank below, or abandons it "
            f"after {self._park_deadline_s:g}s parked -- PP0 is the only "
            f"timeout carrier (#969 §W3)"
        )
        logger.warning("%s %s", LOG_PREFIX, msg)
        return True, msg

    def _arm_as_follower(self, direction: str, source: str) -> Tuple[bool, str]:
        """#969 §W3: below PP0 an arm is an ORDER. Execute it, judge nothing.

        The side-effecting half of ``arm`` runs here -- the pre-arm device-tier
        quiesce, the state entry, the at-arm census, the exposure enforcement.
        What does NOT run is every verdict: guards, storm/backoff limiters, the
        legal-transition check and the pre-arm drain's own ok/not-ok. Each of
        those is a rank-local opinion, and a rank-local opinion that can refuse
        an arm PP0 accepted is precisely how the ranks end up uneins.

        The one thing that IS checked is identity: a direction this build does
        not know cannot be executed, and it means the stream carried something
        this rank cannot be following. That stops the group.
        """
        if direction not in _DIR_ID:
            raise KvReshardError(
                f"{LOG_PREFIX} #969 rank {self._rank} was told to arm an "
                f"unknown flip direction {direction!r} (source {source}). A "
                f"follower cannot execute what it cannot name, and it may "
                f"not invent a local substitute -- the group stops here."
            )
        # Side effect only: the verdict is discarded on purpose (see above).
        try:
            self._prearm_quiesce(direction)
        except Exception as exc:  # noqa: BLE001 - never refuse PP0's order
            logger.warning(
                "%s [#834] pre-arm quiesce raised on follower rank %d: %s. "
                "The arm proceeds: the seam's own quiesce still runs at the "
                "no-return point, and refusing here would split the arm.",
                LOG_PREFIX,
                self._rank,
                exc,
            )
        self._enter_armed_state(direction)
        msg = f"phase flip armed (told by PP0): {direction} (source {source})"
        logger.warning("%s %s", LOG_PREFIX, msg)
        return True, msg

    def _enter_armed_state(self, direction: str) -> None:
        """The state entry every arm performs, decider and follower alike."""
        if self._pending is not None and self._pending != direction:
            logger.warning(
                "%s re-arming %s -> %s", LOG_PREFIX, self._pending, direction
            )
        self._pending = direction
        # A fresh arm starts a fresh round sequence. The epoch already
        # distinguishes this arm from any earlier one, so the round simply
        # restarts at 0 rather than having to be globally unique.
        self._entry_round = 0
        # #969 §W3: a fresh arm can carry no stale told/published decision.
        self._decision_out = None
        self._decision_taken = False
        self._told = None
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
        # #1202: AND SNAPSHOT WHO IS RESIDENT, not just how wide they are.
        # `_parked_extent` records the SIZE of the resident set at this
        # instant; nothing recorded its MEMBERS, so the release one second
        # later enumerated afresh and retracted a different set (boot 9,
        # quoted with that boot's pre-#1205 label -- now `live_reqs`:
        # cur_slot_reqs=1 on all three ranks at arm, 1/0/0 retracted at
        # the release). A fresh arm starts a fresh ledger.
        self._armed_residents = {}
        note_armed_residents(
            self._armed_residents, getattr(self, "_census_scheduler", None)
        )
        # #631 J: census AT ARM. The pre/post-cutover pair proved the move
        # and the cutover innocent (identical unaccounted set on both
        # sides), and a no-flip control boot stayed clean, so the page goes
        # missing somewhere in the ARMED window. This bracket closes it.
        if self._at_arm_census_due():
            self._pool_census("at-arm", direction)
        # #853(i): ENFORCE THE EXPOSURE LAW HERE TOO, BECAUSE THE CUTOVER MAY
        # NEVER COME. W24's stuck phase ran 23.6 minutes with 153 arms and ZERO
        # cutovers, and enforcement was wired to the cutover alone -- so it went
        # quiet for precisely the window it was installed to police. A gate
        # unreachable in the failure mode is not a gate.
        #
        # LAWFUL HERE, established against the tree rather than assumed:
        # `kv_backing_relief` reaches NO COLLECTIVE (the ceiling reads a cached
        # already-agreed floor, never a live reduction), so ranks may run this
        # at different instants without divergent participation; `KvRowCap` is
        # non-destructive by construction -- "only unallocated ids are held
        # back" -- so it cannot disturb the requests still live at arm; and
        # `exposed > backed` is never a sanctioned mid-grow state (a grow adds
        # BACKED rows that stay UNEXPOSED until a collective raises the level),
        # so this can only ever be correcting the #816/#833 defect, never
        # racing a legitimate grow. The same actuator already runs from four
        # other non-quiescent sites.
        #
        # Its verdict is DISCARDED for the reason `_prearm_floor_relief` above
        # states: a rank-local judgement must never decide an arm, or one rank
        # arms while its peers do not and parks at the entry for ever. This
        # corrects the id space and reports; it does not vote.
        self._enforce_exposure_at_seam(f"{direction} arm")

    # -- the per-round hook ---------------------------------------------------
    def on_round(self, require_armed_and_parked: bool = False) -> Optional[dict]:
        """One scheduler round; returns move stats when a flip executed this
        round, else ``None``.

        #969 §W3 -- THE FOLLOWER CUT. RANK 0 DECIDES; EVERY OTHER RANK
        EXECUTES AND FORMS NO OPINION.

        User law, verbatim (2026-08-29): "raenge duerfen sich niemals uneins
        sein. wenn uneins crash stop." and "nein, alles folgt rang0, so wie
        bei tp3 ja auch." The design form is FOLLOWER SEMANTICS, not agreement
        machinery -- exactly what TP3 already does, where rank 0 samples and
        broadcasts and no consensus protocol exists because nobody else ever
        decides. Disagreement is excluded by CONSTRUCTION here for the same
        reason; the checksum below is only a backstop.

        WHAT THIS REPLACED, and why it is a deletion rather than a repair.
        Until #969 §W3 this hook packed a per-rank proposal
        ``[armed, ready, expired, epoch, dir, fp, *vec, tree_digest]`` into a
        blocking element-wise MIN reduction and reconciled the ranks'
        opinions from ``lo``/``hi``. Every term of that payload except the
        identity fields was a VOTE, and the reduce was the machinery that
        settled the votes -- the shape the standing user order names for
        deletion (`upstream-minimal-statt-eigenbau`: a second bookkeeping
        beside an upstream truth is deleted, not repaired). Upstream has no
        such reduction: PP0 admits, the batch travels, and every rank simply
        does what arrives. Deleted with it: ``_park_expired`` (the per-rank
        park clock -- rank 0 is now the ONLY timeout carrier), the presence
        announce/quorum spin that had to exist because the reduce was a
        blocking collective entered at a rank-local cadence, and both flip
        drains, whose whole job was to clean up after ranks that had disarmed
        on their own clocks.

        HOW THE DECISION TRAVELS: on the request stream, which already runs
        in both loop families, so no arc and no collective is added (a new
        collective on this path is RECORDED FATAL, and that stands). See
        ``PhaseFlipDecision`` in io_struct for both carriers.

        WHY A FOLLOWER MAY CUT OVER ON RANK 0's READINESS ALONE, which is the
        one property this whole design rests on: rank 0 holds ``mbs[slot]``
        for a microbatch until that microbatch's OUTPUT has come back around
        the ring, so "rank 0 has no in-flight microbatch" is a statement
        about the WHOLE RING, not about rank 0. Combined with rank 0 having
        stopped admitting at arm time, a decided flip is a flip whose
        pipeline is provably empty on every rank. That is the same fact
        upstream relies on and the reason it needs no readiness exchange.

        The bounded wait a follower does is the request-stream receive it was
        doing anyway. If it expires the receive raises and the group dies --
        which is the required behaviour (`raenge-nie-uneins-crash-stop`), not
        a fallback into a local decision."""
        if not require_armed_and_parked:
            self._round += 1
        # #834 B: PAY THE DEFERRED GROW HERE. Rank-local work only -- no
        # collective is reached on this path, which is what makes a local
        # cadence legal for it and illegal for the levelling it was split
        # from. Unchanged by the follower cut.
        pay_deferred_grow(self)
        # #1202: KEEP THE ARMED-WINDOW RESIDENT LEDGER CURRENT.
        # Rank-local bookkeeping, no collective, no verdict -- it only
        # widens the set the release will reconcile against, and a set
        # that is already whole is unchanged by it. Runs only while a
        # flip is armed, so it is bounded by the park deadline.
        if self._pending is not None:
            note_armed_residents(
                self._armed_residents, getattr(self, "_census_scheduler", None)
            )
        if self._rank == 0:
            return self._round_as_decider()
        return self._round_as_follower()

    def _local_tree_digest(self) -> int:
        """#825: this rank's prefix-tree digest, or 0 when there is no tree."""
        return tree_congruence.tree_digest_of(
            getattr(getattr(self, "_census_scheduler", None), "tree_cache", None)
        )

    def _note_tree_congruence(self, local_digest: int, peer_digest: int) -> None:
        """#825 divergence detector -- KEPT, and deliberately NOT fatal.

        §W3 keeps the detectors and turns their action arms group-fatal
        "where they are refuse-and-continue today". This one is neither: it
        has only ever counted (its ACTION, the tree reset, was withdrawn on
        metal in 2026-08-23 and is off by default), and a divergent tree is
        not the ranks disagreeing about a DECISION. It is the designed,
        measured consequence of the PP phase running with the uniformity
        floors switched off (scheduler.py:4770) -- #825's own note says
        putting it in the fatal family "would take the instance down at the
        first cutover of every boot". So it stays a counter, and the
        deviation from the letter of §W3 is stated here rather than
        performed silently.

        The fatal family is ``_verify_told_identity``: epoch, direction,
        config fingerprint and vector. Those disagreeing IS the ranks being
        out of step about the flip itself, and that stops the group.
        """
        self._tree_congruence = tree_congruence.congruence_verdict(
            local_digest=local_digest,
            group_min=min(local_digest, peer_digest),
            group_neg_min=max(local_digest, peer_digest),
        )
        if not self._tree_congruence.congruent:
            self.tree_divergence_rounds += 1
            if not self._tree_divergence_open:
                self._tree_divergence_open = True
                self.tree_divergence_onsets += 1
                logger.warning(
                    "%s #825 TREE DIVERGENCE ONSET at round %d: %s",
                    LOG_PREFIX,
                    self._round,
                    self._tree_congruence.reason,
                )
        elif self._tree_divergence_open:
            # Recovery edge. #823's lesson: a divergence that is only ever
            # reported at onset can say THAT the trees parted and never that
            # they healed.
            self._tree_divergence_open = False
            self.tree_congruence_recoveries += 1
            logger.warning(
                "%s #825 TREE DIVERGENCE RECOVERED at round %d after %d "
                "divergent rounds (onsets=%d)",
                LOG_PREFIX,
                self._round,
                self.tree_divergence_rounds,
                self.tree_divergence_onsets,
            )

    def _round_as_decider(self) -> Optional[dict]:
        """Rank 0. The ONLY rank that decides, and the only timeout carrier.

        Two passes, deliberately: the verdict is published on the round it is
        taken and EXECUTED on the round after the request stream has actually
        carried it. Executing in the same round would cut over before the
        followers had been told, and rank 0 leaves the PP loop on a committed
        flip (PhaseFlipLoopExit), so a decision not yet on the wire when it
        leaves would never be sent at all. The PP loop commits its chain
        forward (`_pp_commit_pending_req_work`) immediately BEFORE this hook,
        so by the round that executes, the message is gone from this rank;
        in the TP phase the carrier is a broadcast and is complete on return.
        """
        if self._pending is None:
            self._decision_out = None
            self._decision_taken = False
            return None

        if self._decision_out is not None:
            if not self._decision_taken:
                # Published, not yet carried. Nothing to do but let the next
                # request-stream turn pick it up.
                return None
            dec = self._decision_out
            self._decision_out = None
            self._decision_taken = False
            self.desync_checks += 1
            self._entry_round += 1
            # Trivially congruent with itself: rank 0 is the reference the
            # followers compare against, so the pair it reports is its own.
            self._note_tree_congruence(dec.tree_digest, dec.tree_digest)
            if dec.verdict == PhaseFlipDecision.ABORT:
                return self._abandon_parked_flip(0)
            self._last_hold_reason = None
            try:
                return self._execute()
            finally:
                # #760/#834 A insurance, unchanged: a raise anywhere in the
                # seam must not leave the guard reporting a seam for ever,
                # except while a pre-arm hold is deliberately down.
                if not prearm_quiesce_held(self):
                    self.hicache_seam_active = False

        if self._ready_fn():
            verdict = PhaseFlipDecision.PROCEED
        elif (
            self._park_deadline_s > 0
            and self._armed_at is not None
            and (self._clock() - self._armed_at) >= self._park_deadline_s
        ):
            # THE ONE TIMEOUT IN THE SYSTEM. It lives here and nowhere else:
            # a rank-local clock that can put one rank in a different state
            # from its peers is exactly what the flip campaign died of
            # twenty times, and _park_expired was that clock on every rank.
            verdict = PhaseFlipDecision.ABORT
        else:
            self._log_not_ready()
            self._hold(
                f"armed ({self._pending}), waiting for the ring to drain on "
                f"this rank -- rank 0's drained state IS the ring's"
            )
            return None

        self._decision_out = PhaseFlipDecision(
            verdict=verdict,
            epoch=self._epoch,
            dir_id=_DIR_ID[self._pending],
            config_fp=self._fp,
            vector=self._vec,
            tree_digest=self._local_tree_digest(),
        )
        self._decision_taken = False
        logger.warning(
            "%s #969 PP0 DECIDES %s for %s at epoch %d: it rides the request "
            "stream to every rank below, which executes it and decides "
            "nothing. No consensus round, no votes, no reduction.",
            LOG_PREFIX,
            verdict,
            self._pending,
            self._epoch,
        )
        return None

    def _round_as_follower(self) -> Optional[dict]:
        """Every rank below PP0. Executes what it is told; decides nothing."""
        dec = self._told
        if dec is None:
            return None
        self._told = None
        if self._pending is None:
            raise KvReshardError(
                f"{LOG_PREFIX} #969 TOLD A FLIP THIS RANK NEVER ARMED: rank "
                f"{self._rank} was handed PP0's {dec.verdict!r} decision for "
                f"epoch {dec.epoch} while nothing is armed here. The arm and "
                f"the decision ride the SAME request stream, in that order, "
                f"so this can only mean the ranks are out of step -- and an "
                f"out-of-step group STOPS. It does not repair itself locally."
            )
        self._verify_told_identity(dec)
        self.desync_checks += 1
        self._entry_round += 1
        self._note_tree_congruence(self._local_tree_digest(), dec.tree_digest)
        if dec.verdict == PhaseFlipDecision.ABORT:
            # ready is reported, not consulted: this rank is abandoning
            # because PP0 said so, never because of its own reading.
            return self._abandon_parked_flip(1 if self._ready_fn() else 0)
        self._last_hold_reason = None
        try:
            return self._execute()
        finally:
            if not prearm_quiesce_held(self):
                self.hicache_seam_active = False

    def _verify_told_identity(self, dec) -> None:
        """The backstop: detected divergence STOPS THE GROUP.

        Construction already excludes disagreement (only rank 0 decides), so
        reaching a mismatch here means a premise of the design is false. The
        only legal response is to die loudly before any rank moves a byte
        under the wrong layout -- never refuse-and-continue, never a local
        repair (`raenge-nie-uneins-crash-stop`, detection half).
        """
        local_dir = _DIR_ID[self._pending] if self._pending is not None else 0
        mismatches = []
        if int(dec.epoch) != int(self._epoch):
            mismatches.append(f"epoch: pp0={dec.epoch} here={self._epoch}")
        if int(dec.dir_id) != int(local_dir):
            mismatches.append(f"direction: pp0={dec.dir_id} here={local_dir}")
        if int(dec.config_fp) != int(self._fp):
            mismatches.append(f"config_fp: pp0={dec.config_fp} here={self._fp}")
        if tuple(dec.vector) != tuple(self._vec):
            mismatches.append(f"vector: pp0={tuple(dec.vector)} here={self._vec}")
        if not mismatches:
            return
        raise KvReshardError(
            f"{LOG_PREFIX} DESYNC at round {self._round}: rank {self._rank} "
            f"disagrees with PP0 about the flip it was told to execute "
            f"({'; '.join(mismatches)}; this rank: pending={self._pending} "
            f"epoch={self._epoch} phase={self._phase}). Under follower "
            f"semantics this is unreachable by construction, so reaching it "
            f"means a premise is false -- the group stops HERE, before any "
            f"rank moves a byte under the wrong layout."
        )

    def take_flip_decision(self):
        """Rank 0 only: hand the pending decision to the request stream.

        Called by the request origin once per intake turn. Returns the
        decision exactly once; after that the decider knows the followers
        have been told and may execute on its next round.
        """
        if self._rank != 0:
            return None
        dec = self._decision_out
        if dec is None or self._decision_taken:
            return None
        self._decision_taken = True
        return dec

    def apply_flip_decision(self, dec) -> None:
        """Every rank below PP0: record what PP0 decided, to execute this
        round. No verdict is formed here and none is checked -- the checks
        that exist are identity checks, and they kill the group rather than
        adjust anything (see ``_verify_told_identity``)."""
        if self._rank == 0:
            return
        self._told = dec

    def _reconcile_trees_if_diverged(self, direction: str) -> None:
        """#825: repair the prefix trees before the TP phase demands identity.

        DECIDED FROM THE GROUP VERDICT, NEVER LOCALLY. `self._tree_congruence`
        was computed from `lo`/`hi` of the consensus reduction, which is the
        literal output of a collective, so every rank reads the same verdict
        and takes the same branch. A reconcile that some ranks skipped would
        be a new divergence, i.e. the defect this repairs.

        ONLY ON pp_to_tp, and that asymmetry is the point. The TP-decode phase
        requires rank-identical trees: admission is rank-local
        (`scheduler.py:7073`), `#cached-token` is literally
        `len(req.prefix_indices)` per rank, and the per-layer TP collectives
        are entered with whatever token count that produced -- the measured
        0516 divergence (`#cached-token 0 vs 16384`). The PP-prefill phase
        does NOT require it (#791: "Each PP rank is an independent scheduler
        that re-derives its own admission verdict from its own local
        radix-cache state"), so reconciling into PP would pay the capacity
        cost for nothing.

        THE ACTION IS A RESET, and it is chosen for being uniform BY
        CONSTRUCTION rather than for being clever. Intersecting the trees
        would keep more cache, but it needs the ranks to exchange key sets --
        a second, much larger collective at the seam, i.e. exactly the thing
        this design avoids. An empty tree is a state all ranks reach without
        talking. Correctness over capacity is the trade `#824` already makes
        in this family, and the cost is counted (`tree_reconciles`) so it can
        be argued with evidence rather than assumed small.

        NEVER RAISES. This sits between the pre-cutover movers and the
        cutover, with requests parked; a raise here takes the instance down
        for a cache-capacity repair. The same reasoning `_quiesce_hicache`
        below documents.
        """
        verdict = self._tree_congruence
        if verdict is None or verdict.must_reconcile is False:
            return
        if direction != PP_TO_TP:
            return
        # #825 FALSIFIED ON METAL, 2026-08-23 08:55:45, boot_826_review_0912.
        #
        # The reset is OFF by default because it took the instance down on its
        # first real cutover, on all three ranks at once:
        #
        #   cache_finished_req -> dec_lock_ref
        #   -> full_component.py:239  `if cur.id in skip_lock_node_ids`
        #   AttributeError: 'NoneType' object has no attribute 'id'
        #
        # My premise was "requests are parked between the movers and the
        # cutover, so dropping the tree is safe". PARKED IS NOT UNREFERENCED.
        # The cutover carries RESIDENT requests across, and each holds a
        # `last_node` with a lock ref. `reset()` rebuilds the root, orphaning
        # those nodes, so the parent walk in `dec_lock_ref` no longer
        # terminates at the live root and runs off the top into None.
        #
        # The DETECTION half of #825 is unaffected and stays on: the digest
        # rides the consensus reduce, the verdict is group-decided, and the
        # onset/recovery counters are what proved the divergence is real (3
        # onsets inside 35 s of load on this very boot). What is withdrawn is
        # the ACTION, because a reconcile that must not run while any node is
        # locked needs to be built against the lock refs -- evicting only the
        # unlocked portion, or reconciling at a point with no resident reqs --
        # and that is a design, not a flag flip.
        if os.environ.get("SGLANG_TREE_RECONCILE", "").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            if not self._tree_reconcile_suppressed_logged:
                self._tree_reconcile_suppressed_logged = True
                logger.warning(
                    "%s #825 tree divergence detected but the reconcile is "
                    "OFF (SGLANG_TREE_RECONCILE unset): resetting the tree "
                    "under resident lock refs crashed dec_lock_ref on "
                    "2026-08-23. Detection only. %s",
                    LOG_PREFIX,
                    verdict.reason,
                )
            return
        scheduler = getattr(self, "_census_scheduler", None)
        tree = getattr(scheduler, "tree_cache", None)
        reset = getattr(tree, "reset", None)
        if reset is None:
            logger.warning(
                "%s #825 tree reconcile requested but this scheduler exposes "
                "no resettable tree_cache; the TP phase is entering with "
                "divergent prefix trees. %s",
                LOG_PREFIX,
                verdict.reason,
            )
            return
        try:
            reset()
        except Exception as e:  # noqa: BLE001 - the seam must not die here
            logger.error("%s #825 tree reconcile failed: %s", LOG_PREFIX, e)
            return
        self.tree_reconciles += 1
        logger.warning(
            "%s #825 TREE RECONCILE at cutover %s: prefix cache dropped on "
            "every rank so the TP phase starts from a rank-identical tree "
            "(reconcile #%d). %s",
            LOG_PREFIX,
            direction,
            self.tree_reconciles,
            verdict.reason,
        )

    def _quiesce_hicache(self, direction: str) -> float:
        """#760: drain the cache controller's device-tier streams at the seam.

        Reaches the controller through the census scheduler handle (absent in
        unit stubs, where this is a no-op like the census itself). A failure
        is logged, never raised: with requests parked, a raise here takes the
        instance down, and the seam guard above still refuses NEW I/O -- only
        already-in-flight copies remain exposed, which is the pre-fix state,
        not a new hazard.

        #830 F1: RETURNS THE DRAIN IN MILLISECONDS, and the caller records it.
        ``quiesce_device_io``'s own docstring already promised this -- "Returns
        the wait in seconds; the caller logs it into the seam record so a slow
        drain is attributable instead of vanishing into the flip's residual
        (#690)" -- but this method DISCARDED the return value, so the drain
        vanished into exactly the residual that sentence names.

        That mattered, because the drain was filed (ANALYSE_830 F1) as the
        leading cause of the movers inflation (mean movers 2715-2898 ms with
        HiCache off vs 8113-18083 ms with it on) purely on the strength of its
        own docstring saying the copies "outlive their Python call by seconds".
        The one boot where the drain's line survives says otherwise:
        ``boot_window2_0823_1554.log``, current tree, 18 emissions, mean AND
        max 0.0 ms -- against movers of 5915.9 ms in that same boot. So in this
        tree the drain is not the movers cost. It is left in place (removing it
        re-opens the two #760 SIGSEGVs, which is a correctness trade nobody
        should take for latency), and it is now MEASURED at the seam instead of
        being argued about from a docstring.

        The caveat is stated rather than buried: the 08-19 boots carrying the
        big movers numbers predate this drain's log line entirely, so this is a
        measurement on the current tree, not a retro-acquittal of the 08-19
        ones. What it does establish is that today the movers term needs a
        different explanation, and F4's budget is what will catch the drain if
        it ever does become the cost.
        """
        scheduler = getattr(self, "_census_scheduler", None)
        controller = getattr(
            getattr(scheduler, "tree_cache", None), "cache_controller", None
        )
        quiesce = getattr(controller, "quiesce_device_io", None)
        if quiesce is None:
            return 0.0
        try:
            elapsed_s = quiesce(f"phase flip {direction}")
            drain_ms = float(elapsed_s or 0.0) * 1000.0
            # #917: SAY WHICH OF THE TWO THINGS HAPPENED. This line is the one a
            # three-rank log is read by -- it carries the direction and the
            # prefix, which the controller's own line does not -- and it used to
            # report "quiesced" whether or not a stream had drained. In the 0826
            # rerun it printed "quiesced in 0.2 ms" on PP1 and PP2 immediately
            # after both of their streams raised an illegal-access, so a grep of
            # this line found three clean drains in a boot that had one. The
            # controller now publishes which streams failed; a diagnostic that
            # cannot be read as a false all-clear is worth the extra getattr.
            failed = tuple(getattr(controller, "last_quiesce_failed", ()) or ())
            if failed:
                logger.error(
                    "%s [#760] SEAM DRAIN %s: device-tier streams did NOT "
                    "quiesce after %.1f ms -- %s still holds in-flight copies "
                    "at the no-return point. The seam proceeds without them.",
                    LOG_PREFIX,
                    direction,
                    drain_ms,
                    " and ".join(failed),
                )
                return drain_ms
            # Carries LOG_PREFIX deliberately: the controller's own line has no
            # prefix and no direction, so it cannot be correlated with the flip
            # it belongs to in a three-rank log.
            logger.warning(
                "%s [#760] SEAM DRAIN %s: device-tier streams quiesced in "
                "%.1f ms at the no-return point.",
                LOG_PREFIX,
                direction,
                drain_ms,
            )
            return drain_ms
        except Exception as e:  # noqa: BLE001 - the seam must not die here
            logger.error(
                "%s #760 HiCache stream quiesce failed (%s); in-flight "
                "device-tier copies may still race this seam's backing "
                "release. The flip continues -- new I/O is refused by the "
                "seam guard regardless.",
                LOG_PREFIX,
                e,
            )
            return 0.0

    def _prearm_quiesce(self, direction: str) -> Tuple[bool, float, str]:
        """#834 A: disarm the device tier and drain it BEFORE the flip arms.

        ANALYSE_830 F1 filed the shape and named both candidates: "quiesce
        BEFORE arming (so the wait happens while the pipeline still runs), or
        refuse to arm while device-tier I/O is in flight instead of arming and
        then waiting". This does the first and uses the second as its ceiling,
        because the two are the same measurement read twice.

        WHY THIS IS A LATENCY MOVE AND NOT A CORRECTNESS CHANGE. The seam's
        drain closes a real race: a HiCache copy enqueued before the seam rides
        the controller's private streams, outlives its Python call, and lands
        in pool memory the seam has already released (two SIGSEGVs, 2026-08-19,
        each three seconds after a cutover). Closing it requires two things to
        hold together -- no NEW device-tier I/O, and no OLD device-tier I/O
        still in flight -- and today BOTH are established at the no-return
        point, with the requests parked and the ring mid-rebuild. Nothing about
        the race requires that timing. Establishing the same two conditions at
        ARM time closes the same race, and the waiting happens while the
        pipeline is still serving instead of while it is stopped.

        THE ORDER IS THE WHOLE THING, and it is the same order the seam uses:
        raise the guard FIRST, then drain. Raised second, the drain would race
        exactly the admissions it was supposed to cover -- ``quiesce_device_io``
        only waits for what is already enqueued, it does not stop enqueueing
        (cache_controller.py, ``stream.synchronize()`` per stream). So the
        guard goes up, and it STAYS up across the armed window; see
        ``_prearm_quiesce_held``.

        THE COST IS NAMED. Between this call and the cutover the device tier is
        disarmed, so HiCache device-tier writes do not happen in that span. That
        span is bounded by the park deadline the arm itself sets, and every path
        that clears ``_pending`` drops the hold. It is a real capability pause,
        not a free win, and it is why this is behind a flag.

        Returns ``(allow, drain_ms, detail)``. ``allow`` is False only when the
        drain this arm just measured is over the #830 F4 budget -- the refusal
        ANALYSE_830 F1 asked for, now resting on a number measured moments ago
        rather than on the previous flip's.
        """
        if not seam_shrink_prearm_quiesce_enabled():
            return True, 0.0, "prearm quiesce off"
        # THE GUARD BEFORE THE DRAIN. Same ordering law as the seam's own, and
        # the same reason: the drain finishes OLD I/O, the guard refuses NEW.
        self.hicache_seam_active = True
        self._prearm_hold_direction = direction
        drain_ms = self._quiesce_hicache(f"{direction} (prearm)")
        self._prearm_drain_ms = drain_ms
        # THE BUDGET'S PROJECTION GETS BETTER HERE, and that is a side effect
        # worth stating. #830 F4 projects the seam window from the LAST flip's
        # drain because "the drain's duration is a property of the backlog on
        # the controller's private streams, which nothing at arm time can
        # enumerate". With the drain pulled forward, arm time CAN enumerate it:
        # it just did. So the seam-budget consumer in ``_execute`` reads a
        # number from this arm rather than from the previous flip, and F4's own
        # stated limitation -- "cannot catch a one-off spike, only a standing
        # condition" -- is narrowed rather than argued away.
        self._seam_drain_ms = drain_ms
        allow, escalated, detail = flip_seam_budget_verdict(
            drain_ms, int(getattr(self, "_prearm_drain_defers", 0))
        )
        if allow:
            if escalated:
                self._prearm_drain_defers = 0
                logger.warning(
                    "%s [#834] prearm drain ESCALATED %s: %s",
                    LOG_PREFIX,
                    direction,
                    detail,
                )
            return True, drain_ms, detail
        # REFUSED, AND THE HOLD IS RELEASED WITH IT. A refused arm leaves
        # nothing pending, so nothing would ever clear a guard left up here --
        # that is the #742 inert-state class, a capability silently dead for
        # the life of the process.
        self._prearm_drain_defers = int(getattr(self, "_prearm_drain_defers", 0)) + 1
        self._release_prearm_quiesce("arm refused on the drain it measured")
        return (
            False,
            drain_ms,
            (
                f"{PREARM_DRAIN_REFUSED}: {detail} The drain was measured AT "
                f"THIS ARM, not projected from the last flip, and the device "
                f"tier has been re-armed -- nothing is pending, so nothing is "
                f"holding it down."
            ),
        )

    def _prearm_quiesce_held(self) -> bool:
        """#834 A: is a pre-arm disarm currently holding the device tier down?

        Read by ``on_round``'s insurance ``finally``, which unconditionally
        clears ``hicache_seam_active`` after EVERY ``_execute`` -- including
        the great majority that return without flipping, because the group is
        not yet quiescent. That clear is correct insurance for a seam and fatal
        for a pre-arm hold: it would drop the guard on the very next round and
        leave the drain covering nothing.
        """
        if getattr(self, "_prearm_hold_direction", None) is None:
            return False
        # BOUND BY THE ARM, NOT BY A TIMER OF ITS OWN. The hold lives exactly
        # as long as something is pending; the park deadline already bounds
        # that, and every path that abandons or completes a flip clears
        # ``_pending``. A second independent deadline here would be a third
        # damper on one decision, which #662 measured the cost of.
        return self._pending is not None

    def _release_prearm_quiesce(self, why: str) -> None:
        """#834 A: drop the pre-arm hold and re-arm the device tier."""
        if getattr(self, "_prearm_hold_direction", None) is None:
            return
        self._prearm_hold_direction = None
        self.hicache_seam_active = False
        logger.info("%s [#834] prearm device-tier hold released (%s)", LOG_PREFIX, why)

    # -- #834 B: the deferred KV grow ---------------------------------------

    def book_deferred_grow(self, direction: str, level) -> None:
        """Record that the seam levelled but did NOT grow (#834 B step 2).

        Called from ``seam_kv_recover`` with the level the group just agreed
        on, or None when no level could be agreed (the #792 decline). The
        agreed level is the ceiling the grow that follows must clamp itself
        back to; without one there is no ceiling, and a grow with no ceiling is
        the id-space divergence this whole split exists to prevent -- so the
        grow is booked either way and refuses to expose either way.
        """
        self._deferred_grow_pending = True
        self._deferred_grow_level = None if level is None else int(level)
        self._deferred_grow_round = int(getattr(self, "_round", 0))
        logger.warning(
            "%s [#834] GROW DEFERRED %s: the collective levelling ran inside "
            "the seam and agreed level=%s; the rank-local grow "
            "(cuMemCreate/cuMemMap, ~99%% of the cutover term per #830 F2) is "
            "booked for the next round, OUTSIDE the no-return window. It will "
            "back rows without exposing them until a later collective raises "
            "the level.",
            LOG_PREFIX,
            direction,
            self._deferred_grow_level,
        )

    def _grow_relief(self):
        """The relief object, or None. Reached through the census scheduler,
        the same handle the drain and the census use."""
        from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

        scheduler = getattr(self, "_census_scheduler", None)
        if scheduler is None:
            return None
        return getattr(scheduler, KV_BACKING_RELIEF_ATTR, None)

    def _pay_deferred_grow(self) -> None:
        """Run the booked rank-local grow, then CLAMP (#834 B steps 2 and 4).

        ORDER IS THE INVARIANT. ``relief.recover()`` ends in
        ``clamp_exposure_to_backing``, which raises this rank's exposure to its
        OWN new backing -- correct when the levelling follows immediately, and
        exactly the hazard when it does not. So the clamp back to the group's
        agreed level is not a tidy-up after the grow, it is the second half of
        it, and the two are never separated by a return.

        NOT WHILE A FLIP IS PENDING. The seam prices its own affordability at
        the gate, and a grow landing between that pricing and the seam would
        spend memory the gate has already promised to the staging fund. There
        is no urgency that justifies it: the debt has a deadline measured in
        rounds, and an armed flip resolves in far fewer.

        NEVER RAISES. This runs on the ordinary round path with requests live;
        a raise here would take the instance down for a capacity repair.
        """
        if not getattr(self, "_deferred_grow_pending", False):
            self._deferred_grow_debt_check()
            return
        if self._pending is not None:
            return
        relief = self._grow_relief()
        if relief is None:
            # Nothing to grow against. Clear the booking rather than carry a
            # debt nobody can pay -- an unpayable debt is a permanent alarm,
            # which is how a real one gets ignored.
            self._deferred_grow_pending = False
            return
        level = getattr(self, "_deferred_grow_level", None)
        try:
            from sglang.srt.managers.phase_flip_spill import (
                clamp_kv_exposure_to_level,
                grow_kv_backing_local,
            )

            scheduler = self._census_scheduler
            grown = int(grow_kv_backing_local(scheduler))
            # #839 B: THE BOOKED LEVEL IS A FLOOR UNDER THE CLAMP, NEVER A
            # CEILING OVER IT. ``grow_kv_backing_local`` ends in the exposure
            # publication, which since #839 puts this rank at the level of the
            # most recent group verdict -- and that verdict may be HIGHER than
            # the one booked when the grow was deferred, because the group's
            # poorest rank has grown too in the meantime. Clamping back to the
            # booking would then take the payment straight back off the table
            # and re-book the same debt, which is the ratchet with an extra
            # step. Take the higher of the two: both are levels a group
            # verdict put this rank at, so neither exposes an unbacked id.
            try:
                reader = getattr(relief, "published_exposure", None)
                published = None if reader is None else reader()
            except Exception:  # noqa: BLE001 - a reading must not break a round
                published = None
            if published is not None:
                level = published if level is None else max(int(level), published)
            if level is None:
                # NO AGREED LEVEL MEANS NO EXPOSURE, and this branch is the
                # #792 decline arriving here rather than being ignored. The
                # rows are backed and stay invisible until a collective can
                # say how far the group can see. That costs capacity and
                # costs no correctness, which is the right way round.
                clamped = 0
            else:
                clamped = int(clamp_kv_exposure_to_level(scheduler, level))
        except Exception as e:  # noqa: BLE001 - a grow must not kill a round
            logger.error(
                "%s [#834] deferred KV grow failed (%s). The pool keeps the "
                "level the seam agreed, so this is a capacity loss and not an "
                "id-space divergence -- the next recovery retries it",
                LOG_PREFIX,
                e,
            )
            self._deferred_grow_pending = False
            return
        self._deferred_grow_pending = False
        self._deferred_grow_rows = self._unlevelled_rows(relief, level)
        if grown or self._deferred_grow_rows:
            logger.warning(
                "%s [#834] GROW PAID at round %d: %d row(s) backed "
                "rank-locally OUTSIDE the no-return window, exposure held at "
                "the group's agreed level=%s (%+d rows moved). %d row(s) are "
                "backed but not yet exposed and stay that way until a "
                "collective levels the group up to them.",
                LOG_PREFIX,
                int(getattr(self, "_round", 0)),
                grown,
                level,
                clamped,
                self._deferred_grow_rows,
            )
        self._deferred_grow_debt_check()

    @staticmethod
    def _unlevelled_rows(relief, level) -> int:
        """Rows this rank has BACKED but may not EXPOSE, or 0 if unknowable."""
        try:
            backed = int(relief.backed_rows())
        except Exception:  # noqa: BLE001 - a debt reading must not break a round
            return 0
        # #839 B: MEASURE THE DEBT AGAINST WHAT THE POOL ACTUALLY EXPOSES, not
        # against the level that was booked when the grow was deferred.
        #
        # The booked level is a number this runtime wrote down once. The
        # allocator's exposure is what admission is really priced against, and
        # after #839 a seam ballot can RAISE it without this booking hearing
        # about it. Reading the booking would leave the alarm shouting a debt
        # the group has already settled -- a latched indicator for a condition
        # that has gone, which is the failure mode the debt check's own
        # docstring says it was written to avoid ("RE-READ, never trust the
        # booking"). It read the booking anyway; this is the reading it meant.
        exposed = None
        try:
            exposed = int(relief.exposed_rows())
        except Exception:  # noqa: BLE001 - a debt reading must not break a round
            exposed = None
        if level is None and exposed is None:
            # Backed with no agreed level at all: every grown row is unexposed.
            # Reported as the full backing rather than 0, because 0 here would
            # read as "no debt" for the state with the LARGEST debt.
            return max(0, backed)
        if level is None:
            settled = int(exposed)
        elif exposed is None:
            settled = int(level)
        else:
            # The HIGHER of the two, because both are levels this rank is
            # entitled to expose and the debt is what neither covers.
            settled = max(int(level), int(exposed))
        return max(0, backed - settled)

    def _deferred_grow_debt_check(self) -> None:
        """#834 B step 4: an unpaid grow debt is #814's ratchet. Say so.

        THE DESIGN NOTE'S FOURTH STEP, verbatim: "Guard the ratchet explicitly:
        a deferred grow that has not drained after N rounds is a loud error
        naming #814's trap, the way the abort window's 'drain missed' check
        does."

        WHAT IT DOES NOT DO IS SELF-HEAL. The tempting repair -- expose the
        rows once the wait gets embarrassing -- is precisely the abort this
        design exists to prevent: an id one rank exposes and a peer cannot map
        aborts all three inside ``store_kvcache``'s bounds assert. So the guard
        is an alarm and never an actuator. It shouts, repeatedly and at ERROR,
        with the numbers a reader needs to act, and it leaves the id space
        alone.
        """
        rows = int(getattr(self, "_deferred_grow_rows", 0) or 0)
        if rows <= 0:
            self._deferred_grow_round = None
            return
        relief = self._grow_relief()
        level = getattr(self, "_deferred_grow_level", None)
        # RE-READ, never trust the booking. A later collective may already have
        # levelled the group up to these rows, in which case the debt is paid
        # and there is nothing to shout about. Reading it fresh is what stops
        # this guard becoming a latched alarm for a condition that has gone.
        if relief is not None:
            rows = self._unlevelled_rows(relief, level)
            self._deferred_grow_rows = rows
        if rows <= 0:
            self._deferred_grow_round = None
            return
        since = getattr(self, "_deferred_grow_round", None)
        patience = seam_shrink_grow_debt_rounds()
        if since is None or patience <= 0:
            return
        waited = int(getattr(self, "_round", 0)) - int(since)
        if waited < patience:
            return
        logger.error(
            "%s [#834] %s: %d row(s) have been BACKED but not EXPOSED for %d "
            "rounds (agreed level=%s). This is #814's ratchet in a new shape "
            "-- the memory is spent and the pool is not getting it, so "
            "admission is capped against capacity that physically exists. The "
            "levelling that would release it runs on the seam's collective "
            "cadence; if no flip is arming, none will. NOT SELF-HEALED ON "
            "PURPOSE: exposing unlevelled ids is what aborts all three ranks "
            "inside store_kvcache's bounds assert.",
            LOG_PREFIX,
            GROW_DEBT_UNPAID,
            rows,
            waited,
            level,
        )
        # Re-base so the alarm repeats on a cadence instead of once, and so a
        # reader can tell a standing debt from one that keeps recurring.
        self._deferred_grow_round = int(getattr(self, "_round", 0))

    def _unlevelled_exposure_refusal(self) -> Optional[str]:
        """#834 B: is this rank exposing ids the group has not levelled to?

        Returns a named refusal detail, or None when the exposure is within
        the agreed level (which includes every configuration where the gate is
        off, because nothing books a debt there).

        READ FROM THE ALLOCATOR, NOT FROM THE BOOKING. The booking says what
        this runtime intended; the allocator says what the pool will actually
        hand out, and only the second one can abort a peer. An indicator that
        reports the intention would pass cleanly through the exact bug it
        exists to catch.

        AN UNREADABLE EXPOSURE IS NOT A REFUSAL. If the relief object cannot
        answer, this returns None and the flip proceeds: a refusal is the thing
        with a service cost, so it must never rest on a number we do not have
        (#721's rule, applied here unchanged).
        """
        if not seam_shrink_defer_grow_enabled():
            return None
        level = getattr(self, "_deferred_grow_level", None)
        relief = self._grow_relief()
        if relief is None:
            return None
        pending = bool(getattr(self, "_deferred_grow_pending", False))
        debt = int(getattr(self, "_deferred_grow_rows", 0) or 0)
        if not pending and debt <= 0:
            return None
        try:
            exposed = int(relief.exposed_rows())
        except Exception:  # noqa: BLE001 - an unknown must not refuse a flip
            return None
        if level is None:
            # A grow booked with NO agreed level. There is no number the
            # exposure may legally sit at, so any exposure at all against an
            # outstanding grow is the divergence.
            if debt <= 0 and not pending:
                return None
            return (
                f"{UNLEVELLED_EXPOSURE_REFUSED}: a deferred KV grow is "
                f"outstanding and the group agreed NO level to expose it at "
                f"(the #792 decline). This rank exposes {exposed} row(s). "
                f"Entering the seam would risk handing out an id a peer "
                f"cannot map, which aborts all three inside store_kvcache's "
                f"bounds assert -- so the flip is refused instead, which "
                f"loses a flip and no ranks."
            )
        if exposed <= int(level):
            return None
        return (
            f"{UNLEVELLED_EXPOSURE_REFUSED}: this rank exposes {exposed} "
            f"row(s) against a group-agreed level of {int(level)}, with "
            f"{debt} row(s) of deferred grow still unlevelled. An id one rank "
            f"exposes and a peer cannot map aborts ALL THREE inside "
            f"store_kvcache's bounds assert; the flip is refused instead."
        )

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
            # Local, like every other kv_row_ownership import in this file: the
            # ownership module must not become an import-time dependency of the
            # flip runtime.
            from sglang.srt.mem_cache.kv_row_ownership import (
                FREE_ENUMERATED,
                read_free_rows,
            )

            scheduler = self._census_scheduler
            if scheduler is None:
                return
            alloc = getattr(scheduler, "token_to_kv_pool_allocator", None)
            tree = getattr(scheduler, "tree_cache", None)
            if alloc is None or tree is None:
                return
            # #832: ASK THE ALLOCATOR HOW IT ACCOUNTS FREE ROWS, DO NOT ASSUME.
            # This line used to be
            #     free = set(alloc.free_pages.tolist()) | set(...release_pages...)
            # which hard-codes a PAGE-LIST allocator. The two unified composite
            # allocators stub both fields to a permanently empty tensor at
            # construction -- "we use watermark math, not free-lists"
            # (multi_ended_allocator.py:1734) -- so on those flavors `free` came
            # out unconditionally EMPTY and every genuinely free row fell into
            # `unaccounted`. The reading did not stay in this log line: the #822
            # audit below declares the same set as the `free_list` owner, so it
            # derived one false ownership violation per free row, every census.
            #
            # ONE AUTHORITY, TWO CONSUMERS. The census line and the audit both
            # read `free_reading` -- never the allocator's fields directly --
            # so they cannot drift apart again. `read_free_rows` also refuses to
            # answer `0` for an allocator it does not understand: an UNKNOWN
            # free count is reported as UNKNOWN, because zero is the one wrong
            # answer that turns the whole pool into a phantom leak (#606).
            free_reading = read_free_rows(alloc)
            free = set(free_reading.rows) if free_reading.is_enumerable else set()
            cached = set(tree.all_values_flatten().tolist())
            size = int(alloc.size)
            # #814: WITHHELD CAPACITY IS NOT A LEAK, and this census read it as
            # one for 73% of a live pool. When ``KvRowCap`` is engaged the ids
            # above the cap are in NEITHER the free lists nor the tree -- and
            # ``alloc.size`` still reports the full pre-shrink id space, since
            # the shrink never rewrites it (allocator/base.py:38). So they fell
            # straight into ``leaked``: measured on this rig, a cap at 124928
            # of 465190 rows printed ``unaccounted=340262``, one contiguous
            # block, flat for the life of the boot.
            #
            # This is the SAME mistake the scheduler's idle invariant made and
            # ``KvRowCap._publish`` (kv_backing_relief.py:530-549) was written
            # to fix -- "without a term of its own it reads as a LEAK -- and it
            # is a fatal one". It publishes ``residency_withheld_slots`` for
            # that check; this census simply never asked for it.
            #
            # SUBTRACT THE RANGE, NOT THE COUNT. The cap withholds every id
            # ABOVE it, so the withheld set is the TOP of the id space and is
            # known exactly. Subtracting a bare count would also swallow an
            # equal number of genuinely unexplained rows below the cap, which
            # is the one thing this census exists to see.
            #
            # The field is published in TOKENS ("the unit available_size()
            # reports"), so a paged lane must divide by page_size to get ids.
            page = max(1, int(getattr(alloc, "page_size", 1) or 1))
            withheld_n = int(getattr(alloc, "residency_withheld_slots", 0) or 0) // page
            withheld_n = max(0, min(withheld_n, size))
            withheld = (
                set(range(size - withheld_n + 1, size + 1)) if withheld_n else set()
            )
            # #832: THE SET DIFFERENCE ONLY EXISTS WHEN THE IDS DO.
            #
            # A watermark allocator can say HOW MANY rows are free and
            # genuinely cannot say WHICH -- space above the watermark has never
            # been minted as ids, so there is no set to subtract. Two different
            # arithmetics, kept visibly apart rather than blended:
            #
            #   enumerable -> the id difference, exactly as before. `leaked` is
            #                 a real set and its sample is meaningful.
            #   counted    -> a COUNT difference. `leaked` stays empty because
            #                 naming ids here would be inventing them, and the
            #                 sample prints as empty for the same reason.
            #   unknown    -> no arithmetic at all. `unaccounted` prints UNKNOWN.
            #                 Substituting 0 free rows would report the entire
            #                 pool as leaked; substituting 0 unaccounted would
            #                 report a clean bill of health. Both are lies with
            #                 opposite signs, which is why neither is used.
            leaked = set()
            if free_reading.is_enumerable:
                leaked = set(range(1, size + 1)) - free - cached - withheld
                unaccounted = len(leaked)
            elif free_reading.count is None:
                unaccounted = "UNKNOWN"
            else:
                # WHAT THE NUMBER MEANS ON A COMPOSITE, because `size` is not
                # a fixed id space there. `UnifiedMambaTokenToKVPoolAllocator.
                # size` is a DYNAMIC property, `full.schedulable_available_size()
                # + full.allocated_count()` (multi_ended_allocator.py:1759-1766),
                # and its `available_size()` is the first of those two terms
                # exactly. So `size - free` cancels to `allocated_count()`, and
                # `unaccounted` becomes "rows the allocator says are live that
                # no enumerated owner claims" -- which is the question this
                # census exists to ask, arrived at without ever enumerating an
                # id. The SWA composite is not as tidy: its `size` is the static
                # `min(_size_full, _size_swa)` while `available_size()` is a
                # joint BYTE budget, so the difference there is a bound rather
                # than an identity. Named because the two composites are not
                # interchangeable, and a reader who assumes they are will
                # over-read the SWA number.
                #
                # Count arithmetic cannot see overlaps, so it can come out
                # NEGATIVE when the enumerated owners overlap the watermark's
                # count. That is reported as it falls: a negative unaccounted
                # means the allocator's own free count and the tree disagree
                # about the same rows, which is a finding, not a display glitch
                # to be clamped away.
                in_space = (cached | withheld) & set(range(1, size + 1))
                unaccounted = size - free_reading.count - len(in_space)
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
            #
            # #1205 -- THE LABELS NAMED THE WRONG POPULATIONS, in both
            # directions at once, on the first line a post-mortem reads.
            # `cur_slot_reqs` carried `len(_live_reqs(scheduler))`, which is the
            # WIDE count across every slot -- the exact opposite of what "cur
            # slot" says. `resident_reqs` sums `len(mb.reqs)` with no dedup
            # while `_live_reqs` dedups by `id()`, so one request resident in
            # two slots printed `cur_slot_reqs=1 resident_reqs=2`: not a request
            # count at all, but a count of LIST ENTRIES. They are now
            # `live_reqs=` and `resident_slot_entries=`, which is what each of
            # them has always measured.
            resident_slot_entries = 0
            slots_with_reqs = []
            for i, mb in enumerate(getattr(scheduler, "running_mbs", []) or []):
                n = len(getattr(mb, "reqs", []) or [])
                if n:
                    resident_slot_entries += n
                    slots_with_reqs.append(i)
            logger.warning(
                "%s POOL CENSUS %s %s: size=%d free=%s cached=%d "
                "withheld=%d available=%s live_reqs=%d resident_slot_entries=%d "
                "resident_slots=%s unaccounted=%s %s alloc=%s free_src=%s",
                LOG_PREFIX,
                when,
                direction,
                size,
                # #832: the allocator's OWN free count, whatever shape it keeps
                # it in -- not `len(free)`, which is 0 on every watermark
                # allocator and was the whole defect. "UNKNOWN" when the
                # allocator answered in no form this census understands.
                "UNKNOWN" if free_reading.count is None else free_reading.count,
                len(cached),
                # Reported, never merely subtracted: ids out of circulation are
                # the single most important fact about a capped pool, and a fix
                # that only hid them would trade a false leak for a silent
                # capacity loss -- the worse of the two.
                len(withheld),
                getattr(alloc, "available_size", lambda: "?")(),
                len(reqs),
                resident_slot_entries,
                slots_with_reqs,
                unaccounted,
                sorted(leaked)[:12],
                # #832 Fenster-4 criterion: a census that does not name the
                # allocator it read cannot be checked against the allocator's
                # own semantics afterwards -- which is why settling the ~94000
                # reading needed a live probe instead of the existing logs.
                free_reading.allocator,
                free_reading,
            )
            # The reason, verbatim, whenever the reading is anything other than
            # a plain corroborated page list. Kept off the ordinary path so the
            # census line stays one line per census on a healthy pool.
            if free_reading.kind != FREE_ENUMERATED:
                logger.warning(
                    "%s POOL CENSUS %s %s free accounting: %s",
                    LOG_PREFIX,
                    when,
                    direction,
                    free_reading.detail,
                )
            self._census_owner_probe(when, direction, alloc, tree, leaked)
            # #822: the same three sets, asked as a LAW instead of printed as
            # an integer. `unaccounted` above cannot distinguish a leak from an
            # unenumerated owner from an over-exposed id space -- and those are
            # #814, the owner probe's ~94000 rows, and #816 respectively. The
            # authority names which one it is, over the COMMITTED backing.
            #
            # GUARDED SEPARATELY, AND THE ATTRIBUTE LOOKUP IS PART OF WHAT IS
            # GUARDED. The rule "a census must never affect the flip it is
            # watching" extends one level down: an auditor must never affect
            # the census it rides on. `_census_ownership_audit` has its own
            # try/except, but that protects only its BODY -- an unbound
            # `self` (this function is exercised bound to a plain namespace)
            # raises AttributeError before the body is entered, which lands in
            # the census's own handler and replaces its line with a failure
            # message. Measured: six #814 census tests went red on exactly
            # that, and the census's whole reason to exist is to still say
            # something when the thing around it is broken.
            try:
                audit = getattr(self, "_census_ownership_audit", None)
                if audit is not None:
                    # #832: the READING, not the set. Passing `free` here is
                    # what made the audit re-derive the census's page-list
                    # assumption and report every free row on a composite
                    # allocator as an ownership violation.
                    audit(
                        f"{when} {direction}",
                        alloc,
                        size,
                        free_reading,
                        cached,
                        withheld,
                    )
            except Exception:  # noqa: BLE001 -- an instrument, never a gate
                logger.debug(
                    "%s census ownership audit skipped", LOG_PREFIX, exc_info=True
                )
        except Exception as exc:  # noqa: BLE001 - a census never breaks a flip
            logger.warning("%s pool census (%s) failed: %s", LOG_PREFIX, when, exc)

    @staticmethod
    def _owner_pool_of(alloc_obj):
        """The KV pool an allocator front-ends.

        LIKE-FOR-LIKE MATTERS HERE. The census holds an ALLOCATOR
        (`scheduler.token_to_kv_pool_allocator`) while the flip's own reshard
        builder names POOLS (`...model_runner.token_to_kv_pool`,
        phase_flip_runtime.py:2041-2042). Comparing an allocator id against a
        pool id answers nothing, so both sides are reduced to the pool before
        the ids are printed.
        """
        for attr in ("_kvcache", "kvcache"):
            pool = getattr(alloc_obj, attr, None)
            if pool is not None:
                return pool
        return None

    @classmethod
    def _owner_ident(cls, label, alloc_obj, tree_obj, pool_obj=None) -> str:
        if pool_obj is None:
            pool_obj = cls._owner_pool_of(alloc_obj)
        return (
            f"{label}: alloc_id={id(alloc_obj) if alloc_obj is not None else None} "
            f"alloc_size={getattr(alloc_obj, 'size', '?')} "
            f"pool_id={id(pool_obj) if pool_obj is not None else None} "
            f"tree_id={id(tree_obj) if tree_obj is not None else None} "
            f"tree_type={type(tree_obj).__name__ if tree_obj is not None else None}"
        )

    def _committed_backing_rows(self) -> Optional[int]:
        """Rows PHYSICALLY BACKED right now, or None when it cannot be measured.

        #822. The census has ``alloc.size`` -- the id space -- and nothing
        else, which is why it could only ever report an integer. The measured
        committed span lives on ``KvBackingRelief._current_rows``
        (kv_backing_relief.py:1171), whose docstring is explicit that reading
        ``pool.size`` in its place cost a boot on 2026-08-11.

        Returns None rather than a guess. An unmeasurable backing must produce
        "unanswerable", never "sound": substituting the id space here would
        report the #816 state as healthy, which is the whole defect.
        """
        try:
            from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

            relief = getattr(self._census_scheduler, KV_BACKING_RELIEF_ATTR, None)
            if relief is None:
                return None
            rows = int(relief._current_rows())
            return rows if rows > 0 else None
        except Exception:  # noqa: BLE001 -- an instrument, never a gate
            return None

    def _census_ownership_audit(
        self, why, alloc, size, free_reading, cached, withheld
    ) -> None:
        """Route one census reading through the #822 authority.

        #832: ``free_reading`` is a :class:`FreeRowReading`, not a set of ids.
        The audit used to receive the census's ``free`` set and so inherited
        its page-list assumption wholesale -- on a composite allocator that set
        is empty by construction, and every genuinely free row was declared
        unowned. A non-enumerable reading is forwarded as ``free_rows=None``,
        which the authority already understands as "this owner could not be
        enumerated" and which suppresses the unowned half exactly as it does
        for an unenumerable resident set. A count is not a set, and pretending
        otherwise is the defect.

        Read-only and best effort, under the same rule as the census that calls
        it: an auditor must never be able to affect the flip it is watching.
        The authority is attached to this runtime, so its EPOCH is the one the
        cutover retires -- a census taken after a cutover therefore reports any
        surviving pre-cutover claim as a retirement violation rather than as an
        out-of-range row.
        """
        try:
            from sglang.srt.mem_cache.kv_row_ownership import (
                audit_pool_census,
                authority_for,
                free_reading_of,
            )

            # A caller that hands over ids has enumerated them; a caller that
            # hands over a reading has asked the allocator. Both are accepted,
            # neither is inferred (#832).
            free_reading = free_reading_of(free_reading)
            authority = authority_for(self, exposed=int(size))
            # #822 root A: the fourth owner. Rows held by in-flight requests
            # are in neither free list nor tree, and without this term every
            # one of them reads as unowned -- 122 rows against one resident
            # request on the first census of boot_window1_0823_1204, before
            # anything had moved. None means the enumeration had no verdict,
            # and is passed through as such rather than as an empty owner.
            resident = _resident_rows(self._census_scheduler)
            found = audit_pool_census(
                authority,
                exposed=int(size),
                committed=self._committed_backing_rows(),
                free_rows=(free_reading.rows if free_reading.is_enumerable else None),
                free_count=free_reading.count,
                free_detail=free_reading.detail,
                cached_rows=cached,
                withheld_rows=withheld,
                resident_rows=None if resident is None else {"requests": resident},
                why=str(why),
            )
            # #919: NAME THE OWNER THE CENSUS DID NOT ENUMERATE.
            #
            # The UNOWNED line already says what it has historically meant --
            # "an un-enumerated second pool object, not a leak" -- and then
            # leaves the reader to find that object by hand. #919 as filed read
            # the same line as "the tree LOSES 4096 rows without free" and hung
            # #842 on it. This answers the question the line leaves open, at
            # the moment it is asked, instead of arguing about the reading.
            #
            # ADDITIVE ONLY: `found` is not touched, no violation is added,
            # removed or reclassified. One extra line per unowned violation.
            #
            # Candidates are resolved PER ACCESS, never held -- a construction
            # reference to a rebindable pool is the #927 class, and this very
            # census reads its own allocator per access for that reason.
            self._note_unenumerated_owner(found)
            # #912: publish the EXCLUSIVITY law's own "claimed by more than one
            # owner" count onto the allocator, in the same place and the same
            # unit KvRowCap already publishes `residency_withheld_slots`
            # (kv_backing_relief.py:800-849). A row the free list AND the tree
            # (or a resident request) both claim is counted once in
            # `available`/`evictable`/`session_held` for EACH claim, which is a
            # SURPLUS against `total`, not a deficit -- exactly the shape
            # test_free_group_lifecycle_827.py's CORRECTION 2 named and left
            # "filed separately" (PP0, [('free_list', 'radix_cache')], 16384
            # rows) and #912 reproduced at 21-22 rows on five idle checks
            # across two boots. The on_idle invariant (invariant_checker.py)
            # had no term for this and read the surplus as a leak; this is
            # that term, sourced from the law that already detects it rather
            # than guessed at the checker.
            from sglang.srt.mem_cache.kv_row_ownership import (
                EXCLUSIVITY_DOUBLED,
                Law,
            )

            # STRUCTURAL discriminator, not prose-matching. LAW.EXCLUSIVITY
            # covers two non-overlapping shapes -- "claimed by more than one
            # owner" (a SURPLUS against `total`, this term's business) and
            # "belong to no enumerated owner" (a DEFICIT, #814/#902's shape,
            # and never this term's business). `Violation.detail` is
            # documented as "never load-bearing" (kv_row_ownership.py's own
            # docstring); matching a substring of it to pick a code path is
            # exactly the "line_gate-Substring-Defekt -> #908" shape, so the
            # two shapes carry a real field (`Violation.kind`) instead, set
            # once at the point each is constructed
            # (kv_row_ownership.py:~491,~557). Letting the unowned/deficit
            # shape in here by accident would mean subtracting the SIZE OF A
            # REAL LEAK from the ledger and calling the result closed.
            double_owned = sum(
                v.rows
                for v in found
                if v.law == Law.EXCLUSIVITY and v.kind == EXCLUSIVITY_DOUBLED
            )
            # #916: THE LEDGER STILL NEEDS THE SHARE THE LAW STOPPED CALLING A
            # VIOLATION. `resident:requests` is a REFERENCE, so the tree and
            # the request that holds its ids no longer produce an
            # EXCLUSIVITY_DOUBLED row -- correctly, because sharing them is how
            # `cache_unfinished_req` works. The on-idle ledger's arithmetic is
            # unmoved by that: such a row is still counted once in `evictable`
            # and again in `session_held`, so it is still a SURPLUS against
            # `total` and must still be subtracted, or #822 root A's "the
            # working set reads as a leak" comes back through the invariant
            # checker instead of through the census.
            #
            # ASKED OF THE AUTHORITY, NOT DERIVED FROM THE VERDICT. Sourcing a
            # correction term from a violation list means the term changes
            # whenever the verdict does -- which is precisely what #916 just
            # did to it.
            double_owned += int(authority.shared_reference_rows())
            alloc_obj = getattr(
                self._census_scheduler, "token_to_kv_pool_allocator", None
            )
            if alloc_obj is not None:
                try:
                    alloc_obj.double_owned_slots = int(double_owned)
                except Exception:  # pragma: no cover - exotic allocator objects
                    pass
        except Exception:  # noqa: BLE001 -- an instrument, never a gate
            logger.debug("%s census ownership audit skipped", LOG_PREFIX, exc_info=True)

    def _unenumerated_owner_candidates(self) -> list:
        """Pool objects the census does NOT enumerate, resolved PER ACCESS.

        Never cached and never held: a construction reference to a rebindable
        pool is the #927 class, which this module's own census already avoids
        by re-reading its allocator from the scheduler each time.

        Two families, and the second is the one this rig makes likely: the
        DRAFT pool (cutover_participants records it as "#861 the draft KV pool
        was never registered -- a whole pool, missed") and the OTHER PHASE's
        stack, which owns its own pool object across a flip. `_census_owner_probe`
        found ~94000 rows owned by a pool object the census never enumerated;
        this names which one rather than counting them again.
        """
        from sglang.srt.mem_cache.kv_row_ownership import OwnerCandidate

        out = []
        sched = self._census_scheduler
        if sched is None:
            return out
        enumerated_alloc = getattr(sched, "token_to_kv_pool_allocator", None)
        enumerated = id(enumerated_alloc)
        # #1050: THE CENSUS'S OWN ID SPACE, so a candidate can be asked whether
        # its range DISCRIMINATES. Measured 2026-08-31 (boot_855_1048fix): the
        # audit reported "pp_stack_allocator owns ids [1, 578995) and covers
        # every sampled row" as the explanation for 43803 unowned rows -- while
        # the census id space was 578994. A containment test against the whole
        # space is true for every sample that could ever be drawn: it excused a
        # real, monotone, ultimately fatal row loss with a test that had no
        # failing case. The candidate is still reported; it is no longer
        # allowed to close the question.
        enumerated_size = getattr(enumerated_alloc, "size", None)
        if not isinstance(enumerated_size, int) or enumerated_size <= 0:
            enumerated_size = None

        def _add(name, obj):
            if obj is None or id(obj) == enumerated:
                return
            size = getattr(obj, "size", None)
            if not isinstance(size, int) or size <= 0:
                return
            discriminating = (
                enumerated_size is not None and (size + 1) < (enumerated_size + 1)
            )
            out.append(
                OwnerCandidate(
                    name=name, lo=1, hi=size + 1, discriminating=discriminating
                )
            )

        _add(
            "draft_allocator", getattr(sched, "draft_token_to_kv_pool_allocator", None)
        )
        _add("draft_pool", getattr(sched, "draft_token_to_kv_pool", None))
        # #941: THE PP STACK IS NOT ON `stacks`, AND LOOKING FOR IT THERE MADE
        # THIS PROBE VACUOUS IN THE ONLY PHASE THAT NEEDED IT.
        #
        # The loop here read `stacks.tp_worker` and `stacks.pp_worker`.
        # `PhaseFlipStacks` (phase_flip_boot.py:784) carries a `tp_worker` and
        # NO `pp_worker`: the PP stack is the scheduler's own `tp_worker`, which
        # is exactly how `_census_owner_probe` twenty lines below reads it, under
        # the label PP_STACK. So the second name resolved to None on every call,
        # and the first resolved -- while the TP phase was live -- to the
        # allocator the census had ALREADY enumerated, which `_add` skips by
        # design. In the TP phase, the only phase in which this rig reports
        # UNOWNED rows at all, the candidate list came out EMPTY and the verdict
        # was the `not candidates` branch: "no second pool object is reachable
        # from here".
        #
        # THAT IS AN ABSTENTION PRINTED AS A NEGATIVE, and it is the reading that
        # sent #938 hunting release paths and write-through acks. The second pool
        # was one attribute away and it had the rows: on the 2k boot the drop's
        # eviction paid them to the PP stack's allocator (#941), and this line
        # was the one instrument whose job was to say so.
        #
        # BOTH DIRECTIONS, ONE LIST. In the PP phase the PP allocator is the
        # enumerated one and the TP allocator is the candidate; in the TP phase
        # it is the other way round. `_add`'s identity skip does the switching,
        # so this names "the other phase's pool" without asking which phase it
        # is in -- a question this frame has no reliable way to answer.
        stacks = getattr(sched, "phase_flip_stacks", None)
        for label, worker in (
            ("tp_stack_allocator", getattr(stacks, "tp_worker", None)),
            ("pp_stack_allocator", getattr(sched, "tp_worker", None)),
        ):
            runner = getattr(worker, "model_runner", None)
            _add(label, getattr(runner, "token_to_kv_pool_allocator", None))
        return out

    def _note_unenumerated_owner(self, found) -> None:
        """#919: one line per UNOWNED violation, naming the missing owner.

        Three-valued, because the two "not a leak" answers and the one that
        sends the hunt onward are different conclusions and must not share a
        line. Never raises and never changes a verdict.
        """
        try:
            from sglang.srt.mem_cache.kv_row_ownership import (
                EXCLUSIVITY_UNOWNED,
                Law,
                unenumerated_owner_verdict,
            )

            unowned = [
                v
                for v in (found or ())
                if v.law == Law.EXCLUSIVITY and v.kind == EXCLUSIVITY_UNOWNED
            ]
            if not unowned:
                return
            candidates = self._unenumerated_owner_candidates()
            for v in unowned:
                verdict, detail = unenumerated_owner_verdict(v.sample, candidates)
                logger.warning(
                    "%s #919 UNOWNED-BLOCK %s: %d row(s), sample=%s -- %s",
                    LOG_PREFIX,
                    verdict,
                    v.rows,
                    list(v.sample),
                    detail,
                )
        except Exception:  # noqa: BLE001 - an instrument, never a gate
            logger.debug("%s #919 candidate probe skipped", LOG_PREFIX, exc_info=True)

    def _enforce_exposure_at_seam(self, when: str) -> int:
        """#851 F1: hold the allocator to the EXPOSURE law at a seam event.

        THE HALF #822 NAMED AND DID NOT BUILD. `fe43b09e52` states it as its
        own open item -- `_retire_row_id_space` "does not yet REFUSE such an id
        at the allocator -- that is enforcement" -- and O-2 recorded the
        audit-only status as an owner decision on hot-path grounds. The
        reversal is scoped to answer exactly that objection: this runs at the
        TWO SEAM EVENTS where the id space changes regime (cutover retirement,
        shrink restatement), never per allocation. The hot path is untouched.

        WHY W22 NEEDED IT. The authority observed `exposed 470755 > committed
        126976` and said so 48 times in one boot while the instance livelocked
        on that exact gap; the #816 clamp fired twice. An instrument that names
        the root once per arm and binds nobody is not a gate.

        THE ACTUATOR IS NOT NEW. This calls `clamp_exposure_to_backing`, whose
        own contract makes it safe here: it "only ever LOWERS exposure toward
        `_current_rows()` -- a MEASURED committed count, never a remembered one
        -- and it never lowers the BACKING". So it cannot cap below a live set
        (the #722 direction, and the reverted floor-clamp remedy); an
        under-backed pool is REPORTED by that actuator, not papered over.

        Returns rows withdrawn, so a caller or a test can assert on the ACTION
        rather than on the absence of a symptom. Never raises: a seam is
        mid-flight and enforcement that kills the cutover is worse than the
        exposure it corrects -- it refuses, it does not explode.
        """
        try:
            from sglang.srt.managers.phase_flip_spill import (
                KV_BACKING_RELIEF_ATTR as _rung_attr,
            )
        except Exception:  # noqa: BLE001 - never break a seam on an import
            logger.warning(
                "%s #851 EXPOSURE ACTUATOR MISSING at %s: the spill module "
                "would not import, so the exposure law has no actuator to "
                "reach. This is a build/packaging fault, not a pool state.",
                LOG_PREFIX,
                when,
            )
            return 0
        sched = getattr(self, "_census_scheduler", None)
        rung = getattr(sched, _rung_attr, None) if sched is not None else None
        if rung is None:
            logger.warning(
                "%s #851 EXPOSURE NOT ENFORCEABLE at %s: no KV backing rung is "
                "wired to this runtime, so the law is installed and unable to "
                "act. A wiring fault -- distinct from a seam that ran and found "
                "nothing to correct, which reports its own marker.",
                LOG_PREFIX,
                when,
            )
            return 0
        try:
            withdrawn = int(rung.clamp_exposure_to_backing(when) or 0)
        except Exception:  # noqa: BLE001 - a seam may refuse, never explode
            # #853(i): AT WARNING, NOT DEBUG. W24 ran at INFO, so the old DEBUG
            # line meant a clamp that threw on every single seam would leave the
            # boot log identical to a clamp that found nothing wrong.
            logger.warning(
                "%s #851 EXPOSURE CHECK FAILED at %s: the clamp raised and the "
                "seam refused rather than exploding. Exposure is UNVERIFIED for "
                "this event -- not verified-and-clean.",
                LOG_PREFIX,
                when,
                exc_info=True,
            )
            return 0
        if withdrawn <= 0:
            # #853(i) THE READING W24 COULD NOT TAKE. A silent zero here made
            # "EXPOSURE ENFORCED = 0" mean any of five things: healthy, inert,
            # broken, unreachable, or never built. Each of the other four now
            # has its own marker above, which leaves this line to mean exactly
            # one thing -- the law RAN and the id space was already sound.
            logger.info(
                "%s #851 EXPOSURE CHECKED at %s: the exposed id space is "
                "within its backing, nothing withdrawn.",
                LOG_PREFIX,
                when,
            )
        if withdrawn > 0:
            logger.warning(
                "%s #851 EXPOSURE ENFORCED at %s: withdrew %d over-exposed row "
                "id(s) so the allocator can no longer hand out an id with no "
                "page behind it. The id space, not the backing, was corrected.",
                LOG_PREFIX,
                when,
                withdrawn,
            )
        return withdrawn

    def _retire_row_id_space(self, direction) -> None:
        """The cutover retires the whole old id space in ONE step (#822).

        THE GENERALIZATION OF THE LINE BELOW IT. ``self._parked_extent = None``
        (#746) and ``last_req_extent``'s layout tag (#802, 689161de77) each
        clear ONE holder of a pre-cutover row id by hand, because the #802
        rule is stated correctly -- "a row id only
        means something relative to the pool it was enumerated in, so anything
        that stores a row id across a possible cutover has to store WHICH pool
        it came from" -- and then leaves every holder to obey it individually.
        #796's id 344009 surviving against a TP cap of 212992 is what one
        missed holder costs.

        Enumerating holders is the thing that kept being incomplete, so this
        does not enumerate them: it retires the SPACE. Every id minted before
        this instant is unlawful by epoch afterwards, without being touched.

        SCOPE, STATED HONESTLY. This arms the AUDIT: after it, a claim carrying
        a pre-cutover stamp is reported as a retirement violation. It does not
        yet REFUSE such an id at the allocator -- that is enforcement, it
        belongs on the hot path, and it cannot be validated anywhere but on
        metal under a real flip. Carried as an 18-lane window item, not claimed
        here.
        """
        try:
            from sglang.srt.mem_cache.kv_row_ownership import authority_for

            alloc = getattr(self._census_scheduler, "token_to_kv_pool_allocator", None)
            exposed = int(getattr(alloc, "size", 0) or 0)
            committed = self._committed_backing_rows()
            authority = authority_for(self, exposed=exposed)
            dropped = authority.retire(
                exposed=exposed,
                committed=exposed if committed is None else committed,
            )
            # #912: a `double_owned_slots` reading published by the LAST
            # census is a snapshot of the OLD id space's claims. Retirement
            # just dropped every one of those claims by epoch (`dropped`
            # above), so the snapshot is now unconditionally stale -- it
            # cannot be re-validated by comparing it to anything, because
            # there is nothing left to compare it against until the next
            # census (`_pool_census("post-cutover", ...)`, called right after
            # this) republishes a fresh reading. Clearing it to `None` here
            # (never `0`) means the on-idle check reads it via
            # `getattr(..., "double_owned_slots", 0) or 0` as "no correction
            # currently known" and falls back to the pre-#912 raw comparison
            # for the short window before the post-cutover census runs --
            # erring towards a possible false leak flag in that window rather
            # than letting a pre-cutover number silently correct a post-
            # cutover ledger it was never measured against.
            if alloc is not None:
                try:
                    alloc.double_owned_slots = None
                except Exception:  # pragma: no cover - exotic allocator objects
                    pass
            logger.info(
                "%s ID-SPACE RETIRED at %s cutover: epoch=%d dropped=%d "
                "exposed=%d committed=%s",
                LOG_PREFIX,
                direction,
                authority.epoch,
                dropped,
                exposed,
                "unmeasured" if committed is None else committed,
            )
        except Exception:  # noqa: BLE001 -- an instrument, never a gate
            logger.debug("%s id-space retirement skipped", LOG_PREFIX, exc_info=True)

    def _census_owner_probe(self, when, direction, alloc, tree, leaked) -> None:
        """Name every pool object, not just the census's own.

        WHY. `_pool_census` above derives `unaccounted` from EXACTLY ONE
        allocator and ONE tree, both read off the scheduler, and
        `all_values_flatten` collects only BASE-component DEVICE values. So a
        row owned by a different pool object, or hanging off a different tree,
        is unaccounted BY DEFINITION rather than by defect -- and the two are
        indistinguishable in the census line. On the 2026-08-22 r5 flip that
        ambiguity was load-bearing: ~94000 rows (21% of a 448698-row pool)
        read as unaccounted, FLAT across four censuses, which is the signature
        of an unenumerated owner rather than of a leak (a leak accumulates).
        A second owner is known to be possible on exactly the stack those
        flips wedge in -- `model_runner.py` splits `is_draft_pool_worker` from
        `is_draft_worker` on `is_phase_flip_tp_stack`, so a runner there owns
        pools the scheduler's handles do not name.

        Also reports the SHAPE of the unaccounted set. The census prints
        `sorted(leaked)[:12]`, and twelve consecutive ids at the minimum say
        nothing about the rest -- a contiguity claim was made and withdrawn on
        exactly that evidence. `runs`/`longest_run` answer it in bounded
        output instead of dumping ~94000 ids.

        Read-only and best effort, preserving the rule that a census can never
        affect the flip it is watching.
        """
        try:
            scheduler = self._census_scheduler
            lines = [self._owner_ident("CENSUS", alloc, tree)]
            # phase_flip_runtime.py:2041 -- the PP-side pool the reshard
            # builder names. If CENSUS pool_id tracks this one in BOTH phases
            # while the TP stack is live, the census is scoped to the PP stack
            # and the "unaccounted" rows are a measurement artefact.
            pp_runner = getattr(
                getattr(scheduler, "tp_worker", None), "model_runner", None
            )
            lines.append(
                self._owner_ident(
                    "PP_STACK",
                    getattr(pp_runner, "token_to_kv_pool_allocator", None),
                    getattr(pp_runner, "tree_cache", None),
                    pool_obj=getattr(pp_runner, "token_to_kv_pool", None),
                )
            )

            stacks = getattr(scheduler, "phase_flip_stacks", None)
            if stacks is None:
                lines.append("STACKS: None at census time")
            else:
                for label, worker in (
                    ("TP", getattr(stacks, "tp_worker", None)),
                    ("DRAFT", getattr(stacks, "draft_worker", None)),
                ):
                    if worker is None:
                        lines.append(f"{label}: worker None")
                        continue
                    # Mirrors draft_kv_pool()'s traversal exactly, so a null
                    # result here means what it means there.
                    inner = getattr(worker, "draft_worker", None) or worker
                    runner = (
                        getattr(inner, "model_runner", None)
                        or getattr(inner, "draft_runner", None)
                        or inner
                    )
                    lines.append(
                        self._owner_ident(
                            label,
                            getattr(runner, "token_to_kv_pool_allocator", None),
                            getattr(runner, "tree_cache", None),
                            pool_obj=getattr(runner, "token_to_kv_pool", None),
                        )
                        + f" flip_tp={getattr(runner, 'is_phase_flip_tp_stack', '?')}"
                        + f" draft_pool={getattr(runner, 'is_draft_pool_worker', '?')}"
                    )

            if leaked:
                ids = sorted(leaked)
                runs, longest, cur = 1, 1, 1
                for a, b in zip(ids, ids[1:]):
                    if b == a + 1:
                        cur += 1
                    else:
                        runs += 1
                        longest = max(longest, cur)
                        cur = 1
                longest = max(longest, cur)
                lines.append(
                    f"UNACCOUNTED: n={len(ids)} min={ids[0]} max={ids[-1]} "
                    f"runs={runs} longest_run={longest}"
                )
            else:
                lines.append("UNACCOUNTED: n=0")

            logger.warning(
                "%s POOL OWNERS %s %s | %s",
                LOG_PREFIX,
                when,
                direction,
                " | ".join(lines),
            )
        except Exception as exc:  # noqa: BLE001 - a probe never breaks a flip
            logger.warning("%s pool owner probe (%s) failed: %s", LOG_PREFIX, when, exc)

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
        # #834 A: nothing is pending any more, so nothing may keep the
        # device tier disarmed. Released HERE rather than left to the
        # round hook's insurance so the tier comes back in the same
        # step the flip ends in, not one round later.
        release_prearm_quiesce(self, "nothing pending")
        self._parked_extent = None  # #746: a snapshot never outlives its flip
        self._armed_residents = {}  # #1202: nor does the resident ledger
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
        # RUNG 4 JOINS THE DRAFT RESTORE INSIDE THE max(), SUMMED WITH IT.
        # Both are cutover-instant commits, and the cutover's own ordering
        # makes them coexist -- rung 2 restores the weight pages first so the
        # deferred capture has addresses to bake. See _cold_stack_restore_bytes.
        #
        # THE SHAPE OF THIS RETURN IS PINNED, deliberately, by
        # test_phase_flip_arena_tail_631.ThePendingCommitIsPricedTest: the tail
        # is ADDED to the max() and must never become a third argument of it.
        # Keep the tail and the max() on their own lines -- the pin reads the
        # source text, because the difference it guards is one measured
        # corridor breach wide (MERGE-R9 12.4) and is invisible in a value.
        return int(
            self._arena_tail_bytes(direction)
            + max(
                wave_peak,
                self._draft_restore_bytes(direction)
                + self._cold_stack_restore_bytes(direction),
            )
        )

    def _seam_reserve_bytes(self, tr, direction: str, src, dst, waves=None) -> int:
        """What the seam ACTUALLY reserves, which is no longer what a move
        would need (#856).

        TWO DIFFERENT QUESTIONS, TWO NAMES. ``_staging_bytes`` answers "what
        would this move need?" and its formula -- pinned by
        ``test_phase_flip_staging_reserve_631`` and
        ``test_seam_arena_tail_additive_656`` against measured corridor events
        -- is still exactly right for that question. It is simply no longer
        the question the gate asks, because the flip carries no KV: the
        residents are retracted and the plan is rebuilt EMPTY before the wave
        loop (``_release_residents_for_cutover``), so nothing is moved and
        every ``wave_peak`` term prices a transient that cannot occur.

        Collapsing the two into one name is the defect this whole build keeps
        removing, so the move's price keeps its name and its tests, and the
        seam's reserve gets its own.

        WHAT THIS REMOVES, in W25's numbers: a 2339.11 MiB ``tp_to_pp``
        staging ask on PP0, behind 33 refused arms -- 25 of them on the
        staging rate limit -- and 17 FLIP ABANDONED.

        The move's price is still COMPUTED and recorded, because the
        difference between the two IS the funding claim the proof window has
        to check, and a term that vanishes silently cannot be shown to have
        been retired.
        """
        would_move = int(self._staging_bytes(tr, direction, src, dst, waves))
        self._retired_wave_peak_bytes = would_move
        return int(
            self._arena_tail_bytes(direction)
            + self._draft_restore_bytes(direction)
            + self._cold_stack_restore_bytes(direction)
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

    def _cold_stack_restore_bytes(self, direction: str) -> int:
        """Device bytes the pp->tp leg must commit for the DEFERRED cold posts.

        RUNG 4's mirror of ``_draft_restore_bytes``, and it is here for exactly
        the same reason: when the boot deferred the flip TP stack's attention
        workspaces and decode graphs, the pp->tp ``_cutover`` is what builds
        them -- past the point of no return. An allocation failure there cannot
        be unwound, and it is the same failure shape that took all three ranks
        down on 2026-08-09 when rung 2's re-commit was unpriced. Pricing it
        turns that death into a unanimous, free abandon before a byte moves.

        THE NUMBER IS THE MEASURED RESIDUAL, the same constant the KV sizer
        took as a credit (``arena_tail_probe.STACK_RESIDUAL_MIB``). Using one
        constant for both sides is deliberate: the credit and the charge are
        the same bytes seen at the two ends of the deferral, and if they could
        drift apart the pool would be sized against one number and funded
        against another.

        SUMMED WITH THE DRAFT RESTORE, not max()'d with it. Both commits run
        inside the same ``_cutover``, and the ordering makes them coexist: rung
        2 puts the weight pages back FIRST precisely so the capture has
        something to bake addresses into, so at the instant the graphs are
        captured the restored weights are still held. max() would model a peak
        that the code's own ordering rules out.

        Zero on the tp->pp leg, zero at every rung that does not defer, and
        zero once the posts exist -- which is every flip after the first. A
        charge that outlived the build would shrink the staging budget forever
        and abandon flips that fit, against a record of 0 abandons in 402.
        """
        if direction != PP_TO_TP:
            return 0
        # getattr throughout: _staging_bytes is exercised by unit stubs built
        # with object.__new__, which carry none of the runtime's fields.
        scheduler = getattr(self, "_census_scheduler", None)
        if scheduler is None:  # unit stubs
            return 0
        try:
            from sglang.srt.managers.arena_tail_probe import STACK_RESIDUAL_MIB
            from sglang.srt.managers.phase_flip_boot import COLD_STACK_BUILT_ATTR
            from sglang.srt.managers.phase_flip_spill import cold_stack_deferred

            if not cold_stack_deferred(getattr(scheduler, "server_args", None)):
                return 0
            stacks = getattr(scheduler, "phase_flip_stacks", None)
            tp_worker = getattr(stacks, "tp_worker", None)
            if tp_worker is None or getattr(tp_worker, COLD_STACK_BUILT_ATTR, False):
                return 0
            rank = int(self._rank)
            if not 0 <= rank < len(STACK_RESIDUAL_MIB):
                return 0
            return int(STACK_RESIDUAL_MIB[rank]) * 1048576
        except Exception as e:  # pragma: no cover - never block a flip on this
            # A gate that cannot price the build must not also refuse it.
            logger.warning(
                "%s could not price the deferred cold-stack build, gating on "
                "the wave peak alone: %s",
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

    @staticmethod
    def _torch_releasable_cache_bytes():
        """#852: bytes ``empty_cache()`` CAN hand the driver, or None.

        THE DEVICE HALF ONLY. Every judgement -- the arithmetic and all three
        abstentions -- lives in ``releasable_cache_bytes_from_stats``, which
        needs no GPU and is therefore falsifiable in both directions. What is
        left here is the sampling: the counters, and the allocator config the
        expandable-segments abstention keys on. A rule that can only be
        exercised on metal is a rule this corpus has repeatedly shipped inert.
        """
        try:
            if not torch.cuda.is_available():
                return None
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            # #852 R3: the SEGMENT view first, because it is the exact answer
            # and the three-term arithmetic is only a proxy for it. The proxy
            # over-promised a stable 88 MiB in W25 -- free blocks sitting in
            # CUDA-graph PRIVATE pools, which `empty_cache()` provably never
            # returns while the graphs live, but which the device-global
            # `.all` counters happily count. Falls back to the proxy when the
            # snapshot cannot be read, so an unreadable snapshot costs
            # precision, never behaviour.
            try:
                exact = releasable_cache_bytes_from_segments(
                    torch.cuda.memory_snapshot(), alloc_conf=alloc_conf
                )
            except Exception:  # noqa: BLE001 - the proxy is still available
                exact = None
            if exact is not None:
                return exact
            return releasable_cache_bytes_from_stats(
                torch.cuda.memory_stats(),
                alloc_conf=alloc_conf,
            )
        except Exception:  # noqa: BLE001 - a measurement may abstain, never break
            return None

    @staticmethod
    def _torch_graph_pool_free_bytes():
        """#852 R3 device half: free bytes trapped in CUDA-graph pools."""
        try:
            if not torch.cuda.is_available():
                return None
            return graph_pool_free_bytes_from_segments(torch.cuda.memory_snapshot())
        except Exception:  # noqa: BLE001 - an instrument, never a gate
            return None

    def _graph_pool_free_bytes(self):
        """Injectable like the other probes, so the discriminator is
        falsifiable without a GPU."""
        hook = getattr(self, "_mem_graph_pool_free", None)
        if hook is not None:
            try:
                return hook()
            except Exception:  # noqa: BLE001 - an abstaining probe, not a crash
                return None
        return PhaseFlipRuntime._torch_graph_pool_free_bytes()

    def _release_residents_for_cutover(self, direction: str):
        """#856: the flip carries NO KV. Retract the residents, drop the tree.

        Runs AFTER the fence (#703 writeback) and the HiCache quiesce, so every
        prefix worth keeping is already in the canonical store, and BEFORE the
        transfer plan is rebuilt, so the plan it is rebuilt on is empty and the
        wave loop has nothing to move.

        Order is enforced by ``release_residents_for_cutover``; see its
        docstring for why reversing it reproduces #825's three-rank crash.

        RAISES rather than continues if the scheduler cannot supply a
        resettable tree. Entering the next phase with a tree that names rows
        holding no KV is a wrong answer, and the seam's own rule is that a
        wrong answer is worse than a loud failure. There is deliberately no
        fallback to the mover: a flip that cannot honour this contract must
        not happen at all.
        """
        scheduler = getattr(self, "_census_scheduler", None)
        if scheduler is None:
            raise SeamOrderError(
                "#856: no scheduler bound at the seam, so the residents cannot "
                "be retracted and the prefix tree cannot be dropped"
            )
        built = build_cutover_release(scheduler)
        if built is None:
            raise SeamOrderError(
                "#856: this scheduler exposes no resettable tree_cache; the "
                "flip carries no KV, so without dropping the tree the next "
                "phase would read prefixes naming rows that hold no KV"
            )
        retract, reset_tree = built

        def _retract_and_consume(rs):
            out = retract(rs)
            # #856 W27 ROOT FIX: freeing the resources is only half of it. The
            # request must also leave the live universe, or every seam
            # consumer after this point is handed a live request whose rows,
            # mamba slot and tree lock are already gone -- which is exactly
            # how W27 died in `resident_mamba_slots`.
            self._retracted_refs_retired = consume_retracted_from_live_universe(
                scheduler, rs
            )
            # #792 ORDERING: ACK BEFORE THE DROP, or the anchor dies in the
            # cutover it was built for.
            #
            # `reset_tree()` runs on the very next line of
            # `release_residents_for_cutover`, and #924 traced where that goes:
            # `_drop_tree` -> `drop_prefix_tree_returning_rows` -> `tree.evict`
            # -> `_evict_component_and_detach_lru` ->
            # `mamba_component.evict_component` -> `_free_mamba_value` ->
            # `mamba_allocator.free`. The cutover evicts the prefix tree
            # INCLUDING its mamba values. The boundary anchors the retraction
            # just donated are device-side at that instant, so unless their
            # write-through has been ACKED they are freed by the same cutover.
            #
            # The #703 fence is exactly this operation and it already awaits
            # storage acks (`_await_storage_acks`). It ran earlier in the
            # cutover, BEFORE the retraction -- so it covered the prefixes that
            # were already in the tree and could not cover the anchors that did
            # not exist yet. Running it once more here, after the donations and
            # before the drop, is the same proven primitive at the point where
            # the new anchors are the thing that needs persisting.
            #
            # Best-effort by construction: a fence that cannot run leaves the
            # anchor device-only, which degrades to a recompute after the flip
            # -- today's behaviour -- and never to a wrong answer.
            try:
                from sglang.srt.mem_cache.hicache_flip_writeback import (
                    maybe_flip_writeback,
                )

                post_report = maybe_flip_writeback(scheduler)
                if post_report is not None:
                    self._last_retract_writeback_report = post_report
                    logger.info(
                        "%s #792 post-retract writeback fence: %s",
                        LOG_PREFIX,
                        (
                            post_report.as_log()
                            if hasattr(post_report, "as_log")
                            else post_report
                        ),
                    )
            except Exception as exc:  # noqa: BLE001 - never break the cutover
                logger.warning(
                    "%s #792 post-retract writeback fence did not run (%s); the "
                    "boundary anchors stay device-only and are dropped with the "
                    "tree, so re-admission recomputes as it does today.",
                    LOG_PREFIX,
                    exc,
                )
            return out

        # W29: SAY HOW MANY ROWS THE DROP RETURNED. `drop_prefix_tree_
        # returning_rows` computes exactly that and its own docstring says the
        # number "has to be visible" -- and then the seam threw it away, so
        # the drop that silently returned ZERO rows on every flip of the W29
        # boot looked identical in the log to one that returned all of them.
        # The only reader left was the pool census, one pass later, and by
        # then the arithmetic no longer names the owner.
        tree_rows = {"returned": None}

        def _drop_and_record():
            tree_rows["returned"] = reset_tree()
            return tree_rows["returned"]

        # #1202 THE LOAD-BEARING HALF: RETRACT THE SET THE ARM APPROVED.
        #
        # This line read `list(_live_reqs(scheduler))` -- an enumeration
        # taken HERE, one second after the arm read the same authority and
        # got a different answer on two of three ranks. See
        # `cutover_resident_set` for the measured evidence and for why the
        # reconciliation is a FILTER over the armed ledger rather than a
        # union with it.
        #
        # WHY A SNAPSHOT AND NOT A RE-RUN OF THE QUIESCENCE TERM HERE.
        # This point is past the no-return: `hicache_seam_active` is up,
        # the device tier is drained, and PP0's PROCEED decision has
        # already ridden the request stream to every rank. A quiescence
        # term evaluated here could only produce a rank-local verdict at
        # the one place no rank may hold an opinion -- precisely the shape
        # #969 §W3 deleted (`_arm_as_follower`: "a rank-local opinion that
        # can refuse an arm PP0 accepted is how the ranks end up uneins").
        # A snapshot is not a verdict: it changes WHAT is retracted, never
        # WHETHER the group cuts over, and it is inert when the two
        # readings agree.
        reqs, reconcile = cutover_resident_set(
            scheduler, getattr(self, "_armed_residents", None)
        )
        if reconcile["carried_from_arm"] or reconcile["skipped_pool_unreadable"]:
            logger.warning(
                "%s #1202 ARM/RELEASE RESIDENCY RECONCILED for %s: the "
                "release-instant enumeration saw %d request(s); the armed "
                "window saw %d; %d of those had become invisible while "
                "still holding a request-pool row and are retracted here "
                "(%s). Dropped as already-returned: %d; as reallocated to "
                "a live request: %d; as rowless: %d; as unverifiable "
                "(pool unreadable, never retracted blind): %d.",
                LOG_PREFIX,
                direction,
                reconcile["live_now"],
                reconcile["from_arm_ledger"],
                reconcile["carried_from_arm"],
                reconcile["carried_rids"],
                reconcile["skipped_row_free"],
                reconcile["skipped_row_reallocated"],
                reconcile["skipped_no_row"],
                reconcile["skipped_pool_unreadable"],
            )
        self._last_residency_reconcile = reconcile
        released = release_residents_for_cutover(
            reqs, retract=_retract_and_consume, reset_tree=_drop_and_record
        )
        n = len(released or ())
        self.residents_released = int(getattr(self, "residents_released", 0)) + n
        self.tree_rows_returned = tree_rows["returned"]
        # #1028: THE PROMISE IS NOW BOUND TO THE FENCE'S OWN NUMBERS.
        #
        # This line asserted "Their KV is in the canonical store from the
        # fence" UNCONDITIONALLY. Measured on boot_855_wt1016 19:22:45-46, the
        # post-retract fence had just logged `acked=1 outstanding=3` and this
        # line still made the promise one second later; the re-admission then
        # reported `host_hit=0 storage_hit=0` and recomputed 13180 tokens. Two
        # instruments of the same cutover, same second, opposite claims -- the
        # instrument-text class, where the text describes something the code
        # never checked. It now reads the report it is speaking for.
        _post = getattr(self, "_last_retract_writeback_report", None)
        _outstanding = None if _post is None else int(getattr(_post, "outstanding", 0))
        if _outstanding:
            _kv_claim = (
                f"WARNING: {_outstanding} backup(s) were STILL IN FLIGHT at the "
                f"fence, so that many prefixes are NOT in the canonical store "
                f"and their requests will recompute in full rather than serve "
                f"by read-through (#1028)"
            )
        elif _post is None:
            _kv_claim = (
                "the fence reported nothing on this cutover, so whether their KV "
                "reached the canonical store is UNMEASURED here (#1028)"
            )
        else:
            _kv_claim = (
                "their KV is in the canonical store from the fence (acked, "
                "0 outstanding); the new layout re-admits them and serves the "
                "prefix by read-through"
            )
        logger.info(
            "%s RESIDENTS RELEASED for %s: %d request(s) retracted, %d live "
            "reference(s) retired, and the prefix tree dropped returning %s "
            "row(s) to the allocator, in that order (#856). %s. Nothing is "
            "carried across, and nothing retracted is still live.",
            LOG_PREFIX,
            direction,
            n,
            int(getattr(self, "_retracted_refs_retired", 0)),
            "UNKNOWN" if tree_rows["returned"] is None else tree_rows["returned"],
            _kv_claim,
        )
        # W31: RE-ADMIT THEM. THIS IS THE HALF #856 NEVER SHIPPED.
        #
        # The line just logged has promised, on every flip of every boot, that
        # "the new layout re-admits them and serves the prefix by
        # read-through". Nothing did. `released` was computed here and thrown
        # away by the caller, so W30 and W31 both dropped every request the
        # seam retracted -- 78 of them in W31 arm 2, with zero completions and
        # every client timing out at 600 s.
        #
        # THE ABORT PATH IS CORRECT BY CONSTRUCTION BECAUSE IT HAPPENS HERE.
        # Requeuing at the release site means there is NO window in which the
        # list exists and is not owned by someone: if the cutover raises after
        # this point the flip abandons, the layout is unchanged, and the
        # requests are already back on the queue of the SOURCE layout, which
        # is exactly where an abandoned flip should leave them. Deferring the
        # requeue to the end of the cutover would recreate the current defect
        # for precisely the abort case.
        #
        # ORDER (#731 shape): `_retract_and_consume` has already removed every
        # one of these from the live universe. Consume FIRST, requeue SECOND,
        # never both at once -- a request that is simultaneously
        # live-referenced and queued is double-billed by every consumer that
        # sums the two.
        #
        # RANK-UNIFORM: the cutover is group-unanimous and retracts the same
        # resident set on every rank, `kv_arrival_seq` is assigned identically
        # on every rank (its own comment: "the admission order is identical on
        # every TP rank, so the counter is rank-uniform"), and the ordering
        # below is a pure function of it. So every rank rebuilds the same
        # queue in the same order -- the property `waiting_queue` consumers
        # already depend on.
        # #703 FENCE COVERAGE, ASSERTED RATHER THAN ASSUMED.
        #
        # The re-admission's whole premise is that these prefixes are already
        # in the canonical store, so the read-through hits and nothing is
        # recomputed. That premise is the FENCE's output, and
        # `_writeback_fence_ms` returns None for "NO FENCE RAN" -- a state
        # that is real (the fence is skipped outright without a canonical
        # store) and that must never be silently read as "fenced, cost 0 ms".
        # Re-admitting into an unfenced instance is not a wrong answer -- the
        # prefix simply misses and is recomputed -- but it is a silent
        # performance cliff on the exact path this design is built to avoid,
        # so it is named here.
        if (
            released
            and _writeback_fence_ms(getattr(self, "_last_writeback_report", None))
            is None
        ):
            logger.warning(
                "%s #856/W31: re-admitting %d resident(s) for %s with NO "
                "WRITEBACK FENCE RECORDED. Their prefixes may not be in the "
                "canonical store, so the read-through this design promises "
                "can only MISS and the tokens are recomputed. Not a wrong "
                "answer, but the cliff this whole no-carry seam exists to "
                "avoid -- and it is named rather than inferred from a slow "
                "boot later.",
                LOG_PREFIX,
                n,
                direction,
            )
        # #783: AND THE FENCE THAT RAN AND PERSISTED NOTHING.
        #
        # The check above asks "did a fence run". W37-G proved that is the wrong
        # question: a fence over an empty tree returns `elapsed_s=0.0`, not
        # None, so the check above stayed silent on 33 of 39 cutovers while
        # `acked=0` held for every fence of the entire boot. `report.complete`
        # called those same 33 complete, because `outstanding == 0` is trivially
        # true when there was nothing to send. Two instruments, both correct,
        # both blind to the one state that matters -- and `#cached-token: 0` on
        # all 209 prefill batch lines is what it cost.
        #
        # Read defensively: a stand-in report is the ordinary state in tests and
        # an instrument may never be the thing that breaks a flip.
        # #871 THE STREAK, AGGREGATED AT THE INSTRUMENT THAT ALREADY EXISTS.
        #
        # `persisted_nothing` already answers the per-cutover question, and the
        # warning below already fires. What it cannot say is the one thing that
        # decides whether the instance can make progress at all: ONE empty
        # fence is legitimate (nothing was persistable), EVERY empty fence
        # means the canonical store can never be populated, so the read-through
        # the no-carry seam depends on can never hit -- and that shows up only
        # as latency, never as an error. W37-G is the specimen (12 flips, zero
        # completions); the W40 #857 boot is the milder form (57 of 186 fences
        # persisted nothing and `#cached-token: 0` on all 243 prefill batches).
        #
        # NO SECOND COUNTER: this reads `persisted_nothing` off the existing
        # report and keeps one int. A parallel "recomputed prefix tokens"
        # counter would measure what this and `#cached-token` already measure
        # between them, which is how #838 acquired a silent multi-valued zero.
        #
        # GATED ON `released`, MIRRORING #719's BUSY GATE at :8678, and for the
        # same reason that gate exists: a fence over an empty tree is CORRECT
        # to persist nothing, so counting quiet cutovers would build a
        # crying-wolf alarm out of the instrument written to replace one. The
        # streak advances only when this cutover actually retracted residents
        # -- real work was taken away AND nothing was persisted to give it
        # back. Threshold 4, the same "two quiet cutovers are ordinary" reading
        # #719 settled on.
        _persisted_nothing = bool(
            getattr(
                getattr(self, "_last_writeback_report", None),
                "persisted_nothing",
                False,
            )
        )
        try:
            _pn_streak = advance_fence_blind_streak(
                getattr(self, "_fence_persisted_nothing_streak", 0),
                released=bool(released),
                persisted_nothing=_persisted_nothing,
            )
            self._fence_persisted_nothing_streak = _pn_streak
            if _pn_streak >= FENCE_BLIND_STREAK:
                logger.error(
                    "%s #871 FENCE BLIND: the writeback fence PERSISTED NOTHING "
                    "on %d consecutive cutovers that retracted work. The "
                    "canonical store cannot be populated, so every read-through "
                    "after every cutover misses and each retracted prefix is "
                    "recomputed in full -- the instance pays a full re-prefill "
                    "per flip and the cost appears only as latency. Two causes "
                    "carry this, both already diagnosed: the #718/#847 rebind "
                    "refusing on pool coverage (grep '#719 HiCache rebind "
                    "refused' -- if it refuses every cutover the phase host "
                    "tier is not built with the full pool set, see #871), and "
                    "finish-only retention (a request retracted mid-generation "
                    "never inserts its output tokens, so the next fence has "
                    "nothing to persist). A store whose only writer is an event "
                    "the seam preempts can never be read.",
                    LOG_PREFIX,
                    _pn_streak,
                )
        except Exception:  # noqa: BLE001 - telemetry never breaks a seam
            pass
        if released and _persisted_nothing:
            logger.warning(
                "%s #783: re-admitting %d resident(s) for %s after a fence that "
                "RAN AND PERSISTED NOTHING (%s). No storage ack and no host "
                "copy, so neither tier can serve the read-through: the device "
                "tree is dropped by law at this seam and the canonical store "
                "was never written. Every re-admitted prefix MISSES and is "
                "recomputed in full. If this repeats across cutovers the "
                "instance cannot make progress -- requests are retracted before "
                "they finish, retention is finish-only, so nothing is ever "
                "inserted for the next fence to persist (W37-G: 12 flips, zero "
                "completions).",
                LOG_PREFIX,
                n,
                direction,
                (
                    getattr(self, "_last_writeback_report").as_log()
                    if hasattr(getattr(self, "_last_writeback_report", None), "as_log")
                    else "no report"
                ),
            )
        # #1066: RE-ADMISSION IS DEFERRED TO AFTER THE CUTOVER. The requeue
        # (and the fresh prefetch `_add_request_to_queue` issues with it)
        # used to run HERE -- before `_cutover_fn` rebinds the HiCache pools
        # -- so every re-admission prefetch was opened at the OUTGOING
        # binding generation and refused as stale on completion
        # (#937/#1025b; measured boot_855_tiprevert1033: 6/6 refusals,
        # cached=0 on 90/90 prefills). The flip behaves like a freshly
        # started server with a cache hit (user design, 2026-09-01): first
        # everything is nulled and rebound, THEN the residents re-enter
        # through the ordinary intake path, whose prefetch lands on the
        # binding that will serve it. `_execute_body` performs the deferred
        # readmit right after `_cutover_fn` via `_post_cutover_readmit`,
        # which carries the W31 retracted==readmitted assertion with it.
        self._pending_seam_readmit = (list(released or ()), n)
        self._seam_readmitted = 0
        # W36 rung 3: the stale-generation gates report per CUTOVER, from here,
        # because this runs on every flip. A gate that is never reached still
        # produces a line reading checked=0, so "clean" and "blind" can never
        # again be byte-identical (W36: eight flips, zero refusal lines, rung
        # inconclusive).
        try:
            from sglang.srt.managers.cache_controller import gate_heartbeat

            cc = getattr(
                getattr(scheduler, "tree_cache", None), "cache_controller", None
            )
            if cc is not None:
                report = gate_heartbeat(cc)
                logger.info(
                    "%s #719 STALE-GATE HEARTBEAT for %s: %s",
                    LOG_PREFIX,
                    direction,
                    report,
                )
                # #861c: THE HEARTBEAT OF THE HEARTBEAT.
                #
                # W36 built this line so "clean" and "blind" could never again
                # be byte-identical. W37-C then logged `checked=0 refused=0` on
                # ALL EIGHTEEN flips and nobody was woken by it -- the line did
                # its job and the absence of a reader undid the job. A gate that
                # can silently disconnect for a third time is not guarded by a
                # line that merely states the disconnection.
                #
                # So a run of consecutive zero-check cutovers is now an ALARM.
                # Two is the threshold rather than one because a single flip on
                # a genuinely idle instance legitimately checks nothing; two in
                # a row means the gate has not been reached across a whole
                # cutover cycle while the seam was busy enough to flip twice.
                #
                # WHAT IT IS NOT: this does not decide WHY. W37-C's cause was
                # downstream (zero decode -> zero write-back -> empty queues ->
                # both counter sites unreachable), i.e. the gate was blind
                # because nothing flowed, not because it was disarmed. The alarm
                # names the observation and points at both possibilities,
                # because a guard that guesses its own cause is how the wrong
                # participant gets accused (#861c's other half).
                try:
                    # #861e FALSE-POSITIVE FIX. The >=2 threshold fired 24
                    # times on a HEALTHY W37-D boot, which devalues the alarm
                    # into noise -- and an alarm nobody trusts is an alarm that
                    # will be scrolled past on the boot that matters.
                    #
                    # The CONDITION was wrong, not just the number. A cutover
                    # legitimately checks nothing whenever no device-tier I/O
                    # was queued under the outgoing binding, and on this
                    # workload that is common: W37-D read checked=0 on 24
                    # pp_to_tp cutovers and checked=2/3/8 on twelve others, all
                    # in one healthy boot. So a zero is only evidence of
                    # BLINDNESS if traffic existed that SHOULD have been
                    # checked.
                    #
                    # Gate on that instead: count a zero-streak only while the
                    # controller reports device-tier work in flight. With no
                    # traffic, zero is the correct reading and is not counted.
                    # Threshold raised to 4 as well: two consecutive quiet
                    # cutovers is an ordinary idle stretch.
                    #
                    # #1205: THAT ARITHMETIC WAS DESCRIBED HERE AND NEVER
                    # IMPLEMENTED. The busy probe read
                    # `bool(write_queue or load_queue or ack_backup_queue)`,
                    # and `ack_backup_queue` is a `queue.Queue`, which defines
                    # neither `__bool__` nor `__len__` -- so with storage
                    # enabled the chain was a constant True and the gate was
                    # byte-for-byte the pre-#861e gate with a bigger threshold.
                    #
                    # And repairing it to an honest DEPTH is not enough on its
                    # own: this runs one statement after `_quiesce_hicache`,
                    # with `hicache_seam_active` refusing new I/O, so the depth
                    # here is ~always 0 and a depth-only gate would be
                    # permanently silent -- the false all-clear that let W37-C
                    # through eighteen flips. The traffic term therefore also
                    # reads the heartbeat's OWN history: a process that has
                    # reached the gate once has proved it reachable, so zeroes
                    # after that are a disconnection, not an idle stretch. See
                    # `stale_gate_zero_streak`.
                    checked, _refused = parse_gate_heartbeat(report)
                    depth = controller_device_queue_depth(cc)
                    if checked:
                        self._stale_gate_ever_checked = True
                    ever = bool(getattr(self, "_stale_gate_ever_checked", False))
                    streak = stale_gate_zero_streak(
                        int(getattr(self, "_stale_gate_zero_streak", 0) or 0),
                        checked=checked,
                        depth=depth,
                        ever_checked=ever,
                    )
                    self._stale_gate_zero_streak = streak
                    if streak >= STALE_GATE_ZERO_STREAK_ALARM:
                        logger.error(
                            "%s #719 STALE-GATE BLIND: checked=0 on %d "
                            "consecutive cutovers (controller queue depth "
                            "here=%s, this rank has reached the gate before: "
                            "%s). The stale-generation gate "
                            "has not been REACHED across a full flip cycle. "
                            "Either device-tier HiCache traffic stopped "
                            "entirely (check Decode batch / write_backup / "
                            "load_to_device counts -- W37-C's cause) or the "
                            "gate was disconnected from its consume points "
                            "(cache_controller.py:301 sits behind `if not "
                            "queue`, :395 runs per queued operation). A gate "
                            "that is never reached protects nothing.",
                            LOG_PREFIX,
                            streak,
                            "UNMEASURED" if depth is None else depth,
                            ever,
                        )
                except Exception:  # noqa: BLE001 - telemetry never breaks a seam
                    pass
        except Exception:  # noqa: BLE001 - telemetry never breaks a seam
            pass
        return released

    def _releasable_cache_bytes(self):
        """#852: injectable like ``_mem_probe`` / ``_mem_reclaim``, so the
        unit tests can model a fragmented cache, a whole one, and a backend
        that cannot say."""
        hook = getattr(self, "_mem_releasable", None)
        if hook is not None:
            try:
                return hook()
            except Exception:  # noqa: BLE001 - an abstaining probe, not a crash
                return None
        return PhaseFlipRuntime._torch_releasable_cache_bytes()

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
        # #846: one monotonic tick per probe pass, so anything remembered
        # below can say how old it is when the census quotes it later.
        self._seam_probe_seq = int(getattr(self, "_seam_probe_seq", 0)) + 1
        self._last_staging_bytes = int(staging_bytes)
        self._last_staging_bytes_seq = self._seam_probe_seq
        driver_free, cached_free = probe()
        self._record_seam_peak(direction, int(staging_bytes), driver_free, cached_free)
        from_driver = max(0, driver_free - reserve)
        if cached_free > 0 and (staging_bytes > from_driver or driver_free < reserve):
            before = driver_free
            cache_promised = int(cached_free)
            # #852: ask what a draw CAN return before paying for one. W24
            # paid an empty_cache() device sync every 60-75 s for 23 straight
            # minutes, each one measuring the 0 the allocator's own counters
            # already knew (~309-324 MiB cached, all of it fragmented inside
            # in-use segments). A measured 0 skips the draw and the census
            # prices the post honestly from the recorded figure; a nonzero
            # or abstaining (None) measurement keeps the draw AND the law-2
            # delivery measurement exactly as #828 wired them.
            releasable = self._releasable_cache_bytes()
            self._last_cache_releasable_bytes = releasable
            self._last_cache_releasable_seq = self._seam_probe_seq
            if releasable == 0:
                logger.info(
                    "%s staging reclaim skipped: %.0f MiB cached but 0 MiB "
                    "releasable (free blocks fragmented inside in-use "
                    "segments) -- an empty_cache() draw would return nothing "
                    "to the driver, so the seam keeps the sync and prices "
                    "the post honestly at zero",
                    LOG_PREFIX,
                    cache_promised / (1024 * 1024),
                )
            else:
                self._reclaim_cached_blocks()
                driver_free, cached_free = probe()
                from_driver = max(0, driver_free - reserve)
            # #828 LAW 2, MEASURED HERE AND SPENT BY THE CENSUS BELOW. This is
            # the only place in the refusal path that observes what the torch
            # cache ACTUALLY handed the driver. Without it the census prices
            # the same post at `memory_reserved() - memory_allocated()` -- a
            # figure that counts fragmented segments `empty_cache()` cannot
            # return -- and boot_827 printed `covered 1870 MiB ... cause=funded`
            # in the same second this line printed `(+0 returned)`.
            # #846: stamped with the pass that measured them. These two are
            # written ONLY here, inside the reclaim branch, and cleared
            # nowhere -- which is why the census below could not tell a
            # figure from this pass from one several passes old.
            #
            # #852: on the SKIPPED path this records a delivery of zero, and
            # that is a fact rather than a fabricated measurement -- zero bytes
            # reached the driver in this pass, and the skip is exactly why. It
            # never reaches the refusal line as "derated to zero" either: a
            # priced-zero promise carries the fragmentation reason instead, and
            # `creditable` returns that reason before the derate text is built.
            self._last_cache_bytes_seq = self._seam_probe_seq
            self._last_cache_promised_bytes = cache_promised
            self._last_cache_delivered_bytes = max(0, int(driver_free - before))
            mib = 1024 * 1024
            # #852 TELEMETRY, AND IT IS THE POINT. W24 could not settle WHY the
            # draw delivered nothing, because 14490 lines carry no allocator
            # -segment figure at all -- `inactive_split`, `fragment` and
            # `segment` appear zero times. The estimate is printed NEXT TO what
            # the draw actually returned, on every pass, so the next window
            # reads the discriminator directly: agreement confirms the
            # fragmentation account, and a nonzero prediction against a zero
            # delivery falsifies it and indicts this estimator instead. An
            # abstaining backend says so rather than printing a fabricated 0.
            predicted = (
                "unmeasurable" if releasable is None else f"{releasable / mib:.0f} MiB"
            )
            # #852 R3: NAME THE FOURTH TERM IN THE SAME LINE. W25's prediction
            # was a stable 88 MiB against a zero delivery, and the discriminator
            # that settles it is how much free space sits in CUDA-graph PRIVATE
            # pools -- bytes `empty_cache()` can never return while the graphs
            # live, which the device-global `.all` counters nevertheless count.
            # Printed beside the prediction so the next window reads the
            # attribution directly instead of inferring it, exactly as #852
            # printed the prediction beside the delivery.
            trapped = self._graph_pool_free_bytes()
            trapped_txt = (
                "unmeasurable" if trapped is None else f"{trapped / mib:.0f} MiB"
            )
            logger.info(
                "%s staging reclaim: driver free %.0f -> %.0f MiB "
                "(+%.0f returned, predicted releasable %s, graph-pool trapped "
                "%s), %.0f MiB still cached, reserve %.0f MiB, staging needs "
                "%.0f MiB",
                LOG_PREFIX,
                before / mib,
                driver_free / mib,
                (driver_free - before) / mib,
                predicted,
                trapped_txt,
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
            + self._arming_floor_advice()
            + self._kv_rung_verdict()
        )

    def _arming_floor_advice(self) -> str:
        """#770 Defect A: withdraw the retry advice when it names a forbidden state.

        "the flip is retried when occupancy drops" is sound advice only while
        the watermark is reachable at all. On the shipped constants it is not:
        the gate arms at band floor 819 + seam entry reserve 512 = 1331 MiB and
        wants a further 192 MiB margin, against a corridor band that tops out
        at 1229 MiB. A card filled INSIDE its own acceptance band is 294 MiB
        short by construction, so occupancy dropping far enough to clear the
        watermark means leaving the band from above -- which is precisely the
        state the corridor law calls a failed boot acceptance.

        Telling an operator to wait for that is worse than saying nothing: it
        describes the wait as normal when no amount of waiting can end it, and
        18f measured exactly that (draining the load did not lift the lock).

        Never raises: advice that cannot be computed is simply not given.
        """
        try:
            from sglang.srt.managers import corridor_guard as cg
            from sglang.srt.managers.funding_authority import solve_arming_floor
            from sglang.srt.managers.phase_flip_seam_reserve import (
                DEFAULT_ARMING_MARGIN_MIB,
            )

            # #851 F5: READ WHAT THE ACTUATOR ADOPTED, not the shipped constant.
            # This used to pass `cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB` while the
            # gate arms on `cg.seam_entry_reserve_mib_resolved()`. On a boot
            # with the #826 solver armed the two differ, and the advice then
            # diagnoses a contradiction the boot does not have.
            #
            # MEASURED, W22: log 396/411/412 recorded "#826 arming floor 1037
            # MiB, solver-derived, corridor ceiling 1229 MiB" -- it fits, the
            # gate armed -- and every abandon line still appended "UNSATISFIABLE
            # ARMING FLOOR: the gate would arm at 1331", the shipped-constant
            # figure, 14 times. It sent every reader away from the real defect.
            #
            # The advice itself stays: on the shipped default the contradiction
            # is REAL (819 + 512 + 192 = 1523 against a 1229 ceiling), and
            # withdrawing "wait for occupancy to drop" is the whole point of
            # #770 Defect A. It just has to be true of THIS boot.
            sol = solve_arming_floor(
                cg.corridor_band_floor_mib(),
                cg.corridor_band_ceiling_mib(),
                cg.seam_entry_reserve_mib_resolved(),
                DEFAULT_ARMING_MARGIN_MIB,
            )
            if sol.satisfiable:
                return ""
            return (
                f"RETRACTION OF THE RETRY ADVICE ABOVE -- waiting cannot work "
                f"here: {sol.detail} "
            )
        except Exception:  # noqa: BLE001 - advice must not raise
            return ""

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
        return (
            f"corridor gate refused the seam staging: {verdict.detail}"
            f"{self._funding_post_census(int(ask_bytes))}"
        )

    def _funding_post_census(self, want_bytes: int) -> str:
        """#770: name the posts a refusal considered, or say nothing at all.

        STRICTLY OBSERVATIONAL. It takes no decision, spends nothing and
        returns a suffix for the refusal line. The gate's verdict above is
        already final by the time this runs.

        It exists because ``reclaimed 0 MiB from [nothing]`` is three different
        worlds in one string -- no providers, providers that paid zero, and a
        funder that was never in the ladder's list -- and the specimen is the
        third: the same second that printed ``[nothing]`` also printed
        ``KV capacity is the funder`` with an exact draw. A reader could not
        tell those apart, and on 2026-08-16 that cost a morning.

        Like ``_staging_budget_census`` above, it must never raise: a refusal
        that cannot explain itself is bad, and a refusal that CRASHES while
        trying to is worse.
        """
        try:
            from sglang.srt.managers.funding_authority import (
                authority_from_seam_snapshot,
            )
            from sglang.srt.managers.phase_flip_spill import (
                KV_BACKING_RELIEF_ATTR as _RUNG_ATTR,
            )

            cached = 0
            try:
                if torch.cuda.is_available():
                    cached = int(
                        torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                    )
            except Exception:  # noqa: BLE001
                pass

            # #828: what the SAME gate pass measured this post actually paying.
            # None when no reclaim was attempted in this pass, which law 2
            # reads as "unobserved -- trust it once", so a refusal that never
            # tried the cache is priced exactly as it was before.
            #
            # #846 CORRECTION, and the sentence above is kept rather than
            # edited so a reader who relies on it learns that instead of
            # finding it silently gone: THE "None" CASE IS UNREACHABLE AFTER
            # THE FIRST RECLAIM. Both attributes are written only inside the
            # reclaim branch of `_staging_affordable` and are assigned None
            # nowhere in this module, so from the first reclaim onward a pass
            # that attempts no reclaim reads an EARLIER pass's measurement and
            # prices this post with it. The age is now stated in the line
            # below; the branch is deliberately unchanged, because what this
            # census ASSERTS is #828's law-2 question and not this ticket's.
            delivered = getattr(self, "_last_cache_delivered_bytes", None)
            promised = getattr(self, "_last_cache_promised_bytes", None)
            # #852: the figure that makes the PROMISE honest rather than the
            # verdict, written by the same probe pass as the two above and
            # therefore quoted under the same age. A law connected to nothing
            # is this corpus's signature failure mode, and this is the wire.
            # A value that will not coerce ABSTAINS (None -> #828 pricing)
            # instead of raising into the census's blanket except, which would
            # silence the whole line and lose the funder list too.
            try:
                releasable = getattr(self, "_last_cache_releasable_bytes", None)
                releasable = None if releasable is None else int(releasable)
            except Exception:  # noqa: BLE001 - a measurement may abstain
                releasable = None
            cache_age = seam_probe_reading_age(
                getattr(self, "_seam_probe_seq", None),
                getattr(self, "_last_cache_bytes_seq", None),
            )
            if delivered is not None and promised is not None:
                # Price the post at what it PROMISED when the draw was taken,
                # so the ratio is delivered-over-promised for the SAME probe.
                # Using the post-reclaim `cached` as the denominator would
                # derate against a number the draw never saw.
                cached = int(promised)

            sched = getattr(self, "_census_scheduler", None)
            rung = getattr(sched, _RUNG_ATTR, None) if sched is not None else None
            slack_rows = 0
            row_bytes = 0
            granule_rows = 0
            if rung is not None:
                # THE REAL ACCESSORS, verified against kv_backing_relief.py --
                # `_last_proposal_terms` (:1036, keys `current`/`floor_rows`),
                # `_bytes_per_row` and `_min_release_rows()` (:1375). Guessed
                # attribute names would have been swallowed by the except below
                # and left this census permanently silent, which is the exact
                # failure shape the draft-weights provider comment warns about:
                # inert, and indistinguishable in a log from never being needed.
                terms = getattr(rung, "_last_proposal_terms", None)
                if terms:
                    slack_rows = max(
                        0, int(terms["current"]) - int(terms["floor_rows"])
                    )
                row_bytes = int(getattr(rung, "_bytes_per_row", 0) or 0)
                try:
                    granule_rows = int(rung._min_release_rows())
                except Exception:  # noqa: BLE001
                    granule_rows = 0

            auth = authority_from_seam_snapshot(
                allocator_cache_bytes=cached,
                allocator_cache_delivered_bytes=delivered,
                allocator_cache_releasable_bytes=releasable,
                kv_slack_rows=slack_rows,
                row_bytes=row_bytes,
                kv_granule_rows=granule_rows,
                rank=int(getattr(self, "_rank", 0) or 0),
            )
            v = auth.can_fund(int(want_bytes))
            # #846: the verdict and the AGE of the reclaim figures it was
            # priced from, in one line. A funding verdict quoting a cache
            # measurement from several passes ago is not wrong to print -- it
            # is wrong to print WITHOUT SAYING SO.
            return (
                f". #770 FUNDING POSTS: {v.describe()} "
                f"[reclaim figures {seam_probe_age_phrase(cache_age)}]"
            )
        except Exception:  # noqa: BLE001 - a census must not raise
            return ""

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

    def _row_bounds_detail(self, tr: PhaseFlipTransition) -> List[str]:
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
                f"backing drain as their requests finish. "
                f"READ {min_backed} AS A DEFECT REPORT, NOT A CAPACITY VERDICT "
                f"(#833): no rank 'binds' this pool. An id above the group "
                f"floor can only exist because some rank EXPOSED one, and the "
                f"exposure clamp is supposed to hold every rank's id space at "
                f"the group floor precisely so this set cannot be built. If "
                f"this line repeats while ids above {min_backed} keep being "
                f"issued, the defect is in exposure, upstream of the seam -- "
                f"look for the '#833 group exposure floor' line and check "
                f"whether the floor ever reached the clamp on every rank"
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

    def record_host_ram_defer(self, direction: str, host_detail: str) -> int:
        """#830 F6: COUNT THE #721 DEFER, AT WARNING, WITH A STABLE TOKEN.

        ANALYSE_830 open item O1 is "#721 defer frequency: UNMEASURED. No log
        line counted." That is not because the path is silent -- it is because
        what it emitted could not be COUNTED. The ``HOST HEADROOM`` line at the
        call site fires on every flip whether the verdict allows or defers, so
        its count is the flip count and says nothing; the defer's own text was
        folded into that same info-level line and carried no running total. A
        strand reading a boot log could not answer "how often did this fire"
        without parsing prose out of a per-flip line.

        WHY THIS ONE DESERVES A COUNTER. A defer appends to ``too_small``, so
        the rank votes the flip unfit and its ARMED WINDOW ENDS EARLY AND
        RANK-LOCALLY. That is the shape that diverges resume slots between
        peers -- one rank abandoning for a reason its peers never observed,
        because host RAM is measured per rank. The lifetime total is what turns
        "it happened" into a rate, and the rate is what decides whether #721's
        own author was right to call this floor unfitted ("I have NO measured
        projected host transient for the flip").

        Returns the new lifetime total, so a caller or test can assert on the
        count rather than on the string.
        """
        self._host_ram_defer_total = int(getattr(self, "_host_ram_defer_total", 0)) + 1
        logger.warning(
            "%s [#721] HOST-RAM DEFER %s: defer %d of %d in this abandon run, "
            "lifetime=%d. "
            "This rank votes the flip unfit, so its armed window ends early "
            "and RANK-LOCALLY -- peers that saw ample host RAM do not abandon "
            "for this reason. %s",
            LOG_PREFIX,
            direction,
            int(getattr(self, "_host_ram_defers", 0)),
            FLIP_HOST_RAM_MAX_DEFERS,
            self._host_ram_defer_total,
            host_detail,
        )
        return self._host_ram_defer_total

    def record_host_ram_escalation(self, direction: str, host_detail: str) -> int:
        """#830 F6: the defer path's OTHER exit, counted SEPARATELY.

        A defer retries; an escalation PROCEEDS ANYWAY under a floor the guard
        just failed. Those are opposite outcomes -- one costs a flip, the other
        accepts the hazard #721 exists to avoid -- and a single merged counter
        would hide which one a boot actually took.
        """
        self._host_ram_escalation_total = (
            int(getattr(self, "_host_ram_escalation_total", 0)) + 1
        )
        logger.warning(
            "%s [#721] HOST-RAM ESCALATION %s: lifetime=%d. %s",
            LOG_PREFIX,
            direction,
            self._host_ram_escalation_total,
            host_detail,
        )
        return self._host_ram_escalation_total

    def _execute(self) -> Optional[dict]:
        direction = self._pending
        assert direction is not None
        # Seam execution-surface instrument (diagnostic, #855 diag boot):
        # brackets the WHOLE cutover attempt -- including a collective abandon
        # that returns before any bytes move -- in a `cutover:<direction>`
        # coverage context, restored to `serving` in the `finally` on every
        # exit path (return OR an exception this function does not itself
        # catch). No-op end to end unless SGLANG_SEAM_COVERAGE_DIR is set; see
        # sglang.srt.managers.seam_coverage.
        seam_coverage.enter_cutover(direction, rank=self._rank)
        try:
            return self._execute_body(direction)
        finally:
            seam_coverage.exit_cutover(direction, rank=self._rank)

    def _post_cutover_readmit(self, direction: str) -> None:
        """#1066: re-admit the cutover's retractions on the INCOMING binding.

        The requeue used to run inside `_release_residents_for_cutover`,
        BEFORE the cutover rebound the HiCache pools, so the intake prefetch
        `_add_request_to_queue` issues was opened at the OUTGOING binding
        generation and completed into a #937 stale refusal every time
        (boot_855_tiprevert1033: 6/6 refused, cached=0 on 90/90 prefills; the
        one voted re-fetch then #841-declined against recompute-built
        unbacked nodes). Deferred to here -- after `_cutover_fn` -- the flip
        behaves like a freshly started server with a cache hit (user design):
        everything nulled and rebound first, then the residents re-enter
        through the ordinary intake path.

        (#1070: the stale-op sweep that preceded the readmit here is deleted
        -- it counted a constructively empty set; see the comment at its old
        site below. The #943b vote's real replacement is the readmit's own
        post-rebind intake prefetch plus the #937 refusal as safety net.)

        HONEST RESIDUAL: between the retraction (in `_release_residents_for_
        cutover`) and this call the retracted requests are owned by nobody.
        An abort inside that window strands them -- the pre-#1066 code had the
        same exposure for the window up to its earlier requeue point, and an
        abort there is a failed flip (KvReshardError) that takes the instance
        down regardless. Named, not hidden.
        """
        # #1047: ONE CENSUS PER CUTOVER (producer_phase_census section 5).
        # Ended HERE, at the last seam step before this cutover's readmit wave
        # begins, so the census that follows describes THIS cutover's wave and
        # its `worst` is that cutover's worst request. A breach has already
        # forced its line out before this point (`emit_double_prefill` never
        # samples a breach away), so ending the census cannot swallow one.
        try:
            from sglang.srt.mem_cache.producer_phase_census import (
                reset_double_prefill_census as _reset_dpc_1047,
            )

            _reset_dpc_1047()
        except Exception as exc:  # noqa: BLE001 - an instrument may never break a flip
            logger.warning(
                "%s #1047 double-prefill census could not be ended (%s); the "
                "next wave's `worst` may include the previous cutover's.",
                LOG_PREFIX,
                exc,
            )
        stash = getattr(self, "_pending_seam_readmit", None)
        self._pending_seam_readmit = None
        scheduler = getattr(self, "_census_scheduler", None)
        if scheduler is None:
            if stash and stash[0]:
                logger.error(
                    "%s #1066: no scheduler bound at the seam; %d retracted "
                    "request(s) cannot be re-admitted (W31 shape).",
                    LOG_PREFIX,
                    len(stash[0]),
                )
            return
        # #1068 (spec 4.3, G3/G4): THE WHOLE RUN-WILLING POPULATION IS
        # RE-ISSUED HERE -- the retracted residents AND the queue occupants.
        # The #1070 premise that queued requests 'never straddle a cutover'
        # is refuted on Boot 2 (boot_855_968umbauB_228a66db32): occupant
        # d188185a was queued at 21:16:57 and recomputed 0->14921 (log
        # 153596). `_reset_full` drops EVERY prefetch record
        # (ongoing_prefetch={}, cache_controller.reset(), mem_pool_host.clear())
        # regardless of who owned it, so an occupant's intake prefetch dies
        # with the residents' and must be re-issued under the incoming
        # binding generation through the ONE intake site -- which is why the
        # readmit runs even when nothing was retracted (released is empty).
        # No second issue site, no stale-op sweep (upstream-minimal): the
        # scheduler rebuilds the queue in kv_arrival_seq order and every
        # member passes _add_request_to_queue -> _prefetch_kvcache once.
        released, n = (stash[0], stash[1]) if stash else ([], 0)
        readmitted = 0
        summary: dict = {}
        try:
            readmitted = scheduler.readmit_seam_residents(
                list(released), requeue_waiting=True
            )
            summary = dict(getattr(scheduler, "last_seam_readmit", None) or {})
        except Exception:  # noqa: BLE001 - never strand a committed flip
            logger.error(
                "%s #856/W31: RE-ADMISSION FAILED for %s after %d "
                "request(s) were already retracted. Those requests are "
                "now owned by nobody -- the W31 defect happening live; "
                "logged rather than raised because the flip is committed.",
                LOG_PREFIX,
                direction,
                n,
                exc_info=True,
            )
        if readmitted != n:
            # RETRACTED MUST EQUAL READMITTED. Anything else means requests
            # were dropped, and dropping them silently is the whole W31
            # defect. Loud and greppable.
            logger.error(
                "%s #856/W31 RE-ADMISSION MISMATCH for %s: retracted %d but "
                "re-admitted %d. The difference is requests owned by nobody.",
                LOG_PREFIX,
                direction,
                n,
                readmitted,
            )
        self._seam_readmitted = readmitted
        # L7 (G11): ONE aggregate line per cutover with the verdict histogram,
        # so the acceptance can read issued/declined per wave without
        # counting #969C lines by hand.
        try:
            from sglang.srt.mem_cache.hicache_phase_binding import (
                current_generation as _current_generation,
            )

            generation = int(_current_generation())
        except Exception:  # noqa: BLE001 - no binding bound in unit tests
            generation = -1
        pool_id, pool_rows = -1, -1
        try:
            _pool = scheduler.tree_cache.cache_controller.mem_pool_host
            _pool = getattr(getattr(_pool, "anchor_entry", None), "host_pool", None) or _pool
            pool_id = id(_pool)
            pool_rows = int(getattr(_pool, "size", -1))
        except Exception:  # noqa: BLE001 - a diagnostic may never break a flip
            pass
        verdicts = dict(summary.get("verdicts") or {})
        issued = int(verdicts.get("issued", 0))
        declined = sum(int(v) for k, v in verdicts.items() if k != "issued")
        reasons = dict(sorted((k, v) for k, v in verdicts.items() if k != "issued"))
        logger.info(
            "%s #1066 POST-CUTOVER FRESH-FETCH after %s: re-admitted %d/%d "
            "resident(s) + re-issued %d/%d queue occupant(s) on the incoming "
            "binding generation=%d pool_id=%d pool_rows=%d issued=%d "
            "declined=%d reasons=%s dropped_by_queue_limit=%d",
            LOG_PREFIX,
            direction,
            readmitted,
            n,
            int(summary.get("requeued", 0)),
            int(summary.get("occupants", 0)),
            generation,
            pool_id,
            pool_rows,
            issued,
            declined,
            reasons,
            int(summary.get("dropped_by_queue_limit", 0)),
        )

    def _execute_body(self, direction: str) -> Optional[dict]:
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
        # #1028: set when the fence came back with backups still in flight. It
        # becomes a TERM IN THE UNANIMOUS VERDICT below rather than a log line,
        # for the reason the row-bounds paragraph gives two screens down:
        # nothing has been mutated at this point, so the safe answer to "this
        # flip cannot honour its own contract" is that every rank abandons it
        # and keeps serving. See the objection's own comment for the measurement.
        writeback_detail: Optional[str] = None
        try:
            from sglang.srt.mem_cache.hicache_flip_writeback import (
                maybe_flip_writeback,
            )

            report = maybe_flip_writeback(getattr(self, "_census_scheduler", None))
            if report:
                seam_census.mark("flip_writeback")
                if not report.complete:
                    writeback_detail = (
                        f"#1028 writeback fence incomplete: {report.outstanding} "
                        f"backup(s) still in flight ({report.as_log()}). The "
                        f"cutover would retract these residents and drop the "
                        f"prefix tree while claiming their KV is in the "
                        f"canonical store, and the re-admission would then find "
                        f"host_hit=0 and RECOMPUTE the whole prompt"
                    )
                # #856: THE FENCE IS HALF THE NEW VALIDATION METRIC. Once the
                # flip carries no KV, cutover-blocking time is fence + weights
                # refill, and the fence's cost has until now been visible only
                # as a census SEGMENT (`flip_writeback->hicache_quiesce`, 74.8
                # ms in W25) -- a delta between two marks, which nothing that
                # reads `last_stats` can see. The report already carries its
                # own elapsed time; this only stops throwing it away.
                self._last_writeback_report = report
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
        staging_bytes = self._seam_reserve_bytes(tr, direction, src, dst, waves)
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
            staging_bytes = self._seam_reserve_bytes(tr, direction, src, dst, waves)
        # BOUNDS ON THE FINAL PLAN. Computed after the agreement, so the rows
        # the group actually intends to move are the rows the pools are
        # checked against.
        too_small = self._row_bounds_detail(tr)
        if slot_detail:
            too_small.append(slot_detail)
        if corridor_detail:
            too_small.append(corridor_detail)
        affordable, staging_detail = self._staging_affordable(staging_bytes, direction)
        if not affordable:
            too_small.append(staging_detail)

        # #1028: AN EXPIRED FENCE MAY NOT CONVERT INTO A FALSE PROMISE.
        #
        # MEASURED, boot_855_wt1016 19:22:45: the fence reported `acked=1
        # outstanding=3 elapsed=2.000s/2.000s` and the cutover proceeded to log
        # "RESIDENTS RELEASED ... Their KV is in the canonical store from the
        # fence" one second later. It was not. `#969B READMIT-MATCH
        # rid=a5517198 prefix_len=0 host_hit=0 storage_hit=0 input_len=13180`
        # is the receipt: all 13180 tokens were recomputed, the recompute did
        # not fit the one-chunk TP grant, and the decode phase armed away from
        # it mid-recompute -- four flips and 157.7 s of wall for 16 tokens.
        # This is the falsifier #856 named for itself in WINDOW-QUEUE.md: "a
        # flip that completes but leaves requests re-prefilling uncached means
        # the fence is not covering what read-through needs."
        #
        # It joins the existing unanimous abandon rather than becoming a check
        # of its own -- same argument as the staging term above: the answer
        # when the flip cannot be honoured is group-agreed and rank-uniform,
        # and a rank-local abandon would half-flip the group.
        #
        # BOUNDED, because a permanently stuck storage backend must not mean
        # "never flip again" -- that trades a recompute for a wedge, which is
        # the worse of the two and the failure class this campaign keeps
        # paying for. After `_WRITEBACK_DEFER_LIMIT` consecutive defers the
        # flip proceeds and the recompute is ACCEPTED OUT LOUD, which is the
        # honest version of what the code did silently before.
        # #1203 (family A5): THE BUDGET IS SPENT IN THE GROUP'S CURRENCY.
        # The bound below used to count on a RANK-LOCAL counter while the thing
        # it bounds -- the abandon -- is group-unanimous: the flip is abandoned
        # if ANY rank objects, and each rank resets its own counter the moment
        # its own objection clears. Three ranks taking turns objecting then
        # never spend a budget between them and the direction defers for ever,
        # which is the 411-abandon decode wedge reached through the mechanism
        # that exists to prevent it. `_seam_abandons_in_a_row` is booked from
        # the already-reduced fit verdict, so every rank reads the same number;
        # the seam-margin term twelve hundred lines up has been spending that
        # currency since it was written, and this is the same argument applied
        # to the three siblings it was never copied to. The local counter stays
        # as the per-rank instrument it always was -- it is no longer a bound.
        if writeback_detail is not None:
            _wb_defers = int(getattr(self, "_writeback_defers", 0) or 0)
            if _wb_defers < _WRITEBACK_DEFER_LIMIT:
                self._writeback_defers = flip_defer_budget_after(
                    objected=True, escalated=False, prior=_wb_defers
                )
                too_small.append(
                    f"{writeback_detail} -- deferring this flip "
                    f"({self._writeback_defers}/{_WRITEBACK_DEFER_LIMIT}); the "
                    f"residents stay resident and decodable (#1011)"
                )
            else:
                logger.error(
                    "%s #1028 WRITEBACK DEFER LIMIT reached (%d defers in this "
                    "abandon run): "
                    "proceeding with the flip although the fence is incomplete. "
                    "The affected prefixes WILL miss and their requests WILL "
                    "recompute in full -- accepted here deliberately, because a "
                    "flip deferred without end is a wedge and a wedge is worse "
                    "than a recompute. %s",
                    LOG_PREFIX,
                    _wb_defers,
                    writeback_detail,
                )
                self._writeback_defers = flip_defer_budget_after(
                    objected=False, escalated=True, prior=_wb_defers
                )
                # #1068 (G9): the proceed is a LOSS TERM of the wave this
                # cutover re-admits, carried on the #939 census line
                # (`fence_proceeds`, L10) so a `within_bound=false` can be
                # told apart from a store defect. Seeded into the NEXT census
                # (the reset in `_post_cutover_readmit` ends the previous one
                # after this point).
                try:
                    from sglang.srt.mem_cache.producer_phase_census import (
                        note_fence_proceed as _note_fence_proceed,
                    )

                    _note_fence_proceed()
                except Exception:  # noqa: BLE001 - an instrument may never break a flip
                    logger.warning(
                        "%s #1068 the fence proceed could not be noted in the "
                        "#939 census",
                        LOG_PREFIX,
                        exc_info=True,
                    )
        else:
            # NOTHING TO FENCE THIS ROUND IS NOT A CLEARED OBJECTION. The
            # budget is untouched here for the same reason the gate no longer
            # refunds it on a cleared vote: the abandon is group-unanimous.
            self._writeback_defers = flip_defer_budget_after(
                objected=False,
                escalated=False,
                prior=int(getattr(self, "_writeback_defers", 0) or 0),
            )

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
            # #1203 (family A5), REPAIRED: THIS GUARD'S OWN defer count, not
            # the group's abandon book. The verdict below does not merely stop
            # deferring at its limit, it ESCALATES -- proceeds under a failed
            # host floor -- so a budget spendable by corridor, staging and
            # frame-divergence abandons would let the first genuine shortfall
            # through having deferred zero times. The count no longer refunds
            # itself when this rank's own reading clears, which is what A5
            # reached for the group's currency to fix; see
            # `flip_defer_budget_after`.
            _host_defers = int(getattr(self, "_host_ram_defers", 0) or 0)
            allow_host, escalated, host_detail = flip_host_headroom_verdict(
                _host_avail, 0, _host_defers
            )
            logger.info("%s HOST HEADROOM %s: %s", LOG_PREFIX, direction, host_detail)
            # Booked BEFORE the instruments, because `record_host_ram_defer`
            # prints this counter and a probe that reports the pre-verdict
            # number while claiming the post-verdict one is the #1205 defect.
            self._host_ram_defers = flip_defer_budget_after(
                objected=not allow_host, escalated=escalated, prior=_host_defers
            )
            if not allow_host:
                # #830 F6: counted, at warning, with a stable token -- O1.
                self.record_host_ram_defer(direction, host_detail)
                too_small.append(host_detail)
            elif escalated:
                self.record_host_ram_escalation(direction, host_detail)
        except Exception as exc:  # noqa: BLE001 - a guard must not break a flip
            logger.warning(
                "%s host-headroom guard could not run (%r); the flip proceeds "
                "unguarded rather than being refused on an unknown.",
                LOG_PREFIX,
                exc,
            )

        # #834 B: AN UNLEVELLED EXPOSURE REFUSES THE FLIP BY NAME.
        #
        # THE HAZARD THIS IS THE ACTUATOR FOR. A deferred grow backs rows that
        # the group has not levelled to. If anything ever exposes them anyway
        # -- a bug here, a peer on a tree without this split, a levelling that
        # DECLINED under #792 and left no ceiling -- then this rank will hand
        # out an id a peer cannot map, and the result is not a slow flip: it is
        # all three ranks dying inside ``store_kvcache``'s bounds assert.
        #
        # SO IT REFUSES, AND REFUSING IS THE POINT. The alternative shapes are
        # both worse and both have been paid for in this tree: exposing anyway
        # is the three-rank abort, and blocking until the group agrees is a
        # collective entered at a local cadence, which is the boots 9/10 wedge.
        # A refused flip is a lost flip and nothing more -- the same trade
        # ``collective_cap_target`` and the #792 decline already make -- and it
        # votes through ``too_small`` so the whole GROUP declines together
        # rather than this rank declining alone.
        refusal = self._unlevelled_exposure_refusal()
        if refusal is not None:
            logger.error("%s [#834] %s", LOG_PREFIX, refusal)
            too_small.append(refusal)

        # #830 F4: THE SEAM WINDOW GETS A NAMED CEILING.
        #
        # Placed beside #721's host-RAM guard because it is the same kind of
        # object -- an arm-time verdict that votes through ``too_small``,
        # bounded, escalating rather than holding forever -- and because both
        # answer "should this flip enter its no-return window right now".
        #
        # It projects from the LAST measured drain (see flip_seam_budget_verdict
        # for why that is the only honest projector, and for what it therefore
        # cannot catch). Before any flip has measured one, the projection is
        # None and the guard stands down: an unknown must never produce a
        # refusal, which is #721's rule and it applies unchanged here.
        try:
            projected_drain = getattr(self, "_seam_drain_ms", None)
            # #1203 (family A5), REPAIRED: this guard's own defer count -- same
            # argument as the host-RAM bound above, same escalating verdict.
            _sb_defers = int(getattr(self, "_seam_budget_defers", 0) or 0)
            allow_seam, seam_escalated, seam_detail = flip_seam_budget_verdict(
                projected_drain, _sb_defers
            )
            logger.info("%s SEAM BUDGET %s: %s", LOG_PREFIX, direction, seam_detail)
            if not allow_seam:
                self._seam_budget_refusals = (
                    int(getattr(self, "_seam_budget_refusals", 0)) + 1
                )
                logger.warning(
                    "%s [#830] %s %s: defer %d of %d in this abandon run, "
                    "lifetime=%d. %s",
                    LOG_PREFIX,
                    SEAM_BUDGET_REFUSED,
                    direction,
                    _sb_defers + 1,
                    FLIP_SEAM_BUDGET_MAX_DEFERS,
                    self._seam_budget_refusals,
                    seam_detail,
                )
                too_small.append(seam_detail)
            elif seam_escalated:
                self._seam_budget_escalations = (
                    int(getattr(self, "_seam_budget_escalations", 0)) + 1
                )
                logger.warning(
                    "%s [#830] %s ESCALATION %s: lifetime=%d. %s",
                    LOG_PREFIX,
                    SEAM_BUDGET_REFUSED,
                    direction,
                    self._seam_budget_escalations,
                    seam_detail,
                )
            self._seam_budget_defers = flip_defer_budget_after(
                objected=not allow_seam, escalated=seam_escalated, prior=_sb_defers
            )
        except Exception as exc:  # noqa: BLE001 - a guard must not break a flip
            logger.warning(
                "%s seam-budget guard could not run (%r); the flip proceeds "
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
            # #834 A: nothing is pending any more, so nothing may keep the
            # device tier disarmed. Released HERE rather than left to the
            # round hook's insurance so the tier comes back in the same
            # step the flip ends in, not one round later.
            release_prearm_quiesce(self, "nothing pending")
            self._parked_extent = None  # #746: cleared on EVERY exit
            self._armed_residents = {}  # #1202: and so is the ledger
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
            # #814: AND THE CAP MUST NOT SURVIVE THE ABANDON. Recovery is what
            # lifts KvRowCap, and its only other call sites are post-cutover
            # hooks -- so a cap that helps REFUSE the flip would keep itself
            # alive for the life of the process (measured: pool parked at 26.8%
            # of its id space, six refused returns, not one post-cutover
            # census). Safe precisely here and nowhere earlier: nothing moved,
            # no seam is owed the memory, and this exit is reached by the whole
            # group together -- it is downstream of the bit-identical
            # ``reduced_fit`` MIN above, with no return and no raise in
            # between, which is what a collective needs. The grow inside stays
            # rank-local and corridor-bounded; see recover_kv_backing_on_abandon.
            from sglang.srt.managers.phase_flip_spill import (
                recover_kv_backing_on_abandon,
            )

            recover_kv_backing_on_abandon(
                self._census_scheduler,
                self._collective_min,
                direction=direction,
                why="seam fit refused",
            )
            return None

        # The group is going through, so this direction's delay budget is
        # whole again. Reset here rather than in the gate: the gate is
        # rank-local and a rank that cleared while a peer did not has learnt
        # nothing about the group.
        if not hasattr(self, "_seam_abandons_in_a_row"):
            self._seam_abandons_in_a_row = {}
        self._seam_abandons_in_a_row[direction] = 0
        # #1203 A5 (repaired): AND SO ARE THE THREE GUARDS' OWN BUDGETS, and
        # this is the ONLY place they are refunded short of their own
        # escalation. The sentence above is the whole argument -- it was
        # written for the abandon book and applies word for word to the
        # writeback, host-RAM and seam-budget counters, which is why they now
        # live beside it instead of refunding themselves in the gate.
        self._writeback_defers = 0
        self._host_ram_defers = 0
        self._seam_budget_defers = 0
        # #485: and so is the backoff. A seam that went through has proved the
        # demand fundable, so the next refusal starts its damping from zero
        # rather than inheriting a streak the group has already broken.
        if not hasattr(self, "_seam_retry_at_arm"):
            self._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
            self.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
        self._seam_retry_at_arm[direction] = 0

        # #760: THE NO-RETURN POINT IS WHERE THE SEAM BEGINS for HiCache.
        # From here on, bytes move and the outgoing phase's backing is on its
        # way out. Two things must both hold before the first wave:
        #
        #   1. no NEW device-tier HiCache I/O -- the guard refuses it while
        #      ``hicache_seam_active`` is up (this thread is the only producer,
        #      so the flag is a statement, not a race);
        #   2. no OLD device-tier HiCache I/O still in flight -- copies
        #      enqueued legitimately before the seam ride the controller's
        #      private streams and outlive their Python call by seconds under
        #      load. Both #760 crash specimens are exactly such a copy
        #      reaching pool memory the seam had released, 3 s after the
        #      cutover. So the streams are DRAINED here, while every pointer
        #      they hold still names live memory -- finishing them is correct
        #      and bounded; dropping them would only trade durable cache
        #      entries for nothing.
        #
        # Order matters against the #703 writeback hook above: that hook
        # ENQUEUES device->host staging copies (that is its job, and this
        # phase is the one its bindings belong to), so the seam must arm
        # after it and the drain must cover it.
        self.hicache_seam_active = True
        # #830 F1: the drain's cost is RECORDED, not discarded. It lands in
        # last_stats["drain_ms"] and in its own prefixed line, so "is the drain
        # the seam's cost?" is answered by a grep instead of by argument.
        self._seam_drain_ms = self._quiesce_hicache(direction)
        seam_census.mark("hicache_quiesce")

        # #856: THE FLIP CARRIES NO KV, AND THIS IS WHERE THAT BECOMES TRUE.
        #
        # The fence above has persisted the tree's prefixes to the canonical
        # store and the quiesce has drained the staging copies, so every row
        # worth keeping is durable. Retracting now releases each resident
        # request's rows AND its tree lock ref, which is what makes the
        # following reset safe (#825 crashed doing the reset with those locks
        # still held). See `release_residents_for_cutover`.
        self._release_residents_for_cutover(direction)
        seam_census.mark("resident_release")

        # THE PLAN IS REBUILT ON THE NOW-EMPTY LIVE SET. This is the retirement
        # of the KV mover, performed by making its input empty rather than by
        # deleting a wave loop whose extent bookkeeping (finalize_wave, span
        # release, id-space retirement) still has to run. Every downstream
        # figure -- total_slots, send/recv rows, the staged bytes -- follows
        # from `tr`, so an empty `tr` is a flip that provably moves nothing.
        tr = build_phase_flip_transition(
            torch.empty(0, dtype=torch.int64),
            self._map,
            self._n_layers,
            self._vec,
            self._rank,
            direction,
        )

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

        # #856/W28: A PLAN THAT MOVES NOTHING NEEDS NO WAVES.
        #
        # The seam retracts every resident and drops the tree before this
        # point, so under the no-KV design the plan is EMPTY by construction
        # and every wave below packs nothing, exchanges nothing and writes
        # nothing. W27-retry measured the residue: 16 empty waves for ~314 ms,
        # pure backing churn.
        #
        # WHAT THE WAVES STILL DID, and why this is not simply deleting the
        # loop: each one released a slice of the source pool's backing and
        # restored the matching slice of the destination's, and
        # ``finalize_wave`` is what marks the destination RESIDENT again.
        # Skipping the loop without doing that leaves the destination pool
        # answering NO to ``backing_is_resident`` -- the invariant named when
        # this successor was deferred.
        #
        # So the replacement is the whole-pool swap that already exists on the
        # same object: release the source, reclaim, restore the destination.
        # It is not merely equivalent here, it is STRICTLY CHEAPER. Waving
        # exists to bound the transient of holding a source layer live while
        # its destination layer is written; with no bytes crossing there is
        # nothing to bracket, and ``__call__`` releases the source BEFORE
        # restoring the destination, so its peak is max(src, dst) where the
        # wave loop's is a wave's worth above the resting layout.
        if swap is not None and tr.moves_nothing:
            swap(direction)
            seam_census.mark("wave_loop_skipped")
            waves = ()

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
        # #1204: named, recorded, never swallowed -- see run_pre_cutover_movers.
        run_pre_cutover_movers(
            self._pre_cutover_fns,
            direction,
            seam_census.mark,
            note_failure=lambda label, exc: setattr(
                self, "_seam_mover_failed", (direction, label, repr(exc))
            ),
        )
        movers_ms = (self._clock() - t_movers0) * 1000.0
        t_cutover0 = self._clock()
        self._reconcile_trees_if_diverged(direction)
        self._cutover_fn(direction)
        # #822: THE commit instant, so THE retirement instant. Placed here and
        # not beside `self._epoch += 1` below on purpose: the post-cutover
        # census on the next line must already run under the new id space, or
        # a surviving pre-cutover claim reads as an out-of-range row instead of
        # as the #796 shape it is.
        # Guarded including the attribute lookup, for the reason spelled out at
        # the census call site -- and more so here, because this is inside the
        # no-return region: nothing this instrument does may raise into it.
        try:
            retire = getattr(self, "_retire_row_id_space", None)
            if retire is not None:
                retire(direction)
        except Exception:  # noqa: BLE001 -- an instrument, never a gate
            logger.debug("%s id-space retirement skipped", LOG_PREFIX, exc_info=True)
        # #851 F1: SEAM EVENT ONE OF TWO -- the cutover. `retire(direction)`
        # above is the AUDIT half: it stamps a new epoch and reports. This is
        # the enforcement half that #822 named as its own open item ("does not
        # yet REFUSE such an id at the allocator"). Deliberately OUTSIDE the
        # try above -- a swallowed audit must not also swallow the gate, or
        # enforcement inherits the exact silence it exists to end.
        self._enforce_exposure_at_seam(f"{direction} cutover")
        seam_census.mark("cutover")
        # #1066: deferred seam re-admission -- see `_post_cutover_readmit`.
        # Runs after the stacks are swapped and the HiCache pools rebound, so
        # the requeue's intake prefetch opens on the binding that serves it.
        try:
            self._post_cutover_readmit(direction)
        except Exception:  # noqa: BLE001 - never strand a committed flip
            logger.error(
                "%s #1066 post-cutover readmit FAILED after %s (W31 shape)",
                LOG_PREFIX,
                direction,
                exc_info=True,
            )
        seam_census.mark("post_cutover_readmit")
        # #856: the warm-up ledger's clock. Everything served from here until
        # the next cutover is warming the cache back up, which is the price
        # this design pays for carrying no KV. Counting the cutovers is the
        # half that lives in this process; feeding it completed-request
        # latencies is the half that does not (see `warmup_latency`).
        try:
            self.warmup_ledger.note_cutover()
        except Exception:  # noqa: BLE001 - an instrument, never a gate
            pass
        self._pool_census("post-cutover", direction)
        cutover_ms = (self._clock() - t_cutover0) * 1000.0
        self._phase = _PHASE_AFTER[direction]
        # #760: the seam is over -- the phase above is now the truth the
        # guard's authority reports. Cleared AFTER the phase update so the
        # guard never sees a gap in which the seam is down but the phase is
        # still the outgoing one. (The caller's finally is the insurance for
        # a raise anywhere above; this is the ordinary exit.)
        self.hicache_seam_active = False
        self._pending = None
        self._armed_at = None
        # #834 A: nothing is pending any more, so nothing may keep the
        # device tier disarmed. Released HERE rather than left to the
        # round hook's insurance so the tier comes back in the same
        # step the flip ends in, not one round later.
        release_prearm_quiesce(self, "nothing pending")
        # #746: the commit is an exit too -- the packed rows are moved, the
        # extent has no referent, and a surviving snapshot would pin the
        # rung into the next phase for ever (the M5 failure mode).
        self._parked_extent = None
        # #1202: the ledger's referents are retracted and their rows are
        # back in the outgoing pool, so every id in it now names a row the
        # next phase may legally re-mint. Holding it past the commit is
        # the M5 failure mode on the request axis.
        self._armed_residents = {}
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
            # #830 F1: the #760 device-tier drain, which is INSIDE movers_ms.
            # Kept out of the DONE line's parenthesised list on purpose --
            # ANALYSE_830 section 10's reproduction regex pins that list ending
            # at "cutover N ms)", and silently breaking the analysis's own
            # documented grep is the exact failure this ticket is repairing.
            "drain_ms": float(getattr(self, "_seam_drain_ms", 0.0)),
            # #856: the HiCache fence, for the same reason and by the same
            # rule as `drain_ms` above -- recorded in the stats dict, kept OUT
            # of the DONE line's parenthesised list, because ANALYSE_830
            # section 10's reproduction regex pins that list ending at
            # "cutover N ms)" and silently breaking a documented grep is the
            # failure #830 was repairing.
            #
            # None, never 0.0, when no fence ran: "the fence cost nothing" and
            # "no fence happened" are different facts, and a defaulted zero is
            # the #606 shape this build has removed repeatedly.
            "writeback_fence_ms": _writeback_fence_ms(
                getattr(self, "_last_writeback_report", None)
            ),
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
