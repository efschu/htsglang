# SPDX-License-Identifier: Apache-2.0
"""Boot-time construction of the #631 phase flip's TP decode stack.

The server boots as the PP topology (pp_size ranks, tp=1) through the
normal path. This module then builds, in the SAME process on every rank,
the SECONDARY stack that serves the TP decode phase:

1. install the uneven-TP shard plan and the weighted-DCP token vector
   (both derived from --phase-flip-tp-vector, process-global, inert for
   the already-constructed PP stack whose layers cached tp_size=1);
2. snapshot the PP stack's checkpoint weights to a host image and free
   their device storages (the boot-order enabler: PP originals + TP
   originals + arena would not fit the 5090 together);
3. build a TP-shaped TpModelWorker under the flip group routing
   (``set_phase_flip_tp_active``) plus the ``get_parallel().override``
   geometry scope -- the double mechanism is load-bearing: the contextvar
   feeds construction-time caching (weight sharding), the module routing
   feeds forward-time collectives (``tensor_model_parallel_all_reduce``
   reaches groups through ``get_tp_group()``, NOT the contextvar; without
   the routing the TP stack would all-reduce over the primary tp=1 group,
   a silent no-op);
4. pack the TP weights into the boot-allocated shared arena (fixed
   addresses for process life) and image them to host;
5. allocate the TP stack's pools, backends and decode CUDA graphs while
   the TP bytes are live in the arena (graphs bake the arena addresses --
   pin 2: ONLY this stack captures decode graphs);
6. assert full-attention KV row byte-compatibility between the two
   resident pools (pin 3, also pinned hermetically);
7. rebind the PP parameters to their arena views and refill the arena
   from the PP host image -- the boot phase (PP) is live again.

A flip is then: KV/GDN state move on the #297 envelope + ONE contiguous
arena refill from the other layout's image + cutover (PhaseFlipRuntime).

Everything here runs rank-uniformly at boot, gated on
``--enable-phase-flip``; the default path never imports this module.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.model_executor.rotation_executor import allocate_rotation_image
from sglang.srt.model_executor.weights_arena import (
    ArenaLayout,
    allocate_arena,
    RefillLegTiming,
    refill_bound_phrase,
    bind_arena_views,
    host_image_mode,
    image_from_tensors,
    release_host_image,
    pack_into_arena,
    plan_arena_layout,
    require_two_file_preconditions,
    tag_layout_image,
    two_file_images_enabled,
    two_file_leg,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-BOOT"

# Non-checkpoint device parameters excluded from the arena (they persist
# outside it for process life; DESIGN_631 3.3). Matched as substrings of
# the parameter name, the marlin workspace family being the known member.
NON_CHECKPOINT_NAME_PATTERNS: Tuple[str, ...] = ("workspace",)


class PhaseFlipBootError(KvReshardError):
    """Loud failure of the phase-flip boot family (#631)."""


# ---------------------------------------------------------------------------
# #797 THE CALIBRATION EDGE AT THE TP CUTOVER.
#
# `parse_flip_token_vector` produces the SEED: a pre-boot estimate read from
# --phase-flip-tp-vector or SGLANG_UNEVEN_TOKEN_VECTOR. It is installed
# process-globally (step 1) and the TP stack's worker is constructed under it
# (step 3). During that construction the TP runner reaches the install-capable
# calibration sites (_resolve_memory_pool_config -> _maybe_suggest_dcp_token_
# vector with allow_install=True) and may replace the global vector with the
# MEASURED optimum -- safely, because at that point nothing has snapshotted it
# and the TP pools, allocator, backends and graphs are all built AFTERWARDS,
# i.e. under the measured vector.
#
# The seed was then frozen into PhaseFlipStacks.token_vector regardless, and
# `phase_flip_runtime._cutover` reinstalls THAT at every flip. Two consequences,
# and the second is the serious one:
#
#   1. the measured vector never reached the decode phase -- the flip served
#      the estimate the calibration had already superseded;
#   2. the owner rule was pointed at a DIFFERENT vector than the pools were
#      SIZED under, which is exactly the out-of-bounds slot id that the
#      cutover's own comment (phase_flip_runtime.py, at the set_cp_token_ratios
#      call) warns about -- reached not by reinstalling the weight vector, but
#      by reinstalling a stale token vector.
#
# The fix is to read the vector back after the stack is built instead of
# assuming it. This edge INSTALLS NOTHING: `allow_install` inside the
# calibration stays the one and only authority for when a vector goes live,
# and this code only observes what that authority decided. That is also why
# there is no collective here -- the install decision is already rank-uniform
# (server args plus the all-gathered per-rank capacity), so every rank reads
# back the identical vector from its own process-global state.
#
# Three states, reported apart. A single "None means something changed or
# maybe did not" is the defect class this task is about: the verdict is named
# together with its cause, so a log reader can tell "the seed was confirmed"
# from "nothing could be read back".
# ---------------------------------------------------------------------------

FLIP_VECTOR_HOLDS = "holds"
FLIP_VECTOR_RECALIBRATED = "recalibrated"
FLIP_VECTOR_UNDECIDED = "undecided"


@dataclass(frozen=True)
class FlipTokenVectorVerdict:
    """What the TP decode phase will actually run its owner rule under.

    ``state`` is one of the three FLIP_VECTOR_* constants and ``reason`` always
    names why, including for the two states that change nothing.
    """

    state: str
    vector: Tuple[int, ...]
    seed: Tuple[int, ...]
    installed: Optional[Tuple[int, ...]]
    reason: str


def resolve_effective_flip_token_vector(seed, installed) -> FlipTokenVectorVerdict:
    """Decide which token vector the flip's TP stack carries. Pure.

    ``seed`` is what parse_flip_token_vector produced; ``installed`` is what
    get_cp_token_ratios() reports once the TP stack is built. The pools were
    built under ``installed``, so when the two disagree it is the SEED that is
    stale, never the read-back.

    UNDECIDED keeps the seed deliberately. A missing or wrong-length read-back
    means the process-global state cannot be trusted to describe the pools, and
    substituting a vector of the wrong length would turn an unclear situation
    into a certain out-of-bounds slot id.
    """
    seed_vec = tuple(int(x) for x in seed)
    if installed is None:
        return FlipTokenVectorVerdict(
            state=FLIP_VECTOR_UNDECIDED,
            vector=seed_vec,
            seed=seed_vec,
            installed=None,
            reason=(
                "no token vector is installed after the TP stack build, so "
                "there is no measurement to read back. The seed stands, which "
                "is what every pre-#797 boot did unconditionally."
            ),
        )
    installed_vec = tuple(int(x) for x in installed)
    if len(installed_vec) != len(seed_vec):
        return FlipTokenVectorVerdict(
            state=FLIP_VECTOR_UNDECIDED,
            vector=seed_vec,
            seed=seed_vec,
            installed=installed_vec,
            reason=(
                f"the installed vector {list(installed_vec)} has "
                f"{len(installed_vec)} entries against the seed's "
                f"{len(seed_vec)}. A length disagreement means the installed "
                "vector does not describe these ranks, so it cannot be adopted "
                "as the owner rule; the seed stands."
            ),
        )
    if installed_vec == seed_vec:
        return FlipTokenVectorVerdict(
            state=FLIP_VECTOR_HOLDS,
            vector=seed_vec,
            seed=seed_vec,
            installed=installed_vec,
            reason=(
                f"the calibration left {list(seed_vec)} in place -- either it "
                "measured this vector to be the optimum, or no install-capable "
                "site changed it. Either way the pools were built under the "
                "seed and the decode phase runs under the same vector."
            ),
        )
    return FlipTokenVectorVerdict(
        state=FLIP_VECTOR_RECALIBRATED,
        vector=installed_vec,
        seed=seed_vec,
        installed=installed_vec,
        reason=(
            f"the boot's own measurement superseded the seed {list(seed_vec)} "
            f"with {list(installed_vec)}, and the TP pools were built under "
            "the measured vector. Carrying the seed into the cutover would "
            "point the owner rule at a split the pools do not have."
        ),
    )


def effective_flip_token_vector(server_args, seed) -> FlipTokenVectorVerdict:
    """The token vector the flip's TP stack carries, read back and vouched for.

    Called ONCE, after the TP stack is fully built, so that "the calibration
    installed nothing" is a finished fact rather than a not-yet -- the same
    placement rule as assert_seed_superseded.

    The provenance rule (#797) is applied HERE and not only in
    resolve_cp_token_ratios, because the flip's vector never passes through
    that function: parse_flip_token_vector reads --phase-flip-tp-vector and
    SGLANG_UNEVEN_TOKEN_VECTOR directly, so a token vector traced to a
    retracted investigation reached the decode phase unchecked. Refusing at
    boot is the safe direction; the same check inside the cutover would take
    down a serving instance mid-flip.
    """
    from sglang.srt.distributed.utils import (
        _refuse_retracted_token_vector,
        get_cp_token_ratios,
    )

    verdict = resolve_effective_flip_token_vector(seed, get_cp_token_ratios())
    logger.info(
        "%s #797 token-vector calibration at the TP cutover: %s. %s "
        "(seed %s, read back %s, carried into the decode phase %s)",
        LOG_PREFIX,
        verdict.state.upper(),
        verdict.reason,
        list(verdict.seed),
        None if verdict.installed is None else list(verdict.installed),
        list(verdict.vector),
    )
    if verdict.state != FLIP_VECTOR_RECALIBRATED:
        # The carried vector is the DECLARED one, so its declared lineage is
        # what the register must be asked about. A recalibrated vector is
        # exempt for a substantive reason, not for convenience: it was produced
        # by this boot's own per-rank profiling, which is what
        # PROVENANCE_MEASURED means. Asking the register about it would match
        # it by VALUE against a retracted entry and refuse a measurement for
        # resembling a withdrawn estimate.
        _refuse_retracted_token_vector(
            server_args,
            list(verdict.vector),
            "--phase-flip-tp-vector / SGLANG_UNEVEN_TOKEN_VECTOR "
            "(the phase flip's TP decode stack)",
        )
    return verdict


def parse_flip_vector(server_args) -> List[int]:
    vec = [int(x) for x in server_args.phase_flip_tp_vector.split(",")]
    if len(vec) != server_args.pp_size:
        raise PhaseFlipBootError(
            f"flip vector {vec} has {len(vec)} entries but pp_size is "
            f"{server_args.pp_size}"
        )
    return vec


def parse_flip_token_vector(server_args) -> List[int]:
    """The KV TOKEN split for the TP decode phase.

    Defaults to the flip vector, which is what the V1 one-vector rule
    asked for -- but the two ratios optimise against different resources
    and their optima do not coincide:

      * the weight shard follows COMPUTE, so the 5090 takes the largest
        share (30 of 64);
      * the token split must follow each rank's REMAINING memory once its
        weights are placed, and the rank with the biggest weight shard has
        the LEAST left over.

    Sizing KV with the compute vector therefore makes the most
    compute-loaded rank the binding one, and the allocator's min-reduce
    then drags every other rank down to its unit. Measured on this rig at
    vector 30,17,17 (per-rank profiled capacity 12779 / 68661 / 30517
    tokens):

        rank 0: 12750 tok / 30 = unit  425   <- binds the group
        rank 1: 68646 tok / 17 = unit 4038
        rank 2: 30515 tok / 17 = unit 1795
        -> global max_total_num_tokens 27200, ranks 1 and 2 left idle.

    The token-proportional vector 7,39,18 reaches ~108480 tokens (4.0x)
    out of the same physical memory. The server already computes and logs
    that vector after profiling; this is the lane that lets an operator
    act on it, via SGLANG_UNEVEN_TOKEN_VECTOR.

    Unset -> the flip vector, byte-identical to the previous behaviour.
    """
    from sglang.srt import environ as _environ

    raw = _environ.envs.SGLANG_UNEVEN_TOKEN_VECTOR.get()
    flip_vec = parse_flip_vector(server_args)
    if not raw:
        return flip_vec

    try:
        vec = [int(x) for x in str(raw).split(",")]
    except ValueError as e:
        raise PhaseFlipBootError(
            f"SGLANG_UNEVEN_TOKEN_VECTOR={raw!r} is not a comma-separated "
            f"list of integers"
        ) from e
    if len(vec) != len(flip_vec):
        raise PhaseFlipBootError(
            f"SGLANG_UNEVEN_TOKEN_VECTOR={raw!r} has {len(vec)} entries but "
            f"the flip vector has {len(flip_vec)}. The token split is a "
            f"per-rank ratio over the SAME ranks as the weight split, so "
            f"the lengths must agree."
        )
    if any(x < 1 for x in vec):
        raise PhaseFlipBootError(
            f"SGLANG_UNEVEN_TOKEN_VECTOR={raw!r} has a non-positive entry. "
            f"A rank with token ratio 0 would own no KV rows while still "
            f"holding a weight shard, which the owner rule cannot express."
        )
    return vec


# Fields the TP-stack copy overrides, and ONLY these (the #470 REACH
# discipline: every context reader inside the build sees exactly these
# values changed and nothing else). Pinned by test.
TP_STACK_OVERRIDDEN_FIELDS: Tuple[str, ...] = (
    "tp_size",
    "pp_size",
    "rank_tp_ratio",
    "dcp_size",
    "pp_layer_ratio",
    "pp_stage_ratio",
    "enable_phase_flip",
    "phase_flip_tp_vector",
)


def derive_tp_stack_server_args(server_args, pp_id_space: int | None = None):
    """The TP decode phase's ServerArgs: a deepcopy of the boot args with
    the geometry rotated from (tp=1, pp=N) to (tp=N, dcp=N, pp=1).

    V1 rule: ONE vector drives both the weight shard plan (rank_tp_ratio)
    and the token owner rule (the DCP vector) -- the runbook's validated
    ``--rank-tp-ratio 30,17,17`` uneven-TP lane, reached exactly as a
    production TP boot would reach it. The copy clears the flip flags: it
    DESCRIBES the TP-shaped stack, it does not enable a nested flip."""
    import copy

    vec = parse_flip_vector(server_args)
    tp_args = copy.deepcopy(server_args)
    tp_args.tp_size = len(vec)
    tp_args.pp_size = 1
    tp_args.rank_tp_ratio = list(vec)
    tp_args.dcp_size = len(vec)
    tp_args.pp_layer_ratio = None
    tp_args.pp_stage_ratio = None
    tp_args.enable_phase_flip = False
    tp_args.phase_flip_tp_vector = None

    # THE TP POOL IS SIZED TO THE ID SPACE, NOT TO AN OPERATOR NUMBER.
    #
    # The scheduler's allocator is the PP stack's, so the TP layout can only
    # ever be handed slot ids below the PP capacity. A TP pool LARGER than
    # that addresses nothing extra -- the surplus rows are unreachable by
    # construction -- while still costing the VRAM the PP pool needs. Left to
    # size itself against its own budget it does exactly that: measured 788026
    # rows against an id space of 367704, and an uncapped boot dies in
    # cuMemCreate with the corridor guard counting free down to -4 MiB.
    #
    # The historical workaround was to have the operator pass a
    # --max-total-tokens that happened to be small enough. That is a guess
    # about physics wearing the shape of a policy: too low and it silently
    # caps a pool the VRAM would have backed (it also MASKS the per-rank
    # imbalance, because every non-binding rank then reports the cap as its
    # capacity); too high and the boot OOMs.
    #
    # Deriving it removes the guess. The pool becomes exactly what the
    # hardware backs, bounded only by what the allocator can address, and it
    # tracks the id space automatically whenever the PP side grows. The
    # >= check after the TP worker is built stays as the real invariant --
    # this only stops the TP side from overshooting it.
    if pp_id_space is not None:
        tp_args.max_total_tokens = int(pp_id_space)
        # #1030: RECORD THE PROVENANCE, so the reader does not have to guess.
        #
        # The sizing log named this value "--max-total-tokens user limit"
        # whenever it bound -- a label that was true only in the era the
        # comment above describes, when an operator DID pass the flag. Since
        # the derivation the field is populated here, and the log was
        # attributing a machine-derived cap to an operator input nobody made.
        # Measured 2026-08-30 (boot_855_704bgroup2): "projected 1309248 ->
        # EFFECTIVE max_total_num_tokens 613722 (bound by --max-total-tokens
        # user limit 613722)" on a launch whose command line contains no such
        # flag.
        #
        # A stamp rather than an inference: the reader must DISTINGUISH the
        # two provenances, not guess between them, because both are legitimate
        # -- an operator may still pass --max-total-tokens, and that case must
        # keep its own label.
        tp_args.max_total_tokens_from_pp_id_space = int(pp_id_space)
    return tp_args


@contextmanager
def phase_flip_tp_scope(world_rank: int, n: int):
    """Geometry scope for building/running the TP stack.

    Contextvar override for construction-time caching + module routing for
    forward-time collectives (see module docstring). ``world_rank`` is this
    process's flat world rank, which under the primary (tp=1, pp=N)
    topology equals pp_rank and IS the flip-TP rank.

    #785/#791 PART B: RE-ENTRANT w.r.t. an already-armed TP routing. The
    boot-time caller (``build_phase_flip_tp_stack``) always enters this with
    routing OFF, so hard-coding ``set_phase_flip_tp_active(False)`` on exit
    used to be a no-op there. It stopped being a no-op once
    ``restore_deferred_cold_stack`` started opening this scope AT THE PP->TP
    CUTOVER: by that point ``phase_flip_runtime`` has ALREADY called
    ``set_phase_flip_tp_active(True)`` for the whole TP phase, and that phase
    keeps running after this scope returns. Forcing it back to False on exit
    would silently route every later TP-phase collective back onto the
    primary tp=1 groups -- a silent no-op all-reduce, exactly the corruption
    class ``set_phase_flip_tp_active``'s own docstring says the routing
    exists to prevent. Save the value ``phase_flip_tp_routing_active()``
    reports on entry and restore THAT in the finally, instead of hard-coding
    False -- at the boot call site the saved value is always False, so that
    caller's behaviour is unchanged."""
    from sglang.srt.distributed.parallel_state import (
        get_phase_flip_group,
        phase_flip_tp_routing_active,
        set_phase_flip_tp_active,
    )
    from sglang.srt.runtime_context import get_parallel

    flip_tp = get_phase_flip_group("tp")
    flip_dcp = get_phase_flip_group("dcp")
    flip_pp = get_phase_flip_group("pp")
    prior_tp_routing_active = phase_flip_tp_routing_active()
    set_phase_flip_tp_active(True)
    # --pp-layer-ratio exports SGLANG_PP_LAYER_PARTITION process-wide; the
    # TP stack builds with pp_size=1 and get_pp_indices would refuse a
    # 3-way partition (found on the first real-metal flip boot,
    # 2026-08-08). Mask it for the build only; the PP stack read it at
    # primary init and never re-reads it at forward time.
    pp_partition_env = os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
    try:
        with get_parallel().override(
            tp_size=n,
            tp_rank=world_rank,
            moe_tp_size=n,
            moe_tp_rank=world_rank,
            moe_ep_size=1,
            moe_ep_rank=0,
            attn_tp_size=n,
            attn_tp_rank=world_rank,
            dcp_enabled=True,
            dcp_size=n,
            dcp_rank=world_rank,
            attn_dcp_size=n,
            attn_dcp_rank=world_rank,
            pp_size=1,
            pp_rank=0,
            tp_group=flip_tp,
            dcp_group=flip_dcp,
            pp_group=flip_pp,
            attn_tp_group=flip_tp,
        ):
            yield
    finally:
        set_phase_flip_tp_active(prior_tp_routing_active)
        if pp_partition_env is not None:
            os.environ["SGLANG_PP_LAYER_PARTITION"] = pp_partition_env


def checkpoint_param_dict(model) -> Dict[str, torch.nn.Parameter]:
    """Named checkpoint parameters of a model, non-checkpoint families
    excluded by name (NON_CHECKPOINT_NAME_PATTERNS)."""
    return {
        name: p
        for name, p in model.named_parameters()
        if not any(pat in name for pat in NON_CHECKPOINT_NAME_PATTERNS)
    }


def _grade_arena_tail_derivation(primary_runner, world_rank, layout_pp, layout_tp):
    """#785: grade the sizing-time derivation against what this boot measured.

    The derivation is parked on the runner by
    ``ModelRunner._instrument_arena_tail_derivation`` while the pool is being
    sized. Absent means it was not attempted (flips off) or it declined and
    said why -- in both cases the boot sized from the seam record exactly as
    before, and there is nothing to grade.

    ``world_rank`` here and ``_seam_world_rank`` there are the same number by
    construction: the flip's primary topology is (tp=1, pp=N), so the flat
    world rank IS pp_rank. Grading rank 0's derivation against rank 2's
    measurement would be the one way to make this check meaningless, so the
    two are stated rather than assumed.
    """
    derivation = getattr(primary_runner, "_arena_tail_derivation", None)
    if derivation is None:
        return None
    from sglang.srt.managers.arena_tail_probe import grade_derivation

    derived_pp, derived_tp = derivation
    return grade_derivation(
        int(world_rank),
        derived_pp,
        derived_tp,
        int(layout_pp.total_bytes),
        int(layout_tp.total_bytes),
    )


def prime_arena_from_image(arena, layout, image):
    """THE PRIMING FILL: the first H2D of a layout, with nothing to keep.

    Deliberately the SAME call as a warm flip, with ``outgoing_bytes=0``. At
    boot the arena holds nothing worth placing back, so the copy-back has zero
    length and the rotation degenerates to the plain contiguous H2D it has
    always been -- one path, not two, which is what keeps the warm path from
    growing a boot-shaped special case.

    INSTRUMENTED SEPARATELY (P4). The first flip after boot still primes from
    disk, and a steady-state figure averaged over a mean that includes it is
    the measurement error this ticket is most likely to make. The returned
    stats carry ``priming=True`` so the two can never be summed by accident.
    """
    from sglang.srt.model_executor.rotation_executor import (
        rotate_arena,
        rotation_report,
    )
    from sglang.srt.model_executor.weights_arena import (
        _refill_chunk_bytes,
        _refill_depth,
    )

    stats = rotate_arena(
        arena=arena,
        host_image=image,
        incoming_bytes=int(layout.total_bytes),
        outgoing_bytes=0,
        chunk_bytes=_refill_chunk_bytes(),
        depth=_refill_depth(),
        ring=None,
        priming=True,
    )
    try:
        logger.info("%s %s", LOG_PREFIX, rotation_report("boot", stats))
    except Exception:  # noqa: BLE001 - an instrument may never break a boot
        pass
    return stats


def snapshot_and_free(
    named: Dict[str, torch.nn.Parameter],
    layout: ArenaLayout,
    pin: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Host image of ``named`` under ``layout``, then free every device
    original by rebinding ``param.data`` to an empty placeholder.

    Between this call and the later ``bind_arena_views`` + ``arena_refill``
    the parameters are DEAD -- any forward would fail loudly on the 0-sized
    placeholder, which is the wanted behavior (nothing may run mid-boot)."""
    image = image_from_tensors(named, layout, pin=pin, out=out)
    freed = set()
    for name, param in named.items():
        key = param.data.untyped_storage().data_ptr()
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        freed.add(key)
    logger.info(
        "%s snapshotted %d params (%d storages, %.2f MiB image) and freed "
        "their device storages",
        LOG_PREFIX,
        len(named),
        len(freed),
        image.numel() / 1048576.0,
    )
    return image


def _full_attn_row_schema(pool) -> Tuple[int, int, int, str]:
    """(head_num, head_dim, elem_bytes, dtype_str) of a hybrid pool's
    full-attention KV rows -- the byte schema of one token's K (and V)
    row in one layer."""
    full = pool.full_kv_pool
    k0 = full.k_buffer[0]
    # [rows, head_num, head_dim] (page_size folded into rows at size 1).
    head_num = int(k0.shape[-2])
    head_dim = int(k0.shape[-1])
    return head_num, head_dim, k0.element_size(), str(k0.dtype)


def assert_row_schema_compatible(pp_pool, tp_pool) -> None:
    """Operator pin 3 (DESIGN_631 3.6a): the flip's KV move is a pure row
    redistribution ONLY IF both layouts' full-attention rows are the same
    bytes -- this checks that byte equality directly (head_num * head_dim *
    elem_bytes per pool), not a claim about WHY it holds.

    #785/#791 PART D: it is NOT true, as a blanket property of the fork,
    that "weighted DCP replicates KV heads". Row-byte equality here is a
    consequence of ``uneven_dcp_kv_replicated(dcp_size)`` being True for the
    flip TP layout -- true on this rig's flip configuration because
    ``build_phase_flip_tp_stack`` refuses to build unless
    ``server_args.uneven_weighted_dcp_enabled()``, and the flip TP scope
    always sets ``attn_dcp_size`` to the full TP width -- NOT a consequence
    of head-count arithmetic. The LOCAL attention head split is a SEPARATE
    predicate, ``attn_kv_replicated(tp_size, total_num_kv_heads)``, which is
    OFF whenever kv_heads >= tp_size: e.g. 4 total KV heads over tp=3 gives
    per-rank heads [2, 1, 1], head-SHARDED, not replicated, at the attention
    layer. That predicate does not gate what this function checks -- the
    pool's STORED row width is ``uneven_dcp_kv_replicated``'s call, not
    ``attn_kv_replicated``'s, and the two happening to agree on a config with
    a uniform head_num is what let this function pass at boot; it is not
    what the passing check proves. Outside a flip TP scope (DCP disabled, or
    no --rank-tp-ratio plan installed) ``uneven_dcp_kv_replicated`` is False
    and the TP pool would store head-sharded rows -- this check must, and
    does, still fail loudly in that case rather than silently accepting a
    byte-width mismatch. Checked on the real pools at boot, and hermetically
    from the real config in the pin-3 test."""
    pp_schema = _full_attn_row_schema(pp_pool)
    tp_schema = _full_attn_row_schema(tp_pool)
    if pp_schema != tp_schema:
        from sglang.srt.distributed.utils import uneven_dcp_kv_replicated
        from sglang.srt.runtime_context import get_parallel

        observed_dcp_size = get_parallel().attn_dcp_size
        raise PhaseFlipBootError(
            f"full-attention KV row schemas DIVERGE between the phase "
            f"layouts: PP (head_num, head_dim, elem_bytes, dtype) = "
            f"{pp_schema}, TP = {tp_schema}, observed at attn_dcp_size="
            f"{observed_dcp_size} (uneven_dcp_kv_replicated="
            f"{uneven_dcp_kv_replicated(observed_dcp_size)}). The flip's KV "
            f"move is only a row redistribution when both layouts store "
            f"identical row bytes; the TP layout's row width is set by "
            f"DCP-wide pool replication (uneven_dcp_kv_replicated), not by "
            f"the local attn_kv_replicated head split; refusing to boot a "
            f"flip that would scatter mis-shaped rows."
        )


def _pool_full_attn_row_schema_defensive(pool) -> Optional[Tuple[int, int]]:
    """(head_num, head_dim) of ``pool``'s full-attention KV rows, read
    DEFENSIVELY for the Part-C geometry guard, which must skip rather than
    crash on a pool shape it does not recognize -- it is a belt-and-braces
    check, not the source of truth for row compatibility (that is
    ``assert_row_schema_compatible``, checked on the real pools at boot).

    Tries the hybrid-pool path first (``_full_attn_row_schema``, which reads
    ``pool.full_kv_pool.k_buffer``), then falls back to a plain non-hybrid
    pool's own ``k_buffer`` directly. Returns None if neither shape matches."""
    try:
        head_num, head_dim, _, _ = _full_attn_row_schema(pool)
        return head_num, head_dim
    except AttributeError:
        pass
    try:
        k0 = pool.k_buffer[0]
        return int(k0.shape[-2]), int(k0.shape[-1])
    except (AttributeError, IndexError, TypeError):
        return None


def _guard_geometry_before_backend_build(tp_worker, where: str) -> None:
    """#785/#791 PART C: refuse to build attention backends under an ambient
    geometry that disagrees with what the KV pool was already baked for.

    THE FAILURE THIS GUARDS. The attention backends read the TP geometry
    FRESH at construction time from ``get_parallel()`` (contextvars), while
    the KV pool's row width is a BAKED ``ModelRunner`` attribute fixed once
    at pool-construction time. If ``build_cold_stack_posts`` is ever reached
    with no ``phase_flip_tp_scope`` open around it -- as
    ``restore_deferred_cold_stack`` used to, before part A -- the backends
    build under the wrong geometry: ``uneven_dcp_kv_replicated`` reads the
    ambient (not the flip TP) dcp_size, disagrees with the replicated/sharded
    row width the pool was actually built with, ``TritonAttnBackend.uneven_dcp``
    ends up wrong, ``_set_kv_buffer`` skips ``_dcp_write_gather``, and the
    store kernel silently refolds per-rank KV rows into a pool sized for
    gathered replicated rows -- the #785 "store_cache rejected ...
    row_dim=1024 -> k_rows=12" crash, with no error until the first decode.

    CHEAP AND MUST NOT FIRE ON THE HEALTHY PATH: one buffer-shape read, no
    collective, no allocation. On both the boot call site and the (post part
    A) cutover call site, this runs inside an already-open
    ``phase_flip_tp_scope`` where the ambient geometry matches the pool by
    construction, so the predicates below always agree and this is a no-op.
    """
    from sglang.srt.distributed.utils import uneven_dcp_kv_replicated
    from sglang.srt.runtime_context import get_parallel

    model_runner = getattr(tp_worker, "model_runner", None)
    pool = getattr(model_runner, "token_to_kv_pool", None) if model_runner else None
    model_config = getattr(model_runner, "model_config", None) if model_runner else None
    if pool is None or model_config is None:
        return
    schema = _pool_full_attn_row_schema_defensive(pool)
    if schema is None:
        return
    head_num, head_dim = schema
    try:
        total_num_kv_heads = int(model_config.get_total_num_kv_heads())
    except (AttributeError, TypeError, ValueError):
        return

    attn_dcp_size = int(get_parallel().attn_dcp_size)
    pool_baked_replicated = head_num == total_num_kv_heads
    ambient_uneven_dcp = uneven_dcp_kv_replicated(attn_dcp_size)
    if pool_baked_replicated != ambient_uneven_dcp:
        # attn_tp_size is DIAGNOSTIC ONLY (the raise decision above depends
        # solely on attn_dcp_size); reading it can itself raise when no
        # attn-tp group is initialized at all (e.g. this guard firing from a
        # bare hermetic caller), and a diagnostic read must never suppress
        # the refusal it is trying to explain.
        try:
            attn_tp_size = str(int(get_parallel().attn_tp_size))
        except Exception:
            attn_tp_size = "<unavailable: attn-tp group not initialized>"
        raise PhaseFlipBootError(
            f"geometry mismatch building attention backends at {where!r}: "
            f"the KV pool was baked with head_num={head_num} "
            f"(row_dim={head_num * head_dim}) against "
            f"total_num_kv_heads={total_num_kv_heads} "
            f"({'replicated' if pool_baked_replicated else 'sharded'} row "
            f"schema), but the ambient geometry the backends are about to "
            f"build under is attn_dcp_size={attn_dcp_size}, "
            f"attn_tp_size={attn_tp_size}, giving "
            f"uneven_dcp_kv_replicated={ambient_uneven_dcp}. Building the "
            f"attention backends under this ambient geometry would make "
            f"TritonAttnBackend.uneven_dcp disagree with the pool's baked "
            f"row schema, skip _dcp_write_gather in _set_kv_buffer, and "
            f"silently refold per-rank KV rows into a pool sized for "
            f"gathered replicated rows -- refusing to build rather than "
            f"corrupt the KV cache."
        )


#: #690's reference, kept with the conditions that make it transferable --
#: without them it is not a baseline, it is a number. Measured on the PINNED
#: image path (which predates the file-backed arm entirely), pp_to_tp, as a
#: mean over 14 flips with all three ranks moving 9614.9 MiB each:
#: rank1 4.93 GB/s (3080, PCIe x4), rank0 7.08 GB/s (5090, x8), rank2
#: 8.88 GB/s (3080, x8). NOTE_690_gdn_state_spread.md:58-85.
_PINNED_REF_LO_GBPS = 4.93
_PINNED_REF_HI_GBPS = 8.88


def refill_report(
    direction: str, elapsed: float, nbytes: int, file_backed: bool
) -> str:
    """One line describing a refill leg, comparable to something real.

    WHY THIS IS NOT A ONE-LINER. The previous form printed the leg's duration
    beside "the ~3.1 s pinned baseline". That comparison is invalid three ways
    and on 2026-08-22 it produced a briefing that called the flip economy
    broken and went looking for a silent host-RAM fallback:

      SCOPE -- ~3.1 s is a WHOLE FLIP (NOTE_677_floor_components.md:135-143
        uses it as "Against a ~3.1 s flip"), not a refill leg.
      PATH  -- it was measured on the pinned arm, which predates the
        file-backed arm, so it is not a baseline this path ever held.
      BYTES -- it moved 9614.9 MiB/rank; these legs move 8574-16363 MiB, and
        the elapsed time tracks bytes moved (r ~ 0.80), so seconds are not
        comparable across them at all.

    A rate against a rate is comparable; seconds against seconds are not. The
    file-backed arm IS slower, and the line still says so -- what it must not
    do is let that read as a regression against a baseline that never existed
    for it. The arm is an explicit opt-in whose help text names what it buys:
    without it the images are ~68.7 GiB of unreclaimable host RAM on a
    swapless box and the boot is OOM-killed during init.
    """
    mib = nbytes / 1048576.0
    rate = (mib / elapsed) if elapsed > 0 else 0.0
    head = (
        f"REFILL {direction} took {elapsed:.3f} s for {mib:.1f} MiB ({rate:.0f} MiB/s)"
    )
    if not file_backed:
        # The pinned arm IS the reference path, so it is measured against the
        # reference directly and buys nothing it needs to justify.
        return (
            f"{head} -- pinned images, the same path as the "
            f"{_PINNED_REF_LO_GBPS:.2f}-{_PINNED_REF_HI_GBPS:.2f} GB/s "
            "reference (#690, per rank, pp_to_tp)."
        )
    return (
        f"{head} -- file-backed images. Reference for this leg is a RATE, not "
        f"a duration: #690 measured {_PINNED_REF_LO_GBPS:.2f}-"
        f"{_PINNED_REF_HI_GBPS:.2f} GB/s per rank on the PINNED path "
        "(pp_to_tp, 9614.9 MiB/rank), and the ~3.1 s often quoted alongside it "
        "is a whole flip, not this leg. The file-backed arm is slower by "
        "design, not by regression: it is what makes the image post "
        "reclaimable, and without it the images are ~68.7 GiB of "
        "unreclaimable host RAM on a swapless box."
    )


@dataclass
class PhaseFlipStacks:
    """Everything the scheduler flip protocol needs from the boot build."""

    tp_worker: object
    arena: torch.Tensor
    layout_pp: ArenaLayout
    layout_tp: ArenaLayout
    #: #809/W28: ONE host image buffer, sized for the LARGER layout plus its
    #: 8-byte trailer, holding whichever layout is currently RESTING. The two
    #: lifetime images this replaces were the dual pin, and W26 OOM-killed
    #: BOTH its arms in the LAUNCH phase, before any flip ran. At each flip
    #: the resting layout streams out of this buffer while the outgoing one is
    #: placed back into the pages it frees, so RAM holds one layout image plus
    #: the overshoot rather than two whole layouts.
    rotation_image: torch.Tensor
    #: Which layout ``rotation_image`` currently holds: ``"pp"`` or ``"tp"``.
    #: The rotation is a swap, so this alternates with every flip. It is an
    #: INVARIANT, not a hint: a mismatch means the buffer does not contain the
    #: layout about to be served, and under a single-image budget there is no
    #: second image to fall back to, so the refill refuses.
    image_holds: str
    #: The WEIGHT shard vector (--phase-flip-tp-vector): how the TP layout
    #: splits heads/compute across the ranks.
    vector: Tuple[int, ...]
    #: The KV TOKEN vector: how the TP layout splits token ROWS across the
    #: ranks under the weighted owner rule. Equal to :attr:`vector` unless
    #: SGLANG_UNEVEN_TOKEN_VECTOR overrides it (parse_flip_token_vector).
    #: These are NOT interchangeable -- the owner rule and the flip's
    #: transition plan are token-space quantities and must use THIS one,
    #: or rows are routed under a different split than the pools were
    #: sized for.
    token_vector: Tuple[int, ...]
    #: The speculative draft worker for the TP DECODE phase, or None when
    #: the instance runs without speculation. Built on the TP stack and
    #: swapped into the scheduler at cutover (#631); it never participates
    #: in the PP phase.
    draft_worker: object = None
    #: RUNG 3 carrier when --phase-flip-spill-depth >= arena, else None. Holds
    #: the weights arena on a VA-stable reservation so the tail can be handed
    #: back to the driver in the phase whose layout does not reach it.
    arena_carrier: object = None
    #: #1078: the TWO-FILE arm's images -- one per layout, each its own
    #: exact-sized file, each carrying a trailer that is a BOOT CONSTANT.
    #: Both None on the default single-image path, and that is the switch
    #: `_timed_arena_refill` reads: the two arms cannot both be armed, because
    #: a rotation needs one max-sized buffer and this needs two exact ones.
    #: Under this arm `rotation_image` is `image_tp` and is never rotated --
    #: it is simply the resting layout's image at the end of boot, which is
    #: what `image_holds="tp"` already says.
    image_pp: Optional[torch.Tensor] = None
    image_tp: Optional[torch.Tensor] = None

    def two_file_arm(self) -> bool:
        """Is this stack running the #1078 two-file scheme?

        Read off the STACK, not off the env: the images were decided at boot
        and an env flipped mid-process must not change which scheme a leg
        thinks it is running. That is the #742 class in the other direction.
        """
        return self.image_pp is not None and self.image_tp is not None

    def refill(self, direction: str) -> None:
        """The weights leg of a flip: a chunk ROTATION of the arena (#809/W28).

        The target phase's image streams RAM -> VRAM while the outgoing phase's
        arena bytes are placed back into the pages that image vacates, so RAM
        ends holding exactly the now-resting layout, primed for the next flip.
        PCIe is full duplex, so the copy-back rides the idle return direction.

        THE COPY-BACK IS NOT WRITE-BACK. The weights are immutable and nothing
        is saved; it is residency PLACEMENT for the next flip, which is what a
        single-layout RAM budget requires.

        WHAT THIS GIVES UP, and it is a real loss rather than an oversight.
        The old two-image refill passed ``restore=(other_layout, other_image)``
        so that a checksum mismatch rewrote the ACTIVE layout from its own
        separate image and the abort left both layouts byte-exact. That arm
        needs a second lifetime image to read from, which is precisely the
        dual pin W26 proved impossible here. With one buffer it cannot exist:
        a mismatch now declares the arena undefined and refuses loudly instead
        of silently serving it. The single-layout RAM budget is what buys the
        flip its steady state, and this is its price.
        """
        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP

        if direction == PP_TO_TP:
            # COMMIT THE HIGH-WATER FIRST. See _refill_high_water_bytes: the
            # refill writes the TP layout and its restore= arm may rewrite the
            # PP layout, so BOTH must be backed before a byte moves. On a rank
            # where TP is the larger layout this is the difference between a
            # flip and a cudaErrorInvalidValue into the released tail.
            self._commit_refill_high_water()
            self._timed_arena_refill("pp_to_tp", self.layout_tp, self.layout_pp, "tp")
            # RUNG 3, AFTER the refill and not before: the restore= arm above
            # rewrites the PP layout on a checksum mismatch, and the PP layout
            # reaches into the tail wherever PP is the larger layout.
            # Releasing first would fault that recovery path on unbacked
            # memory.
            if self.arena_carrier is not None:
                released = self.arena_carrier.set_active_prefix(
                    self.layout_tp.total_bytes
                )
                if released:
                    logger.info(
                        "%s rung 3 released %.1f MiB of weights-arena tail to "
                        "the driver (TP layout needs %.1f of %.1f MiB); "
                        "arena address unchanged",
                        LOG_PREFIX,
                        released,
                        self.layout_tp.total_bytes / 1048576.0,
                        self.arena.numel() / 1048576.0,
                    )
        elif direction == TP_TO_PP:
            # BEFORE the refill, and to the HIGH-WATER rather than to the PP
            # layout: the refill writes the PP layout and its restore= arm may
            # rewrite the TP layout, so the larger of the two is what has to be
            # backed. This commit is an allocation inside the no-return region
            # -- it is priced into the affordability verdict by
            # PhaseFlipRuntime._arena_tail_bytes before the flip commits.
            self._commit_refill_high_water()
            self._timed_arena_refill("tp_to_pp", self.layout_pp, self.layout_tp, "pp")
            # AND RELEASE AFTER, symmetrically with the pp->tp leg. Without
            # this the tail stays committed for the whole PP phase on a rank
            # whose TP layout is the larger one, which is rung 3's entire
            # purpose given away.
            if self.arena_carrier is not None:
                released = self.arena_carrier.set_active_prefix(
                    self.layout_pp.total_bytes
                )
                if released:
                    logger.info(
                        "%s rung 3 released %.1f MiB of weights-arena tail to "
                        "the driver (PP layout needs %.1f of %.1f MiB); "
                        "arena address unchanged",
                        LOG_PREFIX,
                        released,
                        self.layout_pp.total_bytes / 1048576.0,
                        self.arena.numel() / 1048576.0,
                    )
        else:
            raise PhaseFlipBootError(f"unknown flip direction {direction!r}")

    def refill_high_water_bytes(self) -> int:
        """Bytes of arena that must be BACKED for any refill to be safe.

        THE ASSUMPTION THIS REPLACES, stated so it cannot come back: rung 3
        was written when "PP is the larger layout on every rank of this rig"
        was true, so the tail was committed on tp->pp and released on pp->tp,
        and the pp->tp refill was allowed to run against whatever happened to
        be committed. Change the PP stage ratio -- ``--pp-stage-ratio 15,9,8``
        derives 32,16,16 layers over 64 -- and a middle rank's PP layout drops
        BELOW its TP layout. The tp->pp leg then decommits down to the smaller
        PP layout, and the next pp->tp refill copies the larger TP image
        straight into the released tail: ``CUDA error: invalid argument``,
        inside the no-return region, killing all three ranks at the first
        flip. Measured on this rig, 2026-08-11.

        A refill also has its restore= arm, which rewrites the OTHER layout on
        a checksum mismatch. So the safe span is the MAXIMUM of the two
        layouts on either leg, not the layout being written -- which is what
        makes this a property of the arena rather than of the direction.
        """
        return max(int(self.layout_pp.total_bytes), int(self.layout_tp.total_bytes))

    def _timed_arena_refill(
        self, direction: str, incoming, outgoing, wants: str
    ) -> None:
        """#758 emitter (3 of 3): PER-RANK FLIP REFILL TIME.

        WHY THIS DID NOT EXIST AND HAD TO. The comp4 load ladder
        (2026-08-18) could not accept the file-backed-image arm because the
        acceptance asks for a refill number against the ~3.1 s pinned
        baseline, and NOTHING in the tree emitted one -- a grep of this file
        and weights_arena.py found no elapsed/ms line on the refill path at
        all. The arm's whole cost model is "a pageable H2D copy while cached,
        plus a disk read when the pages were reclaimed"; without a timer that
        claim is unfalsifiable on metal.
        This wraps the copy that the #690 high-water marks already bracket
        (``_commit_refill_high_water`` immediately above every call).

        #873 CORRECTION, because the sentence that stood here -- "so the number
        is the refill leg proper and nothing else" -- is what let this number be
        read as a transfer. It is the refill leg, and the leg is not one
        mechanism. Measured, boot_w40_857strict_0826_0516.log, PP0 pp_to_tp:
        4.818 s = save 4.342 + checksum 0.319 + wait 0.084 + d2h-call 0.026 +
        h2d-call 0.020 + ring 0.001 + plan 0.001. The dominant term on THAT leg
        is ``ops.save`` -- a HOST-TO-HOST memcpy into the staging ring, which
        in-place aliasing forces on 90-97 % of the chunks
        (rotation_executor.py, the ``aliased`` branch). The rate this line
        prints is therefore an aggregate over staging, checksum and transfer;
        it is comparable to another leg's aggregate and to nothing else. The
        decomposition is registered with the seam census below so the two are
        read together.

        #1082 -- ONE LEG OF THAT ARGUMENT IS WITHDRAWN, and it was mine. The
        sentence above used to end "...and not the PCIe transfer the name
        implies", resting on a quoted `gpu-span d2h 0.000s / h2d 0.000s`. That
        zero was never a measurement: ``RotationPhases.gpu_d2h_s`` and
        ``gpu_h2d_s`` had exactly one writer in the whole tree -- their own
        dataclass default -- and one reader, the renderer. Every boot printed
        0.000 s because nothing ever wrote them. The save-dominates conclusion
        stands on the save term itself, which IS measured; the "no device time
        took part" half had no evidence and is removed rather than reworded.
        The fields are measured from #1082 on (CUDA events per lane, read once
        after the drain) and render as `not-measured` when they are not, so a
        future reader can tell the two apart. Note also that the term names
        changed with that ticket: ``d2h_issue_s`` -> ``d2h_call_s``, because on
        a pageable destination the call performs the whole transfer and the
        word "issue" asserted an enqueue that does not happen.

        PER RANK, NOT REDUCED. The flip's cost is the SLOWEST rank's copy --
        the layouts differ in size per rank (pp 17219 / tp 16329 MiB on PP0
        against 8977 on PP1 here), so a mean would hide exactly the rank that
        sets the seam. Each rank logs its own; the reader takes the max.
        """
        import time as _time

        started = _time.perf_counter()
        # #856: this leg is 91% of a tp_to_pp flip (W25 seam census,
        # `refill_highwater->weights_refill` 9516.2 ms of a 10466.8 ms walk),
        # and it reported ONE aggregate rate. The read and the H2D are
        # pipelined, so that rate is min(read, h2d) with no way to say which
        # bound it hit -- which is why the 2.5x direction gap (tp_to_pp
        # 1351-1723 MiB/s vs pp_to_tp 3214-3915 for the same rank and within
        # 2.7% of the same bytes) could not be attributed from the log.
        #
        # #873: AND IT IS NEITHER OF THE TWO THIS SENTENCE OFFERS. "read or
        # h2d" names the only two candidates a reader is given here, and the
        # measured answer is a third that is not on the list -- ``ops.save``,
        # the host-to-host memcpy into the staging ring (4.342 s of a 4.801 s
        # leg on PP0, with both device spans at 0.000 s). This is not a
        # hypothetical cost of leaving the sentence: an independent reader
        # sweeping these segments in 2026-08 classified this leg
        # "SINGLE-MECHANISM, driver = bytes moved / PCIe bandwidth" and cited
        # THIS COMMENT as the authority, having never seen the phase lines. A
        # narrowed candidate set reads as a decomposition. The real one is
        # registered with the seam census below.
        leg_timing = RefillLegTiming()
        if self.two_file_arm():
            self._two_file_refill(
                direction, incoming, outgoing, wants, leg_timing, started
            )
            return
        # #809/W28: the leg IS the rotation. `incoming` streams out of the one
        # host buffer and `outgoing` is placed back into it, so the buffer ends
        # holding the layout that just left the arena.
        from sglang.srt.model_executor.rotation_executor import (
            RotationHazard,
            RotationPhases,
            rotate_arena,
            rotation_phase_report,
            rotation_report,
            rotation_ring,
        )
        from sglang.srt.model_executor.weights_arena import (
            _refill_chunk_bytes,
            _refill_depth,
        )

        if self.image_holds != wants:
            raise RotationHazard(
                f"{LOG_PREFIX} refill {direction}: the host image holds "
                f"{self.image_holds!r} but this leg must stream {wants!r} into "
                f"the arena. Under a single-image budget there is no second "
                f"image to read from, so this is an invariant violation rather "
                f"than a case to fall back on."
            )
        chunk = _refill_chunk_bytes()
        depth = _refill_depth()
        # #809 W28 follow-up: the warm path used to DISCARD the rotation's own
        # record, so `overlapped_steps` -- the duplex falsifier -- was not
        # observable on metal at all, and 97.6 % of a measured 4.833 s leg had
        # nowhere to be attributed. Both are captured here and logged below.
        rot_phases = RotationPhases()
        rot_stats = rotate_arena(
            arena=self.arena,
            host_image=self.rotation_image,
            incoming_bytes=int(incoming.total_bytes),
            outgoing_bytes=int(outgoing.total_bytes),
            chunk_bytes=chunk,
            depth=depth,
            ring=rotation_ring(chunk, depth),
            timing=leg_timing,
            phases=rot_phases,
        )
        # The swap happened, so the marker follows it. Set only on the success
        # path: a rotation that raises has left the arena undefined and said so,
        # and this instance must not serve either layout again. The marker is
        # deliberately NOT updated there -- it would assert a residency that
        # nothing has verified.
        self.image_holds = "pp" if wants == "tp" else "tp"
        elapsed = _time.perf_counter() - started
        # #873: HAND THE CENSUS THE DECOMPOSITION IT WAS MISSING. This leg is
        # the seam's dominant segment on every rank of every flip, and the
        # census reported it as one bar while THIS function already held the
        # #809/W28 phase breakdown and logged it to a separate, unreferenced
        # line. An operator reading the census line could not find that line
        # and hand-fitted a rate-plus-constant model to the bar instead,
        # deriving a 3.0 s "byte-independent constant" that is the intercept of
        # a model applied to four mechanisms with four cost drivers. What that
        # boot's own phase lines actually say, PP0 pp_to_tp: save 4.342 +
        # checksum 0.319 + wait 0.084 + d2h-call 0.026 + h2d-call 0.020 -- on
        # THAT leg the mass is the host-side staging memcpy that in-place
        # aliasing forces.
        #
        # #1082: the clause "gpu-span d2h 0.000s / h2d 0.000s" stood here as
        # the second half of that argument and is WITHDRAWN -- those two fields
        # had no writer, so the zero was a dataclass default, not a device
        # reading. It is also not a general result: on boot_855_1078spec the
        # dominant term is d2h-call at 94.7-96.1 % of the leg on all six legs,
        # with save at 1.5-2.8 s. A decomposition is a per-leg fact.
        #
        # Registered against `weights_refill` because that is the mark the walk
        # stamps when this returns, i.e. the mark that CLOSES this segment.
        from sglang.srt.managers import phase_flip_seam_census as seam_census

        seam_census.explain(
            "weights_refill",
            (
                ("save", rot_phases.save_s),
                ("checksum", rot_phases.checksum_s),
                ("wait", rot_phases.wait_s),
                # #1082: renamed from "d2h-issue"/"h2d-issue". The old labels
                # asserted an enqueue cost; on a pageable destination the call
                # carries the whole transfer, and these two segments were read
                # as cheap plumbing for that reason.
                ("d2h-call", rot_phases.d2h_call_s),
                ("h2d-call", rot_phases.h2d_call_s),
                ("ring", rot_phases.ring_s),
                ("plan", rot_phases.plan_s),
            ),
        )
        # #677: FEED THE ECONOMICS THE MEASURED LEG, NOT A REMEMBERED ONE.
        # The flip policy priced a leg at a 3.2 s pinned-era constant while the
        # file-backed arm measured 22-24 s -- 7.03x too cheap, which moved
        # break-even from 49,248 tokens to 7,004 and is why flips churn. The
        # estimator is inert until it is fed, and this is the only place in the
        # process that knows what a leg actually cost, because it is the timer
        # that brackets the copy.
        # #802 then repriced the leg downward (slowest rank 11.070 -> 4.246 s,
        # whole flip 12.121 -> 4.998 s) by reading the image instead of
        # faulting it. That is exactly why this feeds a MEASUREMENT and not a
        # constant: the regime changed underneath the number, and the estimator
        # followed it.
        # #777: the ESTIMATOR followed it. The THRESHOLD did not -- N is priced
        # once in `config_from_env` and never rebuilt, so "without anyone
        # editing a threshold" described a repricing that does not reach the
        # policy. `observe_flip_cost` now says so on the first flip that proves
        # it; repricing N itself is the planner's call.
        # #819: THE FEED MOVED, and the reason is that this timer measures a
        # STEP. phase_flip_runtime's header calls the weights-arena refill and
        # the KV seam "separate steps of the flip", so what is bracketed here
        # is a component of a leg, not a leg. Priced as if it were the whole,
        # it understated the flip by more than half: this timer reported
        # 3.60287/3.66144/4.62869 s on boot_window3_0823_1733.log while that
        # boot's own PHASE-FLIP DONE lines put a leg at 5681-12023 ms.
        # It also could not see the cutover at all, which is the term
        # SGLANG_SEAM_SHRINK moves -- so the seam shrink could never have
        # reached the threshold through this path.
        # The estimator is now fed the completed leg's own `total_ms`, once,
        # by `observe_flip_leg` at the completion site in scheduler.py. Feeding
        # both a component and its container into one EMA would converge to
        # neither, so this feed is retired rather than kept alongside.
        # The measurement itself is NOT lost: `refill_report` below still logs
        # this leg, which is what the #690 high-water marks bracket.
        try:
            logger.info(
                "%s %s -- %s",
                LOG_PREFIX,
                refill_report(
                    direction,
                    elapsed,
                    int(incoming.total_bytes),
                    self._images_are_file_backed(),
                ),
                # #856: the bound, named. Appended rather than folded into
                # `refill_report` because that function's whole contract is
                # "a rate against a rate is comparable" and this is a
                # different statement about the same leg.
                refill_bound_phrase(leg_timing),
            )
        except Exception:  # noqa: BLE001 - an instrument may never break a flip
            pass
        try:
            logger.info("%s %s", LOG_PREFIX, rotation_report(direction, rot_stats))
            logger.info(
                "%s %s %s", LOG_PREFIX, direction, rotation_phase_report(rot_phases)
            )
        except Exception:  # noqa: BLE001 - an instrument may never break a flip
            pass

    def _two_file_refill(
        self, direction: str, incoming, outgoing, wants: str, leg_timing, started
    ) -> None:
        """#1078: the leg WITHOUT a copy-back.

        THREE TERMS OF THE ROTATION DO NOT HAPPEN HERE, and their measured
        share of the leg is why this exists (boot_855_1078spec, PP0 pp_to_tp,
        63.911 s total): the D2H copy-back 60.692 s (95.0 %), the host-to-host
        staging `save` 2.805 s (4.4 %), and the trailer write. What is left is
        the read, which `arena_refill` already routes through #802's `preadv`
        path for a file-backed image.

        THE MARKER STILL MOVES. `image_holds` means "which layout is resting",
        and under two files that is still exactly one of them -- the one whose
        image the next leg will stream in. Keeping it truthful costs nothing
        and keeps every existing reader (the seam emitters, #758) correct.
        """
        import time as _time

        from sglang.srt.managers import phase_flip_seam_census as seam_census

        incoming_image = self.image_tp if wants == "tp" else self.image_pp
        outgoing_image = self.image_pp if wants == "tp" else self.image_tp
        outgoing_phase = "pp" if wants == "tp" else "tp"
        phases: Dict[str, float] = {}
        two_file_leg(
            arena=self.arena,
            incoming_layout=incoming,
            incoming_image=incoming_image,
            incoming_phase=wants,
            outgoing_layout=outgoing,
            outgoing_image=outgoing_image,
            outgoing_phase=outgoing_phase,
            timing=leg_timing,
            phases=phases,
        )
        # Only after the leg returned. A leg that raised left the arena in a
        # state this marker must not describe -- the same reason the rotation
        # sets it on the success path only.
        self.image_holds = outgoing_phase
        elapsed = _time.perf_counter() - started
        # #873's requirement, met by the arm that replaces it: the census gets
        # the decomposition, not one bar. Two terms only, because there are
        # only two -- naming a phase that does not exist here would be the
        # narrowed-candidate-set defect #873 recorded on the rotation.
        seam_census.explain(
            "weights_refill",
            (
                ("anchor", phases.get("anchor", 0.0)),
                ("read", phases.get("read", 0.0)),
            ),
        )
        try:
            logger.info(
                "%s %s -- %s",
                LOG_PREFIX,
                refill_report(
                    direction,
                    elapsed,
                    int(incoming.total_bytes),
                    self._images_are_file_backed(),
                ),
                refill_bound_phrase(leg_timing),
            )
            logger.info(
                "%s #1078 %s TWO-FILE leg: %.1f MiB in from the %r image, "
                "0.0 MiB back (no copy-back) -- anchor %.3fs + read %.3fs "
                "= %.3fs",
                LOG_PREFIX,
                direction,
                int(incoming.total_bytes) / 1048576.0,
                wants,
                phases.get("anchor", 0.0),
                phases.get("read", 0.0),
                elapsed,
            )
        except Exception:  # noqa: BLE001 - an instrument may never break a flip
            pass

    def _images_are_file_backed(self) -> bool:
        """Report the image mode so the refill number is never read against
        the wrong baseline (a pinned refill and a file-backed one are not
        the same measurement)."""
        import os as _os

        return _os.environ.get("SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED", "") == "1"

    def _commit_refill_high_water(self) -> None:
        from sglang.srt.managers import phase_flip_seam_census as seam_census

        if self.arena_carrier is not None:
            # set_active_prefix GROWS to the high-water here and can never
            # shrink to it, because the high-water is the maximum of both
            # layouts and the arena is never committed above that.
            self.arena_carrier.set_active_prefix(self.refill_high_water_bytes())
        # THE ONE BOUNDARY INSIDE THE REFILL LEG: arena COMMIT above, H2D copy
        # below (refill() calls this immediately before arena_refill on both
        # directions). Without it the whole leg lands in a single
        # 'weights_refill' bar, and a commit that stalls on the driver is
        # indistinguishable from a transfer that is merely bandwidth-bound --
        # the two have different fixes and live in different modules.
        #
        # #873: THIS MARK WORKS, AND ITS ANSWER HAS BEEN IN EVERY CENSUS LINE.
        # The commit is the segment `gdn_state->refill_highwater`, and on
        # boot_w40_857strict_0826_0516.log it reads 0.2 / 7.0 / 0.2 ms across
        # the three ranks -- so page commit is NOT where a flip's seconds go,
        # and that was already established rather than open. It was read as
        # open because `format_timing_line` sorts descending and a 0.2 ms
        # segment that ANSWERS a question about a 4819.9 ms segment lands at the
        # far end of the line where it reads as noise. The boundary below is
        # therefore kept as it is; what #873 changes is that the dominant
        # segment now carries its own decomposition, so a reader is not left to
        # reconstruct one from a sorted bar chart.
        #
        # Marked OUTSIDE the carrier guard on purpose: with no carrier the
        # commit is a no-op and the boundary is a zero-width step, which is
        # itself the answer to "was it the commit?". Skipping the mark there
        # would make a no-carrier rank's bar silently mean something else than
        # its peers'.
        #
        # mark() is a no-op when no census is open and swallows its own
        # exceptions, so this cannot become the reason a flip dies -- the
        # no-return-path contract this instrument has carried since #631.
        seam_census.mark("refill_highwater")


def build_flip_draft_worker(scheduler, tp_worker, tp_args, world_rank):
    """Construct the TP-decode-phase speculative draft worker, or None.

    Mirrors ``Scheduler.maybe_init_draft_worker`` with two deliberate
    substitutions, and no others: the target is the flip's TP-shaped
    worker rather than the boot worker, and the identity arguments are the
    TP-phase ones (``tp_rank`` = world rank, no pp_rank -- the draft
    workers have no PP form, which is exactly why speculation is confined
    to this phase).

    MUST be called inside ``phase_flip_tp_scope`` with ``tp_args``
    published on the context, like everything else on this stack: the
    draft runner reads its geometry from the published server args (#470),
    and a draft built against the target's PP geometry would shard its
    heads for the wrong topology.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo: SpeculativeAlgorithm = scheduler.flip_spec_algorithm
    if algo.is_none():
        return None
    if algo.is_ngram():
        # NGRAM keeps an external corpus manager wired to the tokenizer
        # channel, which the cutover does not rebuild. Refuse rather than
        # half-arm it.
        raise PhaseFlipBootError(
            "speculative algorithm 'ngram' is not supported in the phase "
            "flip's TP decode phase (its external corpus manager is not "
            "part of the cutover rebuild list)"
        )

    DraftWorkerClass = algo.create_worker(tp_args)
    draft = DraftWorkerClass(
        server_args=tp_args,
        gpu_id=scheduler.ps.gpu_id,
        tp_rank=world_rank,
        moe_ep_rank=0,
        nccl_port=scheduler.tp_worker.nccl_port,
        target_worker=tp_worker,
        dp_rank=scheduler.tp_worker.dp_rank,
        attn_cp_rank=scheduler.tp_worker.attn_cp_rank,
        moe_dp_rank=scheduler.tp_worker.moe_dp_rank,
    )
    logger.info(
        "%s TP-phase draft worker built: %s, tp_rank %d",
        LOG_PREFIX,
        type(draft).__name__,
        world_rank,
    )
    return draft


#: Marks a worker whose cold posts are already built. On the TARGET worker
#: rather than on the stacks object because the cutover reaches the worker
#: through ``stacks`` but the boot reaches it as a local, and a flag that
#: lives on the thing being built cannot go stale relative to it.
COLD_STACK_BUILT_ATTR = "_phase_flip_cold_stack_built"


def build_cold_stack_posts(
    tp_worker, draft_worker, draft_carrier, *, where: str
) -> bool:
    """The flip TP stack's PHASE-COLD posts: attention workspaces + graphs.

    Returns True if this call built them, False if they already existed.

    WHY THESE FOUR CALLS ARE ONE FUNCTION. They are the posts that the PP
    phase pays for and cannot use. The PP phase runs on the boot stack, so
    between boot and the first pp->tp cutover no TP forward and no draft
    forward happens -- the workspaces are untouched memory and the captured
    graphs are unreplayed. Measured on this rig they are 2294 / 1188 / 625 MiB
    (``arena_tail_probe.STACK_RESIDUAL_MIB``), and on the binding rank that is
    the whole distance between the pool this cut solves and the 669k plain-TP
    reference.

    THE ORDER IS THE BOOT'S ORDER AND IS LOAD-BEARING. Backends on both
    workers before graphs on either: the target's capture drives the drafter,
    so the drafter's backend must exist before the target captures.

    IDEMPOTENT BY CONSTRUCTION, and that is the point of routing both sites
    through here. The deferral's failure mode is not "never built" -- that
    fails loudly on the first decode -- but "built twice", which allocates a
    second set of workspaces and a second capture, silently, and only shows up
    as a pool that can no longer back itself.

    #785/#791 PART C: a NAMED REFUSAL GUARD runs first
    (``_guard_geometry_before_backend_build``), so a caller that reaches this
    function with no ``phase_flip_tp_scope`` open around it -- the exact bug
    part A fixed at the ``restore_deferred_cold_stack`` call site -- fails
    loudly here instead of silently refolding KV rows on the first decode.
    """
    if getattr(tp_worker, COLD_STACK_BUILT_ATTR, False):
        logger.info(
            "%s cold stack posts already built; %s is a no-op", LOG_PREFIX, where
        )
        return False

    _guard_geometry_before_backend_build(tp_worker, where)
    tp_worker.init_attention_backends()
    if draft_worker is not None:
        draft_worker.init_attention_backends()
    tp_worker.init_cuda_graphs()
    if draft_worker is not None:
        draft_worker.init_cuda_graphs()

    # THE CARRIER PIN (#656 rung 2), checked AFTER capture -- and it travels
    # with the capture rather than staying at the boot site.
    #
    # Capture is the instant the drafter's parameter addresses stop being
    # movable. Asserting here is what catches the failure this ordering exists
    # to prevent: something between the carrier pack (4c) and the capture
    # reallocating a draft parameter, so the graphs bake an address the
    # carrier does not own and the spill later releases pages the graphs still
    # read. There is no runtime symptom to catch that later; the drafter
    # simply produces wrong tokens and the accept rate decays.
    #
    # Left BEFORE the built-flag is set: a refusal must not latch, or a retry
    # would skip the capture and run a drafter with no graphs at all.
    if draft_carrier is not None:
        if not draft_carrier.contains_all_params():
            raise PhaseFlipBootError(
                "PHASE-FLIP-SPILL the draft CUDA graphs were captured "
                "against parameter addresses that do NOT all lie "
                "inside the spill carrier's reservation. Something "
                "between the carrier pack (4c) and the capture "
                f"(at {where}) reallocated a draft parameter. Spilling "
                "would then release pages the graphs still address, which "
                "is silent corruption -- refusing the boot instead."
            )
        logger.info(
            "%s carrier pin OK at %s: all %d draft parameters lie inside the "
            "VA-stable reservation after graph capture; rung 2 may "
            "release their pages without moving an address",
            LOG_PREFIX,
            where,
            len(draft_carrier.param_ptrs()),
        )

    setattr(tp_worker, COLD_STACK_BUILT_ATTR, True)
    logger.info("%s cold stack posts built at %s", LOG_PREFIX, where)
    return True


def restore_deferred_cold_stack(scheduler, stacks) -> bool:
    """pp->tp leg: build the cold posts this instance's boot left out.

    Returns True if this call built them.

    WHERE THIS SITS AND WHY. Called from the cutover's tp_phase branch, AFTER
    rung 2 has restored the draft weights and BEFORE the draft bootstrap. Both
    neighbours are load-bearing:

    * after the weights, because the capture bakes parameter addresses and
      there must be pages under them to bake;
    * before the bootstrap, because ``arm_draft_bootstrap_all_reachable``
      scrubs the drafter's pool and the graphs must exist to be armed.

    THE BYTES ARE ALREADY PAID FOR when control reaches here. The pp->tp
    affordability verdict prices this build (see ``_staging_bytes``), so a
    rank that got this far has agreed it can afford them; the KV pool is the
    relief provider that funds it (``kv_backing_relief`` /
    ``recover_kv_backing``). That is the whole reason the boot was allowed to
    size a pool as if these bytes were absent.

    ONCE PER PROCESS, not once per flip. ``build_cold_stack_posts`` latches,
    so the second and every later pp->tp leg costs nothing. A per-flip
    re-capture is the trade #656 spec item 8 measured at 41% of decode
    throughput and rejected, and this must not reintroduce it by accident.

    #785/#791 PART A: THE BUILD MUST HAPPEN INSIDE ``phase_flip_tp_scope``,
    exactly like the (non-deferred) boot-time call at rung < 4. The two
    calls are NOT symmetric by default: the boot-time call sits inside the
    scope opened by ``build_phase_flip_tp_stack`` for its entire construction,
    but this cutover-time call runs from ``phase_flip_runtime`` with no scope
    of its own. The attention backends read the TP geometry FRESH at
    construction time from ``get_parallel()`` (contextvars --
    ``TritonAttnBackend.__init__`` reads ``attn_dcp_size`` / ``attn_tp_size``
    there, and derives ``uneven_dcp`` from
    ``uneven_dcp_kv_replicated(dcp_size)``), while the KV pool's row width is
    a BAKED ``ModelRunner`` attribute fixed once at boot and therefore
    scope-independent. Left unscoped here, ``get_parallel().attn_dcp_size``
    would read the ambient PRIMARY topology (pp=N, tp=1, dcp disabled) instead
    of the flip TP size the pool was actually sized under: ``uneven_dcp``
    comes up False, ``_set_kv_buffer`` skips ``_dcp_write_gather``, and the
    store kernel is handed the raw per-rank k/v (e.g. 2/1/1 KV heads across
    ranks) instead of the gathered full-head replicated row the pool's
    ``row_dim`` was baked for -- silently refolded rows, observed as
    "store_cache rejected: k(24, 2, 256) v(24, 2, 256) row_dim=1024 ->
    k_rows=12, k_cache(201377, 4, 256) ... size_limit=201377" on rank 0
    (and the 1-head variant on ranks 1/2). ``n`` and ``world_rank`` are
    recovered the same way ``build_phase_flip_tp_stack`` derives them.
    """
    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.managers import phase_flip_spill as spill

    if not spill.cold_stack_deferred(getattr(scheduler, "server_args", None)):
        return False
    draft_worker = getattr(stacks, "draft_worker", None)
    carrier = spill.carrier_of(draft_worker) if draft_worker is not None else None
    n = len(stacks.vector)
    world_rank = get_world_group().rank_in_group
    with phase_flip_tp_scope(world_rank, n):
        return build_cold_stack_posts(
            stacks.tp_worker, draft_worker, carrier, where="pp->tp cutover"
        )


def build_phase_flip_tp_stack(scheduler) -> PhaseFlipStacks:
    """Build the TP decode stack beside the fully-constructed PP stack.

    Called from Scheduler.init_model_worker AFTER the primary worker's
    pools, backends and (eager-only, pin 2) graph init, BEFORE the
    post-capture pool resize -- the resize must see the TP stack's VRAM
    as taken, not as free. Rank-uniform: every collective below runs on
    every rank in the same order."""
    from sglang.srt.distributed.parallel_state import (
        get_world_group,
        phase_flip_groups_initialized,
    )
    from sglang.srt.distributed.utils import (
        set_cp_token_ratios,
        set_tp_partition_ratios,
    )
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.runtime_context import get_context, get_server_args

    server_args = scheduler.server_args
    if not server_args.enable_phase_flip:
        raise PhaseFlipBootError("build_phase_flip_tp_stack without the flag")
    if not phase_flip_groups_initialized():
        raise PhaseFlipBootError(
            "phase-flip secondary groups were not built at primary init "
            "(initialize_phase_flip_secondary_groups)"
        )
    if not server_args.uneven_weighted_dcp_enabled():
        raise PhaseFlipBootError(
            "the flip's TP phase token-shards KV under the WEIGHTED owner "
            "rule; set SGLANG_UNEVEN_DCP=1 and SGLANG_UNEVEN_DCP_WEIGHTED=1 "
            "(the rig runbook's uneven-DCP env pair) -- refusing to build a "
            "stack whose owner rule would silently fall back to even-modulo"
        )
    vec = parse_flip_vector(server_args)
    tok_vec = parse_flip_token_vector(server_args)
    n = len(vec)
    world_rank = get_world_group().rank_in_group
    primary_runner = scheduler.tp_worker.model_runner
    device = primary_runner.device

    # 1. Process-global uneven plan + token vector. Installed only NOW --
    # the PP stack was built with both absent (byte-identical primary
    # path) and its runtime reads are gated on its own cached tp_size=1 /
    # dcp_size=1. Step-6 watch item: any PP-phase code path that consults
    # the global plan with a non-rank-local size would be a bug HERE.
    # The weight shard follows compute; the token split follows what each
    # rank has left after its weights land. Same vector unless an operator
    # overrides the token side (see parse_flip_token_vector).
    set_tp_partition_ratios(list(vec), families=None)
    set_cp_token_ratios(list(tok_vec))
    if tok_vec != vec:
        logger.info(
            "%s KV token split %s differs from the weight shard split %s "
            "(SGLANG_UNEVEN_TOKEN_VECTOR). The weight shard follows "
            "compute; the token split follows each rank's memory left "
            "after its weights land.",
            LOG_PREFIX,
            tok_vec,
            vec,
        )

    # #1078: BEFORE anything is allocated. The two-file arm keeps both layout
    # images for the life of the process, which is only affordable when they
    # are file-backed (reclaimable page cache) rather than pinned. Refusing
    # here means a misconfiguration costs a boot message, not the OOM kill W26
    # took in the LAUNCH phase.
    require_two_file_preconditions()
    two_file = two_file_images_enabled()

    # 2. Snapshot the PP checkpoint weights to host, free device originals
    # (VRAM ledger: PP originals + TP originals + arena never coexist).
    pp_named = checkpoint_param_dict(primary_runner.model)
    layout_pp = plan_arena_layout(pp_named)
    image_pp = snapshot_and_free(pp_named, layout_pp, pin=True)
    if two_file:
        # Its own exact-sized file already -- `snapshot_and_free` without
        # `out=` allocates one. Tagging is what makes a leg handed the wrong
        # image refuse structurally instead of relying on the two layouts
        # happening to differ in size.
        tag_layout_image(image_pp, "pp")
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Build the TP-shaped worker under the flip scope. The server-args
    # copy is PUBLISHED for the whole build (the #470 lesson: context
    # readers must see the stack's own geometry, not the target's).
    tp_args = derive_tp_stack_server_args(
        server_args, pp_id_space=int(primary_runner.max_total_num_tokens)
    )
    logger.info(
        "%s TP pool sized to the PP id space: %d tokens (no --max-total-tokens "
        "needed; the surplus a self-sized TP pool would take is unaddressable)",
        LOG_PREFIX,
        int(primary_runner.max_total_num_tokens),
    )
    ctx = get_context()
    saved_args = get_server_args()
    ctx.set_server_args(tp_args)
    try:
        with phase_flip_tp_scope(world_rank, n):
            # NO pool sharing at construction: the TP stack builds its OWN
            # HybridReqToTokenPool so its mamba pool gets the TP geometry
            # (ALL linear layers, head-sharded) through the normal path --
            # the shared PP req pool carries a PP-shaped mamba pool (stage
            # layers, full heads), which would silently mis-shape every
            # linear-state access. Request-mapping SHARING happens below by
            # tensor rebind instead.
            tp_worker = TpModelWorker(
                server_args=tp_args,
                gpu_id=scheduler.ps.gpu_id,
                tp_rank=world_rank,
                moe_ep_rank=0,
                pp_rank=0,
                attn_cp_rank=scheduler.tp_worker.attn_cp_rank,
                moe_dp_rank=scheduler.tp_worker.moe_dp_rank,
                dp_rank=scheduler.tp_worker.dp_rank,
                nccl_port=scheduler.tp_worker.nccl_port,
                is_draft_worker=True,
                is_phase_flip_tp_stack=True,
            )

            # 4. Arena: sized max(both layouts), fixed for process life.
            #
            # SNAPSHOT-THEN-ALLOCATE, exactly as step 2 does for the PP
            # layout. Packing the live TP originals straight into a fresh
            # arena holds BOTH at once -- originals + arena -- and that
            # transient, not any runtime shape, is what sets the per-rank
            # VRAM budget for the whole process life. Measured on this rig
            # before the reorder (torch allocated_peak_bytes per rank):
            #
            #   rank 0 (5090):  14.70 GiB weights + 14936 MiB arena = 29.27
            #   rank 1 (3080a):  8.91 GiB weights +  7924 MiB arena = 16.64
            #   rank 2 (3080b):  8.91 GiB weights +  9115 MiB arena = 17.81
            #
            # The peak never recurs at runtime -- serving at
            # max_running_requests 4 / chunked_prefill_size 2048 stays
            # ~7 GiB below it -- but because the corridor floor is a
            # CONTINUOUS minimum, the boot spike alone forced
            # --rank-gpu-memory-mib down to roughly half of each card, and
            # every MiB of it was permanent KV pool given away.
            #
            # Snapshotting first makes the peak max(originals, arena)
            # instead of their sum. It costs no extra host RAM: the host
            # image was already built one line later (arena_image), so
            # this is a reordering, not a new allocation. image_from_tensors
            # exists precisely for this and says so in its docstring.
            tp_named = checkpoint_param_dict(tp_worker.model_runner.model)
            layout_tp = plan_arena_layout(tp_named)
            # #785 GATE: this is the first moment in the boot where the
            # MEASURED layout_tp exists, and the pool was sized minutes ago
            # against a derivation of it. Grade the two here, against each
            # other, on one boot's own numbers -- a derivation checked against
            # a remembered rig proves nothing about the rig it ran on.
            _grade_arena_tail_derivation(
                primary_runner, world_rank, layout_pp, layout_tp
            )
            # #809/W28: THE ONE HOST IMAGE. Both layouts are measured by now,
            # which is the first moment the max-sized buffer CAN be sized --
            # the PP snapshot above necessarily predates layout_tp. From here
            # on there is exactly one lifetime image; `image_pp` is a boot
            # transient and is released once the arena carries the PP layout.
            if two_file:
                # #1078: NO max-sized rotation buffer. Each layout gets its own
                # exact-sized file, so a leg reads the incoming layout from its
                # own file and discards the outgoing arena content -- there is
                # no copy-back, hence nothing for a shared buffer to receive.
                # The size asymmetry that the rotation's overshoot existed to
                # cover (rotation_plan.py:51-59) stops being a term at all.
                image_tp = snapshot_and_free(tp_named, layout_tp, pin=True)
                tag_layout_image(image_tp, "tp")
                rotation_image = image_tp
            else:
                rotation_image = allocate_rotation_image(
                    layout_pp.total_bytes, layout_tp.total_bytes, pin=True
                )
                image_tp = snapshot_and_free(
                    tp_named, layout_tp, pin=True, out=rotation_image
                )
            if device == "cuda":
                torch.cuda.empty_cache()
            arena_total = max(layout_pp.total_bytes, layout_tp.total_bytes)
            # RUNG 3: put the arena on a VA-stable reservation so its tail can
            # be released in the phase that does not reach it. Opt-in, so the
            # default path allocates exactly as it always did.
            from sglang.srt.managers.phase_flip_spill import (
                DEPTH_ARENA_TAIL,
                VmmWeightsArenaCarrier,
                resolve_spill_depth,
            )

            arena_carrier = None
            if (
                resolve_spill_depth(getattr(scheduler, "server_args", None))
                >= DEPTH_ARENA_TAIL
                and device == "cuda"
            ):
                arena_carrier = VmmWeightsArenaCarrier(
                    int(tp_worker.model_runner.gpu_id), arena_total
                )
                arena = arena_carrier.tensor
            else:
                arena = allocate_arena(arena_total, tp_worker.model_runner.device)
            # Pure rebind, then one contiguous refill from the host image;
            # arena_refill verifies the checksum on the arena's device
            # after the copy.
            bind_arena_views(layout_tp, arena, rebind=list(tp_named.items()))
            prime_arena_from_image(arena, layout_tp, image_tp)

            # 4b. The TP-phase draft worker (#631 speculation slice).
            # Constructed AFTER the arena is packed, mirroring the boot
            # order (maybe_init_draft_worker precedes the target's
            # alloc_memory_pool) and keeping the arena's one big
            # contiguous allocation away from the draft's. The draft's
            # weights are its OWN model and stay resident across both
            # phases -- they are not arena-backed, because there is no
            # second layout for them to flip between.
            draft_worker = build_flip_draft_worker(
                scheduler, tp_worker, tp_args, world_rank
            )

            # 4c. THE DRAFT-WEIGHT CARRIER (#656 spec item 6, spill rung 2).
            #
            # THIS PLACEMENT IS THE WHOLE CORRECTNESS ARGUMENT. It must be
            # after the draft worker exists and BEFORE its CUDA graphs are
            # captured at step 5, because capture bakes the drafter's
            # parameter addresses into the graphs. Pack afterwards and the
            # graphs address the pre-pack storages, which the pack has
            # already freed -- and that corruption is SILENT: no exception,
            # just wrong draft logits and a quietly collapsing accept rate.
            # The assertion after step 5 exists to make a future reordering
            # of these lines fail loudly instead.
            #
            # The comment above (4b) used to justify leaving these weights
            # un-arena-backed with "there is no second layout for them to
            # flip between". Rung 2 falsifies the premise rather than the
            # reasoning: a spill needs no second layout, only a host image
            # and an empty device.
            # Imported locally: phase_flip_spill reaches back into this module
            # for checkpoint_param_dict/snapshot_and_free, and a module-level
            # import here would close that loop at import time.
            from sglang.srt.managers.phase_flip_spill import (
                DEPTH_DRAFT_WEIGHTS,
                install_draft_weight_carrier,
                resolve_spill_depth,
            )

            draft_spill_depth = resolve_spill_depth(
                getattr(scheduler, "server_args", None)
            )
            draft_carrier = None
            if draft_spill_depth >= DEPTH_DRAFT_WEIGHTS:
                draft_carrier = install_draft_weight_carrier(
                    draft_worker,
                    tp_worker.model_runner.gpu_id,
                    server_args=getattr(scheduler, "server_args", None),
                )

            # 5. TP pools + backends + decode graphs, with the TP arena
            # bytes live (graphs bake the fixed arena addresses; pin 2:
            # ONLY this stack captures decode graphs).
            #
            # BOOT-TIME EXCLUSIVE BACKING (#631). The PP pool's physical
            # pages come out HERE, before the TP stack allocates its own
            # pools and captures its decode graphs, and go back in once the
            # TP pages are released again. Without this the boot holds both
            # layouts' KV at once, and because the corridor floor is a
            # CONTINUOUS minimum that one peak -- not any runtime shape --
            # is what caps --rank-gpu-memory-mib for the whole process
            # life. Peak becomes max(PP, TP) instead of PP + TP.
            #
            # Safe on the PP side by construction: the PP stack captures NO
            # cuda graphs (model_runner logs "PHASE-FLIP PP stack: no CUDA
            # graphs captured by construction"), so no baked address can go
            # stale, and nothing is serving yet at boot. Safe on the TP side
            # because the release is a VMM unmap behind a fixed VA
            # reservation: the graphs captured below bake addresses that
            # never move.
            pp_kv_pool = primary_runner.token_to_kv_pool
            _swappable = hasattr(pp_kv_pool, "release_backing") and getattr(
                pp_kv_pool, "backing_is_resident", False
            )
            if _swappable:
                released = pp_kv_pool.release_backing()
                if device == "cuda":
                    torch.cuda.empty_cache()
                logger.info(
                    "%s released the PP KV backing (%.2f MiB) for the TP "
                    "stack's allocation and graph capture; boot peak is "
                    "max(PP, TP), not PP + TP.",
                    LOG_PREFIX,
                    released / 1048576.0,
                )

            tp_worker.alloc_memory_pool()

            # 5-guard. THE SLOT-ID SPACE MUST FIT BOTH POOLS.
            #
            # The scheduler keeps ONE allocator for process life -- the PP
            # stack's, built by Scheduler.build_kv_cache before this stack
            # exists and never swapped at cutover. That is not an oversight:
            # the flip's transition maps GLOBAL slot ids to each layout's
            # physical rows, so a single id space is what makes a row
            # identifiable across the flip at all.
            #
            # The consequence is an invariant nothing was checking: every id
            # the allocator can issue must be addressable in BOTH layouts.
            # The TP stack derives its own capacity from its own budget and
            # token vector, so it can come out SMALLER than the id space --
            # and then the first decode that touches a high id runs off the
            # end of the TP pool.
            #
            # That is not a graceful failure. It surfaces inside
            # store_kvcache as SGL_DEVICE_ASSERT(index >= 0 && index <
            # size_limit), a device-side assert that takes down all three
            # ranks with an async CUDA error whose traceback points at
            # whatever host call happened to synchronise next. Observed on
            # this rig: PP/allocator C = 46422 against TP C = 27200, which
            # died mid-benchmark exactly as described.
            #
            # Checked here, at boot, where both numbers are on hand and the
            # message can name them. The comparison is deterministic across
            # ranks -- both capacities are group-consistent, each having been
            # min-reduced over the world group by _apply_token_constraints --
            # so a local raise aborts the whole boot identically on every
            # rank rather than half of it.
            pp_capacity = int(primary_runner.max_total_num_tokens)
            tp_capacity = int(tp_worker.model_runner.max_total_num_tokens)
            if tp_capacity < pp_capacity:
                raise PhaseFlipBootError(
                    f"the TP decode stack addresses {tp_capacity} tokens but "
                    f"the scheduler's allocator issues slot ids up to "
                    f"{pp_capacity} (the PP stack's capacity, which is the "
                    f"process-wide id space). Ids above {tp_capacity} would "
                    f"be written past the end of the TP KV pool and abort "
                    f"every rank inside store_kvcache's bounds assert on the "
                    f"first decode that reaches one. Raise the TP stack's "
                    f"capacity (SGLANG_UNEVEN_TOKEN_VECTOR matched to the "
                    f"per-rank profiled capacities, or a larger "
                    f"--rank-gpu-memory-mib) until it is >= {pp_capacity}."
                )

            # 5a. Share the REQUEST MAPPINGS by tensor rebind: both stacks
            # must read the same request->token rows and request->mamba
            # slot mapping (the scheduler writes them ONCE, into the
            # primary pool's tensors; slot ids are the cross-layout keys).
            # The pools themselves stay layout-specific; only the mapping
            # tensors alias. The slot SPACES agree by construction (same
            # max_num_reqs / max_mamba_cache_size in the args copy) --
            # asserted here, loudly.
            pp_req_pool = primary_runner.req_to_token_pool
            tp_req_pool = tp_worker.model_runner.req_to_token_pool
            if tp_req_pool.req_to_token.shape != pp_req_pool.req_to_token.shape:
                raise PhaseFlipBootError(
                    f"req_to_token shapes diverge between stacks: PP "
                    f"{tuple(pp_req_pool.req_to_token.shape)} vs TP "
                    f"{tuple(tp_req_pool.req_to_token.shape)}"
                )
            tp_req_pool.req_to_token = pp_req_pool.req_to_token
            if hasattr(pp_req_pool, "req_index_to_mamba_index_mapping"):
                pp_map = pp_req_pool.req_index_to_mamba_index_mapping
                tp_map = tp_req_pool.req_index_to_mamba_index_mapping
                if tp_map.shape != pp_map.shape:
                    raise PhaseFlipBootError(
                        f"mamba index mapping shapes diverge: PP "
                        f"{tuple(pp_map.shape)} vs TP {tuple(tp_map.shape)}"
                    )
                tp_req_pool.req_index_to_mamba_index_mapping = pp_map
                if pp_req_pool.mamba_pool.size != tp_req_pool.mamba_pool.size:
                    raise PhaseFlipBootError(
                        f"mamba slot spaces diverge: PP "
                        f"{pp_req_pool.mamba_pool.size} vs TP "
                        f"{tp_req_pool.mamba_pool.size} -- the flip relies "
                        f"on slot-id identity across layouts"
                    )

            # 5b. Draft pools/backends/graphs, in the boot's order and
            # from the boot's source of truth: the draft shares the TARGET
            # stack's request pool and KV allocator (Scheduler
            # .init_memory_pools does exactly this), so the draft KV is
            # sized rank-locally inside the TP stack's already-profiled
            # budget rather than profiling a second time against free
            # memory that the two resident pools have already claimed.
            if draft_worker is not None:
                pool, allocator = tp_worker.get_memory_pool()
                draft_worker.alloc_memory_pool(
                    memory_pool_config=tp_worker.model_runner.memory_pool_config,
                    req_to_token_pool=pool,
                    token_to_kv_pool_allocator=allocator,
                )

            # 5b. THE COLD POSTS (spill rung 4), built HERE or deferred to the
            # first pp->tp cutover -- see build_cold_stack_posts, and note that
            # the carrier pin moved INTO it because the pin belongs to the
            # capture, wherever the capture happens.
            #
            # The predicate is the ladder's, not a local one: the KV sizer
            # asked the SAME function before it solved the pool. If these two
            # sites could disagree, the disagreement that matters is a pool
            # sized for a deferral this boot then declines to perform.
            from sglang.srt.managers.phase_flip_spill import cold_stack_deferred

            if cold_stack_deferred(getattr(scheduler, "server_args", None)):
                logger.info(
                    "%s rung 4: DEFERRING the flip TP stack's cold posts (the "
                    "attention-backend workspaces and the decode CUDA graphs) "
                    "to the first pp->tp cutover. The PP phase this boot is "
                    "about to enter cannot execute a TP or draft forward, so "
                    "these bytes would be resident and unusable for the whole "
                    "phase that sizes the KV pool. The KV budget was solved "
                    "with the matching credit already taken.",
                    LOG_PREFIX,
                )
            else:
                build_cold_stack_posts(
                    tp_worker, draft_worker, draft_carrier, where="boot"
                )

            # 5c. EXCLUSIVE-BACKING PIN, measured rather than intended.
            # Between the release above and the restore below exactly one
            # layout may hold pages. Asserting it here -- with the TP pool
            # allocated -- is what makes "exclusive" a checked property
            # instead of a claim about the code's shape.
            #
            # THIS IS THE WORST MOMENT ONLY AT RUNGS BELOW 4. It used to be,
            # unconditionally, because the graphs were captured just above.
            # Under the rung-4 deferral they are not, so the true residency
            # peak moves to the first pp->tp cutover -- which is exactly why
            # that build is priced into the seam's affordability verdict
            # (_cold_stack_restore_bytes) instead of being checked here.
            # What this pin still asserts is unchanged and still exact: the
            # two KV layouts never hold pages at the same time.
            tp_kv_pool = tp_worker.model_runner.token_to_kv_pool
            if _swappable:
                pp_backed = int(getattr(pp_kv_pool, "backed_bytes", 0))
                tp_backed = int(getattr(tp_kv_pool, "backed_bytes", 0))
                if pp_kv_pool.backing_is_resident:
                    raise PhaseFlipBootError(
                        f"PP KV backing is resident while the TP stack holds "
                        f"its own ({pp_backed} B PP, {tp_backed} B TP). The "
                        f"boot peak is then PP + TP, which is the residency "
                        f"this sequence exists to remove."
                    )
                # The floor left behind is one page plus at most one commit
                # chunk per buffer; anything near the TP pool's size means
                # the release did not actually reach the driver.
                if tp_backed and pp_backed > tp_backed // 8:
                    raise PhaseFlipBootError(
                        f"the PP KV release returned only down to "
                        f"{pp_backed} B against the TP pool's {tp_backed} B "
                        f"-- the pages were not handed back to the driver, "
                        f"so the boot peak is still both layouts."
                    )
    finally:
        ctx.set_server_args(saved_args)

    # 5d. Hand the pages back to the boot phase. The TP stack is fully
    # built and captured; from here until the first pp_to_tp flip only the
    # PP layout runs, so only it may hold pages. Ordering is load-bearing:
    # the PP pool must be backed again BEFORE anything can prefill into it,
    # which is why this sits inside the boot path and not on a lazy path.
    if _swappable:
        tp_kv_pool = tp_worker.model_runner.token_to_kv_pool
        tp_released = tp_kv_pool.release_backing()
        if device == "cuda":
            torch.cuda.empty_cache()
        primary_runner.token_to_kv_pool.restore_backing()
        if not primary_runner.token_to_kv_pool.backing_is_resident:
            raise PhaseFlipBootError(
                "the PP KV backing did not come back after the TP stack was "
                "built; the boot phase is PP and would prefill into unmapped "
                "pages"
            )
        logger.info(
            "%s boot-time backing swap done: TP released (%.2f MiB), PP "
            "restored. Exactly one layout is resident from here on.",
            LOG_PREFIX,
            tp_released / 1048576.0,
        )

    # 6. Pin 3 on the real pools.
    assert_row_schema_compatible(
        primary_runner.token_to_kv_pool,
        tp_worker.model_runner.token_to_kv_pool,
    )

    # 7. PP is the boot phase: rebind its params to arena views and refill
    # the arena with the PP image (one contiguous H2D).
    bind_arena_views(layout_pp, arena, rebind=list(pp_named.items()))
    prime_arena_from_image(arena, layout_pp, image_pp)
    # #809/W28: THE BOOT TRANSIENT ENDS HERE. The arena now carries the PP
    # layout, so the PP host image has no reader left: the resting layout is TP
    # and it lives in `rotation_image`. Releasing it is what turns the boot's
    # two-image peak into a one-image steady state -- keeping it would be the
    # dual pin W26 OOM-killed, merely renamed.
    # UNREGISTER BEFORE FREEING. The pages are cudaHostRegister'd; returning
    # them to the allocator while CUDA still maps them makes the next big host
    # allocation fail with rc=712 (W28 attempt 1 died exactly there, in
    # HiCache's 5.6 GB KV buffer).
    #
    # #1078: THE TWO-FILE ARM KEEPS IT, and the sentence above is why that is
    # allowed only here. "Keeping it would be the dual pin W26 OOM-killed"
    # holds for PINNED images and for no others: two pinned lifetime images
    # are 55.99 GiB across this rig's three ranks. A file-backed image is
    # reclaimable page cache and not a pinned post at all, so keeping both
    # costs +27.15 GiB of DISK (501 GiB free) and no locked RAM. That is the
    # whole trade, and `require_two_file_preconditions` at the top of this
    # function is what stops it from being taken under the pinned allocator.
    if not two_file:
        release_host_image(image_pp)
        del image_pp

    # The mode qualifier keeps this line honest for the host ledger: a
    # file-backed image (SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED) is reclaimable
    # page cache, and calling it "pinned" here is exactly how a ledger learns
    # to charge bytes that are not actually locked.
    logger.info(
        "%s TP stack built: vector %s, arena %.2f MiB (pp %.2f / tp %.2f), "
        "images %.2f MiB host (%s)",
        LOG_PREFIX,
        vec,
        arena.numel() / 1048576.0,
        layout_pp.total_bytes / 1048576.0,
        layout_tp.total_bytes / 1048576.0,
        rotation_image.numel() / 1048576.0,
        host_image_mode(),
    )
    # #797: the vector the decode phase actually runs under is the one the TP
    # pools were just built with, which is not necessarily the seed parsed at
    # the top of this function -- the worker construction above may have
    # installed the measured optimum. Read it back rather than assuming it.
    token_verdict = effective_flip_token_vector(server_args, tok_vec)

    return PhaseFlipStacks(
        tp_worker=tp_worker,
        arena=arena,
        layout_pp=layout_pp,
        layout_tp=layout_tp,
        rotation_image=rotation_image,
        image_holds="tp",
        vector=tuple(vec),
        token_vector=token_verdict.vector,
        draft_worker=draft_worker,
        arena_carrier=arena_carrier,
        image_pp=image_pp if two_file else None,
        image_tp=image_tp if two_file else None,
    )


#: #847/#810: how much of a phase's KV view the staging pin has to hold.
#:
#: A STAGING PIN, NOT A MIRROR. #810 draws the line: a `retention` host tier is
#: sized as a ratio of the device pool and exists to KEEP prefixes; a `staging`
#: tier holds only what is in flight and is sized to the work, never to the
#: pool. The rebind needs the second kind. What it must stage is the seam's own
#: re-admission -- the requests one cutover retracts, read back through in the
#: layout it flips into -- so the pin is sized to a re-admission batch and to
#: nothing else: `max_running_requests` requests of one chunk each.
#:
#: A ratio-sized second pool would be the wrong answer twice over: it would
#: duplicate retention the pp-side tier already provides, and on this box the
#: pinned host budget is the binding constraint (DESIGN_706 C1), so it would be
#: charged against the 16 GiB floor for capacity nothing reads.
PHASE_FLIP_STAGING_CHUNKS = 1


def host_tier_of(tree):
    """The HiCache host pool behind ``tree``, whichever route it keeps it on.

    W33 arm 2, AND IT IS THE W29 DEFECT REPEATED BY ME. This read used to be a
    bare ``getattr(tree, "token_to_kv_pool_host", None)``. That attribute is
    ``HiRadixCache``'s; the tree this box actually runs is
    ``UnifiedRadixCache``, which does not have it at all -- it reaches the host
    tier through ``cache_controller.mem_pool_host``. So the read returned
    ``None`` on the live tree, the writer logged its own "no HiCache host tier"
    refusal on every rank, and the rebind could not arm: mechanism installed,
    unreachable.

    That is exactly the shape rooted at W29, when
    ``drop_prefix_tree_returning_rows`` read ``full_evictable_size_`` -- an
    attribute three of the tree types have and ``UnifiedRadixCache`` does not
    -- and ``getattr(..., 0)`` turned the absence into a number that silently
    disabled the eviction. Same family, same tree class, same silent default,
    written by me one strand later.

    So the accessor is NAMED and knows BOTH routes, and a drift-detector test
    asserts every shipped cache is reachable through it. ``None`` here means
    genuinely no host tier -- a real state, reported loudly by the caller --
    never "this tree keeps it somewhere I did not look".
    """
    if tree is None:
        return None
    direct = getattr(tree, "token_to_kv_pool_host", None)
    if direct is not None:
        return direct
    controller = getattr(tree, "cache_controller", None)
    return getattr(controller, "mem_pool_host", None)


def _staging_pin_gib(scheduler, device_pool, fallback_pool=None) -> float:
    """Bytes the phase-matched staging pin needs, in GiB, from measured cells.

    Derived, never guessed: the pool's own per-token cell size times the tokens
    a re-admission batch can present. Returns a float so the caller can ledger
    the real number and round only once, at the allocation boundary.
    """
    sa = scheduler.server_args
    chunk = int(getattr(sa, "chunked_prefill_size", 0) or 0) or 4096
    conc = int(getattr(sa, "max_running_requests", 0) or 0) or 1
    tokens = chunk * conc * PHASE_FLIP_STAGING_CHUNKS
    cell = int(getattr(device_pool, "get_kv_size_per_token", lambda: 0)() or 0)
    if cell <= 0:
        cell = int(getattr(device_pool, "cell_size", 0) or 0)
    if cell <= 0 and fallback_pool is not None:
        # W34 arm 1 printed "0.000 GiB -> 1 GB": neither probe answered on the
        # live TP pool, so the derived size collapsed to zero and only the
        # `max(1, ...)` floor kept it allocatable. A pin sized from nothing is
        # not a derivation. The pp-side tier reports the same per-token cost
        # (both phases hold the same token rows), so it is the honest fallback
        # -- and it is a FALLBACK, named, not the primary reading.
        cell = int(getattr(fallback_pool, "size_per_token", 0) or 0)
    return (tokens * cell) / float(1 << 30)


def _hybrid_pin_entries(*, tp_runner, sa, kv_host, inner_pool, tp_device_pool, logger):
    """#871: the KV+MAMBA entry pair for the 'tp' staging pin, or None.

    MIRRORS ``build_hybrid_mamba_stack`` (hybrid_pool_assembler.py) rather than
    re-deriving it: same primitives, same layer mappings, same
    ``transfer_layer_num`` rule. That assembler is the contract for what a
    kv+mamba ``HostPoolGroup`` looks like, and a second hand-rolled opinion
    about it is exactly the drift #847 warned about when it refused to clone
    ``type(pp_host)``.

    WHAT IS DELIBERATELY NOT REUSED: ``build_hybrid_mamba_stack`` itself, because
    it also constructs a ``HybridCacheController``. This pin needs a host VIEW to
    rebind readers onto; a second controller would be a second writer against the
    same device pool. Entries yes, controller no.

    SIZING IS PER-SLOT, NOT PER-GB, and that is the one place this must NOT copy
    the KV half. The KV pin is sized in bytes from a token count
    (``_staging_pin_gib``). Mamba state is allocated per REQUEST slot, and
    ``MambaPoolHost`` reads ``host_size`` in GB only when it is > 0, otherwise
    ``int(device_pool.size * host_to_device_ratio)``. Passing the KV pin's GB
    figure here would size the mamba view by a budget derived for a different
    unit. Ratio 1.0 with ``host_size=0`` mirrors the device pool exactly, which
    is what "phase-matched staging pin" means: able to stage what the other
    phase can actually hold, and no more.

    Returns ``None`` when the TP side cannot supply the mamba handles. The
    caller then keeps today's KV-only pin, ``check_pool_coverage`` keeps
    refusing, and the operator gets a named reason -- the current state
    preserved rather than a narrowed tier smuggled past a guard.
    """
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        build_pool_entry,
    )
    from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost

    req_pool = getattr(tp_runner, "req_to_token_pool", None)
    mamba_pool = getattr(req_pool, "mamba_pool", None)
    mamba_map = getattr(req_pool, "mamba_map", None)
    mamba_allocator = getattr(req_pool, "mamba_allocator", None)
    missing = [
        n
        for n, v in (
            ("req_to_token_pool", req_pool),
            ("mamba_pool", mamba_pool),
            ("mamba_map", mamba_map),
            ("mamba_allocator", mamba_allocator),
        )
        if v is None
    ]
    if missing:
        logger.error(
            "#871 PHASE-FLIP REBIND: the bound tier describes MAMBA but the TP "
            "phase stack does not expose %s, so the staging pin cannot mirror "
            "it. Keeping the KV-only pin: the rebind will keep refusing on "
            "coverage, which is the correct state, and this line is the reason.",
            ", ".join(missing),
        )
        return None

    # The pool's OWN mapping, exactly as `_MambaStrategy` reads it.
    full_map = dict(getattr(tp_device_pool, "full_attention_layer_id_mapping", {}) or {})
    mamba_map = dict(mamba_map)
    if not full_map:
        logger.error(
            "#871 PHASE-FLIP REBIND: the TP device pool exposes no "
            "full_attention_layer_id_mapping, so the KV half of a hybrid pin "
            "cannot be given the transfer indices the mamba half must not "
            "collide with. Keeping the KV-only pin."
        )
        return None
    transfer_layer_num = len(full_map | mamba_map)

    mamba_host = MambaPoolHost(
        mamba_pool,
        1.0,  # host_to_device_ratio: mirror the device pool, see docstring
        0,  # host_size GB: 0 selects the ratio path
        allocator_type=getattr(sa, "hicache_storage_backend", "default") or "default",
        layout=getattr(sa, "hicache_mem_layout", "layer_first") or "layer_first",
    )
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host,
            device_pool=inner_pool,
            layer_mapping=full_map,
            transfer_layer_num=transfer_layer_num,
            is_anchor=True,
        ),
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host,
            device_pool=mamba_pool,
            layer_mapping=mamba_map,
            transfer_layer_num=transfer_layer_num,
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
        ),
    ]
    return entries, mamba_host


def build_phase_flip_host_pools(scheduler):
    """#847: the WRITER for ``scheduler.phase_flip_host_pools``.

    THE ACTUATOR THAT WAS MISSING, and that is the whole shape of this fix.
    Every other part of the #718 rebind already existed and was already wired:
    ``rebind_for_cutover`` is called at the cutover, the #719 generation stamp
    and ``coherence_check`` are built, and ``phase_pools_for`` knows exactly
    what it wants. It wanted ``scheduler.phase_flip_host_pools[phase]`` -- and
    across the whole tree that name appeared ONLY in its own docstring and its
    own refusal message. Nothing ever wrote it, so the rebind could never arm.

    W32 measured the consequence end to end: no host pool -> ``RebindRefused``
    -> the rebind never arms -> ``bound_phase()`` stays ``"pp"`` ->
    ``device_tier_disarmed("load")`` is True for the whole TP phase ->
    ``HiCacheController.load()`` returns None -> ZERO tokens reach the device.
    The transport prefill logged ``#cached-token: 0`` on what should have been
    a perfect disk hit, and the specimen carries 6 ``#718 hicache-phase-guard``
    warnings beside ``phase_flip_rebind_hicache=False``.

    REFUSAL CONVERSION, NOT GUARD DELETION (#847). ``phase_pools_for`` still
    raises for a genuinely absent or mis-shaped pool -- that guard is correct
    and stays exactly as strict as it is. This turns "structurally impossible"
    into "possible and priced": the pool now exists, and its bytes are a named
    post in the HOST-LEDGER (#721), where the floor never yields to the post.

    FLAG-GATED, so every boot that does not ask for the rebind is
    byte-identical: without ``--phase-flip-rebind-hicache`` this returns an
    empty mapping and allocates nothing.

    The ``pp`` entry is the tier the boot already built -- the rebind needs a
    handle per phase, not a second pp pool. Only the ``tp`` side is new, and it
    is a staging pin (see ``PHASE_FLIP_STAGING_CHUNKS``).
    """
    import logging

    logger = logging.getLogger(__name__)
    sa = getattr(scheduler, "server_args", None)
    if not getattr(sa, "phase_flip_rebind_hicache", False):
        return {}

    tree = getattr(scheduler, "tree_cache", None)
    pp_host = host_tier_of(tree)
    if pp_host is None:
        # The rebind was ASKED for and the instance has no host tier at all.
        # Returning {} here would hand `phase_pools_for` its own refusal one
        # layer later with a less useful message, so say it where the cause is.
        logger.error(
            "#847 PHASE-FLIP REBIND: --phase-flip-rebind-hicache is set but "
            "this boot has no HiCache host tier, so there is nothing to build "
            "a phase-matched pin from. The rebind will refuse at the first "
            "cutover. Enable hierarchical cache, or drop the flag."
        )
        return {}

    stacks = getattr(scheduler, "phase_flip_stacks", None)
    tp_worker = getattr(stacks, "tp_worker", None)
    tp_runner = getattr(tp_worker, "model_runner", None)
    tp_device_pool = getattr(tp_runner, "token_to_kv_pool", None)
    if tp_device_pool is None:
        logger.error(
            "#847 PHASE-FLIP REBIND: no TP device pool at boot, so the "
            "phase-matched staging pin cannot be allocated FROM it (a host "
            "pool is allocated from its device pool -- DESIGN_706 C1). The "
            "rebind will refuse at the first cutover."
        )
        return {"pp": pp_host}

    gib = _staging_pin_gib(scheduler, tp_device_pool, pp_host)
    size_gb = max(1, int(gib + 0.999))
    try:
        # W34: BUILT WITH THE ASSEMBLER'S OWN NAMED PRIMITIVES, not by cloning
        # `type(pp_host)`. The clone was W34 arm 1's defect: the live host tier
        # is a `HostPoolGroup` COMPOSITE whose constructor takes
        # `entries: list[PoolEntry]`, so calling it with the MHA/MLA pool
        # signature died on `unexpected keyword argument 'allocator_type'`.
        # A type cloned without its contract is a guess; these three builders
        # ARE the contract, and reusing them is the one-mover rule (the
        # assembler has five call sites and this must not become a sixth
        # hand-rolled one).
        import copy as _copy

        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            build_kv_host_pool,
            build_pool_entry,
        )
        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup
        from sglang.srt.mem_cache.hicache_storage import PoolName

        # The size override rides a COPY of server_args: `build_kv_host_pool`
        # reads `hicache_ratio`/`hicache_size` off it, and a ratio is the
        # mirror-shaped answer #810 forbids here. Copying leaves the real
        # server_args untouched for every other reader.
        sa_pin = _copy.copy(sa)
        sa_pin.hicache_ratio = 0
        sa_pin.hicache_size = size_gb

        # W34 arm 2: UNWRAP THE WAY THE CONSUMER UNWRAPS. `phase_pools_for`
        # does `inner = getattr(device_pool, "full_kv_pool", device_pool)`
        # before it builds the PhasePools that `check_shapes` reads, so the
        # pool whose `layer_num` must match is the INNER one. Reading the
        # wrapper directly is how arm 2 refused itself with "the TP device
        # pool exposes no layer_num" -- the wrapper has no such attribute and
        # the pool that does was one dereference away. Building the host pool
        # from the wrapper while the shape check reads the inner pool would
        # also be a latent mismatch even if the attribute had existed.
        inner_pool = getattr(tp_device_pool, "full_kv_pool", tp_device_pool)
        layers = int(getattr(inner_pool, "layer_num", 0) or 0)
        if layers <= 0:
            raise ValueError(
                "the TP device pool exposes no layer_num, so the host pool "
                "cannot be shape-matched to it (check_shapes compares exactly "
                "that number)"
            )
        use_mla = "MLA" in type(tp_device_pool).__name__
        kv_host = build_kv_host_pool(
            kv_pool=inner_pool,
            page_size=int(getattr(sa, "page_size", 1) or 1),
            server_args=sa_pin,
            use_mla=use_mla,
        )
        # #871: MIRROR THE BOUND TIER'S POOL SET, NOT JUST ITS KV HALF.
        #
        # THE DEFECT THIS CLOSES, and it is #718/#847's own unmet precondition
        # rather than a new finding. This pin was built with ONE entry, KV. On
        # a hybrid model the live tier carries KV *and* MAMBA, so
        # `check_pool_coverage` computed `missing = {MAMBA}` and refused the
        # rebind -- correctly, every single time. Measured on the W40 #857
        # acceptance boot: 60 refusals, 0 arms, and `#cached-token: 0` on all
        # 243 prefill batches, because a refused rebind leaves the #718 device
        # tier DISARMED and every read-through misses.
        #
        # That guard's docstring states the remedy exactly, and this is it:
        # "A phase host tier has to be built with the FULL POOL SET before this
        # rebind can arm; until then the #718 disarm is the correct state and a
        # read-through miss is the correct cost."
        #
        # THE GUARD IS NOT TOUCHED. It must stop firing because its
        # precondition is MET, never because it was removed -- the difference
        # between a fix and a disarm. `check_pool_coverage` stays exactly as
        # strict as it is, and `test_phase_tier_full_pool_set_871.py` asserts
        # BOTH directions: a full-set tier arms, a narrowed one still refuses.
        #
        # DERIVED FROM THE BOUND TIER, not from the model config: the set that
        # must be covered is whatever the READER actually names, which is the
        # same quantity `check_pool_coverage` compares against. Reading the
        # config instead would be a second opinion about the same fact, and the
        # two would drift.
        _pp_names = set(getattr(pp_host, "entry_map", None) or ())
        _extra = _pp_names - {PoolName.KV}
        entries = [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host,
                device_pool=inner_pool,
                # Identity: this pool carries the TP phase's own layers,
                # so transfer index i IS device layer i.
                layer_mapping={i: i for i in range(layers)},
                transfer_layer_num=layers,
                is_anchor=True,
            )
        ]
        mamba_host = None
        if PoolName.MAMBA in _extra:
            # REBUILDS BOTH ENTRIES, and the reason is the transfer index.
            # `build_hybrid_mamba_stack` sets `transfer_layer_num =
            # len(full_layer_mapping | mamba_layer_mapping)` and gives each
            # entry the pool's OWN mapping. The KV-only pin above uses an
            # identity map over `range(layers)`, which is right while KV is the
            # only entry and wrong the moment a second pool shares the transfer
            # index space -- the two maps would collide at index 0. So the
            # hybrid case is built from the assembler's contract rather than
            # patched onto the identity one.
            _hybrid = _hybrid_pin_entries(
                tp_runner=tp_runner,
                sa=sa,
                kv_host=kv_host,
                inner_pool=inner_pool,
                tp_device_pool=tp_device_pool,
                logger=logger,
            )
            if _hybrid is not None:
                entries, mamba_host = _hybrid
        _unhandled = _extra - {PoolName.MAMBA}
        if _unhandled:
            # NAMED, NOT SWALLOWED. A pool set this builder does not know how to
            # mirror will still be refused by `check_pool_coverage`, which is the
            # safe outcome -- but the operator must learn it here, at the cause,
            # rather than from a coverage message that only says a name is
            # missing. Every such pool is a follow-up of this ticket.
            logger.error(
                "#871 PHASE-FLIP REBIND: the bound tier also describes %s, "
                "which this builder cannot yet mirror into the 'tp' staging "
                "pin. The rebind will keep refusing on coverage until that "
                "pool is added here. The guard is doing its job; the pin is "
                "incomplete.",
                sorted(str(n) for n in _unhandled),
            )
        tp_host = HostPoolGroup(entries)
    except Exception as exc:  # noqa: BLE001 - a refusal must be legible
        logger.error(
            "#847 PHASE-FLIP REBIND: could not allocate the phase-matched "
            "staging pin (%.3f GiB -> %d GB requested): %s. The rebind will "
            "refuse at the first cutover rather than run against a mis-shaped "
            "pool, which is the guard working as designed.",
            gib,
            size_gb,
            exc,
        )
        return {"pp": pp_host}

    # #721 HOST-LEDGER: the pin is a NAMED POST, and the floor never yields to
    # it. Printed here, at the allocation, so the number in the ledger is the
    # number that was actually taken rather than an intention.
    try:
        import subprocess

        free_g = int(
            subprocess.run(["free", "-g"], capture_output=True, text=True)
            .stdout.split("\n")[1]
            .split()[6]
        )
    except Exception:  # noqa: BLE001 - the ledger must not break the boot
        free_g = -1
    # #871: the MAMBA half is a POST OF ITS OWN, priced from what was actually
    # allocated rather than from the intention. An unpriced post is the thing
    # the ledger exists to prevent, and adding a second pinned pool without
    # naming it would have been exactly that.
    mamba_bytes = 0
    if mamba_host is not None:
        try:
            mamba_bytes = int(
                getattr(mamba_host, "size", 0) * getattr(mamba_host, "size_per_token", 0)
            )
        except Exception:  # noqa: BLE001 - the ledger must not break the boot
            mamba_bytes = -1
    logger.warning(
        "HOST-LEDGER POST #847/#871 phase-flip staging pin: %d GB pinned for "
        "the 'tp' phase-matched host view (%.3f GiB derived from %d tok x "
        "cell) + MAMBA half %s (%s slots mirroring the device pool, ratio 1.0 "
        "-- per-SLOT not per-GB), pools=%s, host free after = %s GB against "
        "the 16 GB floor. This post exists so the #718 device tier stays ARMED "
        "across the cutover; without it load() returns None for the whole TP "
        "phase and every read-through misses. The POST shrinks if it does not "
        "fit -- the FLOOR never does.",
        size_gb,
        gib,
        int(getattr(sa, "chunked_prefill_size", 4096) or 4096)
        * int(getattr(sa, "max_running_requests", 1) or 1),
        (
            "not built"
            if mamba_host is None
            else ("unpriceable" if mamba_bytes < 0 else f"{mamba_bytes / 2**30:.3f} GiB")
        ),
        "0" if mamba_host is None else getattr(mamba_host, "size", "?"),
        # getattr, for the SAME STAND-IN reason this module states at its other
        # probes and which I broke by not reading it first: this writer is
        # driven in tests by scheduler and pool STAND-INS carrying only what
        # the writer uses, and `HostPoolGroup` itself is patched there. An
        # instrument may never be the thing that breaks a boot -- reporting
        # "unknown" is the honest answer when the shape is not there.
        sorted(str(n) for n in (getattr(tp_host, "entry_map", None) or ())) or "unknown",
        free_g if free_g >= 0 else "unknown",
    )
    return {"pp": pp_host, "tp": tp_host}
