"""Cross-algorithm speculative decoding (T156 stage 2) -- argument shaping.

Under ``--speculative-cross-algorithm`` ONE server hosts TWO co-resident
speculative rungs over the SAME target model:

* the NEXTN/MTP rung (EAGLEWorkerV2, draft = the target checkpoint's MTP
  head, placement SPLIT -- TP-sharded exactly like a plain NEXTN server), and
* the DFLASH rung (DFlashWorkerV2, draft = a separate DFLASH checkpoint,
  placement SOLO on TP rank 0 -- forced, because DFlashAttention requires
  heads % tp == 0, which is unsatisfiable under uneven tp3).

The two rungs need DIFFERENT values in the same ``speculative_*`` server-args
fields (algorithm, draft path, num_steps, num_draft_tokens, placement, ...).
This module resolves both value sets ("shapes") once at argument-handling
time:

* the FORCED rung's shape (``--speculative-cross-algorithm-force``) is applied
  to the real ServerArgs, so the whole boot pipeline (scheduler, target model
  runner, pool sizing, graph capture, batch stamping) behaves exactly like the
  corresponding single-algorithm server -- that is the stage-2 contract, and
  it is what makes the forced arm comparable to its single-algo control;
* the OTHER rung's shape is stashed on the ServerArgs instance
  (``speculative_cross_shapes``) and consumed by CrossAlgoWorker to build the
  secondary sub-worker from a deep-copied, re-shaped ServerArgs.

Stage 3 (per-batch switching) only changes who stamps
``batch.spec_algorithm``; the shapes and the dual resource build stay.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

CROSS_SHAPES_ATTR = "speculative_cross_shapes"

# ---------------------------------------------------------------------------
# Dual hidden-state capture (stage 3, schedule mode only).
#
# Under per-batch switching BOTH rungs must stay warm on every verify round:
# the NEXTN/MTP draft-extend consumes the target's FINAL hidden states, the
# DFLASH draft-KV append consumes the concatenated AUX layer hidden states.
# With this process-local flag on, LogitsProcessor stores the final hidden
# states in `hidden_states` (the EAGLE convention -- every EAGLE consumer
# stays untouched) and the aux concat in the separate
# `cross_aux_hidden_states` field (read by the DFLASH rung under the cross
# gate). Off (default, incl. the stage-2 force modes and every single-algo
# server): behavior is byte-identical to before.
# ---------------------------------------------------------------------------
_DUAL_CAPTURE_ACTIVE = False


def set_dual_capture_active(enabled: bool) -> None:
    global _DUAL_CAPTURE_ACTIVE
    _DUAL_CAPTURE_ACTIVE = bool(enabled)


def dual_capture_active() -> bool:
    return _DUAL_CAPTURE_ACTIVE


# Numeric shape fields swapped onto the REAL ServerArgs on every rung switch
# (schedule mode). Audited runtime readers that must see the ACTIVE rung's
# value: dflash_info_v2.prepare_for_decode (block size), the eager verify
# sizing fallback, metrics. The algorithm STRING and every boot-sizing
# consumer (max_speculative_num_draft_tokens & friends) stay fixed.
RUNTIME_SHAPE_FIELDS = (
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_dflash_block_size",
)


def apply_runtime_shape(server_args, shape: Dict[str, Any], reason: str) -> None:
    """Atomically point the REAL ServerArgs' numeric spec-shape fields at the
    active rung's values (single override call, single stamp point)."""
    server_args.override(
        reason, **{f: shape[f] for f in RUNTIME_SHAPE_FIELDS}
    )

# Fields a rung shape may override on the deep-copied ServerArgs of the
# secondary sub-worker (and that the forced shape writes on the real one).
_SHAPE_FIELDS = (
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_draft_model_revision",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_dflash_block_size",
    "speculative_draft_attention_backend",
    "speculative_draft_placement",
    "speculative_draft_gpu",
    "speculative_adaptive",
    "speculative_adaptive_config",
    "speculative_draft_window_size",
    "speculative_use_rejection_sampling",
)


def cross_algo_enabled(server_args) -> bool:
    return bool(getattr(server_args, "speculative_cross_algorithm", False))


def cross_schedule_interval(server_args) -> Optional[int]:
    """Verify-round switch interval under force=schedule:N, else None.

    Stage-3 debug policy only; the runtime-switching code paths themselves
    are gated by ``cross_switching_active`` (schedule AND auto).
    """
    if not cross_algo_enabled(server_args):
        return None
    shapes = getattr(server_args, CROSS_SHAPES_ATTR, None)
    if shapes is None or shapes.get("force") != "schedule":
        return None
    return int(shapes["schedule_interval"])


# The per-batch-switching modes (stage 3 schedule + stage 4 auto). Everything
# these modes share -- dual hidden capture, dual prefill, per-round
# warm-keeping, the scheduler pre-decode switch hook, DFLASH request
# validation for every request -- gates on membership here; the static
# force=nextn|dflash modes keep their stage-2 behavior untouched.
SWITCHING_MODES = ("schedule", "auto")


def cross_switching_active(server_args) -> bool:
    """True when the active-rung policy switches rungs at runtime."""
    if not cross_algo_enabled(server_args):
        return False
    shapes = getattr(server_args, CROSS_SHAPES_ATTR, None)
    return shapes is not None and shapes.get("force") in SWITCHING_MODES


def parse_cross_force(force: Optional[str]) -> tuple[str, Optional[int]]:
    """Parse --speculative-cross-algorithm-force into (mode, interval)."""
    if force in ("nextn", "dflash", "auto"):
        return force, None
    if isinstance(force, str) and force.startswith("schedule:"):
        try:
            interval = int(force.split(":", 1)[1])
        except ValueError:
            interval = -1
        if interval < 1:
            _fail(
                "force=schedule:N needs a positive integer switch interval "
                f"(verify rounds); got {force!r}."
            )
        return "schedule", interval
    _fail(
        "--speculative-cross-algorithm-force must be 'nextn', 'dflash', "
        "'schedule:N' (switch every N verify rounds), or 'auto' (stage-4 "
        f"acceptance-driven bandit); got {force!r}."
    )


def get_cross_shapes(server_args) -> Dict[str, Dict[str, Any]]:
    shapes = getattr(server_args, CROSS_SHAPES_ATTR, None)
    if shapes is None:
        raise RuntimeError(
            "--speculative-cross-algorithm is set but the rung shapes were "
            "not resolved (normalize_cross_algorithm_args did not run in "
            "this process; the stash did not survive the process boundary?)."
        )
    return shapes


def _fail(msg: str) -> None:
    raise ValueError(f"--speculative-cross-algorithm: {msg}")


def normalize_cross_algorithm_args(server_args: "ServerArgs") -> None:
    """Validate the cross-algorithm request, resolve both rung shapes, apply
    the forced shape to *server_args* and stash the other shape.

    Must run inside ``handle_speculative_decoding`` AFTER the NEXTN->EAGLE
    alias resolution and BEFORE the per-algorithm handler dispatch, so the
    dispatch sees the forced shape.
    """
    from sglang.srt.arg_groups.speculative_hook import _handle_dflash

    force, schedule_interval = parse_cross_force(
        server_args.speculative_cross_algorithm_force
    )
    if server_args.speculative_algorithm != "EAGLE":
        _fail(
            "requires --speculative-algorithm NEXTN (or EAGLE) as the MTP "
            f"rung; got {server_args.speculative_algorithm!r}."
        )
    if server_args.speculative_draft_model_path is None:
        _fail(
            "requires --speculative-draft-model pointing at the DFLASH draft "
            "checkpoint (the MTP rung drafts from the target checkpoint and "
            "needs no path)."
        )
    if server_args.device != "cuda":
        _fail(f"only supports CUDA; got device={server_args.device!r}.")
    if server_args.pp_size != 1 or server_args.dp_size != 1:
        _fail(
            "supports pure single-node TP only; got "
            f"pp_size={server_args.pp_size}, dp_size={server_args.dp_size}."
        )
    if server_args.enable_dp_attention:
        _fail("does not support --enable-dp-attention.")
    if server_args.ep_size != 1:
        _fail(f"does not support expert parallelism (ep_size={server_args.ep_size}).")
    if server_args.nnodes != 1:
        _fail(f"is single-node only (nnodes={server_args.nnodes}).")
    if server_args.disaggregation_mode != "null":
        _fail(
            "does not support PD disaggregation "
            f"(disaggregation_mode={server_args.disaggregation_mode!r})."
        )
    if server_args.enable_multi_layer_eagle:
        _fail("does not support --enable-multi-layer-eagle.")
    if server_args.speculative_use_rejection_sampling:
        _fail("does not support --speculative-use-rejection-sampling.")
    if (
        server_args.speculative_eagle_topk is not None
        and int(server_args.speculative_eagle_topk) != 1
    ):
        _fail(
            "requires topk == 1 (linear MTP chain); got "
            f"--speculative-eagle-topk {server_args.speculative_eagle_topk}."
        )
    if server_args.speculative_draft_window_size is not None:
        _fail(
            "does not support --speculative-draft-window-size: the DFLASH "
            "rung runs draft-solo on rank 0, and solo placement has no "
            "compact draft cache."
        )
    if server_args.speculative_draft_placement != "split":
        _fail(
            "owns the draft placement (NEXTN rung: split, DFLASH rung: solo "
            "on rank 0); leave --speculative-draft-placement unset."
        )
    if server_args.speculative_draft_gpu is not None:
        _fail("owns the solo rank (rank 0); leave --speculative-draft-gpu unset.")

    user_steps = server_args.speculative_num_steps
    user_adaptive = bool(server_args.speculative_adaptive)
    user_adaptive_config = server_args.speculative_adaptive_config
    dflash_draft_path = server_args.speculative_draft_model_path
    dflash_draft_revision = server_args.speculative_draft_model_revision

    # ------------------------------------------------------------------
    # Resolve the DFLASH shape on a probe copy: _handle_dflash performs the
    # full validation + block-size inference + draft-attention-backend
    # resolution; reusing it verbatim keeps the cross path in lockstep with
    # the single-algorithm DFLASH path.
    # ------------------------------------------------------------------
    probe = copy.deepcopy(server_args)
    probe.speculative_algorithm = "DFLASH"
    probe.speculative_num_steps = None
    probe.speculative_eagle_topk = None
    probe.speculative_num_draft_tokens = None
    probe.speculative_adaptive = False
    probe.speculative_adaptive_config = None
    probe.speculative_draft_placement = "solo"
    _handle_dflash(probe)

    dflash_shape: Dict[str, Any] = {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": dflash_draft_path,
        "speculative_draft_model_revision": dflash_draft_revision,
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": int(probe.speculative_num_draft_tokens),
        "speculative_dflash_block_size": int(probe.speculative_num_draft_tokens),
        "speculative_draft_attention_backend": (
            probe.speculative_draft_attention_backend
        ),
        # DFLASH rung placement is SOLO on rank 0 by requirement:
        # DFlashAttention needs heads % tp == 0, unsatisfiable under uneven
        # tp3; measured 2026-07-22 on bdc1714551.
        "speculative_draft_placement": "solo",
        "speculative_draft_gpu": None,
        "speculative_adaptive": False,
        "speculative_adaptive_config": None,
        "speculative_draft_window_size": None,
        "speculative_use_rejection_sampling": False,
    }

    eagle_steps = int(user_steps) if user_steps is not None else 3
    eagle_shape: Dict[str, Any] = {
        "speculative_algorithm": "EAGLE",
        # The MTP rung drafts from the TARGET checkpoint (ModelConfig swaps
        # the arch to the *MTP variant when is_draft_model=True and the path
        # falls back to model_path) -- exactly the plain NEXTN server shape.
        "speculative_draft_model_path": None,
        "speculative_draft_model_revision": None,
        "speculative_num_steps": eagle_steps,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": eagle_steps + 1,
        "speculative_dflash_block_size": None,
        "speculative_draft_attention_backend": None,
        "speculative_draft_placement": "split",
        "speculative_draft_gpu": None,
        # Adaptive k-laddering is an EAGLE-side feature. When DFLASH is the
        # forced primary, the secondary EAGLE rung is built WITHOUT adaptive:
        # its controller would otherwise clobber the target's active
        # attention backend / decode graph runner on activation, and the
        # launcher decided the torch_memory_saver LD_PRELOAD hook from the
        # GLOBAL (dflash-shaped, adaptive-off) args, so an offload-mode
        # controller in the copy could not initialize. The schedule mode
        # (stage 3) ALSO builds the NEXTN rung without adaptive: the
        # k-controller's _activate would pause the mapped DFLASH rung tag
        # mid-segment (use-after-unmap). The auto mode (stage 4) replaces
        # the k-controller entirely: the bandit runs the NEXTN k rungs as
        # its own arms (design 4b), so adaptive stays off there too.
        "speculative_adaptive": user_adaptive if force == "nextn" else False,
        "speculative_adaptive_config": (
            user_adaptive_config if force == "nextn" else None
        ),
        "speculative_draft_window_size": None,
        "speculative_use_rejection_sampling": False,
    }
    if force != "nextn" and user_adaptive:
        logger.info(
            "--speculative-cross-algorithm force=%s: the NEXTN rung is built "
            "without --speculative-adaptive (%s); see cross_algo_utils for "
            "the rationale.",
            force,
            (
                "the stage-4 bandit owns the k rungs"
                if force == "auto"
                else f"single k={eagle_steps} state"
            ),
        )

    # auto mode (stage 4): the NEXTN k rungs are bandit ARMS. The k set comes
    # from the user's --speculative-adaptive-config (positive candidate steps
    # union; the built-in default yields [1, 2, 3], the high-accept profile
    # [1..5]); the boot k (primary rung) is always included.
    bandit_ks: Optional[list] = None
    if force == "auto":
        from sglang.srt.speculative.adaptive_spec_params import (
            resolve_candidate_steps_from_config,
        )

        bandit_ks = sorted(
            {
                s
                for s in resolve_candidate_steps_from_config(
                    cfg_path=user_adaptive_config, algorithm="EAGLE"
                )
                if s >= 1
            }
            | {eagle_steps}
        )
        dflash_tokens = int(dflash_shape["speculative_num_draft_tokens"])
        if any(k + 1 == dflash_tokens for k in bandit_ks):
            _fail(
                f"bandit k set {bandit_ks} contains k={dflash_tokens - 1}, "
                "whose draft-token stride collides with the DFLASH block "
                f"size {dflash_tokens}; verify results could not be "
                "attributed to a unique rung."
            )

    # schedule/auto mode: the NEXTN shape is the PRIMARY (global) shape. This
    # is load-bearing, not a taste choice: scheduler.spec_algorithm and the
    # FutureMap relay type derive from the global speculative_algorithm, and
    # only the EAGLE type relays topk/hidden draft seeds -- the union both
    # rungs need. The numeric shape fields are re-pointed at the active rung
    # on every switch (CrossAlgoWorker), the algorithm STRING never moves.
    forced_shape = dflash_shape if force == "dflash" else eagle_shape
    other_name = "nextn" if force == "dflash" else "dflash"
    other_shape = eagle_shape if force == "dflash" else dflash_shape

    # Apply the forced shape to the REAL args. handle_speculative_decoding is
    # still inside ServerArgs.__post_init__, so plain assignment is the
    # correct mutation style here (mirrors _handle_dflash & co).
    for field in _SHAPE_FIELDS:
        setattr(server_args, field, forced_shape[field])

    # Stash the other rung's shape for CrossAlgoWorker. Instance attribute
    # (not a dataclass field) on purpose: it is derived state, survives
    # pickling to the scheduler processes via __dict__, and stays off the
    # CLI surface. get_cross_shapes() raises loudly if it went missing.
    # schedule/auto mode additionally stashes BOTH shapes (+ the interval /
    # the bandit k set) so the meta-worker can swap the numeric fields per
    # switch.
    stash: Dict[str, Any] = {other_name: other_shape, "force": force}
    if force in SWITCHING_MODES:
        stash["nextn"] = eagle_shape
        stash["dflash"] = dflash_shape
        if force == "schedule":
            stash["schedule_interval"] = schedule_interval
        else:
            stash["bandit_ks"] = bandit_ks
    object.__setattr__(server_args, CROSS_SHAPES_ATTR, stash)
    logger.info(
        "Cross-algorithm speculative decoding: force=%s%s; primary rung=%s; "
        "secondary rung=%s resident (algorithm=%s, draft_tokens=%s, "
        "placement=%s).",
        force,
        (
            f" (switch every {schedule_interval} verify rounds)"
            if force == "schedule"
            else (f" (bandit arms: NEXTN k={bandit_ks} + DFLASH)" if force == "auto" else "")
        ),
        "nextn" if force != "dflash" else "dflash",
        other_name,
        other_shape["speculative_algorithm"],
        other_shape["speculative_num_draft_tokens"],
        other_shape["speculative_draft_placement"],
    )


def apply_shape_to_args_copy(server_args: "ServerArgs", shape: Dict[str, Any]):
    """Deep-copy *server_args* and apply a rung *shape* to the copy (via the
    audited post-resolution mutation point). The copy is what the secondary
    sub-worker keeps as ``self.server_args`` for its whole lifetime."""
    args_copy = copy.deepcopy(server_args)
    args_copy.override("cross_algo.secondary_shape", **shape)
    return args_copy
