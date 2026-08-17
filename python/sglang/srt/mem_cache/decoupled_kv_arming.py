# SPDX-License-Identifier: Apache-2.0
"""#704b: arm B1 routing only when the pool underneath it was built for it.

B1 (``distributed/parallel_state.py``) routes the cross-rank attention merge
through a decoupled-KV group. R6 (``decoupled_kv_pool_plan.py``) plans a pool
shaped for that routing: ALL attention layers x this rank's token share. The
two are independent switches, and that is the whole hazard this module exists
for -- **arming the routing over a stage-local pool is silently wrong**.

Why silent, concretely. Under decoupled routing a rank may be asked for any
attention layer, because the merge gathers partials from every rank. A
stage-local pool has no rows for a layer whose weights this rank does not own,
so ``HybridLinearKVPool._transfer_full_attention_id`` (``memory_pool.py:
3855-3859``) raises a KeyError at best and, where the id happens to exist,
reads a DIFFERENT layer's rows at worst. Neither is a routing error anyone
would trace back to arming.

The precedent for refusing rather than no-oping is ``set_phase_flip_tp_active``
(``parallel_state.py:2670-2687``), whose docstring names the failure class
exactly: activating a route whose groups were never built leaves "a silent
no-op all-reduce, the exact corruption class this routing exists to prevent".
This module applies the same rule one level down, to the POOL rather than the
group.

Scope: guards and the arming entry point. It does not build pools and does not
own the phase machine; the flip calls in.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.mem_cache.decoupled_kv_pool_plan import DECOUPLED, KvPoolPlan

_RECORDED_PLAN: Optional[KvPoolPlan] = None


class DecoupledKvArmingError(RuntimeError):
    """Arming refused. Never downgraded to a warning: the states this catches
    are silently-wrong-output states, not degraded-performance states."""


def record_pool_plan(plan: Optional[KvPoolPlan]) -> None:
    """Called by the pool BUILD with the plan it actually built.

    Recording the plan the build USED -- rather than the plan someone intended
    -- is the point. An intention cannot be compared against reality, and the
    mismatch this module exists to catch is precisely intention diverging from
    what was allocated.
    """
    global _RECORDED_PLAN
    _RECORDED_PLAN = plan


def recorded_pool_plan() -> Optional[KvPoolPlan]:
    return _RECORDED_PLAN


def reset_recorded_pool_plan() -> None:
    """Test-only seam. Boot records once; nothing in production unrecords."""
    global _RECORDED_PLAN
    _RECORDED_PLAN = None


def assert_pool_supports_arming(all_attn_layer_ids=None) -> KvPoolPlan:
    """Refuse arming unless the built pool can answer a decoupled read."""
    plan = _RECORDED_PLAN
    if plan is None:
        raise DecoupledKvArmingError(
            "B1 arming refused: no KV pool plan was recorded. The pool build "
            "must call record_pool_plan() with the plan it built, so arming "
            "can be checked against what was allocated rather than assumed. "
            "An unrecorded pool is indistinguishable from a stage-local one."
        )
    if plan.mode != DECOUPLED:
        raise DecoupledKvArmingError(
            f"B1 arming refused: the KV pool was built {plan.mode!r}, which "
            f"holds only {len(plan.layer_ids)} attention layer(s) "
            f"{list(plan.layer_ids)}. Decoupled routing may ask this rank for "
            "ANY attention layer, and a stage-local pool has no rows for the "
            "layers this rank does not own -- the read would miss the layer "
            "mapping or, worse, land on another layer's rows. Build the pool "
            "with plan_decoupled() before arming."
        )
    if all_attn_layer_ids is not None:
        expected = tuple(int(i) for i in all_attn_layer_ids)
        if plan.layer_ids != expected:
            missing = [i for i in expected if i not in plan.layer_ids]
            raise DecoupledKvArmingError(
                "B1 arming refused: the recorded decoupled pool is missing "
                f"attention layer(s) {missing}. It claims DECOUPLED mode but "
                "was not built over the full attention set, so it cannot "
                "answer every read the routing will send it."
            )
    return plan


def arm_decoupled_kv(active: bool, all_attn_layer_ids=None) -> None:
    """The single arming entry point. Disarming is always allowed.

    Deliberate asymmetry: arming is checked, disarming is not. Disarming
    returns to the pre-#704b route, which is correct for any pool shape --
    a stage-local read is exactly what an unarmed rank performs. Requiring a
    healthy pool in order to STOP using it would turn a recovery path into a
    second failure.
    """
    from sglang.srt.distributed.parallel_state import set_decoupled_kv_active

    if active:
        assert_pool_supports_arming(all_attn_layer_ids)
    set_decoupled_kv_active(bool(active))


def arm_for_cutover(direction: str, pp_to_tp: str, all_attn_layer_ids=None) -> bool:
    """Arm/disarm B1 across a phase-flip cutover. Returns the armed state.

    B1 is a PP-PREFILL mechanism: it exists so the pipeline's stages can hold
    a phase-uniform KV layout. The TP decode phase routes through the flip's
    own DCP group, which already outranks B1 in ``get_dcp_group`` -- so B1
    must be DISARMED going into TP, not merely out-prioritised. Leaving it
    armed would rely on that precedence ordering staying correct forever,
    which is exactly the kind of implicit dependency that survives until
    someone reorders the branches.

    Insertion point is ``phase_flip_runtime.py`` step 1, alongside
    ``_ps.set_phase_flip_tp_active(tp_phase)`` (``:1196``): after the
    group-wide quiesce (``on_round`` consensus, ``:3066``/``:3121-3128``) and
    after the KV wave-move, but before any group handle is re-derived
    (``:1224``) or any collective runs on the new layout.
    """
    tp_phase = direction == pp_to_tp
    arm_decoupled_kv(not tp_phase, all_attn_layer_ids)
    return not tp_phase
