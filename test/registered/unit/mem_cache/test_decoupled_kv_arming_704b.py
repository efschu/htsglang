# SPDX-License-Identifier: Apache-2.0
"""#704b arming slice: the pool, the routing, and the seam between them.

The hazard this suite exists for is that B1 has TWO independent switches --
the pool SHAPE (R6) and the group ROUTING (B1) -- and only one combination is
wrong in a way nothing reports: routing armed over a stage-local pool. That
case gets the red-first test; the rest pin that the default path did not move.
"""

import pytest

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.mem_cache.decoupled_kv_arming import (
    DecoupledKvArmingError,
    arm_decoupled_kv,
    arm_for_cutover,
    record_pool_plan,
    reset_recorded_pool_plan,
)
from sglang.srt.mem_cache.decoupled_kv_pool_plan import (
    DECOUPLED,
    STAGE_LOCAL,
    plan_for_rank,
    pool_build_args,
)

ATTN = tuple(i for i in range(64) if (i + 1) % 4 == 0)  # 16 full-attention layers
CELL = 2048
T = 436_766
PERIOD = 16
PP_TO_TP = "pp_to_tp"


class _G:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def routes():
    saved = (
        ps._PHASE_FLIP_TP_ACTIVE,
        ps._FLIP_DCP,
        ps._DCP_SPILL_ACTIVE,
        ps._DCP_SPILL,
        ps._DECOUPLED_KV_ACTIVE,
        ps._DECOUPLED_KV,
        ps._DCP,
    )
    ps._FLIP_DCP = _G("flip")
    ps._DCP_SPILL = _G("spill")
    ps._DECOUPLED_KV = _G("b1")
    ps._DCP = _G("primary")
    ps._PHASE_FLIP_TP_ACTIVE = False
    ps._DCP_SPILL_ACTIVE = False
    ps._DECOUPLED_KV_ACTIVE = False
    reset_recorded_pool_plan()
    yield
    reset_recorded_pool_plan()
    (
        ps._PHASE_FLIP_TP_ACTIVE,
        ps._FLIP_DCP,
        ps._DCP_SPILL_ACTIVE,
        ps._DCP_SPILL,
        ps._DECOUPLED_KV_ACTIVE,
        ps._DECOUPLED_KV,
        ps._DCP,
    ) = saved


def _decoupled(share=4 / PERIOD):
    return plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=share, period=PERIOD)


def _stage_local():
    return plan_for_rank(ATTN, 0, 28, T, CELL, armed=False)


# --- (e) the red-first case: armed routing over a stage-local pool -----------


def test_ARMING_OVER_A_STAGE_LOCAL_POOL_IS_REFUSED(routes):
    """THE failure this slice exists to make loud.

    A stage-local pool has rows only for layers 0-27's attention slots. Armed
    routing may ask this rank for any of the 16, and HybridLinearKVPool's
    id mapping (memory_pool.py:3855-3859) either misses or -- where the id
    happens to exist -- returns ANOTHER layer's rows. Wrong output, no error.
    """
    record_pool_plan(_stage_local())
    with pytest.raises(DecoupledKvArmingError, match="was built 'stage_local'"):
        arm_decoupled_kv(True, ATTN)
    assert ps.get_dcp_group().name == "primary", "refusal must not leave it armed"


def test_the_refusal_names_what_the_pool_actually_holds(routes):
    """An error that does not say what is wrong sends the reader to the wrong
    layer of the stack -- routing looks identical in both cases."""
    record_pool_plan(_stage_local())
    with pytest.raises(DecoupledKvArmingError) as e:
        arm_decoupled_kv(True, ATTN)
    msg = str(e.value)
    assert "7 attention layer(s)" in msg  # 3,7,...,27 of the 16
    assert "plan_decoupled" in msg


def test_arming_with_NO_recorded_plan_is_refused(routes):
    """An unrecorded pool is indistinguishable from a stage-local one, so it
    is treated as one rather than assumed healthy."""
    with pytest.raises(DecoupledKvArmingError, match="no KV pool plan was recorded"):
        arm_decoupled_kv(True, ATTN)


def test_a_decoupled_plan_missing_layers_is_refused(routes):
    """Claiming DECOUPLED is not the same as covering the attention set."""
    short = plan_for_rank(
        ATTN[:8], 0, 28, T, CELL, armed=True, share=4 / PERIOD, period=PERIOD
    )
    assert short.mode == DECOUPLED
    record_pool_plan(short)
    with pytest.raises(DecoupledKvArmingError, match="missing attention layer"):
        arm_decoupled_kv(True, ATTN)


def test_a_decoupled_pool_ARMS(routes):
    record_pool_plan(_decoupled())
    arm_decoupled_kv(True, ATTN)
    assert ps.get_dcp_group().name == "b1"


# --- the deliberate asymmetry ------------------------------------------------


def test_DISARMING_is_never_refused(routes):
    """Requiring a healthy pool in order to STOP using it would turn the
    recovery path into a second failure."""
    record_pool_plan(_stage_local())
    arm_decoupled_kv(False, ATTN)
    assert ps.get_dcp_group().name == "primary"
    reset_recorded_pool_plan()
    arm_decoupled_kv(False, ATTN)  # not even a recorded plan is required


# --- parity with the set_phase_flip_tp_active precedent ---------------------


def test_arming_the_flag_with_NO_GROUP_now_refuses(routes):
    """parallel_state.py:2670-2687 refuses this for the flip route, naming
    'a silent no-op all-reduce, the exact corruption class this routing
    exists to prevent'. B1 silently no-oped; it no longer does."""
    ps._DECOUPLED_KV = None
    with pytest.raises(RuntimeError, match="without an initialized"):
        ps.set_decoupled_kv_active(True)


def test_disarming_with_no_group_is_still_fine(routes):
    ps._DECOUPLED_KV = None
    ps.set_decoupled_kv_active(False)


# --- (b) the phase boundary --------------------------------------------------


def test_the_cutover_ARMS_going_into_PP_and_DISARMS_going_into_TP(routes):
    """B1 is a PP-PREFILL mechanism. Going into TP it must be disarmed, not
    merely out-prioritised by get_dcp_group's ordering -- relying on branch
    order is how a later reorder turns into silent capture."""
    record_pool_plan(_decoupled())
    assert arm_for_cutover("tp_to_pp", PP_TO_TP, ATTN) is True
    assert ps.get_dcp_group().name == "b1"
    assert arm_for_cutover(PP_TO_TP, PP_TO_TP, ATTN) is False
    assert ps.get_dcp_group().name == "primary"


def test_the_flip_route_still_outranks_b1_even_if_left_armed(routes):
    """Defence in depth: disarming is the contract, precedence is the net."""
    record_pool_plan(_decoupled())
    arm_decoupled_kv(True, ATTN)
    ps._PHASE_FLIP_TP_ACTIVE = True
    assert ps.get_dcp_group().name == "flip"


# --- (c) the routing chain the attention merge actually uses ----------------


def test_the_attention_merge_group_RESOLVES_PER_ACCESS(routes):
    """The KV pool read/write path takes no group at all -- it is local
    memory. The group enters only at the cross-rank LSE merge, which reads
    `group = get_parallel().dcp_group` (flashinfer_backend.py:5634,
    triton_backend.py:2311 pass it into cp_lse_ag_out_ar_mha_uneven).

    runtime_context.py:392-394 resolves that through _v(), which calls the
    getter every access rather than caching (runtime_context.py:240-242). If
    it ever caches, arming would silently stop taking effect after the first
    read -- so pin the resolution, not just the value.
    """
    from sglang.srt.runtime_context import get_parallel

    record_pool_plan(_decoupled())
    assert get_parallel().dcp_group.name == "primary"
    arm_decoupled_kv(True, ATTN)
    assert get_parallel().dcp_group.name == "b1", (
        "arming did not reach the attention merge's group accessor; if this "
        "fails, dcp_group started caching and B1 arms nothing"
    )
    arm_decoupled_kv(False, ATTN)
    assert get_parallel().dcp_group.name == "primary"


# --- (a) what the build consumes --------------------------------------------


def test_pool_build_args_are_exactly_the_two_the_ctor_takes():
    plan = _decoupled()
    layer_ids, size = pool_build_args(plan)
    assert layer_ids == ATTN, "decoupled build must cover every attention layer"
    assert size == plan.tokens


def test_the_decoupled_build_holds_ALL_layers_and_FEWER_rows():
    """The whole inversion in one assertion."""
    local, dec = _stage_local(), _decoupled()
    assert len(local.layer_ids) == 7 and local.tokens == T
    assert len(dec.layer_ids) == 16 and dec.tokens < T


# --- R6 sizing correction: the shipped ceil, not a second rounding rule -----


def test_the_row_count_comes_from_dcp_compact_pool_rows():
    """CAN-FAIL: round(share * C) would floor here, and flooring is the
    off-by-one that already cost an out-of-bounds-scatter debugging round
    (owner.py:155-181). Chosen so the two rules DISAGREE."""
    from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

    # C divisible by the period is where the ceil is a WHOLE extra block and
    # the two rules are furthest apart: 200 exact rows vs 204 ceiled.
    C, period, ratio = 3200, 64, 4
    plan = plan_for_rank(
        ATTN, 0, 28, C, CELL, armed=True, share=ratio / period, period=period
    )
    assert plan.tokens == dcp_compact_pool_rows(C, period, ratio) == (3200 // 64 + 1) * 4
    assert plan.tokens == 204
    assert round(ratio / period * C) == 200, "the rule this module used to apply"
    assert plan.tokens > round(ratio / period * C), "the ceil must win"


def test_world_conservation_tolerates_the_ceil_but_not_a_lost_rank():
    """The ceil adds at most `period` rows across the world; a missing rank
    loses far more, and the check must still catch that."""
    from sglang.srt.mem_cache.decoupled_kv_pool_plan import validate_world_conservation

    bands = ((0, 2), (2, 7), (7, 16))
    plans = [
        plan_for_rank(
            ATTN, 0, 28, T, CELL, armed=True, share=(hi - lo) / PERIOD, period=PERIOD
        )
        for lo, hi in bands
    ]
    validate_world_conservation(plans, ATTN, T)
    with pytest.raises(Exception, match="world KV total"):
        validate_world_conservation(plans[:2], ATTN, T)


def test_ranks_disagreeing_about_the_period_are_caught():
    from sglang.srt.mem_cache.decoupled_kv_pool_plan import validate_world_conservation

    a = plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.5, period=16)
    b = plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.5, period=8)
    with pytest.raises(Exception, match="disagree about the owner-rule period"):
        validate_world_conservation([a, b], ATTN, T)


# --- (d) unarmed byte-identity, at the build seam ---------------------------


def test_the_build_override_is_INERT_unless_explicitly_enabled(monkeypatch):
    """END-TO-END half of the identity pin: with the gate off the override
    returns None, and None means the caller keeps ITS OWN expression rather
    than a reconstructed copy -- which is what makes 'unchanged' mean
    unchanged instead of 'recomputed the same way'.
    """
    import sglang.srt.model_executor.model_runner_kv_cache_mixin as mx

    monkeypatch.delenv("SGLANG_DECOUPLED_KV", raising=False)
    ids, size = mx.ModelRunnerKVCacheMixin._decoupled_kv_pool_override(
        object(), ATTN, 12345
    )
    assert ids is None and size == 12345


def test_the_build_site_uses_the_override_result(routes):
    """Structural pin: the ctor must read _b1_ids/_b1_size, and the original
    draft-gate expression must remain the else-branch rather than be replaced.
    """
    import ast
    import inspect

    import sglang.srt.model_executor.model_runner_kv_cache_mixin as mx

    tree = ast.parse(inspect.getsource(mx))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "HybridLinearKVPool"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["size"], ast.Name) and kw["size"].id == "_b1_size"
    ids = kw["full_attention_layer_ids"]
    assert isinstance(ids, ast.IfExp) and ids.body.id == "_b1_ids"
    assert isinstance(ids.orelse, ast.IfExp), "the draft-gate branch was lost"
