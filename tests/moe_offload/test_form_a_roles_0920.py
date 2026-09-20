# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for the Form A ROLE layer (slices 2 and 4).

No CUDA, no driver, no model. What is pinned here is that a rank's ROLE and
its dense SHARE say the same thing, and that every seam Form A needs but does
not yet have refuses BY NAME at configuration time instead of dying forty-five
layers into a boot:

  * the role vector is the thing that makes a zero legible. `1,0,0` alone is
    ambiguous -- in every caller older than Form A a zero is an arithmetic
    accident. With a role vector it is a layout, and the two are checked
    against each other in BOTH directions;
  * the plan-level zero admission: `set_tp_partition_ratios(allow_zero=...)`
    is a property of the installed PLAN, not of a call site, so every dense
    dimension in the process splits by the same rule. The default path stays
    byte-identical, and a family vector may not hand a shard to a rank the
    base plan gave width zero;
  * DCP under a host-held KV resolves to dcp_size 1 with the replicated
    geometry off, and an EXPLICIT dcp flag is refused rather than silently
    overridden -- ignoring an operator's flag is how a measurement ends up
    comparing two different layouts;
  * each unwired seam (F3, F4, F5, F6, F9, F10, F11) refuses with its own
    identity and its file:line, and each wired one (F1, F2, F7, F8) does not.
    Eleven, not nine: the slice-2 survey of the call sites found two more
    (the launcher's written 'a zero-head rank cannot happen', and the
    zero-width Linear that computes instead of being skipped).
"""

import pytest

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    set_tp_partition_ratios,
    scoped_tp_partition_ratios,
    tp_partition_allows_zero,
    tp_partition_size,
    tp_partition_sizes,
)
from sglang.srt.rank_role import (
    FormASeamNotWired,
    HOST,
    RankRoleError,
    RankRolePlan,
    SEAMS,
    WORKER,
    guard_collective_subgroup,
    guard_dcp_merge,
    guard_dense_weights,
    guard_draft_worker,
    guard_graph_mode,
    guard_kv_pool,
    parse_rank_roles,
    require_wired,
    resolve_dcp_under_host_kv,
)

FORM_A = RankRolePlan((HOST, WORKER, WORKER))


@pytest.fixture(autouse=True)
def _restore_process_plan():
    """The shard plan is a PROCESS global. Any test that installs one must
    put it back, or it leaks into every later test in the session -- which
    is the kind of cross-test coupling that reads as a real defect."""
    before = get_tp_partition_ratios()
    before_zero = tp_partition_allows_zero()
    yield
    set_tp_partition_ratios(before, None, before_zero)


# ==========================================================================
# 1. The role vector
# ==========================================================================
def test_parse_and_shape():
    assert parse_rank_roles("host,worker,worker") == (HOST, WORKER, WORKER)
    assert parse_rank_roles(" HOST , Worker ") == (HOST, WORKER)
    assert FORM_A.host_rank == 0
    assert FORM_A.worker_ranks == [1, 2]
    assert FORM_A.is_host(0) and FORM_A.is_worker(1) and FORM_A.is_worker(2)
    assert FORM_A.tp_size == 3


def test_bad_role_vectors_are_refused_by_name():
    with pytest.raises(RankRoleError, match="must be one of"):
        parse_rank_roles("host,driver")
    with pytest.raises(RankRoleError, match="one role per rank"):
        parse_rank_roles("host,worker", tp_size=3)
    with pytest.raises(RankRoleError, match="is empty"):
        parse_rank_roles(" , ")
    with pytest.raises(RankRoleError, match="exactly ONE attention host"):
        RankRolePlan((HOST, HOST, WORKER))
    with pytest.raises(RankRoleError, match="exactly ONE attention host"):
        RankRolePlan((WORKER, WORKER))
    with pytest.raises(RankRoleError, match="at least two ranks"):
        RankRolePlan((HOST,))


def test_role_and_dense_ratio_must_agree_in_both_directions():
    assert FORM_A.dense_ratio() == [1, 0, 0]
    FORM_A.check_dense_ratio([1, 0, 0])
    FORM_A.check_dense_ratio([64, 0, 0])
    # A worker with a dense share: the failure that shows up as an OOM on a
    # 3080 that was budgeted for experts alone.
    with pytest.raises(RankRoleError, match="gives them a dense share"):
        FORM_A.check_dense_ratio([39, 13, 12])
    with pytest.raises(RankRoleError, match="gives them a dense share"):
        FORM_A.check_dense_ratio([94, 3, 3])
    # A host with none: nobody computes attention.
    with pytest.raises(RankRoleError, match="nobody computes attention"):
        FORM_A.check_dense_ratio([0, 0, 0])
    with pytest.raises(RankRoleError, match="entries but"):
        FORM_A.check_dense_ratio([1, 0])


def test_role_of_out_of_range_is_refused():
    with pytest.raises(RankRoleError, match="outside the role vector"):
        FORM_A.role_of(3)


# ==========================================================================
# 2. Zero admission is a property of the installed PLAN
# ==========================================================================
def test_default_plan_is_unchanged_and_admits_no_zero():
    set_tp_partition_ratios([39, 13, 12])
    assert tp_partition_allows_zero() is False
    # 24 q packets x 256 head_dim, kv-group aligned -> today's 12/6/6.
    assert tp_partition_sizes(6144, 3, units=24, groups=2) == [3072, 1536, 1536]
    assert tp_partition_size(6144, 3, 0, units=24, groups=2) == 3072


def test_form_a_plan_gives_the_workers_nothing_of_every_dense_dimension():
    set_tp_partition_ratios(FORM_A.dense_ratio(), None, True)
    assert tp_partition_allows_zero() is True
    # q packets (24 heads x 256), kv-group aligned
    assert tp_partition_sizes(6144, 3, units=24, groups=2) == [6144, 0, 0]
    # GDN / linear_attn value heads (48 x 128)
    assert tp_partition_sizes(6144, 3, units=48) == [6144, 0, 0]
    # a plain proportional dimension (no units): o_proj / mixer / vocab
    assert tp_partition_sizes(2560, 3) == [2560, 0, 0]
    # ... and the offsets stay coherent: the workers start where the host ends
    from sglang.srt.distributed.utils import tp_partition_offset

    assert [tp_partition_offset(6144, 3, r, units=24) for r in range(3)] == [
        0,
        6144,
        6144,
    ]


def test_the_plan_flag_is_restored_by_the_scope_not_leaked():
    set_tp_partition_ratios([39, 13, 12])
    assert tp_partition_allows_zero() is False
    with scoped_tp_partition_ratios([1, 0, 0], None, True):
        assert tp_partition_allows_zero() is True
        assert tp_partition_sizes(2560, 3) == [2560, 0, 0]
    assert tp_partition_allows_zero() is False
    assert tp_partition_sizes(2560, 3) == [1560, 520, 480]


def test_a_family_may_not_overrule_a_zero_of_the_base_plan():
    """The MoE family legitimately differs from the base vector -- that is
    how Form A gives a worker experts while giving it no dense share. What it
    may NOT do is hand a worker a share of a DENSE family."""
    # legitimate: moe is not a dense family, so it is set separately and the
    # base zero still governs every dense dimension.
    with pytest.raises(ValueError, match="gave width zero"):
        set_tp_partition_ratios([1, 0, 0], {"mlp": [10, 1, 1]}, True)
    # the same family vector is fine when the base plan has no zeros.
    set_tp_partition_ratios([39, 13, 12], {"mlp": [10, 1, 1]}, False)
    assert get_tp_partition_ratios("mlp") == [10, 1, 1]


def test_an_all_zero_family_is_refused_by_name():
    with pytest.raises(ValueError, match="all zeros"):
        set_tp_partition_ratios([1, 0, 0], {"mlp": [0, 0, 0]}, True)


def test_family_zero_still_refused_without_allow_zero():
    with pytest.raises(ValueError, match="must be positive"):
        set_tp_partition_ratios([39, 13, 12], {"mlp": [10, 0, 1]})


# ==========================================================================
# 3. DCP under a host-held KV
# ==========================================================================
def test_dcp_collapses_when_the_host_holds_the_whole_kv():
    r = resolve_dcp_under_host_kv(FORM_A, requested_dcp_size=3, requested_replicated=True)
    assert r.dcp_size == 1
    assert r.uneven_dcp_kv_replicated is False
    assert "attention host" in r.reason and "no LSE merge" in r.reason


def test_an_explicit_dcp_flag_is_refused_not_overridden():
    """Silently ignoring an operator's flag is how an A/B ends up comparing
    two different layouts."""
    with pytest.raises(RankRoleError, match="was set explicitly"):
        resolve_dcp_under_host_kv(FORM_A, 3, None, forced=True)
    with pytest.raises(RankRoleError, match="nothing to replicate"):
        resolve_dcp_under_host_kv(FORM_A, 1, True, forced=True)
    # forced with a consistent request is fine
    assert resolve_dcp_under_host_kv(FORM_A, 1, False, forced=True).dcp_size == 1


# ==========================================================================
# 4. The seam registry -- named refusals for what is not built
# ==========================================================================
WIRED = ("F1", "F2", "F7", "F8")
UNWIRED = ("F3", "F4", "F5", "F6", "F9", "F10", "F11")


def test_every_seam_has_an_identity_a_place_and_a_verdict():
    """Eleven, not the nine the design note started with: the slice-2 survey
    of the call sites found two more, and a seam that is known but unlisted
    is worse than one that was never looked for."""
    assert set(SEAMS) == {f"F{i}" for i in range(1, 12)}
    for sid, seam in SEAMS.items():
        assert seam.id == sid
        assert ":" in seam.where or ".py" in seam.where, sid
        assert seam.what and seam.note, sid
    assert set(WIRED) | set(UNWIRED) == set(SEAMS)
    assert {s for s, seam in SEAMS.items() if seam.wired} == set(WIRED)


def test_wired_seams_pass_and_unwired_seams_refuse_with_their_own_name():
    for sid in WIRED:
        require_wired(sid)  # must not raise
    for sid in UNWIRED:
        with pytest.raises(FormASeamNotWired) as e:
            require_wired(sid)
        assert sid in str(e.value)
        assert SEAMS[sid].where.split(",")[0] in str(e.value)


def test_the_remaining_work_is_ordered_and_complete():
    from sglang.srt.rank_role import UNWIRED_ORDER

    assert set(UNWIRED_ORDER) == set(UNWIRED)
    assert len(UNWIRED_ORDER) == len(set(UNWIRED_ORDER))
    # F3 first: it is both the largest VRAM gain and the precondition of F11.
    assert UNWIRED_ORDER[0] == "F3"


def test_the_two_seams_the_survey_added_carry_their_evidence():
    """F10 and F11 are the ones a reader will doubt, so they must point at
    the exact line that makes them real."""
    assert "launcher.py:7203" in SEAMS["F10"].where
    assert "cannot happen" in SEAMS["F10"].where
    assert "linear.py:2069" in SEAMS["F11"].where
    assert "1790" in SEAMS["F11"].where  # the 0 % 8 == 0 activation guard


def test_unknown_seam_is_refused():
    with pytest.raises(RankRoleError, match="unknown Form A seam"):
        require_wired("F42")


def test_guards_fire_only_on_the_rank_whose_role_needs_them():
    # The host is never guarded by the worker guards.
    guard_dense_weights(FORM_A, 0)
    guard_draft_worker(FORM_A, 0)
    guard_kv_pool(FORM_A, 0, tokens=262151)
    # A worker is.
    with pytest.raises(FormASeamNotWired, match="F3"):
        guard_dense_weights(FORM_A, 1)
    with pytest.raises(FormASeamNotWired, match="F3"):
        guard_draft_worker(FORM_A, 2)
    with pytest.raises(FormASeamNotWired, match="F4"):
        guard_kv_pool(FORM_A, 1, tokens=1)
    # ... but a worker with zero KV tokens is exactly what Form A wants, so
    # that case must NOT refuse.
    guard_kv_pool(FORM_A, 1, tokens=0)


def test_eager_is_an_allowed_worker_graph_mode_and_a_captured_one_is_not():
    """R6/F9 is deferrable precisely because the first probe boot may run
    decode eager; what must refuse is a CAPTURED dense graph on a rank that
    holds no dense weights."""
    guard_graph_mode(FORM_A, 1, "eager")
    guard_graph_mode(FORM_A, 1, "disabled")
    guard_graph_mode(FORM_A, 0, "full")
    with pytest.raises(FormASeamNotWired, match="F9"):
        guard_graph_mode(FORM_A, 1, "full")


def test_a_zero_width_linear_is_only_guarded_when_it_is_actually_zero():
    """F11's backstop. A positive width is the normal case and must cost
    nothing; a zero width must refuse, because F.linear with K=0 returns
    zeros without raising and the all-reduce adds them."""
    from sglang.srt.rank_role import guard_zero_width_linear

    guard_zero_width_linear(FORM_A, 0, "o_proj", 2560)
    guard_zero_width_linear(FORM_A, 1, "o_proj", 2560)
    with pytest.raises(FormASeamNotWired, match="F11"):
        guard_zero_width_linear(FORM_A, 1, "o_proj", 0)


def test_subgroup_and_dcp_merge_guards_name_their_seams():
    with pytest.raises(FormASeamNotWired, match="F6"):
        guard_collective_subgroup(FORM_A, "dense_all_reduce")
    with pytest.raises(FormASeamNotWired, match="F5"):
        guard_dcp_merge(FORM_A, 0)


def test_form_a_plan_shares_one_definition_of_the_host_with_the_role_vector():
    """Two copies of 'exactly one host' are two rules the moment one is
    edited, so the solver delegates to RankRolePlan."""
    from sglang.srt.form_a_plan import (
        CardBudget,
        ExpertGeometry,
        FormAGeometryInvalid,
        MeasuredPosts,
        RIG_CARDS,
        solve_form_a,
    )

    cards = RIG_CARDS()
    two_hosts = [
        cards[0],
        CardBudget(1, "x", 20480, 17800, 1400, 6.5, HOST),
        cards[2],
    ]
    with pytest.raises(FormAGeometryInvalid, match="exactly ONE attention host"):
        solve_form_a(two_hosts, MeasuredPosts.from_fn8ah(), ExpertGeometry.qwen4_exp())
    # and the role vector of a valid plan round-trips into a RankRolePlan
    plan = solve_form_a(cards, MeasuredPosts.from_fn8ah(), ExpertGeometry.qwen4_exp())
    assert RankRolePlan(tuple(c.role for c in plan.cards)).host_rank == 0
