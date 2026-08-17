"""#704b B1: the decoupled-KV group -- membership, manifest, routing, refusal.

Built on the #616 survey's findings rather than my earlier draft:

* there are NO typed group classes -- a "type" here is a module global, a named
  getter and a routing flag (`parallel_state.py:564`, `:2550`);
* `initialize_phase_flip_secondary_groups` (`:3466+`) ALREADY creates a DCP
  group in the same world as a PP group, with a fixed plan and a world-wide
  manifest all_gather + equality check BEFORE anything is created. B1 reuses
  that pattern instead of inventing one;
* B1 gets its OWN flag rather than reusing `dcp_size`, so
  `ParallelContext.dcp_enabled` stays False on PP prefill ranks and the
  invariant `dcp_group_guard.py:38-42` documents remains true.

Hermetic: pure functions plus module-global manipulation. No CUDA, no
torch.distributed, no process group.
"""

import pytest
from sglang.srt.distributed import parallel_state as ps

# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------


def test_membership_is_the_pipeline_not_the_tp_slice():
    """KV is token-sharded across ONE pipeline's ranks."""
    # tp=1, pp=3: a single pipeline over the whole world.
    assert ps.plan_decoupled_kv_ranks(3, 1, 3) == [[0, 1, 2]]
    # tp=2, pp=3: one group per TP position, each spanning its own stages.
    assert ps.plan_decoupled_kv_ranks(6, 2, 3) == [[0, 2, 4], [1, 3, 5]]


def test_the_plan_matches_the_PP_group_layout_exactly():
    """B1's membership IS the PP layout, so it must reproduce it bit for bit.

    `initialize_model_parallel` builds PP as
    `range(idx, world_size, world_size // pp_size)`. If B1 computed anything
    else the two would disagree about who is in the pipeline.
    """
    for world, tp, pp in ((3, 1, 3), (6, 2, 3), (8, 2, 4), (4, 4, 1)):
        num_pp_groups = world // pp
        expected = [
            list(range(idx, world, num_pp_groups)) for idx in range(num_pp_groups)
        ]
        assert ps.plan_decoupled_kv_ranks(world, tp, pp) == expected


def test_the_plan_does_not_read_the_PP_group():
    """ORDERING, and it is the bug the #616 survey called out.

    `_DCP` is created before `_PP`, so at B1's creation point `_PP` does not
    exist. The plan must therefore be computed inline. Forcing `_PP` to None
    must not change the answer.
    """
    saved = ps._PP
    try:
        ps._PP = None
        assert ps.plan_decoupled_kv_ranks(6, 2, 3) == [[0, 2, 4], [1, 3, 5]]
    finally:
        ps._PP = saved


def test_a_world_that_does_not_factor_is_refused():
    with pytest.raises(ValueError, match="does not factor"):
        ps.plan_decoupled_kv_ranks(7, 2, 3)
    with pytest.raises(ValueError, match="positive"):
        ps.plan_decoupled_kv_ranks(0, 1, 1)


# --------------------------------------------------------------------------
# Manifest agreement -- the phase-flip precedent, reused
# --------------------------------------------------------------------------


def test_agreeing_manifests_pass():
    m = ps.decoupled_kv_manifest([[0, 1, 2]])
    ps.check_manifest_agreement(m, [m, m, m], "DECOUPLED-KV")


def test_a_DIVERGENT_plan_is_caught_before_anything_is_created():
    """CAN-FAIL: a rank that planned a different order must abort the create.

    A divergent create order is the rank-divergent-collective family; dying
    here is the cheap failure, and a half-formed communicator that hangs a
    later collective with no attribution is the expensive one.
    """
    mine = ps.decoupled_kv_manifest([[0, 1, 2]])
    theirs = ps.decoupled_kv_manifest([[0, 2, 1]])  # same members, wrong ORDER
    with pytest.raises(RuntimeError, match="DIVERGES"):
        ps.check_manifest_agreement(mine, [mine, theirs, mine], "DECOUPLED-KV")


def test_the_salt_proves_the_check_can_fail():
    """The precedent carries `_manifest_salt` solely so a test can prove this."""
    a = ps.decoupled_kv_manifest([[0, 1, 2]], salt=0)
    b = ps.decoupled_kv_manifest([[0, 1, 2]], salt=1)
    assert a != b
    with pytest.raises(RuntimeError, match="DIVERGES"):
        ps.check_manifest_agreement(a, [a, b], "DECOUPLED-KV")


def test_the_divergence_message_names_every_rank():
    """An operator must see WHICH rank disagreed, not just that one did."""
    mine = ps.decoupled_kv_manifest([[0, 1]])
    theirs = ps.decoupled_kv_manifest([[1, 0]])
    with pytest.raises(RuntimeError) as ei:
        ps.check_manifest_agreement(mine, [mine, theirs], "DECOUPLED-KV")
    assert "0:" in str(ei.value) and "1:" in str(ei.value)


# --------------------------------------------------------------------------
# The #616 labeled gap, closed loudly
# --------------------------------------------------------------------------


def test_pp_and_dcp_together_are_refused_by_name():
    with pytest.raises(RuntimeError, match="dcp_enabled would report True"):
        ps.refuse_pp_dcp_combination(3, 3)


def test_either_alone_is_fine():
    ps.refuse_pp_dcp_combination(3, 1)  # PP only -- our path
    ps.refuse_pp_dcp_combination(1, 3)  # DCP only -- the TP decode path
    ps.refuse_pp_dcp_combination(1, 1)


# --------------------------------------------------------------------------
# Routing precedence -- first-match is the failure mode, so pin the order
# --------------------------------------------------------------------------


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
    yield
    (
        ps._PHASE_FLIP_TP_ACTIVE,
        ps._FLIP_DCP,
        ps._DCP_SPILL_ACTIVE,
        ps._DCP_SPILL,
        ps._DECOUPLED_KV_ACTIVE,
        ps._DECOUPLED_KV,
        ps._DCP,
    ) = saved


def test_b1_inactive_is_byte_identical_to_the_old_fall_through(routes):
    assert ps.get_dcp_group().name == "primary"


def test_b1_active_claims_only_the_fall_through(routes):
    ps.set_decoupled_kv_active(True)
    assert ps.get_dcp_group().name == "b1"


def test_the_flip_route_still_outranks_b1(routes):
    """B1 is a PP PREFILL mechanism and cannot be legitimately active during
    the TP decode phase. If it captured that forward the flip would read KV
    through the wrong communicator."""
    ps.set_decoupled_kv_active(True)
    ps._PHASE_FLIP_TP_ACTIVE = True
    assert ps.get_dcp_group().name == "flip"


def test_the_spill_route_still_outranks_b1(routes):
    """A spill forward keeps its dedicated serial communicator."""
    ps.set_decoupled_kv_active(True)
    ps._DCP_SPILL_ACTIVE = True
    assert ps.get_dcp_group().name == "spill"


def test_the_full_precedence_chain_in_one_table(routes):
    """flip -> spill -> b1 -> primary, pinned as a table.

    Reading order is not evidence: a future edit that moves the B1 branch up
    would silently capture flip and spill forwards, and only this table would
    notice.
    """
    cases = [
        ((True, True, True), "flip"),
        ((False, True, True), "spill"),
        ((False, False, True), "b1"),
        ((False, False, False), "primary"),
    ]
    for (flip, spill, b1), expected in cases:
        ps._PHASE_FLIP_TP_ACTIVE = flip
        ps._DCP_SPILL_ACTIVE = spill
        ps.set_decoupled_kv_active(b1)
        assert ps.get_dcp_group().name == expected, (flip, spill, b1)


def test_an_armed_flag_with_no_group_falls_through(routes):
    """Arming without initializing must not crash the routing."""
    ps._DECOUPLED_KV = None
    ps.set_decoupled_kv_active(True)
    assert ps.get_dcp_group().name == "primary"


def test_the_getter_refuses_when_uninitialized(routes):
    ps._DECOUPLED_KV = None
    assert ps.get_decoupled_kv_group_no_assert() is None
    with pytest.raises(AssertionError, match="not initialized"):
        ps.get_decoupled_kv_group()
