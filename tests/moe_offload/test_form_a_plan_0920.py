# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for the Form A solve and the zero-width partition seam.

No CUDA, no driver, no model, no NVML. Everything the solve consumes is
injected, so what is pinned here is the CONTRACT, not a table of numbers:

  * the SUBTRACTION order: the attention host pays for dense + undivided KV
    + unsharded draft + GDN states + speculative state + runtime + corridor,
    a worker pays runtime + corridor + row buffer and NOTHING else. Getting
    that order wrong is the whole way Form A can look better than it is;
  * DIRECTION of the spill split: the x4 card is handed FEWER non-resident
    experts than the x8 card of the same capacity -- a spilled expert is
    paid for on the lane that fetches it. A future reader "fixing" this into
    a capacity-proportional split is the failure mode, so it gets a test;
  * the ownership vector covers every expert EXACTLY once (the MoE reduce
    sums a hole otherwise) and no rank owns zero;
  * every impossibility is a NAMED refusal carrying its numbers, and each
    refusal class is distinguishable from the others without parsing prose;
  * Form A is measured against the layout it replaces: the residency ceiling
    it produces on this rig is compared with the 301/515 fn8ah actually ran,
    and the comparison is asserted in the direction the design claims;
  * partition_units(allow_zero=True): a rank with weight 0 owns NOTHING and
    the remaining ranks split byte-identically to a call that never
    mentioned it -- while the DEFAULT path keeps its minimum of one unit per
    rank, unchanged, because there a zero is a bug and not a layout.
"""

import pytest

from sglang.srt.distributed.utils import partition_sizes, partition_units
from sglang.srt.form_a_plan import (
    CardBudget,
    ExpertGeometry,
    FormAGeometryInvalid,
    FormAHostOverBudget,
    FormAOwnershipMismatch,
    FormAWorkerOverBudget,
    MeasuredPosts,
    RIG_CARDS,
    solve_form_a,
)

# --------------------------------------------------------------------------
# What fn8ah actually ran, as the yardstick Form A has to beat.
# ct-stream-presplit resident buffers: 147 / 75 / 79 experts per layer.
# --------------------------------------------------------------------------
FN8AH_RESIDENT_PER_LAYER = 147 + 75 + 79  # = 301 of 512 routed experts


def _plan(**kw):
    return solve_form_a(
        kw.pop("cards", RIG_CARDS(**kw.pop("budgets", {}))),
        kw.pop("posts", MeasuredPosts.from_fn8ah()),
        kw.pop("geometry", ExpertGeometry.qwen4_exp()),
    )


# ==========================================================================
# 1. The posts, and the subtraction order
# ==========================================================================
def test_host_pays_for_every_dense_post_and_the_worker_for_none_of_them():
    posts = MeasuredPosts.from_fn8ah()
    plan = _plan()
    host_fixed, w1_fixed, w2_fixed = plan.fixed_gib

    assert w1_fixed == pytest.approx(w2_fixed)
    # A worker carries runtime + corridor + row buffer + its ROUTER, and
    # nothing else. The router joined the list in slice 6a, when the worker
    # forward settled that a worker picks its own experts out of the
    # broadcast MoE input rather than being handed global topk ids. It is
    # not a new post -- fn8ah measured `moe_gate 0.12` on every rank -- it
    # is one the earlier plan wrongly booked away from the workers, worth
    # about one residence row per worker per layer.
    assert w1_fixed == pytest.approx(
        posts.worker_runtime_gib
        + posts.corridor_gib
        + posts.dispatch_buffer_gib
        + posts.worker_router_gib
    )
    assert posts.worker_router_gib == pytest.approx(0.12)
    # The host carries all seven posts, itemised.
    assert set(plan.host_breakdown) == {
        "dense",
        "kv",
        "draft",
        "gdn_states",
        "spec_state",
        "runtime",
        "corridor",
    }
    assert host_fixed == pytest.approx(sum(plan.host_breakdown.values()))
    # ... and every dense post really is on the host side only.
    for post in ("dense", "kv", "draft", "gdn_states", "spec_state"):
        assert plan.host_breakdown[post] > 0
        assert plan.host_breakdown[post] not in (w1_fixed,)


def test_kv_is_the_whole_context_on_the_host_not_a_dcp_share():
    """Form A's single biggest new host post. fn8ah split 262151 tokens over
    three ranks by the DCP vector [11, 11, 10]; Form A puts all of it on the
    host, and the plan must show the UNDIVIDED figure."""
    posts = MeasuredPosts.from_fn8ah()
    assert posts.kv_bytes_per_token == 14143  # KV pool sizing line, fn8ah
    assert posts.context_tokens == 262151
    whole = 14143 * 262151 / 1024**3
    assert posts.kv_gib == pytest.approx(whole)
    assert posts.kv_gib == pytest.approx(3.45, abs=0.02)
    # The share rank 0 carried under DCP [11, 11, 10]:
    dcp_share = whole * 11 / 32
    assert _plan().host_breakdown["kv"] == pytest.approx(whole)
    assert whole - dcp_share == pytest.approx(2.27, abs=0.05)


def test_one_resident_slot_costs_every_layer_not_one():
    geom = ExpertGeometry.qwen4_exp()
    assert geom.slot_gib == pytest.approx(48 * 2.45 / 1024, rel=1e-6)
    # Cross-check against the measured boot: rank 0 held 147 experts/layer
    # in 16.65 GiB of device expert tensors.
    assert 16.65 / geom.slot_gib == pytest.approx(147, abs=4)


# ==========================================================================
# 2. Capacity, ceiling, ownership
# ==========================================================================
def test_capacity_is_what_is_left_after_the_posts_divided_by_the_slot():
    plan = _plan()
    geom = plan.geometry
    for i, c in enumerate(plan.cards):
        assert plan.expert_gib[i] == pytest.approx(c.budget_gib - plan.fixed_gib[i])
        assert plan.capacity[i] == int(plan.expert_gib[i] // geom.slot_gib)
        assert plan.capacity[i] > 0


def test_ownership_covers_every_expert_exactly_once():
    plan = _plan()
    assert sum(plan.owned) == plan.geometry.num_experts
    assert all(o > 0 for o in plan.owned)
    assert all(s == max(0, o - c) for s, o, c in zip(plan.spill, plan.owned, plan.capacity))


def test_spill_follows_the_LINK_not_the_capacity():
    """DIRECTION test. Ranks 1 and 2 are the same card with the same budget
    and therefore the same capacity; they differ only in their lane (x4 at
    6.5 GiB/s vs x8 at 13.3). The slow lane must be handed FEWER spilled
    experts, because a spilled expert costs a fetch over exactly that lane."""
    plan = _plan()
    r1, r2 = plan.cards[1], plan.cards[2]
    assert r1.link_gib_s < r2.link_gib_s
    assert plan.capacity[1] == plan.capacity[2]
    assert plan.spill[1] < plan.spill[2]
    assert plan.owned[1] < plan.owned[2]
    # ... and therefore the slow card is the MORE resident one.
    assert plan.resident_fraction[1] > plan.resident_fraction[2]


def test_a_capacity_proportional_spill_would_fail_this_file():
    """Companion pin: the property above CAN fail, so it is a real test."""
    plan = _plan()
    equal_lane = [
        CardBudget(c.rank, c.name, c.nameplate_mib, c.budget_mib, c.reserve_mib, 10.0, c.role)
        for c in plan.cards
    ]
    flat = solve_form_a(equal_lane, plan.posts, plan.geometry)
    assert flat.spill[1] >= flat.spill[2]  # no direction left once lanes tie


# ==========================================================================
# 3. The number Form A is bought for
# ==========================================================================
def test_residency_ceiling_beats_the_layout_it_replaces():
    plan = _plan()
    assert plan.residency_total > FN8AH_RESIDENT_PER_LAYER
    assert plan.residency_total == sum(min(c, o) for c, o in zip(plan.capacity, plan.owned))


def test_residency_ceiling_is_far_below_the_512_a_reader_might_assume():
    """The honest half of the same finding: Form A does NOT make the whole
    expert set resident. If this ever passes 512 the posts were dropped."""
    plan = _plan()
    assert plan.residency_total < plan.geometry.num_experts
    # Raising the worker budgets to the physical limit of a 20480 MiB card
    # (less its 1400 MiB user reserve) is the most Form A can ever have.
    best = solve_form_a(
        RIG_CARDS(host_budget_mib=30800, worker_budget_mib=19000),
        plan.posts,
        plan.geometry,
    )
    assert best.residency_total > plan.residency_total
    assert best.residency_total < plan.geometry.num_experts


def test_report_and_flags_name_every_rank():
    plan = _plan()
    flags = plan.flags()
    assert len(flags["--rank-moe-ratio"].split(",")) == len(plan.cards)
    assert len(flags["--rank-moe-resident-fraction"].split(",")) == len(plan.cards)
    assert all(0.0 < f <= 1.0 for f in plan.resident_fraction)
    text = plan.report()
    assert "RESIDENCY CEILING" in text
    assert "fn8ah" in text  # provenance travels with the numbers


# ==========================================================================
# 4. Refusals -- named, numbered, and distinguishable
# ==========================================================================
def test_host_over_budget_is_its_own_refusal_and_names_the_posts():
    with pytest.raises(FormAHostOverBudget) as e:
        solve_form_a(
            RIG_CARDS(host_budget_mib=8000),
            MeasuredPosts.from_fn8ah(),
            ExpertGeometry.qwen4_exp(),
        )
    msg = str(e.value)
    assert "rank 0" in msg and "host" in msg
    assert "kv" in msg and "dense" in msg and "draft" in msg


def test_worker_over_budget_is_a_different_refusal():
    with pytest.raises(FormAWorkerOverBudget) as e:
        solve_form_a(
            RIG_CARDS(worker_budget_mib=2000),
            MeasuredPosts.from_fn8ah(),
            ExpertGeometry.qwen4_exp(),
        )
    assert "rank 1" in str(e.value)
    assert not isinstance(e.value, FormAHostOverBudget)


def test_two_hosts_or_no_host_is_refused_by_name():
    cards = RIG_CARDS()
    two_hosts = [cards[0], CardBudget(1, "x", 20480, 17800, 1400, 6.5, "host"), cards[2]]
    with pytest.raises(FormAGeometryInvalid, match="exactly ONE attention host"):
        solve_form_a(two_hosts, MeasuredPosts.from_fn8ah(), ExpertGeometry.qwen4_exp())
    no_host = [CardBudget(c.rank, c.name, c.nameplate_mib, c.budget_mib, c.reserve_mib, c.link_gib_s) for c in cards]
    with pytest.raises(FormAGeometryInvalid):
        solve_form_a(no_host, MeasuredPosts.from_fn8ah(), ExpertGeometry.qwen4_exp())


def test_a_rank_that_would_own_nothing_is_refused():
    """Fewer experts than ranks: the give-back of surplus capacity must not
    quietly leave a rank owning zero -- a rank that owns nothing is not a
    Form A worker, it is a rank that should not be in the group."""
    geom = ExpertGeometry(num_experts=2, num_layers=1, expert_bytes=1024)
    with pytest.raises(FormAOwnershipMismatch, match="no expert at all"):
        solve_form_a(RIG_CARDS(), MeasuredPosts.from_fn8ah(), geom)


def test_empty_geometry_is_refused():
    with pytest.raises(FormAGeometryInvalid):
        solve_form_a(
            RIG_CARDS(),
            MeasuredPosts.from_fn8ah(),
            ExpertGeometry(num_experts=0, num_layers=48, expert_bytes=1),
        )


def test_a_single_card_is_not_form_a():
    with pytest.raises(FormAGeometryInvalid, match="at least one host and one worker"):
        solve_form_a(
            RIG_CARDS()[:1], MeasuredPosts.from_fn8ah(), ExpertGeometry.qwen4_exp()
        )


# ==========================================================================
# 5. The zero-width partition seam (distributed/utils.py)
# ==========================================================================
def test_default_partition_still_gives_every_rank_at_least_one_unit():
    """Byte-identical pin: without allow_zero nothing changed.

    The two spellings of today's plan, both pinned, because they differ and
    only one of them is the one the rig runs: the RAW split of 24 q packets
    over --rank-tp-ratio 39,13,12 is 15/5/4, and the kv-group-aligned split
    (#116, groups = num_key_value_heads = 2) is the 12/6/6 the boot log
    shows. allow_zero must disturb neither."""
    assert partition_units(24, [39, 13, 12]) == [15, 5, 4]
    assert partition_units(24, [39, 13, 12], groups=2) == [12, 6, 6]
    assert partition_units(24, [94, 3, 3]) == [22, 1, 1]
    # A zero weight on the default path still gets its minimum of one.
    assert partition_units(24, [1, 0, 0]) == [22, 1, 1]
    assert sum(partition_units(24, [1, 0, 0])) == 24


def test_allow_zero_gives_an_expert_only_rank_no_head_at_all():
    """Form A: rank 1 and 2 run no attention, so they own no q packet."""
    assert partition_units(24, [1, 0, 0], allow_zero=True) == [24, 0, 0]
    assert partition_units(48, [3, 0, 1], allow_zero=True) == [36, 0, 12]
    # The ranks that DO own units split exactly as if the empty rank had
    # never been in the vector.
    assert partition_units(48, [3, 0, 1], allow_zero=True) == [
        partition_units(48, [3, 1])[0],
        0,
        partition_units(48, [3, 1])[1],
    ]


def test_allow_zero_reaches_through_partition_sizes():
    # 24 heads x 256 head_dim = 6144 elements, split over q packets.
    assert partition_sizes(6144, [1, 0, 0], units=24, allow_zero=True) == [6144, 0, 0]
    assert partition_sizes(6144, [1, 0, 0], units=24) == [5632, 256, 256]


def test_allow_zero_composes_with_the_kv_group_alignment():
    """Slice 1 refused this combination by name; slice 2 makes it compose,
    because refusing it would have made Form A unbootable at the FIRST
    attention layer -- the q split passes groups=num_key_value_heads.

    An empty rank straddles no kv-head-group boundary, so the aligned split
    over the ranks that DO own packets, with the zeros put back, is the same
    answer the alignment would have given had the empty rank never existed.
    """
    # Form A proper: one host, every packet, alignment trivially satisfied.
    assert partition_units(24, [1, 0, 0], groups=2, allow_zero=True) == [24, 0, 0]
    # A four-rank plan with one worker reproduces today's aligned 12/6/6
    # over the three ranks that are left.
    assert partition_units(24, [39, 13, 12, 0], groups=2, allow_zero=True) == [
        12,
        6,
        6,
        0,
    ]
    assert partition_units(24, [39, 13, 12], groups=2) == [12, 6, 6]


def test_allow_zero_refuses_an_all_zero_vector_by_name():
    with pytest.raises(ValueError, match="all "):
        partition_units(24, [0, 0, 0], allow_zero=True)


def test_negative_weight_is_refused_not_silently_dropped():
    with pytest.raises(ValueError, match="negative weight"):
        partition_units(24, [1, -1, 0], allow_zero=True)


def test_partition_sizes_refuses_a_zero_sum_vector_instead_of_dividing_by_zero():
    with pytest.raises(ValueError, match="sums to 0"):
        partition_sizes(6144, [0, 0, 0])
