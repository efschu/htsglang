# SPDX-License-Identifier: Apache-2.0
"""Slice 4: seam A as a plan object, against stub layout tables."""
from __future__ import annotations

import pytest

from sglang.srt.flip_kv_remap import (
    DISPOSITION_LEG,
    DISPOSITION_REMAP,
    FULL_ATTENTION_LAYERS,
    LINK_GB_S,
    PP3_BYTES_PER_TOKEN_PER_LAYER,
    plan_kv_remap,
)
from sglang.srt.flip_nextflash_plan import (
    FORM_A_KV,
    PP3_KV,
    KvLayout,
    Weg2FlipKvRelayInfeasible,
)

_GB = 1e9


# --------------------------------------------------------------------------
# The measured geometry must reproduce the design's numbers
# --------------------------------------------------------------------------
def test_the_per_layer_cell_is_the_pp3_cells_divided_losslessly():
    """7616 + 3264 + 2176 = 13056 = 12 x 1088. If this ever fails, the PP
    split stopped being lossless and every byte figure moved."""
    assert sum(PP3_KV.cells) == 13056
    assert 13056 == FULL_ATTENTION_LAYERS * PP3_BYTES_PER_TOKEN_PER_LAYER


def test_the_form_a_cell_is_never_compared_with_the_pp3_sum():
    """14143 = 13056 + 1087 -- the 1087 is the solo draft's KV term and the
    undivided mamba amortisation, destination-only terms with no source
    pages. The plan's moved bytes must come from the PER-LAYER cell."""
    assert FORM_A_KV.cells[0] - sum(PP3_KV.cells) == 1087
    plan = plan_kv_remap()
    assert plan.moved_bytes == 13056 * 262144  # not 14143 * 262144


def test_seven_layers_are_already_on_the_destination_card():
    """PP stage 0 sits on the 5090 and so does the Form-A host."""
    plan = plan_kv_remap()
    remaps = [m for m in plan.moves if m.disposition == DISPOSITION_REMAP]
    assert len(remaps) == 7
    assert all(m.src_rank == 0 and m.dst_rank == 0 for m in remaps)


def test_only_five_layers_cross_a_link_and_they_are_1_43_GB():
    plan = plan_kv_remap()
    legs = [m for m in plan.moves if m.disposition == DISPOSITION_LEG]
    assert len(legs) == 5
    assert plan.legged_bytes / _GB == pytest.approx(1.43, abs=0.01)
    # and the full pool would have been 3.42 GB
    assert plan.moved_bytes / _GB == pytest.approx(3.42, abs=0.01)


def test_the_two_legs_split_3_and_2_over_their_own_lanes():
    plan = plan_kv_remap()
    by_src = {}
    for m in plan.moves:
        if m.disposition == DISPOSITION_LEG:
            by_src[m.src_rank] = by_src.get(m.src_rank, 0) + 1
    assert by_src == {1: 3, 2: 2}
    assert plan.per_lane_seconds[1] == pytest.approx(0.856e9 / (6.5e9), abs=0.01)
    assert plan.per_lane_seconds[2] == pytest.approx(0.570e9 / (13.3e9), abs=0.01)


def test_the_seam_is_the_slowest_lane_not_the_sum():
    """Lanes are disjoint and run in parallel. Summing overstates seam A by
    ~2x and points the optimisation at transport, which design §3 says is not
    where the flip's time is."""
    plan = plan_kv_remap()
    assert plan.seam_seconds == max(plan.per_lane_seconds.values())
    assert plan.seam_seconds < sum(plan.per_lane_seconds.values())
    # ~0.13 s, comfortably under the ~2 s physics floor
    assert plan.seam_seconds == pytest.approx(0.13, abs=0.02)


def test_a_remap_costs_no_transport_time():
    plan = plan_kv_remap()
    for m in plan.moves:
        if m.disposition == DISPOSITION_REMAP:
            assert m.seconds == 0.0
            assert m.link_gb_s is None


def test_the_link_rates_are_the_measured_ones_not_hand_numbers():
    """Memory RANG-LINK-ZUORDNUNG: rank 1 is x4, ranks 0 and 2 are x8."""
    assert LINK_GB_S == {0: 14.4, 1: 6.5, 2: 13.3}


def test_every_layer_is_accounted_for_exactly_once():
    plan = plan_kv_remap()
    assert len(plan.moves) == FULL_ATTENTION_LAYERS
    assert sorted(m.layer_index for m in plan.moves) == list(range(12))


def test_the_report_states_remap_and_leg_separately():
    text = plan_kv_remap().report()
    assert "remapped (device-local, no link)" in text
    assert "PARALLEL" in text


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------
def test_an_infeasible_destination_is_W113_before_any_disposition():
    """Scheduling a move into a pool that does not fit is a plan for an OOM."""
    tiny = KvLayout(
        name="tiny host",
        boot="stub",
        cells=(14143, 768, 768),
        avail_bytes=(1 << 20, 1 << 20, 1 << 20),
        full_attn_per_rank=(12, 0, 0),
    )
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        plan_kv_remap(dst=tiny)
    assert "binds at" in str(exc.value)


def test_a_source_that_does_not_cover_every_layer_is_refused():
    """An unwritten KV page is a VALID page -- the hole would read as zeros
    on the first decode after the flip, silently."""
    short = KvLayout(
        name="PP3 short",
        boot="stub",
        cells=(7616, 3264, 2176),
        avail_bytes=PP3_KV.avail_bytes,
        full_attn_per_rank=(7, 3, 1),  # 11, not 12
    )
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        plan_kv_remap(src=short)
    msg = str(exc.value)
    assert "11 full-attention layer(s)" in msg
    assert "read as zeros" in msg


def test_a_destination_with_two_carriers_is_refused():
    two = KvLayout(
        name="two hosts",
        boot="stub",
        cells=(14143, 14143, 768),
        avail_bytes=(8 << 30, 8 << 30, 8 << 30),
        full_attn_per_rank=(6, 6, 0),
    )
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        plan_kv_remap(dst=two)
    assert "exactly ONE attention host" in str(exc.value)


def test_a_destination_holding_fewer_than_all_layers_is_a_different_layout():
    partial = KvLayout(
        name="partial host",
        boot="stub",
        cells=(14143, 768, 768),
        avail_bytes=(8 << 30, 2 << 30, 2 << 30),
        full_attn_per_rank=(11, 0, 0),
    )
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        plan_kv_remap(dst=partial, total_full_attention_layers=11)
    # src still says 12 -> caught on the source check first
    assert "full-attention layer" in str(exc.value)


def test_an_unrated_lane_is_refused_rather_than_costed_as_free():
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        plan_kv_remap(link_gb_s={0: 14.4})  # ranks 1 and 2 unrated
    msg = str(exc.value)
    assert "no measured link rate" in msg
    assert "look free" in msg


def test_a_non_cell_is_refused():
    with pytest.raises(Weg2FlipKvRelayInfeasible):
        plan_kv_remap(bytes_per_token_per_layer=0)


# --------------------------------------------------------------------------
# The trap the plan exists to avoid
# --------------------------------------------------------------------------
def test_a_layout_where_the_host_is_not_pp0_legs_everything():
    """The 1.43 GB saving is not a property of the seam -- it is a property
    of PP0 and the Form-A host being the SAME card. Move the host and the
    whole 3.42 GB crosses links."""
    elsewhere = KvLayout(
        name="host on rank 1",
        boot="stub",
        cells=(768, 14143, 768),
        avail_bytes=(2 << 30, 8 << 30, 2 << 30),
        full_attn_per_rank=(0, 12, 0),
    )
    plan = plan_kv_remap(dst=elsewhere)
    assert all(m.disposition == DISPOSITION_LEG for m in plan.moves if m.src_rank != 1)
    assert plan.legged_bytes / _GB == pytest.approx(2.56, abs=0.02)  # 7+2 layers
    assert plan.seam_seconds > plan_kv_remap().seam_seconds


def test_the_device_local_layers_never_get_a_copy_disposition():
    """Memory S6-REMAP-STATT-ALLOKATION. A copy of PP0's 7 layers needs a
    transient 1.99 GB on a 5090 that has 0.24 GiB free under extend -- an OOM
    with a schedule attached. The disposition is assigned, never accepted."""
    plan = plan_kv_remap()
    local = [m for m in plan.moves if m.src_rank == m.dst_rank]
    assert local and all(m.disposition == DISPOSITION_REMAP for m in local)
    assert plan.remapped_bytes / _GB == pytest.approx(1.99, abs=0.01)
