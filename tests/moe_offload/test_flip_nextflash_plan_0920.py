# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the Next-Flash P/D flip seams (W113/W114/W115).

No torch, no NVML, no device. Every number in here is either injected by the
test or comes from the two measured boots named in ``flip_nextflash_plan``'s
docstring; a test that hard-codes a figure states which log line it came from
so a stale number is visible as a stale boot tag.
"""

import pytest

from sglang.srt.flip_nextflash_plan import (
    FORM_A_KV,
    FORM_A_PINNED_GIB,
    HOST_MARK_GIB,
    PP3_KV,
    PP3_PINNED_GIB,
    CarriedState,
    HostPoolPost,
    KvLayout,
    Weg2FlipDraftStateOrphaned,
    Weg2FlipHostPoolDoubled,
    Weg2FlipKvRelayInfeasible,
    solve_host_pool,
    solve_kv_relay,
    solve_state_carry,
)

CONTEXT_262K = 262144


# --------------------------------------------------------------------------
# Seam (a) -- W113, the KV re-lay
# --------------------------------------------------------------------------
def test_measured_cells_match_the_boot_logs():
    # fn7s lines 263-265; fnFA19 lines 1000-1002. If these ever drift, the
    # defaults were edited without re-reading the log.
    assert PP3_KV.cells == (7616, 3264, 2176)
    assert FORM_A_KV.cells == (14143, 768, 768)
    # The destination cell is NOT the sum of the source cells: 13056 != 14143.
    # The 1087 B difference is the solo draft's KV plus the undivided mamba
    # amortisation, which the source splits differently.
    assert sum(PP3_KV.cells) == 13056
    assert FORM_A_KV.cells[0] - sum(PP3_KV.cells) == 1087


def test_form_a_host_holds_262k_with_headroom():
    plan = solve_kv_relay(PP3_KV, FORM_A_KV, CONTEXT_262K)
    # 7171801088 // 14143 = 507,091 tokens on the host.
    assert plan.dst_binding_rank == 0
    assert plan.dst_binding_tokens == 7171801088 // 14143
    assert plan.headroom_ratio > 1.9
    # ~3.45 GiB of KV, undivided, on the 5090.
    assert 3.4 < plan.dst_bytes[0] / 1024**3 < 3.5
    # The workers hold a stub cell only.
    assert plan.dst_bytes[1] == plan.dst_bytes[2] == 768 * CONTEXT_262K


def test_carrier_moves_only_the_full_attention_ranks():
    plan = solve_kv_relay(PP3_KV, FORM_A_KV, CONTEXT_262K)
    # All three PP stages carry full attention ([7,3,2]), so all three ship.
    assert PP3_KV.carrying_ranks == (0, 1, 2)
    assert plan.carrier_bytes_per_token == 13056
    assert plan.moved_bytes == 13056 * CONTEXT_262K
    assert "KV RE-LAY" in plan.report()


def test_form_a_workers_carry_no_full_attention():
    assert FORM_A_KV.carrying_ranks == (0,)


def test_w113_when_the_host_is_too_small():
    # Same geometry, but the host was profiled only 2 GiB of KV budget.
    starved = KvLayout(
        name="Form A, starved host",
        boot="synthetic",
        cells=(14143, 768, 768),
        avail_bytes=(2 * 1024**3, 2420113408, 2208301056),
        full_attn_per_rank=(12, 0, 0),
    )
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        solve_kv_relay(PP3_KV, starved, CONTEXT_262K)
    msg = str(exc.value)
    assert "W113 Weg2FlipKvRelayInfeasible" in msg
    # The refusal must name the shortfall in GiB, not just say "no".
    assert "GiB" in msg and "262144" in msg


def test_w113_rejects_a_ragged_layout():
    with pytest.raises(Weg2FlipKvRelayInfeasible):
        KvLayout(
            name="ragged",
            boot="synthetic",
            cells=(1, 2, 3),
            avail_bytes=(1, 2),
            full_attn_per_rank=(1, 1, 1),
        )


def test_w113_rejects_a_zero_context():
    with pytest.raises(Weg2FlipKvRelayInfeasible):
        solve_kv_relay(PP3_KV, FORM_A_KV, 0)


# --------------------------------------------------------------------------
# Seam (b) -- W114, the shared page-locked host expert pool
# --------------------------------------------------------------------------
def _six_processes(anon_gib: float = 6.5):
    posts = []
    for rank, gib in enumerate(PP3_PINNED_GIB):
        posts.append(HostPoolPost("PP3", rank, gib, anon_gib))
    for rank, gib in enumerate(FORM_A_PINNED_GIB):
        posts.append(HostPoolPost("FormA", rank, gib, anon_gib))
    return posts


def test_measured_pinned_pools_match_the_boot_logs():
    # fn7s lines 242-244 / fnFA19 lines 989-991.
    assert sum(PP3_PINNED_GIB) == pytest.approx(31.88, abs=0.01)
    assert sum(FORM_A_PINNED_GIB) == pytest.approx(38.86, abs=0.01)


def test_six_private_pools_blow_the_host_mark():
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        solve_host_pool(_six_processes(), shared=False)
    msg = str(exc.value)
    assert "W114 Weg2FlipHostPoolDoubled" in msg
    assert "PER-PROCESS" in msg or "per-process" in msg.lower()
    # 31.88 + 38.86 = 70.74 GiB pinned, plus 6 x 6.5 = 39 GiB anon.
    assert "70.74" in msg
    assert "109.74" in msg


def test_sharing_takes_the_pinned_term_to_the_per_rank_maximum():
    # Shared: max(22.73, 20.62) + max(5.30, 7.36) + max(3.85, 10.88)
    #       = 22.73 + 7.36 + 10.88 = 40.97 GiB
    ledger = solve_host_pool(_six_processes(anon_gib=0.0), shared=True)
    assert ledger.pinned_gib == pytest.approx(40.97, abs=0.01)
    assert ledger.saved_gib == pytest.approx(70.74 - 40.97, abs=0.02)
    assert ledger.total_gib < HOST_MARK_GIB
    assert "SHARED" in ledger.report()


def test_sharing_alone_is_not_enough_at_the_measured_anon_term():
    # 40.97 pinned + 6 x 6.5 anon = 79.97 -- under 88, but only just.
    ledger = solve_host_pool(_six_processes(anon_gib=6.5), shared=True)
    assert ledger.total_gib == pytest.approx(79.97, abs=0.02)
    assert ledger.total_gib < HOST_MARK_GIB
    # One more GiB of anonymous footprint per rank and it is over.
    with pytest.raises(Weg2FlipHostPoolDoubled) as exc:
        solve_host_pool(_six_processes(anon_gib=8.0), shared=True)
    assert "ALREADY shared" in str(exc.value)


def test_anon_never_shares():
    shared = solve_host_pool(_six_processes(anon_gib=1.0), shared=True)
    assert shared.anon_gib == pytest.approx(6.0)


def test_w114_refuses_an_empty_ledger():
    with pytest.raises(Weg2FlipHostPoolDoubled):
        solve_host_pool([], shared=True)


# --------------------------------------------------------------------------
# Seam (c) -- W115, GDN/draft state across the flip
# --------------------------------------------------------------------------
def _next_flash_states():
    return [
        # fnFA19 line 1029: TP0 conv 0.01 / ssm 0.11 GB at max_mamba_cache_size 1.
        CarriedState("gdn_conv_state", "pp-stage", "host", 10_000_000, "carried"),
        CarriedState("gdn_ssm_state", "pp-stage", "host", 110_000_000, "carried"),
        CarriedState("full_attention_kv", "pp-stage", "host", 0, "carried"),
        # The Form-A draft is SOLO: no P-side partner exists.
        CarriedState("mtp_draft_kv", "", "host", 0, "rebuilt", rebuild_tokens=4),
        CarriedState("spec_intermediate", "", "host", 0, "rebuilt", rebuild_tokens=1),
    ]


def test_next_flash_state_plan_prices_the_rebuild():
    plan = solve_state_carry(_next_flash_states(), prefill_tok_s=2729.0)
    assert plan.rebuild_tokens == 5
    # 5 tokens at the measured PP3 binding-stage rate is sub-millisecond --
    # far under the ~2 s physics floor, i.e. the draft rebuild is NOT what
    # makes the flip expensive.
    assert plan.rebuild_seconds < 0.01
    assert plan.carried_bytes == 120_000_000
    assert "STATE CARRY" in plan.report()


def test_w115_a_solo_draft_declared_carried_is_refused():
    states = _next_flash_states()
    states[3] = CarriedState("mtp_draft_kv", "", "host", 512, "carried")
    with pytest.raises(Weg2FlipDraftStateOrphaned) as exc:
        solve_state_carry(states, prefill_tok_s=2729.0)
    msg = str(exc.value)
    assert "W115 Weg2FlipDraftStateOrphaned" in msg
    assert "solo draft" in msg


def test_w115_a_free_rebuild_is_an_undeclared_cost():
    states = _next_flash_states()
    states[3] = CarriedState("mtp_draft_kv", "", "host", 0, "rebuilt", rebuild_tokens=0)
    with pytest.raises(Weg2FlipDraftStateOrphaned) as exc:
        solve_state_carry(states, prefill_tok_s=2729.0)
    assert "undeclared cost" in str(exc.value)


def test_w115_rejects_an_unknown_disposition():
    states = _next_flash_states()
    states[0] = CarriedState("gdn_conv_state", "pp-stage", "host", 1, "maybe")
    with pytest.raises(Weg2FlipDraftStateOrphaned):
        solve_state_carry(states, prefill_tok_s=2729.0)


def test_w115_absent_states_cost_nothing():
    plan = solve_state_carry(
        [CarriedState("vision_tower", "", "", 0, "absent")], prefill_tok_s=2729.0
    )
    assert plan.carried_bytes == 0
    assert plan.rebuild_tokens == 0
