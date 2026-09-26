"""H91 Teil B: the second D seat is PRICED before a boot -- the seat posts on
the Form-A attention host (GDN/Mamba slots, MTP verify state), the per-step
expert-pool rows on every rank, the plan width of the pool step -- and the
Next-Flash launcher's D seat default. H95 (Stufe 2): that default is the
UPPER BOUND 6 of a dynamic seat count per D phase; two seats are n = 2."""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import expert_residency as er  # noqa: E402

NF_MODEL = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
)


# ---- the budget: seat-proportional posts are re-booked, never carried -------

def test_the_built_in_nf_references_were_measured_with_one_seat():
    for ref in (er.D_RESIDENCY_REFERENCE_FNFL2, er.D_RESIDENCY_REFERENCE_FNFL2_H39):
        assert ref.max_running == 1


def test_two_seats_rebook_the_mamba_slots_and_the_spec_post():
    rb = er.seat_rebook(er.D_RESIDENCY_REFERENCE_FNFL2_H39, seats=2)
    assert (rb.slots_ref, rb.slots) == (7, 13)
    assert rb.mamba_mib == (730.2, 0.0, 0.0)  # 393.2 / 7 x 13
    assert rb.spec_mib == (448.6, 0.0, 0.0)  # 224.3 x 2
    assert rb.budget_delta_mib == (561.3, 0.0, 0.0)
    same = er.seat_rebook(er.D_RESIDENCY_REFERENCE_FNFL2_H39, seats=1)
    assert same.budget_delta_mib == (0.0, 0.0, 0.0)


def test_the_h64_rebook_derives_per_req_from_the_seat_rebooked_post():
    """The H64 rebook divides the measured post by the FORM's seats; fed the
    bs1 post with a bs2 form it halved the state per request (the bug the seat
    rebook closes by re-booking FIRST)."""
    tc = {
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_value_heads": 48, "linear_num_key_heads": 16,
        "linear_conv_kernel_dim": 4, "dtype": "bfloat16",
    }
    form = er.ReplaySSMSpecForm(ring_len=16, draft_tokens=4, max_running=2, ssm_dtype="bfloat16")
    rb2 = er.seat_rebook(er.D_RESIDENCY_REFERENCE_FNFL2_H39, seats=2)
    reb = er.replayssm_spec_rebook(spec_mib_ref=rb2.spec_mib, ref_ring_len=None, form=form, text_cfg=tc)
    assert reb.per_req_mib[0] == pytest.approx(56.075, abs=0.01)
    assert reb.spec_mib[0] == pytest.approx(44.6, abs=0.1)  # 2 x the bs1 ring post 22.3


def _fit(rank, E, R, S):
    return types.SimpleNamespace(rank=rank, local_experts=E, resident_rows=R, scratch_rows=S)


def test_the_pool_step_rows_refuse_a_worker_scratch_too_small_for_two_seats():
    # fnFL2x177's D: E 193/145/177, R 12/74/85, Scratch 118/48/48
    fits = [_fit(0, 193, 12, 118), _fit(1, 145, 74, 48), _fit(2, 177, 85, 48)]
    lines, refusal = er.pool_step_rows_check(
        fits, seats=1, verify_tokens=4, top_k=10, pool_mode=True, marker="M", label="D")
    assert refusal is None and "passt" in lines[0]  # bs1: 40 ids
    lines, refusal = er.pool_step_rows_check(
        fits, seats=2, verify_tokens=4, top_k=10, pool_mode=True, marker="M", label="D")
    assert refusal is not None and "[1, 2]" in refusal and "[80, 71, 80]" in refusal
    ok = [_fit(0, 193, 12, 118), _fit(1, 145, 42, 80), _fit(2, 177, 53, 80)]
    _, refusal = er.pool_step_rows_check(
        ok, seats=2, verify_tokens=4, top_k=10, pool_mode=True, marker="M", label="D")
    assert refusal is None
    # outside pool mode the bound is not this rule's business
    assert er.pool_step_rows_check(
        fits, seats=2, verify_tokens=4, top_k=10, pool_mode=False, marker="M", label="D") == ((), None)


@pytest.mark.skipif(not os.path.isdir(NF_MODEL), reason="the NF checkpoint headers are rig-local")
def test_dry_run_of_the_x177_form_at_two_seats():
    env = {
        "SGLANG_MOE_SCRATCH_SLOTS": "118,48,48", "SGLANG_UNEVEN_MOE_EXPERT_SHARD": "1",
        "SGLANG_MOE_OFFLOAD_GRAPH_MODE": "pool", "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL": "1",
        "SGLANG_WEG2_DRAFT_SHARE_EMBED": "1",
    }

    def plan(seats, fr, scratch):
        e = dict(env, SGLANG_MOE_SCRATCH_SLOTS=",".join(str(s) for s in scratch))
        return er.plan_d_residency(
            model_path=NF_MODEL, budgets_mib=[29624, 18664, 18672], ratios=[183, 137, 168],
            fractions=fr, scratch_rows=scratch, rank_tp_ratio="1,0,0", env_d=e,
            reference_logs="", kv_tokens=262144, label="D", marker="M",
            replayssm_spec=er.ReplaySSMSpecForm(
                ring_len=16, draft_tokens=4, max_running=seats, ssm_dtype="bfloat16"),
            seats=seats)

    one = plan(1, [0.06, 0.51, 0.48], [118, 48, 48])
    assert one.refusal is None
    two = plan(2, [0.06, 0.51, 0.48], [118, 48, 48])
    assert two.refusal is not None and "W-SITZE" in two.refusal
    assert two.fits[0].mamba_mib == pytest.approx(730.2)
    assert two.fits[0].spec_mib == pytest.approx(44.6, abs=0.1)
    # budget-neutral worker form: rows R+S unchanged, scratch 80
    ok = plan(2, [0.06, 0.289, 0.299], [118, 80, 80])
    assert ok.refusal is None
    assert [f.buffer_rows for f in ok.fits] == [f.buffer_rows for f in one.fits]
    # the card of the host loses exactly the extra allocation (mamba + ring rows)
    assert one.card_fits[0].headroom_mib - ok.card_fits[0].headroom_mib == pytest.approx(355.1, abs=0.2)
    assert any("NICHT GEBUCHT" in ln for ln in ok.lines)


# ---- the pool step: plan width and the tightened Task #40 bound ------------

def test_the_plan_width_follows_the_widest_captured_step():
    from sglang.srt.layers.moe import expert_pool_device as epd

    one = epd.pool_max_step_ids(graph_bs=[1], max_graph_bs=None, max_running=1, verify_tokens=4, top_k=10)
    two = epd.pool_max_step_ids(graph_bs=[1, 2], max_graph_bs=None, max_running=2, verify_tokens=4, top_k=10)
    assert (one, two) == (40, 80)
    assert epd.plan_width_for(one) == epd.PLAN_WIDTH == 64  # bs1 keeps its kernel
    assert epd.plan_width_for(two) == 128
    assert epd.plan_width_for(None) == 64
    assert epd.pool_max_step_ids(graph_bs=None, max_graph_bs=None, max_running=2,
                                 verify_tokens=None, top_k=None) is None


def test_the_step_bound_counts_only_what_can_miss():
    """80 ids on a rank whose every non-resident expert has a row (scratch
    clamped to E-R) cannot overflow; the ids-only bound refused it."""
    import torch

    from sglang.srt.layers.moe import expert_pool_device as epd

    E, R, S, staging = 40, 10, 30, 4  # E - R == S: every non-resident has a row
    hot = {e: e for e in range(R)}
    host_row = [-1] * R + list(range(E - R))
    tables = epd.allocate_pool_tables("cpu", E, R + S, R, staging, hot, host_row)
    buffers = epd.allocate_step_buffers("cpu", E, epd.plan_width_for(80))
    ids = torch.tensor([i % E for i in range(80)], dtype=torch.int32)
    epd.step_reference(tables, ids, buffers)  # must not raise
    assert int(tables.error[0]) == 0  # and it did not overflow either
    assert epd.step_row_demand(80, E, R) == 30
    assert epd.step_row_demand(40, 512, 12) == 40


# ---- the launcher: the Next-Flash seat default (H95: the bound 6) ----------

def test_nextflash_runs_d_with_up_to_six_seats_unless_told():
    from sglang.srt.weg2 import DEFAULT_D_BS, DEFAULT_D_BS_NEXTFLASH
    from sglang.srt.weg2 import launcher as L

    assert DEFAULT_D_BS_NEXTFLASH == 6  # H95: dynamic 1..6, was H91b's fixed 2
    ns = types.SimpleNamespace(profile=L.PROFILE_NEXTFLASH, d_bs=DEFAULT_D_BS)
    assert L.apply_profile_d_bs_default(ns, ["--profile", "nextflash"]) == 6
    ns = types.SimpleNamespace(profile=L.PROFILE_NEXTFLASH, d_bs=1)
    assert L.apply_profile_d_bs_default(ns, ["--profile", "nextflash", "--d-bs", "1"]) == 1
    ns = types.SimpleNamespace(profile=L.PROFILE_QWEN27B, d_bs=DEFAULT_D_BS)
    assert L.apply_profile_d_bs_default(ns, []) == DEFAULT_D_BS


def test_the_effective_seats_and_the_graph_list_are_read_off_extra_d():
    from sglang.srt.weg2 import launcher as L

    ns = types.SimpleNamespace(
        d_bs=2, extra_d="--max-running-requests 1 --cuda-graph-bs-decode 1 --page-size 64")
    assert L.d_effective_seats(ns) == 1
    lines = L.d_seat_lines(ns)
    assert any("--d-bs 2 != D --max-running-requests 1" in ln for ln in lines)
    ns = types.SimpleNamespace(
        d_bs=2, extra_d="--max-running-requests 2 --cuda-graph-bs-decode 1 --page-size 64")
    lines = L.d_seat_lines(ns)
    assert any("'--cuda-graph-bs-decode 1 2'" in ln for ln in lines)
    ns = types.SimpleNamespace(
        d_bs=2, extra_d="--max-running-requests 2 --cuda-graph-bs-decode 1 2 --page-size 64")
    assert len(L.d_seat_lines(ns)) == 1  # nothing to name
    assert L._argv_int_list("--cuda-graph-bs-decode 1 2 --x 3", "--cuda-graph-bs-decode") == [1, 2]
    # a hand-built namespace without any seat statement is priced as before
    assert L.d_stated_seats(types.SimpleNamespace(extra_d="--rank-tp-ratio 1,0,0")) is None
    assert L.d_stated_seats(types.SimpleNamespace(extra_d="--max-running-requests 2")) == 2


def test_the_d_solve_is_told_the_seats_and_the_main_resolves_the_nf_default():
    from sglang.srt.weg2 import launcher as L

    src = open(L.__file__).read()
    i = src.index("replayssm_spec=d_replayssm_spec_plan_form(ns),")
    assert "seats=d_stated_seats(ns)," in src[i:i + 600]
    j = src.index("    ns = build_parser().parse_args(")  # FL6: the argv goes through _canonical_flags
    assert "apply_profile_d_bs_default(ns," in src[j:j + 600]  # FL6 state-dir link sits before it
