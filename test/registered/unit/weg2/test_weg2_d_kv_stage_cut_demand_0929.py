"""EXPERTEN-KV-DYNAMISCH-D-0929 §6, points 1, 2, 4, 5 (Go 29.09.): the KV stage
form under the production cut, proven on the numbers of z30r3.

z30r3 (ad95392095, 29.09. 05:48-06:22Z, profile -2b-swr-e2cut): cut [0, 1, 1]
(S = 2, host share 0), EFFECTIVE max_total_num_tokens 262144; the profile pins
SGLANG_WEG2_D_KV_STAGE_TOKENS=262144 itself -> 'keine Stufenform vom Launcher',
0x '#251 WAKE-RESHARD'. 20 distinct 'H105 RU FORM-A ADMISSION WAIT
host=NO_TOKEN' (TP1), deficit host_price - host_budget up to 42277 global
tokens. The front's wake demand (HANDOFF-N phase_kv_tokens, global tokens) of
the 68 D wakes is the list below.

WHAT MUST HOLD.
(1) The pin is the switch: one stage token = no form; without it and with
    SGLANG_WEG2_D_KV_STAGE_BY_DEMAND=1 the launcher writes the form for every
    KV rank of the cut (host trim cell 1855 B/token, workers 12288 / 2), every
    stage at every seat count. Profile -e2cut-kvstage = exactly that.
(2) Demand proof: by demand, the z30r3 wakes take S0 36x / S1 30x / S2 2x,
    never over the top; every WAIT epoch but one takes S1 -- epoch 77 (demand
    229392 < S0) stays S0: that WAIT is occupancy, not stage (named, not
    hidden). S1 adds 131072 global tokens against the largest deficit 42277.
    The stage is replicated: every rank of the cut takes the same one.
(4) WEG2-WAKE-KV-TIME: one line per kv_cache resume (ms, need, mapped, stage,
    epoch), its own marker (WEG2-WAKE-TAG-TIME stays the weights legs'),
    wired right after the resume and before the group verdict, and never able
    to stop a wake.
(5) TP0's host cell (QSA keys + draft, 1855 B/token) trims to S0 at birth and
    grows with the stage: +131072 x 1855 B = 231.9 MiB at S1, paid by 3 of its
    6 stage rows (112.5 MiB each), 5 at S2; the top stage's pool (524288 x
    1855 B = 927.5 MiB) lies inside the 1.283 GiB TP0 sized in z30r3.
"""
from __future__ import annotations

import os
import sys
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weg2_d_kv_stage_launcher_251c as T251  # noqa: E402
import test_weg2_d_kv_stage_worker_251c as W251  # noqa: E402

from sglang.srt.planner import expert_residency as er  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.srt.weg2 import wake_kv  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

NF_STAGES = "262144,393216,524288"
S0, S1, S2 = 262144, 393216, 524288
PAGE = 64
HOST_CELL = 1855  # z30r3 TP0 'KV pool sizing ... cell_size=1855'
FA_CELL = 12288  # z30r3 TP1/TP2 'cell_size=12288'
ROW_BYTES = (2534448 - 76848) * 48  # 112.5 MiB, one stage row
MIB = 1 << 20

#: z30r3 front 'HANDOFF-N epoch=E wake=D ... phase_kv_tokens=T', all 68 D wakes
Z30R3_WAKES = (
    (1, 34750), (3, 86662), (5, 108764), (7, 59952), (9, 121996), (11, 64538),
    (13, 129386), (15, 65329), (17, 151972), (19, 104221), (21, 129844), (23, 164339),
    (25, 128022), (27, 155575), (29, 124028), (31, 195902), (33, 142366), (35, 219957),
    (37, 234335), (39, 223016), (41, 200081), (43, 244098), (45, 216935), (47, 235435),
    (49, 268097), (51, 216967), (53, 310375), (55, 315847), (57, 331501), (59, 293466),
    (61, 274774), (63, 255824), (65, 284715), (67, 276863), (69, 209529), (71, 330094),
    (73, 229392), (75, 280731), (77, 229392), (79, 191854), (81, 217937), (83, 275233),
    (85, 240719), (87, 272637), (89, 181937), (91, 277132), (93, 346702), (95, 328984),
    (97, 308282), (99, 253982), (101, 266018), (103, 312363), (105, 322989), (107, 237085),
    (109, 237085), (111, 304406), (113, 302553), (115, 312221), (117, 312379), (119, 312444),
    (121, 329797), (123, 329797), (125, 295635), (127, 295635), (129, 334267), (131, 378623),
    (133, 423295), (135, 413973),
)

#: the wake before each distinct FORM-A ADMISSION WAIT (TP1): epoch -> (demand,
#: largest deficit host_price - host_budget of the WAITs in that phase)
Z30R3_WAIT_EPOCHS = {
    55: (315847, 15059), 75: (280731, 40392), 77: (229392, 40776), 87: (272637, 42277),
    105: (322989, 6487), 111: (304406, 29566), 115: (312221, 25550), 117: (312379, 10190),
    119: (312444, 31343),
}


def _ctx(rank, bounds, **extra):
    """A D rank of the cut with the -e2cut-kvstage form (by demand)."""
    from sglang.srt.environ import envs

    ctx = W251._armed(W251.FakeTms(), rank)
    ctx.stack += [
        envs.SGLANG_WEG2_D_KV_STAGE_TOKENS.override(extra.get("tokens", NF_STAGES)),
        envs.SGLANG_WEG2_D_KV_STAGE_ROWS.override(6),
        envs.SGLANG_WEG2_D_KV_STAGE_ROWS_BY_RANK.override("6,15,15"),
        envs.SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS.override("2,2,2,2,2,2"),
        envs.SGLANG_WEG2_D_KV_STAGE_BY_DEMAND.override(extra.get("by_demand", True)),
        mock.patch("sglang.srt.distributed.utils.uneven_dcp_owner_bounds", lambda: bounds),
    ]
    return ctx


@pytest.fixture(autouse=True)
def _restore_role_plan():
    """_armed installs a Form A role plan; leave it as found (order-free suites)."""
    from sglang.srt import rank_role

    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


#: the cut [0, 1, 1]: (S, lo, hi) per rank
CUT = {0: (2, 0, 0), 1: (2, 0, 1), 2: (2, 1, 2)}


def _fit(rank, cell, stage=0, experts=193, staging=12):
    return types.SimpleNamespace(rank=rank, kv_cell_bytes=cell, kv_tokens=S0,
                                 local_experts=experts, staging_rows=staging,
                                 kv_stage_cell_bytes=stage)


def _cut_plan():
    half = FA_CELL // 2
    return types.SimpleNamespace(fits=[_fit(0, HOST_CELL), _fit(1, half, half, 120, 8),
                                       _fit(2, half, half, 120, 8)])


# ---- (1) the pin is the switch; the launcher writes the form under the cut -----------

def test_the_single_stage_pin_is_no_form_z30r3():
    with _ctx(0, CUT[0], tokens="262144"):
        assert dsv.stage_form() is None
    with _ctx(0, CUT[0]):
        form = dsv.stage_form()
        assert form is not None and form.tokens == (S0, S1, S2) and form.by_demand


def _apply(env_d):
    ns = types.SimpleNamespace(env_d=env_d, d_pool_waves_derived=True)
    lines = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, _cut_plan(), "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=(0, 1, 1))
    return ns, lines


def test_the_launcher_writes_the_form_for_every_kv_rank_of_the_cut():
    base = "SGLANG_MOE_SCRATCH_SLOTS=100,48,48;SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES=2"
    pinned, lines = _apply(base + ";SGLANG_WEG2_D_KV_STAGE_TOKENS=262144")
    assert pinned.env_d.endswith("SGLANG_WEG2_D_KV_STAGE_TOKENS=262144")
    assert any("nennt SGLANG_WEG2_D_KV_STAGE_TOKENS=262144 selbst" in ln for ln in lines)
    ns, lines = _apply(base + ";SGLANG_WEG2_D_KV_STAGE_BY_DEMAND=1")
    env = L.parse_group_env(ns.env_d)
    assert env["SGLANG_WEG2_D_KV_STAGE_TOKENS"] == NF_STAGES
    # host: ceil(262144 x 1855 / 112.5 MiB) + 1; workers: ceil(262144 x 6144 / 112.5 MiB) + 1
    assert env[L.D_KV_STAGE_ROWS_BY_RANK_KEY] == "6,15,15"
    assert env["SGLANG_MOE_SCRATCH_SLOTS"] == "94,33,33"
    assert env["SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS"] == "2,2,2,2,2,2"
    res = [ln for ln in lines if "RESIDENZ (#239 S3g)" in ln]
    assert "S1..S2 schneiden [3, 5]" in res[0] and "Trim-Zelle 1855 B/Tok" in res[0]
    assert all("S1..S2 schneiden [7, 14]" in r and "Trim-Zelle 6144 B/Tok" in r
               for r in res[1:])


# ---- (2) the demand proof on the z30r3 wakes ------------------------------------------

def _choices(rank=0):
    with _ctx(rank, CUT[rank]):
        form = dsv.stage_form()
        return {e: dsv.choose_form_stage(form, 6, d) for e, d in Z30R3_WAKES}


def test_by_demand_the_z30r3_wakes_take_s0_s1_s2_never_over():
    ch = _choices()
    assert len(ch) == 68
    stages = [c.stage for c in ch.values()]
    assert (stages.count(0), stages.count(1), stages.count(2)) == (36, 30, 2)
    assert not any(c.over for c in ch.values())
    assert {e for e, c in ch.items() if c.stage == 2} == {133, 135}


def test_every_wait_epoch_but_the_occupancy_one_takes_s1():
    ch = _choices()
    for e, (demand, _deficit) in Z30R3_WAIT_EPOCHS.items():
        assert dict(Z30R3_WAKES)[e] == demand
        assert ch[e].stage == (0 if e == 77 else 1), e
    # epoch 77: the same rid (weg2-74-110) still waited, demand below S0 -- the
    # tree held the pages, not the stage; by demand keeps S0 there
    assert ch[77].demand == 229392 < S0
    # S1's 131072 extra global tokens cover the largest deficit of every WAIT
    assert max(d for _, d in Z30R3_WAIT_EPOCHS.values()) == 42277 < S1 - S0


def test_every_rank_of_the_cut_takes_the_same_stage():
    per_rank = [_choices(r) for r in (0, 1, 2)]
    for e, _ in Z30R3_WAKES:
        assert len({(c[e].stage, c[e].tokens) for c in per_rank}) == 1, e


def test_without_the_switch_the_table_would_keep_s0_at_five_and_six_seats():
    from sglang.srt.environ import envs

    ctx = _ctx(0, CUT[0], by_demand=False)
    ctx.stack.append(envs.SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS.override("2,1,1,1,0,0"))
    with ctx:
        form = dsv.stage_form()
        c = dsv.choose_form_stage(form, 6, dict(Z30R3_WAKES)[87])
    assert (c.stage, c.over) == (0, True)


# ---- (4) the kv_cache resume, timed ---------------------------------------------------

def test_the_kv_time_line_names_ms_bytes_and_stage():
    st = dsv.PhaseState(epoch="87", stage=1, stage_tokens=S1)
    ln = wake_kv.kv_resume_time_line(41.26, 5438 * MIB, 9000 * MIB, 3562 * MIB, st, "87")
    assert ln == ("WEG2-WAKE-KV-TIME tag=kv_cache resume_ms=41.3 need_mib=5438 "
                  "mapped_mib=5438 stage=S1 tokens=393216 epoch=87")
    # no stage form, no probe: '-' -- an absence is printed as one
    ln = wake_kv.kv_resume_time_line(3.0, 0, None, None, None, None)
    assert ln.endswith("need_mib=0 mapped_mib=- stage=- tokens=- epoch=None")
    assert not ln.startswith("WEG2-WAKE-TAG-TIME")


def _wu_src():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                     "python", "sglang", "srt", "managers", "scheduler_components",
                     "weight_updater.py")
    with open(os.path.normpath(p)) as fh:
        return fh.read()


def test_the_resume_is_timed_between_the_resume_and_the_group_verdict():
    s = _wu_src()
    body = s[s.index("def _weg2_kv_resume_part():"):s.index("def _weg2_kv_clear_part():")]
    t0 = body.index("_t_kv = time.perf_counter()")
    resume = body.index("self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)")
    line = body.index("kv_resume_time_line(")
    verdict = body.index("if not self._weg2_kv_group_verdict(True, _kv_epoch):")
    assert t0 < resume < line < verdict
    # never able to stop the wake: the call sits in its own try, the except logs
    blk = body[body.rindex("try:", 0, line):verdict]
    assert "except Exception" in blk and "WEG2-WAKE-KV-TIME skipped" in blk


def test_the_weights_parser_does_not_read_the_kv_line():
    from sglang.srt.weg2 import wake_credit_pd as W

    src = open(W.__file__).read()
    assert "WEG2-WAKE-TAG-TIME" in src and "WEG2-WAKE-KV-TIME" not in src


# ---- (5) TP0's host cell trims and grows with the stage -------------------------------

def test_the_host_cell_trims_to_s0_and_grows_with_the_stage():
    with _ctx(0, CUT[0]):
        assert dsv.owner_block_for(S2) == (0, 0)  # the whole context on the host
        assert dsv.kv_stage_trims_here(S2) is True
        assert dsv.kv_stage_boot_rows(S2, PAGE) == S0 + PAGE
        assert dsv.stage_mapped_rows(S2, S1, PAGE) == S1 + PAGE
        assert dsv.stage_mapped_rows(S2, S2, PAGE) == S2 + PAGE
    # a worker of the cut trims its compacted FA share (z30r3: '#tokens: 131073')
    with _ctx(1, CUT[1]):
        top = (S2 // 2 + 1) * 1
        assert dsv.owner_block_for(top) == (2, 1)
        assert dsv.kv_stage_trims_here(top) is True
        assert dsv.kv_stage_boot_rows(top, PAGE) == S0 // 2 + 1 + PAGE


def test_the_host_cells_pay_for_the_host_kv_of_every_stage():
    G = dsv.GRANULE_DEFAULT
    kv = dsv.KvTensorGeom(
        dsv.SlotTensorGeom("host_cell", 1, S2 + PAGE, HOST_CELL,
                           dsv.align_up((S2 + PAGE) * HOST_CELL, G)),
        token_pad=PAGE)
    grow = [dsv.kv_mapped_bytes([kv], t, G) for t in (S0, S1, S2)]
    assert abs((grow[1] - grow[0]) - (S1 - S0) * HOST_CELL) <= G  # 231.9 MiB
    assert round((S1 - S0) * HOST_CELL / MIB, 1) == 231.9
    row = dsv.RowTensorGeom("bank", 100, 106, ROW_BYTES, 106 * ROW_BYTES)
    cells = dsv.stage_vram_cells([], [row], [kv], cap=6, pool_size=0, extra_max=6,
                                 stage_tokens=(S0, S1, S2), boot_rows_on=6, granule=G)
    ks = [cells[(6, j)].extra_rows for j in range(3)]
    assert ks == [6, 3, 1]  # the planner's [3, 5] stage rows of 6
    for j in range(3):
        c = cells[(6, j)]
        assert c.feasible and c.mapped <= c.cap_mapped
    # z30r3 TP0 sized 1.283 GiB for this cell; the top stage's pool fits in it
    assert S2 * HOST_CELL <= 1377722368
