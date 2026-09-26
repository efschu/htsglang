"""H95 B (H91 Teil B, Stufe 2): D's seats are DYNAMIC, n = 1..--d-bs per D phase.

User design 25.09.: D decodes every request the P phase handed over (up to 6)
to the end; the batch size follows the count known at the P->D flip; the posts
of unoccupied seats go to the experts; the context is one unified 262k pool
(the youngest parks, H91b).

WHAT MUST HOLD.
(1) The count rides the wake: ``handoff_n``/``parked_n`` are DECLARED fields of
    the kv_cache resume (H91c wrote them, D ignored them), and D's phase seat
    count is a pure function of those two integers and of the boot's cap --
    every rank receives the same request, so no rank can disagree and no
    collective is needed.
(2) The scheduler half: group D keeps the phase's seats and names them once
    per wake; off D and on a wake without the count nothing happens; the
    resume path calls it.
(3) One pricing for every n: the seat table is the D-FRACTION-SOLVE per seat
    count (seat_rebook re-books the posts) plus the H95 pool bound -- bs2 is
    n = 2 of it, reproducing H91b's scratch 80 / FR 0.29 without waves.
(4) The Next-Flash launcher runs D with up to 6 seats (--d-bs the hard bound)
    and two pool waves unless told otherwise.
"""
from __future__ import annotations

import os
import pickle
import types
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import expert_residency as er  # noqa: E402
from sglang.srt.weg2 import d_seats  # noqa: E402

NF_MODEL = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
)


# ---- (1) the count on the wake, one replicated number ----------------------

def test_phase_seats_is_the_handed_count_clamped_to_the_cap():
    s = d_seats.phase_seats(4, 1, cap=6, epoch="e7")
    assert (s.n, s.handoff_n, s.parked_n, s.cap, s.clamped) == (5, 4, 1, 6, False)
    assert d_seats.phase_seats(7, 2, cap=6).n == 6
    assert d_seats.phase_seats(7, 2, cap=6).clamped
    assert d_seats.phase_seats(0, 0, cap=6).n == 1  # a wake with nothing: bs1
    assert d_seats.phase_seats(None, None, cap=6) is None  # no count: no decision
    assert d_seats.phase_seats(2, None, cap=6).n == 2
    line = d_seats.phase_seats(3, 0, cap=6, epoch="e1").line()
    assert "WEG2 D-PHASE-SEATS (H95)" in line and "n=3 of cap 6" in line
    assert "GDN slots in use <= 19 of 38" in line


def test_the_wake_request_declares_the_count_the_front_writes():
    from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput
    from sglang.srt.weg2.front import Front

    front = types.SimpleNamespace(
        _d_parked={"p1": 0.0}, groups={"D": types.SimpleNamespace(outstanding={"a": 1, "b": 1})},
        _ready_for_d=[types.SimpleNamespace(fut=None)], _handoff_in_flight=lambda: 1)
    extra = Front._wake_handoff_fields(front, "D")
    assert extra == {"handoff_n": 4, "parked_n": 1}
    req = ResumeMemoryOccupationReqInput(tags=["kv_cache"], epoch="e3", **extra)
    assert (req.handoff_n, req.parked_n) == (4, 1)
    # every rank unpickles the SAME object -> the same n, no collective
    ranks = [pickle.loads(pickle.dumps(req)) for _ in range(3)]
    ns = {d_seats.phase_seats(r.handoff_n, r.parked_n, cap=6, epoch=r.epoch).n for r in ranks}
    assert ns == {5}
    assert ResumeMemoryOccupationReqInput(tags=["weights"]).handoff_n is None


# ---- (2) the scheduler half ---------------------------------------------------

def _sched(cap=6):
    return types.SimpleNamespace(server_args=types.SimpleNamespace(max_running_requests=cap))


def test_group_d_keeps_the_phase_seats():
    from sglang.srt.weg2 import d_park_runtime

    req = types.SimpleNamespace(handoff_n=3, parked_n=1, epoch="e9")
    with mock.patch.dict(os.environ, {d_seats.GROUP_ENV: "D"}):
        sched = _sched()
        seats = d_park_runtime.note_wake_seats(sched, req)
        assert seats.n == 4 and sched.weg2_d_phase_seats is seats
        sched2 = _sched(cap=2)
        assert d_park_runtime.note_wake_seats(sched2, req).n == 2  # --d-bs is the bound
    with mock.patch.dict(os.environ, {d_seats.GROUP_ENV: "P"}):
        sched = _sched()
        assert d_park_runtime.note_wake_seats(sched, req) is None
        assert not hasattr(sched, "weg2_d_phase_seats")
    with mock.patch.dict(os.environ, {d_seats.GROUP_ENV: "D"}):
        sched = _sched()
        assert d_park_runtime.note_wake_seats(
            sched, types.SimpleNamespace(handoff_n=None, parked_n=None)) is None


def test_the_resume_path_calls_the_seat_note_before_the_tags():
    from sglang.srt.managers import scheduler as sch
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    assert callable(getattr(sch.Scheduler, "weg2_d_note_wake_seats", None))
    src = open(wu.__file__).read()
    i = src.index("    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):")
    body = src[i:i + 4000]
    j = body.index('"weg2_d_note_wake_seats", None)')
    assert j < body.index("tags = recv_req.tags")
    assert j > body.index("return replay")  # a replayed leg decides nothing twice


# ---- (3) one pricing for every n ------------------------------------------

def test_the_scratch_a_seat_count_needs_on_a_worker():
    # x177 TP1 (E 145) / TP2 (E 177) at the rows the arm runs (0.51/0.48 + 48)
    # H91b, no waves: bs2 = scratch 80 -> R 42/53 = FR 0.29/0.30
    assert er.seat_scratch_for(seats=2, verify_tokens=4, top_k=10, local_experts=145,
                               max_rows=122, waves=1) == (80, 42)
    assert er.seat_scratch_for(seats=2, verify_tokens=4, top_k=10, local_experts=177,
                               max_rows=133, waves=1) == (80, 53)
    # without waves bs4 has no split at all: C >= 160 > the rows
    assert er.seat_scratch_for(seats=4, verify_tokens=4, top_k=10, local_experts=145,
                               max_rows=122, waves=1) is None
    # two waves: bs6 with a scratch far below bs1's 48
    s, R = er.seat_scratch_for(seats=6, verify_tokens=4, top_k=10, local_experts=177,
                               max_rows=133, waves=2)
    assert min(240, 177 - R) <= 2 * s and R == 133 - s and s < 48


def _fit(rank, E, R, S, rows_ceiling, mamba=0.0, spec=0.0):
    return types.SimpleNamespace(rank=rank, local_experts=E, resident_rows=R, scratch_rows=S,
                                 staging_rows=12, ceiling_max_rows=rows_ceiling,
                                 mamba_mib=mamba, spec_mib=spec)


def test_the_seat_table_prices_every_n_from_the_same_solve():
    calls = []

    def plan_for(n):
        calls.append(n)
        fits = [_fit(0, 193, 12, 118, 136 - 3 * (n - 1), mamba=393.2 / 7 * d_seats.mamba_slots_for_seats(n),
                     spec=22.3 * n),
                _fit(1, 145, 74, 48, 140), _fit(2, 177, 85, 48, 141)]
        return types.SimpleNamespace(fits=fits, card_fits=(), refusal=None)

    rows = er.seat_table(plan_for, seats_max=6, verify_tokens=4, top_k=10, waves=2)
    assert calls == [1, 2, 3, 4, 5, 6]
    assert [r.mamba_slots for r in rows] == [7, 13, 19, 25, 32, 38]
    assert [r.ids_per_step for r in rows] == [40, 80, 120, 160, 200, 240]
    assert rows[5].host_spec_mib == pytest.approx(133.8)
    # the workers keep their fraction for every n under two waves
    assert {r.fraction_given[1] for r in rows} == {rows[0].fraction_given[1]}
    assert all(r.waves_given[1] <= 2 and r.waves_given[2] <= 2 for r in rows)
    lines = er.describe_seat_table(rows, marker="M", label="D")
    assert len(lines) == 6 and "D-SITZE (H95) n=6" in lines[5]
    # without waves the same scratch cannot carry n >= 2 on the workers
    rows1 = er.seat_table(plan_for, seats_max=3, verify_tokens=4, top_k=10, waves=1)
    assert rows1[1].fraction_given[1] is None and rows1[1].scratch_min[1] == 80


@pytest.mark.skipif(not os.path.isdir(NF_MODEL), reason="the NF checkpoint headers are rig-local")
def test_dry_run_of_the_x177_form_for_one_to_six_seats():
    env = {
        "SGLANG_UNEVEN_MOE_EXPERT_SHARD": "1", "SGLANG_MOE_OFFLOAD_GRAPH_MODE": "pool",
        "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL": "1", "SGLANG_WEG2_DRAFT_SHARE_EMBED": "1",
        "SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES": "2",
    }

    def table(scratch):
        e = dict(env, SGLANG_MOE_SCRATCH_SLOTS=",".join(str(s) for s in scratch))

        def plan_for(n):
            return er.plan_d_residency(
                model_path=NF_MODEL, budgets_mib=[29624, 18664, 18672], ratios=[183, 137, 168],
                fractions=[0.06, 0.51, 0.48], scratch_rows=scratch, rank_tp_ratio="1,0,0",
                env_d=e, reference_logs="", kv_tokens=262144, label="D", marker="M",
                replayssm_spec=er.ReplaySSMSpecForm(
                    ring_len=16, draft_tokens=4, max_running=n, ssm_dtype="bfloat16"),
                seats=n)

        return er.seat_table(plan_for, seats_max=6, verify_tokens=4, top_k=10, waves=2)

    x177 = table([118, 48, 48])
    assert [r.max_rows for r in x177] == [
        (136, 140, 141), (133, 140, 141), (130, 140, 141),
        (127, 140, 141), (123, 140, 141), (120, 140, 141)]
    assert [r.waves_given for r in x177][1:] == [(1, 2, 2)] + [(2, 2, 2)] * 4
    assert x177[5].host_mamba_mib == pytest.approx(2134.5)
    assert x177[5].host_spec_mib == pytest.approx(133.9)
    # the x177 scratch dies on the 5090 card from four seats on (W130) ...
    assert [r.refusal is None for r in x177] == [True, True, True, False, False, False]
    assert "W130" in x177[5].refusal
    # ... TP0 scratch 100 (rows 112 <= 120) carries n = 1..6 with two waves
    assert all(r.refusal is None for r in table([100, 48, 48]))


# ---- (4) the launcher -------------------------------------------------------

def test_nextflash_d_runs_up_to_six_seats_with_two_pool_waves():
    from sglang.srt.weg2 import DEFAULT_D_BS_NEXTFLASH, DEFAULT_D_POOL_WAVES_NEXTFLASH
    from sglang.srt.weg2 import launcher as L

    assert (DEFAULT_D_BS_NEXTFLASH, DEFAULT_D_POOL_WAVES_NEXTFLASH) == (6, 2)
    ns = types.SimpleNamespace(profile=L.PROFILE_NEXTFLASH, env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48")
    line = L.apply_profile_d_pool_waves_default(ns)
    assert "D-POOL-WELLEN (H95)" in line
    assert L.parse_group_env(ns.env_d) == {
        "SGLANG_MOE_SCRATCH_SLOTS": "100,48,48", "SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES": "2"}
    told = types.SimpleNamespace(profile=L.PROFILE_NEXTFLASH,
                                 env_d="SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES=0")
    assert L.apply_profile_d_pool_waves_default(told) is None
    assert told.env_d == "SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES=0"
    other = types.SimpleNamespace(profile=L.PROFILE_QWEN27B, env_d="")
    assert L.apply_profile_d_pool_waves_default(other) is None and other.env_d == ""
    ns = types.SimpleNamespace(
        d_bs=6, extra_d="--max-running-requests 6 --cuda-graph-bs-decode 1 2 3 4 5 6")
    lines = L.d_seat_lines(ns)
    assert len(lines) == 1 and "H95: 6 ist die OBERGRENZE" in lines[0]


def test_the_launcher_main_applies_both_nextflash_defaults_first():
    from sglang.srt.weg2 import launcher as L

    src = open(L.__file__).read()
    j = src.index("    ns = build_parser().parse_args(")  # FL6: the argv goes through _canonical_flags
    head = src[j:j + 600]
    assert "apply_profile_d_bs_default(ns," in head
    assert "apply_profile_d_pool_waves_default(ns)" in head
    i = src.index("    for _ln in d_seat_table_lines(ns, _er, dict(")
    assert i < src.index("        raise Weg2LaunchRefused(plan.refusal)", i)
