"""#251d: D's KV stage follows the next wake's demand alone (switch, default off).

User order (27./28.09., "dynamisch leistung gegen mehr verfuegbaren unified kv
cache falls noetig", uneven-DCP KV before YaRN): the stage choice comes from the
measured demand of the next wake -- the KV tokens of the phase's seats against
the pool -- not from a fixed table per seat count, so that S1/S2 become
reachable at n = 5/6 when the demand needs them. D only, switched only at the
wake (a phase boundary), performance first (the KV grows only when the demand
exceeds it), no extra host RAM (the stage rows are expert rows that already
live in the store).

Metal before the switch (KV-STUFEN-PREIS-NF-0928.md): 15 of 19 over-demand
phases were at n = 5/6, where the launcher's table 2,1,1,1,0,0 keeps S0 --
'#251 WAKE-RESHARD n=5 stage=S0 ... over=yes', the youngest parked.

WHAT MUST HOLD.
(1) SGLANG_WEG2_D_KV_STAGE_BY_DEMAND=1: every stage is open to every seat
    count -- n = 5/6 with 300k demand take S1, with 450k S2 (the table would
    keep S0 and park).
(2) Performance first: a demand the smaller stage holds keeps it, at every n.
(3) Off (the default): the table, byte for byte.
(4) The capture floor covers the lowest stage at the cap; TP0's cell check
    covers every (n, j).
(5) The rank: the wake takes the demand's stage above the table and names
    'form=demand' on its stage line (only then).
(6) The launcher: the switch writes the top stage for every n and raises the
    derived overflow-wave cap to what the lowered capture floor needs; a cap
    told in --env-d is refused by name (W169, through d_seat_table_lines too);
    the undo takes it all back; without the switch no wave key moves.
(7) The W168 operator arm 2,1,1,1,1,1 (profile -kv211111): rc12z22 derived
    SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES=2 (D log 14:51:34, bs6 D=181 over
    C=91), S1 at n=5/6 lowers C to 75 -> 3 waves -> the capture's own bound
    ('Step ids exceed the LRU rows plus the staging rows') would stop TP0.
    The launcher raises the cap to 3; the s2probe arm (2 waves) keeps 2.
"""
from __future__ import annotations

import logging
import os
import sys
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weg2_d_kv_stage_launcher_251c as lt  # noqa: E402
import test_weg2_d_kv_stage_runtime_251c as rt  # noqa: E402

from sglang.srt.planner import expert_residency as er  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402


@pytest.fixture(autouse=True)
def _kv_stage_table_path():
    """29.09.: #251d is the default now; this file covers the table path, so
    the switch is pinned off here (a test that wants demand overrides it)."""
    from sglang.srt.environ import envs

    with envs.SGLANG_WEG2_D_KV_STAGE_BY_DEMAND.override(False):
        yield


NF_STAGES = "262144,393216,524288"
TABLE = "2,1,1,1,0,0"
WAVES = "SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES"
DEMAND = "SGLANG_WEG2_D_KV_STAGE_BY_DEMAND"
MAX_BY = "SGLANG_WEG2_D_KV_STAGE_MAX_BY_SEATS"


def _env(by_demand, **extra):
    from sglang.srt.environ import envs

    return rt._Ctx(rt._env(**extra) + [envs.SGLANG_WEG2_D_KV_STAGE_BY_DEMAND.override(by_demand)])


# ---- (1)-(3) the choice -------------------------------------------------------------

def test_by_demand_every_stage_is_open_to_every_seat_count():
    with _env(True, tokens=NF_STAGES, rows=33, max_by=TABLE):
        form = dsv.stage_form()
        assert form.by_demand and [form.max_stage(n) for n in range(1, 7)] == [2] * 6
        c5 = dsv.choose_form_stage(form, 5, 300000)
        c6 = dsv.choose_form_stage(form, 6, 450000)
    assert (c5.stage, c5.tokens, c5.over) == (1, 393216, False)
    assert (c6.stage, c6.tokens, c6.over) == (2, 524288, False)


def test_performance_first_the_kv_grows_only_when_the_demand_exceeds_it():
    with _env(True, tokens=NF_STAGES, rows=33, max_by=TABLE):
        form = dsv.stage_form()
        for n in range(1, 7):
            for d in (None, 1, 200000, 262144):
                assert dsv.choose_form_stage(form, n, d).stage == 0, (n, d)
            assert dsv.choose_form_stage(form, n, 262145).stage == 1
            assert dsv.choose_form_stage(form, n, 393217).stage == 2
        # above the top stage: the top one, the youngest parks (as before)
        top = dsv.choose_form_stage(form, 6, 600000)
        assert (top.stage, top.over) == (2, True)


def test_off_the_table_holds_byte_for_byte():
    with _env(False, tokens=NF_STAGES, rows=33, max_by=TABLE):
        form = dsv.stage_form()
        assert not form.by_demand
        assert [form.max_stage(n) for n in range(1, 7)] == [2, 1, 1, 1, 0, 0]
        c5 = dsv.choose_form_stage(form, 5, 300000)
    assert (c5.stage, c5.tokens, c5.over) == (0, 262144, True)
    assert dsv.form_seats_text(form) == [2, 1, 1, 1, 0, 0]
    assert dsv.form_seats_text(dsv.StageForm(tokens=(1, 2))) == "all"


# ---- (4) the capture floor and TP0's cell check -------------------------------------

def test_the_capture_floor_covers_the_lowest_stage_at_the_cap():
    table = dsv.StageForm(tokens=(262144, 393216, 524288), rows_on=33,
                          max_by_seats=(2, 1, 1, 1, 0, 0))
    demand = dsv.StageForm(tokens=table.tokens, rows_on=33, max_by_seats=table.max_by_seats,
                           by_demand=True)
    assert dsv.capture_floors(table, lt._cells(), 6) == (14, 22, 22, 22, 33, 33)
    # every stage at every n: the 6-seat phase at S2 (1 row ON) floors every batch
    assert dsv.capture_floors(demand, lt._cells(), 6) == (1,) * 6
    cells = {k: types.SimpleNamespace(feasible=True) for k in lt._cells()}
    dsv.check_form_against_cells(demand, cells, 6)
    cells[(6, 2)] = types.SimpleNamespace(feasible=False)
    dsv.check_form_against_cells(table, cells, 6)  # the table never lets n=6 take S2
    with pytest.raises(dsv.Weg2DSeatVramRefused, match=r"\(6, 2\)"):
        dsv.check_form_against_cells(demand, cells, 6)


# ---- (5) the rank -------------------------------------------------------------------

def _wake(tms, caplog, by_demand):
    from sglang.srt.environ import envs

    with rt._Ctx(rt._armed(tms, max_by="0,0").stack
                 + [envs.SGLANG_WEG2_D_KV_STAGE_BY_DEMAND.override(by_demand)]):
        r = rt._rank(tms)
        for p in list(tms.allocs):
            tms.pause(p)
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        for c in r.caches:
            for buf in c._resident.values():
                tms.resume(buf.data_ptr())
        caplog.clear()
        caplog.set_level(logging.INFO)
        kv = types.SimpleNamespace(epoch="e1", handoff_n=2, parked_n=0, phase_kv_tokens=100)
        st = dsv.on_wake(r.sched, kv, rt._seats(2))
        lines = [m for m in caplog.messages if m.startswith(dsv.STAGE_MARK + " n=")]
        return st, lines


def test_the_rank_takes_the_demands_stage_and_names_the_form(caplog):
    # the table allows S0 only (max_by 0,0); demand 100 > S0 (64 tokens)
    st, lines = _wake(rt.FakeTms(), caplog, True)
    assert (st.stage, st.stage_tokens, st.over) == (1, 128, False)
    assert len(lines) == 1 and "stage=S1" in lines[0] and lines[0].endswith(" form=demand")


def test_off_the_rank_keeps_the_tables_stage_and_its_line(caplog):
    st, lines = _wake(rt.FakeTms(), caplog, False)
    assert (st.stage, st.stage_tokens, st.over) == (0, 64, True)
    assert len(lines) == 1 and "stage=S0" in lines[0] and "form=" not in lines[0]


# ---- (6)/(7) the launcher -----------------------------------------------------------

def _ns(env_d, derived=True):
    return types.SimpleNamespace(env_d=env_d, d_pool_waves_derived=derived)


def _apply(ns):
    from sglang.srt.weg2 import launcher as L

    return L.apply_d_kv_stage_form(ns, er, lt._rows(), lt.FORM, lt._plan(), "D",
                                   verify_tokens=4, top_k=10)


def test_the_switch_writes_every_stage_and_raises_the_derived_wave_cap():
    from sglang.srt.weg2 import launcher as L

    before = "SGLANG_MOE_SCRATCH_SLOTS=100,48,48;%s=2;%s=1" % (WAVES, DEMAND)
    ns = _ns(before)
    lines = _apply(ns)
    env = L.parse_group_env(ns.env_d)
    assert env[MAX_BY] == "2,2,2,2,2,2"
    # the fixture's capture waves at the top stage everywhere: 1,2,2,3,3,3
    assert env[WAVES] == "3"
    assert env["SGLANG_MOE_SCRATCH_SLOTS"] == "67,48,48" and env[DEMAND] == "1"
    dem = [ln for ln in lines if "BEDARF (#251d)" in ln]
    assert len(dem) == 1 and "Zusatzwellen je bs [0, 1, 0, 1, 1, 1]" in dem[0]
    wav = [ln for ln in lines if "WELLEN (#251d)" in ln]
    assert len(wav) == 1 and "%s 2 -> 3" % WAVES in wav[0]
    assert "[1, 2, 2, 3, 3, 3] statt [1, 1, 2, 2, 2, 2]" in wav[0]
    # a second solve pass: everything this form wrote goes back, then again
    L.d_kv_stage_undo(ns)
    assert L.parse_group_env(ns.env_d) == L.parse_group_env(before)
    _apply(ns)
    assert L.parse_group_env(ns.env_d)[WAVES] == "3"


def test_a_told_wave_cap_below_the_need_is_refused_by_name():
    from sglang.srt.weg2 import launcher as L

    told = "SGLANG_MOE_SCRATCH_SLOTS=100,48,48;%s=2;%s=1" % (WAVES, DEMAND)
    ns = _ns(told, derived=False)
    with pytest.raises(L.Weg2DKvStageWavesRefused, match="W169 Weg2DKvStageWavesRefused"):
        _apply(ns)
    assert ns.env_d == told  # nothing written
    assert issubclass(L.Weg2DKvStageWavesRefused, L.Weg2DKvStageMaxRefused)
    # a told cap that carries the need passes untouched
    ok = _ns("SGLANG_MOE_SCRATCH_SLOTS=100,48,48;%s=3;%s=1" % (WAVES, DEMAND), derived=False)
    lines = _apply(ok)
    assert L.parse_group_env(ok.env_d)[WAVES] == "3"
    assert not any("WELLEN (#251d)" in ln for ln in lines)


def test_the_operator_arm_kv211111_gets_the_waves_its_capture_needs():
    from sglang.srt.weg2 import launcher as L

    ns = _ns("SGLANG_MOE_SCRATCH_SLOTS=100,48,48;%s=2;%s=2,1,1,1,1,1" % (WAVES, MAX_BY))
    lines = _apply(ns)
    env = L.parse_group_env(ns.env_d)
    assert env[MAX_BY] == "2,1,1,1,1,1" and env[WAVES] == "3"
    assert any("WELLEN (#251d)" in ln and "2 -> 3" in ln for ln in lines)
    # the s2probe arm needs 2 waves at bs2 -- the cap of 2 carries it
    probe = _ns("SGLANG_MOE_SCRATCH_SLOTS=100,48,48;%s=2;%s=2,2,2,1,0,0" % (WAVES, MAX_BY))
    lines = _apply(probe)
    assert L.parse_group_env(probe.env_d)[WAVES] == "2"
    assert not any("WELLEN (#251d)" in ln for ln in lines)


def test_without_the_switch_no_wave_key_moves():
    from sglang.srt.weg2 import launcher as L

    for env_d in ("SGLANG_MOE_SCRATCH_SLOTS=100,48,48;%s=2" % WAVES,
                  "SGLANG_MOE_SCRATCH_SLOTS=100,48,48;%s=2;%s=0" % (WAVES, DEMAND)):
        ns = _ns(env_d)
        lines = _apply(ns)
        env = L.parse_group_env(ns.env_d)
        assert env[MAX_BY] == "2,1,1,1,0,0" and env[WAVES] == "2"
        assert not any("#251d" in ln for ln in lines)


def test_the_capture_waves_are_todays_plus_the_extra():
    t = lt._table()
    for mb in (t.max_by_seats, (2, 1, 1, 1, 1, 1), (2,) * 6, (2, 2, 2, 1, 0, 0)):
        assert tuple(a - b for a, b in zip(t.capture_waves(mb), t.waves)) == t.extra_waves(mb)
    assert t.capture_waves((2, 1, 1, 1, 1, 1)) == (1, 1, 2, 2, 3, 3)


def test_the_refusal_passes_the_seat_tables_except():
    from sglang.srt.weg2 import launcher as L

    src = open(L.__file__).read()
    k = src.index("def d_seat_table_lines(")
    body = src[k:src.index("\ndef ", k + 10)]
    # W169 is a W168 subclass: the named except before the informational one lets it through
    assert body.index("except Weg2DKvStageMaxRefused:") < body.index("except Exception as exc:")
