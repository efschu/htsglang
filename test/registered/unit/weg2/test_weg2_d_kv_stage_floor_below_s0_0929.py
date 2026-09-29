"""29.09. (Nutzer 12:35Z / 12:42Z, Grundgesetz): KV stages BELOW the booked S0.

D booked S0 = 262144 tokens of KV even when the wake's sessions hold 17k; the
bytes of the unused KV belong to experts. With SGLANG_WEG2_D_KV_STAGE_FLOOR_TOKENS
the stage ladder starts at the floor (floor, 2 x floor, ... < S0, then S0 and the
stages above as today): the KV between the floor and S0 is born unmapped and its
bytes fund expert rows (rounded down); the wake takes the stage holding its KNOWN
tokens, the D-MEM-SCHED tick grows it between rounds. S0 and every stage above
keep exactly today's rows, so the capture floor and the waves do not move.
"""

import os
import sys
import types

from sglang.test.test_utils import CustomTestCase  # noqa: F401

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weg2_d_kv_stage_launcher_251c as T251  # noqa: E402
from sglang.srt.planner import expert_residency as er  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

FLOOR = 32768


def _table(floor):
    return er.kv_stage_table(T251._rows(), T251.FORM, kv_cell_bytes=14143, kv_tokens=262144,
                             local_experts=193, verify_tokens=4, top_k=10, staging_rows=12,
                             floor_tokens=floor)


def test_no_floor_is_todays_table():
    assert _table(0) == T251._table()
    assert _table(262144) == T251._table()  # a floor at S0 adds nothing


def test_the_floor_ladder_funds_rows_and_keeps_s0_and_above():
    old, new = _table(0), _table(FLOOR)
    assert new.tokens == tuple(range(FLOOR, 262144, FLOOR)) + old.tokens
    # the unmapped KV between the floor and S0: 229376 x 14143 B = 3093.7 MiB
    # over 112.5 MiB rows -> 27 rows, rounded DOWN (never a byte above the KV)
    assert new.low_rows == (262144 - FLOOR) * 14143 // int(112.5 * 2**20) == 27
    assert new.low_rows * 112.5 * 2**20 <= (262144 - FLOOR) * 14143
    s0 = new.tokens.index(262144)
    # S0 and every stage above: exactly today's stage rows and capacities
    assert tuple(r - new.low_rows for r in new.stage_rows[s0:]) == old.stage_rows
    assert new.rows == old.rows + new.low_rows
    for caps_new, caps_old in zip(new.capacity, old.capacity):
        assert caps_new[s0:] == caps_old
        # below S0 the phase has MORE rows ON
        assert all(c >= caps_old[0] for c in caps_new[:s0])
    assert new.capacity[0][0] == old.capacity[0][0] + new.low_rows
    # the waves of every captured batch are today's (the top stage is unchanged)
    assert new.waves == old.waves and new.need == old.need
    top = tuple(len(new.tokens) - 1 for _ in new.max_by_seats)
    top_old = tuple(len(old.tokens) - 1 for _ in old.max_by_seats)
    assert new.capture_waves(top) == old.capture_waves(top_old)


def test_the_launcher_writes_the_ladder_and_grows_the_bank_by_the_low_rows():
    before = "SGLANG_MOE_SCRATCH_SLOTS=100,48,48;SGLANG_WEG2_D_SEAT_EXPERT_ROWS=14,0,0"
    ns = types.SimpleNamespace(env_d=before + ";SGLANG_WEG2_D_KV_STAGE_FLOOR_TOKENS=32768")
    L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, T251._plan(), "D",
                            verify_tokens=4, top_k=10)
    env = L.parse_group_env(ns.env_d)
    assert env["SGLANG_WEG2_D_KV_STAGE_TOKENS"].split(",")[0] == "32768"
    assert env["SGLANG_WEG2_D_KV_STAGE_TOKENS"].split(",")[-3:] == ["262144", "393216", "524288"]
    assert env["SGLANG_WEG2_D_KV_STAGE_ROWS"] == str(33 + 27)
    # the scratch loses only today's 33 stage rows; the 27 low rows are NEW bank
    # rows (their KV is born unmapped): seat rows 14 + 33 + 27
    assert env["SGLANG_MOE_SCRATCH_SLOTS"] == "67,48,48"
    assert env["SGLANG_WEG2_D_SEAT_EXPERT_ROWS"] == "74,0,0"


def test_the_rank_runs_a_floor_ladder_by_the_known_tokens():
    """The rank reads only the stage list and the rows: a floor ladder is a
    longer list. RT geometry: one 64-token step = 16 rows; a floor of 32 below
    S0 = 64 funds 8 more rows -> 40 rows ON at the floor (bank X = 40)."""
    import test_weg2_d_kv_stage_runtime_251c as RT

    from sglang.srt.environ import envs
    from sglang.srt.weg2 import d_seat_vram as dsv

    tms = RT.FakeTms()
    with RT._armed(tms, tokens="32,64,128,192", rows=40), \
            envs.SGLANG_WEG2_D_KV_STAGE_BY_DEMAND.override(True):
        r = RT._rank(tms)
        for p in list(tms.allocs):
            tms.pause(p)
        st = dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
        ctl = r.sched._weg2_d_seat_vram
        # the weights leg has no demand: the floor, every stage row ON
        assert (st.stage, st.stage_tokens, ctl.applied.extra_rows) == (0, 32, 40)
        for c in r.caches:
            for buf in c._resident.values():
                tms.resume(buf.data_ptr())
        # the KNOWN tokens of the wake choose the stage: 50 -> S0's 64
        kv = types.SimpleNamespace(epoch="e1", handoff_n=2, parked_n=0, phase_kv_tokens=50)
        st = dsv.on_wake(r.sched, kv, RT._seats(2))
        assert (st.stage, st.stage_tokens) == (1, 64)
        assert ctl.applied.extra_rows == 40 - 8
        assert r.kv_alloc.available_size() == 64  # the allocator hands out S0's pages


def test_the_floor_key_reads_the_group_env_and_is_off_by_default():
    assert L.d_kv_stage_floor_tokens({}) == 0
    assert L.d_kv_stage_floor_tokens({L.D_KV_STAGE_FLOOR_KEY: "32768"}) == 32768
    assert L.d_kv_stage_floor_tokens({L.D_KV_STAGE_FLOOR_KEY: "junk"}) == 0
