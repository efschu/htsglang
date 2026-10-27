"""27B NACHT2 22.10. (desk/flliper-27b-nacht2-1022): the review Auflagen and the NF ports of the integration.

(A) W65 review, SPILL-TP (1b7d45edb9) left one rank-local release of a shared mamba slot unguarded: the H19 FULL
    displacement in ``_pdflip_mamba_claim_h19`` (release a victim, claim again). On a TP group off Form A the claim
    right after succeeds only on the LAST releaser -- one rank backs the node, the others do not (the W65 class).
    Not hit in the 27B INT8 rounds 3b/3c (0 ``DISPLACE why=full``), guarded by the same rule.
(B) Dual review (66b171e36e): every wake from sleep waits for ``sleep_leg_done``; a sleep leg that RAISED left the
    ladder 'sleeping' with the flag down for the rest of the boot (P held, silent). Before the release: back to
    'stopped'; after it: the named stop.
(C) Extend line port (NF rc33 bc9dcc7520, abgestimmt with a4f4b8fec1): the width vote keeps THIS card's derived
    corridor floor (guard door, the number the rank's breach instrument grades against), not the stated 1024 law.
    Metal round 3c (1022_015348): TP1 graded 700, voted cap=1 at post 864 under the 1024 line -> 256 floor chunk.
(D) envs override restores in finally (NF review cd5b187437).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache import form_a_host_shadow as _r12  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as URC  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------------------------------------------------
# (A) H19 FULL on a shared slot
# ---------------------------------------------------------------------------------------------------------------------

class _SharedArena:
    """One full arena whose victim slot every rank of the group references; a release frees it only at the last."""

    def __init__(self, ranks):
        self.refs = ranks
        self.free = 0

    def release(self):
        self.refs -= 1
        if self.refs == 0:
            self.free += 1
            return 1
        return 0


def _h19_rank(arena, tp, role, monkeypatch):
    monkeypatch.setattr(_r12, "role", lambda: role)
    released = []
    victim = types.SimpleNamespace(node=types.SimpleNamespace(id=282), depth=51200, slots=[7], rid="v")

    class _Mp:
        arena_slots = 112

        def alloc_write(self, keys):
            if arena.free > 0:
                arena.free -= 1
                return [1]
            return None

    t = types.SimpleNamespace(tp_world_size=tp, pp_size=1, root_node=None, _pdflip_rid_anchor_cfg=4)
    t._pdflip_anchor_ledger = types.SimpleNamespace(of=lambda rid: object())
    t._pdflip_told_unheld = lambda owned: owned
    t._pdflip_anchor_of = lambda n, mp: None
    t._pdflip_full_victim = lambda node, rid, owned, mp: victim
    t._pdflip_dual_claim_retry = lambda node, mp, h: None
    t._pdflip_note_anchor_claim = lambda *a, **k: None
    t._r12_anchor_deletable = lambda x: x

    def _release(v, mp, st, why, for_rid):
        released.append(why)
        return arena.release()

    t._pdflip_release_anchor = _release
    node = types.SimpleNamespace(id=307, pdflip_anchor_rid="pdflip-4-73", pdflip_arena_displaced=False)
    monkeypatch.setattr(URC._mad, "rid_anchor_cap", lambda configured, arena_slots: 4)
    monkeypatch.setattr(URC._mad, "ancestor_path", lambda target, root: ([], 51712))
    monkeypatch.setattr(URC._mad, "owned_anchors", lambda path, rid, anchor_of: [])
    rows = UnifiedRadixCache._pdflip_mamba_claim_h19(t, node, _Mp(), 0xABC)
    return rows, released


def _h19_group(tp, role, monkeypatch):
    arena = _SharedArena(tp)
    return [_h19_rank(arena, tp, role, monkeypatch) for _ in range(tp)]


def test_h19_full_on_a_shared_tp_group_ends_alike_on_every_rank(monkeypatch, caplog):
    UnifiedRadixCache._spill_tp_skip_n = 0
    with caplog.at_level(logging.INFO):
        out = _h19_group(3, None, monkeypatch)
    rows = [r for r, _ in out]
    assert rows == [None, None, None], "no rank backs the node alone"
    assert all(rel == [] for _, rel in out), "no rank-local release of the shared slot"
    assert "SPILL SKIPPED site=full tp_size=3" in caplog.text


def test_h19_full_mutant_without_the_guard_splits_the_ranks(monkeypatch):
    # the danger direction (ranks disagree): without the guard only the last releaser backs the node
    monkeypatch.setattr(UnifiedRadixCache, "_shared_slot_spill_skipped", lambda self, node, site: False)
    rows = [r for r, _ in _h19_group(3, None, monkeypatch)]
    assert rows == [None, None, [1]]


@pytest.mark.parametrize("tp,role", [(1, None), (3, "host")])
def test_h19_full_single_rank_and_form_a_still_displace(monkeypatch, tp, role):
    rows, released = _h19_group(tp, role, monkeypatch)[-1]
    assert released == ["full"]
    if tp == 1:
        assert rows == [1]


# ---------------------------------------------------------------------------------------------------------------------
# (B) the dual sleep leg never leaves the ladder blind
# ---------------------------------------------------------------------------------------------------------------------

def _front(quiesce, leg_rpc):
    from flliper.srt.pdflip import dual_d_priority as DP
    from flliper.srt.pdflip import front as FR

    stops = []
    st = DP.PressureStages(sleep_capable=True, sleep_after=1)
    st.p_state, st.sleep_leg_done = "sleeping", False          # the ladder ISSUED the sleep (stage 2)
    f = types.SimpleNamespace(groups={"P": "P"}, counters=collections.Counter(), weight_chunks=8, boot_epoch=1,
                              _dual_stages_obj=st, quiesce=quiesce, leg_rpc=leg_rpc)
    f._dual_p_weights_tags = types.MethodType(FR.Front._dual_p_weights_tags, f)
    f.do_stop = lambda code, why: stops.append((code, why))
    return FR.Front._dual_p_sleep, f, st, stops


def test_dual_sleep_leg_raising_before_the_release_returns_to_stopped():
    async def quiesce(_g):
        raise ConnectionResetError("flush_cache connection reset")

    async def leg_rpc(*_a):
        raise AssertionError("no release after a failed quiesce")

    sleep, f, st, stops = _front(quiesce, leg_rpc)
    asyncio.run(sleep(f))
    assert st.p_state == "stopped" and st.sleep_leg_done is True, "nothing parked: the ladder may ask again"
    assert stops == [] and f.counters["dual_kv_pressure_sleep_raised"] == 1
    # the next ticks can climb the ladder again (not stuck in 'sleeping' with the flag down)
    kw = dict(p_committed=0, free_min=20 << 30, p_grant_bytes=64 << 20, d_air_bytes=64 << 20, weights_bytes=1 << 30,
              card_room=[(20 << 30, 2 << 30)] * 3)
    actions = [st.tick(pressure=1, seats_done=0, **kw)[0] for _ in range(4)]
    assert "sleep" in actions


def test_dual_sleep_leg_raising_after_the_release_is_the_named_stop():
    async def quiesce(_g):
        return True, ""

    async def leg_rpc(*_a):
        raise TimeoutError("release answer lost")

    sleep, f, st, stops = _front(quiesce, leg_rpc)
    asyncio.run(sleep(f))
    assert [c for c, _ in stops] == ["W-DUAL-P-SLEEP PdFlipDualPSleepRefused"]
    assert "raised after the release was sent" in stops[0][1]
    assert st.sleep_leg_done is False, "P's VRAM state is unknown: no wake is ever issued on it"


def test_dual_sleep_leg_happy_path_unchanged():
    calls = []

    async def quiesce(_g):
        return True, ""

    async def leg_rpc(_g, path, payload, _t):
        calls.append(path)
        return 200, "ok"

    sleep, f, st, stops = _front(quiesce, leg_rpc)
    asyncio.run(sleep(f))
    assert calls == ["/release_memory_occupation"] and st.sleep_leg_done is True and stops == []


# ---------------------------------------------------------------------------------------------------------------------
# (C) the extend line is the card's derived corridor floor
# ---------------------------------------------------------------------------------------------------------------------

RATE = 0.3091
CONFIGURED = 4096


class _Cuda:
    def __init__(self, free, reserved=17800, allocated=17000):
        self.free, self.reserved, self.allocated = free, reserved, allocated

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return int(self.free * (1 << 20)), int(20480 * (1 << 20))

    def memory_reserved(self):
        return int(self.reserved * (1 << 20))

    def memory_allocated(self):
        return int(self.allocated * (1 << 20))

    def synchronize(self):
        pass

    def empty_cache(self):
        pass


def _door(mib, source="MEASURED-D"):
    from flliper.srt.managers import corridor_guard as CG

    calls = []

    def door(*a, **k):
        calls.append(1)
        return types.SimpleNamespace(mib=mib, source=source,
                                     verdict_floor_mib=CG.verdict_floor_for_mib(mib, source))

    return mock.patch.object(CG, "corridor_floor_for_current_device", door), calls


def _vote(cuda, floor="1"):
    env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": str(RATE)}
    if floor:
        env["FLLIPER_PDFLIP_EXTEND_CAP_FLOOR"] = floor
    with mock.patch.dict(os.environ, env, clear=False):
        for k in ("FLLIPER_PDFLIP_TORCH_CACHE_CAP", "FLLIPER_PDFLIP_EXTEND_TRIM_MIB",
                  "FLLIPER_PDFLIP_EXTEND_RATE_MEASURE", "FLLIPER_CORRIDOR_LAW_FLOOR_MIB"):
            os.environ.pop(k, None)
        if not floor:
            os.environ.pop("FLLIPER_PDFLIP_EXTEND_CAP_FLOOR", None)
        return ET.width_vote(cuda, 1, CONFIGURED, 1, True)


@pytest.fixture(autouse=True)
def _reset_et(monkeypatch):
    # EXTEND-CAP-SOFT (27.10.): the vote tests below pin the pre-soft vote (the FLLIPER_PDFLIP_EXTEND_SOFT_LINE=0
    # path); the soft rule is pinned in test_pdflip_extend_soft_line_1027
    monkeypatch.setenv("FLLIPER_PDFLIP_EXTEND_SOFT_LINE", "0")
    ET.reset_for_tests()
    yield
    ET.reset_for_tests()


def test_round3c_tp1_votes_to_its_own_700_law_not_the_1024_floor_chunk(caplog):
    patch, calls = _door(700)
    with patch, caplog.at_level(logging.INFO):
        w = _vote(_Cuda(free=864))
    assert w == 530                                              # floor((864 - 700) / 0.3091)
    assert 864 - w * RATE >= 700, "the priced end keeps the card's law"
    assert "EXTEND-CORRIDOR-LINE rank=1 line_mib=700 source=MEASURED-D:700" in caplog.text


def test_round3c_old_line_1024_was_the_floor_chunk():
    # the pre-port behaviour at the same metal state, for the record: the stated 1024 law -> cap 1 -> floor chunk 256
    with mock.patch.object(ET, "_read_corridor_line", lambda: (1024.0, "stated-law:1024")):
        assert _vote(_Cuda(free=864)) == ET.floor_rows(CONFIGURED, 1) == 256


def test_line_is_read_once_per_process():
    patch, calls = _door(701)
    with patch:
        for free in (1044, 900, 1500):
            _vote(_Cuda(free=free))
    assert len(calls) == 1


def test_danger_grid_no_vote_above_the_floor_chunk_ends_under_the_card_line():
    # danger direction "extend under the corridor": every vote the line funds (no Q-710 raise) ends at or above it
    for line in (700, 701, 767, 1024):
        for free in range(300, 4000, 37):
            ET.reset_for_tests()
            patch, _ = _door(line)
            with patch:
                w = _vote(_Cuda(free=free), floor="")
            if w is None:
                assert free - CONFIGURED * RATE >= line - 1
            elif w > 1:
                assert free - w * RATE >= line - 1, (line, free, w)


def test_unmeasured_card_falls_back_to_the_stated_law_itself():
    # NACHT2 review 22.10.: the line is the floor's TARGET -- on an unmeasured card the stated law (the pre-port line),
    # never the -20 % band edge (819) the guard grades a breach against (test_27b_nacht2_review_1022 R-LINE)
    patch, _ = _door(1024, source="UNMEASURED-FALLBACK")
    with patch:
        assert ET.corridor_line_mib() == 1024.0


def test_no_door_reads_the_stated_law_and_never_under_the_halt_line():
    from flliper.srt.managers import corridor_guard as CG

    def boom(*a, **k):
        raise RuntimeError("no nvml")

    with mock.patch.object(CG, "corridor_floor_for_current_device", boom), \
            mock.patch.dict(os.environ, {"FLLIPER_CORRIDOR_LAW_FLOOR_MIB": "100"}):
        assert ET.corridor_line_mib() == float(ET.CHUNKING_FLOOR_MIB)


def test_mutant_line_back_to_the_stated_law_is_red():
    # mutant: the old reader (stated 1024 law) at the round-3c TP1 state -> the floor chunk, not the card's 530 rows
    with mock.patch.object(ET, "_read_corridor_line", lambda: (1024.0, "stated-law:1024")):
        w = _vote(_Cuda(free=864))
    assert w != 530


# ---------------------------------------------------------------------------------------------------------------------
# (D) envs override restores in finally
# ---------------------------------------------------------------------------------------------------------------------

def test_env_override_restores_when_the_body_raises(monkeypatch):
    from flliper.srt.environ import envs

    var = envs.FLLIPER_PDFLIP_EXTEND_TRIM_MIB
    monkeypatch.delenv(var.name, raising=False)
    with pytest.raises(RuntimeError):
        with var.override("1,2,3"):
            assert os.environ[var.name] == "1,2,3"
            raise RuntimeError("red test body")
    assert var.name not in os.environ
