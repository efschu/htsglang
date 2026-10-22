"""27B NACHT2 review 22.10. (desk/flliper-27b-nacht2-1022-review, adversarial review of 37602df8dd / 1c5c3b4d11).

(R-B1) The dual stage-2 leg that ends BEFORE its release (quiesce refused, or -- since 37602df8dd -- the leg raised)
       falls back to 'stopped'. The sleep is issued only from 'lent', so P's awake loan is still out; from 'stopped'
       the ladder RESUMED P without the reclaim (P prefills while the card pool still counts its loan: over-promise).
       Fixed in PressureStages.tick: 'stopped' with p_lent > 0 goes back through the reclaim, as from 'lent'.
(R-LINE) The extend width line (1c5c3b4d11) took the guard's VERDICT floor: on an UNMEASURED-FALLBACK card that is
       the -20 % band edge (819 at 1024), i.e. the vote got WIDER than before the port on exactly the card nobody
       measured (after the rig's card swap the hardware fingerprint changes and every card reads the fallback).
       Fixed: the line is the floor's TARGET (CorridorFloor.mib) -- equal to the verdict floor on a measured card
       (the port's gain stays), the stated law on an unmeasured one (the pre-port line, never wider).
(R-H19) The A1 guard covers H19 FULL; H19 SHARE + a full arena keeps the same split on a shared TP group (only the
       LAST releaser's claim succeeds). Latent: the 27B profile keeps the rid cap at 0 (grid4096), NF D is Form A.
       Pinned here as the premise (cap 0 on the 27B) plus a strict xfail that names the residual.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import os
import textwrap
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache import form_a_host_shadow as _r12  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as URC  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import dual_d_priority as DP  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GiB = 1 << 30
KW = dict(p_grant_bytes=64 << 20, d_air_bytes=64 << 20, weights_bytes=GiB, card_room=[(20 * GiB, 2 * GiB)] * 3)


# ---------------------------------------------------------------------------------------------------------------------
# (R-B1) a stage-2 leg that ended before its release never resumes P around its loan
# ---------------------------------------------------------------------------------------------------------------------

def _ladder_at_sleep(stages_cls=None):
    st = (stages_cls or DP.PressureStages)(sleep_capable=True, sleep_after=1, reclaim_after=1)
    acts = [st.tick(pressure=5, p_committed=1, free_min=0, seats_done=0, **KW)[0],
            st.tick(pressure=5, p_committed=0, free_min=0, seats_done=0, **KW)[0],
            st.tick(pressure=5, p_committed=0, free_min=0, seats_done=0, p_lent=3 * GiB, **KW)[0]]
    assert acts == ["stop", "lend", "sleep"] and st.p_state == "sleeping"
    return st


def _sleep_leg(st, quiesce, leg_rpc):
    from flliper.srt.pdflip import front as FR

    stops = []
    f = types.SimpleNamespace(groups={"P": "P"}, counters=collections.Counter(), weight_chunks=8, boot_epoch=1,
                              _dual_stages_obj=st, quiesce=quiesce, leg_rpc=leg_rpc)
    f._dual_p_weights_tags = types.MethodType(FR.Front._dual_p_weights_tags, f)
    f.do_stop = lambda code, why: stops.append((code, why))
    asyncio.run(FR.Front._dual_p_sleep(f))
    return stops


async def _raise_reset(_g):
    raise ConnectionResetError("flush_cache connection reset")


async def _refuse(_g):
    return False, "flush_cache 400 for 90 s"


async def _no_release(*_a):
    raise AssertionError("no release after a failed quiesce")


@pytest.mark.parametrize("quiesce", [_raise_reset, _refuse], ids=["leg-raised", "quiesce-refused"])
def test_fallback_with_the_loan_out_goes_back_through_the_reclaim_never_a_plain_resume(quiesce):
    st = _ladder_at_sleep()
    assert _sleep_leg(st, quiesce, _no_release) == []
    assert st.p_state == "stopped" and st.sleep_leg_done
    # the pressure is gone, the card has room, P's awake loan (3 GiB) is still with the card pool
    act, line = st.tick(pressure=0, p_committed=0, free_min=20 * GiB, seats_done=1, p_lent=3 * GiB, **KW)
    assert act == "reclaim" and st.p_state == "reclaiming", (act, st.p_state)
    assert "from=stopped" in line
    # the loan stays out while D holds part of it: P stays held, asked again
    assert st.tick(pressure=0, p_committed=0, free_min=20 * GiB, seats_done=1, p_lent=3 * GiB, **KW)[0] == "reclaim"
    assert st.p_state == "reclaiming"
    # the loan is back: only now P serves
    assert st.tick(pressure=0, p_committed=0, free_min=20 * GiB, seats_done=1, p_lent=0, **KW)[0] == "resume"
    assert st.p_state == "serving"


def test_fallback_without_room_holds_and_pressure_relends():
    st = _ladder_at_sleep()
    _sleep_leg(st, _raise_reset, _no_release)
    tight = dict(KW, card_room=[(GiB, 2 * GiB)] * 3)
    assert st.tick(pressure=0, p_committed=0, free_min=GiB, seats_done=1, p_lent=3 * GiB, **tight)[0] is None
    assert st.p_state == "stopped", "no room for the return: P stays held"
    assert st.tick(pressure=5, p_committed=0, free_min=0, seats_done=1, p_lent=3 * GiB, **KW)[0] == "lend"
    assert st.p_state == "lent", "the pressure is back: the ladder climbs again from the loan"


def test_stopped_without_a_loan_resumes_as_before():
    st = DP.PressureStages(sleep_capable=True, sleep_after=9)
    assert st.tick(pressure=5, p_committed=1, free_min=0, seats_done=0, **KW)[0] == "stop"
    assert st.tick(pressure=0, p_committed=0, free_min=20 * GiB, seats_done=0, p_lent=0, **KW)[0] == "resume"


def test_mutant_without_the_loan_check_resumes_p_around_its_loan(monkeypatch):
    src = textwrap.dedent(inspect.getsource(DP.PressureStages))
    fixed = "if int(p_lent) > 0:"
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant"
    ns = dict(vars(DP))
    exec(compile(src.replace(fixed, "if False:"), DP.__file__, "exec"), ns)
    st = _ladder_at_sleep(ns["PressureStages"])
    _sleep_leg(st, _raise_reset, _no_release)
    act, _ = st.tick(pressure=0, p_committed=0, free_min=20 * GiB, seats_done=1, p_lent=3 * GiB, **KW)
    assert act == "resume" and st.p_state == "serving", "the danger direction the fix closes"


# ---------------------------------------------------------------------------------------------------------------------
# (R-LINE) the extend line is the floor's TARGET: never wider than the pre-port line on an unmeasured card
# ---------------------------------------------------------------------------------------------------------------------

RATE = 0.3091
CONFIGURED = 4096


class _Cuda:
    def __init__(self, free):
        self.free = free

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self):
        return int(self.free * (1 << 20)), int(20480 * (1 << 20))

    def memory_reserved(self):
        return 17800 << 20

    def memory_allocated(self):
        return 17000 << 20

    def synchronize(self):
        pass

    def empty_cache(self):
        pass


@pytest.fixture(autouse=True)
def _reset_et():
    ET.reset_for_tests()
    yield
    ET.reset_for_tests()


def _real_floor(mib, source):
    from flliper.srt.managers import corridor_guard as CG

    return CG.CorridorFloor(card_uuid="GPU-test", group="D", transient_mib=mib, reserve_mib=0, source=source)


def _vote(free, line_patch):
    env = {"FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB": str(RATE)}
    with mock.patch.dict(os.environ, env, clear=False):
        for k in ("FLLIPER_PDFLIP_TORCH_CACHE_CAP", "FLLIPER_PDFLIP_EXTEND_TRIM_MIB", "FLLIPER_PDFLIP_EXTEND_CAP_FLOOR",
                  "FLLIPER_PDFLIP_EXTEND_RATE_MEASURE", "FLLIPER_CORRIDOR_LAW_FLOOR_MIB"):
            os.environ.pop(k, None)
        ET.reset_for_tests()
        with line_patch:
            return ET.width_vote(_Cuda(free), 1, CONFIGURED, 1, True)


@pytest.mark.parametrize("source,mib,want", [
    ("MEASURED-D", 700, 700.0),               # the port's gain: the card's measured law (round 3c TP1)
    ("UNMEASURED-FALLBACK", 1024, 1024.0),    # the pre-port line, NOT the 819 band edge
    ("ENV-OVERRIDE", 900, 900.0),             # an operator's hand-set law is the line itself (pre-port semantics)
])
def test_line_is_the_floor_target(source, mib, want):
    from flliper.srt.managers import corridor_guard as CG

    with mock.patch.object(CG, "corridor_floor_for_current_device", lambda *a, **k: _real_floor(mib, source)):
        assert ET.corridor_line_mib() == want


def test_unmeasured_card_never_votes_wider_than_before_the_port():
    # danger direction "too wide on a card nobody measured" (card swap -> fingerprint mismatch -> fallback everywhere)
    from flliper.srt.managers import corridor_guard as CG

    for free in range(300, 4000, 41):
        port = _vote(free, mock.patch.object(CG, "corridor_floor_for_current_device",
                                             lambda *a, **k: _real_floor(1024, "UNMEASURED-FALLBACK")))
        pre = _vote(free, mock.patch.object(ET, "_read_corridor_line", lambda: (1024.0, "stated-law:1024")))
        assert (port or CONFIGURED) <= (pre or CONFIGURED), (free, port, pre)


def test_no_door_reads_the_stated_law_itself():
    from flliper.srt.managers import corridor_guard as CG

    def boom(*a, **k):
        raise RuntimeError("no nvml")

    with mock.patch.object(CG, "corridor_floor_for_current_device", boom):
        os.environ.pop("FLLIPER_CORRIDOR_LAW_FLOOR_MIB", None)
        assert ET.corridor_line_mib() == float(CG.corridor_law_mib())


def test_mutant_verdict_floor_line_is_wider_on_an_unmeasured_card():
    from flliper.srt.managers import corridor_guard as CG

    src = inspect.getsource(ET._read_corridor_line)
    assert src.count("return float(fl.mib)") == 1
    ns = dict(vars(ET))
    exec(compile(textwrap.dedent(src).replace("return float(fl.mib)", "return float(fl.verdict_floor_mib)"),
                 ET.__file__, "exec"), ns)
    with mock.patch.object(CG, "corridor_floor_for_current_device",
                           lambda *a, **k: _real_floor(1024, "UNMEASURED-FALLBACK")):
        assert ns["_read_corridor_line"]()[0] == 819.0, "the band edge the fix keeps the vote off"


# ---------------------------------------------------------------------------------------------------------------------
# (R-H19) the A1 premise on the 27B and the named SHARE residual
# ---------------------------------------------------------------------------------------------------------------------

def test_27b_profile_keeps_the_rid_cap_off_so_h19_share_and_full_are_unreachable():
    from flliper.srt.pdflip import form as F
    from flliper.srt.pdflip import mamba_arena_displace as MAD

    sw = F.PROFILES[F.PROFILE_QWEN27B].switch_defaults()
    assert sw["FLLIPER_PDFLIP_MAMBA_ARENA_RID_ANCHORS"] == 0, (
        "the 27B left grid4096: H19 SHARE on a shared TP slot splits the ranks (see the xfail below) -- guard it first")
    assert MAD.rid_anchor_cap(configured=0, arena_slots=112) == 0


class _SharedArena:
    def __init__(self, ranks):
        self.refs, self.free = ranks, 0

    def release(self):
        self.refs -= 1
        if self.refs == 0:
            self.free += 1
            return 1
        return 0


def _share_rank(arena, monkeypatch):
    monkeypatch.setattr(_r12, "role", lambda: None)
    own = URC._mad.OwnedAnchor(node=types.SimpleNamespace(id=282), depth=51200, rid="pdflip-4-73", slots=(7,))

    class _Mp:
        arena_slots = 112

        def alloc_write(self, keys):
            if arena.free > 0:
                arena.free -= 1
                return [1]
            return None

    t = types.SimpleNamespace(tp_world_size=3, pp_size=1, root_node=None, _pdflip_rid_anchor_cfg=1)
    t._pdflip_anchor_ledger = types.SimpleNamespace(of=lambda rid: object())
    t._pdflip_told_unheld = lambda owned: owned
    t._pdflip_anchor_of = lambda n, mp: None
    t._pdflip_full_victim = lambda node, rid, owned, mp: None
    t._pdflip_dual_claim_retry = lambda node, mp, h: None
    t._pdflip_note_anchor_claim = lambda *a, **k: None
    t._r12_anchor_deletable = lambda x: x
    t._pdflip_release_anchor = lambda v, mp, st, why, for_rid: arena.release()
    node = types.SimpleNamespace(id=307, pdflip_anchor_rid="pdflip-4-73", pdflip_arena_displaced=False)
    monkeypatch.setattr(URC._mad, "rid_anchor_cap", lambda configured, arena_slots: 1)
    monkeypatch.setattr(URC._mad, "ancestor_path", lambda target, root: ([], 51712))
    monkeypatch.setattr(URC._mad, "owned_anchors", lambda path, rid, anchor_of: [own])
    return UnifiedRadixCache._pdflip_mamba_claim_h19(t, node, _Mp(), 0xABC)


@pytest.mark.xfail(strict=True, reason="R-H19 residual, latent (27B rid cap 0, NF D Form A): H19 SHARE + a full "
                                       "arena on a shared TP slot -- only the last releaser's claim succeeds")
def test_h19_share_on_a_full_shared_arena_ends_alike_on_every_rank(monkeypatch):
    arena = _SharedArena(3)
    rows = [_share_rank(arena, monkeypatch) for _ in range(3)]
    assert rows in ([None] * 3, [[1]] * 3), rows
