"""LANEPARK-YIELD-1014 (NF h16 a2cfc77f76, boot ..._1014_151524, group D = Form A under the token cut).

Two lane-1 image turns stalled behind lane-0 spans the lane park had just written into the shared KV arena:

* weg2-8-68, 15:27:38: ``park(lane) rids=[6-60, 6-62, 2-51, 4-55, 4-57] ... retracted with the span retained and
  a forced host write-through`` -> ``ARENA-REF-HOLDERS ... park_kept=5840/5840`` of 6485 slots; the lane-1 store
  read found the arena FULL: ``L3-FILL ROOM-WAIT stems=730 claimed=0 still_full=730`` at 15:27:55/:28:00/:06/:11,
  each read ended at 65536 tokens of 175872 the store held (``#1324c ... store_hit=175872 delivered=64576``).
* weg2-10-90, 15:32:44-48: every backup of the parked spans was refused (``#1421 BACKUP-REFUSED why=arena_claim``,
  ``parent_unbacked``), the peel paid nothing (``EVICT-FRONTIER-CENSUS ... delivered_after_repair=0``) and
  ``D-WALL-HEAD need_rows=186821 available=99072 payable_evictable=0`` answered W50.

Not one ``W3-ARENA SPILL`` line in the whole boot: ``_w3_arena_spill`` returned 0 under every Form A role. The
spill releases exactly these pages -- host copies of nodes whose KV is still on the device, each with its L3 copy
first. Under the token cut every rank's host rows are the page's bytes (R12 (d)) and the spill's group-equal arena
order decides alike on every rank. Fix (SGLANG_WEG2_W3_SPILL_FORM_A_CUT, default on): the spill runs there -- at a
refused claim (the backup path, case 10-90) and before a short store read whose rest the store holds is re-issued
(case 8-68).
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_w3_arena_spill_0929 import FULL, _claimer, _full_tree  # noqa: E402

from sglang.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402
from sglang.srt.mem_cache import unified_radix_cache as URC  # noqa: E402

ENV = "SGLANG_WEG2_W3_SPILL_FORM_A_CUT"


@pytest.fixture
def form_a(monkeypatch):
    def _set(role, cut):
        monkeypatch.setattr(R12, "role", lambda: role)
        monkeypatch.setattr(R12, "cut_role", lambda: role if cut else None)
    return _set


def test_default_on(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    from sglang.srt.environ import envs
    assert envs.SGLANG_WEG2_W3_SPILL_FORM_A_CUT.get()


@pytest.mark.parametrize("role", ["host", "worker"])
def test_weg2_10_90_under_the_token_cut_the_refused_claim_spills_and_gets_its_room(tmp_path, form_a, monkeypatch, role):
    monkeypatch.delenv(ENV, raising=False)
    form_a(role, cut=True)
    p, arena, root, t, nodes = _full_tree(tmp_path)
    pre = t._weg2_direct_claim(_claimer(t))
    assert pre is not False, f"claim refused: {t.refused} (the #1421 arena_claim of 15:32:44)"
    spilled = [n for n in nodes if n.component_data[FULL].host_value is None]
    assert spilled and all(n.component_data[FULL].value is not None and n.l3_present for n in spilled)


def test_form_a_without_the_cut_keeps_the_old_guard(tmp_path, form_a, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    form_a("worker", cut=False)
    p, arena, root, t, nodes = _full_tree(tmp_path)
    assert t._weg2_direct_claim(_claimer(t)) is False
    assert t.refused == ["arena_claim"]


def test_switch_off_is_the_old_guard(tmp_path, form_a, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    form_a("host", cut=True)
    p, arena, root, t, nodes = _full_tree(tmp_path)
    assert t._weg2_direct_claim(_claimer(t)) is False


def test_weg2_8_68_the_read_spill_frees_arena_room_before_the_reissue(tmp_path, form_a, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    form_a("host", cut=True)
    p, arena, root, t, nodes = _full_tree(tmp_path)
    t.W3_SPILL_MIN_PAGES = 1
    released = URC.UnifiedRadixCache.weg2_spill_for_read(t, 3)
    assert released >= 3
    assert sum(1 for n in nodes if n.component_data[FULL].host_value is None) >= 3
    for n in nodes:
        if n.component_data[FULL].host_value is None:
            assert (root / (n.hash_value[0] + "_sfx.bin")).exists(), "never out of L2 without its L3 copy"


def test_the_read_spill_without_a_pool_is_a_no_op(form_a):
    form_a("host", cut=True)
    t = object.__new__(URC.UnifiedRadixCache)
    t._weg2_direct_pool = lambda: None
    assert URC.UnifiedRadixCache.weg2_spill_for_read(t, 10) == 0
    assert URC.UnifiedRadixCache.weg2_spill_for_read(t, 0) == 0


# ---- the drain hook (scheduler) ------------------------------------------------------------------

def test_the_store_short_drain_asks_for_room_when_the_store_holds_the_rest(monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weg2"))
    from test_weg2_store_short_fallback_rc12y import _sched  # noqa: E402

    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    monkeypatch.delenv(ENV, raising=False)
    s, r = _sched(176133, 64576, 175872, 7005)
    asked = []
    s.tree_cache.weg2_spill_for_read = lambda n: asked.append(n) or n
    s._weg2_note_store_shortfall(r)
    assert asked == [175872 - 64576]          # page_size 1 in the harness
    assert r._weg2_read_spilled == 175872 - 64576


def test_the_drain_hook_off(monkeypatch):
    from test_weg2_store_short_fallback_rc12y import _sched  # noqa: E402

    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    monkeypatch.setenv(ENV, "0")
    s, r = _sched(176133, 64576, 175872, 7005)
    asked = []
    s.tree_cache.weg2_spill_for_read = lambda n: asked.append(n) or n
    s._weg2_note_store_shortfall(r)
    assert asked == []
