"""S86b (rc22 boot ..._9c40fe40a2_1020_094747, D.log from line 79109, 10:11:49Z, 6x on every D rank):

    sleep_l3_release.py:161 local_veto -> :126 _l3_proven -> memory_pool_host.py:2370 __getattr__
    AttributeError: secure_rows_to_l3
    S86 SLEEP-L3-RELEASE veto raised: every slot kept ... vetoed_by_group=6485 ... group_released=0

D's direct pool is the hybrid ``HostPoolGroup`` (KV + Mamba). It forwards the arena calls to its anchor pool -- the
KV ``ArenaMHAHostPool`` -- but not ``secure_rows_to_l3``: every veto failed closed and S86 released nothing on D.

The fix (``sleep_l3_release._arena_pool``): the step works on the pool that holds the arena -- the group's anchor
pool. Hermetic, the REAL classes: ``HostPoolGroup`` (its ``__getattr__`` and ``arena`` property) over the real
``ArenaMHAHostPool`` ranks of the S86 fixture (real C arena, one file, two ranks in threads). RED on 9c40fe40a2."""
import os
import sys
import threading
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_s86_sleep_l3_release_1020 as B  # noqa: E402
from test_s86_sleep_l3_release_1020 import _thread_roles, rc19  # noqa: E402,F401

from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup  # noqa: E402
from flliper.srt.pdflip import sleep_l3_release as S86  # noqa: E402

FULL, SLOTS = B.FULL, B.SLOTS


def _hybrid(pool):
    """D's pool type: a HostPoolGroup whose anchor entry is the KV arena pool (the Mamba entry is not read here)."""
    g = object.__new__(HostPoolGroup)
    g.anchor_entry = types.SimpleNamespace(host_pool=pool)
    return g


def _both(g, pools):
    grp = B._Group(2)
    res = [None, None]

    def run(i, role, t, p):
        B._ROLE.r = role
        try:
            res[i] = S86.run(t, grp.member(i), p)
        except BaseException as e:  # noqa: BLE001
            res[i] = e
        finally:
            B._ROLE.r = None

    ths = [threading.Thread(target=run, args=(0, "host", g.t0, pools[0])),
           threading.Thread(target=run, args=(1, "worker", g.t1, pools[1]))]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    return res, grp


def test_the_hybrid_group_really_lacks_the_call():
    """The live cause, pinned on the real class: the group does not forward secure_rows_to_l3 (W3 reads hasattr and
    keeps its own switch -- S86b does not change the group)."""
    with pytest.raises(AttributeError):
        _hybrid(object()).secure_rows_to_l3     # noqa: B018


def test_rc22_d_hybrid_pool_the_sleep_release_frees_the_l3_pages(rc19):
    """RED on 9c40fe40a2: every veto raised (AttributeError), vetoed 8/8, nothing released. GREEN: the anchor pool
    secures and the group releases all 8 slots on both ranks, the nodes stay on the device, l3_present."""
    g = rc19
    res, grp = _both(g, (_hybrid(g.p0), _hybrid(g.p1)))
    assert not any(isinstance(r, BaseException) for r in res), res
    assert res[0]["why"] == {} and res[1]["why"] == {}, (res[0].get("why"), res[1].get("why"))
    assert res[0]["pages"] == res[1]["pages"] == SLOTS and res[0]["group_released"] == SLOTS
    assert B._refs(g) == [0] * SLOTS
    for a, b in zip(g.n0, g.n1):
        for n in (a, b):
            assert n.component_data[FULL].host_value is None and n.l3_present
            assert n.component_data[FULL].value is not None


def test_rc22_d_the_tree_s_own_direct_pool_is_the_group(rc19):
    """As on D: no pool passed, the tree's ``_pdflip_direct_pool`` hands back the hybrid group."""
    g = rc19
    g.t0._pdflip_direct_pool = lambda: _hybrid(g.p0)
    g.t1._pdflip_direct_pool = lambda: _hybrid(g.p1)
    res, grp = _both(g, (None, None))
    assert not any(isinstance(r, BaseException) for r in res), res
    assert res[0]["group_released"] == SLOTS and B._refs(g) == [0] * SLOTS


def test_a_group_whose_anchor_cannot_secure_vetoes_everything(rc19):
    """No pool that holds the call: the full veto (A3b, fail closed), every collective still posted."""
    g = rc19
    blind = types.SimpleNamespace(anchor_entry=types.SimpleNamespace(host_pool=object()))
    res, grp = _both(g, (_hybrid(g.p0), blind))
    assert not any(isinstance(r, BaseException) for r in res), res
    assert B._refs(g) == [2] * SLOTS
    assert grp.calls[0] == grp.calls[1] and len(grp.calls[0]) == 3


def test_an_unpaired_page_stays_through_the_group_too(rc19, monkeypatch):
    """A3a holds on the hybrid path: an unpaired page keeps its node, whatever the strict switch says."""
    import torch

    g = rc19
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PRESENT_STRICT", "0")
    victim = [g.n0[4].component_data[FULL].host_value, g.n1[4].component_data[FULL].host_value]
    for p in (g.p0, g.p1):
        orig = p.secure_rows_to_l3

        def wrapped(hv, _orig=orig):
            sec = dict(_orig(hv))
            if any(torch.equal(hv, v) for v in victim):
                sec.update(unpaired=1, on_disk=max(0, int(sec.get("on_disk", 0)) - 1))
            return sec
        p.secure_rows_to_l3 = wrapped
    res, grp = _both(g, (_hybrid(g.p0), _hybrid(g.p1)))
    assert not any(isinstance(r, BaseException) for r in res), res
    assert B._refs(g)[4] == 2
    assert g.n0[4].component_data[FULL].host_value is not None and g.n1[4].component_data[FULL].host_value is not None
