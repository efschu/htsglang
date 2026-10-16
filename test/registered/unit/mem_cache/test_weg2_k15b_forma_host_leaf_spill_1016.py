"""K15b (NF h16q2 1cfbf32584, boot 1016_051456, D 06:25:38, weg2-66-470 h109): the D>P sleep flush
could not publish weg2-66-458's tail -- ``#1470 FLUSH-PUBLISH ARENA-FULL-STOP unbacked=54
arena_refused=54`` on TP0, ``unbacked_left=59`` on TP1/TP2, ``#1421 arena_claim depth=57472 / 66240 /
66304`` -- while ``W3-ARENA SPILL need=2 released_pages=0 nodes=0 candidates=0``: the KV arena was held
by D's HOST-ONLY nodes, which the W3 device round never takes and the W3-HOST-LEAF round only took on
the local-PP floor (P). The reset dropped the tail; the follow-up resumed at the store's 56512
(``#988 LOADBACK anchor_depth=56512``; TP1/TP2 print ``extent=40640`` = 56512 - the device head 15872,
the extent is the host part on a worker -- one prefix on every rank) and computed 8779, priced 1739.

Pinned: on a Form A group TP0 (role host) runs the host-leaf round, and every node it drops goes to
the workers as an R12 state event (the evict_host path) -- never a rank-local drop on a worker.
RED on 394d6dd7e7 (the round is not armed on Form A)."""
import os
import shutil
import sys
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_w3_arena_spill_forma_dual15_0930 as F  # noqa: E402

from sglang.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from sglang.srt.weg2 import host_only_spill as HOS  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
group = F.group
FULL = F.FULL


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(HOS.FORMA_ENV, raising=False)


def test_armed_on_form_a_host_only():
    t = types.SimpleNamespace()
    with mock.patch.object(R, "role", lambda: "host"):
        assert HOS.armed(t) is True
    with mock.patch.object(R, "role", lambda: "worker"):
        assert HOS.armed(t) is False           # a worker never drops on its own
    with mock.patch.object(R, "role", lambda: None):
        assert HOS.armed(t) is False           # classic, off the local-PP floor: as before


def test_switch_off_is_the_old_gate(monkeypatch):
    monkeypatch.setenv(HOS.FORMA_ENV, "0")
    with mock.patch.object(R, "role", lambda: "host"):
        assert HOS.armed(types.SimpleNamespace()) is False


def test_h109_host_only_arena_tp0_spills_host_leaves_and_records_them(group, monkeypatch):
    """The metal state: every slot held by HOST-ONLY nodes (device value gone) -> the device round finds
    candidates=0; TP0's host-leaf round drops leaves (L3 copy first) and records each one for the workers."""
    g = group
    for n in g.n0:
        n.component_data[FULL].value = None          # host-only, as D's tree at the flush
    dropped = []

    def _fake_spill(cache, pool, want, page_size, skip, **kw):
        # the real loop's effect on TP0: the deepest leaf leaves the tree whole, its references go back
        leaf = g.n0[-1]
        leaf.parent.children.pop(tuple(leaf.hash_value), None)
        cache._r12_rec.append(leaf)                  # what _evict_component_and_detach_lru(HOST) records
        dropped.append(leaf)
        return {"released": 1, "leaves": 1, "candidates": 1, "unsecured": 0}

    monkeypatch.setattr(HOS, "spill_host_only", _fake_spill)
    with mock.patch.object(R, "role", lambda: "host"):
        R.register_tree(g.t0)
        released = g.t0._w3_arena_spill(g.p0, 1)
        sent = R.attach([])
    assert released >= 1 and dropped == [g.n0[-1]]
    assert g.t0._r12_rec is None, "the recorder is closed after the round"
    states = [e for e in sent[0].events if e[0] == R.STATE] if sent else []
    key = R.node_key(g.n0[-1])
    assert any((e[1], int(e[2])) == key and int(e[3]) == 0 for e in states), (
        "TP0 dropped a host leaf the workers never hear of -- rank-local divergence; events=%s" % (states,))


def test_worker_takes_no_host_leaf_round(group, monkeypatch):
    g = group
    for n in g.n1:
        n.component_data[FULL].value = None
    called = []
    monkeypatch.setattr(HOS, "spill_host_only", lambda *a, **k: called.append(1) or {"released": 0})
    with mock.patch.object(R, "role", lambda: "worker"):
        assert g.t1._w3_arena_spill(g.p1, 1) == 0
    assert called == []
