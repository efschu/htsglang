"""K21 (NF h16u2 1017_132112, D-TP0 13:35:14Z): the write-back peel issues write_backup(node),
then drains the in-flight writes (writing_check(write_back=True)) and demotes the node with
``_evict_to_host`` -- whose first line asserts ``not node.evicted and node.backuped``.  The drain
runs the node's own ack: an A2 ORPHAN (the claim lost its key, the device copy is the only copy)
takes the host layer back on TP0 (K12 A2-ORPHAN REPUBLISH, ``record_state``), so the node is an
UN-backed device node again and the assert killed the rank.  The node must stay on the device
(the publish sweep writes it from its device copy) -- the peel moves on, nothing is freed, nothing
raises.  The stub keeps the live assert of ``_evict_to_host``."""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def _peel(drain_effect):
    t = object.__new__(UnifiedRadixCache)
    calls = []
    t._is_device_leaf = lambda n: True
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.ongoing_write_through = {}
    t.write_backup = lambda n, write_back=False, **kw: (calls.append("write"), 1)[1]
    t._weg2_l3_drop = staticmethod(lambda *a, **k: False)
    node = types.SimpleNamespace(backuped=False, evicted=False, id=142)

    def drain(write_back=False):
        calls.append("drain")
        drain_effect(node)

    t.writing_check = drain
    def evict(n, tr):
        calls.append("evict")
        assert not n.evicted and n.backuped  # the live assert (unified_radix_cache.py:_evict_to_host)

    t._evict_to_host = evict
    t._evict_device_leaf(node, {})
    return calls, node


def test_backed_after_the_drain_is_demoted():
    calls, _ = _peel(lambda n: setattr(n, "backuped", True))
    assert calls == ["write", "drain", "evict"]


def test_orphan_republish_in_the_drain_keeps_the_node_on_the_device():
    # the ack inside the drain gave the host layer back: backuped -> False
    def republish(n):
        n.backuped = False

    calls, node = _peel(republish)
    assert "evict" not in calls, calls  # red on the base: AssertionError (live K21)
    assert calls == ["write", "drain"]
    assert node.evicted is False


def test_node_gone_from_the_device_in_the_drain_is_not_demoted_twice():
    calls, _ = _peel(lambda n: (setattr(n, "backuped", True), setattr(n, "evicted", True)))
    assert "evict" not in calls
