"""A7 TOLD-HEAD (weg2-106-554 class; review F1 A7, 16.10.): the #1417 span pin host-locks a read's span
from ``deepest`` up to (excluding) ``stop_at`` -- the matched head ABOVE it was unprotected. A
device-resident head node lost its host copy to the W3 spill (``l3_present``); a device eviction before
the admission then left the told prefix short (pp0_admissible below told).

Repro on the real C arena (dual15 harness): chain a0..a7, told open for a span a3..a7 (stop_at a2), arena
pressure (W3 spill for every page) -> the head a0..a2 keeps its host copy; after a device eviction of the
head the prefix reachable from the root (device or host) is still the told depth. Both spill paths: the
per-rank device round and the F1 room set (PP0 decide).
RED on fd344ad7a1."""
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_w3_arena_spill_forma_dual15_0930 as F  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
group = F.group
FULL = F.FULL


def _reachable_pages(nodes):
    depth = 0
    for n in nodes:
        cd = n.component_data[FULL]
        if cd.value is None and cd.host_value is None:
            break
        depth += len(n.hash_value)
    return depth


def _host_lock_stub(t):
    """The hand-built tree has no lock plumbing: the pin's host lock as the production one sets it."""
    import types

    def inc(node):
        for cd in node.component_data:
            cd.host_lock_ref = int(getattr(cd, "host_lock_ref", 0) or 0) + 1
        return types.SimpleNamespace(to_dec_params=lambda: None)

    def dec(node, params):
        for cd in node.component_data:
            cd.host_lock_ref = int(getattr(cd, "host_lock_ref", 0) or 0) - 1
    t.inc_host_lock_ref, t.dec_host_lock_ref = inc, dec


def _told_then_pressure(g, monkeypatch, pp):
    t, nodes = g.t0, g.n0
    _host_lock_stub(t)
    if pp:
        t.pp_size, t.pp_rank = 3, 0
    told = len(nodes)
    assert t._pin_prefetched_span("weg2-106-554", nodes[-1], nodes[2]) == len(nodes) - 3
    t._w3_spill_quiet_until = 0.0
    t._w3_arena_spill(g.p0, len(nodes))
    for n in nodes[:3]:
        n.component_data[FULL].value = None       # the device eviction before the admission
    return told, nodes


@pytest.mark.parametrize("pp", [False, True], ids=["device_round", "room_set"])
def test_told_head_keeps_its_host_copy_and_the_told_depth(group, monkeypatch, pp):
    from sglang.srt.weg2 import w3_room_set as RS

    monkeypatch.delenv(RS.ENV, raising=False)
    told, nodes = _told_then_pressure(group, monkeypatch, pp)
    assert all(n.component_data[FULL].host_value is not None for n in nodes[:3]), "head lost its host copy"
    assert _reachable_pages(nodes) == told


def test_after_the_admission_the_head_is_a_candidate_again(group):
    t, nodes = group.t0, group.n0
    _host_lock_stub(t)
    t._pin_prefetched_span("r", nodes[-1], nodes[2])
    assert {id(n) for n in nodes[:3]} == t._told_head_ids()
    t._unpin_prefetched_span("r")
    assert t._told_head_ids() == set()
