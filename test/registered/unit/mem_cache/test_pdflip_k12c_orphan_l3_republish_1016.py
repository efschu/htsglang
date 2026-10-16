"""K12c review A3 (nf-review-k12c-1016): a direct write whose pages A2 ORPHANED
is not in L3 -- no l3_present -- and the node is republished from its device copy.

Before: ``_pdflip_direct_complete`` set ``node.l3_present = True`` after
``_a2_settle_lost`` whatever it orphaned. L3-DROP could then drop the device
copy -- for the orphaned pages the ONLY copy -- and the publish sweep never
looked at the node again (backuped / l3_present): the next read recomputed
them (on D: the named read stop and the detour to P). It also bypassed H1 (no
TP0 verdict, no record_l3 under Form A).

Now: orphaned > 0 -> no l3_present; the host layer goes back (TP0 decides on a
Form A group and sends its STATE over R12; a worker follows, never decides; a
pipeline tree without R12 does not drop host stage-locally) and the publish
sweep writes the node from the device copy. Without orphans: R12 -> TP0 marks
and sends L3, a worker waits for it; off R12 -> the rank marks, as before.
RED on 7c2c16eb7a (l3_present True after an orphaning ack; no republish).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U  # noqa: E402

PAGE = 64
STAGING = 4096


class _Pool:
    def __init__(self, orphaned):
        self.arena = object()
        self.staging_rows = STAGING
        self.orphaned = orphaned

    def complete_write(self, hv):
        return int(hv.numel()) // PAGE

    def settle_lost_rows(self, hv, hashes=None):
        return {"lost": self.orphaned, "repointed": 0, "orphaned": self.orphaned, "moved": []}


class _Tree:
    _pdflip_direct_complete = U._pdflip_direct_complete
    _a2_settle_lost = U._a2_settle_lost

    def __init__(self, orphaned, pp_size=1):
        self.pp_size = pp_size
        self.cache_controller = SimpleNamespace(mem_pool_host=_Pool(orphaned), mem_pool_host_draft=None)
        self.released = []

    def _pdflip_mamba_pool(self):
        return None

    def _pdflip_release_chain_piece_host(self, node):
        # the real release returns early under a host lock or without a device copy
        cd = node.component_data[0]
        if cd.host_lock_ref or cd.value is None:
            return
        cd.host_value = None
        self.released.append(node.id)


def _node(locked=False):
    hv = torch.arange(STAGING, STAGING + 2 * PAGE)
    cd = SimpleNamespace(host_value=hv, value=torch.arange(2 * PAGE), host_lock_ref=1 if locked else 0)
    return SimpleNamespace(id=5, component_data=[cd], hash_value=["a", "b"], l3_present=False)


@pytest.fixture
def r12(monkeypatch):
    calls = {"l3": [], "state": []}
    st = {"role": None}
    monkeypatch.setattr(R, "role", lambda: st["role"])
    monkeypatch.setattr(R, "l3_group_on", lambda env=None: True)
    monkeypatch.setattr(R, "record_l3", lambda tree, node: calls["l3"].append(node.id) or True)
    monkeypatch.setattr(R, "record_state", lambda tree, node, why="": calls["state"].append((node.id, why)))
    monkeypatch.delenv("FLLIPER_PDFLIP_L3_PRESENT_STRICT", raising=False)
    return st, calls


def test_orphaned_classic_no_l3_and_republished(r12):
    t, n = _Tree(orphaned=24), _node()
    assert t._pdflip_direct_complete(n) is True
    assert n.l3_present is False
    assert t.released == [5] and n.component_data[0].host_value is None  # sweep sees an un-backed device node


def test_clean_classic_marks_l3(r12):
    t, n = _Tree(orphaned=0), _node()
    t._pdflip_direct_complete(n)
    assert n.l3_present is True and t.released == []


def test_orphaned_pipeline_tree_keeps_host_no_l3(r12):
    t, n = _Tree(orphaned=24, pp_size=3), _node()
    t._pdflip_direct_complete(n)
    assert n.l3_present is False and t.released == []
    assert n.component_data[0].host_value is not None


def test_orphaned_form_a_host_releases_and_sends_state(r12):
    st, calls = r12
    st["role"] = "host"
    t, n = _Tree(orphaned=24), _node()
    t._pdflip_direct_complete(n)
    assert n.l3_present is False and t.released == [5]
    assert calls["state"] == [(5, "k12-orphan-republish")] and calls["l3"] == []


def test_clean_form_a_host_marks_and_sends_l3(r12):
    st, calls = r12
    st["role"] = "host"
    t, n = _Tree(orphaned=0), _node()
    t._pdflip_direct_complete(n)
    assert n.l3_present is True and calls["l3"] == [5] and calls["state"] == []


@pytest.mark.parametrize("orphaned", [0, 24])
def test_form_a_worker_never_decides(r12, orphaned):
    st, calls = r12
    st["role"] = "worker"
    t, n = _Tree(orphaned=orphaned), _node()
    t._pdflip_direct_complete(n)
    assert n.l3_present is False and t.released == []
    assert calls == {"l3": [], "state": []}


def test_host_lock_held_keeps_the_node_no_l3(r12):
    t, n = _Tree(orphaned=24), _node(locked=True)
    t._pdflip_direct_complete(n)
    assert n.l3_present is False and t.released == []


def test_strict_off_is_the_old_mark(r12, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PRESENT_STRICT", "0")
    t, n = _Tree(orphaned=24), _node()
    t._pdflip_direct_complete(n)
    assert n.l3_present is True and t.released == []
