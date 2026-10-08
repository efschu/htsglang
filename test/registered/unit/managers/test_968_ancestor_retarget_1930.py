"""Q-1930 #968 ANCESTOR-RETARGET (27B NVFP4 dual hb 0ceb3699f8, boot
dkr27bnvfp4dual1mpsleepsharegreentop112bar1fs10050428, P PP1 W17 04:33:13Z,
rid pdflip-0-4).

The race replayed on the follower's tree shape (debug-hold rank1_pid787):
PP0 lost its 53248 anchor (node 51788..53248 evicted host=False, a tree edge
only PP0 had after its PP0-only L3 read of pdflip-0-3 fell back to told=0), so
PP0 decided prefix_len=49152 from its host anchor at 49152. PP1 still held the
node 49152..53248 with a host-backed state, its best_match_node was that node,
the load-back served 53248 with the anchor adopted at 53248 -> #968 SHORTFALL.
PP1 ALSO held the 49152 node with a host-backed state.

Pinned: with FLLIPER_PDFLIP_968_ANCESTOR_RETARGET=1 the follower loads back from
the ancestor ending at the decision (exactly 49152, state at 49152); without
the gate, or without a host state at the decision, the #968 stop stands.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

import flliper.srt.managers.pp_admission_congruence as congruence  # noqa: E402
from flliper.srt.managers.pp_admission_congruence import (  # noqa: E402
    ENV_ANCESTOR_RETARGET,
    execute_scheduled_prefix,
)
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
)

DECISION = 49152
DEEP = 53248


class _Node:
    _id = 0

    def __init__(self, parent, n, host_state=True, device_state=False):
        _Node._id += 1
        self.id = _Node._id
        self.parent = parent
        self.key = list(range(n))
        self.component_data = {
            ComponentType.MAMBA: SimpleNamespace(
                value=object() if device_state else None,
                host_value=object() if host_state else None,
            )
        }


class _Tree:
    """init_load_back as the real one behaves on this path: the KV of every
    node from best_match_node up to req.last_node, and the MAMBA half plants
    the best_match_node's host state into the request's slot (FIX-3 flag)."""

    enable_storage = True

    def __init__(self):
        self.loaded_from = []

    def init_load_back(self, params):
        node, req = params.best_match_node, params.req
        self.loaded_from.append(node)
        n, cur = 0, node
        while cur is not req.last_node:
            n += len(cur.key)
            cur = cur.parent
        comp = node.component_data[ComponentType.MAMBA]
        if comp.host_value is not None and comp.value is None:
            req.mamba_loadback_anchor_adopted = True
        return torch.arange(n), node


def _follower(state_at_decision=True):
    root = _Node(None, 0)
    a = _Node(root, 45056)                                   # 0..45056
    b = _Node(a, DECISION - 45056, host_state=state_at_decision)  # ..49152
    c = _Node(b, DEEP - DECISION)                            # ..53248, host state
    req = SimpleNamespace(
        rid="pdflip-0-4",
        prefix_indices=torch.arange(0),
        cache_protected_len=0,
        host_hit_length=DECISION,
        best_match_node=c,
        last_node=root,
        mamba_loadback_anchor_adopted=False,
        full_untruncated_fill_ids=list(range(53890)),
    )
    return req, b, c


@pytest.fixture(autouse=True)
def _short_bound(monkeypatch):
    monkeypatch.setattr(congruence, "MATERIALISE_BASE_S", 0.0)
    monkeypatch.setattr(congruence, "MATERIALISE_MAX_S", 0.05)


def test_metal_shape_gate_on_executes_the_decision_from_the_ancestor(monkeypatch):
    """RED on 0ceb3699f8 (no retarget): the 53248 node is loaded, the anchor
    is adopted at 53248, #968 SHORTFALL. Gate on: 49152 from node b."""
    monkeypatch.setenv(ENV_ANCESTOR_RETARGET, "1")
    req, b, _c = _follower()
    tree = _Tree()
    loaded = execute_scheduled_prefix(req, tree, DECISION)
    assert loaded == DECISION
    assert len(req.prefix_indices) == DECISION
    assert req.last_node is b
    assert tree.loaded_from == [b]
    assert req.mamba_loadback_anchor_adopted is True


def test_gate_off_keeps_the_968_stop(monkeypatch):
    """Default OFF: byte-identical behaviour -- the metal death."""
    monkeypatch.delenv(ENV_ANCESTOR_RETARGET, raising=False)
    req, _b, c = _follower()
    tree = _Tree()
    with pytest.raises(RuntimeError, match="#968 PREFIX MATERIALISATION SHORTFALL"):
        execute_scheduled_prefix(req, tree, DECISION)
    assert tree.loaded_from == [c]


def test_no_state_at_the_decision_keeps_the_968_stop(monkeypatch):
    """Danger direction: never resume at the decision over KV without its
    recurrent state -- an ancestor with no host state is not taken."""
    monkeypatch.setenv(ENV_ANCESTOR_RETARGET, "1")
    req, _b, c = _follower(state_at_decision=False)
    tree = _Tree()
    with pytest.raises(RuntimeError, match="#968 PREFIX MATERIALISATION SHORTFALL"):
        execute_scheduled_prefix(req, tree, DECISION)
    assert tree.loaded_from == [c]


def test_no_edge_at_the_decision_keeps_the_968_stop(monkeypatch):
    """No node ends exactly at the decision: nothing is guessed."""
    monkeypatch.setenv(ENV_ANCESTOR_RETARGET, "1")
    req, _b, c = _follower()
    tree = _Tree()
    with pytest.raises(RuntimeError, match="#968 PREFIX MATERIALISATION SHORTFALL"):
        execute_scheduled_prefix(req, tree, DECISION - 100)
    assert tree.loaded_from == [c]


def test_device_state_ancestor_is_not_taken(monkeypatch):
    """A device-resident state is not planted by the load-back: not ours."""
    monkeypatch.setenv(ENV_ANCESTOR_RETARGET, "1")
    req, b, c = _follower()
    b.component_data[ComponentType.MAMBA].value = object()
    tree = _Tree()
    with pytest.raises(RuntimeError):
        execute_scheduled_prefix(req, tree, DECISION)
    assert tree.loaded_from == [c]


def test_exact_match_is_untouched(monkeypatch):
    """best_match_node already ends at the decision: unchanged path."""
    monkeypatch.setenv(ENV_ANCESTOR_RETARGET, "1")
    req, b, _c = _follower()
    req.best_match_node = b
    tree = _Tree()
    assert execute_scheduled_prefix(req, tree, DECISION) == DECISION
    assert tree.loaded_from == [b]
